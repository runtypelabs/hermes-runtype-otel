"""Hermes observer plugin exporting one OTLP trace per agent turn to Runtype."""

from __future__ import annotations

import atexit
import json
import logging
import os
import re
import secrets
import threading
import time
from urllib import error, request


_ADAPTER_NAME = "@runtypelabs/hermes-adapter"
_ADAPTER_VERSION = "0.1.0"
_DEFAULT_ENDPOINT = "https://api.runtype.com/v1/otel/v1/traces"
_MAX_CONTENT = 4096
_MAX_MESSAGES = 16
_MAX_ACTIVE_TURNS = 256
_EXPORT_TIMEOUT_SECONDS = 2.0
# Turns still open at exit are flushed within this budget so a restart is never held up for long.
_EXIT_FLUSH_BUDGET_SECONDS = 5.0
_LOCK = threading.RLock()
_TURNS: dict[str, dict] = {}
_EVICTED_TURNS: dict[str, None] = {}
_CAPACITY_WARNING_EMITTED = False
_LOGGER = logging.getLogger("runtype.hermes.adapter")
_SECRET_PATTERN = re.compile(
    r"(?i)\b(?:sk-[a-z0-9_-]{12,}|rt_[a-z0-9_-]{12,})\b|\bBearer\s+\S+"
)


def _enabled() -> bool:
    return bool(
        os.environ.get("RUNTYPE_AGENT_ID") and os.environ.get("RUNTYPE_OTEL_API_KEY")
    )


def _capture_content() -> bool:
    return os.environ.get("RUNTYPE_OTEL_CAPTURE_CONTENT", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _redact(value: str) -> str:
    for name in ("RUNTYPE_OTEL_API_KEY", "OPENAI_API_KEY"):
        secret = os.environ.get(name, "")
        if secret:
            value = value.replace(secret, "[REDACTED]")
    return _SECRET_PATTERN.sub("[REDACTED]", value)


def _redact_structured(value: object, depth: int = 0) -> object:
    if depth >= 16:
        return "[OMITTED:DEPTH_LIMIT]"
    if isinstance(value, str):
        return _redact(value)
    if isinstance(value, dict):
        return {
            _redact(str(key)): _redact_structured(item, depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_structured(item, depth + 1) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _redact(str(value))


def _content_text(value: object) -> str:
    if isinstance(value, str):
        text = value
    elif isinstance(value, list):
        text = "\n".join(_content_text(item) for item in value)
    elif isinstance(value, dict):
        content = value.get("text") or value.get("content")
        text = (
            content
            if isinstance(content, str)
            else json.dumps(_redact_structured(value), default=str)
        )
    else:
        text = str(value or "")
    return _redact(text)[:_MAX_CONTENT]


def _messages(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    rows = []
    for item in value[-_MAX_MESSAGES:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if role not in {"system", "developer", "user", "assistant", "tool"}:
            continue
        content = _content_text(item.get("content"))
        if content:
            rows.append({"role": role, "content": content})
    return rows


def _attribute(key: str, value: object) -> dict:
    if isinstance(value, bool):
        encoded = {"boolValue": value}
    elif isinstance(value, int):
        encoded = {"intValue": str(value)}
    elif isinstance(value, float):
        encoded = {"doubleValue": value}
    elif isinstance(value, (list, tuple)) and all(
        isinstance(item, str) for item in value
    ):
        encoded = {"arrayValue": {"values": [{"stringValue": item} for item in value]}}
    else:
        encoded = {"stringValue": str(value)}
    return {"key": key, "value": encoded}


def _span(
    trace_id: str,
    parent_id: str | None,
    name: str,
    attributes: dict,
    start: int | None = None,
) -> dict:
    span = {
        "traceId": trace_id,
        "spanId": secrets.token_hex(8),
        "name": name,
        "kind": 1,
        "startTimeUnixNano": str(start or time.time_ns()),
        "endTimeUnixNano": "0",
        "attributes": [
            _attribute(key, value)
            for key, value in attributes.items()
            if value is not None
        ],
    }
    if parent_id:
        span["parentSpanId"] = parent_id
    return span


def _set_attribute(span: dict, key: str, value: object) -> None:
    span["attributes"] = [item for item in span["attributes"] if item["key"] != key]
    span["attributes"].append(_attribute(key, value))


def _finish_span(
    span: dict, *, failed: bool = False, error_type: str = "agent_error"
) -> None:
    if span["endTimeUnixNano"] != "0":
        return
    span["endTimeUnixNano"] = str(
        max(time.time_ns(), int(span["startTimeUnixNano"]) + 1)
    )
    span["status"] = {"code": 2 if failed else 1}
    if failed:
        _set_attribute(span, "error.type", error_type)


def _key(turn_id: str = "", task_id: str = "", session_id: str = "", **_kwargs) -> str:
    return str(turn_id or task_id or session_id)


def _start_turn(
    key: str,
    model: str = "",
    platform: str = "",
    user_message: object = None,
    session_id: str = "",
) -> dict | None:
    global _CAPACITY_WARNING_EMITTED
    if not key or not _enabled():
        return None
    evicted = None
    with _LOCK:
        if key in _EVICTED_TURNS:
            return None
        state = _TURNS.get(key)
        if state:
            return state
        if len(_TURNS) >= _MAX_ACTIVE_TURNS:
            oldest_key = next(iter(_TURNS))
            evicted = _TURNS.pop(oldest_key)
            for child in evicted["children"]:
                _finish_span(
                    child, failed=True, error_type="telemetry_capacity_eviction"
                )
            _set_attribute(evicted["root"], "runtype.stop_reason", "error")
            _finish_span(
                evicted["root"],
                failed=True,
                error_type="telemetry_capacity_eviction",
            )
            evicted["root"]["status"]["message"] = (
                "Hermes telemetry capacity reached before turn completed"
            )
            _EVICTED_TURNS[oldest_key] = None
            if len(_EVICTED_TURNS) > _MAX_ACTIVE_TURNS:
                _EVICTED_TURNS.pop(next(iter(_EVICTED_TURNS)))
            if not _CAPACITY_WARNING_EMITTED:
                _LOGGER.warning(
                    "Hermes OTLP adapter reached capacity (%d active turns); "
                    "oldest incomplete traces are exported before eviction",
                    _MAX_ACTIVE_TURNS,
                )
                _CAPACITY_WARNING_EMITTED = True
        else:
            _CAPACITY_WARNING_EMITTED = False
        trace_id = secrets.token_hex(16)
        attrs = {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "Hermes"}
        if model:
            attrs["gen_ai.request.model"] = model
        if platform:
            attrs["hermes.platform"] = platform
        if _capture_content() and user_message is not None:
            attrs["gen_ai.input.messages"] = json.dumps(
                [{"role": "user", "content": _content_text(user_message)}]
            )
        root = _span(trace_id, None, "invoke_agent Hermes", attrs)
        state = {
            "trace_id": trace_id,
            "session_id": session_id,
            "root": root,
            "children": [],
            "models": {},
            "tools": {},
        }
        _TURNS[key] = state
    if evicted is not None:
        _export(evicted)
    return state


def on_pre_llm_call(
    *,
    session_id: str = "",
    task_id: str = "",
    turn_id: str = "",
    user_message=None,
    model: str = "",
    platform: str = "",
    **_kwargs,
) -> None:
    _start_turn(
        _key(turn_id, task_id, session_id), model, platform, user_message, session_id
    )


def on_pre_api_request(
    *,
    session_id: str = "",
    task_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
    api_call_count: int = 0,
    model: str = "",
    provider: str = "",
    request_messages=None,
    user_message=None,
    platform: str = "",
    **_kwargs,
) -> None:
    key = _key(turn_id, task_id, session_id)
    state = _start_turn(key, model, platform, user_message, session_id)
    if state is None:
        return
    model_key = str(api_request_id or api_call_count)
    with _LOCK:
        if model_key in state["models"]:
            return
    attrs = {
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": model,
        "gen_ai.provider.name": provider,
        "runtype.iteration": max(0, int(api_call_count or 1) - 1),
    }
    if _capture_content():
        messages = _messages(request_messages)
        if messages:
            attrs["gen_ai.input.messages"] = json.dumps(messages)
    child = _span(state["trace_id"], state["root"]["spanId"], "chat", attrs)
    with _LOCK:
        state["children"].append(child)
        state["models"][model_key] = child


def _message_content(message: object) -> str:
    if isinstance(message, dict):
        return _content_text(message.get("content"))
    return _content_text(getattr(message, "content", ""))


def on_post_api_request(
    *,
    session_id: str = "",
    task_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
    api_call_count: int = 0,
    model: str = "",
    response_model: str = "",
    usage=None,
    assistant_message=None,
    finish_reason: str = "",
    **_kwargs,
) -> None:
    with _LOCK:
        state = _TURNS.get(_key(turn_id, task_id, session_id))
        child = (
            state["models"].pop(str(api_request_id or api_call_count), None)
            if state
            else None
        )
        if child is None:
            return
        if response_model or model:
            _set_attribute(child, "gen_ai.response.model", response_model or model)
        if isinstance(usage, dict):
            input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
            output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
            cache_read_tokens = usage.get("cache_read_tokens")
            cache_write_tokens = usage.get("cache_write_tokens")
            if type(input_tokens) is int and input_tokens >= 0:
                _set_attribute(child, "gen_ai.usage.input_tokens", input_tokens)
            if type(output_tokens) is int and output_tokens >= 0:
                _set_attribute(child, "gen_ai.usage.output_tokens", output_tokens)
            if type(cache_read_tokens) is int and cache_read_tokens >= 0:
                _set_attribute(
                    child, "gen_ai.usage.cache_read.input_tokens", cache_read_tokens
                )
            if type(cache_write_tokens) is int and cache_write_tokens >= 0:
                _set_attribute(
                    child,
                    "gen_ai.usage.cache_creation.input_tokens",
                    cache_write_tokens,
                )
        if finish_reason:
            _set_attribute(child, "gen_ai.response.finish_reasons", [finish_reason])
        if _capture_content() and assistant_message is not None:
            content = _message_content(assistant_message)
            if content:
                _set_attribute(
                    child,
                    "gen_ai.output.messages",
                    json.dumps([{"role": "assistant", "content": content}]),
                )
        _finish_span(child)


def on_api_request_error(
    *,
    session_id: str = "",
    task_id: str = "",
    turn_id: str = "",
    api_request_id: str = "",
    api_call_count: int = 0,
    **_kwargs,
) -> None:
    with _LOCK:
        state = _TURNS.get(_key(turn_id, task_id, session_id))
        child = (
            state["models"].get(str(api_request_id or api_call_count))
            if state
            else None
        )
        if child:
            _set_attribute(child, "hermes.api.error_observed", True)


def on_pre_tool_call(
    *,
    session_id: str = "",
    task_id: str = "",
    turn_id: str = "",
    tool_call_id: str = "",
    tool_name: str = "",
    args=None,
    **_kwargs,
) -> None:
    with _LOCK:
        state = _TURNS.get(_key(turn_id, task_id, session_id))
        if state is None:
            return
        attrs = {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": tool_name}
        if tool_call_id:
            attrs["gen_ai.tool.call.id"] = tool_call_id
        child = _span(
            state["trace_id"],
            state["root"]["spanId"],
            "execute_tool " + tool_name,
            attrs,
        )
        state["children"].append(child)
        state["tools"].setdefault(str(tool_call_id or tool_name), []).append(child)


def on_post_tool_call(
    *,
    session_id: str = "",
    task_id: str = "",
    turn_id: str = "",
    tool_call_id: str = "",
    tool_name: str = "",
    args=None,
    result=None,
    status: str = "",
    **_kwargs,
) -> None:
    with _LOCK:
        state = _TURNS.get(_key(turn_id, task_id, session_id))
        queue = state["tools"].get(str(tool_call_id or tool_name)) if state else None
        child = queue.pop(0) if queue else None
        if queue is not None and not queue:
            state["tools"].pop(str(tool_call_id or tool_name), None)
        if child is None:
            return
        if _capture_content() and args is not None:
            serialized_args = json.dumps(_redact_structured(args), default=str)
            if len(serialized_args) <= _MAX_CONTENT:
                _set_attribute(child, "gen_ai.tool.call.arguments", serialized_args)
        if _capture_content() and result is not None:
            _set_attribute(child, "gen_ai.tool.call.result", _content_text(result))
        _finish_span(
            child, failed=status not in {"ok", "success"}, error_type="tool_error"
        )


def on_post_llm_call(
    *,
    session_id: str = "",
    task_id: str = "",
    turn_id: str = "",
    assistant_response=None,
    **_kwargs,
) -> None:
    if not _capture_content() or assistant_response is None:
        return
    with _LOCK:
        state = _TURNS.get(_key(turn_id, task_id, session_id))
        if state:
            content = _content_text(assistant_response)
            if content:
                _set_attribute(
                    state["root"],
                    "gen_ai.output.messages",
                    json.dumps([{"role": "assistant", "content": content}]),
                )


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _export(state: dict, timeout: float = _EXPORT_TIMEOUT_SECONDS) -> None:
    endpoint = os.environ.get("RUNTYPE_OTEL_ENDPOINT", _DEFAULT_ENDPOINT)
    api_key = os.environ.get("RUNTYPE_OTEL_API_KEY", "")
    agent_id = os.environ.get("RUNTYPE_AGENT_ID", "")
    if not (api_key and agent_id):
        return
    resource = {
        "service.name": "hermes-agent",
        "runtype.agent.id": agent_id,
        "runtype.adapter.name": _ADAPTER_NAME,
        "runtype.adapter.version": _ADAPTER_VERSION,
    }
    payload = {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        _attribute(key, value) for key, value in resource.items()
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": _ADAPTER_NAME, "version": _ADAPTER_VERSION},
                        "spans": [state["root"], *state["children"]],
                    }
                ],
            }
        ]
    }
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    try:
        outgoing = request.Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + api_key,
            },
        )
        with request.build_opener(_NoRedirect).open(
            outgoing, timeout=timeout
        ) as response:
            response.read(1024)
    except error.HTTPError as exc:
        exc.close()
    except Exception:
        pass


def on_session_end(
    *,
    session_id: str = "",
    task_id: str = "",
    turn_id: str = "",
    completed: bool = False,
    failed: bool = False,
    interrupted: bool = False,
    turn_exit_reason: str = "",
    **_kwargs,
) -> None:
    with _LOCK:
        key = _key(turn_id, task_id, session_id)
        state = _TURNS.pop(key, None)
        _EVICTED_TURNS.pop(key, None)
        if state is None and session_id and not turn_id and not task_id:
            matching_key = next(
                (
                    candidate
                    for candidate, active in reversed(_TURNS.items())
                    if active["session_id"] == session_id
                ),
                None,
            )
            state = _TURNS.pop(matching_key, None) if matching_key else None
        if state is None:
            return
        for child in state["children"]:
            if child["endTimeUnixNano"] == "0":
                _finish_span(child, failed=True, error_type="incomplete_operation")
        _set_attribute(
            state["root"],
            "runtype.iterations",
            len(
                [
                    child
                    for child in state["children"]
                    if any(
                        item["key"] == "gen_ai.operation.name"
                        and item["value"].get("stringValue") == "chat"
                        for item in child["attributes"]
                    )
                ]
            ),
        )
        stop_reason = (
            "max_turns"
            if isinstance(turn_exit_reason, str)
            and turn_exit_reason.startswith("max_iterations_reached")
            else "cancelled"
            if interrupted
            else "error"
            if failed or not completed
            else "end_turn"
        )
        _set_attribute(state["root"], "runtype.stop_reason", stop_reason)
        _finish_span(
            state["root"],
            failed=failed or interrupted or not completed,
            error_type="interrupted" if interrupted else "agent_error",
        )
    _export(state)


def _finalize_open_turns() -> None:
    with _LOCK:
        states = list(_TURNS.values())
        _TURNS.clear()
        _EVICTED_TURNS.clear()
        for state in states:
            for child in state["children"]:
                _finish_span(child, failed=True, error_type="process_exit")
            _set_attribute(state["root"], "runtype.stop_reason", "error")
            _finish_span(state["root"], failed=True, error_type="process_exit")
    deadline = time.monotonic() + _EXIT_FLUSH_BUDGET_SECONDS
    for state in states:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        _export(state, timeout=min(_EXPORT_TIMEOUT_SECONDS, remaining))


def register(ctx) -> None:
    hooks = (
        ("pre_llm_call", on_pre_llm_call),
        ("post_llm_call", on_post_llm_call),
        ("pre_api_request", on_pre_api_request),
        ("post_api_request", on_post_api_request),
        ("api_request_error", on_api_request_error),
        ("pre_tool_call", on_pre_tool_call),
        ("post_tool_call", on_post_tool_call),
        ("on_session_end", on_session_end),
    )
    for name, callback in hooks:
        ctx.register_hook(name, callback)


atexit.register(_finalize_open_turns)
