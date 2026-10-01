import importlib.util
import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch


PLUGIN_PATH = Path(__file__).resolve().parents[1] / "__init__.py"


def load_plugin():
    spec = importlib.util.spec_from_file_location(
        "runtype_hermes_test_plugin", PLUGIN_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CaptureHandler(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        self.requests.append(
            (self.path, dict(self.headers), json.loads(self.rfile.read(length)))
        )
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *_args):
        pass


class RedirectHandler(BaseHTTPRequestHandler):
    destination = ""

    def do_POST(self):
        self.send_response(302)
        self.send_header("Location", self.destination)
        self.end_headers()

    def log_message(self, *_args):
        pass


class DestinationHandler(BaseHTTPRequestHandler):
    requests = []

    def do_GET(self):
        self.requests.append(dict(self.headers))
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        self.requests.append(dict(self.headers))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *_args):
        pass


def attributes(span):
    return {
        item["key"]: next(iter(item["value"].values())) for item in span["attributes"]
    }


class HermesAdapterTest(unittest.TestCase):
    def setUp(self):
        CaptureHandler.requests = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), CaptureHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.env = patch.dict(
            os.environ,
            {
                "RUNTYPE_AGENT_ID": "agent_test123",
                "RUNTYPE_OTEL_API_KEY": "rt_fake_secret",
                "RUNTYPE_OTEL_ENDPOINT": f"http://127.0.0.1:{self.server.server_port}/v1/traces",
                "RUNTYPE_OTEL_CAPTURE_CONTENT": "true",
            },
        )
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_successful_turn_exports_canonical_root_and_model_span_with_real_content_and_usage(
        self,
    ):
        plugin = load_plugin()
        plugin.on_pre_llm_call(
            session_id="session-1",
            turn_id="turn-1",
            task_id="task-1",
            user_message="What is 2 + 2?",
            model="gpt-4o-mini",
            platform="api",
        )
        plugin.on_pre_api_request(
            session_id="session-1",
            turn_id="turn-1",
            task_id="task-1",
            api_request_id="request-1",
            api_call_count=1,
            model="gpt-4o-mini",
            provider="openai-api",
            request_messages=[{"role": "user", "content": "What is 2 + 2?"}],
        )
        plugin.on_post_api_request(
            session_id="session-1",
            turn_id="turn-1",
            task_id="task-1",
            api_request_id="request-1",
            api_call_count=1,
            model="gpt-4o-mini",
            response_model="gpt-4o-mini",
            provider="openai-api",
            usage={"input_tokens": 16, "output_tokens": 7},
            assistant_message={"content": "4"},
            finish_reason="stop",
        )
        plugin.on_post_llm_call(
            session_id="session-1",
            turn_id="turn-1",
            task_id="task-1",
            assistant_response="4",
        )
        plugin.on_session_end(
            session_id="session-1",
            turn_id="turn-1",
            task_id="task-1",
            completed=True,
            failed=False,
            interrupted=False,
        )

        self.assertEqual(len(CaptureHandler.requests), 1)
        path, headers, payload = CaptureHandler.requests[0]
        self.assertEqual(path, "/v1/traces")
        self.assertEqual(headers["Authorization"], "Bearer rt_fake_secret")
        resource = payload["resourceSpans"][0]
        resource_attrs = attributes({"attributes": resource["resource"]["attributes"]})
        self.assertEqual(resource_attrs["runtype.agent.id"], "agent_test123")
        self.assertEqual(
            resource_attrs["runtype.adapter.name"], "@runtypelabs/hermes-adapter"
        )
        root, model = resource["scopeSpans"][0]["spans"]
        root_attrs = attributes(root)
        model_attrs = attributes(model)
        self.assertEqual(root_attrs["gen_ai.operation.name"], "invoke_agent")
        self.assertEqual(root_attrs["runtype.stop_reason"], "end_turn")
        self.assertEqual(root_attrs["runtype.iterations"], "1")
        self.assertEqual(model_attrs["gen_ai.operation.name"], "chat")
        self.assertEqual(model_attrs["runtype.iteration"], "0")
        self.assertEqual(model["parentSpanId"], root["spanId"])
        self.assertEqual(model["traceId"], root["traceId"])
        self.assertEqual(model_attrs["gen_ai.usage.input_tokens"], "16")
        self.assertEqual(model_attrs["gen_ai.usage.output_tokens"], "7")
        self.assertEqual(
            json.loads(root_attrs["gen_ai.input.messages"]),
            [{"role": "user", "content": "What is 2 + 2?"}],
        )
        self.assertEqual(
            json.loads(root_attrs["gen_ai.output.messages"]),
            [{"role": "assistant", "content": "4"}],
        )
        self.assertNotIn("rt_fake_secret", json.dumps(payload))

    def test_tool_calls_without_ids_are_matched_in_order_and_payloads_stay_private_by_default(
        self,
    ):
        with patch.dict(os.environ, {"RUNTYPE_OTEL_CAPTURE_CONTENT": "false"}):
            plugin = load_plugin()
            plugin.on_pre_llm_call(
                session_id="session-2",
                turn_id="turn-2",
                user_message="private prompt",
            )
            plugin.on_pre_tool_call(
                session_id="session-2",
                turn_id="turn-2",
                tool_name="lookup",
                args={"query": "first private query"},
            )
            plugin.on_pre_tool_call(
                session_id="session-2",
                turn_id="turn-2",
                tool_name="lookup",
                args={"query": "second private query"},
            )
            plugin.on_post_tool_call(
                session_id="session-2",
                turn_id="turn-2",
                tool_name="lookup",
                result="first private result",
                status="success",
            )
            plugin.on_post_tool_call(
                session_id="session-2",
                turn_id="turn-2",
                tool_name="lookup",
                result="second private result",
                status="success",
            )
            plugin.on_session_end(
                session_id="session-2",
                turn_id="turn-2",
                completed=True,
            )

        payload = CaptureHandler.requests[0][2]
        root, first, second = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
        self.assertEqual(root["status"]["code"], 1)
        self.assertEqual(first["status"]["code"], 1)
        self.assertEqual(second["status"]["code"], 1)
        self.assertEqual(attributes(first)["gen_ai.operation.name"], "execute_tool")
        self.assertEqual(attributes(second)["gen_ai.operation.name"], "execute_tool")
        self.assertNotIn("private", json.dumps(payload))

    def test_reduced_cli_exit_hook_closes_turn_by_session_without_claiming_success(
        self,
    ):
        plugin = load_plugin()
        plugin.on_pre_llm_call(
            session_id="session-3",
            turn_id="turn-3",
            task_id="task-3",
            user_message="Stop before finishing",
            model="gpt-4o-mini",
        )
        plugin.on_session_end(session_id="session-3", completed=False, interrupted=True)

        self.assertEqual(len(CaptureHandler.requests), 1)
        root = CaptureHandler.requests[0][2]["resourceSpans"][0]["scopeSpans"][0][
            "spans"
        ][0]
        self.assertEqual(root["status"]["code"], 2)
        self.assertEqual(attributes(root)["error.type"], "interrupted")
        self.assertEqual(attributes(root)["runtype.stop_reason"], "cancelled")

    def test_process_exit_exports_unclosed_turn_as_failure(self):
        plugin = load_plugin()
        plugin.on_pre_llm_call(
            session_id="session-4", turn_id="turn-4", user_message="unfinished"
        )
        plugin._finalize_open_turns()

        self.assertEqual(len(CaptureHandler.requests), 1)
        root = CaptureHandler.requests[0][2]["resourceSpans"][0]["scopeSpans"][0][
            "spans"
        ][0]
        self.assertEqual(root["status"]["code"], 2)
        self.assertEqual(attributes(root)["error.type"], "process_exit")

    def test_process_exit_flush_stops_at_its_time_budget(self):
        plugin = load_plugin()
        for index in range(3):
            plugin.on_pre_llm_call(
                session_id=f"session-budget-{index}", turn_id=f"turn-budget-{index}"
            )
        exported = []
        clock = iter([0.0, 0.0, 6.0, 6.0])
        with (
            patch.object(plugin.time, "monotonic", lambda: next(clock)),
            patch.object(
                plugin, "_export", lambda state, timeout: exported.append(timeout)
            ),
        ):
            plugin._finalize_open_turns()

        self.assertEqual(exported, [2.0])
        self.assertEqual(plugin._TURNS, {})

    def test_session_only_exit_hook_closes_the_newest_turn_of_that_session(self):
        plugin = load_plugin()
        plugin.on_pre_llm_call(
            session_id="session-9", turn_id="stranded", user_message="a"
        )
        plugin.on_pre_llm_call(
            session_id="session-9", turn_id="current", user_message="b"
        )
        plugin.on_session_end(session_id="session-9", completed=True)

        self.assertEqual(len(CaptureHandler.requests), 1)
        self.assertEqual(
            list(plugin._TURNS), [plugin._key("stranded", "", "session-9")]
        )

    def test_content_redacts_credentials_before_truncating(self):
        plugin = load_plugin()
        plugin.on_pre_llm_call(
            session_id="session-5",
            turn_id="turn-5",
            user_message="x" * 4092 + "rt_fake_secret",
        )
        plugin.on_session_end(session_id="session-5", turn_id="turn-5", completed=True)

        payload = CaptureHandler.requests[0][2]
        root = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        message = json.loads(attributes(root)["gen_ai.input.messages"])[0]["content"]
        self.assertNotIn("rt_f", message)

    def test_export_rejects_cross_origin_redirect_before_sending_bearer_token(self):
        DestinationHandler.requests = []
        destination = ThreadingHTTPServer(("127.0.0.1", 0), DestinationHandler)
        destination_thread = threading.Thread(
            target=destination.serve_forever, daemon=True
        )
        destination_thread.start()
        redirect = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
        redirect_thread = threading.Thread(target=redirect.serve_forever, daemon=True)
        redirect_thread.start()
        RedirectHandler.destination = (
            f"http://127.0.0.1:{destination.server_port}/steal"
        )
        try:
            with patch.dict(
                os.environ,
                {
                    "RUNTYPE_OTEL_ENDPOINT": f"http://127.0.0.1:{redirect.server_port}/v1/traces"
                },
            ):
                plugin = load_plugin()
                plugin.on_pre_llm_call(
                    session_id="session-6", turn_id="turn-6", user_message="hello"
                )
                plugin.on_session_end(
                    session_id="session-6", turn_id="turn-6", completed=True
                )
            self.assertEqual(DestinationHandler.requests, [])
        finally:
            redirect.shutdown()
            redirect.server_close()
            redirect_thread.join()
            destination.shutdown()
            destination.server_close()
            destination_thread.join()

    def test_provider_retry_keeps_one_model_span_until_recovered_success(self):
        plugin = load_plugin()
        plugin.on_pre_llm_call(
            session_id="session-7", turn_id="turn-7", user_message="retry"
        )
        plugin.on_pre_api_request(
            session_id="session-7",
            turn_id="turn-7",
            api_request_id="request-7",
            api_call_count=1,
            model="gpt-4o-mini",
            provider="openai-api",
        )
        plugin.on_api_request_error(
            session_id="session-7",
            turn_id="turn-7",
            api_request_id="request-7",
            api_call_count=1,
            retryable=True,
            status_code=429,
        )
        self.assertEqual(CaptureHandler.requests, [])
        plugin.on_pre_api_request(
            session_id="session-7",
            turn_id="turn-7",
            api_request_id="request-7",
            api_call_count=1,
            model="gpt-4o-mini",
            provider="openai-api",
        )
        plugin.on_post_api_request(
            session_id="session-7",
            turn_id="turn-7",
            api_request_id="request-7",
            api_call_count=1,
            response_model="gpt-4o-mini",
            usage={"input_tokens": 20, "output_tokens": 5},
            assistant_message={"content": "Recovered"},
        )
        plugin.on_post_llm_call(
            session_id="session-7", turn_id="turn-7", assistant_response="Recovered"
        )
        plugin.on_session_end(session_id="session-7", turn_id="turn-7", completed=True)

        root, model = CaptureHandler.requests[0][2]["resourceSpans"][0]["scopeSpans"][
            0
        ]["spans"]
        self.assertEqual(root["status"]["code"], 1)
        self.assertEqual(model["status"]["code"], 1)
        self.assertEqual(attributes(model)["gen_ai.usage.input_tokens"], "20")
        self.assertEqual(
            json.loads(attributes(model)["gen_ai.output.messages"])[0]["content"],
            "Recovered",
        )

    def test_timed_out_tool_is_not_reported_as_success(self):
        plugin = load_plugin()
        plugin.on_pre_llm_call(
            session_id="session-8", turn_id="turn-8", user_message="tool"
        )
        plugin.on_pre_tool_call(
            session_id="session-8",
            turn_id="turn-8",
            tool_call_id="call-8",
            tool_name="slow_tool",
        )
        plugin.on_post_tool_call(
            session_id="session-8",
            turn_id="turn-8",
            tool_call_id="call-8",
            tool_name="slow_tool",
            status="timeout",
            result="timed out",
        )
        plugin.on_session_end(session_id="session-8", turn_id="turn-8", completed=True)

        tool = CaptureHandler.requests[0][2]["resourceSpans"][0]["scopeSpans"][0][
            "spans"
        ][1]
        self.assertEqual(tool["status"]["code"], 2)
        self.assertEqual(attributes(tool)["error.type"], "tool_error")

    def test_null_exit_reason_does_not_drop_completed_trace(self):
        plugin = load_plugin()
        plugin.on_pre_llm_call(
            session_id="session-9", turn_id="turn-9", user_message="hello"
        )
        plugin.on_session_end(
            session_id="session-9",
            turn_id="turn-9",
            completed=True,
            turn_exit_reason=None,
        )

        self.assertEqual(len(CaptureHandler.requests), 1)
        root = CaptureHandler.requests[0][2]["resourceSpans"][0]["scopeSpans"][0][
            "spans"
        ][0]
        self.assertEqual(attributes(root)["runtype.stop_reason"], "end_turn")

    def test_tool_arguments_reflect_executed_post_policy_args_not_pre_policy_args(self):
        plugin = load_plugin()
        plugin.on_pre_llm_call(
            session_id="session-10", turn_id="turn-10", user_message="tool"
        )
        plugin.on_pre_tool_call(
            session_id="session-10",
            turn_id="turn-10",
            tool_call_id="call-10",
            tool_name="lookup",
            args={"query": "secret before policy"},
        )
        plugin.on_post_tool_call(
            session_id="session-10",
            turn_id="turn-10",
            tool_call_id="call-10",
            tool_name="lookup",
            args={"query": "safe after policy"},
            result="found",
            status="ok",
        )
        plugin.on_session_end(
            session_id="session-10", turn_id="turn-10", completed=True
        )

        payload = CaptureHandler.requests[0][2]
        tool = payload["resourceSpans"][0]["scopeSpans"][0]["spans"][1]
        tool_args = json.loads(attributes(tool)["gen_ai.tool.call.arguments"])
        self.assertEqual(tool_args, {"query": "safe after policy"})
        self.assertNotIn("secret before policy", json.dumps(payload))

    def test_oversized_tool_arguments_are_omitted_instead_of_emitting_broken_json(self):
        plugin = load_plugin()
        plugin.on_pre_llm_call(
            session_id="session-11", turn_id="turn-11", user_message="tool"
        )
        plugin.on_pre_tool_call(
            session_id="session-11",
            turn_id="turn-11",
            tool_call_id="call-11",
            tool_name="write_file",
        )
        plugin.on_post_tool_call(
            session_id="session-11",
            turn_id="turn-11",
            tool_call_id="call-11",
            tool_name="write_file",
            args={"content": "x" * 5000},
            result="wrote",
            status="ok",
        )
        plugin.on_session_end(
            session_id="session-11", turn_id="turn-11", completed=True
        )

        tool = CaptureHandler.requests[0][2]["resourceSpans"][0]["scopeSpans"][0][
            "spans"
        ][1]
        self.assertNotIn("gen_ai.tool.call.arguments", attributes(tool))

    def test_cached_prompt_tokens_are_included_in_model_input_usage(self):
        plugin = load_plugin()
        plugin.on_pre_llm_call(
            session_id="session-12", turn_id="turn-12", user_message="cached"
        )
        plugin.on_pre_api_request(
            session_id="session-12",
            turn_id="turn-12",
            api_request_id="request-12",
            api_call_count=1,
            model="gpt-4o-mini",
            provider="openai-api",
        )
        plugin.on_post_api_request(
            session_id="session-12",
            turn_id="turn-12",
            api_request_id="request-12",
            api_call_count=1,
            response_model="gpt-4o-mini",
            usage={
                "input_tokens": 100,
                "prompt_tokens": 130,
                "cache_read_tokens": 20,
                "cache_write_tokens": 10,
                "output_tokens": 25,
            },
            assistant_message={"content": "Done"},
        )
        plugin.on_session_end(
            session_id="session-12", turn_id="turn-12", completed=True
        )

        model = CaptureHandler.requests[0][2]["resourceSpans"][0]["scopeSpans"][0][
            "spans"
        ][1]
        usage = attributes(model)
        self.assertEqual(usage["gen_ai.usage.input_tokens"], "130")
        self.assertEqual(usage["gen_ai.usage.cache_read.input_tokens"], "20")
        self.assertEqual(usage["gen_ai.usage.cache_creation.input_tokens"], "10")
        self.assertEqual(usage["gen_ai.usage.output_tokens"], "25")

    def test_object_tool_result_without_text_field_is_captured_as_json(self):
        plugin = load_plugin()
        plugin.on_pre_llm_call(
            session_id="session-13", turn_id="turn-13", user_message="tool"
        )
        plugin.on_pre_tool_call(
            session_id="session-13",
            turn_id="turn-13",
            tool_call_id="call-13",
            tool_name="count",
        )
        plugin.on_post_tool_call(
            session_id="session-13",
            turn_id="turn-13",
            tool_call_id="call-13",
            tool_name="count",
            args={},
            result={"count": 3},
            status="ok",
        )
        plugin.on_session_end(
            session_id="session-13", turn_id="turn-13", completed=True
        )

        tool = CaptureHandler.requests[0][2]["resourceSpans"][0]["scopeSpans"][0][
            "spans"
        ][1]
        self.assertEqual(
            json.loads(attributes(tool)["gen_ai.tool.call.result"]), {"count": 3}
        )

    def test_structured_tool_payloads_redact_configured_secret_before_json_escaping(
        self,
    ):
        secret = 'sk-test"quoted\\backslash'
        with patch.dict(os.environ, {"OPENAI_API_KEY": secret}):
            plugin = load_plugin()
            plugin.on_pre_llm_call(
                session_id="session-14", turn_id="turn-14", user_message="tool"
            )
            plugin.on_pre_tool_call(
                session_id="session-14",
                turn_id="turn-14",
                tool_call_id="call-14",
                tool_name="lookup",
            )
            plugin.on_post_tool_call(
                session_id="session-14",
                turn_id="turn-14",
                tool_call_id="call-14",
                tool_name="lookup",
                args={"nested": [{"credential": secret}]},
                result={"token": secret},
                status="ok",
            )
            plugin.on_session_end(
                session_id="session-14", turn_id="turn-14", completed=True
            )

        tool = CaptureHandler.requests[0][2]["resourceSpans"][0]["scopeSpans"][0][
            "spans"
        ][1]
        tool_attrs = attributes(tool)
        self.assertEqual(
            json.loads(tool_attrs["gen_ai.tool.call.arguments"]),
            {"nested": [{"credential": "[REDACTED]"}]},
        )
        self.assertEqual(
            json.loads(tool_attrs["gen_ai.tool.call.result"]),
            {"token": "[REDACTED]"},
        )

    def test_provider_finish_reason_uses_otlp_string_array(self):
        plugin = load_plugin()
        plugin.on_pre_llm_call(
            session_id="session-15", turn_id="turn-15", user_message="reason"
        )
        plugin.on_pre_api_request(
            session_id="session-15",
            turn_id="turn-15",
            api_request_id="request-15",
            api_call_count=1,
        )
        plugin.on_post_api_request(
            session_id="session-15",
            turn_id="turn-15",
            api_request_id="request-15",
            api_call_count=1,
            finish_reason="stop",
        )
        plugin.on_session_end(
            session_id="session-15", turn_id="turn-15", completed=True
        )

        model = CaptureHandler.requests[0][2]["resourceSpans"][0]["scopeSpans"][0][
            "spans"
        ][1]
        reason = next(
            item["value"]
            for item in model["attributes"]
            if item["key"] == "gen_ai.response.finish_reasons"
        )
        self.assertEqual(reason, {"arrayValue": {"values": [{"stringValue": "stop"}]}})

    def test_capacity_exports_incomplete_oldest_trace_and_retains_new_turn(self):
        plugin = load_plugin()
        try:
            with self.assertLogs("runtype.hermes.adapter", level="WARNING") as logs:
                for index in range(257):
                    plugin.on_pre_llm_call(
                        session_id=f"session-cap-{index}",
                        turn_id=f"turn-cap-{index}",
                        user_message="bounded",
                    )
            self.assertEqual(len(CaptureHandler.requests), 1)
            evicted_root = CaptureHandler.requests[0][2]["resourceSpans"][0][
                "scopeSpans"
            ][0]["spans"][0]
            self.assertEqual(evicted_root["status"]["code"], 2)
            self.assertEqual(
                evicted_root["status"]["message"],
                "Hermes telemetry capacity reached before turn completed",
            )
            self.assertEqual(
                attributes(evicted_root)["error.type"], "telemetry_capacity_eviction"
            )
            self.assertIn("turn-cap-256", plugin._TURNS)
            self.assertNotIn("turn-cap-0", plugin._TURNS)
            plugin.on_session_end(
                session_id="session-cap-0", turn_id="turn-cap-0", completed=True
            )
            self.assertEqual(len(CaptureHandler.requests), 1)
            plugin.on_session_end(
                session_id="session-cap-256", turn_id="turn-cap-256", completed=True
            )
            self.assertEqual(len(CaptureHandler.requests), 2)
            self.assertIn("capacity", logs.output[0].lower())
            self.assertLessEqual(len(plugin._TURNS), 256)
        finally:
            plugin._TURNS.clear()
            if hasattr(plugin, "_EVICTED_TURNS"):
                plugin._EVICTED_TURNS.clear()


if __name__ == "__main__":
    unittest.main()
