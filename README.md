# Runtype plugin for Hermes Agent

Send [Hermes Agent](https://github.com/NousResearch/hermes-agent) runs to [Runtype](https://runtype.com) as OpenTelemetry traces, so you can inspect them, review them, and turn them into eval cases.

Each Hermes turn becomes one trace:

- an `invoke_agent Hermes` span for the whole turn
- a `chat` span for each model request, with model, token usage, and finish reason
- an `execute_tool <name>` span for each tool call

The plugin only observes Hermes through its plugin hooks. It does not change model calls or tools, it has no dependencies beyond the Python standard library, and it works in both the Hermes CLI and the gateway.

## Install

```bash
hermes plugins install runtypelabs/hermes-runtype-otel --enable
```

If Hermes runs in Docker, run the command inside the container. After installing, restart a running gateway with `hermes gateway restart`.

## Configure

Set these variables in the Hermes process environment or in your profile's `.env` file:

```bash
RUNTYPE_AGENT_ID=agent_...
RUNTYPE_OTEL_API_KEY=rt_...
```

| Variable | Required | Description |
| --- | --- | --- |
| `RUNTYPE_AGENT_ID` | Yes | The Runtype agent that the traces belong to. |
| `RUNTYPE_OTEL_API_KEY` | Yes | A Runtype API key with the `TELEMETRY:WRITE` permission. |
| `RUNTYPE_OTEL_CAPTURE_CONTENT` | No | Set to `true` to include prompts, responses, and tool payloads. Defaults to `false`. |
| `RUNTYPE_OTEL_ENDPOINT` | No | An OTLP/HTTP traces URL. Defaults to `https://api.runtype.com/v1/otel/v1/traces`. |

The plugin stays inactive until both required variables are set.

## Verify

Run a single turn:

```bash
hermes chat --oneshot -q "Reply with one sentence"
```

A new execution appears on the agent in Runtype within a few seconds.

## What gets sent

By default the plugin sends identifiers, model names, token usage, timings, and status. It sends no message or tool content.

With `RUNTYPE_OTEL_CAPTURE_CONTENT=true` it also sends:

- The user message, the model's response, and each model request's recent messages. Only the last 16 messages per request are sent.
- Tool arguments and results.

You need content capture to turn a run into an eval case.

Captured text is limited to 4,096 characters per value, and tool arguments over that limit are left out. Before anything is sent, the plugin removes the values of `RUNTYPE_OTEL_API_KEY` and `OPENAI_API_KEY` and common API key and bearer token patterns. This filtering is best effort. If your prompts or tool results may contain secrets that must not leave the host, leave content capture off.

## Behavior

- **One request per turn.** The trace is sent when the turn ends. The request times out after 2 seconds. Export errors are ignored and never fail the Hermes turn.
- **Turn status.** Completed turns end with `end_turn`, and hitting the iteration limit ends with `max_turns`. Interrupted turns are recorded as `cancelled` and failed turns as `error`.
- **Provider errors.** Hermes may retry or switch providers after an API error, so one failed request doesn't end the turn. If Hermes stops without reporting that the turn ended, the trace is sent when the process exits. Pending traces get 5 seconds in total at exit, and any not sent by then are dropped.
- **Capacity.** The plugin tracks up to 256 turns at once. If that limit is reached, the oldest turn is sent early and marked incomplete.

## Limits

- A trace covers a single turn, not a whole multi-turn session.
- Auxiliary model calls and the internal steps of subagents are not included.
- The plugin reports runs that start in Hermes. It does not make Hermes callable from Runtype.

## Development

The tests use the standard library and a local HTTP receiver. They need no API key or network access:

```bash
python3 -m unittest discover -s tests -v
```

## Documentation

- [Reporting external telemetry](https://docs.runtype.com/developer-guides/guides/reporting-external-telemetry)
- [Bring your own agent](https://docs.runtype.com/developer-guides/guides/bring-your-own-agent)

## License

[MIT](LICENSE)
