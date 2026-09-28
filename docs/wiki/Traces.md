# Traces

`ambits trace` exports a session as a trace: every tool call a span, from
the call to its result, with each subagent's calls nested under the
delegation that launched it. Open it in the viewer you already use for
distributed traces or profiles.

```bash
ambits -p . trace > session.otlp.json                    # OTLP/JSON (default)
ambits -p . trace --format chrome > session.trace.json   # Chrome trace events
ambits -p . --session <id> --agent a1b2 trace            # one subagent's subtree
```

The [TUI's trace view](TUI#trace-view) (`t`) shows the same trace live.

The trace is rebuilt from the Claude Code logs of the session (`--session`, or
the latest), including its subagents' logs. Nothing is stored.

## What a span is

| Span | Start | End |
|---|---|---|
| Session (the root) | first call | last end |
| A tool call | the call | its result |
| A delegation (`Agent`/`Task`) | the call | when its agent last stopped, or its subagent's last span, whichever is later |

A background agent returns at once (`async_launched`) and reports later, so
a delegation ends when Claude Code enqueued its last *task notification*. An
agent that was resumed with `SendMessage` stopped more than once; its
delegation covers all its rounds, idle time included.

A call with no result yet (still running, or the session ended) has no end:
it is exported with zero length and marked `ambits.open`.
A failed call, or an agent that stopped with any status but `completed`, is
an error.

Compactions are instants on the session: span events in OTLP, `ph:"i"`
events in Chrome.

## OTLP/JSON

The [OTLP/JSON encoding](https://opentelemetry.io/docs/specs/otlp/#json-protobuf-encoding)
of one `ExportTraceServiceRequest`, so a collector's `otlpjsonfile` receiver
or any OTLP/HTTP endpoint accepts it:

```bash
curl -X POST -H 'Content-Type: application/json' \
  --data-binary @session.otlp.json http://localhost:4318/v1/traces
```

Ids are derived from the session and the tool-use ids, so exporting twice
gives the same trace. Attributes follow the
[GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
where one fits:

| Attribute | Value |
|---|---|
| `gen_ai.operation.name` | `execute_tool`, or `invoke_agent` for a delegation |
| `gen_ai.tool.name` | `Read`, `Edit`, `Bash`, … |
| `gen_ai.tool.call.id` | the tool-use id |
| `gen_ai.agent.id` | the agent that made the call |
| `code.file.path` | project-relative file, when the call names one |
| `code.function.name` | the symbol, when the call targets one |
| `ambits.read.depth` | how deeply a read saw its target |
| `ambits.write` | `true` for a write tool |
| `ambits.subagent.id` | on a delegation, the agent it launched |
| `ambits.open` | `true` for a call with no result |

## Chrome trace events

The [trace event format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU)
that [Perfetto](https://ui.perfetto.dev) and `chrome://tracing` open: one
thread per agent (the session is `main`; a subagent is labelled with its
delegation's description), complete events categorised `read`, `write`,
`delegate` or `other`, and a flow arrow from each delegation to its
subagent's first call.
