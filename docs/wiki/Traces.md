# Traces

`ambits trace` exports a session as a trace: every tool call a span, from
the call to its result, every model response a span beside the calls it
made, with each subagent's calls nested under the delegation that launched
it. Open it in the viewer you already use for
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
| A prompt | when you sent it | the last end of the calls and responses made answering it |
| A model response | when its request was sent | its last block |
| A tool call | the call | its result |
| A delegation (`Agent`/`Task`) | the call | when its agent last stopped, or its subagent's last span, whichever is later |

Each prompt you typed is a span under the session, and every call made
answering it is a span under the prompt: the calls from that prompt until
the next one, and, through its delegations, their subagents' calls. Slash
commands count as prompts (`/compact`); notifications, interruptions and
text Claude Code injects do not. Calls made before any prompt hang off the
session.

A background agent returns at once (`async_launched`) and reports later, so
a delegation ends when Claude Code enqueued its last *task notification*. An
agent that was resumed with `SendMessage` stopped more than once; its
delegation covers all its rounds, idle time included.

A call with no result yet (still running, or the session ended) has no end:
it is exported with zero length and marked `ambits.open`.
A failed call, or an agent that stopped with any status but `completed`, is
an error. A call failed when its result says `is_error`, or when its only
result is text that reads as a failure: it starts `Error`, is wrapped in
`<tool_use_error>`, or is Claude Code's message for a call the user rejected.
Other plain text, such as `StructuredOutput`'s "Structured output provided
successfully", is a success.

## Model responses

Claude Code logs a response as several lines, one per block (thinking,
text, each tool call), sharing its message id; they make one span. It
starts when the request was sent: the `api-request` line newer logs name it
by, else the last line before the response that was not written with it.
It sits beside the calls it made under the same prompt or delegation, and
each of those calls links to it.

A response carries what it was and what it cost: the model (and the one
asked for, when the log names it), the stop reason, uncached input, cache
reads and writes, output and thinking tokens, how full its context was (all
of its input), when its first block came, and how it was asked (effort,
speed, service tier). The text it wrote is never exported; the
[TUI](TUI#trace-view) shows it, and Claude Code mostly keeps no thinking
text to show.

A subagent's lines often never say how a response stopped. Its output count
is then where the stream began, a floor rather than a total, and the span
is marked `ambits.inference.usage_partial`. A message Claude Code wrote
itself (model `<synthetic>`) carries no usage: an API error is an error
span with the error's kind, and anything else, such as the note an
interrupted turn gets, counts for nothing. A compaction's summary is a model
request too, but the log keeps no response for it: only its moment.

Compactions and git commits are moments rather than spans: span events in
OTLP — on the prompt they happened during, else on the session — and
`ph:"i"` events in Chrome. Commits are those on local branches or `HEAD`
whose committer time falls within the session, each with
`vcs.ref.head.revision` and `ambits.commit.subject`.

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
| `gen_ai.operation.name` | `execute_tool`, or `invoke_agent` for a prompt or a delegation |
| `gen_ai.tool.name` | `Read`, `Edit`, `Bash`, … |
| `gen_ai.tool.call.id` | the tool-use id |
| `gen_ai.agent.id` | the agent that made the call |
| `code.file.path` | project-relative file, when the call names one |
| `code.function.name` | the symbol, when the call targets one |
| `ambits.read.depth` | how deeply a read saw its target |
| `ambits.write` | `true` for a write tool |
| `ambits.subagent.id` | on a delegation, the agent it launched |
| `ambits.open` | `true` for a call with no result |
| `ambits.prompt` | `true` on a prompt's span, named by the prompt's first line |

A tool call made by a model response has a span `link` to it. A response is
a `CLIENT` span named `chat {model}`, with:

| Attribute | Value |
|---|---|
| `gen_ai.operation.name` | `chat` |
| `gen_ai.provider.name` | `anthropic` |
| `gen_ai.conversation.id` | the session |
| `gen_ai.agent.id` | the agent it answered |
| `gen_ai.request.model`, `gen_ai.response.model` | the model asked for (the one that answered, when the log does not say), and the one that answered |
| `gen_ai.response.id` | the message id |
| `gen_ai.response.finish_reasons` | its stop reason |
| `gen_ai.request.max_tokens` | the output limit asked, when the request line is logged |
| `gen_ai.usage.input_tokens` | all its input, cache reads and writes included: how full its context was |
| `gen_ai.usage.cache_read.input_tokens`, `gen_ai.usage.cache_write.input_tokens` | the cache's share of that input |
| `gen_ai.usage.output_tokens`, `gen_ai.usage.reasoning.output_tokens` | what it wrote, and the part of it spent thinking |
| `error.type` | the API error, for one |
| `ambits.inference.first_block_ms` | from the request to its first block |
| `ambits.inference.thinking_ms` | how long it thought, when the log says |
| `ambits.inference.tool_calls` | how many calls it made |
| `ambits.inference.usage_partial` | `true` when its output count is a floor |
| `ambits.inference.synthetic` | `true` for a message Claude Code wrote |
| `ambits.inference.request_id`, `effort`, `speed`, `service_tier`, `geo`, `advisor_model` | how it was asked and served |
| `ambits.inference.cache_write_5m_tokens`, `cache_write_1h_tokens`, `web_search_requests`, `web_fetch_requests` | the rest of its usage |

The GenAI conventions are still marked Development; these follow them as
of November 2026.

## Chrome trace events

The [trace event format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU)
that [Perfetto](https://ui.perfetto.dev) and `chrome://tracing` open: one
thread per agent (the session is `main`; a subagent is labelled with its
delegation's description), complete events categorised `prompt`, `read`,
`write`, `delegate`, `inference` or `other`, with a response's attributes as
its `args`, and a flow arrow from each delegation to its subagent's first
call.
