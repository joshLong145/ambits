# Reports

`ambits report` writes one session as a single JSON document: its trace,
what each agent read of every symbol, and every write with whether it is
still on disk and which commit it landed in. It is made for a viewer that has
neither the session logs nor the checkout, such as a page published after an
agent harness's run.

```bash
ambits -p . report > report.json
ambits -p <clone> --log-dir <logs> -s <session> --state-dir <dir> report --max-bytes 1000000
```

The report is rebuilt from the session's logs, as the [TUI](TUI) does at
launch, and folds in the session's [journal](Configuration#the-read-journal)
when there is one: its reads keep the content hash they were journaled with,
and its writes are included even if the logs have since lost them. It writes
nothing. To have the journal hold everything first, run `cache sync`.

## As a page

`--format html` writes the same document as one self-contained HTML page:
a summary, the files with each symbol's read depth and the writes that
touched it, the writes grouped by the commit they landed in, and the trace as
a timeline with one lane per agent above every call. A call expands in place
when clicked, as in the TUI: its arguments, and the change it made, the
text it read with line numbers, or what it printed. Open it in any browser; it needs no
server.

```bash
ambits -p . report --format html > report.html
ambits report --format html --from-json screened.json > report.html
```

`--from-json` renders a document `report` printed earlier, exactly as
given, without reading any session, log or checkout: build the JSON, screen
it, then render the screened copy. It takes no `--max-bytes`; cut the JSON
when it is built.

The page loads nothing. Its Content-Security-Policy is `default-src 'none'`
with only its own script and stylesheet allowed, by hash, and it shows every
value from the document as text, never as markup. The tab is in the URL
(`report.html#writes`), and `#trace:<span id>` opens one call.

## What it carries

Everything the session's logs said about each call, as the
[TUI's call view](TUI) shows it: its arguments, the diff of every write, the
text every read returned, and every other call's output (a command's, a
search's, a subagent's answer). A report is therefore as sensitive
as the transcript itself. It holds file contents, and a command or a call's
description says whatever it said, secrets included, so screen it before
publishing it. Unlike a [snapshot](Snapshots), which never holds contents.

Files are named by project-relative path, path arguments included, and the
project by its directory's name, never its path.

## Fields

`schema_version` (1) is bumped on any breaking change.

| Field | What it holds |
|---|---|
| `session_id`, `ambits_version`, `generated_at` | Which session, made by which ambits, when (RFC 3339) |
| `project` | `name`, the project directory's name; `head`, the commit checked out, or `null` |
| `totals` | `files`, `symbols`, `symbols_read`, `symbols_read_full`, `symbols_written`, `writes`, `spans`, `agents` |
| `truncated` | What `--max-bytes` cut, in order; empty when nothing was |
| `agents` | `id`, `label`, `parent` (`null` for the session's own agent) |
| `spans` | One per tool call or prompt, as in [Traces](Traces) |
| `instants` | `compaction`, `commit` (`sha`, `subject`) and `snapshot` events, by `t_ms` |
| `files` | Every file, with `symbols` listed for those something read or wrote |
| `writes` | Every attributed write, oldest first |

A span: `i` (its id, which `parent` and a write's `span` refer to), `parent`,
`agent`, `kind` (`prompt`, `read`, `write`, `delegate`, `other`), `tool`,
`name`, `detail`, `file`, `symbol`, `start_ms`, `end_ms` (`null` while it had
no result), `error`, `message`, `depth` (for a read), `read` (the symbols it
was credited with, each with its depth), `read_count`, `child_agent` (for a
delegation), `args` and `content`.

`args` lists the call's arguments as `key`, `value` and `bulk`; a `bulk`
argument held file contents the call wrote and is summed up (`8 lines · in
the change below`), since `content` shows them. `content` is `null`, or:

- `{"type": "change", "exact", "cut", "hunks"}` for a write: each hunk's
  `old_start`, `new_start` and `lines` as `[op, text]` with `op` one of ` `,
  `+`, `-`. `exact` when the hunks are the tool's own patch; otherwise they
  are what the call asked for and carry no line numbers.
- `{"type": "read", "start", "lines", "cut"}` for a read: the text it
  returned, its first line numbered `start`.
- `{"type": "output", "lines", "cut"}` for any other call: what it
  returned, failed calls included.

`cut` counts lines left out past the TUI's limit of 5000 per call.

A symbol, under its file: `id`, `name`, `kind` (the language's word: `fn`,
`class`, `def`), `level` (0 for top level), `lines` (1-based, inclusive),
`reads` (each agent's depth), `writes` (the `op`s that touched, created or
removed it).

A write: `op`, `agent`, `time`, `tool`, `file`, `level` (`symbol` or `file`,
see [Agent Writes](Agent-Writes)), `outside_symbols`, `symbols` (each `id`
with its `status`), `created`, `removed`, `status` for the whole file
(`current`, `changed`, `removed`, `unknown`), `landed` (as
[`touched`](Agent-Writes) reports it), and `span`.

## Size

`--max-bytes N` cuts the document until it fits, in this order, naming each
cut in `truncated`:

1. `unwritten_file_symbols`: the symbol lists of files nothing wrote.
2. `read_text`: the text reads returned.
3. `output`: every other call's output.
4. `span_reads`: every span's `read` list; `read_count` stays.
5. `arguments`: every call's `args`.
6. `diffs`: every write's `content`.
7. `other_spans`: spans that neither read, wrote nor delegated.

Writes, totals and the remaining spans are never cut, so a report can still
exceed the limit.
