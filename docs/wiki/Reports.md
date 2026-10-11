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
a timeline with one lane per agent. Open it in any browser; it needs no
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
(`report.html#writes`), so a link can open one.

## What it carries, and what it does not

No file contents and no absolute paths: symbol ids, project-relative paths,
and the project directory's name. It does carry what the session logs said
about each call (a shell command, the first line of a prompt, an error
message), so screen it for secrets before publishing it.

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
delegation).

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
2. `span_reads`: every span's `read` list; `read_count` stays.
3. `other_spans`: spans that neither read, wrote nor delegated.

Writes, totals and the remaining spans are never cut, so a report can still
exceed the limit.
