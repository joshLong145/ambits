# Ambits

[![e2e](https://github.com/joshLong145/ambits/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/joshLong145/ambits/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/joshLong145/ambits/graph/badge.svg?token=9Q8GWA8H6Y)](https://codecov.io/gh/joshLong145/ambits)

**A code-reading tool for AI agents, and a memory of what they have read.**

Coding agents read whole files to find one function, re-read code they already know, and lose all of it the moment the context window compacts. ambits addresses both halves of that:

- **Reading** — `ambits show` returns a symbol's definition as structured JSON, addressed by name or by content hash. The agent asks for `App/process_compaction`, not for 2,000 lines of `app.rs`.
- **Remembering** — ambits records every symbol the agent reads, at what depth, throughout the session. After a compaction it can hand that history back, so the agent knows what it already understands instead of rediscovering it.

Both surfaces are plain text and JSON with no vendor coupling, so the record can be handed between agents, or between providers. Ingestion is currently built and tested against Claude Code.

There is also a live TUI, for when you want to watch what your agent is actually looking at.

![screenshot](./images/screenshot.png)

## Quick start

```bash
cargo install ambits

# Read a symbol instead of a file
ambits -p . show 'src/app.rs::App/process_compaction'

# What has this session read so far?
ambits -p . restore-context

# Hand that history back automatically after every compaction
ambits hook install --project .

# Watch it live
ambits -p .
```

---

# For the agent

## Finding symbols

```bash
ambits -p . find 'parse_selector'     # where is this defined?
ambits -p . find 'src/app.rs::'       # everything in a file
ambits -p . find '::App/'             # every member of a type
ambits -p . find 'ui::render'         # scoped to a directory
```

```
restore::classify — 2 matches
  [fn] src/restore.rs::classify  L336-404
  [fn] src/restore.rs::tests/rehydrate_and_classify_agree_on_where_a_symbol_lives  L1000-1020
```

The pattern is `[path]::[name]`, or a bare name; both halves are optional and
case-insensitive.

The path half matches whole **components**, not substrings — so `ui` matches
`src/ui/` but not `src/tui.rs`, while `app` still matches `app.rs` and
`ui/stats` matches `src/ui/stats.rs`.

The name half matches the **leaf** name, unless the pattern contains `/`, in
which case it matches the whole name path. That distinction matters at scale:
on this repo `test` matches 42 symbols by leaf but 486 by full path, since
every `tests/…` child matches through its parent. Writing `::App/` opts into
path matching deliberately, which is how you ask for a type's members.

Results carry **coverage context** when the TUI has been running: a depth
column showing what this session already read, `—` for unread, and a per-query
count. That answers the question a search is usually a step toward — do I need
to read this?

```
fmt:: — 6 matches (5 read)
  [fn ] src/fmt.rs::tokens  L10-18  full
  [mod] src/fmt.rs::tests  L36-62  full

parser/typescript::extract — 5 matches (0 read)
  [fn] src/parser/typescript.rs::extract_symbols  L157-245  —
```

Without a coverage journal the column is omitted entirely rather than shown
empty — "unknown" and "unread" are different answers, and only one of them
means go read it. In JSON, a `coverage` object on the envelope is what
distinguishes them; `show` carries the same annotation per match.

Results are capped at 100 per pattern (`--limit`), and a truncated result says
how many it withheld. `--format json` emits the same fields as `show --no-body`, so `find` feeds
straight into `show` — except that `children` comes back as a `children_count`,
since listing them made a broad search 72% child ids by byte.

Unlike grep, results carry their kind — `struct`, `impl`, `fn` — so a name that
appears as a type, its impl block, and a method inside it comes back as three
labelled, distinguishable entries. But this searches **definitions, not
usages**: a method call is not a symbol, so `find is_none_or` returns nothing.

## Finding callers

```bash
ambits -p . callers centered_rect
```

```
centered_rect — 2 call sites in 2 callers
  src/ui/alignment.rs::render  (src/ui/alignment.rs:16)
  src/ui/compaction.rs::render  (src/ui/compaction.rs:15)
```

Call sites come from the grammar's own tags query, so a mention in a comment or
inside a string literal is never reported — the answer is a call node or it is
not there. Each site is attributed to the innermost symbol containing it, and
that id goes straight into `show`.

**Matching is by name, not by resolution.** tree-sitter parses; it does not do
type inference, so a call to `new()` cannot be tied to one of the twelve
definitions named `new`. On this repo 838 of 916 function names are unique, so
most answers are exact — but `callers new` returns every call to anything named
`new`. `--format json` sets `name_matched_only: true` so a consumer cannot
mistake this for a resolved call graph.

References are extracted on demand rather than stored, so `find`, `show`, and
the TUI pay nothing for this. It costs about 0.8s on this repo against 0.1s for
`find`.

## Reading code by symbol

```bash
ambits -p . show 'src/digest.rs::format_tokens'
```

```json
{"schema_version":1,"results":[{"query":"src/digest.rs::format_tokens","selector":"id",
"matches":[{"id":"src/digest.rs::format_tokens","name":"format_tokens","file":"src/digest.rs",
"lines":[143,149],"bytes":[5621,5766],"content_hash":"b3:178ab30e…","label":"fn",
"estimated_tokens":56,"definition":"fn format_tokens(n: u64) -> String {\n …"}]}]}
```

A selector is either a **symbol id** — `<path>::<name-path>`, exactly what `restore-context` prints — or a **content hash**, full or an 8-character prefix. Several resolve per invocation, so a batch of lookups costs one process:

```bash
ambits -p . show b3:5a60f75c 'src/digest.rs::grouped' 'src/app.rs::App/handle_key'
```

`definition` is the exact source span, sliced by byte offset rather than reconstructed from line numbers. `--no-body` returns location metadata only; `--max-bytes N` caps each definition and flags it `"truncated": true`, since a cut definition is no longer valid source.

**Ambiguity is reported, not resolved.** `matches` is an array, because ids are
not guaranteed unique — Rust allows a type several inherent impl blocks in one
file, and nothing in the name distinguishes them. A content hash always names
exactly one symbol. Empty `matches` means no such symbol; `"selector": "unrecognized"` means the query was neither an id nor a hash. The command exits `0` either way: "nothing matches" is an answer, not a failure.

In practice this is a large saving. Reading the six implementation symbols of a 316-line module costs ~630 tokens against ~3,500 for the file.

## Knowing what it has already read

Every read is tracked per symbol and per agent, at the depth the tool implies — a `Read` gives full body, a grep match gives overview, a glob gives name only. `restore-context` reports that history:

```bash
ambits -p . restore-context
```

```
### src/app.rs — 85 symbols (~40.4k tok)
App/process_compaction:340-388, App/rebuild_tree_rows:391-458,
App/handle_key:460-533, …

### src/digest.rs — 43 symbols (~11.6k tok)
format_tokens:143-149 (was src/ui/stats.rs), grouped:63-83, …
```

Entries are `name:first-last`. Line numbers come from a fresh scan at print time rather than from storage, so they stay correct in files edited since the read — the agent can read that range directly instead of pulling the file.

`(was src/ui/stats.rs)` marks a symbol that moved between files. ambits identifies symbols by content as well as by path, so hoisting a helper into a shared module does not lose it.

`--max-tokens N` fits a budget (default 3000). `--format json` gives the same data structurally, including each symbol's `content_hash` for exact `show` lookups.

### Automatic hand-back after compaction

```bash
ambits hook install --project .
```

Registers a `SessionStart` hook with `matcher: "compact"` in `.claude/settings.json`, so Claude Code runs `restore-context` and injects the result the moment a compaction completes. It merges into existing settings, is safe to re-run, and emits nothing when there is nothing to restore.

### Lookups count as reads

ambits parses `show` invocations out of the session log and credits the symbols they name, so reading efficiently costs nothing in coverage versus a plain `Read`. (`--no-body` credits name-level only — the agent learned where a symbol is, not what it says.)

Credit is best-effort: it is reconstructed from the logged command text, so a selector passed through a shell variable or command substitution is not visible. It fails toward under-reporting, never over.

## Portability

Nothing in the record is tied to a vendor or a machine:

| Piece | Form |
|---|---|
| Symbol ids | `<project-relative-path>::<name-path>` |
| Content hashes | BLAKE3 over whitespace-normalized source |
| Digest | Markdown, or schema-versioned JSON |
| `show` output | Schema-versioned JSON |
| Journal | NDJSON, one record per read |

The same digest means the same thing on another checkout, another machine, or in front of another model. Any agent that can run a command and read text can consume it — no MCP server, no SDK, no wire protocol.

What *is* provider-shaped, and where the seams are:

- **Session ingestion** is Claude Code's JSONL format today. `SessionIngester` is the extension point — "implement this to add support for a new LLM session format."
- **Tool mappings** are data, not code. Another provider's tool names are taught in `.ambit/tools.toml` rather than patched in; `ToolCallMapper` exists to "plug in alternative tool-name conventions."
- **`--format hook`** emits Claude Code's `SessionStart` envelope specifically. `--format markdown` and `--format json` carry the same content with no envelope.

Claude Code is what this is built and tested against. The formats are deliberately boring so that need not stay true.

---

# For you

## The TUI

```bash
ambits -p .
```

Tails the session log and updates live. Three panels — symbol tree, coverage stats, activity feed — cycled with `Tab`.

- **Depth-aware coloring** — every symbol shaded by how deeply it was read
- **Per-file counts** — `seen/total` on each file header, so partial coverage shows without expanding
- **Sortable tree** — alphabetical, or grouped by coverage to surface half-read files first
- **Search** — `/` to jump to a symbol by name
- **Compaction history** — `C` for this session's compaction boundaries
- **Sub-agent alignment** — `d` compares two agents file by file: where they read the same code, and where only one looked

Symbols carried over from before a compaction render dimmed — the read happened, but it is no longer in the agent's live context.

### Keybindings

| Key | Action |
|---|---|
| `j` / `k` | Navigate up/down (tree or agent list, depending on focus) |
| `h` / `l` | Collapse / expand tree nodes |
| `Enter` | Expand node, or select agent when Stats is focused |
| `Tab` | Cycle panel focus (Tree / Stats / Activity) |
| `/` | Search symbols |
| `s` | Toggle sort (alphabetical / coverage) |
| `a` / `A` | Cycle agent filter forward / backward |
| `d` | Sub-agent alignment view |
| `C` | Compaction history (`[` / `]` to page) |
| `g` / `G` | Jump to first / last |
| `PgUp` / `PgDn` | Scroll by page |
| `q` | Quit |

### Color legend

**Symbols**, by read depth:

| Color | Meaning |
|---|---|
| Dark gray | Unseen |
| Light gray | Name only (appeared in a glob or listing) |
| Pale blue | Overview (grep match, symbol listing) |
| Blue | Signature seen |
| Green | Full body read |

**File headers**, by coverage:

| Color | Meaning |
|---|---|
| White | Nothing seen |
| Amber | Partially covered |
| Yellow-green | All symbols seen, not all at full depth |
| Green | Every symbol read in full |

## Coverage reports

```bash
ambits -p . --coverage
```

```
Coverage Report (session: 34e212cf-…)
─────────────────────────────────────────────────────────────────────────────
File                                      Symbols    Seen    Full   Seen%   Full%
─────────────────────────────────────────────────────────────────────────────
src/events.rs                                   3       3       3    100%    100%
src/parser/mod.rs                               8       8       1    100%     12%
src/app.rs                                     89      89      85    100%     95%
…
─────────────────────────────────────────────────────────────────────────────
TOTAL                                         214     182     175     85%     82%
```

- **Seen%** — symbols the agent has any awareness of
- **Full%** — symbols read completely

```bash
ambits -p . --coverage --format json | jq '.totals.full_percent'
```

## Multi-agent sessions

When a session spawns sub-agents with the Task tool, ambits tracks each independently:

```
Agents: 5
  ▶ [All]              Seen: 95%
  ├─ 7842313b          35%
  │  ├─ a38e68c        20%
  │  ├─ a9fe23c        41%
  │  └─ a845182        15%
  └─ compact-0aff      10%
```

`Tab` to the Stats panel, `j`/`k` to move, `Enter` to filter — tree, activity feed, and depth breakdown all follow. `a` cycles agents from any panel. `d` opens the alignment view, which scores each pair of agents file by file — useful for spotting sub-agents that duplicated each other's exploration.

Outside the TUI:

```bash
ambits -p . --coverage --agent a9fe23c
ambits -p . --coverage --agent a9fe        # prefix match
```

A prefix matching no agent, or several, warns rather than guessing.

---

# Configuration

## The read journal

While the TUI runs it maintains an append-only NDJSON record at `.ambit/coverage/<session>.ndjson` — one entry per `(symbol, agent)` read. This is what lets `restore-context` answer after the fact, and it survives restarts.

```bash
ambits -p . cache status              # sessions, symbols, size on disk
ambits -p . cache clear --session <id>
ambits -p . cache clear --all
```

Size is bounded by what an agent can read in one session, not by repository size — a full day of heavy work on this repo is around 150 KB. Nothing is pruned automatically, and `cache clear` requires naming a target, because a journal is the only record of what a past session read.

Disable with `--no-journal`; tune the write interval with `--flush-interval-ms`.

## Restricting scope

```bash
ambits -p . --filter src/parser              # by path component
ambits -p . --filter-regex '^src/.*\.rs$'    # by regex
```

`--filter` matches whole path components, so `src/parser` matches `src/parser/rust.rs` but not `src/parser_extra.rs`.

## Tool mappings

How a tool call becomes a symbol read is data. Drop a `.ambit/tools.toml` into your project to teach ambits a tool it does not know, or to change the depth an existing one grants:

```toml
version = 1

[[tool]]
names         = ["MyCustomReader"]
path_keys     = ["path"]
depth         = { type = "fixed", value = "FullBody" }
description   = "MyCustomReader {path}"
```

Project config merges over the built-in defaults; a user-global config is picked up automatically. `--tools-config` points at a specific file.

## Parsing backends

| Backend | Languages |
|---|---|
| Tree-sitter (default) | Rust, Python, TypeScript |
| Serena MCP | Any language [Serena](https://github.com/oraios/serena) supports |

```bash
ambits -p . --serena
```

## Claude Code skill

```bash
ambits skill install --global      # all projects
ambits skill install               # current project
ambits skill install --project /path/to/project
```

Installs a [skill](https://code.claude.com/docs/en/skills) that teaches the agent when to check its own coverage and how to fetch definitions. Global installs go to `~/.claude/skills/ambit/`, project installs to `.claude/skills/ambit/`.

# CLI reference

| Command | Description |
|---|---|
| `ambits -p <path>` | Launch the TUI |
| `ambits … find <pattern>…` | Search symbols by `[path]::[name]` |
| `ambits … callers <name>…` | List call sites and their enclosing symbol |
| `ambits … show <selector>…` | Print symbol definitions as JSON |
| `ambits … restore-context` | Print this session's read history |
| `ambits … --coverage` | Print a coverage report and exit |
| `ambits … --dump` | Print the symbol tree and exit |
| `ambits … cache status\|clear` | Inspect or remove read journals |
| `ambits hook install` | Register the post-compaction hook |
| `ambits skill install` | Install the Claude Code skill |

| Flag | Description |
|---|---|
| `--project`, `-p` | Project root (required) |
| `--session`, `-s` | Session ID (auto-detects latest) |
| `--agent`, `-a` | Filter to one agent ID (prefix matching) |
| `--filter` / `--filter-regex` | Restrict analysis to a subpath or regex |
| `--format` | `table` (default) or `json` |
| `--serena` | Use Serena's LSP symbol cache |
| `--tools-config` | Custom tool-mapping TOML |
| `--no-journal` | Disable the read journal |
| `--flush-interval-ms` | Journal write interval |
| `--log-dir` | Claude Code log directory (auto-derived) |
| `--log-output` | Write processed events to a directory |

# Building from source

Requires Rust 1.82+ (declared as `rust-version` in `Cargo.toml`).

```bash
cargo build --release
cargo test
```
