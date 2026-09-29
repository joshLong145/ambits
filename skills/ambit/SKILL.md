---
name: ambit
description: "Coverage-aware agent workflow tool. Use PROACTIVELY to search and read code and Markdown docs — `ambits rg`/`ambits grep` find matches and name the symbol or section each lands in, `ambits show` fetches exactly that definition — and before modifying files, making architecture decisions, debugging, or reviewing code. Reports which symbols Claude has read (Seen%) vs read in full (Full%). Use MCP tools mcp__ambit__coverage or mcp__ambit__coverage_file when available; fall back to bash."
allowed-tools: Bash(ambits *)
---

# /ambit - Coverage-Aware Agent Workflow

Ambit tracks which parts of a codebase this agent has actually read, at symbol
resolution. Use it to verify you have enough context before acting — and to
identify blind spots before making changes that touch unfamiliar code.

## PROACTIVE USE: When to Check Without Being Asked

**BEFORE modifying a file** — if you haven't verified you've read it this session:
```
mcp__ambit__coverage_file  →  check the specific file(s) you're about to change
```

**BEFORE architectural decisions** — if you're proposing design changes across
multiple files:
```
mcp__ambit__coverage  →  check overall module coverage; proceed only if Seen% > 60%
```

**When a bug is hard to reproduce or diagnose** — low coverage on the relevant
file is a likely root cause of bad recommendations:
```
mcp__ambit__coverage_file  →  if Full% < 50%, read the file before diagnosing
```

**After receiving a multi-file task** — check coverage of the involved files
before starting; flag which ones need reading first.

**When your suggestion might be wrong** — if a user pushes back, check coverage.
You may have missed implementation details.

**When looking for code or docs** — search with `ambits rg`, then `show` the
symbol or Markdown section it found, instead of `Grep` and a whole-file `Read`
(see [Finding and Reading Code and Docs](#finding-and-reading-code-and-docs)).

## Finding and Reading Code and Docs

Search, then fetch exactly what the search found — not the whole file:

```bash
ambits -p . rg 'fn enclosing'                                   # 1. find it
# src/symbols/mod.rs:158:9:[— FileSymbols/enclosing]     pub fn enclosing(…
ambits -p . show 'src/symbols/mod.rs::FileSymbols/enclosing'    # 2. read that symbol
```

Every match names the symbol it landed in and how deeply you have read it, so
the search tells you what to fetch and whether you need to. `show` returns
that one definition — a function, a type, or a section of a Markdown file —
and **counts as reading it**; a `Read` of the whole file would cost the rest
of the file too. Use `ambits rg` in place of your `Grep` tool and of shell
`grep`/`rg`, and `show` in place of `Read` whenever you know the symbol.

### Searching: `ambits rg`

ripgrep's flags — the dialect your `Grep` tool is built on:

```bash
ambits -p . rg 'depth_of'                    # every use and definition
ambits -p . rg -F 'Vec<(String, u32)>'       # a literal, no regex escaping
ambits -p . rg -w -i 'journal'               # whole word, any case
ambits -p . rg 'fn enclosing' -t rust        # one file type (-t md for Markdown)
ambits -p . rg 'TODO' -g '!tests/**'         # globs; ! excludes
ambits -p . rg 'Journal::open' -C 3          # context: -A after, -B before, -C both
ambits -p . rg 'unwrap\(\)' -c               # matching lines per file
ambits -p . rg 'Matcher' -l                  # just the files
ambits -p . rg 'fn [a-z_]+' -o src/text.rs   # only the matched text, scoped to a path
```

Scope a broad search first (`-l`, `-c`), then narrow: output is capped at
**200 matches** (`--head-limit 0` for all, `-m N` per file) and lines at 300
columns (`-M 0`), and all of it lands in your context.

Each match is `file:line:column:[depth symbol] line`:

| Bracket | Meaning |
|---|---|
| `[full name]` | You have read this symbol in full: no need to `show` it |
| `[signature name]`, `[overview name]`, `[name name]` | Seen at that depth — `name` is what an earlier search earns — but not its body: `show` it before relying on it |
| `[— name]` | You have **not** seen it |
| `[name]` | No coverage journal loaded: *unknown*, not unread |
| `[-]` | Not inside any symbol — a `use` line, or a file no parser handles |

The `show` id is the file and the bracket's name path joined by `::` —
`src/symbols/mod.rs:158:9:[— FileSymbols/enclosing]` is
`src/symbols/mod.rs::FileSymbols/enclosing` — or take `symbol.id` verbatim
from `--json` (ripgrep's JSON Lines, with a `symbol` on each match and a
`coverage` object on the summary).

- **Exit codes are grep's**: `0` matched, `1` nothing matched, `2` error — safe
  to chain with `&&`.
- Results are sorted by path, line, column. Every text file is searched, not
  only parseable ones: a hit in TOML is real, it just has no symbol.
- Not supported: `-P/--pcre2`, `-r/--replace`, `-f/--file`. No backreferences
  or lookaround — ripgrep's regex engine, and its limits.

**`ambits grep`** is the same search with GNU grep's flags, for when you are
writing grep by habit. The two give the same letters opposite meanings (`-L`,
`-z`, `-r`), so they are separate commands. Under `grep`, line numbers are
opt-in (`-n`), there is no column, `-h` is `--no-filename` (help is `--help`),
and `-P` is refused rather than silently matching something else:

```bash
ambits -p . grep -rn 'break-lock' docs/
```

### Markdown: sections are symbols

In a Markdown file every heading is a symbol, nested under the headings above
it, so the same search-then-fetch works on documentation:

```bash
ambits -p . rg '^#{1,3} ' docs/wiki/Sharing.md          # its outline, with what you have read
ambits -p . rg -t md 'break-lock'                        # which sections mention it
# docs/wiki/Sharing.md:53:51:[— Sharing/Push] … `ambits push --break-lock` removes it …
ambits -p . show 'docs/wiki/Sharing.md::Sharing/Push'    # that section, heading to the next
```

A section's id is the file and its heading path: `# Sharing` › `## Push` is
`docs/wiki/Sharing.md::Sharing/Push`. Fetching the top heading returns the
whole document, so fetch the section you need. `ambits -p . --dump --depth 2`
outlines every file (Markdown headings included) with line ranges and token
estimates.

### Fetching: `ambits show`

```bash
ambits -p . show 'src/app.rs::App/process_compaction'
ambits -p . show 'src/digest.rs::grouped' 'src/app.rs::App/handle_key'   # batched: one call
ambits -p . show b3:2b6c1e45                                               # by content_hash, 8+ characters
ambits -p . show --no-body 'docs/wiki/TUI.md::TUI'                        # where, and how big
ambits -p . show --max-bytes 4000 'src/main.rs::main'                     # capped
```

It returns JSON: `id`, `name`, `lines`, `bytes`, `content_hash`, `label`
(`fn`, `struct`, `h2`, …), `estimated_tokens`, and `definition` — the exact
source span. Check a large one's size with `--no-body` first; a definition cut
by `--max-bytes` is flagged `"truncated": true`, as it is no longer valid source.

**Ambiguity is reported, not resolved.** `matches` is an array: ids are not
unique (a type may have several inherent impl blocks in one file). Prefer the
hash when you need exactly one. An empty `matches` means no such symbol;
`"selector": "unrecognized"` means the query was neither an id nor a hash.

### What counts as reading

Coverage is credited from the session log, for what the tool calls could have
put in front of you:

| Command | Credited |
|---|---|
| `show <id>` | Each symbol named, as read in **full** |
| `show --no-body <id>` | Each symbol named, at **name** depth only |
| `rg` / `grep` | Each symbol a printed match sits in, at **name** depth — seen, not read |

A search counts toward Seen%, never Full%: after one, `show` what you need
rather than reasoning from the matched lines — it is both the complete
definition and the credit for having read it. `-l`, `-c` and `-q` print no
symbols and credit nothing.

Each `ambits` invocation in a command is credited on its own (`ambits show A
--no-body && ambits show B` reads `B` in full), found by the word `ambits` —
so call it by that name, not through a shell variable.

### Finding callers

```bash
ambits -p . callers centered_rect
```

Each call site and the symbol containing it, as an id `show` takes. Comments
and strings are never reported — the answer comes from parsed call nodes, not
text — so prefer it over `rg` when you want calls specifically. Matching is by
callee **name**: `callers new` returns calls to every `new` (`--format json`
marks this `name_matched_only: true`).

## Decision Thresholds

### Threshold Arguments

Thresholds can be passed directly when invoking the skill:

```
/ambit --proceed=80 --adequate=50
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `--proceed=N` | 80 | Minimum Full% to proceed without reading more |
| `--adequate=N` | 50 | Minimum Full% for targeted/interface-only changes |
| `--module=N` | 60 | Minimum Seen% across a module for architectural tasks |

If no thresholds are provided and the task is non-trivial, **ask the user**:
> "What coverage level is acceptable before I proceed? (default: 80% full body)"

Use the user's answer for all subsequent threshold decisions in the session.

### Applying Thresholds

| Full% on a file vs `--proceed` | Action |
|-------------------------------|--------|
| ≥ proceed                     | Sufficient context — proceed |
| ≥ adequate, < proceed         | Adequate for targeted changes — note gaps |
| < adequate                    | **Read the file before making changes** |
| 0%                            | **Do not make recommendations without reading first** |

For architectural or refactoring tasks, also check module-level Seen% against `--module`.

## Primary Interface: MCP Tools

Prefer these over bash when the MCP server is available (check `mcp__ambit__*`
in your allowed tools):

| Tool | When to use |
|------|-------------|
| `mcp__ambit__coverage` | Overall project coverage for the current session |
| `mcp__ambit__coverage_file` | Coverage for one specific file — fastest check before editing |
| `mcp__ambit__symbol_tree` | Full symbol tree — use when you need to understand project structure |
| `mcp__ambit__list_sessions` | Find available sessions — use when session context is unclear |

## After a Compaction

When context has just been compacted, the summary describes what a model
*remembered* while it was losing that context. `restore-context` instead
reports what this session demonstrably read:

```bash
ambits -p . restore-context
```

Treat the symbols it lists as known — no need to re-read them. Anything it does
not list is not covered, and a `Not included` line names files holding symbols
it withheld, so you can read those directly.

A heading marked `reconstructed from session logs` means the coverage journal
was unavailable and the history was rebuilt from the logs afterwards, rather
than recorded as the session ran.

Entries are listed as `name:first-last` — current line numbers, taken from a
fresh scan rather than stored, so they are accurate even for files edited since
the read. Use them: read that range directly instead of re-reading the file or
spending a `find_symbol` call to locate the symbol.

An entry marked `(was src/old.rs)` moved since you read it. Its body is
byte-identical — you still know it — but the address you remember is stale, so
use the one given. Renames are not tracked this way and will appear as
no-longer-present instead.

Use `--max-tokens N` to fit a budget (default 3000), and `--format json` for
programmatic use.

### Automatic injection

To have this happen without being asked, register the hook once:

```bash
ambits hook install --project .
```

That adds a `SessionStart` hook with `matcher: "compact"` to
`.claude/settings.json`, so Claude Code runs `restore-context` and injects the
result the moment a compaction completes. It merges into existing settings and
is safe to re-run. When there is nothing to restore it emits nothing.

### Inspecting the journal

```bash
ambits -p . cache status        # sessions, symbols, size on disk
ambits -p . cache clear --session <id>
ambits -p . cache clear --all
```

Nothing is deleted automatically. A journal is the only record of what a
session read *and what the code looked like at the time*, so removing one
means any later restore of that session must be reconstructed from logs
instead.
Journals are small — one entry per symbol read, bounded by what an agent can
read in a session, not by repository size.

## Bash Fallback

If MCP tools are unavailable:

```bash
# Coverage for the current session
ambits -p . --coverage

# Coverage for a specific file (pipe and grep)
ambits -p . --coverage | grep "src/my_file.rs"

# Specific session
ambits -p . --coverage --session <session-id>

# Full symbol tree
ambits -p . --dump
```

## Reading the Output

```
File                              Seen%    Full%
──────────────────────────────────────────────────────────
src/app.rs                        100.0%   85.0%   ← well understood
src/parser/rust.rs                 40.0%   10.0%   ← blind spot
src/ingest/mod.rs                  0.0%    0.0%   ← unread
```

- **Seen%** — symbols viewed at any depth (name, signature, overview, or full body)
- **Full%** — symbols where the complete implementation was read

A file at 100% Seen / 10% Full means you saw the signatures but not the bodies.
For anything you're actively modifying, Full% matters more than Seen%.

## Interpreting Low Coverage

**High Seen%, Low Full%** — You know the structure but not the implementations.
Safe for interface-only changes; risky for behaviour changes.

**Low Seen%, Low Full%** — Genuine blind spot. Read the file before touching it.

**Specific symbol at 0%** — If the task involves that symbol, read it first:
`ambits show <id>`, Read, or Serena's `find_symbol` with `include_body: true`.

## Coverage Improvement Loop

If coverage on files you need is insufficient:

1. Identify low-coverage files with `mcp__ambit__coverage_file` or `ambits --coverage`
2. Read the specific symbols you need: `ambits show <id>` (ids from `rg`, or
   from `restore-context`), `find_symbol` with `include_body: true`, or `Read`
   for the full file
3. Re-check — coverage updates immediately after each read
4. Proceed once thresholds are met

## Flags Reference (Bash)

| Flag | Description |
|------|-------------|
| `-p` | Project root path (required) |
| `-s` | Session ID (auto-detects latest) |
| `--coverage` | Print coverage report |
| `--dump` | Print symbol tree |
| `--serena` | Use Serena LSP symbols (more languages, finer detail) |
| `--agent` | Filter to a specific agent ID |

## Troubleshooting

**No session found** — Sessions live in `~/.claude/projects/<slug>/` where the
slug is your project path with `/` replaced by `-`. The latest `.jsonl` file is
the current session.

**Coverage shows 0% for a file you've read** — Coverage tracks read tool calls
only (Read, grep, find_symbol, etc.). Edits are recorded as writes, not reads —
editing a symbol does not mark it read. Mentioning a file in conversation
without reading it via a tool does not count.

**When was this last changed by an agent?** — `ambits touched <file|symbol-id>`
shows the latest agent write, whether that version is still on disk, and which
commit it landed in. For a
symbol it counts only writes attributed to that symbol or one nested in it
(`App` covers `App/handle_key`). A write the log could attribute only to the
whole file is not counted, so "no writes" for a symbol means none that ambits
could attribute; query the file to see those.

**File not in the coverage report** — The file may not have parseable symbols
(empty file, non-code file, or unsupported language without `--serena`).
