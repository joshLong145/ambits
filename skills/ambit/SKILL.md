---
name: ambit
description: "Coverage-aware agent workflow tool. Use PROACTIVELY before modifying files, making architecture decisions, debugging, or reviewing code. Reports which symbols Claude has read (Seen%) vs read in full (Full%). Use MCP tools mcp__ambit__coverage or mcp__ambit__coverage_file when available; fall back to bash."
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
reports what was demonstrably read and demonstrably has not changed since:

```bash
ambits -p . restore-context
```

Treat the symbols it lists as known — no need to re-read them. Anything it does
not list is not covered, and it names the files that changed since they were
read so you can re-read only those.

If the output is labelled `UNVERIFIED`, it was recovered from session logs
rather than the coverage journal. Those logs do not record what a file looked
like when it was read, so drift cannot be detected there — re-read before
relying on it.

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
silently downgrades any later restore of that session to `UNVERIFIED`.
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

**Specific symbol at 0%** — If the task involves that symbol, read it first using
Read or Serena's `find_symbol` with `include_body: true`.

## Coverage Improvement Loop

If coverage on files you need is insufficient:

1. Identify low-coverage files with `mcp__ambit__coverage_file` or `ambits --coverage`
2. Read the specific symbols you need (`find_symbol` with `include_body: true`,
   or `Read` for the full file)
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

**Coverage shows 0% for a file you've read** — Coverage tracks tool calls only
(Read, Edit, find_symbol, etc.). Mentioning a file in conversation without reading
it via a tool does not count.

**File not in the coverage report** — The file may not have parseable symbols
(empty file, non-code file, or unsupported language without `--serena`).
