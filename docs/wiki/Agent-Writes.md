# Agent Writes

ambits records which files — and, where the session log allows, which
symbols — an agent **changed**, alongside what it read.

## Writes are not reads

`Edit`, `Write`, `MultiEdit`, `NotebookEdit` and Serena's editing tools are
**writes**. A write grants no read credit: an agent that changed a function
has not necessarily read it, and usually reads the file again afterwards. So a
symbol an agent edited but never read does not count toward coverage, and an
edit that was rejected counts for nothing at all.

Nor does a write refresh an earlier read. If an agent read a function, the
function then changed, and the agent edited it without reading it again, the
old read stays marked as out of date — `restore-context` will not hand it back
as current.

## What is recorded

For each successful write, from the tool's result in the session log:

| Field | Meaning |
|---|---|
| file | Project-relative path. Writes outside the project are not recorded |
| symbols | The innermost symbols whose lines changed, with their hash **after** the write |
| removed | Symbols the write deleted |
| level | `symbol` when the log carried enough to attribute lines to symbols; otherwise `file` |
| agent, time, tool | Who, when, with what |

Symbol-level attribution needs the file's text before the write, which Claude
Code includes only for smaller files (about 10 KB and under) and for newly
created files. Larger files get file-level writes. Attribution checks every
changed line against the reconstructed text and falls back to file level
rather than guess.

**File contents are never stored** — only paths, symbol names and hashes.

## `ambits touched`

```bash
ambits -p . touched src/app.rs
ambits -p . touched 'src/app.rs::App/handle_key'
ambits -p . touched --format json src/app.rs
```

```
src/app.rs::App/handle_key — last written 2026-09-27T10:00:01Z by agent-3f9c (Edit)
  session 9a1c…, write toolu_01…
  unchanged since the agent wrote it
```

It searches every session's journal and reports whether the agent's version
is still on disk: `current`, `changed`, `removed`, or `unknown` for a
file-level edit with no hash to compare. Which git commit a write landed in is
planned.

Writes are journaled by the [TUI](TUI), like reads — see
[Configuration → The read journal](Configuration#the-read-journal).
