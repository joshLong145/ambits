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
| created | Those of the symbols the write created: no symbol had the id before it |
| level | `symbol` when the log carried enough to attribute lines to symbols; otherwise `file` |
| agent, time, tool | Who, when, with what |

Symbol-level attribution needs the file's text before the write, which Claude
Code includes only for smaller files (about 10 KB and under) and for newly
created files. Larger files get file-level writes, and so does any write whose
log entry carries no usable patch. Attribution checks that the logged changes
turn the old text into exactly the new one, and falls back to file level
rather than guess — so a symbol a symbol-level write does not name really was
left alone.

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
  landed in 3054724 (verified)
```

For a symbol, only writes attributed to that symbol count — or to one nested
inside it, so `App` finds an edit to `App/handle_key`. A file-level write
changed *something* in the file, but not provably this symbol, so it is not
reported. Ask about the file to see it.

It searches every session's journal and reports whether the agent's version
is still on disk: `current`, `changed`, `removed` (still absent — a symbol
that came back after the agent deleted it is `changed`), or `unknown` for a
file-level edit with no hash to compare.

The [TUI](TUI#reading-the-tree) marks the current session's writes with the same
rule: `✎` on each written symbol, its parents and its file.

## Which commit it landed in

`touched` also finds the commit the agent's write landed in, on any local
branch:

| Result | Meaning |
|---|---|
| `landed in <commit> (verified)` | That commit contains the symbol exactly as the agent wrote it (or, for a whole-file `Write`, the file byte for byte) |
| `landed in <commit> (unverified: …)` | The write left nothing to compare — a file-level `Edit` — so this is only the first commit to touch the file after it |
| `partly landed in … ; the rest is in no commit` | Some of a write's symbols are committed (perhaps separately, `git add -p`); the rest are in no commit — not committed yet, or changed again before committing |
| `uncommitted` | In no commit on any branch |

A write that touched several symbols can land in several commits; all are
listed. The file is followed through renames. Answers are cached in
`.ambits/links.ndjson` (and writes found in no commit yet, in
`.ambits/cache/never-landed.ndjson`), and a cached commit that was amended or
rebased away is looked up again. Symbol hashes ignore whitespace, so a commit that differs
from the agent's version only in whitespace still counts as verified.

To keep the cache warm, install the optional git hook:

```bash
ambits hook install --git      # remove with: ambits hook uninstall --git
```

After each commit it records where recent writes landed, in the background.
It keeps any `post-commit` hook you already have (and runs it first), and can
never fail or delay a commit. It acts only when `.ambits/` is at the
repository's top level, so a project in a subdirectory of its repository is
not refreshed by it; `touched` still resolves on demand. It refuses to
install when `core.hooksPath` is set in your global or system git config,
since the hook would then run for every repository.

Not seen: changes made while resolving a merge conflict (merge commits list
none), and files whose names contain `:` or `\`.

Writes are journaled by the [TUI](TUI), like reads — see
[Configuration → The read journal](Configuration#the-read-journal).
