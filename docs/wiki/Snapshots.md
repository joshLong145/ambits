# Snapshots

A snapshot records a session's state in one durable, content-addressed unit:
the project's **symbol tree**, what agents **read**, what they **wrote**,
the **git commit** it sits on, and any **uncommitted changes**. File contents
are never stored — only paths, symbol names, spans and hashes.

```bash
ambits -p . snapshot -m "before the refactor"
ambits -p . log
ambits -p . gc
```

```
snapshot 871171228aa7  2026-09-27T16:36:33Z
  git 42d1050 (dirty: 1 file)   reads 532   writes 189
  parents 930eb2a85b72
  before the refactor
```

Snapshots are manual. They are the groundwork for restoring a session and
syncing coverage to a remote, both of which are planned. For now they are
local history.

## Taking a snapshot

`ambits snapshot` snapshots the current session (or `--session <id>`).

- **Nothing changed, nothing written.** A snapshot's id is derived from its
  inputs, so running it again with no new reads, writes or edits prints
  `nothing changed: <id>` and writes nothing.
- **Anything changed makes a new snapshot** on top of the last one: a new
  journal record, an edited file (even whitespace), a different commit, a
  different parser version. Touching a file without changing it does not
  count.
- **History never rewinds.** Undoing a change makes a *new* snapshot whose
  parent is the latest, not a return to the earlier id.
- **Uncommitted changes are included** and counted (`dirty: N files` in the
  log), fingerprinted by their raw bytes. `--require-clean` refuses a dirty
  working tree instead.
- `-m` stores a message in the snapshot's local note.

A project's first snapshot writes every symbol and can take a few seconds.
Each object is flushed to disk so a crash can never leave one half-written.
Later snapshots only write what changed and take well under a second.

## History

`ambits log [ref]` lists a snapshot and its ancestors, newest first: time,
git commit and dirty count, reads, writes, parents and message. `ref` is a
session id, a snapshot id, or a unique prefix of one (7+ hex digits); it
defaults to the current session.

Snapshots made with `--serena` are marked **non-reproducible**: they depend
on Serena's cache, not just the commit.

## Leaving things out

```toml
[sync]
ignore = ["secrets/**", "vendor/"]
```

`.gitignore` syntax, in `tools.toml`. An ignored path appears nowhere in a
snapshot: not in the tree, not in reads or writes, not among the dirty files.
Patterns in your user-global `~/.config/ambit/tools.toml` are applied
separately, and a project cannot re-include what they exclude. Only a digest
of the patterns is recorded, never the patterns themselves.

Tightening `ignore` does not rewrite earlier snapshots.

## Garbage collection

`ambits gc` deletes objects that no session ref, and no reflog entry from the
last 90 days, can reach. Unreachable objects younger than the grace period
(`--grace-days`, default 14) are kept, so a snapshot in progress is never
disturbed. It is always safe to interrupt.

## On disk

Everything lives under `.ambits/`, private to your user:

| Path | Holds |
|---|---|
| `objects/ab/cdef….json` | Symbols, files, directories, coverage, writes and snapshots, as canonical JSON |
| `refs/sessions/<id>` | The latest snapshot of each session |
| `logs/refs/sessions/<id>` | Every move of that ref (the reflog) |
| `notes/<snapshot>.json` | Time, message and ambits version — never host, branch or path |

The format is specified in the design spike,
[`docs/spikes/coverage-snapshots.md`](https://github.com/joshLong145/ambits/blob/main/docs/spikes/coverage-snapshots.md).
