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

Snapshots are manual. A snapshot can be [restored](#restoring) into another
session on this machine; syncing to a remote is planned.

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
- A file whose path cannot be stored — a name with `:` or a control
  character, or two paths differing only in case — is left out with a
  warning; the rest of the snapshot goes ahead.
- **What git does not report is not seen.** Changes inside a submodule, and
  to files marked `skip-worktree` or `assume-unchanged`, do not make a new
  snapshot.

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

## Restoring

```bash
ambits -p . restore <ref>                    # into a new session (id printed)
ambits -p . restore <ref> --into <session>   # into an existing one
```

```
restored snapshot 89ad0e4badaa into session 3f1c…
  warning: the snapshot is pinned to 6d8c21f, but HEAD is 1f32c0b; nothing was checked out
  src/app.rs: 41 verified, 2 drifted
  src/linkage.rs: 12 verified, 1 moved
  src/time.rs: 0 verified, 3 unverifiable here (dirty when snapshotted)
  56 read(s) and 4 write(s) appended
```

Each read in the snapshot is checked against the project as it is now,
exactly as [`restore-context`](Restoring-Context) checks a journal:

| Per file | Meaning |
|---|---|
| verified | Still what was read: restored. A symbol that moved to another file since is restored at its new address (`moved`) |
| drifted | Changed since it was read: not restored |
| unverifiable here | The file had uncommitted changes when snapshotted, and what was read is not on this machine any more: not restored |

The snapshot's writes are restored as *history* — a record of what an agent
did, not writes of the new session, so `touched` does not report them as
its own. Everything goes to the session's `<session>.restore.ndjson`
journal, which the [TUI](TUI) and `restore-context` read with the rest. The
session's ref is set to the snapshot, so its next snapshot descends from
it. Restoring twice changes nothing. A snapshot pinned to another commit is
restored with a warning; nothing is ever checked out.

Restoring from another machine arrives with syncing.

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
last 90 days (`--reflog-expiry-days`), can reach. Unreachable objects younger than the grace period
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
