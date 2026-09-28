# Sharing

A session's [snapshots](Snapshots) can be shared through a **remote**: a
plain directory — on a local disk, a shared mount, anything with atomic
rename and exclusive create — laid out as `.ambits/` is. There is no server.
Another clone of the project fetches from it, pulls a session's history into
its own session of the same id, or restores it into a new one.

```bash
ambits -p . remote add origin /mnt/team/ambits   # once per clone
ambits -p . snapshot
ambits -p . push                                  # this session's history to origin

# on another machine
ambits -p . fetch                                 # every session's, verified
ambits -p . restore origin/<session>              # into a new session here
ambits -p . --session <session> pull              # or merge into the same session
```

## Push

`ambits push [remote]` copies the current session's snapshot history (or
`--session <id>`'s) to the remote, then moves the session's ref there.

- Objects go first, each verified where it lands; the ref moves only once
  the whole history is in place, so an interrupted push leaves no ref
  pointing at missing objects.
- If the remote already has this tip, or history after it, there is nothing
  to push.
- If the remote has snapshots of this session you do not, the push is
  refused: pull, snapshot, and push again. `--force-with-lease` overwrites
  them instead, but only if the remote still holds what you last fetched.
- `--dry-run` says how many objects and links would go, and sends nothing.
- Links (which commit a write [landed in](Agent-Writes#which-commit-it-landed-in))
  go with it, filtered by `[sync] ignore`, as hints the other side re-checks.

The remote's ref is guarded by `refs.lock`, a file naming the store that
took it (a random id from `.ambits/config`, never a host name), a pid and a
time. A crash can leave it behind; `ambits push --break-lock` removes it only
when it is over 10 minutes old, and then without asking only if this store
took it and that process is gone — a lock from another machine needs your
confirmation, since breaking a live push's lock can lose that push.

## Fetch

`ambits fetch [remote]` copies every session's history from the remote and
records where each points, as `<remote>/<session>`. Nothing is trusted:
every object is re-hashed, every snapshot's id and contents recomputed,
symlinks and oversized files refused. A fetch that fails verification
records nothing.

A remote session that was overwritten (someone used `--force-with-lease`) is
still followed, and reported as forced; the old tip stays in the reflog. A
fetch also brings back objects lost locally.

## Pull

`ambits pull [remote]` fetches, then merges the remote's history of the
current session into the same session here, writing to its
`.pull` journal shard:

| Remote record | Here |
|---|---|
| A read of code that is the same here | Added, unless already known at that depth |
| A read of code that differs here | Kept as history, once: it never lowers what this session knows |
| A write with a newer attribution | Added, marked with where it came from |
| The same write, attributed differently | Yours kept; theirs kept as history (a conflict) |

When anything was added, or the two histories diverged, the pull records a
merge: the next `ambits snapshot` has both tips as parents, and pushes as a
fast-forward. When you are merely behind and the remote adds nothing, a pull
writes nothing — so pull, snapshot, push rounds settle instead of minting
snapshots back and forth.

`ambits touched` says `pulled from <remote>` for a write that came this way.

## Restoring across machines

After a fetch, `ambits restore <remote>/<session>` restores the remote's
latest snapshot of that session into a new session here, with the same
checks as a [local restore](Snapshots#restoring).

## Configuration

`.ambits/config` holds this store's id and its remotes:

```toml
store_id = "…"

[remotes.origin]
path = "/mnt/team/ambits"
```

`ambits remote add <name> <path>`, `remote list`, `remote remove <name>`. A
world-writable remote is warned about: anyone who can write it can rewrite
the history everyone pulls. A dumb remote proves nothing about who wrote
what; signatures are future work.

## What leaves the machine

Symbol names and paths (including untracked files), agent ids, timestamps,
tool names, snapshot messages, and links. Never file contents, host names,
branch names or the project's path. Keep files out with
[`[sync] ignore`](Configuration).
