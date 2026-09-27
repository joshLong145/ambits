# Spike: Portable Coverage Snapshots, Agent Writes, and a Git-Style Remote

**Status**: Draft for review
**Branch**: `spec-coverage-snapshots` (spec only; implementation branches per phase)
**Spec revision**: 3
**Estimated scope**: the spec (phase 1) plus six implementation phases; phases 2–5 are local-only and independently useful

### Revision history

**Revision 3** — after a consistency review and an explicit-state model check
of the sync protocol (§16):

- **Snapshot identity** now includes parents: `id = H(inputs_digest ‖ sorted
  parents)` (D17). Both the model and the reviewer independently found that
  the rev 2 key let a reverted state rewind history, and let a pull that
  appended nothing leave two machines unable to converge.
- **gc and fetch** fixed after the model found a crash → gc → fetch sequence
  that fails forever (§8).
- **Journal writers**: `restore` and `pull` write their own shards (§2.6).
- **Restore** narrowed to a new session on the same machine (D19).
- **Links** keyed by `(op, symbol, hash)`, pushed as hints with the ignore
  filter applied at push time; the never-landed cache stays local (D18).
- **Key completeness**: all scanner inputs, generated parser versions, and raw
  bytes for every dirty file.
- **Deferred**: `blame` and the `share_provenance` opt-in (D20).
- **Second model pass** (§16), which found two more faults in the first rev 3
  draft, now fixed. First, fast-forward-only tracking refs with no
  `--force` wedged every clone after one force-with-lease; tracking refs now
  mirror the remote (§12.2). Second, an unconditional merge record meant sync
  never reached a fixed point; pull now appends nothing when merely behind
  and reads the fold already has are skipped (§12.3). The pass also fixed the
  `--break-lock` liveness check, which used a pid with no host (store id,
  §12.2), and made `gc.lock` a `flock`. All ablations now produce
  counterexamples.

**Revision 2** — after three expert reviews (version control, codebase
feasibility, security): attribution rebuilt on what the logs contain; journal
bytes as identity; content-addressed `coverage`/`writes`; §9 Security; honest
privacy; no-regression merge; reflog, gc grace period.

---

## Background

ambits records what an agent has *read*, per symbol, in a per-session journal
(`.ambits/coverage/<session>.ndjson`). That record is machine-independent —
project-relative symbol ids, BLAKE3 content hashes — but it lives on one
machine, is tied to one session, and says nothing about what the agent
*changed*.

This spec adds, each useful before the next exists:

1. **Writes** — which files and symbols an agent changed.
2. **Snapshots** — a portable, deterministic capture of a session's symbol
   tree, reads and writes, pinned to the git commit the code came from.
3. **A remote** — `push` / `fetch` / `pull` on the git model: immutable
   objects plus movable refs.

And one question answered over them: *when did an agent last touch this file
or symbol, and which commit did that change land in?*

## Goals

- Restore a session's knowledge into another session or checkout,
  **verified** per symbol, never assumed.
- Know which agent last changed a file — or, where the log allows, a symbol —
  when, and (where verifiable) in which git commit it landed.
- Sync that record to an origin and back, converging without regressing
  either side.
- Deterministic snapshot ids: snapshotting twice with nothing changed yields
  the same id and writes nothing.

## Non-goals

- **Storing file contents.** No blobs, no diffs. (Identifiers and paths *are*
  stored — §10.)
- **Human edits.**
- **A server** in these phases; the format allows one later.
- **Authenticity.** A dumb remote proves nothing about who wrote what
  (§9.4).

---

## Decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | Snapshots hold **symbols + coverage + writes**, pinned to a git commit; **no file contents** | Identifiers and paths are still shipped (§10) |
| D2 | First origin is a **dumb remote** | No server to build or secure |
| D3 | **Canonical JSON** (RFC 8785 rules, §5.2) | Debuggable, like the journal |
| D4 | Snapshots are **manual** | Predictable |
| D5 | **Nested `dir` objects** | Sharing and cheap diffs |
| D6 | Sync exclusion is **gitignore syntax in `tools.toml`** | The `ignore` crate is already a dependency |
| D7 | **Dirty working trees are tracked** | Agent sessions are dirty almost all the time |
| D8 | Snapshot ids are **derived from inputs** (option B), not from hashing objects | Deterministic; robust to encoding changes |
| D9 | **Writes do not count as reads** | Agents usually read after writing |
| D10 | "Touched" = **changed lines**, on today's symbol types unchanged | §2.3 |
| D11 | Writes name **innermost symbols only** | Parents and files by rollup |
| D12 | **No human edits** | Git knows who committed |
| D13 | Journal identity = **hash of journal bytes** up to the last newline | §6.2 |
| D14 | `coverage` and `writes` objects are **content-addressed** | No collisions, no stale ignored paths |
| D15 | Symbol-level attribution **only from the log** | Reproducible; honest about coverage (§2.2) |
| D16 | Pushed notes **never** include `host`, `gitBranch` or `project_root` | Private by default |
| D17 | `snapshot_id = H(inputs_digest ‖ sorted parents)`; the no-op rule compares `inputs_digest` | History can't rewind; merges always produce a snapshot (§6) |
| D18 | Links are **pushed as hints**, keyed by `(op, symbol, hash)`, filtered at push time; the never-landed cache is local | §3.2 |
| D19 | **Restore** (phase 5) = into a **new session on the same machine**; cross-machine behaviour arrives with the remote | Narrow, shippable scope |
| D20 | **Deferred**: `blame`; the `share_provenance` opt-in | Scope |

---

## 1. Ingestion changes (prerequisite)

Today the Claude ingester maps tool **calls** only. `parse_jsonl_line`
(`src/ingest/claude.rs`) ignores `type:"user"` lines except compaction and
`/clear`, and tool **results** — `toolUseResult` and the result block's
`tool_use_id` — live only on those lines. Nothing in `AgentToolCall`,
`SessionEvent` or `TailerOutput` can carry a write.

Add:

- `ParsedLine::WriteResult` — from a `type:"user"` line whose content holds a
  `tool_result` for a write tool (§2.1) and whose `toolUseResult` is an
  object. **Skip** errors (`is_error: true`, or a string `toolUseResult`:
  rejections, "String to replace not found").
- `SessionEvent::Write { op, agent, t, tool, path, kind, before: Option<String>, after: Option<String>, hunks }`.
- Attribution (§2) runs in **core** (the ingester holds no
  `ParserRegistry`) on a **worker thread**, never the render thread: tailing
  and replay run synchronously in `handle_tick`.

## 2. Writes

### 2.1 Declaring writes

A tool stanza gains an optional `effect`, default `"read"`. Validation becomes
**"exactly one of `depth` or `effect = "write"`"**; `extends` copies `effect`.

```toml
[[tool]]
names       = ["Edit", "mcp__acp__Edit", "mcp__plugin_serena_serena__replace_content"]
path_keys   = ["file_path", "relative_path"]
effect      = "write"
description = "Edit {file_path|short}"
```

| Tool | Attribution |
|---|---|
| `Edit`, `mcp__acp__Edit` | §2.2 |
| `Write`, `mcp__acp__Write` | §2.2 |
| `MultiEdit` | **new stanza** (untracked today); §2.2 when its result carries `originalFile`, else file-level |
| `NotebookEdit` | file-level |
| Serena `replace_content`, `create_text_file`, `replace_symbol_body`, `insert_after_symbol`, `insert_before_symbol`, `rename_symbol` | file-level (no patch; §2.5) |

### 2.2 Attribution sources (D15)

| Case | *before* | *after* | Level |
|---|---|---|---|
| `Edit` / `MultiEdit` with `originalFile` | `originalFile` | `originalFile` with each `oldString` → `newString` (all if `replaceAll`) | symbol |
| `Write`, `type: "create"` | empty | `content` | symbol — every symbol in *after* is touched (**no hunk walk**; creates have no hunks) |
| `Write` update with `originalFile` | `originalFile` | `content` | symbol |
| `originalFile` null (≈79% of `Edit`s here; capped near 10 KB), `userModified: true`, no parser, reconstruction fails, Serena mode | — | — | **file** |

For files over ~10 KB, most writes are file-level. Revisit if Claude Code
changes the cap.

### 2.3 Which symbols changed (D10, D11)

Walk each hunk's `lines`, tracking old and new line numbers from
`oldStart`/`newStart` (hunk *ranges* include context and would
over-attribute):

- `+` at new line *n* → innermost symbol in *after* containing *n*:
  **touched**;
- `-` at old line *n* → innermost symbol *s* in *before* containing *n*:
  **touched** if any symbol in *after* has `s`'s id (ids are not unique; any
  match counts), else **removed**;
- ` ` (context) → nothing.

A changed line inside no symbol (a `use` line, a gap, a detached comment) sets
`outside_symbols` on the record (§2.6) — never a silent drop.

`SymbolNode::line_range` is a `Range<u32>` whose `end` is **inclusive**:
containment is `start <= n && n <= end`, not `Range::contains`.

### 2.4 Constraint: no change to the symbol representation

Attribution is a pure function over existing `FileSymbols`/`SymbolNode`
(`parse_file(path, &str)` takes in-memory source). If a future need would
require changing those types, attribution falls back to post-write hash
comparison. Parser identity for `file` objects comes from `ParserRegistry`.

### 2.5 Serena backend

Tree ids come from the LSP cache and may not match a tree-sitter parse, so
**all** Serena-mode writes are file-level.

### 2.6 Journal records (schema v3)

```json
{"kind":"write","op":"toolu_01…","av":1,"a":"agent-3f9c","t":"2026-09-26T14:02:11Z","tool":"Edit",
 "file":"src/app.rs","level":"symbol","outside_symbols":false,
 "syms":[["src/app.rs::App/handle_key","b3:…"]],"removed":["src/app.rs::App/old_fn"],"fh":null}
```

- **`level`**: `symbol` (attributed from the log) or `file` (fallback).
  **`outside_symbols`**: some changed lines hit no symbol.
- **`fh`**: for a `Write`, BLAKE3 of the raw `content` — lets a file-level
  `Write` be linked to a commit (§3.2). Hashing is allowed; storing contents
  is not (§9.6).
- **Keying**: one record per `(session, op)`, `op` = the tool call's
  `tool_use_id`. The fold keeps the **highest `av`** (attribution version),
  so re-attribution after an upgrade replaces rather than duplicates. Symbol
  entries carry their hash because ids are not unique.
- **`history` records** (defined here, used from phase 2): `{"kind":"history",
  "of":"read"|"write", …}` — kept in the journal, **ignored by the fold**,
  **local-only**: never copied into `coverage`/`writes` objects.
- **`merge` records**: `{"kind":"merge","remote":"<name>","tip":"<snapshot id>"}`
  — appended by `pull` (§12.3); make a pending merge durable.
- **Writers and shards**: the TUI owns the primary shard. `restore` and `pull`
  (CLI) append to their **own** shards (`<session>.restore.ndjson`,
  `<session>.pull.ndjson`) — never to the file the TUI holds open. Writers
  seed their `op` set from **all** shards (`read_journal_session`), so replay
  never duplicates.
- Paths **outside the project root** are dropped.
- The watcher's stale marking is unchanged.

### 2.7 Effects of D9 on existing code

- Tests asserting edits grant `FullBody` change: `claude.rs` `map_edit_tool`,
  `map_write_tool`, `map_notebook_edit`; integration `tool_edit_full_body`,
  `tool_write_full_body`, the NotebookEdit and `mcp__acp__Edit` cases,
  `parse_log_file_all_tool_stanzas`.
- The TUI activity feed drops events with no read depth
  (`App::process_agent_event`); it must render writes explicitly.
- Coverage: a symbol edited but never read drops out — release notes. Rejected
  edits, which today still grant `FullBody`, no longer do.
- A write call leaves existing read state untouched: no stale-clear, no
  provenance or hash refresh. The same holds for a read that resolves to
  `Unseen`. Both are guarded in `apply_tool_call` (the one path every replay
  shares) and in `ContextLedger::record`, which ignores `Unseen`. Otherwise
  editing a drifted symbol makes the stale read look current, and the journal
  records it as a fresh read at the post-edit hash.

---

## 3. Git linkage

### 3.1 Still current?

Symbol writes: compare the write's hash with the symbol's current hash —
*unchanged since agent X wrote it* / *changed since*. File-level `Write`s:
compare `fh` with the file's current bytes. Other file-level writes report
time only.

### 3.2 Which commit did it land in?

Resolved lazily, per `(op, symbol, hash)` — one write's symbols can land in
different commits (`git add -p`):

1. `git log --branches --full-history -M --name-status --format=%H --reverse --since=<t − 1 day> -- <path>`.
2. For each commit C: `git cat-file blob C:<path-at-C>`; for a symbol write,
   parse and compare the symbol's hash; for a file-level `Write`, compare
   BLAKE3 of the blob with `fh`.
3. The first match is where it **landed**.

File-level `Edit`s (no hash) **cannot be verified**. `touched` may show
"first commit touching the path after t", always labelled **unverified**.

The content hash is whitespace-normalized, so a whitespace-only difference
can match; documented. A cached result is **re-checked for reachability**
(`git merge-base --is-ancestor C <branch>` over `--branches`) before display,
so amends and rebases re-resolve.

**Links index** (`.ambits/links/`), one entry per `(op, symbol, hash)`:

- **Pushed as hints** (D18). The ignore filter (§4) is applied **at push
  time** — links are written after snapshots, by `touched` or the hook.
  Fetched links are re-verified locally before display (§9.4).
- The **never-landed** cache (per `HEAD`) is **local-only**, never pushed.
- Links are not objects and not gc roots.

**Optional prewarm:** `ambits hook install --git` (§9.3).

### 3.3 Queries

| Command | Answers |
|---|---|
| `ambits touched <file\|symbol-id>` | Latest agent write: agent, session, time, landed commit (verified, unverified, or *uncommitted*), still current or changed since. Origin (local or which remote) from phase 6 |

`ambits blame` is deferred (D20).

---

## 4. Sync exclusion

```toml
[sync]
ignore = ["secrets/**", "vendor/"]
```

- `.gitignore` syntax via the `ignore` crate.
- **Applied to every record type**: at snapshot time for tree objects, reads,
  writes, the dirty list and notes; **at push time** for links.
- **Layering**: project patterns first, then **user-global patterns last, as a
  separate matcher that cannot be negated** by the project.
- **Not retroactive**: tightening `ignore` does not rewrite older snapshots,
  and `push` sends ancestors reachable through `parents`.
  **`push --dry-run` lists what would leave, including ancestors**, so the
  user can see it.
- Snapshots record only a **digest** of the effective patterns, never the
  patterns themselves (they may name confidential directories).

---

## 5. Object model

### 5.1 Types

| Type | Payload | Id |
|---|---|---|
| `symbol` | name, category, label, `line_range`, `byte_range`, `content_hash`, child symbol ids | content |
| `file` | top-level symbol ids, `total_lines`, parser identity | content |
| `dir` | sorted entries `{name, kind: file\|dir, id}` | content |
| `coverage` | sorted reads `(symbol id, hash at read, depth, agent)` — the fold's result, **no `history`, no origin** | content (D14) |
| `writes` | write records (§2.6), sorted by canonical bytes — no `history`, no origin | content (D14) |
| `snapshot` | §6.3 | derived (D17) |

`estimated_tokens` is excluded from `symbol` payloads (recomputed on restore).

### 5.2 Canonical JSON and object ids

- RFC 8785 rules; integers only; paths project-relative, `/`-separated,
  **NFC-normalized**. Readers **reject** non-canonical input.
- Object id = `BLAKE3("ambits-obj v1\0" ‖ type ‖ "\0" ‖ len ‖ "\0" ‖ payload)`.
- The existing `content_hash` and merkle hash cannot identify objects;
  neither changes.

---

## 6. Snapshots and deterministic ids

### 6.1 Inputs and id (D8, D17)

```
inputs_digest = BLAKE3("ambits-inputs v1\0" ‖ canonical(
    session_id,
    journal_digest,
    git_commit | "none",
    parsers,
    scan_inputs,
    sync_ignore_digest,
    dirty: sorted [(path, BLAKE3(raw bytes))]
))

snapshot_id = BLAKE3("ambits-snapshot v1\0" ‖ inputs_digest ‖ sorted parents)
```

| Input | Determines |
|---|---|
| `git_commit` + `parsers` + `scan_inputs` | the tree for clean files |
| `session_id` + `journal_digest` | reads, writes, and pending merges |
| `dirty` | dirty files (§7) |
| `sync_ignore_digest` | what is included |
| `parents` | history (§6.4) |

- **`parsers`**: backend, grammar versions **generated from `Cargo.lock` at
  build time** (today's list in `EnvironmentManifest::capture` is
  hand-maintained), and a per-language **`SYMBOL_SCHEMA`** constant bumped
  whenever ambits' own extraction changes (#33 changed spans with no grammar
  change).
- **`scan_inputs`**: everything the walker honours that is not in the commit —
  `--filter`/`--filter-regex`, `.git/info/exclude`, global git excludes,
  untracked `.ignore` files, and walker flags. `snapshot` records its own
  effective values.

**Not inputs** (stored as a note): time, host, message, ambits crate
version, tool config.

### 6.2 Journal identity (D13)

`journal_digest` = BLAKE3 over each of the session's shards, sorted by name:
the shard name, then its bytes up to and including the last newline.
`snapshot` reads each shard **once** and derives the key and objects from
that prefix — a torn last line, a concurrent append, multiple shards and v2
journals are all handled by this one rule.

A journal **schema upgrade** appends a new header (`Journal::open_at`), so the
id changes **once** on upgrade (§6.4).

### 6.3 The snapshot object

```json
{"inputs":"b3:…","session":"…","journal":"b3:…","git":"<sha>|none","parsers":[…],
 "scan":"b3:…","ignore":"b3:…","dirty":[["src/app.rs","b3:…"]],
 "root":"<dir id>","coverage":"<id>","writes":"<id>","parents":["<snapshot id>",…],"state_digest":"b3:…"}
```

`state_digest` = BLAKE3 over `root`, `coverage`, `writes` and `parents`. With
parents in the id, **the same id always implies the same `state_digest`**;
finding an existing id with a different digest is a **hard error** (it means
corruption or a forged object), never an idempotent skip.

### 6.4 Snapshot, the no-op rule, and properties

`ambits snapshot`:

1. Compute `inputs_digest`. Parents = the session ref's tip, plus the pending
   merge tips: the `tip` of every `merge` record in the journal prefix that
   the session ref's snapshot did not cover, dropping any that is an ancestor
   of another parent. Usually that leaves one; after a force-with-lease on the
   remote between two pulls it can leave two, and a snapshot may then have
   three parents.
2. **No-op** if the tip's `inputs_digest` equals the current one **and** no
   merge is pending: print `nothing changed: <tip>` and write nothing.
3. Otherwise write objects (§8 order), the snapshot, the note; advance the
   ref.

| Change | Result |
|---|---|
| Nothing | **Same id** (no-op) |
| A read, write or merge recorded | New snapshot (journal grew) |
| A dirty file edited, even whitespace or same length | New snapshot (raw bytes changed) |
| A dirty file only `touch`ed | Same id |
| Edited then **reverted** to an earlier state | New snapshot whose parent is the tip — history never rewinds (rev 2 bug) |
| `pull` of a diverged tip that appended nothing but a merge record | New **two-parent** snapshot, so the next push fast-forwards (rev 2 bug) |
| `pull` when merely behind and the remote adds nothing | Nothing appended; same id (no-op), so sync reaches a fixed point |
| Symbol-extraction change (`SYMBOL_SCHEMA`) | New snapshot |
| Journal schema upgrade | New snapshot, once |
| Same state in another session | Different id |

**Serena backend**: parsers record `serena@<cache fingerprint>`; snapshots are
flagged **non-reproducible**.

---

## 7. Dirty state

- **Detected** with `git status --porcelain=v1 -z` (§9.2): modified tracked
  files and untracked files the scanner would parse. No git or not a repository
  ⇒ `git = "none"` and every file is dirty.
- **Fingerprint** = BLAKE3 of the file's **raw bytes**, for every dirty file:
  `file` object ids miss same-length edits outside symbols and whitespace-only
  edits inside them (`content_hash` is normalized).
- `ambits snapshot --require-clean` refuses dirty trees.
- `log` shows `a1b2c3d (dirty: 4 files)`.

| Implication | Why acceptable |
|---|---|
| Dirty content cannot be restored elsewhere | Follows from D1; restore reports it |
| A snapshot is not "exactly commit C" | Surfaced in `log` and `restore` |
| Wrong restores | None: every read and symbol write is verified by hash |
| Uncommitted names and paths leave on push | `[sync] ignore`; `push --dry-run` |

---

## 8. Store layout, refs and gc

```
.ambits/
  coverage/                          # journal shards (v3)
  objects/ab/cdef….json              # loose objects, uncompressed canonical JSON
  refs/sessions/<session-id>
  refs/remotes/<remote>/sessions/<id>
  logs/refs/…                        # reflog
  links/…                            # links index (§3.2), keyed by (op, symbol, hash)
  cache/never-landed/…               # local-only (§3.2)
  notes/<snapshot-id>.json           # time, message, version — no host, branch or root (D16)
  config
```

- Store mode `0700`, files `0600`.
- **Object writes**: temp, fsync, rename; **children before parents**,
  snapshot object last. Loose objects uncompressed (zstd with packs, phase 7).
- **Ref updates**: create `<ref>.lock` exclusively containing **pid, start time
  and a random token** (no host, D16); re-read the ref after locking; write;
  rename; append to the reflog.
- **Reflog** entries expire after 90 days (configurable).
- **gc** (found necessary by the model, §16):
  - `gc` holds `.ambits/gc.lock` **exclusively**; `snapshot`, `fetch` and
    `pull` hold it **shared** for their duration. It is an OS advisory lock
    (`flock` on Unix, `LockFileEx` on Windows — e.g. via the `fs2` crate),
    not a create-exclusive file, so a crashed holder releases it and cannot
    block gc forever.
  - Roots: refs, `refs/remotes/*`, unexpired reflog entries.
  - Deletes only unreachable objects older than a **grace period** (default
    14 days), **re-checking age at the moment of deletion**, and in
    **parents-first order**, so a present object always has its children.
  - `snapshot`, `fetch` and `pull` **refresh the age** of every object they
    find already present, not only the ones they write.
  - Removes orphaned notes. **Never runs on a remote.**

---

## 9. Security

### 9.1 Untrusted names and paths

Validated before use (from a remote, and defensively locally): object and
snapshot ids `^[0-9a-f]{64}$`; session ids by the UUID grammar (`is_uuid`);
`op` `^toolu_[A-Za-z0-9]+$`; `dir` entry names are one component — no `.`,
`..`, `/`, `\`, NUL, `:`, drive prefixes or control characters, no duplicates
after case-folding and NFC; paths in records are relative, normalized, no
`..`. Every path goes through **one checked join**; symlinks are never
followed (`symlink_metadata`; non-regular files skipped).

### 9.2 Invoking git

`--end-of-options` and `--` before paths; commit ids match
`^[0-9a-f]{40,64}$` and come from our own `git` output, never a remote;
`--since` times re-emitted as RFC 3339; `git cat-file blob` only after path
validation; hardened with `-c core.fsmonitor=false -c diff.external=
--no-pager` and `GIT_CONFIG_NOSYSTEM=1`.

### 9.3 The optional `post-commit` hook

Directory from `git rev-parse --git-path hooks` (respects `core.hooksPath`);
**chains** to any existing hook; shell-quoted absolute ambits path; runs in
the background with a timeout and `|| true`, so it never blocks or fails a
commit; acts only if `.ambits/` exists at the repo top level; never global;
`ambits hook uninstall --git`.

### 9.4 Verification on fetch

Re-hash every content-addressed object; for each snapshot recompute
`state_digest` and `snapshot_id` from its stored `inputs` and `parents`, and
check every referenced object exists and verifies; re-verify links locally
before display. Records merged from a remote carry an **origin**
(`origin=<remote>`) — local-only, like `history`. A dumb remote provides
**no authenticity**; signatures are future work.

### 9.5 Resource limits

Object size cap (16 MiB, checked before reading); serde_json's default
recursion limit; iterative tree walks with a depth cap and a visited set; a
cap on entries per object.

### 9.6 File contents are never persisted

`originalFile`, `content`, `oldString`, `newString` go only to `parse_file`
(and, for `fh`, to BLAKE3) and are dropped. Errors and logs carry path and
length only. A canary test asserts an edited file's marker string appears
nowhere in the journal, objects, links or notes.

---

## 10. Privacy: what leaves the machine

| Leaves on push | Mitigation |
|---|---|
| Symbol names, labels, Markdown headings, paths (including untracked files) | `[sync] ignore`; `push --dry-run` |
| Agent ids, timestamps, tool names | inherent |
| Snapshot messages | user-authored |
| Links (op, symbol, commit) | ignore filter at push time |
| `host`, `gitBranch`, `project_root` | **never** (D16; the opt-in is deferred, D20) |
| Ignore and scan patterns | digests only (§4) |
| File contents | **never** (§9.6) |

---

## 11. Commands

| Command | Behavior |
|---|---|
| `ambits snapshot [-m msg] [--require-clean]` | §6.4 |
| `ambits log [ref]` | Time, git pin, dirty count, reads, writes, parents, message |
| `ambits restore <ref> [--into <session>]` | §12.1 |
| `ambits touched <file\|symbol>` | §3.3 |
| `ambits remote add <name> <path>` | Record in `.ambits/config`; warn if world-writable |
| `ambits push [remote] [ref] [--force-with-lease] [--dry-run]` | §12.2 |
| `ambits fetch [remote]` / `ambits pull [remote]` | §12.2, §12.3 (tracking refs always mirror the remote) |
| `ambits gc` | §8 |
| `ambits hook install --git` / `hook uninstall --git` | §9.3 |

`restore` (snapshot → session) is distinct from the existing
`restore-context` (journal → digest for a compaction); the help text says so.

---

## 12. Restore, sync and merge

### 12.1 Restore (phase 5: same machine, new session — D19)

`ambits restore <ref> [--into <session>]`; without `--into`, a new session id
is minted.

1. Load the snapshot; if its git pin differs from `HEAD`, **say so** — never
   check out.
2. Rebuild the tree and classify each read with the **existing
   `restore::classify`** (valid, drifted, removed — including symbols that
   moved between files).
3. Append valid reads, and all writes as `history{of:"write"}`, to the target
   session's `<session>.restore.ndjson` shard, **skipping records the target's
   fold already has** — restoring twice changes nothing.
4. Set `refs/sessions/<target>` to the restored snapshot, so the target's next
   snapshot descends from it.
5. Report per file: **verified**, **drifted**, **unverifiable here** (dirty at
   snapshot, content not on this machine).

Cross-machine restore is the same operation after a `fetch` (phase 6).

### 12.2 Push and fetch (dumb remote over a filesystem path)

- **push**: copy missing objects children first, fsync each, snapshot object
  last; **verify the remote holds the tip's full closure**, including objects
  skipped as already present; then take the remote ref lock,
  compare-and-swap (fast-forward, or `--force-with-lease` against the last
  fetched value), write, release. An existing id with a different
  `state_digest` is refused. Links go with it, filtered (§3.2).
- **fetch**: holds `gc.lock` shared; verify everything (§9.4); refresh ages
  of already-present objects; then **always** set `refs/remotes/<remote>/*`
  to the remote's value — a remote-tracking ref mirrors the remote, as git's
  `+refs/…` refspecs do. A non-fast-forward move (someone used
  `--force-with-lease`) is **warned about and logged to the reflog**, whose
  old entry keeps the previous tip reachable. Fast-forward-only tracking
  refs, with no `--force`, wedge every other clone after one
  force-with-lease: their fetch and pull are rejected forever and their push
  is non-fast-forward (model, §16).
- **Orphaned remote lock**: a crash can leave `<ref>.lock` held forever (the
  model confirms). The lock records a random **store id** (minted into
  `.ambits/config` at init; not a host name, D16), pid, start time and token.
  `--break-lock` removes a lock only if its start time exceeds a threshold
  **and** either the store id is this store's and the pid is not alive, or the
  user confirms interactively. A pid check alone is meaningless for a lock
  taken from another machine. Breaking a live, slow pusher's lock can lose
  that push, so the threshold is conservative and the command says so.
- **Supported filesystems**: local disks and mounts with atomic rename and
  exclusive create; not sync services that create conflict copies.

### 12.3 Pull and merge

Merges into the **same session id** locally.

1. Fetch.
2. **Nothing to do** if the remote tip equals, or is an ancestor of, the local
   tip.
3. **Reads**: a remote read is appended only if its hash equals the local
   symbol's current hash **and** the local fold does not already hold that
   hash at equal or greater depth; other remote reads become
   `history{of:"read"}`, once. (The fold lets a new hash supersede, so
   appending stale reads would regress coverage — the model confirms the rule
   prevents it.)
4. **Writes**: folded per `(session, op)`. A remote record with a **higher
   `av`** is appended (the fold takes the highest `av`); at the **same `av`**
   with different contents the **local** record wins and the remote one is
   kept as `history{of:"write", conflict:true}`. Such conflicts therefore do
   not converge between clones — each keeps its own — and `touched` shows the
   flag.
5. **Merge record**: if step 3 or 4 appended any `read`/`write` record, **or**
   the local tip is not an ancestor of the remote tip (the histories
   diverged), append `{"kind":"merge","remote":…,"tip":<remote tip>}` to the
   `.pull` shard, so the merge is durable and the next snapshot takes two
   parents. Otherwise — the local tip is merely behind and the remote adds
   nothing — append nothing: an unconditional merge record makes every
   pull → snapshot → push round mint two new snapshots forever (model, §16).
6. Symbol trees are never merged.

---

## 13. Compatibility and migration

- **Journal v2 → v3**: adds `write`, `history` and `merge` records; v2 stays
  readable; upgrades only append (one id change, §6.4).
- **D9 behavior change** (§2.7).
- **Formats** versioned by id prefixes; `SYMBOL_SCHEMA` versions extraction.

---

## 14. Phases

| # | Phase | Delivers | Key tests |
|---|---|---|---|
| 1 | Spec | This document | — |
| 2 | Writes, locally | §1; §2 on a worker thread; journal v3 (`write`, `history`); shards; tool stanzas and validation; `touched` without commits; TUI shows writes | only changed lines attribute; pure deletion; create touches every symbol; `userModified` ⇒ file; null `originalFile` ⇒ file; `outside_symbols`; outside-project dropped; errors skipped; replay twice ⇒ idempotent; higher `av` replaces; no read credit; **canary** |
| 3 | Objects + snapshots | §5–§8 locally: objects, D17 ids, no-op rule, `log`, reflog, gc | snapshot twice ⇒ same id; `touch` ⇒ same id; whitespace or same-length dirty edit ⇒ new; **revert ⇒ new snapshot, parent = tip**; journal append ⇒ new; torn line ignored; schema upgrade ⇒ one new id; `SYMBOL_SCHEMA` ⇒ new; `\`/`/`, NFC/NFD, shuffled dirty ⇒ same; ignore covers every record type; user-global ignore not negatable; **gc: crash → gc → re-snapshot never loses an object; parents-first deletion; age refresh** |
| 4 | Git linkage | §3: lazy resolution, links index, verified/unverified, reachability re-check, optional hook | write → commit → link; `add -p` across two commits; file-level `Write` via `fh`; later edit ⇒ changed since; amend and rebase ⇒ re-resolved; other branch; rename; never-landed cached locally; hook chains and never fails a commit |
| 5 | Restore (same machine) | §12.1 | into a new session; restore twice ⇒ no change; ref set ⇒ next snapshot descends; moved symbols; drifted; different commit ⇒ warning |
| 6 | Dumb remote | §9, §12.2, §12.3; cross-machine restore; origin in `touched` | two clones converge both ways; **repeated pull/snapshot/push with no new reads ⇒ no new snapshots after one round**; **diverged pull with nothing to append ⇒ two-parent snapshot ⇒ fast-forward**; **after B's force-with-lease, A's fetch and pull still work**; pulled reads the fold already has are not re-appended; stale remote read ⇒ history; concurrent push rejected; force-with-lease; **crash → gc → fetch recovers**; push verifies skipped closure; colliding id refused; hostile names, symlinks, oversized and cyclic objects rejected; interrupted push leaves no dangling ref; break-lock rules |
| 7 | Network transports | SSH / object storage; packs with an index and zstd | — |
| — | *(later)* | Smart `ambits serve`; signed snapshots; `blame`; `share_provenance` | — |

## 15. Risks and open questions

- **Attribution coverage** is bounded by the log (D15).
- **`MultiEdit` / `NotebookEdit` result shapes** unverified; both fall back to
  file-level.
- **CRLF files** unverified; attribution must count lines as the parser does.
- **Lazy commit resolution cost** grows with a file's history; bounded,
  cached, prewarmable.
- **Parser determinism** assumed by D8; guarded by phase 3 tests and
  `SYMBOL_SCHEMA`.
- **`--break-lock`** can lose a slow live push; conservative defaults only.
- **`--force-with-lease` after a fetch** passes the lease even if the fetched
  tip was never merged (git's known gap). Consider git's `--force-if-includes`
  rule: the lease holds only if the remote tip is an ancestor of the local
  tip or appears in a `merge` record.

## 16. Formal model

An explicit-state model checker (Python, standard library; two clients
sharing a session id, one remote, one symbol, hashes {h1,h2}, depths {1,2})
ran BFS over all interleavings of atomic steps, including crashes and gc. The
model was a review aid and is **not kept in the repository**; this section is
the record of what it checked and found, and a re-check means rebuilding it
from this description. It checked four bounded scenarios, none truncated:

- **SYNC**: journal ≤ 2 records, 3 snapshots, 2 pushes, 2 fetches, 1 edit;
  convergence (P4) is checked from every quiescent state.
- **SNAPID**: 2 edits (edit and revert).
- **CRASH**: 1 crash during snapshot, push or fetch.
- **GC**: gc, time passing beyond the grace period, 1 crash.

The results hold only within these bounds. Writes (and so write convergence)
and more than one symbol were not exercised. Push's closure verification
(§12.2) is not modeled: children-first copying alone passes within the
bounds, so that check is defence in depth against partial remotes the model
does not produce (for example, a remote written by another tool).

| Property | Rev 2 | Rev 3 as first drafted | Rev 3 as now written |
|---|---|---|---|
| P1 no dangling refs | pass | pass | pass |
| P2 no lost update (CAS under lock) | pass | pass | pass |
| P3 no coverage regression on pull | pass | pass | pass |
| P4 convergence after exchange | **fail** — diverged pull with nothing to append never fast-forwards | **fail** — one force-with-lease wedges other clones (fast-forward-only tracking refs, no `--force`) | pass |
| P5 same id ⇒ same state | **fail** — revert rewinds history | pass (D17) | pass |
| P6 gc never deletes needed objects | **fail** — crash → gc → fetch fails forever | pass (§8) | pass |
| Fixed point (no new snapshots once synced) | — | **fail** — unconditional merge record, two new snapshots per round | reached after one round |
| L1 remote lock can't be held forever | fail (expected) | fail (expected) | fail (expected; §12.2) |

States explored for the current design: SYNC 982,862; SNAPID 976,359;
CRASH 124,036; GC 56,417 (an independent re-run of the final model version
explored 58,223 GC states with identical results; the counts varied slightly
as the model evolved). The independent re-run also reproduced the GC and
copy-order ablations below.

**Ablations** (each mitigation removed from the current design; a
counterexample must appear):

| Removed | Scenario | Result |
|---|---|---|
| gc grace period and `gc.lock` | GC | P1, P6 fail (gc deletes a running snapshot's children; the ref dangles) |
| gc age re-check and `gc.lock`, arbitrary delete order | GC | P1, P6 fail (crash → age → gc marks → re-snapshot refreshes → gc deletes) |
| children-first copy (arbitrary order) | CRASH | P1 fails (crash after the snapshot object is copied; retry skips it and the ref dangles) |
| hash-match merge (append every remote read) | SYNC | P3 fails |
| CAS under the lock (blind overwrite after an unlocked check) | SYNC | P2 fails |
| multiple parents (local tip only) | SYNC | P4 fails |
