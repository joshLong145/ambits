# Spike: Grep-Style `find` with Symbol Attribution

**Status**: Ready for implementation
**Branch**: `feat/grep-style-find` (suggested)
**Spec revision**: 1
**Estimated scope**: ~8 new/modified files, ~40 new tests

---

## Background

`ambits find` today searches the **symbol index** with a bespoke `[path]::[name]`
grammar (`src/find.rs`). It answers "where is this name defined", "what is in this
file", "what hangs off this type" — and nothing else. Its own module doc is explicit
about the limit:

> This searches *definitions*. It has no notion of usages: a method call like
> `is_none_or` returns nothing, because no symbol in the tree is named that.
> For call sites, grep remains the right tool.

That is a real gap for the product's purpose. ambit exists so an agent can find code
*and know what it has already read*; the moment the agent falls back to `grep` or the
`Grep` tool for anything content-shaped, it leaves the coverage system entirely — the
read is attributed by `default_tools.toml` to a path at `Overview` depth, with no
symbol resolution and no drift detection.

Meanwhile the pattern grammar is a second dialect agents must learn, and it is not one
they already know.

Neither `find` nor `callers` has shipped. There are no external consumers and no
schema compatibility to preserve.

---

## Goal

Replace `find`'s interface with a **ripgrep-compatible content search** whose every hit
carries the symbol it landed in and how deeply that symbol has already been read, and
which records what it showed the agent through the existing coverage journal.

```
$ ambits -p . find 'depth_of'
src/find.rs:178:26:[full run] .filter(|(_, n)| c.depth_of(&n.id).is_some())
src/lookup.rs:214:22:[—   describe] read_depth: coverage.and_then(|c| c.depth_of(…
```

**Compatibility target is ripgrep, not POSIX grep.** Claude Code's `Grep` tool is
ripgrep-backed, so the agents this serves already speak rg's flags (`-g`, `-t`,
`output_mode`, `head_limit`, `multiline`). Where `grep(1)` and `rg` disagree, rg wins.

Where rg and ambit's own existing CLI conventions disagree, **rg wins** — our commands
are unreleased and can be refactored.

---

## Settled Decisions

Reached during review; recorded here so implementation does not relitigate them.

| Decision | Resolution | Rationale |
| --- | --- | --- |
| What the pattern matches | File **content**, attributed to the enclosing symbol | The distinctive capability; name-only search is a grep any tool can do |
| Compatibility target | ripgrep | Claude Code's `Grep` is rg-backed |
| Migration | Replace `find` in place; delete `Pattern` and the `[path]::[name]` grammar | Nothing released; two overlapping search commands is the confusion the `find`/`show` split already warns about |
| Schema version | **No bump.** `find::SCHEMA_VERSION` is retired with the envelope it described | Pre-release |
| Exit codes | `0` match · `1` no match · `2` error | grep convention; agents chain with `&&` |
| `--head-limit` | Default **200** | Deliberate deviation: an uncapped grep piped into a context window is a hazard |
| Journal credit | A symbol that **contained a printed hit** is recorded at `FullBody` | Explicit product decision: ambit scanned the whole body at a verified hash |
| Staleness | A drifted symbol that matched gets its hash refreshed; one with no hit stays stale | Falls out of `ContextLedger::record` for free |
| Journaling mechanism | The TUI's existing path: `ContextLedger::record` + `Journal::sync` | No new journaling system. See §6.6 |
| Symbol enumeration (`find 'src/app.rs::'`) | **Cut.** Not reimplemented as `ambits outline` | It is `--dump --filter` plus a depth column; folding it there is a follow-up, not new surface |

---

## Current State

### Call chain today

```
main.rs::main
  └─ registry.scan_project(root, filter)          // main.rs:359 — parses EVERY file
       └─ ProjectTree { files: Vec<FileSymbols> }
  └─ CoverageIndex::load(root, session_id)        // main.rs:409 — journal → read set
  └─ Commands::Find { pattern, limit, format }    // main.rs:430
       └─ find::run(tree, patterns, limit, json, coverage)   // find.rs:180
            └─ find::search(tree, &Pattern::parse(q), limit) // find.rs:130
                 └─ tree.walk()                   // symbols/mod.rs:168 — flat Vec of all symbols
                 └─ filter by Pattern::matches_path / matches_name
            └─ lookup::describe_summary(file, node, coverage) // lookup.rs:194
```

### What the existing structures already provide

| Capability | Where | Reusable as-is? |
| --- | --- | --- |
| Symbol byte spans (`byte_range: Range<u32>`) | `symbols/mod.rs:47` | **Yes** — the basis of attribution |
| Innermost-symbol-containing-a-byte | `callers.rs:248` `fn enclosing` (private) | **Yes**, after being made public and moved |
| Read depth by symbol id | `restore.rs:249` `CoverageIndex::depth_of` | **Yes** |
| "No journal" vs "nothing read" distinction | `lookup.rs:139` `CoverageDto` | **Yes** |
| Path scoping, component-wise | `filter.rs` `PathFilter::Literal` | **Yes** — reused for `PATH...` args |
| gitignore/hidden-aware walk | `parser/mod.rs:113` `WalkBuilder` | **Yes**, after extraction |
| Read recording + staleness | `tracking/mod.rs:111` `ContextLedger::record` | **Yes** — see §6.6 |
| Append-only journal with dedup | `journal.rs:499` `Journal::open` / `:603` `sync` | **Yes**, with one signature change |
| Glob (`-g`) and type (`-t`) filtering | `ignore::overrides`, `ignore::types` | **Yes** — already a dependency, unused features |
| Regex engine | `regex` crate | **Yes** — the same crate ripgrep uses, so pattern syntax parity is exact, including the shared absence of backreferences and lookaround |

### The three gaps

1. **Source text is dropped.** `LanguageParser::parse_file(path, &str)` returns
   `FileSymbols` and the text goes out of scope. `lookup.rs` re-reads on demand;
   `callers.rs:261` re-reads every file. Nothing retains bytes, so nothing can match
   content.

2. **The pipeline is inverted for grep.** `main.rs:359` scans and parses the entire
   project before dispatching. Per `parser/mod.rs:100`, the walk is effectively free
   while parsing is ~95 ms of the ~112 ms before a command can answer. Grep wants the
   opposite order: match raw bytes first, parse only files that hit.

3. **No persistent index of any kind.** `cache.rs` is coverage journals, not symbols.
   Every invocation is a cold rebuild.

### Measurements (this repo: 40 files, ~20k lines, 1313 symbols)

| | |
| --- | --- |
| Full scan + parse, warm page cache | ~25–30 ms |
| Same, cold | ~350 ms |
| Symbols scanned by one `find depth_of` | 1313 |
| Files containing `depth_of` | 7 |
| Symbols in those files | 345 |
| Matched lines | 74 (≈30 distinct symbols) |

**Verdict: no new index is required, and building one first would be wrong.** Inverting
to prefilter-first *deletes* parse work — grep-find will be cheaper than today's `find`
on any repo where most files do not match. An index would need invalidation, a storage
format, and a staleness story of its own. Deferred; see §10.

---

## Proposed Design

### 6.1 Interface

```
ambits [-p <root>] [-a <agent>] find [OPTIONS] PATTERN [PATH...]
```

`PATTERN` is a regex over file content. `PATH...` are files or directories, resolved
relative to the **current working directory** (as in grep), then made project-relative;
a path outside `--project` is an error (exit 2). Matching is component-wise via
`PathFilter::Literal`, so `src/parser` never matches `src/parser_extra.rs`.

`PATH...` intersects (AND) with the global `--filter` / `--filter-regex`.

**rg-compatible — same spelling, same meaning:**

| Flag | Meaning |
| --- | --- |
| `-e, --regexp <P>` | Additional pattern; repeatable. Replaces today's repeatable positional |
| `-i, --ignore-case` | |
| `-w, --word-regexp` | |
| `-x, --line-regexp` | |
| `-F, --fixed-strings` | |
| `-v, --invert-match` | |
| `-U, --multiline` | Pattern may span lines; line/col report the match **start** |
| `-g, --glob <G>` | Repeatable, `!` negates. `ignore::overrides::OverrideBuilder` |
| `-t, --type <T>` | `ignore::types::TypesBuilder::add_defaults()` — the full rg type list, not just our three parsers |
| `-A/-B/-C <N>` | Context lines, printed with `-` separators as in grep/rg |
| `-n / -N` | Line numbers on/off (on by default) |
| `--column / --no-column` | Column on by default (rg's default is off; see deviations) |
| `-l, --files-with-matches` | |
| `-c, --count` | Matching lines per file |
| `--count-matches` | Total matches per file |
| `-o, --only-matching` | |
| `-q, --quiet` | No output; exit code only |
| `-m, --max-count <N>` | Per-file match cap |
| `-M, --max-columns <N>` | Truncate long lines, marked |
| `--hidden`, `--no-ignore` | Passthrough to `WalkBuilder` |
| `--heading / --no-heading` | |
| `--color <when>` | `auto` (default) / `always` / `never` |
| `--json` | rg JSON Lines event stream; see §6.2 |

**ambit extensions — long-only.** Short letters belong to rg; we do not squat on them
(note `-s` is `--case-sensitive` in rg and `--no-messages` in grep, which is why symbol
enumeration did not become `-s`).

| Flag | Meaning |
| --- | --- |
| `--head-limit <N>` | Global cap across all files. Default 200, `0` = unlimited |
| `--no-symbol` | Drop the attribution field → byte-identical rg output |

The existing global `--no-journal` additionally suppresses find's journal writes, as
does `enabled = false` in the `[cache]` stanza of `tools.toml`.

**Deliberate deviations from rg**, each documented in `--help`:

1. **Always sorted** by `(path, line, col)`. rg is order-nondeterministic under
   parallelism unless `--sort path`. Deterministic output is worth more to an agent
   than the microseconds, and the scanner already sorts for the same reason
   (`parser/mod.rs:184`).
2. **`--head-limit` defaults to 200**, with a trailer naming what was withheld.
3. **`-M/--max-columns` defaults to 300**, truncating rather than omitting the line
   (rg's default is unlimited; rg omits and needs `--max-columns-preview` to show).
   Same rationale as `--head-limit`: output lands in a context window.
4. **`--column` defaults on.** The column is what makes attribution unambiguous when a
   line has several matches.

**Reserved, unimplemented, failing loudly** rather than silently ignored: `-r/--replace`,
`-P/--pcre2`, `-f/--file`, `-z/--search-zip`, `-a/--text`, `--pre`.

**Known wart:** ambit's globals precede the subcommand (`ambits -p . find …`), so `-p`
reads oddly against rg's `-p/--pretty`. Renaming a flag every command shares costs more
than it buys. Left alone.

### 6.2 Output contract

**Piped (how agents call it).** The `file:line:col:` prefix is sacred; ambit's field
goes at the head of the text so `file:line:col:`-splitting parsers (quickfix, editors)
keep working:

```
src/find.rs:178:26:[full run] .filter(|(_, n)| c.depth_of(&n.id).is_some())
src/lookup.rs:214:22:[—   describe] read_depth: coverage.and_then(|c| c.depth_of(…
```

The bracketed field is `[<depth> <name-path>]`, where `<depth>` is one of
`full`/`signature`/`overview`/`name`, `—` for a symbol with no recorded read, and the
whole field is **omitted entirely when no coverage journal exists** — an empty column
would read as "unread" when the truth is "unknown", the same rule `find.rs:229` applies
today. A hit in a file no parser handles renders `[— -]`.

`--no-symbol` produces byte-identical-to-rg output.

**TTY.** rg's `--heading` grouping; the annotation moves to its own dim column:

```
src/find.rs
  178:26  [full]  run       .filter(|(_, n)| c.depth_of(&n.id).is_some())
  184:13  [full]  run       read,
```

**Trailer goes to stderr**, not stdout, so stdout stays strictly grep-shaped:

```
… 1,842 more matches withheld (--head-limit 0 for all)
```

**`--json`** is rg's event stream (`begin` / `match` / `end` / `summary`) with exactly
two additions:

```jsonc
{"type":"match","data":{ /* …rg fields… */,
  "symbol":{"id":"src/find.rs::run","label":"fn","lines":[160,241],"read_depth":"full"}}}
{"type":"summary","data":{ /* …rg stats… */,
  "coverage":{"session_id":"…","symbols_read":312},
  "withheld":1842}}
```

`symbol` is `null` for a hit in an unparseable file; `read_depth` is omitted when
unread; `coverage` is absent when no journal was loaded — preserving the distinction
`CoverageDto` exists to carry. `symbol.id` remains a valid `show` selector, so
`find | show` still composes.

With `-l`/`-c`/`--count-matches`, `--json` emits `begin`/`end`/`summary` with stats and
no `match` events. `--json` overrides `--heading` and `--color`.

### 6.3 Exit codes

| Code | Meaning |
| --- | --- |
| 0 | At least one match |
| 1 | No match (not an error; nothing on stderr) |
| 2 | Usage error, bad regex, unreadable project, path outside root |

This replaces today's deliberate exit-0-on-no-match (`find.rs:178`). The reasoning
there — "no symbol is named that is an answer, not a failure" — still holds for `show`,
which keeps its behavior; but a grep-shaped tool in a shell pipeline must honor grep's
convention.

### 6.4 Pipeline

```
walk (ignore: gitignore + hidden + overrides(-g) + types(-t) + PATH args + --filter)
  └─ read bytes
      └─ NUL byte in first 8 KiB ──────────────► skip (binary)
      └─ regex::bytes prefilter over the buffer ── no hit ─► drop  (NEVER PARSED)
          └─ collect line matches (byte offset, line, col, span)
              └─ parse the file ONCE ──► FileSymbols
                  └─ enclosing(byte) ──► &SymbolNode
                      ├─ CoverageIndex::depth_of(id) ──► display
                      └─ ledger.record(...)          ──► journal (§6.6)
```

Notes:

- **`regex::bytes::Regex` throughout.** Its internal literal/memchr prefilters are what
  make the file-level reject fast. Line text is decoded lossily only for display; a file
  that is not valid UTF-8 still reports hits but cannot be parsed, so `symbol` is `null`.
- **Parallelism** mirrors `scan_project`: `std::thread::scope` over chunks, results
  merged and sorted before printing.
- **`-l` / `-c` / `-q` short-circuit before the parse** — they need no attribution.
- A file is read exactly once and parsed at most once.

### 6.5 Attribution

`callers.rs:248 fn enclosing` already implements "innermost symbol whose byte range
contains this offset", and its docstring already argues the right semantic (a call
inside a method belongs to the method, not the wrapping `impl`). It moves to
`symbols/mod.rs` as an inherent method:

```rust
impl FileSymbols {
    /// The innermost symbol whose byte range contains `byte`.
    pub fn enclosing(&self, byte: u32) -> Option<&SymbolNode>;
}
```

`callers.rs` drops its private copy and calls this instead — one containment search,
one place.

**Symbol ids are not unique.** `struct Foo` and `impl Foo` in one file both yield
`<path>::Foo`, and several inherent impls in one file still collide. Hits are therefore
keyed internally by `(file, byte_range.start)`, **never** by id; the id is a display and
journal value only.

Context lines (`-A/-B/-C`) get **no** annotation — attribution is a property of the
match, and a context line may belong to a different symbol.

### 6.6 Journaling — the TUI's path, not a new one

`ambits find` prints source into an agent's context, so it must record what it showed.
It does that through the **exact** API the TUI uses. No new record type, no new writer,
no new file.

```rust
// after collecting hits, before exit
let mut ledger = ContextLedger::new();        // empty: sync diffs against disk, not this
for sym in printed_symbols {
    ledger.record(sym.id.clone(), ReadDepth::FullBody, sym.content_hash,
                  agent_id.clone(), sym.estimated_tokens as usize);
}
Journal::open(&root, &session_id, Duration::ZERO, || manifest).sync(&ledger);
```

Every decision in §Settled falls out of behavior that already exists:

| Requirement | Existing mechanism |
| --- | --- |
| Credit matched symbols `FullBody` | `ContextLedger::record` (`tracking/mod.rs:111`) — depth is upgrade-only |
| Refresh the hash **only** if it matched | `record` sets `content_hash_at_read`, `stale = false`, `provenance = Live` on *every* read (`tracking/mod.rs:148`). Symbols with no hit are never recorded, so they keep their old hash and stay stale |
| Never double-write | `Journal::open` seeds its `journaled` map from disk; `sync` appends only `(symbol, agent)` pairs whose `(hash, depth)` moved |
| Drifted-but-matched supersedes the old record | Append-only + last-record-wins; already pinned by `re_read_after_drift_appends_the_new_hash` |
| Concurrent TUI and CLI writers | `journal.rs:367 fn fold`: same hash → `max(depth)`, different hash → supersede. A TUI later appending `Overview` at the same hash folds **up** to `FullBody` rather than clobbering it. Records are a single `write_all` under `O_APPEND`, ~150 bytes, which `journal.rs:56` already designed for |

**The credit rule, stated precisely:** a symbol is journaled iff **at least one of its
matched lines was actually printed**. Therefore `-q`, `-l`, `-c` and `--count-matches`
credit nothing (the agent saw no source), and symbols whose matches fell past
`--head-limit` credit nothing. The journal must record what the agent saw, not what the
process computed.

**Agent id** reuses the existing global `-a/--agent`, defaulting to the session's most
recent agent — which, when an agent shells out to `ambits find`, is the agent whose
context received the output. With no resolvable session there is no journal to write to
and journaling is skipped.

**The one required API change.** `Journal::open` takes an `EnvironmentManifest`, and
`EnvironmentManifest::capture` (`journal.rs:181`) needs a full `ProjectTree` for
`tree_fingerprint` — the very scan grep-find is designed to skip. Make the manifest
lazy:

```rust
pub fn open(root: &Path, session: &str, interval: Duration,
            manifest: impl FnOnce() -> EnvironmentManifest) -> Self
```

`open` only writes a header when the file has none or its version is stale, so the full
scan is paid on the first find of a session and never again. The TUI passes
`|| EnvironmentManifest::capture(&tree, backend, filter)` and is otherwise untouched.

**Two documented invariants change and must be updated in the same commit:**

- `journal.rs:62` — *"Only the TUI writes. Readers (`restore-context`) never open the
  file for writing, which removes concurrent-writer concerns rather than managing
  them."* Now managed; document `fold` and `O_APPEND` as the mechanism.
- `cache.rs:109` — *"Journals are written by the TUI."*

**Honest caveat, recorded deliberately:** a symbol credited `FullBody` because one line
matched will be reported to a future agent as known in full. The hash is verified, so
drift detection stays sound; the exposure is over-crediting at restore time. Accepted as
a product decision.

**Why not merkle?** `merkle_hash` is computed by all three parsers and read in exactly
one non-test place — `journal.rs:289`, folding top-level symbols into
`tree_fingerprint`. Every staleness path compares `content_hash`, correctly: a subtree
hash says *something under here changed*, not *which symbol drifted*, and the journal is
keyed per symbol. Merkle also cannot save the work — knowing a file's symbols are
unchanged requires reading and hashing them, which is the parse we are skipping. Its
real use is as the invalidation key for a future parse cache (§10), where those
already-computed hashes are currently dead weight.

### 6.7 Tool-config parity — **dropped, with reasons**

The plan was to mirror `default_tools.toml:123`'s bash `grep `/`rg ` → `Overview`
mapping with an `ambits find` prefix. Investigating it showed the line would be
cosmetic: a `Bash` tool call carries no `file_path` (`path_keys = []`, and
`empty_path_keys_produces_none_file_path` pins it), so the stanza's depth never reaches
a symbol. It tints the activity feed and nothing else. With find journaling itself
authoritatively (§6.6), an inert config line is noise that reads like a mechanism.

**A real interaction turned up instead.** The `Bash` stanza's `target_selectors` rule
scans the *command string* for anything that parses as a symbol id and credits it
`FullBody`. That was right when a `::` in an `ambits` command was the old find's query
syntax. Now the pattern is a regex over content, so `ambits find 'src/app.rs::App'` —
or any search for text containing `::` — credits a symbol the search may never have
shown, which is exactly the over-crediting §6.6 is careful to avoid. Expressing
"except for `find`" needs an `excludes` counterpart to `requires` in the config schema,
which is an ingest-side change. Filed as a follow-up rather than smuggled into this
work.

---

## Code Changes by File

| File | Change |
| --- | --- |
| `src/find.rs` | **Rewritten.** `Matcher` (patterns + flags), `search_file`, `Hit`, printers, `run`. `Pattern`, `search`, `SCHEMA_VERSION`, `DEFAULT_LIMIT` deleted. Module doc rewritten — the current one argues for a grammar we are removing and states a "definitions, not usages" contract that inverts |
| `src/symbols/mod.rs` | Add `FileSymbols::enclosing(byte)` |
| `src/callers.rs` | Delete private `enclosing`; call the shared one |
| `src/parser/mod.rs` | Extract `walk_files(root, filter, overrides, types) -> Vec<(PathBuf, PathBuf)>` from `scan_project`; `scan_project` calls it |
| `src/main.rs` | New `Find` clap variant; **dispatch `find` before `scan_project`** (done in P0, as `Cache` does); `scan_tree` helper; exit-code plumbing |
| `src/journal.rs` | Lazy manifest in `open`; module-doc invariant update |
| `tests/helpers/mod.rs` | `sym_with_bytes` fixture for byte-range attribution tests |
| `src/cache.rs` | Doc line update |
| `src/ingest/default_tools.toml` | `ambits find` prefix pattern |
| `src/lookup.rs` | `describe_summary` may become unused by `find`; keep or retire with `show`'s needs in mind |
| `skills/ambit/SKILL.md` | §"Finding symbols" rewritten; the "**This searches definitions, not usages** … Use grep for call sites" paragraph is now false and must go |
| `skills/ambit/examples.md`, `README.md` | Examples updated |

---

## Testing Plan

### Unit — `src/find.rs`

Matcher and flags:

1. `a_bare_pattern_is_a_regex_over_content`
2. `fixed_strings_disables_metacharacters`
3. `ignore_case_and_word_boundaries_compose`
4. `line_regexp_anchors_the_whole_line`
5. `invert_match_reports_non_matching_lines`
6. `multiline_reports_the_line_of_the_match_start`
7. `max_count_caps_per_file_and_head_limit_caps_globally`
8. `max_columns_truncates_and_marks`
9. `an_invalid_regex_is_an_error_not_an_empty_result`

Attribution:

10. `a_hit_is_attributed_to_the_innermost_symbol`
11. `a_hit_between_symbols_has_no_symbol`
12. `a_hit_in_an_unparseable_file_has_no_symbol`
13. `colliding_ids_do_not_merge_hits` — `struct Foo` and `impl Foo` in one file
14. `context_lines_carry_no_attribution`

Coverage display:

15. `a_read_symbol_reports_the_depth_it_was_read_at` (ported)
16. `an_unread_symbol_renders_an_em_dash`
17. `no_journal_omits_the_column_entirely` (ported — the load-bearing one)

Output:

18. `piped_output_keeps_the_file_line_col_prefix`
19. `no_symbol_produces_rg_identical_output`
20. `json_match_events_carry_the_symbol`
21. `json_summary_carries_coverage_and_withheld`
22. `results_are_sorted_by_path_line_col`
23. `the_withheld_trailer_goes_to_stderr`

### Unit — journaling

24. `a_printed_hit_records_full_body_for_its_symbol`
25. `a_symbol_with_no_hit_is_not_recorded`
26. `a_matched_symbol_that_drifted_gets_a_fresh_hash`
27. `an_unmatched_drifted_symbol_stays_stale`
28. `quiet_and_files_with_matches_record_nothing`
29. `symbols_past_head_limit_record_nothing`
30. `no_journal_flag_suppresses_writes`
31. `cache_disabled_in_tools_toml_suppresses_writes`

### Unit — `src/journal.rs`

32. `open_does_not_build_a_manifest_when_a_header_exists` — the lazy closure must not
    run; pin with a closure that panics
33. `concurrent_appends_all_parse` — N threads appending, every line valid JSON
34. `fold_upgrades_depth_at_an_equal_hash` — TUI `Overview` after find's `FullBody`
    folds to `FullBody`

### Unit — `src/symbols/mod.rs`

35. `enclosing_prefers_the_innermost_symbol`
36. `enclosing_is_none_outside_every_range`

### Integration — `tests/e2e.rs` (first e2e coverage for `find`)

37. `find_exits_zero_on_match_one_on_no_match_two_on_bad_regex`
38. `glob_and_type_filters_narrow_the_walk`
39. `path_args_intersect_with_the_global_filter`
40. `a_path_outside_the_project_root_is_an_error`
41. `binary_files_are_skipped`
42. `find_then_show_composes_on_the_emitted_id`

### Benchmarks — `benches/`

43. `non_matching_files_are_never_parsed` — the load-bearing performance invariant of
    the whole design. Assert parse count, not wall time
44. Grep throughput on this repo, matching and non-matching patterns

---

## Non-Goals for v1

- `--replace`, `--pcre2`, `--file`, `--search-zip`, `--text`, `--pre`
- A persistent symbol or content index (§10)
- Reimplementing symbol enumeration as a new command
- `--coverage` consulting the journal (§10)
- Hot-reload, daemon mode, or an MCP surface for find
- Journal sharding per process

---

## Follow-Ups (filed separately, not in this spike)

1. **Symbol enumeration → `--dump`.** `find 'src/app.rs::'` and `find '::App/'` are
   `--dump` scoped by path plus a read-depth column. Fold them into `--dump` rather
   than adding a third search-ish command.
2. **`--coverage` reads the journal.** It replays session logs, so find-credited reads
   will not appear in its table. Same root cause as the known inflation issue.
3. **Parse cache keyed by per-file merkle hash** under `.ambit/index/`, maintained
   opportunistically by the TUI's existing `notify` watcher. Only if cold-start on a
   large repo is measured to hurt. A trigram content index is the *last* thing to
   build — ripgrep's thesis is that brute force with good prefilters beats index
   maintenance below ~1M files.
4. **Journal shards per process** (`<session>.<pid>.ndjson`) if the concurrency stress
   test ever tears. Ship single-file first.
5. **`target_selectors` must not credit `find`'s pattern.** See §6.7: a search for text
   containing `::` is credited as a read of that symbol id. Needs an `excludes` field
   beside `requires` in the tool-config schema.

---

## Implementation Order

Sequenced to keep the build green at each step.

**P0 — preparation (mechanical, independently landable) — DONE**

1. ✅ `FileSymbols::enclosing` in `symbols/mod.rs`, with the recursive half private
   beside it and tests 35–36. `sym_with_bytes` added to `tests/helpers/mod.rs`.
2. ✅ `callers.rs`'s private `enclosing` deleted; `find_callers` calls
   `file.enclosing(byte)`. Its `attribution_picks_the_innermost_enclosing_symbol` test
   and the two local range fixtures moved to `symbols/mod.rs` with the code they cover.
3. ✅ `parser::walk_files` extracted, **with `WalkOptions` in the same step** rather
   than as a second pass — extracting a signature only to rewrite it one step later is
   churn. Defaults reproduce the scanner's behaviour exactly; the `parser_for`
   extension test stays in `scan_project`, since a content search wants every text
   file in scope.
4. ✅ (folded into 3.)
5. ✅ `find` dispatches ahead of the scan in `main.rs`, as `cache` does. Session and
   log-dir resolution hoisted above it and switched from consuming `cli.log_dir` /
   `cli.session` to cloning, so `--coverage` still resolves its own from the raw CLI
   values. The two scan call sites share a new `main::scan_tree` helper. `find` still
   asks for a full tree here — scaffolding P1 removes.

Verified: 530 tests pass (443 lib + 87 integration), no new clippy warnings, and
`find` output is unchanged across `render`, `src/app.rs::`, `::App/`, `ui::render`,
and a no-match query.

**P1 — core search**

6. `Matcher` + flag parsing in the new `find.rs`; tests 1–9.
7. `search_file` with the byte prefilter and binary skip; test 41.
8. Attribution via `enclosing`; tests 10–14.
9. Text output, both modes; tests 18–19, 22–23.
10. JSON output; tests 20–21.
11. Exit codes; test 37.
12. Delete `Pattern`, `search`, `SCHEMA_VERSION`, `DEFAULT_LIMIT`. Rewrite the module
    doc.

**P2 — journaling — DONE**

13. ✅ `Journal::open` takes `manifest: impl FnOnce() -> EnvironmentManifest`, with
    `interval` moved ahead of it so the closure sits last. `App::enable_journal` passes
    a closure; test 32 pins it with one that panics.
14. ✅ `find::journal_reads` builds a `ContextLedger`, records each shown symbol at
    `FullBody`, and `sync`s once. `shown_symbols` extracted from `run` so the credit
    rule is testable without capturing stdout. Tests 24–27 plus two more: modes that
    print no source credit nothing, and repeated hits in one symbol are one read.
15. ✅ `journal_enabled` hoisted so `find` shares the TUI's `--no-journal` /
    `[cache] enabled` decision. Verified end-to-end against a temp project; CLI-level
    tests 30–31 move to P4 with the other binary-level tests.
16. ✅ Tests 33–34: four concurrent writers produce 100 intact records with no
    warnings, and a later `Overview` at an equal hash folds **up** to `FullBody`.
17. ✅ `journal.rs` module doc now states how concurrent writers are held safe
    (`O_APPEND` whole-line appends, `open`'s dedup seed, `fold`'s greatest-depth rule)
    rather than claiming only the TUI writes. `cache.rs` status text updated.
18. ❌ Dropped — see §6.7.

**P3 — surface**

19. `SKILL.md`, `examples.md`, `README.md`. The false "use grep for call sites"
    paragraph must go.

**P4 — guardrails**

20. Remaining e2e tests 38–40, 42.
21. Benchmarks 43–44.
22. `cargo test --all` green; manual smoke against a real Claude Code session,
    verifying the depth column and that the journal grows by exactly the symbols shown.
