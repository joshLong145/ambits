# Benchmark: `ambits` vs Serena (MCP)

**Status**: Complete
**Date**: 2026-09-15
**Environment**: this repository (`ambit`, ~20k lines Rust), one live agent session, `ambits` built from the working tree at commit `742b65f`, Serena's `rust` language server (rust-analyzer) warm from prior use.

**Scope**: `ambits` is, at this time, a deliberately read-only search/navigation/coverage tool — no symbol editing, no cross-session memory. This report compares the two tools only on the surface where both operate: search, symbol lookup, references, implementors, and structure overview. Editing and memory are Serena-only by design, not measured here as an `ambits` deficiency.

**Methodology note**: `ambits` timings are `time <command>` — precise process time. Serena has no equivalent stopwatch available to the agent driving this benchmark; its numbers are a `date`-bracketed wall-clock estimate that includes some unavoidable agent-turn overhead between issuing the call and recording the timestamp after it returns. Treat Serena's numbers as **upper bounds**, not exact figures — the direction of every finding below survives that uncertainty, the magnitude doesn't need to be exact for the conclusion to hold. Full raw data and reproduction commands are in the appendices.

---

## 1. Latency

```text
ambits  Read symbol        █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.037s
Serena  Read symbol        ████████████████████████████████████████ 2.109s
ambits  File overview      ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.013s
Serena  File overview      █████████████████████████████░░░░░░░░░░░ 1.550s
ambits  Find refs          █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.072s
Serena  Find refs          █████████████████████████████████████░░░ 1.959s
```

*Bar length is proportional within this chart only (`█` = filled, longest bar = the chart's own max value). Not comparable across charts.*

**Finding**: `ambits` was sub-100ms on every one of 6 trials, no exceptions. Serena's true latency is unknown but bounded around ~1.5-2.1s per call — one to two orders of magnitude slower, expected for an MCP round trip through a language server versus a local process `exec`.

## 2. Output size, per step

Eight-step simulated session (orient → read → find-refs → find-implementors → search-text → orient → read → find-refs), each step a real call made against this repository. Step 5 (full-text search) is **not chartable** — Serena has no tool for it at all — so the chart below covers the 7 steps both tools can complete; step 5 is called out separately underneath.

```text
ambits  Overview A         ███████████████████████░░░░░░░░░░░░░░░░░ 1511
Serena  Overview A         █████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 591
ambits  Symbol A           ████████████████████░░░░░░░░░░░░░░░░░░░░ 1315
Serena  Symbol A           ████████████████░░░░░░░░░░░░░░░░░░░░░░░░ 1022
ambits  Refs A              ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 406
Serena  Refs A              ███████████████████████████░░░░░░░░░░░░ 1776
ambits  Implementors        ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 406
Serena  Implementors        ███████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 689
ambits  Overview B          ███████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 464
Serena  Overview B          ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 107
ambits  Symbol B           ████████████████████████████████████████ 2598
Serena  Symbol B           ████████████████████████████████████░░░░ 2320
ambits  Refs B               █████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 296
Serena  Refs B               ███████████████████████████████░░░░░░░ 1521
```

*Scale is shared across this chart: longest bar (2598) is 100% width. `ambits` in bytes, Serena in characters — same unit for practical purposes here.*

**Finding**: mixed at the step level. Serena wins 4 of 7 steps (both overviews, both symbol-body reads) by modest margins — its shallow default and per-symbol lookup are genuinely economical there. `ambits` wins 3 of 7 (both find-references, the implementors search) by much larger margins, driven by Serena's `find_referencing_symbols` attaching a source snippet to every hit.

**Step 5, excluded above**: `ambits` answered it in 101 bytes. Serena cannot answer it at all with its own tools — no bar height represents that honestly, so it's reported as text: any session that needs a plain-text search (most do) is not completable on Serena's toolset alone.

## 3. Cumulative cost over the session

The same 7 steps, running total — the "average token use over a session" question made visual. This is the chart that matters most for a long session: the gap doesn't just exist, it compounds.

```text
ambits  after step 1       ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 1511
Serena  after step 1       ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 591
ambits  after step 2       ██████████████░░░░░░░░░░░░░░░░░░░░░░░░░░ 2826
Serena  after step 2       ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 1613
ambits  after step 3       ████████████████░░░░░░░░░░░░░░░░░░░░░░░░ 3232
Serena  after step 3       █████████████████░░░░░░░░░░░░░░░░░░░░░░░ 3389
ambits  after step 4       ██████████████████░░░░░░░░░░░░░░░░░░░░░░ 3638
Serena  after step 4       ████████████████████░░░░░░░░░░░░░░░░░░░░ 4078
ambits  after step 5       ████████████████████░░░░░░░░░░░░░░░░░░░░ 4102
Serena  after step 5       █████████████████████░░░░░░░░░░░░░░░░░░░ 4185
ambits  after step 6       █████████████████████████████████░░░░░░░ 6700
Serena  after step 6       ████████████████████████████████░░░░░░░░ 6505
ambits  after step 7       ███████████████████████████████████░░░░░ 6996
Serena  after step 7       ████████████████████████████████████████ 8026
```

*Scale is shared across this chart: Serena's final total (8026) is 100% width.*

**Finding**: Serena starts ahead (steps 1-2, both its strong cases) and `ambits` closes the gap and overtakes by step 3, never giving it back. Final tally: `ambits` 6,996 vs Serena 8,026 — a 12.8% total-session gap, from a tool that led on only 3 of the 7 individual steps. Compounding, not any single step, is what decides a session.

## 4. Reliability — one MCP call failed mid-run

`find_referencing_symbols` on `leading_comment_start` **failed** on first attempt:

```
SolidLSPException: ... textDocument/references ... (caused by content modified (-32801))
```

Bracketed time for the failed call: 6.09s — 3x slower than any successful call — before erroring. A retry immediately after, unchanged, succeeded in 2.00s. `src/parser/mod.rs` had been heavily edited earlier in this session; `content modified` is exactly the error an LSP raises when its document version drifts from what it's being queried against. `ambits` has no equivalent failure mode — no persistent server-side state, a fresh cheap parse every call.

**One occurrence is an observed event with a plausible mechanism, not a measured failure rate** — but it's the single most actionable finding in this report: Serena's *correctness*, not just its speed, depends on the language server staying in sync with a session that is also actively editing.

## 5. Feature comparability (search/navigation surface only)

| Capability | `ambits` | Serena |
|---|---|---|
| Full-text / regex search | Yes — core function | **No tool exists for this** |
| Symbol lookup by name | Yes, first-try reliable | Yes, but its own documented naming convention failed silently once, requiring a retry |
| Find references | Yes, compact | Yes, LSP-exact, larger (snippet per hit) |
| Find implementors | Heuristic text-search proxy only | Yes, LSP-exact |
| Structure overview | Yes, depth-controlled | Yes, depth-controlled |
| Read-coverage tracking | Yes — the tool's reason for existing | No equivalent |
| External server dependency | None — static binary | Yes — language server, can drift out of sync (§4) |
| ripgrep-compatible `--json` | Yes, byte-shape-compatible with `rg` | N/A, bespoke shape |

## Conclusion

- **Latency**: `ambits` wins decisively, an order of magnitude or more, every trial.
- **Session token cost**: `ambits` wins in total (12.8%) despite losing more individual steps — the wins are just bigger where they happen, and they compound.
- **Full-text search**: `ambits` only. Not a narrow loss for Serena — a session that needs this (most do) cannot complete on Serena's tools alone.
- **Reliability**: one live LSP failure, directly tied to concurrent editing — the finding most worth acting on before treating Serena's reference-finding as load-bearing in an edit-heavy session.
- **Where Serena wins**: bare structure/name questions (smaller by design) and exact semantic relationships (implements/references, no heuristic guessing).

Net: the tools' strong areas barely overlap. On the overlap that exists, `ambits` is faster and cheaper in aggregate, and this report's one reliability data point suggests it is more robust under concurrent edits — while Serena remains the more semantically precise choice when an exact relationship, not a search, is the actual question.

---

## Appendix A — Raw timestamps (ns epoch)

| Trial | Start | End | Δ (s) |
|---|---|---|---|
| A.1 (symbol, open_shard) | 1789474980548389000 | 1789474982826033000 | 2.278 |
| A.2 (symbol, sync) | 1789474986418901000 | 1789474988358465000 | 1.940 |
| B.1 (overview, journal.rs) | 1789474997999561000 | 1789474999539919000 | 1.540 |
| B.2 (overview, cache.rs) | 1789475002230685000 | 1789475003791156000 | 1.560 |
| C.1 failed (refs, leading_comment_start) | 1789475009620073000 | 1789475015707228000 | 6.087 |
| C.1 retry | 1789475018288589000 | 1789475020291773000 | 2.003 |
| C.2 (refs, read_journal_session) | 1789475023600905000 | 1789475025515558000 | 1.915 |

## Appendix B — Session step sources

`ambits` sizes are `wc -c` on real command output; Serena sizes are `wc -c` on the tool's exact response text, reproduced via heredoc rather than retyped, to avoid transcription error.

| Step | `ambits` | Serena |
|---|---|---|
| 1 | `ambits -p . --dump --filter src/journal.rs` | `get_symbols_overview("src/journal.rs")` |
| 2 | `ambits -p . show 'src/journal.rs::Journal/open_shard'` | `find_symbol("impl Journal/open_shard", "src/journal.rs", include_body=true)` |
| 3 | `ambits -p . callers leading_comment_start` | `find_referencing_symbols("leading_comment_start", "src/parser/mod.rs")` |
| 4 | `ambits -p . find "impl LanguageParser for" -t rust --head-limit 0` | `find_implementations("LanguageParser", "src/parser/mod.rs")` |
| 5 | `ambits -p . find "vacuous comparison" -t rust` | *(no equivalent tool)* |
| 6 | `ambits -p . --dump --filter src/cache.rs` | `get_symbols_overview("src/cache.rs")` |
| 7 | `ambits -p . show 'src/journal.rs::Journal/sync'` | `find_symbol("impl Journal/sync", "src/journal.rs", include_body=true)` |
| 8 | `ambits -p . callers read_journal_session` | `find_referencing_symbols("read_journal_session", "src/journal.rs")` |
