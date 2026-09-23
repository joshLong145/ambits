# Ambits

[![e2e](https://github.com/joshLong145/ambits/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/joshLong145/ambits/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/joshLong145/ambits/graph/badge.svg?token=9Q8GWA8H6Y)](https://codecov.io/gh/joshLong145/ambits)

**A code-reading tool for AI agents, and a memory of what they have read.**

Coding agents read whole files to find one function, re-read code they already know, and lose all of it the moment the context window compacts. ambits addresses both halves of that:

- **Reading** — search, callers and symbol lookup that answer in symbols rather than files. The agent asks for `App/process_compaction`, not for 2,000 lines of `app.rs`.
- **Remembering** — ambits records every symbol the agent reads, at what depth, throughout the session. After a compaction it hands that history back, so the agent knows what it already understands instead of rediscovering it.

Both surfaces are plain text and JSON with no vendor coupling. Ingestion is currently built and tested against Claude Code.

![screenshot](./images/screenshot.png)

## Quick start

```bash
cargo install ambits

# Hand read history back automatically after every compaction
ambits hook install --project .

# Watch the session live (and keep the read journal)
ambits -p .
```

## For the agent

**Search, with every hit attributed to a symbol** — ripgrep's flags, or GNU grep's via `ambits grep`:

```bash
ambits -p . rg 'is_binary' -t rust
```
```
src/search.rs:390:4:[full is_binary] fn is_binary(buf: &[u8]) -> bool {
src/search.rs:495:8:[— search_file]     if is_binary(&buf) || !matcher.worth_searching(&buf) {
```

The bracket shows the enclosing symbol and how deeply this session has read it. → [Searching Code](https://github.com/joshLong145/ambits/wiki/Searching-Code)

**Find callers** from the grammar's call nodes, never from comments or strings:

```bash
ambits -p . callers centered_rect
```
→ [Finding Callers](https://github.com/joshLong145/ambits/wiki/Finding-Callers)

**Read one symbol, not the file** — by id or content hash, as JSON:

```bash
ambits -p . show 'src/app.rs::App/process_compaction' b3:53a28842
```
→ [Reading by Symbol](https://github.com/joshLong145/ambits/wiki/Reading-by-Symbol)

**Recall what it has already read**, with line ranges that stay correct after edits:

```bash
ambits -p . restore-context
```
```
### src/digest.rs — 47 symbols (~13.7k tok)
grouped:64-84, symbol_label:104-119, fit_names:126-141, …
```
→ [Restoring Context](https://github.com/joshLong145/ambits/wiki/Restoring-Context)

## For you

**The TUI** (`ambits -p .`) tails the session log live: a symbol tree shaded by read depth, coverage stats, an activity feed, per-agent filtering for sub-agents, and compaction history. → [TUI](https://github.com/joshLong145/ambits/wiki/TUI)

**Coverage reports** for scripts and CI:

```bash
ambits -p . --coverage                                   # table
ambits -p . --coverage --format json | jq '.totals.full_percent'
```
→ [Coverage and Multi-Agent](https://github.com/joshLong145/ambits/wiki/Coverage-and-Multi-Agent)

## Languages

| Backend | Languages |
|---|---|
| tree-sitter (default) | Rust, Python, TypeScript, Markdown |
| [Serena](https://github.com/oraios/serena) (`--serena`) | Anything Serena supports |

Search covers every text file; a file no parser handles simply has no symbols.

## Documentation

Full docs are in the **[wiki](https://github.com/joshLong145/ambits/wiki)**:

- [Getting Started](https://github.com/joshLong145/ambits/wiki/Getting-Started) — install, hook, skill, first session
- [Configuration](https://github.com/joshLong145/ambits/wiki/Configuration) — read journal, scope filters, `tools.toml`, logging
- [Portability](https://github.com/joshLong145/ambits/wiki/Portability) — formats and provider seams
- [Architecture](https://github.com/joshLong145/ambits/wiki/Architecture) — the memory loop and query pipelines
- [CLI Reference](https://github.com/joshLong145/ambits/wiki/CLI-Reference) — every command and flag

The wiki's source is [`docs/wiki/`](docs/wiki/); edit it there.

## Building from source

Requires Rust 1.82+ (declared as `rust-version` in `Cargo.toml`).

```bash
cargo build --release
cargo test
```
