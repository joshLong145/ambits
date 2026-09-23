# ambits

**A code-reading tool for AI agents, and a memory of what they have read.**

Coding agents read whole files to find one function, re-read code they already
know, and lose all of it the moment the context window compacts. ambits
addresses both halves of that:

- **Reading** — search, callers and symbol lookup that answer in symbols rather
  than files. The agent asks for `App/process_compaction`, not for 2,000 lines
  of `app.rs`.
- **Remembering** — every symbol the agent reads is recorded, at what depth,
  throughout the session. After a compaction that history is handed back, so
  the agent knows what it already understands instead of rediscovering it.

## Pages

**Start here**
- [Getting Started](Getting-Started) — install, hook, skill, first session

**For the agent**
- [Searching Code](Searching-Code) — `rg` / `grep`, with every hit attributed to a symbol
- [Finding Callers](Finding-Callers) — call sites from the grammar, not from text
- [Reading by Symbol](Reading-by-Symbol) — `show`: definitions as JSON, by id or hash
- [Restoring Context](Restoring-Context) — `restore-context` and the post-compaction hook
- [Portability](Portability) — the formats, and where the provider-specific seams are

**For you**
- [TUI](TUI) — panels, keybindings, colors, opening symbols in your editor
- [Coverage and Multi-Agent](Coverage-and-Multi-Agent) — reports, JSON, sub-agent filtering and alignment
- [Configuration](Configuration) — the read journal, scope filters, `tools.toml`, backends, logging

**Reference**
- [CLI Reference](CLI-Reference) — every command and flag
- [Architecture](Architecture) — the memory loop and how a query is served
- [Benchmarks](Benchmarks)

> These pages are generated from [`docs/wiki/`](https://github.com/joshLong145/ambits/tree/main/docs/wiki)
> in the main repository. Edit them there, not here — changes made in the wiki
> UI are overwritten the next time the wiki is published.
