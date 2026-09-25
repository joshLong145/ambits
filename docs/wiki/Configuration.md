# Configuration

## The read journal

While the [TUI](TUI) runs it maintains an append-only NDJSON record at
`.ambits/coverage/<session>.ndjson` — one entry per `(symbol, agent)` read,
including the content hash the symbol had at the time. This is what lets
`restore-context` answer after the fact, and it survives restarts.

The TUI is the only writer: a search or `show` run with no TUI attached to the
session earns no coverage credit. That is a deliberate trade for never needing
two processes to reason about writing the same journal.

```bash
ambits -p . cache status              # journals on disk, size and age
ambits -p . cache clear --session <id>
ambits -p . cache clear --all
```

Size is bounded by what an agent can read in one session, not by repository
size — a full day of heavy work on this repository runs to a few hundred KB.
Nothing is pruned automatically, and `cache clear` requires naming a target:
a journal is the only record of what a past session read *and what it looked
like at the time*. Deleting one downgrades any later restore of that session to
the UNVERIFIED session-log fallback.

| Setting | CLI | `tools.toml` |
|---|---|---|
| Disable the journal | `--no-journal` | `[cache] enabled = false` |
| Write interval | `--flush-interval-ms N` | `[cache] flush_interval_ms = N` |

CLI flags win over the config file.

## Restricting scope

```bash
ambits -p . --filter src/parser              # by path component
ambits -p . --filter-regex '^src/.*\.rs$'    # by regex
```

`--filter` matches whole path components, so `src/parser` matches
`src/parser/rust.rs` but not `src/parser_extra.rs`; a leading slash is
accepted, and a path that does not exist is an error. `--filter-regex` is
unanchored unless you anchor it. The two are mutually exclusive.

## `tools.toml`

One file configures tool mappings, the editor and the journal. Only one user
file is used — the first found of:

1. `--tools-config <path>`
2. `.ambits/tools.toml` in the **current working directory** (not the `-p` path)
3. `~/.config/ambit/tools.toml`

A `tools.toml` left in the pre-0.21 location, `.ambit/tools.toml`, is still
read when no `.ambits/tools.toml` exists, with a warning to move it. Journals
in the old `.ambit/coverage/` are moved to `.ambits/coverage/` automatically
the first time any command needs them.

That file merges over the built-in defaults
([`src/ingest/default_tools.toml`](https://github.com/joshLong145/ambits/blob/main/src/ingest/default_tools.toml)),
which are also the best reference for the format.

### Tool mappings

How a tool call becomes a symbol read is data. Add a stanza to teach ambits a
tool it does not know, or to change the depth an existing one grants:

```toml
version = 1

[[tool]]
names         = ["MyCustomReader"]
path_keys     = ["path"]
depth         = { type = "fixed", value = "FullBody" }
description   = "MyCustomReader {path}"
```

| Key | Meaning |
|---|---|
| `names` | Tool names this stanza applies to |
| `path_keys` | Input keys that may hold the file path |
| `pattern_keys` | Input keys that may hold a search pattern |
| `path_required` | Default `true`; set `false` for tools where the path is an optional filter (Glob, Grep, Bash) |
| `depth` | How deeply a call reads the symbols it touches (below) |
| `description` | Activity-feed line; `{key}` interpolates an input value |
| `extends` | Name of a built-in stanza to inherit unset fields from |
| `target_symbol`, `target_lines`, `target_selectors` | Narrow the read to specific symbols rather than the whole file; see the built-ins |

Depths are `Unseen`, `NameOnly`, `Overview`, `Signature`, `FullBody`. A
`depth` is one of:

```toml
# Always the same
depth = { type = "fixed", value = "Overview" }

# Chosen by a boolean input key
depth = { type = "conditional", condition_key = "…", if_true = "FullBody", if_false = "Signature", default = "Signature" }

# Chosen by matching an input string; first match wins
depth = { type = "pattern_match", key = "command", default = "NameOnly", patterns = [
    { prefix = "cat ",  depth = "FullBody" },
    { prefix = "head ", depth = "Signature" },
    { prefix = "rg",    match_type = "ambits_subcommand", depth = "Overview" },
] }
```

`match_type` is `prefix` (the default), `contains`, `exact`, or
`ambits_subcommand` — which matches an `ambits` subcommand wherever it sits in
the command line, since global flags come between the binary and the
subcommand (`ambits -p . grep …`).

### Editor

```toml
[editor]
command = "zed"                         # or a template: "code -g {file}:{line}"
```

See [TUI → Opening a symbol in your editor](TUI#opening-a-symbol-in-your-editor)
for how commands and templates are resolved.

### Cache

```toml
[cache]
enabled           = true    # default
flush_interval_ms = 5000    # default
```

## Parsing backends

| Backend | Languages |
|---|---|
| tree-sitter (default) | Rust, Python, TypeScript, Markdown |
| [Serena](https://github.com/oraios/serena) MCP | Any language Serena supports |

```bash
ambits -p . --serena
```

Markdown symbols are ATX headings (`#` through `######`), nested by level; a
heading's span runs to the next heading of equal or shallower depth. Content
before the first heading becomes a `preamble` symbol.

## Claude Code skill

```bash
ambits skill install --global      # ~/.claude/skills/ambit/ — all projects
ambits skill install               # .claude/skills/ambit/ in the current project
ambits skill install --project /path/to/project
```

Installs a [skill](https://code.claude.com/docs/en/skills) that teaches the
agent when to check its own coverage and how to fetch definitions.

## Debug logging

```bash
ambits -p . --log-output ./logs
```

Writes one structured JSON-lines log per session to `<dir>/<session>.log`:
session-log ingest, tool/depth resolution, symbol updates, parsing, and one
activity line per tool call. It is file-only, never stderr, because stray
output would corrupt the TUI; with no `--log-output` no logger is installed at
all.

`Info` is the default level. `RUST_LOG=ambits=debug` adds the noisier
internal-diagnostics targets (`ambits::…`), and `RUST_LOG` can override in
either direction as usual.
