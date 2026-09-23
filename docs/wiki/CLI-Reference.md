# CLI Reference

`ambits --help` and `ambits <command> --help` are authoritative; this page
collects them in one place.

## Commands

| Command | Description |
|---|---|
| `ambits -p <path>` | Launch the [TUI](TUI) |
| `ambits -p <path> rg <pattern> [path…]` | [Search](Searching-Code), ripgrep's flags; every hit names its symbol |
| `ambits -p <path> grep <pattern> [path…]` | The same search, GNU grep's flags |
| `ambits -p <path> callers <name>…` | [Call sites](Finding-Callers) and their enclosing symbol |
| `ambits -p <path> show <selector>…` | [Symbol definitions](Reading-by-Symbol) as JSON |
| `ambits -p <path> restore-context` | [This session's read history](Restoring-Context) |
| `ambits -p <path> --coverage` | [Coverage report](Coverage-and-Multi-Agent) |
| `ambits -p <path> --dump` | Print the symbol tree |
| `ambits -p <path> cache status\|clear` | Inspect or remove [read journals](Configuration#the-read-journal) |
| `ambits hook install` | Register the post-compaction hook |
| `ambits skill install` | Install the Claude Code skill |

## Global options

| Flag | Description |
|---|---|
| `-p`, `--project <PATH>` | Project root. Required by every command that reads the project |
| `-s`, `--session <ID>` | Session to track (auto-detects the latest) |
| `-a`, `--agent <ID>` | Filter coverage to one agent (prefix match) |
| `--log-dir <DIR>` | Claude Code log directory (auto-derived) |
| `--dump` | Print the symbol tree to stdout instead of launching the TUI |
| `--depth <N>` | With `--dump`, levels of children to show (default `0`; hidden children marked `+N`) |
| `--full` | With `--dump`, the whole tree regardless of `--depth` |
| `--coverage` | Print a coverage report instead of launching the TUI |
| `--format table\|json` | Output format for `--coverage` (default `table`) |
| `--filter <SUBPATH>` | Restrict to a project-relative subpath, by path component |
| `--filter-regex <REGEX>` | Restrict to paths matching a regex (unanchored); exclusive with `--filter` |
| `--serena` | Use Serena's LSP symbol cache instead of tree-sitter |
| `--tools-config <FILE>` | Use this [`tools.toml`](Configuration#toolstoml) |
| `--editor <TEMPLATE>` | [Editor](TUI#opening-a-symbol-in-your-editor) for `Enter` in the TUI |
| `--no-journal` | Disable the read journal |
| `--flush-interval-ms <MS>` | Journal write interval (default 5000) |
| `--log-output <DIR>` | Write a JSON-lines [debug log](Configuration#debug-logging) per session |

## `ambits rg`

`ambits -p . rg [OPTIONS] PATTERN [PATH…]` — every argument is a PATH when
`-e` or `-f` is given.

| Group | Flags |
|---|---|
| Patterns | `-e PATTERN` (repeatable), `-f FILE`, `-F`, `-i`, `-w`, `-x`, `-U` (multiline), `-v` |
| Files | `-g GLOB` (`!` excludes), `-t TYPE`, `--type-list`, `--hidden`, `--no-ignore` |
| Context | `-A N`, `-B N`, `-C N` |
| Output | `-n` (default) / `-N`, `--no-column`, `-o`, `--heading` / `--no-heading`, `--vimgrep`, `--no-symbol`, `--color auto\|always\|never`, `--json` |
| Summaries | `-l`, `--files-without-match`, `-c`, `--count-matches`, `-q` |
| Limits | `-m NUM` per file, `-M NUM` columns (default 300), `--head-limit NUM` total (default 200); `0` lifts either default |

## `ambits grep`

`ambits -p . grep [OPTIONS] PATTERN [PATH…]` — every argument is a PATH when
`-e` or `-f` is given. `--help` is long-only.

| Group | Flags |
|---|---|
| Patterns | `-e PATTERN`, `-f FILE`, `-F`, `-i`, `-w`, `-x`, `-v`; `-E` / `-G` accepted, no effect; `-P` refused |
| Files | `--include GLOB`, `--exclude GLOB`, `--exclude-dir DIR`, `--hidden`, `--no-ignore`; `-r` / `-R` accepted, no effect |
| Context | `-A N`, `-B N`, `-C N` |
| Output | `-n` (off by default), `--column`, `-H` (default) / `-h`, `-Z`, `-o`, `--no-symbol`, `--color` |
| Summaries | `-l`, `-L`, `-c`, `-q` |
| Limits | `-m NUM`, `--max-columns NUM` (default 300), `--head-limit NUM` (default 200) |

`-z` is refused. Both commands exit `0` on a match, `1` on none, `2` on error.

## `ambits callers`

`ambits -p . callers [--format text|json] NAME…`

## `ambits show`

`ambits -p . show [--no-body] [--max-bytes N] SELECTOR…` — a selector is a
symbol id (`<path>::<name-path>`) or a content hash (`b3:<hex>`, or at least 8
hex characters).

## `ambits restore-context`

`ambits -p . restore-context [--max-tokens N] [--format markdown|json|hook]` —
default budget 3000 tokens, default format `markdown`.

## `ambits cache`

| Command | Description |
|---|---|
| `cache status` | Journals on disk, with size and age |
| `cache clear --session <ID>` | Delete one session's journal |
| `cache clear --all` | Delete every journal for this project |

## `ambits hook install`

| Flag | Description |
|---|---|
| `-p`, `--project <DIR>` | Project to install for (default: current directory) |
| `-g`, `--global` | Install to `~/.claude/settings.json` instead |

## `ambits skill install`

| Flag | Description |
|---|---|
| `-p`, `--project <DIR>` | Install to this project's `.claude/skills/ambit/` |
| `-g`, `--global` | Install to `~/.claude/skills/ambit/` |

With neither, installs into the current project.
