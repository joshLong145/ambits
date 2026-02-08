# ambits

Tool for visualizing parts of your codebase an LLM agent has stored within a session log.

![screenshot](./images/screenshot.png)

## What it does
- Monitors Claude Code's JSONL session logs in real time
- Colors each symbol by how deeply the agent has read it: unseen, name-only, overview, signature, or full body
- Detects when source files change and marks previously-read symbols as stale
- Supports parsing [Serena MCP](https://github.com/oraios/serena) symbol artifacts.
- Supports Tree sitter parsing for
  - Rust
  - Python
- Symbol dumps and coverage reports

## Todo
- Multi agent hierarchies
- Multi session visualization

## Supported languages

**Tree-sitter parsing**
- Rust
- Python

## Building from source

Requires Rust 1.70+.

```
cargo build --release
```

## Installing through Cargo

```
cargo install ambits
```

## Usage

```
ambits --project <path>
```

### Flags

| Flag | Description |
|---|---|
| `--project`, `-p` | Path to the project root (required) |
| `--session`, `-s` | Session ID to track (auto-detects latest) |
| `--dump` | Print symbol tree to stdout and exit |
| `--coverage` | Print coverage report to stdout and exit |
| `--serena` | Use Serena's LSP symbol cache instead of tree-sitter |
| `--log-dir` | Path to Claude Code log directory (auto-derived) |
| `--log-output` | Output directory for event logs |

### Examples

```bash
# Launch TUI for current project
ambits -p .

# Dump symbol tree without TUI
ambits -p . --dump

# Print coverage report
ambits -p . --coverage

# Use Serena's symbol cache (more languages, finer detail)
ambits -p . --serena
```

### Coverage Report

The `--coverage` flag outputs a tabular report showing per-file symbol visibility:

```
Coverage Report (session: 34e212cf-a176-4059-ba12-eca94b56e43b)
─────────────────────────────────────────────────────────────────────────────
File                                      Symbols    Seen    Full   Seen%   Full%
─────────────────────────────────────────────────────────────────────────────
src/events.rs                                   3       0       0      0%      0%
src/parser/mod.rs                               8       8       1    100%     12%
src/app.rs                                     23      23      23    100%    100%
─────────────────────────────────────────────────────────────────────────────
TOTAL                                         214     182     175     85%     82%
```

- **Seen%**: Symbols the agent has any awareness of (name, overview, signature, or full body)
- **Full%**: Symbols the agent has read completely (full body)

### Keybindings

| Key | Action |
|---|---|
| `j` / `k` | Navigate up/down |
| `h` / `l` | Collapse/expand |
| `Enter` | Toggle expand |
| `/` | Search symbols |
| `a` | Cycle agent filter |
| `Tab` | Switch panel focus |
| `q` | Quit |

### Color legend

| Color | Meaning |
|---|---|
| Dark gray | Unseen |
| Light gray | Name only (appeared in glob/listing) |
| Pale blue | Overview (grep match, symbol listing) |
| Blue | Signature seen |
| Green | Full body read |
| Orange | Stale (source changed since last read) |
