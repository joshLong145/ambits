# marker

Tool for visualizing parts of your codebase an Claude has stored within a session log.

![screenshot](./images/screenshot.png)

## What it does

- Parses your project into a hierarchical symbol tree (functions, structs, classes, methods, etc.)
- Monitors Claude Code's JSONL session logs in real time
- Colors each symbol by how deeply the agent has read it: unseen, name-only, overview, signature, or full body
- Detects when source files change and marks previously-read symbols as stale
- supports parsing [Serena MCP](https://github.com/oraios/serena) symbol artifacts.

## Todo
- multi agent heirchies
- 
## Supported languages

**Tree-sitter parsing:**
- Rust
- Python

## Build

Requires Rust 1.70+.

```
cargo build --release
```

## Usage

```
marker --project <path>
```

### Flags

| Flag | Description |
|---|---|
| `--project`, `-p` | Path to the project root (required) |
| `--dump` | Print symbol tree to stdout and exit |
| `--serena` | Use Serena's LSP symbol cache instead of tree-sitter |
| `--session`, `-s` | Session ID to track (auto-detects latest) |
| `--log-dir` | Path to Claude Code log directory (auto-derived) |

### Examples

```
# Launch TUI for current project
marker -p .

# Dump symbol tree without TUI
marker -p . --dump

# Use Serena's symbol cache (more languages, finer detail)
marker -p . --serena
```

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
