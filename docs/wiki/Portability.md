# Portability

Nothing in the record is tied to a vendor or a machine:

| Piece | Form |
|---|---|
| Symbol ids | `<project-relative-path>::<name-path>` |
| Content hashes | BLAKE3 over whitespace-normalized source |
| Digest (`restore-context`) | Markdown, or schema-versioned JSON |
| `show` / `callers --format json` output | Schema-versioned JSON |
| `rg --json` output | ripgrep's JSON Lines events plus a `symbol` field |
| Journal | NDJSON, one record per read |

The same digest means the same thing on another checkout, another machine, or
in front of another model. Any agent that can run a command and read text can
consume it — no MCP server, no SDK, no wire protocol.

## What *is* provider-shaped

- **Session ingestion** reads Claude Code's JSONL format today.
  `SessionIngester` (`src/ingest/mod.rs`) is the extension point — "implement
  this to add support for a new LLM session format."
- **Tool mappings** are data, not code. Another provider's tool names are
  taught in `.ambits/tools.toml` rather than patched in (see
  [Configuration](Configuration#tool-mappings)); `ToolCallMapper` exists to
  "plug in alternative tool-name conventions."
- **`restore-context --format hook`** emits Claude Code's `SessionStart`
  envelope specifically. `--format markdown` and `--format json` carry the same
  content with no envelope.

Claude Code is what this is built and tested against. The formats are
deliberately boring so that need not stay true.
