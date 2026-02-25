# Spike: Configuration-Driven Tool Call Mappings

**Status**: Ready for implementation
**Branch**: `feat/config-driven-tool-mappings` (suggested)
**Spec revision**: 4
**Estimated scope**: ~6 new/modified files, ~25 new tests

---

## Background

ambit watches Claude Code session `.jsonl` logs and maps each agent tool call to a
`ReadDepth` value and an `AgentToolCall` event that drives coverage tracking. The
mapping lives entirely in `fn map_tool_call` in `src/ingest/claude.rs` — a 260-line
`match` over hard-coded tool name strings.

Every new MCP plugin, custom Claude tool, or alternative agent framework requires
editing that `match` and recompiling ambit. Users cannot extend coverage tracking
without forking the project.

---

## Goal

Extract the hard-coded mappings into a declarative TOML configuration file loaded at
startup. All existing built-in mappings become the shipped default config. Users can
extend or override mappings by placing a `tools.toml` in `.ambit/` or
`~/.config/ambit/` — no recompilation required.

---

## Current State

### Call chain

```
main.rs
  └─ LogTailer::new(files)           // no config today
       └─ read_new_events()
            └─ parse_jsonl_line(line, default_agent_id)
                 └─ map_tool_call(tool_name, input, agent_id, timestamp)
                      └─ hard-coded match → Option<AgentToolCall>

main.rs (dump/coverage mode)
  └─ parse_log_file(path)
       └─ parse_jsonl_line(line, default_agent_id)
            └─ map_tool_call(...)
```

### `map_tool_call` today (`src/ingest/claude.rs:313–577`)

A single `match tool_name { ... }` with ~13 arms. Each arm:
1. Extracts a `file_path` from `input` JSON (tool-specific key names)
2. Determines a `ReadDepth` (sometimes conditional on an input field, e.g. `include_body`)
3. Formats a `description` string
4. Optionally extracts `target_symbol` and `target_lines`
5. Returns `Some(AgentToolCall { ... })` or `None`

Unknown tool names fall through to `_ => return None`.

### `LogTailer` today (`src/ingest/claude.rs:592–678`)

```rust
pub struct LogTailer {
    files: Vec<PathBuf>,
    positions: HashMap<PathBuf, u64>,
}

impl LogTailer {
    pub fn new(files: Vec<PathBuf>) -> Self { ... }
    pub fn add_file(&mut self, path: PathBuf) { ... }
    pub fn read_new_events(&mut self) -> TailerOutput { ... }
}
```

No config reference today. `parse_jsonl_line` is called inline with no shared state.

### Existing tests (`src/ingest/claude.rs` `mod tests`)

13 `map_*_tool` tests, each constructing a JSON `Value` and asserting the returned
`AgentToolCall` fields. These tests must all continue to pass after the refactor (only
their call signature changes: a `config` argument is added).

---

## Proposed Design

### New files

| File | Purpose |
|---|---|
| `src/ingest/tool_config.rs` | All new types + `ToolMappingConfig` impl |
| `src/ingest/default_tools.toml` | Normative built-in TOML (13 stanzas) |
| `tests/integration_tool_config.rs` | End-to-end pipeline integration tests |

### Modified files

| File | Change |
|---|---|
| `src/ingest/mod.rs` | `pub mod tool_config;` |
| `src/ingest/claude.rs` | Refactor `map_tool_call`; `parse_jsonl_line` + `parse_log_file` gain `config` param; `LogTailer` gains `Arc<ToolMappingConfig>` field |
| `src/main.rs` | Call `ToolMappingConfig::resolve()`; display warnings; thread `Arc` |
| `Cargo.toml` | Add `toml = { version = "0.8", default-features = false, features = ["parse"] }` |

---

## Type Definitions (`src/ingest/tool_config.rs`)

```rust
#[derive(Debug, Clone, Deserialize)]
pub struct ToolMappingConfig {
    pub version: u32,
    #[serde(rename = "tool", default)]
    pub tools: Vec<ToolMapping>,
    #[serde(skip)]
    pub(crate) index: HashMap<String, usize>,  // built at load time; NOT in TOML
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolMapping {
    pub names: Vec<String>,
    pub path_keys: Vec<String>,
    #[serde(default)]
    pub pattern_keys: Vec<String>,
    pub depth: DepthSpec,
    pub description: String,
    #[serde(default)]
    pub target_symbol: Option<TargetSymbolSpec>,
    #[serde(default)]
    pub target_lines: Option<TargetLinesSpec>,
    /// Inherit unset fields from a named built-in stanza.
    #[serde(default)]
    pub extends: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DepthSpec {
    Fixed { value: ReadDepthDe },
    Conditional {
        condition_key: String,
        if_true: ReadDepthDe,
        if_false: ReadDepthDe,
        default: ReadDepthDe,
    },
}

/// Serde mirror of `ReadDepth`. Kept separate so `tracking::ReadDepth` stays
/// free of a `serde` dependency. If `ReadDepth` ever derives `Deserialize`,
/// delete this type and the `From` impl below.
#[derive(Debug, Clone, Deserialize)]
pub enum ReadDepthDe { NameOnly, Overview, Signature, FullBody }

impl From<ReadDepthDe> for ReadDepth { ... }

#[derive(Debug, Clone, Deserialize)]
pub struct TargetSymbolSpec { pub key: String }

#[derive(Debug, Clone, Deserialize)]
pub struct TargetLinesSpec { pub offset_key: String, pub limit_key: String }

#[derive(Debug, Clone)]
pub enum ConfigWarning {
    ParseError { path: String, message: String },
    UnsupportedVersion { path: String, found: u32, supported: u32 },
    EmptyNames { stanza_index: usize },
    DuplicateName { name: String, kept_index: usize, dropped_index: usize },
    ConditionalKeyNonBoolean { tool_name: String, key: String },
}
```

---

## Config File Discovery

Resolution order — first **loadable** file wins. "Loadable" = exists + parses without
error + `version <= SUPPORTED_VERSION`. A file that fails any condition emits a
`ConfigWarning` and the search continues.

1. `--tools-config <path>` CLI flag *(to be added to `Cli` struct)*
2. `.ambit/tools.toml` in the working directory
3. `~/.config/ambit/tools.toml`
4. Built-in defaults (always succeeds; enforced by CI test)

---

## `ToolMappingConfig` Methods

```rust
impl ToolMappingConfig {
    pub const SUPPORTED_VERSION: u32 = 1;

    /// Parse the embedded built-in config. Returns Err rather than panicking.
    /// CI test `builtin_config_parses` asserts this never returns Err on main.
    pub fn builtin() -> Result<Self, toml::de::Error>;

    /// Load a user config file. Returns (None, warnings) on any error.
    pub fn load(path: &Path) -> (Option<Self>, Vec<ConfigWarning>);

    /// Full resolution: discover + load + merge + wrap in Arc.
    /// Never panics. Never writes to stderr.
    pub fn resolve(cli_override: Option<&Path>) -> (Arc<Self>, Vec<ConfigWarning>);

    /// Merge semantics: user names shadow built-in names (any-element rule).
    /// User config is deduplicated by name first (last-wins), then merged.
    pub fn merge(base: Self, user: Self, warnings: &mut Vec<ConfigWarning>) -> Self;

    /// Build the HashMap<tool_name, stanza_index> dispatch index.
    /// Must be called after any mutation of `tools`.
    fn build_index(&mut self);
}
```

---

## Merge Semantics

> A user stanza whose `names` list shares **any** element with a built-in stanza
> **replaces** that built-in stanza entirely.

**Deduplication within user config** happens first (last stanza in file wins per
shared name), producing a canonical user config; that canonical config is then merged
against built-ins.

**Worked example:**

```
Built-ins:   [A: names=["Read","mcp__acp__Read"]]
             [B: names=["Edit"]]

User file:   [X: names=["Read","my_read"]]   ← line 1
             [Y: names=["Read"]]              ← line 2

After user dedup:
  Y wins for "Read" (last occurrence)
  X survives only for "my_read"

After merge vs built-ins:
  A dropped (shares "Read" with Y)
  Y inserted (covers "Read")
  X inserted as entry for "my_read"
  B preserved (no overlap)

Final: [Y:"Read"], [X:"my_read"], [B:"Edit"]
```

### `extends` — field-level inheritance

A user stanza may add `extends = "<built-in-name>"` to inherit any fields it does not
explicitly set from the named built-in stanza. Prevents silent loss of `target_symbol`
or `target_lines` when copying a template.

```toml
[[tool]]
names       = ["find_symbol", "my_custom_find"]
extends     = "mcp__serena__find_symbol"
description = "FindSym {name_path_pattern} (custom)"
# depth, target_symbol, path_keys inherited from built-in
```

---

## TOML Schema

### `depth` field

```toml
# Fixed:
depth = { type = "fixed", value = "FullBody" }
# values: "NameOnly" | "Overview" | "Signature" | "FullBody"

# Conditional (boolean key in input drives choice):
depth = { type = "conditional", condition_key = "include_body",
          if_true = "FullBody", if_false = "Signature", default = "Signature" }
# `default` used when key absent, null, or non-boolean (emits ConfigWarning)
```

### `description` — generic `{key}` templates

`{key}` is replaced with `input[key].as_str()`. Unknown key → `"?"`.

Optional `|short` modifier applies `short_path()` (last 2 path components):

```toml
description = "Read {file_path|short}"
description = "Symbol {name_path_pattern}"
description = "Search \"{pattern}\""
```

Valid placeholder syntax: `{[a-zA-Z_][a-zA-Z0-9_]*(|short)?}`.
Malformed `{...}` expressions are treated as literal text.

### `target_symbol` and `target_lines`

```toml
target_symbol = { key = "name_path_pattern" }
target_lines  = { offset_key = "offset", limit_key = "limit" }
```

### Version semantics

| `version` | Behaviour |
|---|---|
| `== 1` (SUPPORTED) | Load normally |
| `> 1` | `ConfigWarning::UnsupportedVersion`, skip user config, use built-ins |
| Missing | Deserialisation error → `ConfigWarning::ParseError` |

Adding new `#[serde(default)]` optional fields is **non-breaking** and does not
increment `version`. Renaming or removing a required field is breaking and **must**
increment `version`.

---

## Normative Default Config (`src/ingest/default_tools.toml`)

```toml
version = 1

[[tool]]
names        = ["Read", "mcp__acp__Read", "mcp__plugin_serena_serena__read_file"]
path_keys    = ["file_path", "relative_path"]
pattern_keys = []
depth        = { type = "fixed", value = "FullBody" }
description  = "Read {file_path|short}"
target_lines = { offset_key = "offset", limit_key = "limit" }

[[tool]]
names        = ["Edit", "mcp__acp__Edit", "mcp__plugin_serena_serena__replace_content"]
path_keys    = ["file_path", "relative_path"]
pattern_keys = []
depth        = { type = "fixed", value = "FullBody" }
description  = "Edit {file_path|short}"

[[tool]]
names        = ["Write", "mcp__acp__Write", "mcp__plugin_serena_serena__create_text_file"]
path_keys    = ["file_path", "relative_path"]
pattern_keys = []
depth        = { type = "fixed", value = "FullBody" }
description  = "Write {file_path|short}"

[[tool]]
names        = ["Glob", "mcp__serena__find_file", "mcp__serena__list_dir",
                "mcp__plugin_serena_serena__find_file", "mcp__plugin_serena_serena__list_dir"]
path_keys    = ["path", "relative_path"]
pattern_keys = ["pattern", "file_mask"]
depth        = { type = "fixed", value = "NameOnly" }
description  = "Glob {pattern}"

[[tool]]
names        = ["Grep", "mcp__serena__search_for_pattern",
                "mcp__plugin_serena_serena__search_for_pattern"]
path_keys    = ["path", "relative_path"]
pattern_keys = ["pattern", "substring_pattern"]
depth        = { type = "fixed", value = "Overview" }
description  = "Search \"{pattern}\""

[[tool]]
names        = ["mcp__serena__get_symbols_overview",
                "mcp__plugin_serena_serena__get_symbols_overview"]
path_keys    = ["relative_path"]
pattern_keys = []
depth        = { type = "fixed", value = "Overview" }
description  = "Overview {relative_path|short}"

[[tool]]
names        = ["mcp__serena__find_symbol", "mcp__plugin_serena_serena__find_symbol"]
path_keys    = ["relative_path"]
pattern_keys = []
depth        = { type = "conditional", condition_key = "include_body",
                 if_true = "FullBody", if_false = "Signature", default = "Signature" }
description  = "Symbol {name_path_pattern}"
target_symbol = { key = "name_path_pattern" }

[[tool]]
names        = ["mcp__serena__find_referencing_symbols",
                "mcp__plugin_serena_serena__find_referencing_symbols"]
path_keys    = ["relative_path"]
pattern_keys = []
depth        = { type = "fixed", value = "Overview" }
description  = "FindRefs {name_path}"
target_symbol = { key = "name_path" }

[[tool]]
names        = ["mcp__serena__replace_symbol_body",
                "mcp__plugin_serena_serena__replace_symbol_body"]
path_keys    = ["relative_path"]
pattern_keys = []
depth        = { type = "fixed", value = "FullBody" }
description  = "ReplaceSymbol {name_path}"
target_symbol = { key = "name_path" }

[[tool]]
names        = ["mcp__serena__insert_after_symbol",
                "mcp__plugin_serena_serena__insert_after_symbol"]
path_keys    = ["relative_path"]
pattern_keys = []
depth        = { type = "fixed", value = "FullBody" }
description  = "InsertAfter {name_path}"
target_symbol = { key = "name_path" }

[[tool]]
names        = ["mcp__serena__insert_before_symbol",
                "mcp__plugin_serena_serena__insert_before_symbol"]
path_keys    = ["relative_path"]
pattern_keys = []
depth        = { type = "fixed", value = "FullBody" }
description  = "InsertBefore {name_path}"
target_symbol = { key = "name_path" }

[[tool]]
names        = ["mcp__serena__rename_symbol", "mcp__plugin_serena_serena__rename_symbol"]
path_keys    = ["relative_path"]
pattern_keys = []
depth        = { type = "fixed", value = "FullBody" }
description  = "Rename {name_path}"
target_symbol = { key = "name_path" }

[[tool]]
names        = ["NotebookEdit"]
path_keys    = ["notebook_path"]
pattern_keys = []
depth        = { type = "fixed", value = "FullBody" }
description  = "NotebookEdit {notebook_path|short}"
```

---

## Refactored `map_tool_call`

### Signature change

```rust
// Before
fn map_tool_call(tool_name: &str, input: &Value, agent_id: &str, timestamp_str: &str)
    -> Option<AgentToolCall>

// After
pub fn map_tool_call(
    config: &ToolMappingConfig,
    tool_name: &str,
    input: &Value,
    agent_id: &str,
    timestamp_str: &str,
) -> Option<AgentToolCall>
```

### Dispatch algorithm

```
1. config.index.get(tool_name) → stanza index i   [O(1)]
   If absent → return None

2. mapping = &config.tools[i]

3. file_path:
   iterate mapping.path_keys; use first key present in input as str
   if path_keys non-empty but no key found → return None

4. pattern:
   iterate mapping.pattern_keys; first hit or "?"

5. read_depth from DepthSpec:
   Fixed { value }          → ReadDepthDe::into()
   Conditional { ... }:
     key absent / null / non-bool → use default (+ warn if non-bool)
     key = true/false             → if_true / if_false

6. description:
   for each {key} or {key|short} in template:
     resolve input[key].as_str(), apply short_path if |short
     fallback "?" if key absent or non-string

7. target_symbol: if TargetSymbolSpec present, input[spec.key].as_str()

8. target_lines: if TargetLinesSpec present, both keys as u64 or None

9. return Some(AgentToolCall { ... })
```

### `LogTailer` change

```rust
pub struct LogTailer {
    files: Vec<PathBuf>,
    positions: HashMap<PathBuf, u64>,
    config: Arc<ToolMappingConfig>,   // ← added
}

impl LogTailer {
    pub fn new(files: Vec<PathBuf>, config: Arc<ToolMappingConfig>) -> Self { ... }
    // read_new_events passes &*self.config to parse_jsonl_line
}
```

### `parse_log_file` change

```rust
pub fn parse_log_file(path: &Path, config: &ToolMappingConfig) -> Vec<AgentToolCall>
```

### `parse_jsonl_line` change

```rust
pub fn parse_jsonl_line(
    line: &str,
    default_agent_id: &str,
    config: &ToolMappingConfig,   // ← added
) -> ParsedLine
```

---

## Propagation Through `main.rs`

```rust
// At startup, before TUI or dump/coverage modes:
let (tool_config, config_warnings) = ToolMappingConfig::resolve(cli.tools_config.as_deref());

// Display warnings in TUI status bar or to stdout (non-TUI modes) before event loop.
// Never write to stderr — would corrupt ratatui alternate screen.

// Thread Arc into LogTailer:
let tailer = LogTailer::new(log_files, Arc::clone(&tool_config));

// Thread ref into parse_log_file (dump/coverage modes):
let events = parse_log_file(&path, &tool_config);
```

**Session reset**: when `LogTailer` is rebuilt on `/clear`, clone the existing
`Arc<ToolMappingConfig>` — do **not** call `resolve()` again. Config changes require
application restart.

---

## Validation Behaviour

| Condition | Behaviour |
|---|---|
| `version > SUPPORTED_VERSION` | `ConfigWarning::UnsupportedVersion`, skip user config |
| `version` field absent | Deserialisation `Err` → `ConfigWarning::ParseError`, skip |
| Malformed TOML | Deserialisation `Err` → `ConfigWarning::ParseError`, skip |
| Stanza missing required field | Deserialisation `Err` for whole file → same as malformed |
| `names` array empty | `ConfigWarning::EmptyNames`, skip that stanza, continue |
| Duplicate name within user config | `ConfigWarning::DuplicateName`, last wins |
| `condition_key` value is non-boolean | `ConfigWarning::ConditionalKeyNonBoolean`, use `default` |
| Template placeholder unresolvable | Substitute `"?"` silently |
| `extends` names unknown built-in | Skip `extends` silently, treat stanza as standalone |

---

## Testing Plan

### Preserved tests (`src/ingest/claude.rs` `mod tests`)

All 13 existing `map_*_tool` tests continue to pass. Only change: each constructs
`ToolMappingConfig::builtin().unwrap()` and passes it as the first argument to
`map_tool_call`.

### New unit tests (`src/ingest/tool_config.rs` `mod tests`)

| Test | What it proves |
|---|---|
| `builtin_config_parses` | `builtin()` returns `Ok`; `tools.len() == 13` |
| `builtin_covers_all_tool_names` | All historical match-arm names present in `index` |
| `builtin_index_correct_length` | `index.len()` == sum of all stanza `names` lengths |
| `merge_user_replaces_builtin` | Overlapping name → built-in stanza dropped |
| `merge_user_extends_builtin` | Novel name appended; built-ins untouched |
| `merge_no_overlap` | All 13 built-in stanzas survive |
| `merge_dedup_user_first` | Two user stanzas sharing a name → last wins; earlier's other names survive |
| `merge_extends_inherits_target_symbol` | User stanza with `extends`, no `target_symbol` → inherits built-in's |
| `conditional_depth_key_missing` | Key absent → `default` depth |
| `conditional_depth_key_null` | Key = JSON null → `default` depth |
| `conditional_depth_key_non_boolean` | Key = `"yes"` string → `default` + `ConfigWarning` |
| `conditional_depth_with_body` | `true` → `if_true` |
| `conditional_depth_without_body` | `false` → `if_false` |
| `pattern_keys_fallback` | `Glob` with `file_mask` (not `pattern`) → pattern extracted |
| `empty_path_keys` | `path_keys = []` → `file_path = None` |
| `missing_required_path` | `path_keys = ["file_path"]`, key absent → `None` |
| `unknown_tool_name` | Returns `None` |
| `unknown_placeholder_substitutes_question_mark` | `{typo_key}` absent → `"?"` in description |
| `short_modifier_applies_short_path` | `{file_path\|short}` → last 2 components |
| `user_config_bad_version` | `load()` → `(None, [UnsupportedVersion])` |
| `user_config_malformed_toml` | `load()` → `(None, [ParseError])` |
| `resolve_falls_back_to_builtin` | Bad user config → 13 built-in tools in Arc |
| `dispatch_index_matches_linear_scan` | Index lookup returns same stanza as linear scan for all 13 names |

### New dispatch tests (`src/ingest/claude.rs` `mod tests`)

| Test | What it proves |
|---|---|
| `merged_config_dispatches_user_tool` | Novel user tool name → user stanza behavior |
| `merged_config_dispatches_replaced_builtin` | Shadowed `"Read"` → user behavior, not built-in |

### Integration tests (`tests/integration_tool_config.rs`)

`parse_jsonl_line_with_builtin_config_integration` — for each of the 13 stanzas,
supply a representative raw JSON line (as it appears in a real Claude Code `.jsonl`
file), call `parse_jsonl_line` with `ToolMappingConfig::builtin().unwrap()`, and assert
all `AgentToolCall` fields (`file_path`, `read_depth`, `description`, `target_symbol`,
`target_lines`). This test simultaneously validates the TOML, deserialization, index
dispatch, template resolution, and `parse_jsonl_line` wiring.

---

## Dependency

```toml
# Cargo.toml — add to [dependencies]
toml = { version = "0.8", default-features = false, features = ["parse"] }
```

No other new dependencies. `serde` with `derive` feature is already present.

---

## Non-Goals for v1

- Hot-reload of config during a running session
- TUI editor for tool mappings
- Mappings for non-tool-call events (session clear, compact)
- Chained `extends` (one level only)
- Generalised `extractions` map for future `AgentToolCall` fields (tracked as a v2
  consideration)

---

## Implementation Order

Suggested sequencing to keep the build green at each step:

1. **Add `toml` dep** to `Cargo.toml`. Build passes.
2. **Create `src/ingest/default_tools.toml`** with the 13 normative stanzas.
3. **Create `src/ingest/tool_config.rs`** with all types and a stub `ToolMappingConfig::builtin()`. Add `pub mod tool_config;` to `src/ingest/mod.rs`. Build passes.
4. **Implement `ToolMappingConfig::builtin()`, `build_index()`, `load()`, `merge()`, `resolve()`**. Add `builtin_config_parses` test — must pass.
5. **Refactor `map_tool_call`** in `claude.rs` to accept `config: &ToolMappingConfig` and dispatch via index. Update `parse_jsonl_line` and `parse_log_file` signatures. Update all 13 existing tests to pass `config`. All existing tests must pass.
6. **Add `Arc<ToolMappingConfig>` to `LogTailer`**. Update `LogTailer::new` signature.
7. **Update `main.rs`**: add `--tools-config` CLI arg, call `resolve()`, display warnings, thread `Arc` to `LogTailer` and `parse_log_file`. Build passes end-to-end.
8. **Write remaining unit tests** in `tool_config.rs` `mod tests`.
9. **Write dispatch tests** in `claude.rs` `mod tests`.
10. **Write integration tests** in `tests/integration_tool_config.rs`.
11. **Run `cargo test --all`** — all tests green.
12. **Manual smoke test**: run ambit against a real Claude Code session; verify coverage display is identical to pre-refactor behaviour.
