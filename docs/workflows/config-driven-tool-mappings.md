# Workflow: Configuration-Driven Tool Call Mappings

**Source spike**: `docs/spikes/config-driven-tool-mappings.md`
**Suggested branch**: `feat/config-driven-tool-mappings`
**Execute with**: `/sc:implement` — one phase at a time

## Tooling conventions for this workflow

- **Code exploration**: use Serena MCP tools (`get_symbols_overview`, `find_symbol`,
  `find_referencing_symbols`, `search_for_pattern`) rather than raw `cat`/`grep`.
- **Coverage checkpoints**: run `ambits --project . --coverage` (or `--dump`) after
  each gate to confirm coverage of affected files increases as expected.
- **Build gate**: `cargo build` must pass at the end of every phase before proceeding.
- **Test gate**: stated `cargo test` invocations must pass before moving to the next phase.

---

## Phase overview

| # | Phase | Key files | Gate |
|---|---|---|---|
| 1 | Dependency | `Cargo.toml` | `cargo build` |
| 2 | Default config asset | `src/ingest/default_tools.toml` | file parses as TOML |
| 3 | Type scaffolding | `src/ingest/tool_config.rs`, `src/ingest/mod.rs` | `cargo build` |
| 4 | Core logic | `src/ingest/tool_config.rs` | `builtin_config_parses` passes |
| 5 | Refactor dispatch | `src/ingest/claude.rs` | all 13 existing tests pass |
| 6 | `LogTailer` wiring | `src/ingest/claude.rs` | `cargo build` |
| 7 | `main.rs` integration | `src/main.rs` | `cargo run -- --project . --dump` works |
| 8 | Unit tests | `src/ingest/tool_config.rs` | 23 new tests pass |
| 9 | Dispatch tests | `src/ingest/claude.rs` | 2 new tests pass |
| 10 | Integration tests | `tests/integration_tool_config.rs` | integration test passes |
| 11 | Full suite | — | `cargo test --all` green |
| 12 | Smoke test | — | manual + ambit self-coverage check |

**Hard invariant**: `cargo build` must succeed at the end of every phase.
Never leave the build broken between phases.

---

## Phase 1 — Dependency

**Goal**: Add the `toml` crate. Nothing else changes.

### Before starting

Use Serena `get_symbols_overview` on `Cargo.toml` is not applicable (non-Rust file).
Instead verify the current `[dependencies]` section does not already contain `toml`.

### Tasks

- [ ] **1.1** In `Cargo.toml` under `[dependencies]`, add:
  ```toml
  toml = { version = "0.8", default-features = false, features = ["parse"] }
  ```

### Gate

```bash
cargo build
```

---

## Phase 2 — Default Config Asset

**Goal**: Create `src/ingest/default_tools.toml` — the normative built-in config
embedded at compile time. This is the authoritative source of truth for all 13 built-in
tool stanzas.

### Before starting

Use Serena `search_for_pattern` on `src/ingest/claude.rs` with pattern
`"mcp__acp__Read|mcp__serena__find_symbol|NotebookEdit"` to confirm the full set of
tool name strings currently in the hard-coded `match` before writing the TOML.

### Tasks

- [ ] **2.1** Create `src/ingest/default_tools.toml`:

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
names         = ["Glob", "mcp__serena__find_file", "mcp__serena__list_dir",
                 "mcp__plugin_serena_serena__find_file", "mcp__plugin_serena_serena__list_dir"]
path_keys     = ["path", "relative_path"]
pattern_keys  = ["pattern", "file_mask"]
path_required = false
depth         = { type = "fixed", value = "NameOnly" }
description   = "Glob {pattern}"
# Note: existing code falls back to "*" when pattern absent; config-driven fallback is "?".
# This is a minor cosmetic change; file_path is still None when path absent.

[[tool]]
names         = ["Grep", "mcp__serena__search_for_pattern",
                 "mcp__plugin_serena_serena__search_for_pattern"]
path_keys     = ["path", "relative_path"]
pattern_keys  = ["pattern", "substring_pattern"]
path_required = false
depth         = { type = "fixed", value = "Overview" }
description   = "Search \"{pattern}\""

[[tool]]
names         = ["mcp__serena__get_symbols_overview",
                 "mcp__plugin_serena_serena__get_symbols_overview"]
path_keys     = ["relative_path"]
pattern_keys  = []
path_required = false
depth         = { type = "fixed", value = "Overview" }
description   = "Overview {relative_path|short}"

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

### Gate

No build gate yet — the file is only used via `include_str!` in Phase 3/4.
Verify it is syntactically valid TOML with any available TOML linter, or defer
validation to the `builtin_config_parses` test in Phase 4.

---

## Phase 3 — Type Scaffolding

**Goal**: Define all new types in `src/ingest/tool_config.rs` and register the module.
Method bodies are stubs (`todo!()`). Build must pass.

### Before starting

Use Serena `get_symbols_overview` on `src/ingest/mod.rs` to confirm current module
declarations. Use `get_symbols_overview` on `src/tracking/mod.rs` to confirm
`ReadDepth` variants before writing `ReadDepthDe`.

### Tasks

- [ ] **3.1** In `src/ingest/mod.rs`, add the line:
  ```rust
  pub mod tool_config;
  ```

- [ ] **3.2** Create `src/ingest/tool_config.rs` with all type definitions and stub impls:

  **Imports** (top of file):
  ```rust
  use std::collections::HashMap;
  use std::path::{Path, PathBuf};
  use std::sync::Arc;
  use serde::Deserialize;
  use crate::tracking::ReadDepth;
  ```

  **Types** — write all structs/enums in full (not stubs):
  - `ToolMappingConfig` — `version: u32`, `#[serde(rename="tool", default)] tools: Vec<ToolMapping>`, `#[serde(skip)] pub(crate) index: HashMap<String, usize>`
  - `ToolMapping` — `names`, `path_keys`, `#[serde(default)] pattern_keys`, `#[serde(default)] depth: Option<DepthSpec>`, `description`, `#[serde(default = "default_path_required")] path_required: bool`, `#[serde(default)] target_symbol`, `#[serde(default)] target_lines`, `#[serde(default)] extends`
    > **Technical note**: `path_required` defaults to `true` (via `fn default_path_required() -> bool { true }`).
    > Set `path_required = false` for tools like Glob and Grep where the path is an optional filter, not a
    > mandatory target. This preserves the existing behavioral contract: `Read`/`Edit`/`Write` return `None`
    > from `map_tool_call` if no path is found; `Glob`/`Grep`/`Overview` proceed with `file_path = None`.
    > **Technical note**: `depth` is `Option<DepthSpec>` (not bare `DepthSpec`) because the `extends`
    > inheritance logic must distinguish "user explicitly set `Fixed(NameOnly)`" from "user omitted the field".
    > Using a bare `DepthSpec` with any sentinel value (e.g. `Fixed(NameOnly)`) is incorrect — a user
    > legitimately wanting `NameOnly` depth would have it silently overwritten by the base stanza.
    > With `Option<DepthSpec>`, `None` means "inherit from base"; a standalone stanza (no `extends`) with
    > `depth = None` is a config error and should emit `ConfigWarning::MissingDepth { stanza_index }`.
    > In `map_tool_call`, resolve depth via `mapping.depth.as_ref().expect("depth must be Some after merge/load")`
    > — the `expect` is unreachable for correctly loaded configs.
  - `DepthSpec` — `#[serde(tag="type", rename_all="snake_case")]` enum with `Fixed { value: ReadDepthDe }` and `Conditional { condition_key, if_true, if_false, default }` variants
  - `ReadDepthDe` — enum with `NameOnly`, `Overview`, `Signature`, `FullBody`; `impl From<ReadDepthDe> for ReadDepth`
  - `TargetSymbolSpec { pub key: String }`
  - `TargetLinesSpec { pub offset_key: String, pub limit_key: String }`
  - `ConfigWarning` — enum with `ParseError { path, message }`, `UnsupportedVersion { path, found, supported }`, `EmptyNames { stanza_index }`, `DuplicateName { name, kept_index, dropped_index }`, `ConditionalKeyNonBoolean { tool_name, key }`, `MissingDepth { stanza_index }`

  **Method stubs** — compile but `todo!()`:
  ```rust
  impl ToolMappingConfig {
      pub const SUPPORTED_VERSION: u32 = 1;
      pub fn builtin() -> Result<Self, toml::de::Error> { todo!() }
      pub fn load(_path: &Path) -> (Option<Self>, Vec<ConfigWarning>) { todo!() }
      pub fn resolve(_cli: Option<&Path>) -> (Arc<Self>, Vec<ConfigWarning>) { todo!() }
      pub fn merge(_base: Self, _user: Self, _w: &mut Vec<ConfigWarning>) -> Self { todo!() }
      fn build_index(&mut self) { todo!() }
      fn find_user_config(_cli: Option<&Path>) -> Option<PathBuf> { todo!() }
      fn empty() -> Self { todo!() }
  }
  ```

### Gate

```bash
cargo build
```

`todo!()` warnings are acceptable. No errors.

### Coverage checkpoint

```bash
ambits --project . --coverage
```

`src/ingest/tool_config.rs` should now appear as a new file in the coverage output
(unseen, since no agent has read it yet in this session — that's fine; it confirms
the file was picked up by the parser).

---

## Phase 4 — Core Logic

**Goal**: Implement all `ToolMappingConfig` methods. The anchor test
`builtin_config_parses` must pass by the end of this phase.

### Before starting

Use Serena `find_symbol` with `include_body=true` on:
- `ToolMappingConfig/builtin` (stub) to confirm location before replacing
- `ToolMappingConfig/merge` (stub) to confirm location

### Tasks

- [ ] **4.1** Implement `fn empty() -> Self`:
  ```rust
  fn empty() -> Self {
      Self { version: 0, tools: vec![], index: HashMap::new() }
  }
  ```

- [ ] **4.2** Implement `fn build_index(&mut self)`:
  - Compute `total_names = self.tools.iter().map(|m| m.names.len()).sum::<usize>()`
  - `self.index = HashMap::with_capacity(total_names)` — pre-sizes the map to avoid incremental rehashing; typical built-in count is ~37 names across 13 stanzas
  - `for (i, mapping) in self.tools.iter().enumerate()` → for each name in `mapping.names` → `self.index.insert(name.clone(), i)`

- [ ] **4.3** Implement `pub fn builtin() -> Result<Self, toml::de::Error>`:
  ```rust
  let mut cfg: Self = toml::from_str(include_str!("default_tools.toml"))?;
  cfg.build_index();
  Ok(cfg)
  ```

- [ ] **4.4** Implement `pub fn load(path: &Path) -> (Option<Self>, Vec<ConfigWarning>)`:
  - `std::fs::read_to_string(path)` — on `Err` emit `ConfigWarning::ParseError`, return `(None, warnings)`
  - `toml::from_str(&content)` — on `Err` emit `ConfigWarning::ParseError`, return `(None, warnings)`
  - If `cfg.version > Self::SUPPORTED_VERSION` emit `ConfigWarning::UnsupportedVersion`, return `(None, warnings)`
  - Call `cfg.build_index()`, return `(Some(cfg), vec![])`

- [ ] **4.5** Implement `fn find_user_config(cli: Option<&Path>) -> Option<PathBuf>`:
  - If `cli.is_some()` and that path `exists()` → return it
  - Check `std::env::current_dir().ok()?.join(".ambit/tools.toml")` — return if exists
  - For the user-global path, use `std::env::home_dir()` (deprecated but still functional on all platforms) or, preferably, the `dirs` crate's `dirs::config_dir()`:
    > **Technical note**: `std::env::var("HOME")` works on Unix but silently returns `Err` on Windows
    > where the home directory is `USERPROFILE` or `FOLDERID_RoamingAppData`. `dirs::config_dir()`
    > returns `~/.config` on Linux/macOS and `AppData\Roaming` on Windows, which is the correct
    > platform-native config location. The `dirs` crate is a zero-dependency crate commonly used
    > in Rust CLI tools; check if it is already a transitive dependency before adding it explicitly.
    > If not available, `std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE"))` is an
    > acceptable fallback that covers both Unix and Windows without a new crate.
  - Return `None`

- [ ] **4.6** Implement `pub fn merge(base: Self, user: Self, warnings: &mut Vec<ConfigWarning>) -> Self`:

  **Step A — deduplicate within user config (last-wins per name)**:
  - Walk `user.tools` in order; build `HashMap<name, last_stanza_index>`
  - For stanzas displaced (earlier occurrence of a shared name): emit `ConfigWarning::DuplicateName`
  - Produce `canonical_user: Vec<ToolMapping>` — only winning stanzas in file order

  **Step B — validate: skip stanzas with empty `names`**:
  - For each stanza in `canonical_user` with `names.is_empty()`: emit `ConfigWarning::EmptyNames { stanza_index }`, skip

  **Step C — apply `extends` field-level inheritance**:
  - For each user stanza with `extends = Some(ref name)`:
    - Find the built-in stanza (from `base.tools`) whose `names` contains `name`
    - If found: for each field on the user stanza that is `None`/empty, substitute the built-in value
    - Fields eligible for inheritance (only if not explicitly set by user):
      - `path_keys`: if `user.path_keys.is_empty()`, inherit `base.path_keys.clone()`
      - `pattern_keys`: if `user.pattern_keys.is_empty()`, inherit `base.pattern_keys.clone()`
      - `depth`: **`depth` is `Option<DepthSpec>`; if `user.depth.is_none()`, inherit `Some(base.depth.clone().unwrap())`**
      - `target_symbol`: if `user.target_symbol.is_none()`, inherit `base.target_symbol.clone()`
      - `target_lines`: if `user.target_lines.is_none()`, inherit `base.target_lines.clone()`
    - Fields never inherited: `names`, `description`, `extends`
    - If built-in name not found: skip `extends` silently (no warning — user may be extending a future built-in)
  - For stanzas without `extends`, validate that `depth.is_some()`; if `None`, emit
    `ConfigWarning::MissingDepth { stanza_index }` and skip the stanza

  **Step D — merge canonical user against base**:
  - Collect `user_name_set: HashSet<&str>` from all canonical user stanza `names`
  - Filter `base.tools`: keep only stanzas with no name in `user_name_set` → `filtered_base: Vec<ToolMapping>`
  - Allocate result with `Vec::with_capacity(filtered_base.len() + canonical_user.len())` before extending — avoids a reallocation when appending user entries
  - Result: `filtered_base` + `canonical_user` (base first, user appended)

  **Step E** — build `ToolMappingConfig { version: base.version, tools: result, index: HashMap::new() }`, call `build_index()`, return

- [ ] **4.7** Implement `pub fn resolve(cli: Option<&Path>) -> (Arc<Self>, Vec<ConfigWarning>)`:
  ```rust
  let mut warnings = Vec::new();
  let builtin = match Self::builtin() {
      Ok(b) => b,
      Err(e) => {
          warnings.push(ConfigWarning::ParseError {
              path: "<built-in>".into(),
              message: e.to_string(),
          });
          return (Arc::new(Self::empty()), warnings);
      }
  };
  let (user, mut uw) = Self::find_user_config(cli)
      .map(|p| Self::load(&p))
      .unwrap_or((None, vec![]));
  warnings.append(&mut uw);
  let merged = match user {
      Some(u) => Self::merge(builtin, u, &mut warnings),
      None    => builtin,
  };
  (Arc::new(merged), warnings)
  ```

- [ ] **4.8** Add the CI anchor test at the bottom of `src/ingest/tool_config.rs`:
  ```rust
  #[cfg(test)]
  mod tests {
      use super::*;

      #[test]
      fn builtin_config_parses() {
          let cfg = ToolMappingConfig::builtin().expect("built-in config must parse");
          assert_eq!(cfg.tools.len(), 13);
          assert!(!cfg.index.is_empty());
      }
  }
  ```

### Gate

```bash
cargo test -p ambits ingest::tool_config::tests::builtin_config_parses
```

Must pass. `cargo build` must also be error-free.

---

## Phase 5 — Refactor `map_tool_call`

**Goal**: Replace the 260-line hard-coded `match` in `claude.rs` with config-driven
dispatch. Update all dependent function signatures. All 13 existing tests must pass.

### Before starting

Use Serena `find_symbol` with `include_body=true` on:
- `map_tool_call` in `src/ingest/claude.rs` — read the full current body to understand all extraction patterns before replacing
- `parse_jsonl_line` in `src/ingest/claude.rs` — confirm call sites of `map_tool_call` within it
- `parse_log_file` in `src/ingest/claude.rs` — confirm call site of `parse_jsonl_line`

Use Serena `find_referencing_symbols` on `map_tool_call` to confirm there are no other callers outside `claude.rs`.

### Tasks

- [ ] **5.1** Add imports to `src/ingest/claude.rs`:
  ```rust
  use crate::ingest::tool_config::{ToolMappingConfig, DepthSpec, ConfigWarning};
  use std::sync::Arc;
  ```

- [ ] **5.2** Add the description template renderer as a `fn` private to `claude.rs`:
  ```rust
  fn render_description(template: &str, input: &Value, short_path_fn: impl Fn(&str) -> String) -> String
  ```
  Implementation — **use a manual character-scan, not regex**:
  > **Performance note**: `render_description` is called once per parsed JSONL event in the
  > hot loop. Do NOT introduce a `regex` crate dependency here — it adds compile time and
  > requires per-call state or lazy initialization. A single linear pass over the template
  > bytes is sufficient and allocates exactly one `String`.

  - Walk `template` byte-by-byte, copying literal characters into an output `String`
  - When `{` is encountered, scan forward accumulating the key name (`[a-zA-Z_][a-zA-Z0-9_]*`) and optional `|short` modifier until `}` is found
  - If the closing `}` is not found before end-of-string, or if the key name contains invalid characters, push the original `{` as a literal and resume scanning from the character after `{`
  - For each valid placeholder: look up `input[key].as_str()`; if `|short` apply `short_path_fn`; fallback `"?"` if key absent or non-string
  - Return the resulting `String`

- [ ] **5.3** Implement the new `map_tool_call` body (replace the `match`):
  ```rust
  pub fn map_tool_call(
      config: &ToolMappingConfig,
      tool_name: &str,
      input: &Value,
      agent_id: &str,
      timestamp_str: &str,
  ) -> Option<AgentToolCall>
  ```

  9-step algorithm:
  1. `config.index.get(tool_name)` → `i`; absent → `return None`
  2. `mapping = &config.tools[i]`
  3. `file_path`: first key in `mapping.path_keys` found in `input.get(key)?.as_str()`; if none found: if `mapping.path_required` is `true` → `return None`; otherwise `file_path = None`
  4. `pattern`: first key in `mapping.pattern_keys` found as `str`; fallback `"?"`
  5. `read_depth`: `let depth_spec = mapping.depth.as_ref().expect("depth must be Some after load/merge — MissingDepth stanzas are dropped during config construction");` then match `depth_spec`:
     - `DepthSpec::Fixed { value }` → `ReadDepth::from(value.clone())`
     - `DepthSpec::Conditional { condition_key, if_true, if_false, default }`:
       - `input.get(condition_key)`:
         - `None` or `Value::Null` → `ReadDepth::from(default.clone())`
         - `Value::Bool(true)` → `ReadDepth::from(if_true.clone())`
         - `Value::Bool(false)` → `ReadDepth::from(if_false.clone())`
         - other → `ReadDepth::from(default.clone())` *(non-boolean; ConfigWarning emitted by caller if needed)*
  6. `description`: `render_description(&mapping.description, input, |s| short_path(s))`
  7. `target_symbol`: `mapping.target_symbol.as_ref().and_then(|spec| input.get(&spec.key)?.as_str().map(String::from))`
  8. `target_lines`: `mapping.target_lines.as_ref().and_then(|spec| { let off = input.get(&spec.offset_key)?.as_u64()? as usize; let lim = input.get(&spec.limit_key)?.as_u64()? as usize; Some(off..off+lim) })`
  9. Return `Some(AgentToolCall { agent_id: agent_id.to_string(), tool_name: tool_name.to_string(), file_path: file_path.map(PathBuf::from), read_depth, description, timestamp_str: timestamp_str.to_string(), target_symbol, target_lines, label: agent_id.to_string() })`

- [ ] **5.4** Update `parse_jsonl_line` signature — add `config: &ToolMappingConfig` parameter; pass it to `map_tool_call`.

- [ ] **5.5** Update `parse_log_file` signature — add `config: &ToolMappingConfig` parameter; pass it to `parse_jsonl_line`.

- [ ] **5.6** Update all 13 existing tests in `mod tests` of `claude.rs`:
  - Add `let config = ToolMappingConfig::builtin().unwrap();` at the top of each test function
  - Add `&config` as the first argument to every `map_tool_call(...)` call

### Gate

```bash
cargo test -p ambits
```

All previously passing tests must pass. Zero regressions.

### Coverage checkpoint

```bash
ambits --project . --coverage
```

`src/ingest/claude.rs` and `src/ingest/tool_config.rs` should show increased coverage
(the test suite now exercises both files meaningfully).

---

## Phase 6 — `LogTailer` Wiring

**Goal**: Add `Arc<ToolMappingConfig>` to `LogTailer`; thread it through `read_new_events`.
Call sites in `main.rs` will break (expected) — fixed in Phase 7.

### Before starting

Use Serena `find_symbol` with `depth=1` on `LogTailer` in `src/ingest/claude.rs` to
confirm the current struct fields and method list before editing.

### Tasks

- [ ] **6.1** Update `LogTailer` struct:
  ```rust
  pub struct LogTailer {
      files: Vec<PathBuf>,
      positions: HashMap<PathBuf, u64>,
      config: Arc<ToolMappingConfig>,
  }
  ```

- [ ] **6.2** Update `LogTailer::new`:
  ```rust
  pub fn new(files: Vec<PathBuf>, config: Arc<ToolMappingConfig>) -> Self {
      let mut positions = HashMap::new();
      for f in &files {
          if let Ok(meta) = fs::metadata(f) {
              positions.insert(f.clone(), meta.len());
          }
      }
      Self { files, positions, config }
  }
  ```

- [ ] **6.3** Update `read_new_events`: pass `&self.config` to `parse_jsonl_line`:
  ```rust
  match parse_jsonl_line(line.trim(), &default_id, &self.config) { ... }
  ```

### Gate

```bash
cargo build 2>&1 | grep -v "main.rs"
```

Only `main.rs` call-site errors are expected. `src/ingest/claude.rs` itself must
compile cleanly.

---

## Phase 7 — `main.rs` Integration

**Goal**: Wire `ToolMappingConfig::resolve()` at startup; display warnings; thread
`Arc` to `LogTailer` and `parse_log_file`. Full `cargo run` must work.

### Before starting

Use Serena `get_symbols_overview` on `src/main.rs` to see all top-level symbols,
then `find_symbol` with `include_body=true` on `main` function to understand the
current startup flow and all call sites of `LogTailer::new` and `parse_log_file`.

### Tasks

- [ ] **7.1** Add `--tools-config` to the `Cli` struct in `main.rs`:
  ```rust
  /// Path to a custom tool call mapping config (TOML).
  /// Overrides project-local (.ambit/tools.toml) and user-global configs.
  #[arg(long)]
  tools_config: Option<PathBuf>,
  ```

- [ ] **7.2** Add import at top of `main.rs`:
  ```rust
  use ambits::ingest::tool_config::ToolMappingConfig;
  ```

- [ ] **7.3** At the top of the main execution path (after subcommand handling, before
  project scan), resolve the config:
  ```rust
  let (tool_config, config_warnings) =
      ToolMappingConfig::resolve(cli.tools_config.as_deref());
  ```

- [ ] **7.4** Display `config_warnings` before entering the event loop:
  - **TUI mode**: collect into `Vec<String>` and render in the initial status bar or
    startup overlay before `terminal.draw(...)` enters the main loop.
    **Never use `eprintln!`** — it corrupts the ratatui alternate screen.
  - **`--dump` / `--coverage` modes**: print to stdout:
    ```rust
    for w in &config_warnings {
        println!("[ambit warning] {w:?}");
    }
    ```

- [ ] **7.5** Update every `LogTailer::new(...)` call to pass `Arc::clone(&tool_config)`.

- [ ] **7.6** Update every `parse_log_file(path)` call to
  `parse_log_file(path, &tool_config)`.

- [ ] **7.7** Session reset — wherever `LogTailer` is rebuilt after a `/clear` event,
  clone the existing `Arc` rather than calling `resolve()` again:
  ```rust
  let new_tailer = LogTailer::new(new_files, Arc::clone(&tool_config));
  ```

### Gate

```bash
cargo build
cargo run -- --project . --dump
```

Output must be identical to pre-refactor `--dump` output. No warnings displayed
(expected: none, since no user config is present in the project root).

### Coverage checkpoint

```bash
ambits --project . --coverage
```

`src/main.rs` should show increased coverage as the tool_config integration path
has now been read and edited.

---

## Phase 8 — Unit Tests (`tool_config.rs`)

**Goal**: Write all 23 unit tests in `src/ingest/tool_config.rs` `mod tests`.

### Before starting

Use Serena `find_symbol` with `depth=1` on `ToolMappingConfig` in
`src/ingest/tool_config.rs` to confirm all method signatures are in place before
writing tests that call them.

### Tasks

Add the following tests to the existing `#[cfg(test)] mod tests` block in
`src/ingest/tool_config.rs`. Use `tempfile::NamedTempFile` (already in `[dev-dependencies]`)
for file-based tests.

Each test listed below is a single `#[test]` function. Names are exact — the gate
command uses them.

- [ ] **8.1** `builtin_covers_all_tool_names` — assert all historical match-arm names present in `builtin().unwrap().index`. Minimum set: `"Read"`, `"mcp__acp__Read"`, `"Edit"`, `"mcp__acp__Edit"`, `"Write"`, `"mcp__acp__Write"`, `"Glob"`, `"Grep"`, `"mcp__serena__search_for_pattern"`, `"mcp__serena__get_symbols_overview"`, `"mcp__serena__find_symbol"`, `"mcp__serena__find_referencing_symbols"`, `"mcp__serena__replace_symbol_body"`, `"mcp__serena__insert_after_symbol"`, `"mcp__serena__insert_before_symbol"`, `"mcp__serena__rename_symbol"`, `"NotebookEdit"`

- [ ] **8.2** `builtin_index_correct_length` — `index.len()` == sum of `names.len()` across all 13 stanzas

- [ ] **8.3** `merge_user_replaces_builtin` — user stanza `names=["Read"]`, custom `description`; after merge, `config.index["Read"]` points to the user stanza (verify via `tools[i].description`)

- [ ] **8.4** `merge_user_extends_builtin` — user stanza `names=["my_custom_read"]`; after merge, `tools.len() == 14` and `"my_custom_read"` in index

- [ ] **8.5** `merge_no_overlap` — user stanza with novel name; assert all 13 historical built-in names still present in merged index

- [ ] **8.6** `merge_dedup_user_first` — two user stanzas: `[names=["Read","my_read"]]` then `[names=["Read"]]`; after merge, `"Read"` maps to the second (last) user stanza; `"my_read"` maps to a surviving entry from the first stanza

- [ ] **8.7** `merge_extends_inherits_target_symbol` — user stanza `extends="mcp__serena__find_symbol"`, only `names` + `description` set; after merge the stanza for `"mcp__serena__find_symbol"` has `target_symbol = Some(TargetSymbolSpec { key: "name_path_pattern" })`

- [ ] **8.8** `conditional_depth_key_missing` — call `map_tool_call` with `"mcp__serena__find_symbol"` and input with no `include_body`; assert `ReadDepth::Signature`

- [ ] **8.9** `conditional_depth_key_null` — input has `"include_body": null`; assert `ReadDepth::Signature`

- [ ] **8.10** `conditional_depth_key_non_boolean` — input has `"include_body": "yes"`; assert `ReadDepth::Signature`

- [ ] **8.11** `conditional_depth_with_body` — input has `"include_body": true`; assert `ReadDepth::FullBody`

- [ ] **8.12** `conditional_depth_without_body` — input has `"include_body": false`; assert `ReadDepth::Signature`

- [ ] **8.13** `pattern_keys_fallback` — `"Glob"` tool, input `{"file_mask": "*.rs"}`; assert description contains `"*.rs"` not `"?"`

- [ ] **8.14** `empty_path_keys` — stanza with `path_keys=[]`, any input; assert result `file_path == None`

- [ ] **8.15** `missing_required_path` — stanza `path_keys=["file_path"]`, input has no `file_path`; assert `map_tool_call` returns `None`

- [ ] **8.16** `unknown_tool_name` — `map_tool_call` with `"NonexistentTool_xyz"` → `None`

- [ ] **8.17** `unknown_placeholder_substitutes_question_mark` — stanza `description="tool: {typo_key}"`, input has no `typo_key`; assert description is `"tool: ?"`

- [ ] **8.18** `short_modifier_applies_short_path` — stanza `description="{file_path|short}"`, input `file_path="/a/b/c/d/e.rs"`; assert description is `"d/e.rs"`

- [ ] **8.19** `user_config_bad_version` — write `version = 999` to temp file; `load()` → `(None, warnings)` where warnings contains `ConfigWarning::UnsupportedVersion { found: 999, .. }`

- [ ] **8.20** `user_config_malformed_toml` — write `"[[["` to temp file; `load()` → `(None, warnings)` where warnings contains `ConfigWarning::ParseError`

- [ ] **8.21** `resolve_falls_back_to_builtin` — `resolve()` with `cli_override` pointing to malformed temp file; result `Arc` has `tools.len() == 13`

- [ ] **8.22** `dispatch_index_matches_linear_scan` — for every name in every stanza of `builtin()`, assert `index.get(name)` returns same stanza index as a linear scan through `tools` looking for that name

### Gate

```bash
cargo test -p ambits ingest::tool_config::tests
```

All 23 tests (including `builtin_config_parses` from Phase 4) must pass.

---

## Phase 9 — Dispatch Tests (`claude.rs`)

**Goal**: 2 tests verifying correctness of merged-config dispatch end-to-end.

### Before starting

Use Serena `find_symbol` on `mod tests` in `src/ingest/claude.rs` with `depth=1` to
see the existing test function names before adding new ones.

### Tasks

Add to `mod tests` in `src/ingest/claude.rs`:

- [ ] **9.1** `merged_config_dispatches_user_tool`:
  - Build user `ToolMappingConfig` inline with stanza: `names=["my_custom_read"]`, `path_keys=["source_path"]`, `depth=Fixed(FullBody)`, `description="Custom {source_path|short}"`, `pattern_keys=[]`
  - Merge with `ToolMappingConfig::builtin().unwrap()`
  - Call `map_tool_call(&merged, "my_custom_read", &json!({"source_path": "/a/b/c.rs"}), "agent", "ts")`
  - Assert: `result.is_some()`, `read_depth == ReadDepth::FullBody`, `description == "Custom b/c.rs"`

- [ ] **9.2** `merged_config_dispatches_replaced_builtin`:
  - User stanza: `names=["Read"]`, `path_keys=["file_path"]`, `depth=Fixed(NameOnly)`, `description="UserRead {file_path|short}"`
  - Merge with built-ins
  - Call `map_tool_call(&merged, "Read", &json!({"file_path": "/x/y/z.rs"}), "agent", "ts")`
  - Assert: `read_depth == ReadDepth::NameOnly` (not `FullBody` from built-in), `description == "UserRead y/z.rs"`

### Gate

```bash
cargo test -p ambits ingest::claude::tests::merged_config_dispatches_user_tool
cargo test -p ambits ingest::claude::tests::merged_config_dispatches_replaced_builtin
```

---

## Phase 10 — Integration Tests

**Goal**: End-to-end pipeline test from raw JSONL line to fully populated `AgentToolCall`
for all 13 tool types using the built-in config.

### Before starting

Use Serena `get_symbols_overview` on `tests/helpers/mod.rs` and `tests/e2e.rs` to
understand the existing test helper utilities before writing the new integration test.

### Tasks

- [ ] **10.1** Create `tests/integration_tool_config.rs`.

- [ ] **10.2** Implement `parse_jsonl_line_with_builtin_config_integration`:

  Use `ambits::ingest::claude::parse_jsonl_line` and `ambits::ingest::tool_config::ToolMappingConfig`.

  For each row in the table below, construct a raw JSON string matching the Claude Code
  `.jsonl` format (an assistant message containing a `tool_use` content block), call
  `parse_jsonl_line(raw, "test-agent", &config)`, match on `ParsedLine::Events(events)`,
  and assert the specified fields on `events[0]`:

  | Tool name | Key input fields in `tool_use.input` | Expected `read_depth` | Expected `file_path` present | `target_symbol` present | `target_lines` present |
  |---|---|---|---|---|---|
  | `Read` | `file_path`, `offset=10`, `limit=20` | `FullBody` | yes | no | yes (10..30) |
  | `Read` | `file_path` only | `FullBody` | yes | no | no |
  | `Edit` | `file_path` | `FullBody` | yes | no | no |
  | `Write` | `file_path` | `FullBody` | yes | no | no |
  | `Glob` | `pattern="*.rs"`, `path="/src"` | `NameOnly` | yes | no | no |
  | `Glob` | `file_mask="*.ts"` (no `pattern`) | `NameOnly` | no | no | no |
  | `Grep` | `pattern="foo"`, `path="/src"` | `Overview` | yes | no | no |
  | `mcp__serena__get_symbols_overview` | `relative_path="src/lib.rs"` | `Overview` | yes | no | no |
  | `mcp__serena__find_symbol` | `relative_path`, `include_body=true`, `name_path_pattern="Foo/bar"` | `FullBody` | yes | yes | no |
  | `mcp__serena__find_symbol` | `relative_path`, `include_body=false` | `Signature` | yes | yes | no |
  | `mcp__serena__find_referencing_symbols` | `relative_path`, `name_path="Foo/bar"` | `Overview` | yes | yes | no |
  | `mcp__serena__replace_symbol_body` | `relative_path`, `name_path="Foo/bar"` | `FullBody` | yes | yes | no |
  | `NotebookEdit` | `notebook_path="/a/b.ipynb"` | `FullBody` | yes | no | no |

### Gate

```bash
cargo test --test integration_tool_config
```

All sub-cases must pass.

---

## Phase 11 — Full Suite

**Goal**: Confirm zero regressions across the entire test suite.

### Tasks

- [ ] **11.1** Run:
  ```bash
  cargo test --all
  ```

- [ ] **11.2** Zero failures. Zero new `unused` or `dead_code` warnings introduced by
  this feature (pre-existing warnings are acceptable — do not mask them).

- [ ] **11.3** Run clippy:
  ```bash
  cargo clippy -- -D warnings
  ```
  Must produce the same warning count as `main` before this branch.

### Gate

```bash
cargo test --all    # green
```

### Coverage checkpoint

```bash
ambits --project . --coverage
```

After all tests pass, ambit has read/executed every new file in this feature. The
coverage report for `src/ingest/tool_config.rs` and `tests/integration_tool_config.rs`
should show high coverage (most symbols read or exercised).

---

## Phase 12 — Smoke Test

**Goal**: Manual verification that the running TUI is identical to pre-refactor, and
that user config extension works.

### Tasks

- [ ] **12.1** Run TUI mode with no user config:
  ```bash
  cargo run -- --project <some-project-path>
  ```
  Coverage display must be identical to pre-refactor. No warnings shown.

- [ ] **12.2** Create `.ambit/tools.toml` with a novel stanza:
  ```toml
  version = 1
  [[tool]]
  names        = ["my_test_tool"]
  path_keys    = ["source_path"]
  pattern_keys = []
  depth        = { type = "fixed", value = "Overview" }
  description  = "MyTool {source_path|short}"
  ```
  Run ambit and confirm it starts without errors; all 13 built-in tools still tracked.

- [ ] **12.3** Create `.ambit/tools.toml` with an unsupported version:
  ```toml
  version = 999
  ```
  Run with `--dump` and confirm the warning `UnsupportedVersion` is printed to stdout
  and built-in defaults are used.

- [ ] **12.4** Test the `--tools-config` flag:
  ```bash
  cargo run -- --project . --dump --tools-config /tmp/custom-tools.toml
  ```
  With a valid config at that path: confirm it loads. With a missing path: confirm
  fallback to built-ins with a `ParseError` warning.

### Final coverage checkpoint

```bash
ambits --project . --coverage
```

Record the final coverage percentages for `src/ingest/tool_config.rs`,
`src/ingest/claude.rs`, and `src/main.rs` as a baseline for future work.

---

## Dependency graph

```
Phase 1 (dep)
  └─ Phase 2 (default_tools.toml)
       └─ Phase 3 (scaffolding)
            └─ Phase 4 (core logic)          ← gate: builtin_config_parses
                 └─ Phase 5 (dispatch)        ← gate: 13 existing tests
                      ├─ Phase 6 (LogTailer)
                      │    └─ Phase 7 (main.rs) ← gate: cargo run works
                      │         └─ Phase 12 (smoke)
                      └─ Phase 8 (unit tests)   ← can run in parallel with 6
                           └─ Phase 9 (dispatch tests)
                                └─ Phase 10 (integration tests)
                                     └─ Phase 11 (full suite)
```

Phases 6 and 8 may proceed in parallel once Phase 5 is complete.

---

## Rollback

The only shared state between this branch and `main` is the `Cargo.toml` dependency
addition (Phase 1). All other changes are additive (new files) or isolated to
`src/ingest/`. The branch can be abandoned at any phase boundary cleanly.
