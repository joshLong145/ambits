use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Deserialize;

use crate::tracking::ReadDepth;

// ---------------------------------------------------------------------------
// Top-level config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct ToolMappingConfig {
    pub version: u32,
    #[serde(rename = "tool", default)]
    pub tools: Vec<ToolMapping>,
    /// Name → index into `tools`. Built after deserialization via `build_index()`.
    #[serde(skip)]
    pub(crate) index: HashMap<String, usize>,
}

// ---------------------------------------------------------------------------
// Per-stanza mapping
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct ToolMapping {
    pub names: Vec<String>,
    pub path_keys: Vec<String>,
    #[serde(default)]
    pub pattern_keys: Vec<String>,
    /// `None` means "inherit from base stanza via `extends`".
    /// A standalone stanza (no `extends`) with `depth = None` is a config error.
    #[serde(default)]
    pub depth: Option<DepthSpec>,
    pub description: String,
    /// When `true` (default), `map_tool_call` returns `None` if no path key is found.
    /// Set to `false` for tools like Glob/Grep where the path is an optional filter.
    #[serde(default = "default_path_required")]
    pub path_required: bool,
    #[serde(default)]
    pub target_symbol: Option<TargetSymbolSpec>,
    #[serde(default)]
    pub target_lines: Option<TargetLinesSpec>,
    /// Name of a built-in stanza to inherit fields from.
    #[serde(default)]
    pub extends: Option<String>,
}

fn default_path_required() -> bool {
    true
}

// ---------------------------------------------------------------------------
// DepthSpec
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DepthSpec {
    Fixed {
        value: ReadDepthDe,
    },
    Conditional {
        condition_key: String,
        if_true: ReadDepthDe,
        if_false: ReadDepthDe,
        default: ReadDepthDe,
    },
}

/// String-deserializable mirror of `ReadDepth`.
/// Kept separate so `tracking::ReadDepth` stays free of serde.
#[derive(Debug, Clone, Deserialize)]
pub enum ReadDepthDe {
    NameOnly,
    Overview,
    Signature,
    FullBody,
}

impl From<ReadDepthDe> for ReadDepth {
    fn from(d: ReadDepthDe) -> Self {
        match d {
            ReadDepthDe::NameOnly  => ReadDepth::NameOnly,
            ReadDepthDe::Overview  => ReadDepth::Overview,
            ReadDepthDe::Signature => ReadDepth::Signature,
            ReadDepthDe::FullBody  => ReadDepth::FullBody,
        }
    }
}

// ---------------------------------------------------------------------------
// Sub-specs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct TargetSymbolSpec {
    pub key: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TargetLinesSpec {
    pub offset_key: String,
    pub limit_key: String,
}

// ---------------------------------------------------------------------------
// Warnings (never written to stderr — returned to caller)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum ConfigWarning {
    ParseError             { path: String, message: String },
    UnsupportedVersion     { path: String, found: u32, supported: u32 },
    EmptyNames             { stanza_index: usize },
    DuplicateName          { name: String, kept_index: usize, dropped_index: usize },
    ConditionalKeyNonBoolean { tool_name: String, key: String },
    MissingDepth           { stanza_index: usize },
}

impl std::fmt::Display for ConfigWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigWarning::ParseError { path, message } =>
                write!(f, "tool config parse error in '{}': {}", path, message),
            ConfigWarning::UnsupportedVersion { path, found, supported } =>
                write!(f, "tool config '{}': unsupported version {} (max {}); using built-in defaults",
                    path, found, supported),
            ConfigWarning::EmptyNames { stanza_index } =>
                write!(f, "tool config stanza {} has an empty 'names' list; skipping", stanza_index),
            ConfigWarning::DuplicateName { name, kept_index, dropped_index } =>
                write!(f, "tool config: name '{}' in stanzas {} and {}; stanza {} wins",
                    name, dropped_index, kept_index, kept_index),
            ConfigWarning::ConditionalKeyNonBoolean { tool_name, key } =>
                write!(f, "tool '{}': conditional depth key '{}' is not boolean; using default",
                    tool_name, key),
            ConfigWarning::MissingDepth { stanza_index } =>
                write!(f, "tool config stanza {} has no 'depth' and no 'extends'; skipping",
                    stanza_index),
        }
    }
}

// ---------------------------------------------------------------------------
// ToolMappingConfig methods
// ---------------------------------------------------------------------------

impl ToolMappingConfig {
    pub const SUPPORTED_VERSION: u32 = 1;

    /// Load and index the embedded built-in config.
    /// Returns `Err` only if the bundled TOML is malformed — caught by CI test.
    pub fn builtin() -> Result<Self, toml::de::Error> {
        let mut cfg: Self = toml::from_str(include_str!("default_tools.toml"))?;
        cfg.build_index();
        Ok(cfg)
    }

    /// Load a user config file. On any error, emits a warning and returns `None`.
    pub fn load(path: &Path) -> (Option<Self>, Vec<ConfigWarning>) {
        let mut warnings = Vec::new();
        let path_str = path.to_string_lossy().into_owned();

        let content = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                warnings.push(ConfigWarning::ParseError {
                    path: path_str,
                    message: e.to_string(),
                });
                return (None, warnings);
            }
        };

        let mut cfg: Self = match toml::from_str(&content) {
            Ok(c) => c,
            Err(e) => {
                warnings.push(ConfigWarning::ParseError {
                    path: path_str,
                    message: e.to_string(),
                });
                return (None, warnings);
            }
        };

        if cfg.version > Self::SUPPORTED_VERSION {
            warnings.push(ConfigWarning::UnsupportedVersion {
                path: path_str,
                found: cfg.version,
                supported: Self::SUPPORTED_VERSION,
            });
            return (None, warnings);
        }

        cfg.build_index();
        (Some(cfg), warnings)
    }

    /// Discover user config (CLI override → project-local → user-global) and
    /// merge with built-ins. Returns `(Arc<config>, warnings)`.
    pub fn resolve(cli: Option<&Path>) -> (Arc<Self>, Vec<ConfigWarning>) {
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

        let user_path = Self::find_user_config(cli);
        let (user, mut uw) = match user_path {
            Some(p) => Self::load(&p),
            None    => (None, vec![]),
        };
        warnings.append(&mut uw);

        let merged = match user {
            Some(u) => Self::merge(builtin, u, &mut warnings),
            None    => builtin,
        };

        (Arc::new(merged), warnings)
    }

    /// Merge user config over built-in base.
    /// Steps: A=dedup user, B=skip empty names, C=extends inheritance, D=filter+append
    pub fn merge(base: Self, user: Self, warnings: &mut Vec<ConfigWarning>) -> Self {
        // Step A — deduplicate user stanzas (last stanza per name wins).
        // Stanzas with empty `names` are passed through to Step B for warning emission.
        let mut name_to_winner: HashMap<&str, usize> = HashMap::new();
        for (i, mapping) in user.tools.iter().enumerate() {
            for name in &mapping.names {
                if let Some(prev_i) = name_to_winner.insert(name.as_str(), i) {
                    warnings.push(ConfigWarning::DuplicateName {
                        name: name.clone(),
                        kept_index: i,
                        dropped_index: prev_i,
                    });
                }
            }
        }
        // Collect winning indices plus any empty-names stanza indices.
        let mut winning_indices: Vec<usize> = name_to_winner.values().copied().collect();
        for (i, mapping) in user.tools.iter().enumerate() {
            if mapping.names.is_empty() {
                winning_indices.push(i);
            }
        }
        winning_indices.sort_unstable();
        winning_indices.dedup();
        let canonical_user: Vec<ToolMapping> = winning_indices
            .into_iter()
            .map(|i| user.tools[i].clone())
            .collect();

        // Step B — skip stanzas with empty `names`.
        let mut valid_user: Vec<ToolMapping> = Vec::with_capacity(canonical_user.len());
        for (i, stanza) in canonical_user.into_iter().enumerate() {
            if stanza.names.is_empty() {
                warnings.push(ConfigWarning::EmptyNames { stanza_index: i });
            } else {
                valid_user.push(stanza);
            }
        }

        // Step C — apply `extends` field inheritance from base stanzas.
        let mut resolved_user: Vec<ToolMapping> = Vec::with_capacity(valid_user.len());
        for (idx, mut stanza) in valid_user.into_iter().enumerate() {
            if let Some(ref base_name) = stanza.extends.clone() {
                if let Some(base_stanza) = base.tools.iter().find(|b| b.names.contains(base_name)) {
                    if stanza.path_keys.is_empty() {
                        stanza.path_keys = base_stanza.path_keys.clone();
                    }
                    if stanza.pattern_keys.is_empty() {
                        stanza.pattern_keys = base_stanza.pattern_keys.clone();
                    }
                    if stanza.depth.is_none() {
                        stanza.depth = base_stanza.depth.clone();
                    }
                    if stanza.target_symbol.is_none() {
                        stanza.target_symbol = base_stanza.target_symbol.clone();
                    }
                    if stanza.target_lines.is_none() {
                        stanza.target_lines = base_stanza.target_lines.clone();
                    }
                }
                // base_name not found: skip silently
            }

            // depth must be Some by now (either explicit or inherited via extends).
            if stanza.depth.is_none() {
                warnings.push(ConfigWarning::MissingDepth { stanza_index: idx });
                continue;
            }

            resolved_user.push(stanza);
        }

        // Step D — filter base: drop stanzas whose names overlap with user names.
        let user_name_set: std::collections::HashSet<&str> = resolved_user
            .iter()
            .flat_map(|m| m.names.iter().map(|n| n.as_str()))
            .collect();

        let filtered_base: Vec<ToolMapping> = base.tools
            .into_iter()
            .filter(|b| !b.names.iter().any(|n| user_name_set.contains(n.as_str())))
            .collect();

        let mut result = Vec::with_capacity(filtered_base.len() + resolved_user.len());
        result.extend(filtered_base);
        result.extend(resolved_user);

        let mut merged = ToolMappingConfig {
            version: base.version,
            tools: result,
            index: HashMap::new(),
        };
        merged.build_index();
        merged
    }

    /// Build the O(1) dispatch index: name → index into `self.tools`.
    fn build_index(&mut self) {
        let total_names: usize = self.tools.iter().map(|m| m.names.len()).sum();
        self.index = HashMap::with_capacity(total_names);
        for (i, mapping) in self.tools.iter().enumerate() {
            for name in &mapping.names {
                self.index.insert(name.clone(), i);
            }
        }
    }

    /// Discover user config path:
    /// 1. CLI override, 2. `.ambit/tools.toml` in CWD, 3. `~/.config/ambit/tools.toml`
    fn find_user_config(cli: Option<&Path>) -> Option<PathBuf> {
        if let Some(p) = cli {
            if p.exists() {
                return Some(p.to_path_buf());
            }
        }

        if let Ok(cwd) = std::env::current_dir() {
            let local = cwd.join(".ambit/tools.toml");
            if local.exists() {
                return Some(local);
            }
        }

        let home = std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .map(PathBuf::from);
        if let Some(home) = home {
            let global = home.join(".config/ambit/tools.toml");
            if global.exists() {
                return Some(global);
            }
        }

        None
    }

    /// Empty config — fallback when built-in fails to parse.
    fn empty() -> Self {
        Self {
            version: Self::SUPPORTED_VERSION,
            tools: vec![],
            index: HashMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use serde_json;

    // -----------------------------------------------------------------------
    // 1. builtin_config_parses
    // -----------------------------------------------------------------------
    #[test]
    fn builtin_config_parses() {
        let cfg = ToolMappingConfig::builtin().expect("built-in config must parse");
        assert_eq!(cfg.tools.len(), 13);
        assert!(!cfg.index.is_empty());
    }

    // -----------------------------------------------------------------------
    // 2. builtin_covers_all_tool_names
    // -----------------------------------------------------------------------
    #[test]
    fn builtin_covers_all_tool_names() {
        let cfg = ToolMappingConfig::builtin().unwrap();
        let all_names: HashSet<&str> = cfg.tools.iter()
            .flat_map(|m| m.names.iter().map(|n| n.as_str()))
            .collect();

        let expected = [
            "Read", "Edit", "Write", "Glob", "Grep", "NotebookEdit",
            "mcp__serena__find_symbol",
            "mcp__serena__find_referencing_symbols",
            "mcp__serena__replace_symbol_body",
            "mcp__serena__insert_after_symbol",
            "mcp__serena__insert_before_symbol",
            "mcp__serena__rename_symbol",
            "mcp__serena__get_symbols_overview",
            "mcp__serena__list_dir",
            "mcp__serena__search_for_pattern",
            "mcp__plugin_serena_serena__find_symbol",
            "mcp__plugin_serena_serena__read_file",
            "mcp__plugin_serena_serena__create_text_file",
            "mcp__plugin_serena_serena__replace_content",
            "mcp__plugin_serena_serena__find_file",
            "mcp__plugin_serena_serena__list_dir",
            "mcp__plugin_serena_serena__search_for_pattern",
            "mcp__plugin_serena_serena__get_symbols_overview",
            "mcp__plugin_serena_serena__find_referencing_symbols",
            "mcp__plugin_serena_serena__replace_symbol_body",
            "mcp__plugin_serena_serena__insert_after_symbol",
            "mcp__plugin_serena_serena__insert_before_symbol",
            "mcp__plugin_serena_serena__rename_symbol",
            "mcp__acp__Read", "mcp__acp__Edit", "mcp__acp__Write",
            "mcp__serena__find_file",
        ];
        for name in &expected {
            assert!(all_names.contains(name), "missing tool name: {name}");
        }
    }

    // -----------------------------------------------------------------------
    // 3. dispatch_index_matches_linear_scan
    // -----------------------------------------------------------------------
    #[test]
    fn dispatch_index_matches_linear_scan() {
        let cfg = ToolMappingConfig::builtin().unwrap();
        for name in cfg.index.keys() {
            let idx_result = cfg.index.get(name).copied();
            let linear_result = cfg.tools.iter()
                .position(|m| m.names.contains(name));
            assert_eq!(idx_result, linear_result, "index mismatch for '{name}'");
        }
    }

    // -----------------------------------------------------------------------
    // 4. merge_user_replaces_builtin
    // -----------------------------------------------------------------------
    #[test]
    fn merge_user_replaces_builtin() {
        let base = ToolMappingConfig::builtin().unwrap();
        let user_toml = r#"
version = 1
[[tool]]
names        = ["Read"]
path_keys    = ["custom_path"]
pattern_keys = []
depth        = { type = "fixed", value = "NameOnly" }
description  = "custom Read {custom_path}"
"#;
        let mut user: ToolMappingConfig = toml::from_str(user_toml).unwrap();
        user.build_index();

        let mut warnings = Vec::new();
        let merged = ToolMappingConfig::merge(base, user, &mut warnings);

        // The user stanza for "Read" replaces the built-in.
        let idx = *merged.index.get("Read").expect("Read must exist");
        let mapping = &merged.tools[idx];
        assert_eq!(mapping.path_keys, vec!["custom_path"]);
        assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");

        // mcp__acp__Read was in the same built-in stanza — it should also be dropped.
        assert!(!merged.index.contains_key("mcp__acp__Read"),
            "mcp__acp__Read should be dropped when Read is overridden");
    }

    // -----------------------------------------------------------------------
    // 5. merge_user_extends_builtin (appends novel names)
    // -----------------------------------------------------------------------
    #[test]
    fn merge_user_extends_builtin() {
        let base = ToolMappingConfig::builtin().unwrap();
        let base_count = base.tools.len();

        let user_toml = r#"
version = 1
[[tool]]
names        = ["MyCustomTool"]
path_keys    = ["path"]
pattern_keys = []
depth        = { type = "fixed", value = "Overview" }
description  = "Custom {path}"
"#;
        let mut user: ToolMappingConfig = toml::from_str(user_toml).unwrap();
        user.build_index();

        let mut warnings = Vec::new();
        let merged = ToolMappingConfig::merge(base, user, &mut warnings);

        assert_eq!(merged.tools.len(), base_count + 1);
        assert!(merged.index.contains_key("MyCustomTool"));
        assert!(warnings.is_empty());
    }

    // -----------------------------------------------------------------------
    // 6. merge_no_overlap
    // -----------------------------------------------------------------------
    #[test]
    fn merge_no_overlap() {
        let base = ToolMappingConfig::builtin().unwrap();
        let base_count = base.tools.len();

        // An empty user config — all built-ins must be preserved.
        let user_toml = "version = 1\n";
        let mut user: ToolMappingConfig = toml::from_str(user_toml).unwrap();
        user.build_index();

        let mut warnings = Vec::new();
        let merged = ToolMappingConfig::merge(base, user, &mut warnings);

        assert_eq!(merged.tools.len(), base_count);
        assert!(warnings.is_empty());
    }

    // -----------------------------------------------------------------------
    // 7. merge_dedup_user_first (last user stanza per name wins)
    // -----------------------------------------------------------------------
    #[test]
    fn merge_dedup_user_first() {
        let base = ToolMappingConfig::builtin().unwrap();
        let user_toml = r#"
version = 1
[[tool]]
names        = ["NewTool"]
path_keys    = ["old_key"]
pattern_keys = []
depth        = { type = "fixed", value = "NameOnly" }
description  = "old"

[[tool]]
names        = ["NewTool"]
path_keys    = ["new_key"]
pattern_keys = []
depth        = { type = "fixed", value = "FullBody" }
description  = "new"
"#;
        let mut user: ToolMappingConfig = toml::from_str(user_toml).unwrap();
        user.build_index();

        let mut warnings = Vec::new();
        let merged = ToolMappingConfig::merge(base, user, &mut warnings);

        let idx = *merged.index.get("NewTool").unwrap();
        assert_eq!(merged.tools[idx].path_keys, vec!["new_key"]);
        // Expect one DuplicateName warning.
        assert!(warnings.iter().any(|w| matches!(w, ConfigWarning::DuplicateName { name, .. } if name == "NewTool")));
    }

    // -----------------------------------------------------------------------
    // 8. merge_extends_inherits_target_symbol
    // -----------------------------------------------------------------------
    #[test]
    fn merge_extends_inherits_target_symbol() {
        let base = ToolMappingConfig::builtin().unwrap();
        // Extend "mcp__serena__find_symbol" — inherits path_keys + target_symbol.
        let user_toml = r#"
version = 1
[[tool]]
names        = ["MyFindSymbol"]
path_keys    = []
pattern_keys = []
description  = "Find {name_path_pattern}"
extends      = "mcp__serena__find_symbol"
"#;
        let mut user: ToolMappingConfig = toml::from_str(user_toml).unwrap();
        user.build_index();

        let mut warnings = Vec::new();
        let merged = ToolMappingConfig::merge(base, user, &mut warnings);

        let idx = *merged.index.get("MyFindSymbol").expect("MyFindSymbol must exist");
        let mapping = &merged.tools[idx];
        // target_symbol inherited from find_symbol base stanza.
        assert!(mapping.target_symbol.is_some());
        // path_keys inherited (relative_path).
        assert!(!mapping.path_keys.is_empty());
        // depth inherited (Conditional).
        assert!(matches!(mapping.depth, Some(DepthSpec::Conditional { .. })));
    }

    // -----------------------------------------------------------------------
    // 9. conditional_depth_with_body_true
    // -----------------------------------------------------------------------
    #[test]
    fn conditional_depth_with_body_true() {
        let spec = DepthSpec::Conditional {
            condition_key: "include_body".into(),
            if_true: ReadDepthDe::FullBody,
            if_false: ReadDepthDe::Signature,
            default: ReadDepthDe::Signature,
        };
        let input = serde_json::json!({ "include_body": true });
        let depth = match &spec {
            DepthSpec::Conditional { condition_key, if_true, if_false, default } => {
                match input.get(condition_key) {
                    Some(serde_json::Value::Bool(true))  => ReadDepth::from(if_true.clone()),
                    Some(serde_json::Value::Bool(false)) => ReadDepth::from(if_false.clone()),
                    _                                    => ReadDepth::from(default.clone()),
                }
            }
            DepthSpec::Fixed { value } => ReadDepth::from(value.clone()),
        };
        assert_eq!(depth, ReadDepth::FullBody);
    }

    // -----------------------------------------------------------------------
    // 10. conditional_depth_with_body_false
    // -----------------------------------------------------------------------
    #[test]
    fn conditional_depth_with_body_false() {
        let spec = DepthSpec::Conditional {
            condition_key: "include_body".into(),
            if_true: ReadDepthDe::FullBody,
            if_false: ReadDepthDe::Signature,
            default: ReadDepthDe::Signature,
        };
        let input = serde_json::json!({ "include_body": false });
        let depth = match &spec {
            DepthSpec::Conditional { condition_key, if_true, if_false, default } => {
                match input.get(condition_key) {
                    Some(serde_json::Value::Bool(true))  => ReadDepth::from(if_true.clone()),
                    Some(serde_json::Value::Bool(false)) => ReadDepth::from(if_false.clone()),
                    _                                    => ReadDepth::from(default.clone()),
                }
            }
            DepthSpec::Fixed { value } => ReadDepth::from(value.clone()),
        };
        assert_eq!(depth, ReadDepth::Signature);
    }

    // -----------------------------------------------------------------------
    // 11. conditional_depth_absent_key
    // -----------------------------------------------------------------------
    #[test]
    fn conditional_depth_absent_key() {
        let spec = DepthSpec::Conditional {
            condition_key: "include_body".into(),
            if_true: ReadDepthDe::FullBody,
            if_false: ReadDepthDe::Signature,
            default: ReadDepthDe::Signature,
        };
        let input = serde_json::json!({ "name_path_pattern": "Foo" }); // no include_body
        let depth = match &spec {
            DepthSpec::Conditional { condition_key, if_true, if_false, default } => {
                match input.get(condition_key) {
                    Some(serde_json::Value::Bool(true))  => ReadDepth::from(if_true.clone()),
                    Some(serde_json::Value::Bool(false)) => ReadDepth::from(if_false.clone()),
                    _                                    => ReadDepth::from(default.clone()),
                }
            }
            DepthSpec::Fixed { value } => ReadDepth::from(value.clone()),
        };
        assert_eq!(depth, ReadDepth::Signature);
    }

    // -----------------------------------------------------------------------
    // 12. user_config_bad_version
    // -----------------------------------------------------------------------
    #[test]
    fn user_config_bad_version() {
        let bad_toml = r#"
version = 999
[[tool]]
names        = ["Foo"]
path_keys    = ["path"]
pattern_keys = []
depth        = { type = "fixed", value = "NameOnly" }
description  = "Foo"
"#;
        let tmp = write_temp_toml(bad_toml);
        let (result, warnings) = ToolMappingConfig::load(tmp.path());
        assert!(result.is_none(), "should reject unsupported version");
        assert!(warnings.iter().any(|w| matches!(w, ConfigWarning::UnsupportedVersion { .. })));
    }

    // -----------------------------------------------------------------------
    // 13. user_config_malformed_toml
    // -----------------------------------------------------------------------
    #[test]
    fn user_config_malformed_toml() {
        let bad_toml = "this is not valid toml [[[";
        let tmp = write_temp_toml(bad_toml);
        let (result, warnings) = ToolMappingConfig::load(tmp.path());
        assert!(result.is_none(), "should reject malformed TOML");
        assert!(warnings.iter().any(|w| matches!(w, ConfigWarning::ParseError { .. })));
    }

    // -----------------------------------------------------------------------
    // 14. resolve_falls_back_to_builtin
    // -----------------------------------------------------------------------
    #[test]
    fn resolve_falls_back_to_builtin() {
        // Pass a non-existent path — resolve() should silently fall back.
        let (cfg, warnings) = ToolMappingConfig::resolve(Some(std::path::Path::new("/nonexistent/tools.toml")));
        assert_eq!(cfg.tools.len(), 13, "should have 13 built-in tools");
        // No warnings since the file simply doesn't exist (no ParseError).
        assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
    }

    // -----------------------------------------------------------------------
    // 15. empty_path_keys_produces_none_file_path
    // -----------------------------------------------------------------------
    #[test]
    fn empty_path_keys_produces_none_file_path() {
        let toml = r#"
version = 1
[[tool]]
names        = ["PathlessCmd"]
path_keys    = []
path_required = false
pattern_keys = []
depth        = { type = "fixed", value = "Overview" }
description  = "no path here"
"#;
        let mut cfg: ToolMappingConfig = toml::from_str(toml).unwrap();
        cfg.build_index();

        let input = serde_json::json!({ "unrelated": "value" });
        let mapping = cfg.tools.iter().find(|m| m.names.contains(&"PathlessCmd".to_string())).unwrap();
        let file_path_str: Option<&str> = mapping.path_keys.iter()
            .find_map(|k| input.get(k).and_then(|v| v.as_str()));
        // path_required = false: no file_path but call should NOT return None.
        assert!(file_path_str.is_none());
        assert!(!mapping.path_required);
    }

    // -----------------------------------------------------------------------
    // 16. missing_required_path_returns_none  (verified via map_tool_call in claude.rs tests)
    //     — here we just verify the mapping config flag is set correctly for Read.
    // -----------------------------------------------------------------------
    #[test]
    fn read_tool_has_path_required_true() {
        let cfg = ToolMappingConfig::builtin().unwrap();
        let idx = *cfg.index.get("Read").unwrap();
        assert!(cfg.tools[idx].path_required, "Read must require a path");
    }

    // -----------------------------------------------------------------------
    // 17. glob_tool_has_path_required_false
    // -----------------------------------------------------------------------
    #[test]
    fn glob_tool_has_path_required_false() {
        let cfg = ToolMappingConfig::builtin().unwrap();
        let idx = *cfg.index.get("Glob").unwrap();
        assert!(!cfg.tools[idx].path_required, "Glob must not require a path");
    }

    // -----------------------------------------------------------------------
    // 18. merge_empty_names_stanza_emits_warning
    // -----------------------------------------------------------------------
    #[test]
    fn merge_empty_names_stanza_emits_warning() {
        let base = ToolMappingConfig::builtin().unwrap();
        let user_toml = r#"
version = 1
[[tool]]
names        = []
path_keys    = ["path"]
pattern_keys = []
depth        = { type = "fixed", value = "NameOnly" }
description  = "empty names"
"#;
        let mut user: ToolMappingConfig = toml::from_str(user_toml).unwrap();
        user.build_index();

        let mut warnings = Vec::new();
        let merged = ToolMappingConfig::merge(base, user, &mut warnings);
        // Stanza must be skipped and a warning emitted.
        assert!(!merged.index.is_empty());
        assert!(warnings.iter().any(|w| matches!(w, ConfigWarning::EmptyNames { .. })));
    }

    // -----------------------------------------------------------------------
    // 19. find_symbol_has_conditional_depth
    // -----------------------------------------------------------------------
    #[test]
    fn find_symbol_has_conditional_depth() {
        let cfg = ToolMappingConfig::builtin().unwrap();
        let idx = *cfg.index.get("mcp__serena__find_symbol").unwrap();
        assert!(
            matches!(cfg.tools[idx].depth, Some(DepthSpec::Conditional { .. })),
            "find_symbol must have conditional depth"
        );
    }

    // -----------------------------------------------------------------------
    // 20. find_symbol_has_target_symbol
    // -----------------------------------------------------------------------
    #[test]
    fn find_symbol_has_target_symbol() {
        let cfg = ToolMappingConfig::builtin().unwrap();
        let idx = *cfg.index.get("mcp__serena__find_symbol").unwrap();
        assert!(cfg.tools[idx].target_symbol.is_some());
    }

    // -----------------------------------------------------------------------
    // 21. read_tool_has_target_lines
    // -----------------------------------------------------------------------
    #[test]
    fn read_tool_has_target_lines() {
        let cfg = ToolMappingConfig::builtin().unwrap();
        let idx = *cfg.index.get("Read").unwrap();
        assert!(cfg.tools[idx].target_lines.is_some());
    }

    // -----------------------------------------------------------------------
    // 22. grep_tool_has_pattern_keys
    // -----------------------------------------------------------------------
    #[test]
    fn grep_tool_has_pattern_keys() {
        let cfg = ToolMappingConfig::builtin().unwrap();
        let idx = *cfg.index.get("Grep").unwrap();
        let mapping = &cfg.tools[idx];
        assert!(mapping.pattern_keys.contains(&"pattern".to_string()));
        assert!(mapping.pattern_keys.contains(&"substring_pattern".to_string()));
    }

    // -----------------------------------------------------------------------
    // Helper
    // -----------------------------------------------------------------------

    fn write_temp_toml(content: &str) -> tempfile::NamedTempFile {
        use std::io::Write as IoWrite;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }
}
