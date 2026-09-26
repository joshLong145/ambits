use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Deserialize;

use crate::tracking::ReadDepth;

// ---------------------------------------------------------------------------
// Top-level config
// ---------------------------------------------------------------------------

/// `[cache]` stanza: project-level defaults for coverage journaling.
/// CLI flags override anything set here. Every field is optional so an absent
/// stanza means "use the built-in defaults" rather than "disable".
#[derive(Debug, Clone, Default, Deserialize)]
pub struct CacheConfig {
    /// Set `false` to disable journaling for this project.
    #[serde(default)]
    pub enabled: Option<bool>,
    /// How often to diff the ledger into the journal.
    #[serde(default)]
    pub flush_interval_ms: Option<u64>,
}

/// `[editor]` stanza: how to open a symbol's file in an external editor.
/// Every field is optional; an absent stanza (or absent field) falls through
/// to `$VISUAL`/`$EDITOR`.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct EditorConfig {
    /// Editor command, e.g. `"code"`, `"nvim"`, or a template containing
    /// `{file}`/`{line}` placeholders, e.g. `"code -g {file}:{line}"`.
    #[serde(default)]
    pub command: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolMappingConfig {
    pub version: u32,
    #[serde(rename = "tool", default)]
    pub tools: Vec<ToolMapping>,
    /// Coverage-journal settings. See [`CacheConfig`].
    #[serde(default)]
    pub cache: CacheConfig,
    /// "Open in editor" settings. See [`EditorConfig`].
    #[serde(default)]
    pub editor: EditorConfig,
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
    #[serde(default)]
    pub target_selectors: Option<TargetSelectorSpec>,
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
    /// Match the string value of `key` in the tool input against an ordered
    /// list of patterns. The first matching pattern wins; unmatched falls back
    /// to `default`. Currently supports `prefix` (starts_with) matching.
    PatternMatch {
        /// The input JSON key whose string value is tested (e.g. `"command"`).
        key: String,
        /// Ordered list of patterns; first match wins.
        patterns: Vec<CommandPattern>,
        /// Depth assigned when no pattern matches (or key is absent).
        default: ReadDepthDe,
    },
}

impl DepthSpec {
    /// Resolve the read depth for the given tool input JSON.
    ///
    /// Centralises the dispatch logic so callers don't duplicate `match` arms.
    pub fn resolve(&self, input: &serde_json::Value) -> crate::tracking::ReadDepth {
        use crate::tracking::ReadDepth;
        let (depth, variant, detail) = match self {
            DepthSpec::Fixed { value } => (ReadDepth::from(*value), "fixed", String::new()),
            DepthSpec::Conditional { condition_key, if_true, if_false, default } => {
                match input.get(condition_key) {
                    Some(serde_json::Value::Bool(true)) =>
                        (ReadDepth::from(*if_true), "conditional", format!("{condition_key}=true")),
                    Some(serde_json::Value::Bool(false)) =>
                        (ReadDepth::from(*if_false), "conditional", format!("{condition_key}=false")),
                    _ =>
                        (ReadDepth::from(*default), "conditional", format!("{condition_key} absent/non-bool, default")),
                }
            }
            DepthSpec::PatternMatch { key, patterns, default } => {
                let cmd = input.get(key).and_then(|v| v.as_str()).unwrap_or("");
                match patterns
                    .iter()
                    .enumerate()
                    .find(|(_, p)| match p.match_type {
                        MatchType::Prefix   => cmd.starts_with(p.prefix.as_str()),
                        MatchType::Contains => cmd.contains(p.prefix.as_str()),
                        MatchType::Exact    => cmd == p.prefix.as_str(),
                        MatchType::AmbitsSubcommand => {
                            cmd.contains("ambits")
                                && cmd.split_whitespace().any(|t| t == p.prefix.as_str())
                        }
                    }) {
                    Some((i, p)) =>
                        (ReadDepth::from(p.depth), "pattern_match", format!("pattern[{i}] '{}' ({:?})", p.prefix, p.match_type)),
                    None =>
                        (ReadDepth::from(*default), "pattern_match", "no pattern matched, default".to_string()),
                }
            }
        };
        log::debug!(
            target: "ambits::depth_resolution",
            variant = variant,
            detail = detail,
            depth:? = depth;
            "resolved"
        );
        depth
    }
}

/// How the `prefix` field of a [`CommandPattern`] is compared against the input value.
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MatchType {
    /// `value.starts_with(prefix)` — the default; backward-compatible with existing configs.
    #[default]
    Prefix,
    /// `value.contains(prefix)` — useful for piped or compound commands.
    Contains,
    /// `value == prefix` — exact equality.
    Exact,
    /// `prefix` names an `ambits` subcommand (e.g. `"grep"`); matches when any
    /// whitespace-separated token of the command equals it. Global flags sit
    /// between the binary and its subcommand (`ambits -p . grep ...`), so a
    /// `starts_with`/`contains` test on a fixed literal can't span the two —
    /// this mirrors `TargetSelectorSpec`'s own token scan for the same reason.
    AmbitsSubcommand,
}

/// A single pattern entry in a `PatternMatch` depth spec.
#[derive(Debug, Clone, Deserialize)]
pub struct CommandPattern {
    /// The literal to compare against the input value (field name kept for TOML compatibility).
    pub prefix: String,
    /// How to compare `prefix` against the input value. Defaults to `Prefix` (starts_with).
    #[serde(default)]
    pub match_type: MatchType,
    pub depth: ReadDepthDe,
}

/// String-deserializable mirror of `ReadDepth`.
/// Kept separate so `tracking::ReadDepth` stays free of serde.
#[derive(Debug, Clone, Copy, Deserialize)]
pub enum ReadDepthDe {
    Unseen,
    NameOnly,
    Overview,
    Signature,
    FullBody,
}

impl From<ReadDepthDe> for ReadDepth {
    fn from(d: ReadDepthDe) -> Self {
        match d {
            ReadDepthDe::Unseen    => ReadDepth::Unseen,
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

/// Extract symbol selectors from a free-form command string.
///
/// Exists because `ambits show <id-or-hash>...` reads code without naming a
/// file, so the ordinary `path_keys` route records nothing. Without this,
/// using ambit's own lookup makes coverage *fall*, which is precisely backwards
/// — an agent that reads efficiently would look less informed than one that
/// pulls whole files.
///
/// The `requires` marker, the optional `subcommand`, and the selector grammar
/// decide what is picked up. Flags need no special handling: `--no-body` and a numeric `--max-bytes`
/// argument simply fail to parse as a selector and are ignored, so the spec
/// does not have to track the command's option list.
#[derive(Debug, Clone, Deserialize)]
pub struct TargetSelectorSpec {
    /// Input key holding the command string (`command` for Bash).
    pub key: String,
    /// Substring the command must contain before any selector is taken.
    /// Keeps an unrelated command that merely mentions a symbol id from
    /// registering a read.
    pub requires: String,
    /// Subcommand that must appear in an invocation before its selectors count,
    /// and after which they are read.
    ///
    /// `requires` alone cannot express this: global flags sit between the
    /// binary and its subcommand (`ambits -p . show <id>`), so no fixed
    /// substring spans the two. Matching a whole token instead is exact, and
    /// taking selectors only from what follows it mirrors the grammar —
    /// everything before the subcommand is a global flag and cannot be a
    /// selector anyway.
    ///
    /// Without it, `ambits rg 'src/app.rs::App'` credited a full read of that
    /// symbol. A search pattern is a regex over file content, not a request for
    /// a definition, and `find` already journals precisely what it displayed.
    /// Optional, so a tool whose every invocation returns definitions needs no
    /// such marker.
    #[serde(default)]
    pub subcommand: Option<String>,
    /// Depth credited when the command returns definitions.
    pub depth: ReadDepthDe,
    /// Flag that makes the command return metadata only.
    #[serde(default)]
    pub shallow_flag: Option<String>,
    /// Depth credited when `shallow_flag` is present — the caller learned the
    /// symbol exists and where it is, not what it says.
    #[serde(default)]
    pub shallow_depth: Option<ReadDepthDe>,
}

impl TargetSelectorSpec {
    /// Pull selectors and the depth each one earns out of a tool input.
    ///
    /// Resolution is **per invocation**, not per command. One shell command can
    /// hold several invocations — chained with `&&`, on separate lines, or
    /// inside a substitution — and they need not agree about depth. Testing
    /// `shallow_flag` against the whole command string credited every selector
    /// in `ambits show A --no-body && ambits show B` at the shallow depth,
    /// including `B`, which was read in full.
    ///
    /// Splitting on the `requires` marker separates them. The leading segment
    /// is whatever preceded the first invocation and is discarded; a marker
    /// that appears inside a path contributes a segment with no selectors,
    /// which costs nothing.
    pub fn resolve(&self, input: &serde_json::Value) -> Option<Vec<(String, ReadDepth)>> {
        let cmd = input.get(&self.key)?.as_str()?;
        if !cmd.contains(&self.requires) {
            return None;
        }

        let full = ReadDepth::from(self.depth);
        let shallow = self.shallow_depth.map(ReadDepth::from);

        let mut out = Vec::new();
        for segment in cmd.split(&self.requires).skip(1) {
            let depth = match (self.shallow_flag.as_deref(), shallow) {
                (Some(flag), Some(d)) if segment.contains(flag) => d,
                _ => full,
            };
            // `any` consumes through the match, so what remains is exactly
            // the subcommand's own arguments — which is where selectors live,
            // and the only place they can mean "return this definition".
            let mut tokens = segment.split_whitespace();
            if let Some(subcommand) = &self.subcommand {
                if !tokens.any(|t| t == subcommand) {
                    continue;
                }
            }

            for token in tokens {
                let token = token.trim_matches(|c| c == '\'' || c == '"');
                if matches!(
                    crate::lookup::parse_selector(token),
                    crate::lookup::Selector::Id(_) | crate::lookup::Selector::Hash(_)
                ) {
                    out.push((token.to_string(), depth));
                }
            }
        }

        if out.is_empty() {
            return None;
        }
        Some(out)
    }
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
    EmptyPatterns          { tool_name: String },
    MissingDepth           { stanza_index: usize },
    LegacyConfigPath       { path: String, moved_to: String },
    MissingOverride        { path: String },
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
            ConfigWarning::EmptyPatterns { tool_name } =>
                write!(f, "tool '{}': pattern_match depth has an empty 'patterns' list; will always use default",
                    tool_name),
            ConfigWarning::MissingDepth { stanza_index } =>
                write!(f, "tool config stanza {} has no 'depth' and no 'extends'; skipping",
                    stanza_index),
            ConfigWarning::LegacyConfigPath { path, moved_to } =>
                write!(f, "tool config '{}' is in the legacy .ambit/ directory; move it to '{}'",
                    path, moved_to),
            ConfigWarning::MissingOverride { path } =>
                write!(f, "--tools-config '{}' does not exist; using the project and user configs instead",
                    path),
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

    /// Discover user config (CLI override → `project_root`'s → user-global) and
    /// merge with built-ins. Returns `(Arc<config>, warnings)`.
    pub fn resolve(cli: Option<&Path>, project_root: &Path) -> (Arc<Self>, Vec<ConfigWarning>) {
        Self::resolve_with(cli, project_root, Self::user_global_config().as_deref())
    }

    /// [`resolve`](Self::resolve) with the user-global config path given
    /// rather than read from `$HOME`, so tests control every layer.
    fn resolve_with(
        cli: Option<&Path>,
        project_root: &Path,
        global: Option<&Path>,
    ) -> (Arc<Self>, Vec<ConfigWarning>) {
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

        // Each layer merges over everything before it, so a later layer wins
        // stanza by stanza and `[cache]`/`[editor]` field by field.
        let mut merged = builtin;
        for path in Self::config_layers(cli, project_root, global, &mut warnings) {
            let (layer, mut lw) = Self::load(&path);
            warnings.append(&mut lw);
            if let Some(layer) = layer {
                merged = Self::merge(merged, layer, &mut warnings);
            }
        }

        (Arc::new(merged), warnings)
    }

    /// Merge user config over built-in base.
    /// Steps: A=dedup user, B=skip empty names, C=extends inheritance, D=filter+append
    pub fn merge(base: Self, user: Self, warnings: &mut Vec<ConfigWarning>) -> Self {
        // `[cache]` is plain scalar settings, not a mergeable stanza list: the
        // user file simply wins where it says anything. Captured up front
        // because `user` is consumed below.
        let cache = CacheConfig {
            enabled: user.cache.enabled.or(base.cache.enabled),
            flush_interval_ms: user.cache.flush_interval_ms.or(base.cache.flush_interval_ms),
        };
        // `[editor]` follows the same plain-scalar merge as `[cache]`.
        let editor = EditorConfig {
            command: user.editor.command.clone().or(base.editor.command.clone()),
        };
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
                    // Inherited like every other optional field. It was left
                    // out, so a user stanza extending `Bash` to add one command
                    // prefix silently lost `ambits show` crediting — the exact
                    // coverage hole the spec exists to close.
                    if stanza.target_selectors.is_none() {
                        stanza.target_selectors = base_stanza.target_selectors.clone();
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
            cache,
            editor,
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

    /// The config files to merge over the built-ins, lowest precedence first:
    ///
    /// 1. `global` — the user's `~/.config/ambit/tools.toml`
    /// 2. `.ambits/tools.toml` in the project root, else the legacy
    ///    `.ambit/tools.toml` there with a warning
    ///
    /// so a project setting overrides a personal one, and a personal setting
    /// the project says nothing about still applies. `--tools-config` replaces
    /// both, as its help promises; one that does not exist is warned about and
    /// the normal layers are used instead, rather than silently ignoring a
    /// file the user named.
    ///
    /// The project root, not the working directory: the config belongs to the
    /// project, and `ambits -p ../other` or a run from `src/` must pick up the
    /// same file as a run from the root.
    fn config_layers(
        cli: Option<&Path>,
        project_root: &Path,
        global: Option<&Path>,
        warnings: &mut Vec<ConfigWarning>,
    ) -> Vec<PathBuf> {
        if let Some(p) = cli {
            if p.exists() {
                return vec![p.to_path_buf()];
            }
            warnings.push(ConfigWarning::MissingOverride {
                path: p.display().to_string(),
            });
        }

        let mut layers: Vec<PathBuf> = global.map(Path::to_path_buf).into_iter().collect();

        let local = project_root.join(crate::state_dir::STATE_DIR).join("tools.toml");
        let legacy = project_root.join(crate::state_dir::LEGACY_STATE_DIR).join("tools.toml");
        if local.exists() {
            layers.push(local);
        } else if legacy.exists() {
            warnings.push(ConfigWarning::LegacyConfigPath {
                path: legacy.display().to_string(),
                moved_to: local.display().to_string(),
            });
            layers.push(legacy);
        }

        layers
    }

    /// `~/.config/ambit/tools.toml`, when it exists.
    fn user_global_config() -> Option<PathBuf> {
        let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))?;
        let global = PathBuf::from(home).join(".config/ambit/tools.toml");
        global.exists().then_some(global)
    }

    /// Empty config — fallback when built-in fails to parse.
    fn empty() -> Self {
        Self {
            version: Self::SUPPORTED_VERSION,
            tools: vec![],
            cache: CacheConfig::default(),
            editor: EditorConfig::default(),
            index: HashMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// ToolCallMapper impl
// ---------------------------------------------------------------------------

impl super::ToolCallMapper for ToolMappingConfig {
    fn map_tool_call(
        &self,
        tool_name: &str,
        input: &serde_json::Value,
        agent_id: &str,
        timestamp_str: &str,
    ) -> Option<super::AgentToolCall> {
        crate::ingest::claude::map_tool_call(self, tool_name, input, agent_id, timestamp_str)
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

    fn bash_input(cmd: &str) -> serde_json::Value {
        serde_json::json!({ "command": cmd })
    }

    fn selector_spec(cfg: &ToolMappingConfig) -> &TargetSelectorSpec {
        let idx = cfg.index["Bash"];
        cfg.tools[idx]
            .target_selectors
            .as_ref()
            .expect("Bash carries a selector spec")
    }

    #[test]
    fn show_command_yields_its_selectors_at_full_body() {
        let cfg = ToolMappingConfig::builtin().unwrap();
        let got = selector_spec(&cfg)
            .resolve(&bash_input(
                "ambits -p . show 'src/filter.rs::PathFilter/matches' 6e42b7a3",
            ))
            .expect("selectors found");
        assert_eq!(
            got,
            vec![
                ("src/filter.rs::PathFilter/matches".to_string(), ReadDepth::FullBody),
                ("6e42b7a3".to_string(), ReadDepth::FullBody),
            ]
        );
    }

    /// `--no-body` returns location metadata only, so it earns a shallower
    /// depth than a call that actually printed the source.
    #[test]
    fn a_metadata_only_lookup_earns_a_shallower_depth() {
        let cfg = ToolMappingConfig::builtin().unwrap();
        let got = selector_spec(&cfg)
            .resolve(&bash_input("ambits -p . show 'a.rs::x' --no-body"))
            .unwrap();
        assert_eq!(got, vec![("a.rs::x".to_string(), ReadDepth::NameOnly)]);
    }

    /// Flags are excluded by the selector grammar rather than by listing them,
    /// so the spec does not rot when the command grows an option.
    #[test]
    fn flags_and_their_values_are_not_mistaken_for_selectors() {
        let cfg = ToolMappingConfig::builtin().unwrap();
        let got = selector_spec(&cfg)
            .resolve(&bash_input("ambits show a.rs::x --max-bytes 4000 --no-body"))
            .unwrap();
        assert_eq!(got, vec![("a.rs::x".to_string(), ReadDepth::NameOnly)]);
    }

    /// The bug this splitting exists for: one shell command holding two
    /// invocations, only one of which asked for metadata. Testing the flag
    /// against the whole command credited `B` at the shallow depth too, even
    /// though it was read in full.
    #[test]
    fn each_invocation_gets_its_own_depth() {
        let cfg = ToolMappingConfig::builtin().unwrap();
        let got = selector_spec(&cfg)
            .resolve(&bash_input(
                "ambits show a.rs::shallow --no-body && ambits show b.rs::deep",
            ))
            .unwrap();
        assert_eq!(
            got,
            vec![
                ("a.rs::shallow".to_string(), ReadDepth::NameOnly),
                ("b.rs::deep".to_string(), ReadDepth::FullBody),
            ]
        );
    }

    /// The marker appearing in a path must not create a phantom invocation.
    #[test]
    fn a_marker_inside_a_path_contributes_nothing() {
        let cfg = ToolMappingConfig::builtin().unwrap();
        let got = selector_spec(&cfg)
            .resolve(&bash_input("./target/debug/ambits -p . show a.rs::x"))
            .unwrap();
        assert_eq!(got, vec![("a.rs::x".to_string(), ReadDepth::FullBody)]);
    }

    /// The marker keeps an unrelated command that merely mentions a symbol id
    /// from registering a read.
    #[test]
    fn a_command_without_the_marker_yields_nothing() {
        let cfg = ToolMappingConfig::builtin().unwrap();
        assert!(selector_spec(&cfg)
            .resolve(&bash_input("grep -n 'src/app.rs::App' notes.txt"))
            .is_none());
    }

    /// The regression this `subcommand` field exists for. `find`'s pattern is a
    /// regex over file *content*, so a search for text that happens to look
    /// like a symbol id is not a request for that symbol — and crediting it
    /// would claim a read of something the search may never have displayed.
    /// `find` journals exactly what it showed; it needs no help from here.
    #[test]
    fn a_search_pattern_that_looks_like_an_id_is_not_credited() {
        let cfg = ToolMappingConfig::builtin().unwrap();
        assert!(selector_spec(&cfg)
            .resolve(&bash_input("ambits -p . rg 'src/app.rs::App'"))
            .is_none());
    }

    /// …and the same command run for real: a search and a lookup chained
    /// together credit only the lookup.
    #[test]
    fn a_search_beside_a_lookup_credits_only_the_lookup() {
        let cfg = ToolMappingConfig::builtin().unwrap();
        let got = selector_spec(&cfg)
            .resolve(&bash_input(
                "ambits -p . rg 'a.rs::pattern' && ambits -p . show b.rs::real",
            ))
            .unwrap();
        assert_eq!(got, vec![("b.rs::real".to_string(), ReadDepth::FullBody)]);
    }

    /// Every other subcommand is excluded by the same rule, without listing
    /// them: `callers` takes a bare name, but nothing stops one being written
    /// as a path-qualified string.
    #[test]
    fn other_subcommands_credit_nothing() {
        let cfg = ToolMappingConfig::builtin().unwrap();
        for cmd in [
            "ambits -p . callers 'src/app.rs::App'",
            "ambits -p . cache clear --session src/app.rs::App",
            "ambits -p . --dump src/app.rs::App",
        ] {
            assert!(
                selector_spec(&cfg).resolve(&bash_input(cmd)).is_none(),
                "{cmd} does not print a definition, so it credits nothing"
            );
        }
    }

    #[test]
    fn a_marker_with_no_selectors_yields_nothing() {
        let cfg = ToolMappingConfig::builtin().unwrap();
        assert!(selector_spec(&cfg)
            .resolve(&bash_input("ambits -p . --coverage"))
            .is_none());
    }

    // -----------------------------------------------------------------------
    // 1. builtin_config_parses
    // -----------------------------------------------------------------------
    #[test]
    fn builtin_config_parses() {
        let cfg = ToolMappingConfig::builtin().expect("built-in config must parse");
        assert_eq!(cfg.tools.len(), 22);
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
            "Bash", "TodoWrite",
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

    /// Extending `Bash` to add one command prefix must not cost the selector
    /// spec: without inheritance, a user config that customises Bash silently
    /// stopped crediting `ambits show`.
    #[test]
    fn merge_extends_inherits_target_selectors() {
        let user = r#"
version = 1
[[tool]]
names   = ["Bash"]
extends = "Bash"
path_keys = []
description = "custom bash"
"#;
        let mut warnings = Vec::new();
        let user_cfg: ToolMappingConfig = toml::from_str(user).unwrap();
        let merged = ToolMappingConfig::merge(
            ToolMappingConfig::builtin().unwrap(),
            user_cfg,
            &mut warnings,
        );

        let mapping = &merged.tools[merged.index["Bash"]];
        let spec = mapping
            .target_selectors
            .as_ref()
            .expect("the selector spec survives an extends");
        assert_eq!(spec.subcommand.as_deref(), Some("show"));
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

    #[test]
    fn conditional_depth_with_body_true() {
        let spec = DepthSpec::Conditional {
            condition_key: "include_body".into(),
            if_true: ReadDepthDe::FullBody,
            if_false: ReadDepthDe::Signature,
            default: ReadDepthDe::Signature,
        };
        let input = serde_json::json!({ "include_body": true });
        assert_eq!(spec.resolve(&input), ReadDepth::FullBody);
    }

    #[test]
    fn conditional_depth_with_body_false() {
        let spec = DepthSpec::Conditional {
            condition_key: "include_body".into(),
            if_true: ReadDepthDe::FullBody,
            if_false: ReadDepthDe::Signature,
            default: ReadDepthDe::Signature,
        };
        let input = serde_json::json!({ "include_body": false });
        assert_eq!(spec.resolve(&input), ReadDepth::Signature);
    }

    #[test]
    fn conditional_depth_absent_key() {
        let spec = DepthSpec::Conditional {
            condition_key: "include_body".into(),
            if_true: ReadDepthDe::FullBody,
            if_false: ReadDepthDe::Signature,
            default: ReadDepthDe::Signature,
        };
        let input = serde_json::json!({ "name_path_pattern": "Foo" }); // no include_body
        assert_eq!(spec.resolve(&input), ReadDepth::Signature);
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
    /// Write `body` (prefixed with `version = 1`) to `dir/rel`.
    fn write_config(dir: &Path, rel: &str, body: &str) -> PathBuf {
        let path = dir.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, format!("version = 1\n{body}")).unwrap();
        path
    }

    /// A minimal valid stanza for a tool named `name`, at `depth`.
    fn stanza(name: &str, depth: &str) -> String {
        format!(
            "[[tool]]\nnames = [\"{name}\"]\npath_keys = [\"path\"]\n\
             depth = {{ type = \"fixed\", value = \"{depth}\" }}\ndescription = \"{name}\"\n"
        )
    }

    fn tool<'c>(cfg: &'c ToolMappingConfig, name: &str) -> Option<&'c ToolMapping> {
        cfg.tools.iter().find(|t| t.names.iter().any(|n| n == name))
    }

    /// A `--tools-config` that does not exist is warned about — it used to be
    /// skipped silently, so a typo quietly ran on other settings — and the
    /// normal layers apply instead (here: none, so the built-ins).
    #[test]
    fn a_missing_override_warns_and_falls_back() {
        let root = tempfile::tempdir().unwrap();
        let (cfg, warnings) = ToolMappingConfig::resolve_with(
            Some(Path::new("/nonexistent/tools.toml")),
            root.path(),
            None,
        );
        assert_eq!(cfg.tools.len(), 22, "should have 22 built-in tools");
        assert!(
            matches!(warnings.as_slice(), [ConfigWarning::MissingOverride { path }] if path == "/nonexistent/tools.toml"),
            "{warnings:?}"
        );
    }

    /// The project's config is found under the project root, whatever the
    /// working directory — `cargo test` runs from the crate root, which is
    /// not this temp dir.
    #[test]
    fn the_project_config_is_found_under_the_project_root() {
        let root = tempfile::tempdir().unwrap();
        write_config(root.path(), ".ambits/tools.toml", "[editor]\ncommand = \"from-project-root\"\n");

        let (cfg, _) = ToolMappingConfig::resolve_with(None, root.path(), None);
        assert_eq!(cfg.editor.command.as_deref(), Some("from-project-root"));
    }

    #[test]
    fn a_legacy_config_under_the_project_root_is_read_with_a_warning() {
        let root = tempfile::tempdir().unwrap();
        write_config(root.path(), ".ambit/tools.toml", "[editor]\ncommand = \"from-legacy\"\n");

        let (cfg, warnings) = ToolMappingConfig::resolve_with(None, root.path(), None);
        assert_eq!(cfg.editor.command.as_deref(), Some("from-legacy"));
        assert!(
            warnings.iter().any(|w| matches!(w, ConfigWarning::LegacyConfigPath { .. })),
            "{warnings:?}"
        );
    }

    /// User-global and project configs layer rather than one hiding the
    /// other: a personal setting the project says nothing about still applies.
    /// Before, the first file found won outright, so any project config
    /// silently dropped every user-global setting.
    #[test]
    fn global_and_project_configs_layer() {
        let home = tempfile::tempdir().unwrap();
        let root = tempfile::tempdir().unwrap();
        let global = write_config(
            home.path(),
            "tools.toml",
            &format!("[editor]\ncommand = \"global-editor\"\n[cache]\nflush_interval_ms = 1234\n{}",
                stanza("GlobalTool", "Overview")),
        );
        write_config(
            root.path(),
            ".ambits/tools.toml",
            &format!("[editor]\ncommand = \"project-editor\"\n{}", stanza("ProjectTool", "FullBody")),
        );

        let (cfg, warnings) = ToolMappingConfig::resolve_with(None, root.path(), Some(&global));
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(cfg.editor.command.as_deref(), Some("project-editor"), "project wins where both speak");
        assert_eq!(cfg.cache.flush_interval_ms, Some(1234), "global applies where the project is silent");
        assert!(tool(&cfg, "GlobalTool").is_some(), "global stanzas survive");
        assert!(tool(&cfg, "ProjectTool").is_some(), "project stanzas are added");
        assert!(tool(&cfg, "Read").is_some(), "built-ins remain underneath");
    }

    /// A stanza for the same tool in both layers: the project's replaces the
    /// global's, exactly as either replaces a built-in.
    #[test]
    fn a_project_stanza_overrides_the_global_one() {
        let home = tempfile::tempdir().unwrap();
        let root = tempfile::tempdir().unwrap();
        let global = write_config(home.path(), "tools.toml", &stanza("Shared", "Overview"));
        write_config(root.path(), ".ambits/tools.toml", &stanza("Shared", "FullBody"));

        let (cfg, _) = ToolMappingConfig::resolve_with(None, root.path(), Some(&global));
        let shared: Vec<_> = cfg.tools.iter().filter(|t| t.names.iter().any(|n| n == "Shared")).collect();
        assert_eq!(shared.len(), 1, "one stanza per name");
        assert!(
            matches!(shared[0].depth, Some(DepthSpec::Fixed { value: ReadDepthDe::FullBody })),
            "the project's stanza: {:?}",
            shared[0].depth
        );
    }

    /// `--tools-config` replaces both layers, as its help says.
    #[test]
    fn an_existing_override_replaces_both_layers() {
        let home = tempfile::tempdir().unwrap();
        let root = tempfile::tempdir().unwrap();
        let global = write_config(home.path(), "tools.toml", "[editor]\ncommand = \"global-editor\"\n");
        write_config(root.path(), ".ambits/tools.toml", &stanza("ProjectTool", "FullBody"));
        let over = write_config(home.path(), "override.toml", "[cache]\nflush_interval_ms = 42\n");

        let (cfg, _) = ToolMappingConfig::resolve_with(Some(&over), root.path(), Some(&global));
        assert_eq!(cfg.cache.flush_interval_ms, Some(42));
        assert_eq!(cfg.editor.command, None, "the global layer does not apply");
        assert!(tool(&cfg, "ProjectTool").is_none(), "nor does the project layer");
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
    // 23. bash_tool_has_pattern_match_depth
    // -----------------------------------------------------------------------
    #[test]
    fn bash_tool_has_pattern_match_depth() {
        let cfg = ToolMappingConfig::builtin().unwrap();
        let idx = *cfg.index.get("Bash").unwrap();
        assert!(
            matches!(cfg.tools[idx].depth, Some(DepthSpec::PatternMatch { .. })),
            "Bash must use pattern_match depth"
        );
    }

    /// `ambits grep`/`ambits rg` do the same regex-over-content search as bare
    /// `grep`/`rg`, just through ambit's own subcommands — they must not fall
    /// through to the pattern_match default, or an agent using ambit's search
    /// would be credited *less* than one bypassing it with raw grep/rg.
    ///
    /// The `-p .` case is the regression this test guards: a live transcript
    /// capture showed Claude Code actually invoking `ambits -p . grep ...`,
    /// which a literal `contains("ambits grep")` check does not match because
    /// the global flag sits between the binary and the subcommand.
    #[test]
    fn ambits_grep_and_rg_earn_overview_depth() {
        let cfg = ToolMappingConfig::builtin().unwrap();
        let idx = *cfg.index.get("Bash").unwrap();
        let depth = cfg.tools[idx].depth.as_ref().unwrap();
        for cmd in [
            "ambits grep 'foo' -p .",
            "ambits rg 'foo' --max-bytes 100",
            "ambits -p . grep \"fn dirs_home\" src/ingest/claude.rs 2>&1",
            "cd /repo && ambits -p . rg 'fn dirs_home' src/ingest/claude.rs",
        ] {
            assert_eq!(
                depth.resolve(&bash_input(cmd)),
                ReadDepth::Overview,
                "{cmd} should earn Overview depth"
            );
        }
    }

    // -----------------------------------------------------------------------
    // 24. command_pattern_default_match_type_is_prefix
    // -----------------------------------------------------------------------
    #[test]
    fn command_pattern_default_match_type_is_prefix() {
        // Omitting match_type must deserialize as Prefix (backward compat).
        let toml = r#"
version = 1
[[tool]]
names         = ["T"]
path_keys     = []
path_required = false
pattern_keys  = ["command"]
depth         = { type = "pattern_match", key = "command", default = "NameOnly", patterns = [
    { prefix = "cat ", depth = "FullBody" },
] }
description = "T {pattern}"
"#;
        let cfg: ToolMappingConfig = toml::from_str(toml).unwrap();
        let mapping = cfg.tools.first().unwrap();
        let patterns = match mapping.depth.as_ref().unwrap() {
            DepthSpec::PatternMatch { patterns, .. } => patterns,
            _ => panic!("expected PatternMatch"),
        };
        assert!(matches!(patterns[0].match_type, MatchType::Prefix));
    }

    // -----------------------------------------------------------------------
    // 25. command_pattern_contains_match_type_deserializes
    // -----------------------------------------------------------------------
    #[test]
    fn command_pattern_contains_match_type_deserializes() {
        let toml = r#"
version = 1
[[tool]]
names         = ["T"]
path_keys     = []
path_required = false
pattern_keys  = ["command"]
depth         = { type = "pattern_match", key = "command", default = "NameOnly", patterns = [
    { prefix = "| grep", match_type = "contains", depth = "Overview" },
] }
description = "T {pattern}"
"#;
        let cfg: ToolMappingConfig = toml::from_str(toml).unwrap();
        let mapping = cfg.tools.first().unwrap();
        let patterns = match mapping.depth.as_ref().unwrap() {
            DepthSpec::PatternMatch { patterns, .. } => patterns,
            _ => panic!("expected PatternMatch"),
        };
        assert!(matches!(patterns[0].match_type, MatchType::Contains));
    }

    // -----------------------------------------------------------------------
    // 26. command_pattern_exact_match_type_deserializes
    // -----------------------------------------------------------------------
    #[test]
    fn command_pattern_exact_match_type_deserializes() {
        let toml = r#"
version = 1
[[tool]]
names         = ["T"]
path_keys     = []
path_required = false
pattern_keys  = ["command"]
depth         = { type = "pattern_match", key = "command", default = "NameOnly", patterns = [
    { prefix = "cargo test", match_type = "exact", depth = "NameOnly" },
] }
description = "T {pattern}"
"#;
        let cfg: ToolMappingConfig = toml::from_str(toml).unwrap();
        let mapping = cfg.tools.first().unwrap();
        let patterns = match mapping.depth.as_ref().unwrap() {
            DepthSpec::PatternMatch { patterns, .. } => patterns,
            _ => panic!("expected PatternMatch"),
        };
        assert!(matches!(patterns[0].match_type, MatchType::Exact));
    }

    // -----------------------------------------------------------------------
    // 27. command_pattern_ambits_subcommand_match_type_deserializes
    // -----------------------------------------------------------------------
    #[test]
    fn command_pattern_ambits_subcommand_match_type_deserializes() {
        let toml = r#"
version = 1
[[tool]]
names         = ["T"]
path_keys     = []
path_required = false
pattern_keys  = ["command"]
depth         = { type = "pattern_match", key = "command", default = "NameOnly", patterns = [
    { prefix = "grep", match_type = "ambits_subcommand", depth = "Overview" },
] }
description = "T {pattern}"
"#;
        let cfg: ToolMappingConfig = toml::from_str(toml).unwrap();
        let mapping = cfg.tools.first().unwrap();
        let patterns = match mapping.depth.as_ref().unwrap() {
            DepthSpec::PatternMatch { patterns, .. } => patterns,
            _ => panic!("expected PatternMatch"),
        };
        assert!(matches!(patterns[0].match_type, MatchType::AmbitsSubcommand));
    }

    /// A command that merely mentions "ambits" without the subcommand as its
    /// own token must not match — otherwise a path or description containing
    /// the word would falsely earn credit.
    #[test]
    fn ambits_subcommand_requires_both_marker_and_token() {
        let cfg = ToolMappingConfig::builtin().unwrap();
        let idx = *cfg.index.get("Bash").unwrap();
        let depth = cfg.tools[idx].depth.as_ref().unwrap();
        assert_eq!(
            depth.resolve(&bash_input("echo 'no ambits binary here' && rgrep foo")),
            ReadDepth::NameOnly,
            "no literal 'ambits' token, and 'rgrep' is not the 'rg' token, so this must not match"
        );
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
