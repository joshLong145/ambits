use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::Value;
use crate::tracking::ReadDepth;
use super::{AgentToolCall, EventTailer, SessionIngester, TailerOutput, ToolCallMapper};
use super::tool_config::ToolMappingConfig;

/// Derive the Claude Code log directory for a given project path.
/// Claude stores logs at ~/.claude/projects/<slug>/ where slug is the
/// absolute path with `/` replaced by `-` and leading `-`.
pub fn log_dir_for_project(project_path: &Path) -> Option<PathBuf> {
    let canonical = project_path.canonicalize().ok()?;
    let slug = canonical
        .to_string_lossy()
        .replace('/', "-")
        .replace('.', "-");  // Claude Code also replaces dots with hyphens
    let home = dirs_home()?;
    let dir = home.join(".claude").join("projects").join(&slug);
    if dir.is_dir() {
        Some(dir)
    } else {
        None
    }
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

/// Find the most recent session ID by scanning for UUID-named .jsonl files
/// and selecting the one with the most recent modification time.
pub fn find_latest_session(log_dir: &Path) -> Option<String> {
    find_session_from_files(log_dir)
}

/// Find the latest session by scanning for UUID-named .jsonl files.
fn find_session_from_files(log_dir: &Path) -> Option<String> {
    let entries = fs::read_dir(log_dir).ok()?;

    entries
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            let name = path.file_name()?.to_str()?;
            // Match UUID pattern: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.jsonl
            if !name.ends_with(".jsonl") {
                return None;
            }
            let stem = name.strip_suffix(".jsonl")?;
            if !is_uuid(stem) {
                return None;
            }
            // Skip empty files.
            let meta = fs::metadata(&path).ok()?;
            if meta.len() == 0 {
                return None;
            }
            let mtime = meta.modified().ok()?;
            Some((stem.to_string(), mtime))
        })
        .max_by_key(|(_, mtime)| *mtime)
        .map(|(session_id, _)| session_id)
}

/// Check if a string looks like a UUID (8-4-4-4-12 hex chars, exactly 36 bytes).
/// Uses direct byte-position checks — no allocation.
fn is_uuid(s: &str) -> bool {
    let b = s.as_bytes();
    if b.len() != 36 {
        return false;
    }
    // Dashes must sit at positions 8, 13, 18, 23.
    if b[8] != b'-' || b[13] != b'-' || b[18] != b'-' || b[23] != b'-' {
        return false;
    }
    b.iter().enumerate().all(|(i, &c)| {
        matches!(i, 8 | 13 | 18 | 23) || c.is_ascii_hexdigit()
    })
}

/// List all JSONL files in the log directory that belong to a session
/// (the main session file + any agent-*.jsonl files that reference it).
/// Supports both old format (agent files flat in log dir) and new format
/// (agent files in `<session-id>/subagents/`).
pub fn session_log_files(log_dir: &Path, session_id: &str) -> Vec<PathBuf> {
    let mut files = Vec::new();

    // Main session file.
    let main_file = log_dir.join(format!("{session_id}.jsonl"));
    if main_file.exists() {
        files.push(main_file);
    }

    // New format: <log_dir>/<session-id>/subagents/agent-*.jsonl
    // All files in this directory belong to the session by definition.
    let subagents_dir = log_dir.join(session_id).join("subagents");
    if subagents_dir.is_dir() {
        if let Ok(entries) = fs::read_dir(&subagents_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("");
                if name.starts_with("agent-") && name.ends_with(".jsonl") {
                    files.push(path);
                }
            }
        }
    }

    // Old format: <log_dir>/agent-*.jsonl (check sessionId in first lines).
    if let Ok(entries) = fs::read_dir(log_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");
            if name.starts_with("agent-")
                && name.ends_with(".jsonl")
                && agent_belongs_to_session(&path, session_id)
            {
                files.push(path);
            }
        }
    }

    files
}

fn agent_belongs_to_session(path: &Path, session_id: &str) -> bool {
    let file = match fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return false,
    };
    let reader = BufReader::new(file);
    // Check first few lines for the sessionId.
    for line in reader.lines().take(3).map_while(Result::ok) {
        if let Ok(obj) = serde_json::from_str::<Value>(&line) {
            if let Some(sid) = obj.get("sessionId").and_then(|v| v.as_str()) {
                return sid == session_id;
            }
        }
    }
    false
}

/// Extract a human-readable label for an agent from its JSONL log file.
///
/// Reads the first line of the file and looks for:
/// 1. The `message.content` field (the task prompt given to the agent) — truncated to 50 chars
/// 2. Falls back to the filename stem (e.g., `agent-a9fe23c` → `a9fe23c`)
pub fn extract_agent_label(path: &Path) -> String {
    let fallback = path
        .file_stem()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .strip_prefix("agent-")
        .unwrap_or_else(|| {
            path.file_stem()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
        })
        .to_string();

    let file = match fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return fallback,
    };
    let mut reader = BufReader::new(file);
    let mut first_line = String::new();
    if reader.read_line(&mut first_line).is_err() {
        return fallback;
    }

    let obj: Value = match serde_json::from_str(first_line.trim()) {
        Ok(v) => v,
        Err(_) => return fallback,
    };

    // Try to get the task prompt from the first user message content.
    let content = obj
        .pointer("/message/content")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    if content.is_empty() {
        return fallback;
    }

    // Truncate to a short label: first line, max 50 chars.
    let first_line_content = content.lines().next().unwrap_or(content);
    if first_line_content.len() <= 50 {
        first_line_content.to_string()
    } else {
        format!("{}...", &first_line_content[..47])
    }
}

/// The result of parsing a single JSONL line.
pub enum ParsedLine {
    Events(Vec<AgentToolCall>),
    SessionCleared,
    Ignored,
}

/// Parse all events from a JSONL log file.
/// SessionCleared signals are ignored in batch mode — dump/coverage modes show
/// aggregate historical coverage, not live session state.
pub fn parse_log_file(path: &Path, config: &ToolMappingConfig) -> Vec<AgentToolCall> {
    parse_log_file_with_mapper(path, config)
}

fn parse_log_file_with_mapper(path: &Path, mapper: &dyn ToolCallMapper) -> Vec<AgentToolCall> {
    let mut events = Vec::new();
    let file = match fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return events,
    };
    let reader = BufReader::new(file);

    // Derive a default agent ID from the filename.
    let default_id = path
        .file_stem()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();

    // Extract a human-readable label for this agent (Arc so each clone is free).
    let label: Arc<str> = extract_agent_label(path).into();

    for line in reader.lines().map_while(Result::ok) {
        if let ParsedLine::Events(mut line_events) = parse_jsonl_line(&line, &default_id, mapper) {
            for ev in &mut line_events {
                ev.label = label.clone();
            }
            events.extend(line_events);
        }
    }
    events
}

/// Parse a single JSONL line from a Claude Code session log.
/// Returns a `ParsedLine` indicating tool call events, a session clear signal, or nothing.
pub fn parse_jsonl_line(line: &str, default_agent_id: &str, mapper: &dyn ToolCallMapper) -> ParsedLine {
    let obj: Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(_) => return ParsedLine::Ignored,
    };

    let msg_type = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");

    // Detect /clear command in user messages.
    if msg_type == "user" {
        let content = obj
            .pointer("/message/content")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if content.contains("<command-name>/clear</command-name>") {
            return ParsedLine::SessionCleared;
        }
        return ParsedLine::Ignored;
    }

    if msg_type != "assistant" {
        return ParsedLine::Ignored;
    }

    // Intern agent_id once per line so per-event clones are free.
    let agent_id: Arc<str> = Arc::from(
        obj.get("agentId")
            .or_else(|| obj.get("sessionId"))
            .and_then(|v| v.as_str())
            .unwrap_or(default_agent_id),
    );

    let timestamp_str = obj
        .get("timestamp")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let content = match obj.pointer("/message/content") {
        Some(Value::Array(arr)) => arr,
        _ => return ParsedLine::Ignored,
    };

    let mut events = Vec::new();
    for block in content {
        if block.get("type").and_then(|v| v.as_str()) != Some("tool_use") {
            continue;
        }

        let tool_name = match block.get("name").and_then(|v| v.as_str()) {
            Some(n) => n,
            None => continue,
        };

        let input = block.get("input").cloned().unwrap_or(Value::Null);

        let event = mapper.map_tool_call(tool_name, &input, &agent_id, &timestamp_str)
            .unwrap_or_else(|| AgentToolCall {
                agent_id: agent_id.clone(),
                tool_name: Arc::from(tool_name),
                file_path: None,
                read_depth: ReadDepth::Unseen,
                description: format!("{tool_name} (untracked)"),
                timestamp_str: timestamp_str.clone(),
                target_symbol: None,
                target_lines: None,
                label: agent_id.clone(),
            });
        events.push(event);
    }

    ParsedLine::Events(events)
}

/// Render a description template by substituting `{key}`, `{key|short}`, and `{key|cmd}`
/// placeholders. Uses `str::find` for scanning — correct UTF-8, no per-byte dispatch.
///
/// `display_override`: when `Some((key, value))`, that key resolves to `value` before
/// falling back to `input.get(key)`. Used to inject the resolved pattern/display value
/// without cloning the entire input `Value`.
///
/// Modifiers:
/// - `|short` — passes value through `short_path_fn` (last 2 path components)
/// - `|cmd`   — truncates to first 200 chars, appending "…" if longer
fn render_description(
    template: &str,
    input: &Value,
    display_override: Option<(&str, &str)>,
    short_path_fn: impl Fn(&str) -> String,
) -> String {
    let mut out = String::with_capacity(template.len() + 16);
    let mut rest = template;
    loop {
        let Some(brace) = rest.find('{') else {
            out.push_str(rest);
            break;
        };
        out.push_str(&rest[..brace]);
        rest = &rest[brace + 1..]; // advance past '{'

        let Some(close) = rest.find('}') else {
            // Unclosed brace — emit literal '{' then the rest verbatim.
            out.push('{');
            out.push_str(rest);
            break;
        };
        let inner = &rest[..close];

        // Split on `|` to detect modifiers.
        let (key, modifier) = if let Some(k) = inner.strip_suffix("|short") {
            (k, "short")
        } else if let Some(k) = inner.strip_suffix("|cmd") {
            (k, "cmd")
        } else {
            (inner, "")
        };

        // Validate key: [a-zA-Z_][a-zA-Z0-9_]*
        let valid = !key.is_empty() && {
            let mut chars = key.chars();
            let first = chars.next().unwrap();
            (first.is_ascii_alphabetic() || first == '_')
                && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
        };
        if !valid {
            // Not a valid placeholder — emit literal '{' and re-scan from `inner`.
            // (`rest` already points to `inner`, so the loop naturally re-scans it.)
            out.push('{');
            continue;
        }

        // Resolve value: check override first, then input JSON.
        let value = display_override
            .and_then(|(ok, ov)| if ok == key { Some(ov) } else { None })
            .or_else(|| input.get(key).and_then(|v| v.as_str()))
            .unwrap_or("?");

        match modifier {
            "short" => out.push_str(&short_path_fn(value)),
            "cmd" => {
                const CMD_MAX: usize = 200;
                if value.len() <= CMD_MAX {
                    out.push_str(value);
                } else {
                    // Truncate at a char boundary within the limit.
                    let truncated = value.char_indices()
                        .take_while(|(i, _)| *i < CMD_MAX)
                        .last()
                        .map(|(i, c)| &value[..i + c.len_utf8()])
                        .unwrap_or(&value[..CMD_MAX]);
                    out.push_str(truncated);
                    out.push('…');
                }
            }
            _ => out.push_str(value),
        }
        rest = &rest[close + 1..]; // advance past '}'
    }
    out
}

/// Map a tool call to an `AgentToolCall` using config-driven dispatch.
pub fn map_tool_call(
    config: &ToolMappingConfig,
    tool_name: &str,
    input: &Value,
    agent_id: &str,
    timestamp_str: &str,
) -> Option<AgentToolCall> {
    // O(1) lookup via pre-built index.
    let &idx = config.index.get(tool_name)?;
    let mapping = &config.tools[idx];

    // Extract file_path from the first matching path_key.
    let file_path_str: Option<&str> = mapping
        .path_keys
        .iter()
        .find_map(|k| input.get(k).and_then(|v| v.as_str()));

    let file_path: Option<PathBuf> = match file_path_str {
        Some(s) => Some(PathBuf::from(s)),
        None if mapping.path_required => return None,
        None => None,
    };

    // Resolve pattern from pattern_keys (first hit).
    let pattern_val: Option<&str> = mapping
        .pattern_keys
        .iter()
        .find_map(|k| input.get(k).and_then(|v| v.as_str()));

    // Resolve read depth via centralised DepthSpec logic.
    let read_depth = mapping
        .depth
        .as_ref()
        .expect("depth must be Some after load/merge — MissingDepth stanzas are dropped")
        .resolve(input);

    // For TodoWrite-style tools: if no pattern_keys matched, try to extract the
    // first todo item's "content" field as a display label.
    let first_todo_content: Option<&str> = if pattern_val.is_none() {
        input.get("todos")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|t| t.get("content"))
            .and_then(|c| c.as_str())
    } else {
        None
    };

    // Build the display value injected under the canonical "pattern" key:
    //   1. Resolved pattern_keys value (e.g. "command", "file_mask")
    //   2. First todo content (TodoWrite)
    //   3. file_path_str fallback (e.g. list_dir with no pattern key)
    let display_val: Option<&str> = pattern_val
        .or(first_todo_content)
        .or(file_path_str);

    let description = render_description(
        &mapping.description,
        input,
        display_val.map(|dv| ("pattern", dv)),
        short_path,
    );

    // Extract target_symbol.
    let target_symbol = mapping
        .target_symbol
        .as_ref()
        .and_then(|spec| input.get(&spec.key)?.as_str().map(String::from));

    // Extract target_lines.
    let target_lines = mapping.target_lines.as_ref().and_then(|spec| {
        let offset = input.get(&spec.offset_key)?.as_u64()? as usize;
        let limit  = input.get(&spec.limit_key)?.as_u64()? as usize;
        Some(offset..offset + limit)
    });

    // Intern agent_id once and share between agent_id and label fields.
    let agent_arc: Arc<str> = Arc::from(agent_id);

    Some(AgentToolCall {
        agent_id: agent_arc.clone(),
        tool_name: Arc::from(tool_name),
        file_path,
        read_depth,
        description,
        timestamp_str: timestamp_str.to_string(),
        target_symbol,
        target_lines,
        label: agent_arc,
    })
}

/// Shorten a file path for display: returns the last 2 path components joined by `/`.
/// Uses `str::rfind` — no Vec allocation.
fn short_path(path: &str) -> String {
    let trimmed = path.trim_end_matches('/');
    let Some(last_sep) = trimmed.rfind('/') else {
        return trimmed.to_string();
    };
    let last = &trimmed[last_sep + 1..];
    let before = &trimmed[..last_sep];
    let second = match before.rfind('/') {
        Some(p) => &before[p + 1..],
        None    => before,
    };
    format!("{second}/{last}")
}

/// Incrementally tails a set of JSONL log files, tracking read positions.
pub struct LogTailer {
    files: Vec<PathBuf>,
    positions: std::collections::HashMap<PathBuf, u64>,
    mapper: Arc<dyn ToolCallMapper>,
}

impl LogTailer {
    /// Create a tailer for the given log files, starting from the end of each
    /// (i.e., only new lines will be read on subsequent calls).
    pub fn new(files: Vec<PathBuf>, mapper: Arc<dyn ToolCallMapper>) -> Self {
        let mut positions = std::collections::HashMap::new();
        for f in &files {
            // Start at the current end of file so we only get new events.
            if let Ok(meta) = fs::metadata(f) {
                positions.insert(f.clone(), meta.len());
            }
        }
        Self { files, positions, mapper }
    }



    /// Add a new file to tail (e.g., a newly created agent log).
    pub fn add_file(&mut self, path: PathBuf) {
        if !self.positions.contains_key(&path) {
            self.positions.insert(path.clone(), 0);
            self.files.push(path);
        }
    }

    /// Read new lines from all tracked files since last read.
    /// Returns a `TailerOutput` with any new agent tool call events and a
    /// `session_cleared` flag set to `true` if a `/clear` command was detected.
    pub fn read_new_events(&mut self) -> TailerOutput {
        let mut output = TailerOutput {
            events: Vec::new(),
            session_cleared: false,
        };

        // Index-based loop so we can update `positions` via `get_mut` without
        // cloning the `PathBuf` key on every iteration.
        for i in 0..self.files.len() {
            let pos = self.positions.get(&self.files[i]).copied().unwrap_or(0);
            let current_len = fs::metadata(&self.files[i])
                .map(|m| m.len())
                .unwrap_or(0);

            if current_len <= pos {
                continue;
            }

            let default_id = self.files[i]
                .file_stem()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string();

            if let Ok(file) = fs::File::open(&self.files[i]) {
                use std::io::{Seek, SeekFrom};
                let mut reader = BufReader::new(file);
                if reader.seek(SeekFrom::Start(pos)).is_ok() {
                    let mut line = String::new();
                    loop {
                        line.clear();
                        match reader.read_line(&mut line) {
                            Ok(0) => break,
                            Ok(_) => match parse_jsonl_line(line.trim(), &default_id, &*self.mapper) {
                                ParsedLine::Events(events) => output.events.extend(events),
                                ParsedLine::SessionCleared => output.session_cleared = true,
                                ParsedLine::Ignored => {}
                            },
                            Err(_) => break,
                        }
                    }
                }
            }

            // Update position in-place — key is already present from `new()` or `add_file()`.
            if let Some(p) = self.positions.get_mut(&self.files[i]) {
                *p = current_len;
            }
        }

        output
    }
}

// ---------------------------------------------------------------------------
// ClaudeIngester — implements SessionIngester for Claude Code .jsonl logs
// ---------------------------------------------------------------------------

/// A `SessionIngester` that understands Claude Code's `.jsonl` log format.
pub struct ClaudeIngester {
    mapper: Arc<dyn ToolCallMapper>,
}

impl ClaudeIngester {
    pub fn new(mapper: Arc<dyn ToolCallMapper>) -> Self {
        Self { mapper }
    }
}

impl SessionIngester for ClaudeIngester {
    fn log_dir_for_project(&self, project_path: &Path) -> Option<PathBuf> {
        log_dir_for_project(project_path)
    }
    fn find_latest_session(&self, log_dir: &Path) -> Option<String> {
        find_latest_session(log_dir)
    }
    fn session_log_files(&self, log_dir: &Path, session_id: &str) -> Vec<PathBuf> {
        session_log_files(log_dir, session_id)
    }
    fn parse_log_file(&self, path: &Path) -> Vec<AgentToolCall> {
        parse_log_file_with_mapper(path, &*self.mapper)
    }
    fn new_tailer(&self, files: Vec<PathBuf>) -> Box<dyn EventTailer> {
        Box::new(LogTailer::new(files, Arc::clone(&self.mapper)))
    }
}

impl EventTailer for LogTailer {
    fn add_file(&mut self, path: PathBuf) {
        LogTailer::add_file(self, path);
    }
    fn read_new_events(&mut self) -> TailerOutput {
        LogTailer::read_new_events(self)
    }
}

#[cfg(test)]
#[path = "../../tests/helpers/mod.rs"]
#[allow(dead_code)]
mod helpers;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Convenience: unwrap `ParsedLine::Events` for tests that don't care about clear detection.
    fn parse_events(line: &str, agent_id: &str) -> Vec<AgentToolCall> {
        let config = ToolMappingConfig::builtin().expect("builtin config");
        match parse_jsonl_line(line, agent_id, &config) {
            ParsedLine::Events(evs) => evs,
            _ => vec![],
        }
    }

    #[test]
    fn test_parse_read_tool_call() {
        let line = r#"{"type":"assistant","sessionId":"abc-123","message":{"role":"assistant","content":[{"type":"tool_use","name":"mcp__acp__Read","input":{"file_path":"/foo/bar/src/main.rs"}}]}}"#;
        let events = parse_events(line, "default");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].read_depth, ReadDepth::FullBody);
        assert_eq!(
            events[0].file_path.as_ref().unwrap(),
            &PathBuf::from("/foo/bar/src/main.rs")
        );
    }

    #[test]
    fn test_parse_grep_tool_call() {
        let line = r#"{"type":"assistant","sessionId":"abc","message":{"role":"assistant","content":[{"type":"tool_use","name":"Grep","input":{"pattern":"AuthService"}}]}}"#;
        let events = parse_events(line, "default");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].read_depth, ReadDepth::Overview);
    }

    #[test]
    fn test_ignores_user_messages() {
        let line = r#"{"type":"user","message":{"role":"user","content":"hello"}}"#;
        let events = parse_events(line, "default");
        assert!(events.is_empty());
    }

    #[test]
    fn test_ignores_type_a() {
        // "type":"A" does not appear in real logs; only "assistant" should be accepted.
        let line = r#"{"type":"A","sessionId":"abc","message":{"role":"assistant","content":[{"type":"tool_use","name":"mcp__acp__Read","input":{"file_path":"/foo.rs"}}]}}"#;
        let events = parse_events(line, "default");
        assert!(events.is_empty());
    }

    #[test]
    fn test_is_uuid() {
        assert!(is_uuid("c4d0275f-5c57-4192-962e-ada3c2efec60"));
        assert!(is_uuid("07f66211-6835-43d3-91d5-e3468d705fc5"));
        assert!(!is_uuid("agent-a09c164"));
        assert!(!is_uuid("sessions-index"));
        assert!(!is_uuid("not-a-uuid-at-all"));
        assert!(!is_uuid(""));
    }

    #[test]
    fn test_find_session_from_files() {
        // Create a temp dir with UUID-named .jsonl files.
        let tmp = tempfile::tempdir().unwrap();
        let uuid1 = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee";
        let uuid2 = "11111111-2222-3333-4444-555555555555";

        // Write uuid1 first, then uuid2 (uuid2 should be newer).
        let f1 = tmp.path().join(format!("{uuid1}.jsonl"));
        fs::write(&f1, r#"{"type":"user","sessionId":"aaa"}"#).unwrap();
        // Small sleep to ensure different mtimes.
        std::thread::sleep(std::time::Duration::from_millis(50));
        let f2 = tmp.path().join(format!("{uuid2}.jsonl"));
        fs::write(&f2, r#"{"type":"user","sessionId":"bbb"}"#).unwrap();

        // Also create an agent file that should NOT be picked.
        fs::write(
            tmp.path().join("agent-abc123.jsonl"),
            r#"{"type":"user"}"#,
        )
        .unwrap();

        // Also create an empty UUID file that should be skipped.
        fs::File::create(tmp.path().join("00000000-0000-0000-0000-000000000000.jsonl")).unwrap();

        let result = find_session_from_files(tmp.path());
        assert_eq!(result, Some(uuid2.to_string()));
    }

    #[test]
    fn test_session_log_files_subagents_dir() {
        // Create a temp dir mimicking the new format:
        // <log_dir>/<session-id>.jsonl
        // <log_dir>/<session-id>/subagents/agent-*.jsonl
        let tmp = tempfile::tempdir().unwrap();
        let session = "abcd1234-abcd-abcd-abcd-abcd12345678";

        // Main session file.
        let main_file = tmp.path().join(format!("{session}.jsonl"));
        fs::write(&main_file, r#"{"type":"user"}"#).unwrap();

        // Subagents directory.
        let subagents_dir = tmp.path().join(session).join("subagents");
        fs::create_dir_all(&subagents_dir).unwrap();
        let agent_file = subagents_dir.join("agent-abc1234.jsonl");
        fs::write(&agent_file, r#"{"type":"user","sessionId":"xxx"}"#).unwrap();

        let files = session_log_files(tmp.path(), session);
        assert!(files.contains(&main_file));
        assert!(files.contains(&agent_file));
        assert_eq!(files.len(), 2);
    }

    #[test]
    fn test_session_log_files_flat_agents() {
        // Create a temp dir mimicking the old format:
        // <log_dir>/<session-id>.jsonl
        // <log_dir>/agent-*.jsonl (with matching sessionId)
        let tmp = tempfile::tempdir().unwrap();
        let session = "abcd1234-abcd-abcd-abcd-abcd12345678";

        let main_file = tmp.path().join(format!("{session}.jsonl"));
        fs::write(&main_file, r#"{"type":"user"}"#).unwrap();

        // Agent file that belongs to this session.
        let agent_ok = tmp.path().join("agent-match01.jsonl");
        let mut f = fs::File::create(&agent_ok).unwrap();
        writeln!(f, r#"{{"type":"user","sessionId":"{session}"}}"#).unwrap();

        // Agent file that belongs to a different session.
        let agent_other = tmp.path().join("agent-other01.jsonl");
        fs::write(&agent_other, r#"{"type":"user","sessionId":"different-session"}"#).unwrap();

        let files = session_log_files(tmp.path(), session);
        assert!(files.contains(&main_file));
        assert!(files.contains(&agent_ok));
        assert!(!files.contains(&agent_other));
    }

    // --- map_tool_call coverage tests (via parse_jsonl_line) ---

    use super::helpers::{jsonl_assistant, jsonl_user_msg};

    #[test]
    fn map_edit_tool() {
        let line = jsonl_assistant("mcp__acp__Edit", r#"{"file_path":"/src/app.rs","old_string":"a","new_string":"b"}"#);
        let events = parse_events(&line, "d");
        assert_eq!(events[0].read_depth, ReadDepth::FullBody);
        assert_eq!(events[0].file_path.as_ref().unwrap(), &PathBuf::from("/src/app.rs"));
    }

    #[test]
    fn map_write_tool() {
        let line = jsonl_assistant("mcp__acp__Write", r#"{"file_path":"/src/new.rs","content":"fn main(){}"}"#);
        let events = parse_events(&line, "d");
        assert_eq!(events[0].read_depth, ReadDepth::FullBody);
    }

    #[test]
    fn map_glob_tool() {
        let line = jsonl_assistant("Glob", r#"{"pattern":"**/*.rs","path":"/src"}"#);
        let events = parse_events(&line, "d");
        assert_eq!(events[0].read_depth, ReadDepth::NameOnly);
        assert!(
            events[0].description.contains("**/*.rs"),
            "Glob description should contain pattern, got: {:?}",
            events[0].description
        );
    }

    #[test]
    fn map_find_file() {
        let line = jsonl_assistant("mcp__serena__find_file", r#"{"file_mask":"*.rs","relative_path":"src"}"#);
        let events = parse_events(&line, "d");
        assert_eq!(events[0].read_depth, ReadDepth::NameOnly);
        assert!(
            events[0].description.contains("*.rs"),
            "find_file description should contain file_mask pattern, got: {:?}",
            events[0].description
        );
    }

    #[test]
    fn map_list_dir_shows_path_not_question_mark() {
        // list_dir has no pattern key — description should fall back to the path.
        let line = jsonl_assistant("mcp__serena__list_dir", r#"{"relative_path":"src/ingest","recursive":false}"#);
        let events = parse_events(&line, "d");
        assert_eq!(events[0].read_depth, ReadDepth::NameOnly);
        assert!(
            !events[0].description.contains('?'),
            "list_dir description should not show '?', got: {:?}",
            events[0].description
        );
        assert!(
            events[0].description.contains("src/ingest"),
            "list_dir description should show the path, got: {:?}",
            events[0].description
        );
    }

    #[test]
    fn map_symbols_overview() {
        let line = jsonl_assistant("mcp__serena__get_symbols_overview", r#"{"relative_path":"src/app.rs"}"#);
        let events = parse_events(&line, "d");
        assert_eq!(events[0].read_depth, ReadDepth::Overview);
        assert_eq!(events[0].file_path.as_ref().unwrap(), &PathBuf::from("src/app.rs"));
    }

    #[test]
    fn map_find_symbol_no_body() {
        let line = jsonl_assistant("mcp__serena__find_symbol", r#"{"name_path_pattern":"App","relative_path":"src/app.rs","include_body":false}"#);
        let events = parse_events(&line, "d");
        assert_eq!(events[0].read_depth, ReadDepth::Signature);
        assert_eq!(events[0].target_symbol.as_deref(), Some("App"));
    }

    #[test]
    fn map_find_symbol_with_body() {
        let line = jsonl_assistant("mcp__serena__find_symbol", r#"{"name_path_pattern":"App/new","relative_path":"src/app.rs","include_body":true}"#);
        let events = parse_events(&line, "d");
        assert_eq!(events[0].read_depth, ReadDepth::FullBody);
        assert_eq!(events[0].target_symbol.as_deref(), Some("App/new"));
    }

    #[test]
    fn map_find_referencing() {
        let line = jsonl_assistant("mcp__serena__find_referencing_symbols", r#"{"name_path":"ProjectTree","relative_path":"src/symbols/mod.rs"}"#);
        let events = parse_events(&line, "d");
        assert_eq!(events[0].read_depth, ReadDepth::Overview);
        assert_eq!(events[0].target_symbol.as_deref(), Some("ProjectTree"));
    }

    #[test]
    fn map_replace_symbol() {
        let line = jsonl_assistant("mcp__serena__replace_symbol_body", r#"{"name_path":"App/new","relative_path":"src/app.rs","body":"pub fn new() {}"}"#);
        let events = parse_events(&line, "d");
        assert_eq!(events[0].read_depth, ReadDepth::FullBody);
        assert_eq!(events[0].target_symbol.as_deref(), Some("App/new"));
    }

    #[test]
    fn map_insert_after() {
        let line = jsonl_assistant("mcp__serena__insert_after_symbol", r#"{"name_path":"App","relative_path":"src/app.rs","body":"fn foo() {}"}"#);
        let events = parse_events(&line, "d");
        assert_eq!(events[0].read_depth, ReadDepth::FullBody);
        assert_eq!(events[0].target_symbol.as_deref(), Some("App"));
    }

    #[test]
    fn map_rename_symbol() {
        let line = jsonl_assistant("mcp__serena__rename_symbol", r#"{"name_path":"old_fn","relative_path":"src/app.rs","new_name":"new_fn"}"#);
        let events = parse_events(&line, "d");
        assert_eq!(events[0].read_depth, ReadDepth::FullBody);
        assert_eq!(events[0].target_symbol.as_deref(), Some("old_fn"));
    }

    #[test]
    fn map_notebook_edit() {
        let line = jsonl_assistant("NotebookEdit", r#"{"notebook_path":"/nb/analysis.ipynb","new_source":"print(1)"}"#);
        let events = parse_events(&line, "d");
        assert_eq!(events[0].read_depth, ReadDepth::FullBody);
        assert_eq!(events[0].file_path.as_ref().unwrap(), &PathBuf::from("/nb/analysis.ipynb"));
    }

    #[test]
    fn map_unknown_tool() {
        let line = jsonl_assistant("SomeRandomTool", r#"{"data":"value"}"#);
        let events = parse_events(&line, "d");
        // Unknown tools still produce an event, but with Unseen depth.
        assert_eq!(events[0].read_depth, ReadDepth::Unseen);
    }

    #[test]
    fn read_with_offset_limit() {
        let line = jsonl_assistant("mcp__acp__Read", r#"{"file_path":"/src/main.rs","offset":10,"limit":20}"#);
        let events = parse_events(&line, "d");
        assert_eq!(events[0].read_depth, ReadDepth::FullBody);
        assert_eq!(events[0].target_lines, Some(10..30));
    }

    #[test]
    fn parse_malformed_json() {
        let events = parse_events("not valid json {{{", "d");
        assert!(events.is_empty());
    }

    #[test]
    fn parse_multi_tool_message() {
        let line = r#"{"type":"assistant","sessionId":"s","message":{"role":"assistant","content":[{"type":"tool_use","name":"mcp__acp__Read","input":{"file_path":"/a.rs"}},{"type":"tool_use","name":"Grep","input":{"pattern":"foo"}}]}}"#;
        let events = parse_events(line, "d");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].read_depth, ReadDepth::FullBody);
        assert_eq!(events[1].read_depth, ReadDepth::Overview);
    }

    #[test]
    fn parse_log_file_skips_non_assistant() {
        let tmp = tempfile::tempdir().unwrap();
        let log = tmp.path().join("test.jsonl");
        let mut f = fs::File::create(&log).unwrap();
        writeln!(f, "{}", jsonl_user_msg()).unwrap();
        writeln!(f, "{}", jsonl_assistant("mcp__acp__Read", r#"{"file_path":"/a.rs"}"#)).unwrap();
        writeln!(f, "{}", jsonl_user_msg()).unwrap();
        writeln!(f, "{}", jsonl_assistant("Grep", r#"{"pattern":"x"}"#)).unwrap();
        drop(f);

        let config = ToolMappingConfig::builtin().expect("builtin config");
        let events = parse_log_file(&log, &config);
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn extract_label_from_subagent_log() {
        let tmp = tempfile::tempdir().unwrap();
        let log = tmp.path().join("agent-abc1234.jsonl");
        let content = r#"{"agentId":"abc1234","type":"user","message":{"role":"user","content":"Explore the parser module"},"sessionId":"sess-1","timestamp":"2025-01-01T00:00:00Z"}"#;
        fs::write(&log, content).unwrap();

        let label = extract_agent_label(&log);
        assert_eq!(label, "Explore the parser module");
    }

    #[test]
    fn extract_label_truncates_long_prompt() {
        let tmp = tempfile::tempdir().unwrap();
        let log = tmp.path().join("agent-abc1234.jsonl");
        let long_prompt = "A".repeat(100);
        let content = format!(
            r#"{{"agentId":"abc1234","type":"user","message":{{"role":"user","content":"{long_prompt}"}},"sessionId":"sess-1","timestamp":"2025-01-01T00:00:00Z"}}"#
        );
        fs::write(&log, content).unwrap();

        let label = extract_agent_label(&log);
        assert_eq!(label.len(), 50); // "AAA...AAA..."
        assert!(label.ends_with("..."));
    }

    #[test]
    fn extract_label_falls_back_to_filename() {
        let tmp = tempfile::tempdir().unwrap();
        let log = tmp.path().join("agent-def5678.jsonl");
        // Empty content field → falls back to filename stem sans "agent-" prefix.
        let content = r#"{"agentId":"def5678","type":"user","message":{"role":"user","content":""},"sessionId":"sess-1"}"#;
        fs::write(&log, content).unwrap();

        let label = extract_agent_label(&log);
        assert_eq!(label, "def5678");
    }

    #[test]
    fn extract_label_nonexistent_file() {
        let label = extract_agent_label(Path::new("/nonexistent/agent-xyz.jsonl"));
        assert_eq!(label, "xyz");
    }

    #[test]
    fn extract_label_main_session_file() {
        // Main session files don't have "agent-" prefix.
        let tmp = tempfile::tempdir().unwrap();
        let log = tmp.path().join("abcd1234-abcd-abcd-abcd-abcd12345678.jsonl");
        let content = r#"{"type":"user","message":{"role":"user","content":"Hello"},"sessionId":"abcd1234"}"#;
        fs::write(&log, content).unwrap();

        let label = extract_agent_label(&log);
        assert_eq!(label, "Hello");
    }

    #[test]
    fn parse_log_file_sets_label_on_events() {
        let tmp = tempfile::tempdir().unwrap();
        let log = tmp.path().join("agent-test123.jsonl");
        let mut f = fs::File::create(&log).unwrap();
        // First line: user message with task prompt.
        writeln!(f, r#"{{"agentId":"test123","type":"user","message":{{"role":"user","content":"Check the coverage"}},"sessionId":"sess-1","timestamp":"2025-01-01T00:00:00Z"}}"#).unwrap();
        // Second line: assistant with a tool call.
        writeln!(f, "{}", jsonl_assistant("mcp__acp__Read", r#"{"file_path":"/a.rs"}"#)).unwrap();
        drop(f);

        let config = ToolMappingConfig::builtin().unwrap();
        let events = parse_log_file(&log, &config);
        assert_eq!(events.len(), 1);
        assert_eq!(&*events[0].label, "Check the coverage");
    }

    #[test]
    fn agent_id_prefers_agent_id_over_session_id() {
        // Subagent lines have both agentId and sessionId; agentId should win.
        let line = r#"{"type":"assistant","agentId":"a9fe23c","sessionId":"7842313b-63c5-49db-97d1-c78ba563278e","message":{"role":"assistant","content":[{"type":"tool_use","name":"mcp__acp__Read","input":{"file_path":"/foo/bar.rs"}}]}}"#;
        let events = parse_events(line, "fallback");
        assert_eq!(events.len(), 1);
        assert_eq!(&*events[0].agent_id, "a9fe23c");
    }

    #[test]
    fn agent_id_falls_back_to_session_id() {
        // Main session lines have sessionId but no agentId.
        let line = r#"{"type":"assistant","sessionId":"7842313b","message":{"role":"assistant","content":[{"type":"tool_use","name":"mcp__acp__Read","input":{"file_path":"/foo/bar.rs"}}]}}"#;
        let events = parse_events(line, "fallback");
        assert_eq!(events.len(), 1);
        assert_eq!(&*events[0].agent_id, "7842313b");
    }

    #[test]
    fn agent_id_falls_back_to_default() {
        // No agentId or sessionId → uses default_agent_id.
        let line = r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","name":"mcp__acp__Read","input":{"file_path":"/foo/bar.rs"}}]}}"#;
        let events = parse_events(line, "my-default");
        assert_eq!(events.len(), 1);
        assert_eq!(&*events[0].agent_id, "my-default");
    }

    // --- /clear detection tests ---

    #[test]
    fn parse_clear_command_returns_session_cleared() {
        // Exact JSONL shape observed in the wild (empty command-args).
        let line = r#"{"type":"user","sessionId":"abc","message":{"role":"user","content":"<command-name>/clear</command-name>\n            <command-message>clear</command-message>\n            <command-args></command-args>"}}"#;
        let config = ToolMappingConfig::builtin().expect("builtin config");
        assert!(matches!(parse_jsonl_line(line, "d", &config), ParsedLine::SessionCleared));
    }

    #[test]
    fn parse_clear_with_nonempty_args() {
        // /clear invoked with explicit args (as seen in some sessions: command-args=clear).
        let line = r#"{"type":"user","sessionId":"abc","message":{"role":"user","content":"<command-name>/clear</command-name>\n            <command-message>clear</command-message>\n            <command-args>clear</command-args>"}}"#;
        let config = ToolMappingConfig::builtin().expect("builtin config");
        assert!(matches!(parse_jsonl_line(line, "d", &config), ParsedLine::SessionCleared));
    }

    #[test]
    fn parse_compact_command_returns_ignored() {
        // /compact must NOT trigger SessionCleared.
        let line = r#"{"type":"user","sessionId":"abc","message":{"role":"user","content":"<command-name>/compact</command-name>\n            <command-message>compact</command-message>\n            <command-args></command-args>"}}"#;
        let config = ToolMappingConfig::builtin().expect("builtin config");
        assert!(matches!(parse_jsonl_line(line, "d", &config), ParsedLine::Ignored));
    }

    #[test]
    fn tailer_sets_session_cleared_flag() {
        let tmp = tempfile::tempdir().unwrap();
        let log = tmp.path().join("session.jsonl");

        // Create an empty file first so the tailer starts at position 0.
        fs::write(&log, "").unwrap();
        let config: Arc<dyn ToolCallMapper> = Arc::new(ToolMappingConfig::builtin().expect("builtin config"));
        let mut tailer = LogTailer::new(vec![log.clone()], config);

        // Now append the /clear line as "new" content.
        let clear_line = r#"{"type":"user","sessionId":"abc","message":{"role":"user","content":"<command-name>/clear</command-name>\n<command-message>clear</command-message>\n<command-args></command-args>"}}"#;
        fs::write(&log, format!("{clear_line}\n")).unwrap();

        let output = tailer.read_new_events();
        assert!(output.session_cleared);
        assert!(output.events.is_empty());
    }

    #[test]
    fn tailer_emits_events_after_clear() {
        let tmp = tempfile::tempdir().unwrap();
        let log = tmp.path().join("session.jsonl");

        // Create an empty file first so the tailer starts at position 0.
        fs::write(&log, "").unwrap();
        let config: Arc<dyn ToolCallMapper> = Arc::new(ToolMappingConfig::builtin().expect("builtin config"));
        let mut tailer = LogTailer::new(vec![log.clone()], config);

        // Append a /clear line followed by a tool use as "new" content.
        let clear_line = r#"{"type":"user","sessionId":"abc","message":{"role":"user","content":"<command-name>/clear</command-name>\n<command-message>clear</command-message>\n<command-args></command-args>"}}"#;
        let tool_line = jsonl_assistant("mcp__acp__Read", r#"{"file_path":"/src/main.rs"}"#);
        fs::write(&log, format!("{clear_line}\n{tool_line}\n")).unwrap();

        let output = tailer.read_new_events();
        assert!(output.session_cleared);
        assert_eq!(output.events.len(), 1);
        assert_eq!(output.events[0].read_depth, ReadDepth::FullBody);
    }

    // -----------------------------------------------------------------------
    // Phase 9: Dispatch tests — merged config
    // -----------------------------------------------------------------------

    /// A merged config where "Read" is replaced by a user stanza dispatches
    /// correctly for the new input key and rejects the old key.
    #[test]
    fn merged_config_dispatches_replaced_builtin() {
        use std::io::Write as IoWrite;

        let builtin = ToolMappingConfig::builtin().unwrap();

        // Write user config to a temp file so load() builds the index.
        let user_toml = r#"
version = 1
[[tool]]
names        = ["Read"]
path_keys    = ["custom_file"]
pattern_keys = []
depth        = { type = "fixed", value = "NameOnly" }
description  = "CustomRead {custom_file}"
"#;
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(user_toml.as_bytes()).unwrap();
        let (user_opt, _) = ToolMappingConfig::load(tmp.path());
        let user = user_opt.expect("user config must parse");

        let mut warnings = Vec::new();
        let merged = ToolMappingConfig::merge(builtin, user, &mut warnings);

        // Dispatch with the new input key.
        let input = serde_json::json!({ "custom_file": "/foo/bar.rs" });
        let result = map_tool_call(&merged, "Read", &input, "agent1", "2025-01-01");
        assert!(result.is_some(), "Read should dispatch via merged config");
        let call = result.unwrap();
        assert_eq!(call.read_depth, ReadDepth::NameOnly);
        assert!(call.description.contains("CustomRead"));

        // The old key "file_path" should now be unknown → None (path_required = true by default).
        let old_input = serde_json::json!({ "file_path": "/foo/bar.rs" });
        let missing = map_tool_call(&merged, "Read", &old_input, "agent1", "2025-01-01");
        assert!(missing.is_none(), "Read with old key should return None after override");
    }

    /// A merged config with a novel user tool should dispatch it end-to-end.
    #[test]
    fn merged_config_dispatches_user_tool() {
        use std::io::Write as IoWrite;

        let builtin = ToolMappingConfig::builtin().unwrap();
        let user_toml = r#"
version = 1
[[tool]]
names        = ["UserTool"]
path_keys    = ["target"]
pattern_keys = []
depth        = { type = "fixed", value = "Overview" }
description  = "UserTool {target}"
"#;
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(user_toml.as_bytes()).unwrap();
        let (user_opt, _) = ToolMappingConfig::load(tmp.path());
        let user = user_opt.expect("user config must parse");

        let mut warnings = Vec::new();
        let merged = ToolMappingConfig::merge(builtin, user, &mut warnings);

        let input = serde_json::json!({ "target": "/some/path.rs" });
        let result = map_tool_call(&merged, "UserTool", &input, "agentX", "ts");
        assert!(result.is_some());
        let call = result.unwrap();
        assert_eq!(call.read_depth, ReadDepth::Overview);
        assert_eq!(&*call.tool_name, "UserTool");
        assert!(call.file_path.is_some());
    }

    // -----------------------------------------------------------------------
    // Bash per-command depth dispatch
    // -----------------------------------------------------------------------

    fn bash_event(command: &str) -> AgentToolCall {
        let config = ToolMappingConfig::builtin().expect("builtin config");
        let input = serde_json::json!({ "command": command });
        map_tool_call(&config, "Bash", &input, "agent", "ts")
            .expect("Bash must always produce an event")
    }

    #[test]
    fn bash_cat_gets_full_body() {
        assert_eq!(bash_event("cat src/main.rs").read_depth, ReadDepth::FullBody);
    }

    #[test]
    fn bash_head_gets_signature() {
        assert_eq!(bash_event("head -n 20 src/lib.rs").read_depth, ReadDepth::Signature);
    }

    #[test]
    fn bash_tail_gets_signature() {
        assert_eq!(bash_event("tail -n 50 logs/output.log").read_depth, ReadDepth::Signature);
    }

    #[test]
    fn bash_grep_gets_overview() {
        assert_eq!(bash_event("grep 'fn main' src/").read_depth, ReadDepth::Overview);
    }

    #[test]
    fn bash_rg_gets_overview() {
        assert_eq!(bash_event("rg 'ReadDepth' --type rust").read_depth, ReadDepth::Overview);
    }

    #[test]
    fn bash_git_gets_unseen() {
        assert_eq!(bash_event("git status").read_depth, ReadDepth::Unseen);
        assert_eq!(bash_event("git log --oneline -10").read_depth, ReadDepth::Unseen);
    }

    #[test]
    fn bash_unknown_command_gets_name_only() {
        // Unrecognised commands fall back to the PatternMatch default (NameOnly).
        assert_eq!(bash_event("cargo build --release").read_depth, ReadDepth::NameOnly);
        assert_eq!(bash_event("npm install").read_depth, ReadDepth::NameOnly);
    }
}
