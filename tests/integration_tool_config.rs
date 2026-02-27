//! Integration tests for config-driven tool call mappings.
//!
//! Covers all 13 built-in tool stanzas end-to-end:
//!   JSONL line → parse_jsonl_line → AgentToolCall fields verified.

use ambits::ingest::claude::{map_tool_call, parse_log_file};
use ambits::ingest::tool_config::ToolMappingConfig;
use ambits::tracking::ReadDepth;

/// Build a minimal JSONL assistant line for a tool call.
fn assistant_line(tool_name: &str, input_json: &str) -> String {
    format!(
        r#"{{"type":"assistant","agentId":"test-agent","sessionId":"sess1","message":{{"role":"assistant","content":[{{"type":"tool_use","name":"{tool_name}","input":{input_json}}}]}}}}"#
    )
}

fn builtin() -> ToolMappingConfig {
    ToolMappingConfig::builtin().expect("built-in config must parse")
}

// ---------------------------------------------------------------------------
// 1. Read
// ---------------------------------------------------------------------------
#[test]
fn tool_read_file_path() {
    let cfg = builtin();
    let input = serde_json::json!({ "file_path": "/src/main.rs" });
    let call = map_tool_call(&cfg, "Read", &input, "a", "ts").unwrap();
    assert_eq!(call.read_depth, ReadDepth::FullBody);
    assert_eq!(call.file_path.unwrap().to_str().unwrap(), "/src/main.rs");
    assert!(call.target_lines.is_none());
}

#[test]
fn tool_read_with_target_lines() {
    let cfg = builtin();
    let input = serde_json::json!({ "file_path": "/src/lib.rs", "offset": 10, "limit": 20 });
    let call = map_tool_call(&cfg, "Read", &input, "a", "ts").unwrap();
    assert!(call.target_lines.is_some());
    let lines = call.target_lines.unwrap();
    assert_eq!(lines, 10..30);
}

#[test]
fn tool_read_missing_path_returns_none() {
    let cfg = builtin();
    let input = serde_json::json!({ "other_key": "/irrelevant" });
    assert!(map_tool_call(&cfg, "Read", &input, "a", "ts").is_none());
}

// ---------------------------------------------------------------------------
// 2. Edit
// ---------------------------------------------------------------------------
#[test]
fn tool_edit_full_body() {
    let cfg = builtin();
    let input = serde_json::json!({ "file_path": "/src/foo.rs" });
    let call = map_tool_call(&cfg, "Edit", &input, "a", "ts").unwrap();
    assert_eq!(call.read_depth, ReadDepth::FullBody);
    assert!(call.description.contains("foo.rs"));
}

// ---------------------------------------------------------------------------
// 3. Write
// ---------------------------------------------------------------------------
#[test]
fn tool_write_full_body() {
    let cfg = builtin();
    let input = serde_json::json!({ "file_path": "/new/file.rs" });
    let call = map_tool_call(&cfg, "Write", &input, "a", "ts").unwrap();
    assert_eq!(call.read_depth, ReadDepth::FullBody);
}

// ---------------------------------------------------------------------------
// 4. Glob — path_required = false, pattern_keys used
// ---------------------------------------------------------------------------
#[test]
fn tool_glob_no_path_needed() {
    let cfg = builtin();
    let input = serde_json::json!({ "pattern": "**/*.rs" });
    let call = map_tool_call(&cfg, "Glob", &input, "a", "ts").unwrap();
    assert_eq!(call.read_depth, ReadDepth::NameOnly);
    assert!(call.file_path.is_none());
    assert!(call.description.contains("**/*.rs"), "description was: {}", call.description);
}

#[test]
fn tool_glob_file_mask_fallback() {
    let cfg = builtin();
    // `file_mask` is the second pattern_key for Glob.
    let input = serde_json::json!({ "file_mask": "*.toml" });
    let call = map_tool_call(&cfg, "Glob", &input, "a", "ts").unwrap();
    assert!(call.description.contains("*.toml"), "description was: {}", call.description);
}

// ---------------------------------------------------------------------------
// 5. Grep — pattern_keys: ["pattern", "substring_pattern"]
// ---------------------------------------------------------------------------
#[test]
fn tool_grep_pattern_key() {
    let cfg = builtin();
    let input = serde_json::json!({ "pattern": "fn main" });
    let call = map_tool_call(&cfg, "Grep", &input, "a", "ts").unwrap();
    assert_eq!(call.read_depth, ReadDepth::Overview);
    assert!(call.description.contains("fn main"), "description was: {}", call.description);
}

#[test]
fn tool_grep_substring_pattern_fallback() {
    let cfg = builtin();
    let input = serde_json::json!({ "substring_pattern": "struct Foo" });
    let call = map_tool_call(&cfg, "Grep", &input, "a", "ts").unwrap();
    assert!(call.description.contains("struct Foo"), "description was: {}", call.description);
}

// ---------------------------------------------------------------------------
// 6. get_symbols_overview
// ---------------------------------------------------------------------------
#[test]
fn tool_get_symbols_overview() {
    let cfg = builtin();
    let input = serde_json::json!({ "relative_path": "src/lib.rs" });
    let call = map_tool_call(&cfg, "mcp__serena__get_symbols_overview", &input, "a", "ts").unwrap();
    assert_eq!(call.read_depth, ReadDepth::Overview);
    assert!(call.file_path.is_some());
}

// ---------------------------------------------------------------------------
// 7. find_symbol — conditional depth
// ---------------------------------------------------------------------------
#[test]
fn tool_find_symbol_with_body() {
    let cfg = builtin();
    let input = serde_json::json!({ "relative_path": "src/foo.rs", "name_path_pattern": "Foo/bar", "include_body": true });
    let call = map_tool_call(&cfg, "mcp__serena__find_symbol", &input, "a", "ts").unwrap();
    assert_eq!(call.read_depth, ReadDepth::FullBody);
    assert_eq!(call.target_symbol.as_deref(), Some("Foo/bar"));
}

#[test]
fn tool_find_symbol_without_body() {
    let cfg = builtin();
    let input = serde_json::json!({ "relative_path": "src/foo.rs", "name_path_pattern": "Foo/bar", "include_body": false });
    let call = map_tool_call(&cfg, "mcp__serena__find_symbol", &input, "a", "ts").unwrap();
    assert_eq!(call.read_depth, ReadDepth::Signature);
}

#[test]
fn tool_find_symbol_absent_include_body() {
    let cfg = builtin();
    let input = serde_json::json!({ "relative_path": "src/foo.rs", "name_path_pattern": "Foo" });
    let call = map_tool_call(&cfg, "mcp__serena__find_symbol", &input, "a", "ts").unwrap();
    assert_eq!(call.read_depth, ReadDepth::Signature);
}

// ---------------------------------------------------------------------------
// 8. find_referencing_symbols
// ---------------------------------------------------------------------------
#[test]
fn tool_find_referencing_symbols() {
    let cfg = builtin();
    let input = serde_json::json!({ "relative_path": "src/foo.rs", "name_path": "Foo" });
    let call = map_tool_call(&cfg, "mcp__serena__find_referencing_symbols", &input, "a", "ts").unwrap();
    assert_eq!(call.read_depth, ReadDepth::Overview);
    assert_eq!(call.target_symbol.as_deref(), Some("Foo"));
}

// ---------------------------------------------------------------------------
// 9. replace_symbol_body
// ---------------------------------------------------------------------------
#[test]
fn tool_replace_symbol_body() {
    let cfg = builtin();
    let input = serde_json::json!({ "relative_path": "src/foo.rs", "name_path": "MyFn" });
    let call = map_tool_call(&cfg, "mcp__serena__replace_symbol_body", &input, "a", "ts").unwrap();
    assert_eq!(call.read_depth, ReadDepth::FullBody);
    assert_eq!(call.target_symbol.as_deref(), Some("MyFn"));
}

// ---------------------------------------------------------------------------
// 10. insert_after_symbol
// ---------------------------------------------------------------------------
#[test]
fn tool_insert_after_symbol() {
    let cfg = builtin();
    let input = serde_json::json!({ "relative_path": "src/foo.rs", "name_path": "last_fn" });
    let call = map_tool_call(&cfg, "mcp__serena__insert_after_symbol", &input, "a", "ts").unwrap();
    assert_eq!(call.read_depth, ReadDepth::FullBody);
}

// ---------------------------------------------------------------------------
// 11. insert_before_symbol
// ---------------------------------------------------------------------------
#[test]
fn tool_insert_before_symbol() {
    let cfg = builtin();
    let input = serde_json::json!({ "relative_path": "src/foo.rs", "name_path": "first_fn" });
    let call = map_tool_call(&cfg, "mcp__serena__insert_before_symbol", &input, "a", "ts").unwrap();
    assert_eq!(call.read_depth, ReadDepth::FullBody);
}

// ---------------------------------------------------------------------------
// 12. rename_symbol
// ---------------------------------------------------------------------------
#[test]
fn tool_rename_symbol() {
    let cfg = builtin();
    let input = serde_json::json!({ "relative_path": "src/foo.rs", "name_path": "old_name" });
    let call = map_tool_call(&cfg, "mcp__serena__rename_symbol", &input, "a", "ts").unwrap();
    assert_eq!(call.read_depth, ReadDepth::FullBody);
    assert_eq!(call.target_symbol.as_deref(), Some("old_name"));
}

// ---------------------------------------------------------------------------
// 13. NotebookEdit
// ---------------------------------------------------------------------------
#[test]
fn tool_notebook_edit() {
    let cfg = builtin();
    let input = serde_json::json!({ "notebook_path": "/work/notebook.ipynb" });
    let call = map_tool_call(&cfg, "NotebookEdit", &input, "a", "ts").unwrap();
    assert_eq!(call.read_depth, ReadDepth::FullBody);
    assert!(call.file_path.is_some());
}

// ---------------------------------------------------------------------------
// Unknown tool — returns None
// ---------------------------------------------------------------------------
#[test]
fn unknown_tool_returns_none() {
    let cfg = builtin();
    let input = serde_json::json!({ "file_path": "/foo.rs" });
    assert!(map_tool_call(&cfg, "UnknownTool", &input, "a", "ts").is_none());
}

// ---------------------------------------------------------------------------
// mcp__acp__ aliases all resolve to the same stanza
// ---------------------------------------------------------------------------
#[test]
fn mcp_acp_read_alias() {
    let cfg = builtin();
    let input = serde_json::json!({ "file_path": "/alias.rs" });
    let call = map_tool_call(&cfg, "mcp__acp__Read", &input, "a", "ts").unwrap();
    assert_eq!(call.read_depth, ReadDepth::FullBody);
}

#[test]
fn mcp_acp_edit_alias() {
    let cfg = builtin();
    let input = serde_json::json!({ "file_path": "/alias.rs" });
    let call = map_tool_call(&cfg, "mcp__acp__Edit", &input, "a", "ts").unwrap();
    assert_eq!(call.read_depth, ReadDepth::FullBody);
}

// ---------------------------------------------------------------------------
// parse_log_file end-to-end: all 13 tool stanzas produce events
// ---------------------------------------------------------------------------
#[test]
fn parse_log_file_all_tool_stanzas() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let cfg = ToolMappingConfig::builtin().unwrap();
    let mut tmp = NamedTempFile::new().unwrap();

    let lines = [
        assistant_line("Read",            r#"{"file_path":"/a.rs"}"#),
        assistant_line("Edit",            r#"{"file_path":"/b.rs"}"#),
        assistant_line("Write",           r#"{"file_path":"/c.rs"}"#),
        assistant_line("Glob",            r#"{"pattern":"**/*.rs"}"#),
        assistant_line("Grep",            r#"{"pattern":"fn foo"}"#),
        assistant_line("mcp__serena__get_symbols_overview",      r#"{"relative_path":"src/lib.rs"}"#),
        assistant_line("mcp__serena__find_symbol",               r#"{"relative_path":"src/lib.rs","name_path_pattern":"Foo","include_body":true}"#),
        assistant_line("mcp__serena__find_referencing_symbols",  r#"{"relative_path":"src/lib.rs","name_path":"Foo"}"#),
        assistant_line("mcp__serena__replace_symbol_body",       r#"{"relative_path":"src/lib.rs","name_path":"Foo"}"#),
        assistant_line("mcp__serena__insert_after_symbol",       r#"{"relative_path":"src/lib.rs","name_path":"Foo"}"#),
        assistant_line("mcp__serena__insert_before_symbol",      r#"{"relative_path":"src/lib.rs","name_path":"Foo"}"#),
        assistant_line("mcp__serena__rename_symbol",             r#"{"relative_path":"src/lib.rs","name_path":"Foo"}"#),
        assistant_line("NotebookEdit",    r#"{"notebook_path":"/nb.ipynb"}"#),
    ];

    for line in &lines {
        writeln!(tmp, "{}", line).unwrap();
    }

    let events = parse_log_file(tmp.path(), &cfg);
    assert_eq!(events.len(), 13, "expected 13 events, got {}: {:?}", events.len(),
        events.iter().map(|e| &e.tool_name).collect::<Vec<_>>());
}
