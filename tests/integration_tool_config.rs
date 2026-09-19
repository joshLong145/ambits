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
// 16. Agent
// ---------------------------------------------------------------------------
#[test]
fn tool_agent_maps_with_description() {
    let cfg = builtin();
    let input = serde_json::json!({ "description": "Explore parser", "subagent_type": "Explore", "prompt": "..." });
    let call = map_tool_call(&cfg, "Agent", &input, "a", "ts").unwrap();
    assert_eq!(call.read_depth, ReadDepth::Unseen);
    assert!(call.description.contains("Explore parser"), "description was: {}", call.description);
}

// ---------------------------------------------------------------------------
// 17. ToolSearch
// ---------------------------------------------------------------------------
#[test]
fn tool_search_gets_unseen() {
    let cfg = builtin();
    let input = serde_json::json!({ "query": "select:Read,Edit" });
    let call = map_tool_call(&cfg, "ToolSearch", &input, "a", "ts").unwrap();
    assert_eq!(call.read_depth, ReadDepth::Unseen);
    assert!(call.description.contains("select:Read,Edit"), "description was: {}", call.description);
}

// ---------------------------------------------------------------------------
// 18. Skill
// ---------------------------------------------------------------------------
#[test]
fn tool_skill_gets_name_only() {
    let cfg = builtin();
    let input = serde_json::json!({ "skill": "rust-skills:m01-ownership" });
    let call = map_tool_call(&cfg, "Skill", &input, "a", "ts").unwrap();
    assert_eq!(call.read_depth, ReadDepth::NameOnly);
    assert!(call.description.contains("rust-skills:m01-ownership"), "description was: {}", call.description);
}

// ---------------------------------------------------------------------------
// 19. SendMessage
// ---------------------------------------------------------------------------
#[test]
fn tool_send_message_gets_unseen() {
    let cfg = builtin();
    let input = serde_json::json!({ "to": "agent-abc", "message": "hi" });
    let call = map_tool_call(&cfg, "SendMessage", &input, "a", "ts").unwrap();
    assert_eq!(call.read_depth, ReadDepth::Unseen);
    assert!(call.description.contains("agent-abc"), "description was: {}", call.description);
}

// ---------------------------------------------------------------------------
// 20. TaskList
// ---------------------------------------------------------------------------
#[test]
fn tool_task_list_gets_unseen() {
    let cfg = builtin();
    let input = serde_json::json!({});
    let call = map_tool_call(&cfg, "TaskList", &input, "a", "ts").unwrap();
    assert_eq!(call.read_depth, ReadDepth::Unseen);
}

// ---------------------------------------------------------------------------
// 21. AskUserQuestion
// ---------------------------------------------------------------------------
#[test]
fn tool_ask_user_question_gets_unseen() {
    let cfg = builtin();
    let input = serde_json::json!({ "question": "Proceed?", "options": ["Yes", "No"] });
    let call = map_tool_call(&cfg, "AskUserQuestion", &input, "a", "ts").unwrap();
    assert_eq!(call.read_depth, ReadDepth::Unseen);
}

// ---------------------------------------------------------------------------
// 22. WebFetch / WebSearch
// ---------------------------------------------------------------------------
#[test]
fn tool_web_fetch_gets_unseen() {
    let cfg = builtin();
    let input = serde_json::json!({ "url": "https://example.com" });
    let call = map_tool_call(&cfg, "WebFetch", &input, "a", "ts").unwrap();
    assert_eq!(call.read_depth, ReadDepth::Unseen);
    assert!(call.description.contains("https://example.com"), "description was: {}", call.description);
}

#[test]
fn tool_web_search_gets_unseen() {
    let cfg = builtin();
    let input = serde_json::json!({ "query": "rust error handling" });
    let call = map_tool_call(&cfg, "WebSearch", &input, "a", "ts").unwrap();
    assert_eq!(call.read_depth, ReadDepth::Unseen);
    assert!(call.description.contains("rust error handling"), "description was: {}", call.description);
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
// 14. Bash — pattern from "command" key, truncated with |cmd
// ---------------------------------------------------------------------------
#[test]
fn tool_bash_short_command() {
    let cfg = builtin();
    let input = serde_json::json!({ "command": "cargo build" });
    let call = map_tool_call(&cfg, "Bash", &input, "a", "ts").unwrap();
    assert_eq!(call.read_depth, ReadDepth::NameOnly);
    assert!(call.file_path.is_none());
    assert!(call.description.contains("cargo build"), "description was: {}", call.description);
}

#[test]
fn tool_bash_long_command_truncated() {
    let cfg = builtin();
    // Command longer than 200 chars should be truncated with "…"
    let long_cmd = "cargo test --all-features -- --nocapture 2>&1 | tee output.log && \
        echo 'done' && cargo clippy --all-targets --all-features -- -D warnings && \
        cargo fmt --check && echo 'all checks passed' && cargo build --release";
    assert!(long_cmd.len() > 200, "test command must exceed 200 chars, got {}", long_cmd.len());
    let input = serde_json::json!({ "command": long_cmd });
    let call = map_tool_call(&cfg, "Bash", &input, "a", "ts").unwrap();
    assert!(call.description.contains('…'), "expected truncation ellipsis, got: {}", call.description);
    // The description body (excluding "Bash " prefix) should be at most 200 chars + "…"
    let body = call.description.strip_prefix("Bash ").unwrap_or(&call.description);
    // 200 chars + "…" (3 bytes) = 203 bytes max
    assert!(body.len() <= 204, "truncated body too long: {}", body);
}

/// The panic this guards: a multi-byte character straddling the truncation
/// offset. `unwrap_or` evaluated its fallback slice unconditionally, so every
/// long command was cut at a fixed 200 bytes regardless of where characters
/// began — and a box-drawing `─` in a heredoc took down the TUI and
/// `--coverage` alike, from nothing worse than reading the session log back.
#[test]
fn tool_bash_truncates_a_multibyte_command_on_a_char_boundary() {
    let cfg = builtin();
    // `─` occupies bytes 198..201, so a naive cut at 200 lands inside it.
    let cmd = format!("{}─ and more text after the cut", "x".repeat(198));
    assert!(cmd.len() > 200);

    let input = serde_json::json!({ "command": cmd });
    let call = map_tool_call(&cfg, "Bash", &input, "a", "ts").unwrap();

    assert!(call.description.contains('…'), "still truncated");
    assert!(
        call.description.is_char_boundary(call.description.len()),
        "the result is valid UTF-8 by construction"
    );
}

#[test]
fn tool_bash_description_key_fallback() {
    let cfg = builtin();
    // When "command" is absent, fall back to "description" pattern_key.
    let input = serde_json::json!({ "description": "List files" });
    let call = map_tool_call(&cfg, "Bash", &input, "a", "ts").unwrap();
    assert!(call.description.contains("List files"), "description was: {}", call.description);
}

// ---------------------------------------------------------------------------
// 14b. Bash — selectors, which only `ambits show` earns
// ---------------------------------------------------------------------------

/// `ambits show` reads code without naming a file, so this is the only route
/// by which it earns coverage at all.
#[test]
fn tool_bash_show_command_credits_its_selectors() {
    let cfg = builtin();
    let input = serde_json::json!({ "command": "ambits -p . show src/app.rs::App/render" });
    let call = map_tool_call(&cfg, "Bash", &input, "a", "ts").unwrap();

    assert_eq!(
        call.target_selectors,
        vec![("src/app.rs::App/render".to_string(), ReadDepth::FullBody)]
    );
    assert_eq!(
        call.read_depth,
        ReadDepth::FullBody,
        "the selector's depth overrides the generic Bash default"
    );
}

/// …and `rg`/`grep` do not earn selector (symbol-level) credit. Their pattern
/// is a regex over file content, so a search for text shaped like an id is
/// not a request for that symbol — and the search subcommands journal what
/// they actually displayed on their own.
///
/// The overall depth is a different question: `ambits rg`/`ambits grep` do
/// the same exploratory search as bare `rg`/`grep`, so they earn the same
/// `Overview` depth those commands get — not the generic Bash fallback.
#[test]
fn tool_bash_find_pattern_credits_nothing() {
    let cfg = builtin();
    let input = serde_json::json!({ "command": "ambits -p . rg 'src/app.rs::App/render'" });
    let call = map_tool_call(&cfg, "Bash", &input, "a", "ts").unwrap();

    assert!(
        call.target_selectors.is_empty(),
        "a search pattern is not a selector, got {:?}",
        call.target_selectors
    );
    assert_eq!(
        call.read_depth,
        ReadDepth::Overview,
        "search depth matches bare rg/grep, not the generic Bash default"
    );
}

// ---------------------------------------------------------------------------
// 15. TodoWrite — first todo's content shown truncated
// ---------------------------------------------------------------------------
#[test]
fn tool_todo_write_shows_first_todo() {
    let cfg = builtin();
    let input = serde_json::json!({
        "todos": [
            { "content": "Run tests", "status": "pending", "activeForm": "Running tests" },
            { "content": "Deploy to prod", "status": "pending", "activeForm": "Deploying" }
        ]
    });
    let call = map_tool_call(&cfg, "TodoWrite", &input, "a", "ts").unwrap();
    assert_eq!(call.read_depth, ReadDepth::NameOnly);
    assert!(call.file_path.is_none());
    assert!(call.description.contains("Run tests"), "description was: {}", call.description);
}

#[test]
fn tool_todo_write_long_content_truncated() {
    let cfg = builtin();
    let long_content = "Implement the full authentication system with OAuth2 support and refresh tokens, \
        including token storage, silent renewal, logout flow, PKCE challenge, and integration \
        with the existing user profile service and role-based access control middleware";
    let input = serde_json::json!({
        "todos": [
            { "content": long_content, "status": "pending", "activeForm": "Implementing" }
        ]
    });
    let call = map_tool_call(&cfg, "TodoWrite", &input, "a", "ts").unwrap();
    assert!(call.description.contains('…'), "expected truncation, got: {}", call.description);
}

#[test]
fn tool_todo_write_empty_todos() {
    let cfg = builtin();
    // Empty todos array — description falls back to static "TodoWrite ?"
    let input = serde_json::json!({ "todos": [] });
    let call = map_tool_call(&cfg, "TodoWrite", &input, "a", "ts").unwrap();
    // No panic, description contains "TodoWrite"
    assert!(call.description.starts_with("TodoWrite"), "description was: {}", call.description);
}

// ---------------------------------------------------------------------------
// parse_log_file end-to-end: all 15 tool stanzas produce events
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
        assistant_line("Bash",            r#"{"command":"cargo build"}"#),
        assistant_line("TodoWrite",       r#"{"todos":[{"content":"Run tests","status":"pending","activeForm":"Running tests"}]}"#),
    ];

    for line in &lines {
        writeln!(tmp, "{}", line).unwrap();
    }

    let events = parse_log_file(tmp.path(), &cfg);
    let tool_names: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            ambits::ingest::SessionEvent::ToolCall(tc) => Some(tc.tool_name.as_ref()),
            _ => None,
        })
        .collect();
    assert_eq!(tool_names.len(), 15, "expected 15 events, got {}: {:?}", tool_names.len(), tool_names);
}

// ---------------------------------------------------------------------------
// [cache] stanza — coverage journal settings
// ---------------------------------------------------------------------------

fn load_cfg(toml_src: &str) -> ToolMappingConfig {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tools.toml");
    std::fs::write(&path, toml_src).unwrap();
    let (cfg, warnings) = ToolMappingConfig::load(&path);
    assert!(warnings.is_empty(), "{warnings:?}");
    cfg.expect("config should parse")
}

#[test]
fn cache_stanza_is_optional() {
    // The built-in config has no [cache]; absence must mean "defaults", not
    // "disabled", or journaling would silently never start.
    let cfg = builtin();
    assert_eq!(cfg.cache.enabled, None);
    assert_eq!(cfg.cache.flush_interval_ms, None);
}

#[test]
fn cache_stanza_parses() {
    let cfg = load_cfg("version = 1\n[cache]\nenabled = false\nflush_interval_ms = 250\n");
    assert_eq!(cfg.cache.enabled, Some(false));
    assert_eq!(cfg.cache.flush_interval_ms, Some(250));
}

#[test]
fn cache_stanza_fields_are_individually_optional() {
    let cfg = load_cfg("version = 1\n[cache]\nflush_interval_ms = 900\n");
    assert_eq!(cfg.cache.enabled, None, "unset stays unset rather than defaulting to false");
    assert_eq!(cfg.cache.flush_interval_ms, Some(900));
}

#[test]
fn user_cache_settings_survive_merge_with_builtin() {
    let user = load_cfg("version = 1\n[cache]\nflush_interval_ms = 1234\n");
    let mut warnings = Vec::new();
    let merged = ToolMappingConfig::merge(builtin(), user, &mut warnings);
    assert_eq!(merged.cache.flush_interval_ms, Some(1234));
    // Tool mappings must still come through.
    assert!(!merged.tools.is_empty());
}
