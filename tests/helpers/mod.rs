use std::path::PathBuf;

use crate::ingest::AgentToolCall;
use crate::symbols::{FileSymbols, ProjectTree, SymbolCategory, SymbolNode};
use crate::tracking::ReadDepth;

/// Create a mock SymbolNode for testing.
/// The `file_path` is left empty; `file()` sets it to match the parent `FileSymbols`.
pub fn sym(id: &str, name: &str) -> SymbolNode {
    let hash = crate::symbols::merkle::content_hash(name);
    SymbolNode {
        id: id.to_string(),
        name: name.to_string(),
        category: SymbolCategory::Function,
        label: "fn".to_string(),
        file_path: PathBuf::new(),
        byte_range: 0..100,
        line_range: 1..10,
        content_hash: hash,
        merkle_hash: hash,
        children: Vec::new(),
        estimated_tokens: 30,
    }
}

/// Create a SymbolNode with children.
pub fn sym_with_children(id: &str, name: &str, children: Vec<SymbolNode>) -> SymbolNode {
    let mut s = sym(id, name);
    s.children = children;
    s
}

/// Create a SymbolNode with a custom line range.
pub fn sym_with_lines(id: &str, name: &str, start: usize, end: usize) -> SymbolNode {
    let mut s = sym(id, name);
    s.line_range = start..end;
    s
}

/// Create a FileSymbols entry, setting each symbol's `file_path` to match.
pub fn file(path: &str, symbols: Vec<SymbolNode>) -> FileSymbols {
    let file_path = PathBuf::from(path);
    let symbols = symbols
        .into_iter()
        .map(|mut s| {
            s.file_path = file_path.clone();
            s
        })
        .collect();
    FileSymbols {
        file_path,
        symbols,
        total_lines: 100,
    }
}

/// Create a ProjectTree from files.
pub fn project(files: Vec<FileSymbols>) -> ProjectTree {
    ProjectTree {
        root: PathBuf::from("/test/project"),
        files,
    }
}

/// Create a basic AgentToolCall.
pub fn tool_call(tool: &str, path: &str, depth: ReadDepth) -> AgentToolCall {
    AgentToolCall {
        agent_id: "agent-1".to_string(),
        tool_name: tool.to_string(),
        file_path: Some(PathBuf::from(path)),
        read_depth: depth,
        description: format!("{tool} {path}"),
        timestamp_str: "2025-01-01T00:00:00Z".to_string(),
        target_symbol: None,
        target_lines: None,
    }
}

/// Create a tool call targeting a specific symbol.
pub fn tool_call_targeted(tool: &str, path: &str, depth: ReadDepth, symbol: &str) -> AgentToolCall {
    let mut tc = tool_call(tool, path, depth);
    tc.target_symbol = Some(symbol.to_string());
    tc
}

/// Create a tool call targeting a line range.
pub fn tool_call_lines(tool: &str, path: &str, depth: ReadDepth, start: usize, end: usize) -> AgentToolCall {
    let mut tc = tool_call(tool, path, depth);
    tc.target_lines = Some(start..end);
    tc
}

/// Build a JSONL assistant message with a single tool_use block.
pub fn jsonl_assistant(tool_name: &str, input_json: &str) -> String {
    format!(
        r#"{{"type":"assistant","sessionId":"test-session","timestamp":"2025-01-01T00:00:00Z","message":{{"role":"assistant","content":[{{"type":"tool_use","name":"{tool_name}","input":{input_json}}}]}}}}"#
    )
}

/// Build a JSONL user message (should be ignored by parser).
pub fn jsonl_user_msg() -> String {
    r#"{"type":"user","message":{"role":"user","content":"hello"}}"#.to_string()
}
