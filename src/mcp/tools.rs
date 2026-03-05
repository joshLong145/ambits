//! Helpers shared by the MCP tool implementations in `server.rs`.

use std::time::Duration;

use rmcp::model::{Content, CreateMessageRequestParam, Role, SamplingMessage};
use rmcp::service::{Peer, RoleServer};

use ambits::symbols::SymbolNode;

// ── error helpers ────────────────────────────────────────────────────────────

/// Wrap any display-able error as an MCP internal error.
pub(super) fn mcp_err(e: impl std::fmt::Display) -> rmcp::Error {
    rmcp::Error::internal_error(e.to_string(), None)
}

// ── JSON serialisation helpers ────────────────────────────────────────────────

/// Recursively convert `SymbolNode` trees to JSON.
pub(super) fn symbols_to_json(symbols: &[SymbolNode]) -> Vec<serde_json::Value> {
    symbols
        .iter()
        .map(|s| {
            serde_json::json!({
                "id":         s.id,
                "name":       s.name,
                "category":   s.category.to_string(),
                "label":      s.label,
                "line_range": [s.line_range.start, s.line_range.end],
                "children":   symbols_to_json(&s.children),
            })
        })
        .collect()
}

/// Serialize a `ProjectTree` to JSON (the type does not derive `Serialize`).
pub(super) fn project_tree_to_json(tree: &ambits::symbols::ProjectTree) -> serde_json::Value {
    serde_json::json!({
        "root": tree.root.to_string_lossy(),
        "files": tree.files.iter().map(|f| serde_json::json!({
            "file_path":     f.file_path.to_string_lossy(),
            "total_lines":   f.total_lines,
            "total_symbols": f.total_symbols(),
            "symbols":       symbols_to_json(&f.symbols),
        })).collect::<Vec<_>>(),
    })
}

/// Returns true when `s` looks like a UUID (8-4-4-4-12 hex).
pub(super) fn is_uuid(s: &str) -> bool {
    let b = s.as_bytes();
    if b.len() != 36 {
        return false;
    }
    if b[8] != b'-' || b[13] != b'-' || b[18] != b'-' || b[23] != b'-' {
        return false;
    }
    b.iter()
        .enumerate()
        .all(|(i, &c)| matches!(i, 8 | 13 | 18 | 23) || c.is_ascii_hexdigit())
}

// ── sampling helper ───────────────────────────────────────────────────────────

/// Attempt to get an AI interpretation of a coverage report via MCP sampling.
/// Returns `None` on timeout (30 s), sampling unavailable, or any error.
pub(super) async fn try_interpret_coverage(
    peer: &Peer<RoleServer>,
    raw_report: &str,
) -> Option<String> {
    let prompt = format!(
        "You are reviewing a code coverage report from ambit, a tool that tracks \
         which source files and symbols an AI coding agent read during a session.\n\n\
         Coverage report (JSON):\n```json\n{raw_report}\n```\n\n\
         Provide a concise interpretation: which parts of the codebase were \
         well-explored vs. overlooked, and any patterns worth noting. \
         Be specific about file names and percentages."
    );
    let req = CreateMessageRequestParam {
        messages: vec![SamplingMessage {
            role: Role::User,
            content: Content::text(prompt),
        }],
        system_prompt: Some(
            "You are a helpful assistant that analyses code coverage data from AI coding sessions."
                .into(),
        ),
        model_preferences: None,
        include_context: None,
        temperature: None,
        max_tokens: 1024,
        stop_sequences: None,
        metadata: None,
    };
    match tokio::time::timeout(Duration::from_secs(30), peer.create_message(req)).await {
        Ok(Ok(result)) => result.message.content.as_text().map(|t| t.text.clone()),
        _ => None,
    }
}
