use std::ops::Range;
pub mod claude;

use std::path::PathBuf;

use crate::tracking::ReadDepth;

/// A parsed agent tool call event.
#[derive(Debug, Clone)]
pub struct AgentToolCall {
    pub agent_id: String,
    pub tool_name: String,
    pub file_path: Option<PathBuf>,
    pub read_depth: ReadDepth,
    pub description: String,
    pub timestamp_str: String,
    /// Optional symbol name path to target (e.g. "MyClass/my_method").
    pub target_symbol: Option<String>,
    /// Optional line range to target (1-based, e.g. 10..25).
    pub target_lines: Option<Range<usize>>,
    /// Human-readable label for the agent (e.g. "Explore parser and symbol types").
    /// Falls back to agent_id if no label could be extracted from the session log.
    pub label: String,
}

/// Trait for agent event sources.
/// Implement this to support different agent frameworks.
pub trait AgentEventSource {
    /// Parse all events from existing log files.
    fn parse_existing(&self) -> color_eyre::Result<Vec<AgentToolCall>>;
}
