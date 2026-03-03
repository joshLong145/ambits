use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;
pub mod claude;
pub mod tool_config;

use crate::tracking::ReadDepth;

/// A parsed agent tool call event.
#[derive(Debug, Clone)]
pub struct AgentToolCall {
    pub agent_id: Arc<str>,
    pub tool_name: Arc<str>,
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
    pub label: Arc<str>,
}

/// Output from a single incremental poll of an event tailer.
pub struct TailerOutput {
    pub events: Vec<AgentToolCall>,
    pub session_cleared: bool,
}

/// Maps a raw tool call (name + JSON input) to an `AgentToolCall`.
/// Implement this to plug in alternative tool-name conventions.
pub trait ToolCallMapper: Send + Sync {
    fn map_tool_call(
        &self,
        tool_name: &str,
        input: &serde_json::Value,
        agent_id: &str,
        timestamp_str: &str,
    ) -> Option<AgentToolCall>;
}

/// Stateless session-format operations: discovery, listing, batch parsing.
/// Implement this to add support for a new LLM session format.
pub trait SessionIngester: Send + Sync {
    /// Derive the log directory from a project root path.
    fn log_dir_for_project(&self, project_path: &Path) -> Option<PathBuf>;
    /// Find the ID of the most recently active session in `log_dir`.
    fn find_latest_session(&self, log_dir: &Path) -> Option<String>;
    /// List all log files belonging to `session_id` within `log_dir`.
    fn session_log_files(&self, log_dir: &Path, session_id: &str) -> Vec<PathBuf>;
    /// Parse all events from a single log file in batch.
    fn parse_log_file(&self, path: &Path) -> Vec<AgentToolCall>;
    /// Create a new incremental event tailer for the given set of files.
    fn new_tailer(&self, files: Vec<PathBuf>) -> Box<dyn EventTailer>;
}

/// Stateful incremental reader. Created via `SessionIngester::new_tailer`.
pub trait EventTailer: Send {
    fn add_file(&mut self, path: PathBuf);
    fn read_new_events(&mut self) -> TailerOutput;
}
