use std::collections::BTreeSet;
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

/// Point-in-time ledger snapshot captured at a compaction boundary.
#[derive(Debug, Clone)]
pub struct LedgerSnapshot {
    pub tool_call_count: usize,
    pub files_accessed: BTreeSet<PathBuf>,
    pub symbols_seen: usize,
    pub seen_percent: f64,
}

/// A detected compaction event with summary and pre-compaction state.
#[derive(Debug, Clone)]
pub struct CompactionEvent {
    pub sequence: u32,
    pub timestamp: String,
    pub agent_id: Arc<str>,
    pub summary: String,
    pub ledger_before: LedgerSnapshot,
}

/// An ordered session event emitted by batch parsing.
#[derive(Debug, Clone)]
pub enum SessionEvent {
    ToolCall(AgentToolCall),
    Compacted { summary: String, timestamp: String, agent_id: Arc<str> },
    SessionCleared,
}

/// A compaction event surfaced by the incremental tailer (no ledger snapshot
/// here — that's filled in by `App::process_compaction` at receipt time).
pub struct TailedCompaction {
    pub summary: String,
    pub timestamp: String,
    pub agent_id: Arc<str>,
}

/// Output from a single incremental poll of an event tailer.
pub struct TailerOutput {
    pub events: Vec<AgentToolCall>,
    pub compactions: Vec<TailedCompaction>,
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
    fn parse_log_file(&self, path: &Path) -> Vec<SessionEvent>;
    /// Create a new incremental event tailer for the given set of files.
    fn new_tailer(&self, files: Vec<PathBuf>) -> Box<dyn EventTailer>;

    /// Return the human-readable slug for a session (e.g. "crispy-crunching-nova").
    /// Default returns None; implementations override this for slug-carrying formats.
    fn session_slug(&self, log_dir: &Path, session_id: &str) -> Option<String> {
        let _ = (log_dir, session_id);
        None
    }

    /// Parse all events from a log file, remapping paths when the agent ran in a
    /// worktree whose cwd differs from `project_root`.
    /// Default delegates to `parse_log_file` (no remapping).
    fn parse_log_file_with_root(&self, path: &Path, project_root: &Path) -> Vec<SessionEvent> {
        let _ = project_root;
        self.parse_log_file(path)
    }
}

/// Stateful incremental reader. Created via `SessionIngester::new_tailer`.
pub trait EventTailer: Send {
    fn add_file(&mut self, path: PathBuf);
    fn read_new_events(&mut self) -> TailerOutput;
}
