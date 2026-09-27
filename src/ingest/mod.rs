use std::collections::BTreeSet;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;
pub mod claude;
pub mod tool_config;

use crate::tracking::ReadDepth;

/// Whether a tool reads code or changes it (spec §2.1). Declared per tool
/// stanza in `tools.toml`; a write grants no read credit (D9).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Effect {
    #[default]
    Read,
    Write,
}

/// One diff hunk, as the session log records it. Re-exported from the
/// attribution core so ingestion and attribution share one type.
pub use crate::writes::Hunk;

/// What a write tool's result tells us about the change (spec §2.2).
///
/// File contents live here only until attribution; they are never persisted
/// or logged (spec §9.6).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WriteSource {
    /// An `Edit`: `original` is `originalFile`, which the log omits for most
    /// edits (capped near 10 KB) — then only a file-level write is possible.
    Edit {
        original: Option<String>,
        old: String,
        new: String,
        replace_all: bool,
        /// `structuredPatch`; `None` when missing or malformed, which makes
        /// the write file-level.
        hunks: Option<Vec<Hunk>>,
        user_modified: bool,
    },
    /// A `Write`: the full new `content`; `original` is null on a create.
    Write {
        original: Option<String>,
        content: String,
        create: bool,
        /// As for `Edit`; unused on a create.
        hunks: Option<Vec<Hunk>>,
        user_modified: bool,
    },
    /// A write tool whose result carries no usable text (Serena tools,
    /// `NotebookEdit`, unrecognized shapes): file-level only.
    Opaque,
}

/// A completed agent write: a write tool's call, correlated with its
/// successful result by `op` (the tool call's `tool_use_id`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteEvent {
    pub op: Arc<str>,
    pub agent_id: Arc<str>,
    pub tool_name: Arc<str>,
    pub path: PathBuf,
    pub timestamp: String,
    pub source: WriteSource,
}

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
    pub target_lines: Option<Range<u32>>,
    /// Symbol selectors named directly by the command, for tools that read
    /// code without naming a file — `ambits show <id-or-hash>...` being the
    /// case this exists for.
    ///
    /// A selector carries its own location (a symbol id embeds the path, a
    /// content hash identifies the body outright), so these are resolved
    /// against the symbol tree rather than through `file_path`, and one call
    /// can legitimately name symbols in several files. Empty for every other
    /// tool.
    ///
    /// Each carries its own depth: a single shell command may hold several
    /// invocations, one asking for definitions and another for metadata only,
    /// and they earn different credit.
    pub target_selectors: Vec<(String, ReadDepth)>,
    /// Human-readable label for the agent (e.g. "Explore parser and symbol types").
    /// Falls back to agent_id if no label could be extracted from the session log.
    pub label: Arc<str>,
    /// The tool call's `tool_use_id`, which its result line refers back to.
    /// `None` for calls whose log line carried no id.
    pub tool_use_id: Option<Arc<str>>,
    /// Read or write, from the tool's stanza. A write carries
    /// `ReadDepth::Unseen`: it grants no read credit (D9).
    pub effect: Effect,
}

/// Point-in-time ledger snapshot captured at a compaction boundary.
#[derive(Debug, Clone)]
pub struct LedgerSnapshot {
    pub tool_call_count: usize,
    pub files_accessed: BTreeSet<PathBuf>,
    pub symbols_seen: usize,
    pub seen_percent: f64,
}

/// Metadata extracted from the `compact_boundary` system record that
/// precedes the `isCompactSummary` user record in Claude Code logs.
#[derive(Debug, Clone)]
pub struct CompactionMetadata {
    /// "manual", "auto", "warning" — corresponds to the trigger that initiated the compaction.
    pub trigger: String,
    pub pre_tokens: u64,
    pub post_tokens: u64,
    pub duration_ms: u64,
}

/// A detected compaction event with summary and pre-compaction state.
#[derive(Debug, Clone)]
pub struct CompactionEvent {
    pub sequence: u32,
    pub timestamp: String,
    pub agent_id: Arc<str>,
    pub summary: String,
    pub ledger_before: LedgerSnapshot,
    /// Token counts and trigger info from the boundary record. `None` if the
    /// boundary record was missing or unparseable (older Claude Code versions).
    pub metadata: Option<CompactionMetadata>,
}

/// An ordered session event emitted by batch parsing.
#[derive(Debug, Clone)]
pub enum SessionEvent {
    ToolCall(AgentToolCall),
    Compacted {
        summary: String,
        timestamp: String,
        agent_id: Arc<str>,
        metadata: Option<CompactionMetadata>,
    },
    SessionCleared,
    /// A write tool call that completed successfully (spec §1).
    Write(WriteEvent),
}

/// A compaction event surfaced by the incremental tailer (no ledger snapshot
/// here — that's filled in by `App::process_compaction` at receipt time).
pub struct TailedCompaction {
    pub summary: String,
    pub timestamp: String,
    pub agent_id: Arc<str>,
    pub metadata: Option<CompactionMetadata>,
}

/// A batch replay of one log file: its events, and where a tailer should
/// continue so that nothing is read twice or missed.
#[derive(Default)]
pub struct FileReplay {
    pub events: Vec<SessionEvent>,
    /// Byte offset the replay read up to.
    pub offset: u64,
    /// Write calls whose results the replay had not reached (spec §1): a
    /// permission prompt or a long write can straddle the handoff.
    pub awaiting: Vec<AgentToolCall>,
}

/// What a session's replay hands to the tailer that follows it.
#[derive(Default)]
pub struct Handoff {
    /// Each log file, with the offset its replay stopped at.
    pub files: Vec<(PathBuf, u64)>,
    pub awaiting: Vec<AgentToolCall>,
}

/// Output from a single incremental poll of an event tailer.
pub struct TailerOutput {
    pub events: Vec<AgentToolCall>,
    pub compactions: Vec<TailedCompaction>,
    pub session_cleared: bool,
    /// Write tool calls whose successful result arrived in this poll.
    pub writes: Vec<WriteEvent>,
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

    /// Replay a log file for a tailer to continue from ([`Self::resume_tailer`]).
    /// Default: parse it, and hand over the file's end afterwards with
    /// nothing awaited — lines landing in between are missed, which is what
    /// implementations override this to avoid.
    fn replay_log_file(&self, path: &Path, project_root: &Path) -> FileReplay {
        let events = self.parse_log_file_with_root(path, project_root);
        let offset = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        FileReplay { events, offset, awaiting: Vec::new() }
    }

    /// A tailer that continues exactly where a replay stopped. Default: a
    /// fresh tailer from each file's current end.
    fn resume_tailer(&self, handoff: Handoff) -> Box<dyn EventTailer> {
        self.new_tailer(handoff.files.into_iter().map(|(file, _)| file).collect())
    }
}

/// Stateful incremental reader. Created via `SessionIngester::new_tailer`.
pub trait EventTailer: Send {
    fn add_file(&mut self, path: PathBuf);
    fn read_new_events(&mut self) -> TailerOutput;
}
