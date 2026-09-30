use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind};

use crate::coverage::count_symbols;
use crate::expansion::{Expansion, RowKind};
use crate::filter::PathFilter;
use crate::symbols::{ProjectTree, SymbolNode};
use crate::tracking::ReadDepth;
use crate::tracking::ContextLedger;
use crate::tracking::agents::{AgentTree, AgentNode};
use crate::ingest::AgentToolCall;

/// How files are sorted in the tree view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortMode {
    Alphabetical,
    ByCoverage,
}

/// Four-state coverage classification for files.
/// Variant order gives the desired sort: Partially → AllSeen → Fully → Not Covered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FileCoverageStatus {
    PartiallyCovered,
    AllSeen,
    FullyCovered,
    NotCovered,
}

/// A flattened row in the tree view, ready for rendering.
#[derive(Debug, Clone)]
pub struct TreeRow {
    pub symbol_id: String,
    pub display_name: String,
    pub label: &'static str,  // Language-specific label (e.g., "class", "def", "fn")
    pub depth: usize,         // nesting depth for indentation
    pub kind: RowKind,
    pub is_expanded: bool,
    pub has_children: bool,
    pub line_range: String,
    pub token_count: usize,
    pub read_depth: ReadDepth,
    /// Content changed since this symbol was read. Orthogonal to `read_depth`.
    pub stale: bool,
    /// Read predates a compaction, so the model no longer holds it in context.
    pub restored: bool,
    /// Coverage of what the row stands for but does not show: the whole file
    /// for a file row, the descendants of a collapsed symbol. `None` for any
    /// other symbol row, whose own color already tells the whole story.
    pub coverage_status: Option<FileCoverageStatus>,
    pub coverage_seen: usize,
    pub coverage_total: usize,
    /// Of `coverage_total`, how many were read in full and are unchanged.
    pub coverage_full: usize,
    /// Of what the row summarizes, how many read symbols changed since.
    pub stale_count: usize,
    /// How this session's agents wrote the row (the filtered agent's, when
    /// one is): a symbol through itself or anything nested in it, a file
    /// through any write to it.
    pub write: Option<crate::write_index::WriteMark>,
}

impl TreeRow {
    /// True for file headers.
    pub fn is_file(&self) -> bool {
        self.kind == RowKind::File
    }
}

/// One load of a call's content: the call, and whether it had ended. Its
/// result reaches the log when it ends, so what was loaded while it ran is
/// asked for again once it has.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContentKey {
    pub id: Arc<str>,
    pub ended: bool,
}

/// The selected call's content, loaded off the render thread from its
/// agent's log and held in memory only: never written (spec §9.6).
#[derive(Debug, Default)]
pub struct CallContents {
    loaded: Option<(ContentKey, Option<Arc<crate::ingest::content::CallDetail>>)>,
    asked: Option<ContentKey>,
}

/// Where a call's content stands.
#[derive(Debug, Clone, Copy)]
pub enum ContentState<'a> {
    Loading,
    /// The log has none: no id, no log, or not there.
    Missing,
    Loaded(&'a crate::ingest::content::CallDetail),
}

/// The full-width view of a call's content (`o`).
#[derive(Debug)]
pub struct ContentView {
    pub span: usize,
    /// The first row shown.
    pub scroll: usize,
    /// Rows and columns the view last had room for, set while rendering.
    pub height: std::cell::Cell<usize>,
    pub width: std::cell::Cell<usize>,
}

/// Where the trace view drew its time axis and rows, so a mouse position
/// maps to a moment and a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceGeometry {
    pub bars_x: u16,
    pub bars_width: u16,
    pub rows_y: u16,
    pub rows: u16,
    /// The row index drawn first (the scroll offset).
    pub first_row: usize,
}

/// What the right-hand pane shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RightPane {
    /// The selected row's states in words, and the traces that touched it.
    #[default]
    Inspector,
    /// Session totals, agents and compactions.
    Session,
}

/// What the trace view's right-hand panel is about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PanelSubject {
    /// A whole trace (its root span): on the list, or its prompt selected.
    Trace(usize),
    /// One call in the open trace.
    Call(usize),
    /// A commit or compaction.
    Instant(usize),
    Nothing,
}

/// Which panel is focused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusPanel {
    /// The file tree, or the trace view in its place.
    Left,
    /// The inspector, the session pane, or the trace panel.
    Right,
    /// The activity feed, when shown.
    Feed,
}

/// How the TUI journals a session (see [`App::attach_journal`]).
#[derive(Debug, Clone)]
pub struct JournalSettings {
    /// Symbol backend, recorded in the journal header.
    pub backend: &'static str,
    pub interval: std::time::Duration,
}

/// What attaching a session's journal found.
#[derive(Debug, Default)]
pub struct JournalAttach {
    pub rehydrated: Option<crate::restore::RehydrateStats>,
    /// Non-fatal complaints from opening the journal.
    pub warnings: Vec<String>,
}

pub struct App {
    pub project_tree: ProjectTree,
    pub project_root: PathBuf,
    pub ledger: ContextLedger,
    pub should_quit: bool,

    // Tree view state.
    pub tree_rows: Vec<TreeRow>,
    pub selected_index: usize,
    /// Private: every change must rebuild `tree_rows`, so it goes through
    /// [`App::set_expanded`].
    expansion: Expansion,

    // Activity feed.
    pub activity: Vec<AgentToolCall>,
    /// How many lines the user has scrolled up from the bottom in the activity feed (0 = pinned to latest).
    pub activity_scroll_offset: usize,

    // Agents seen.
    pub agents_seen: Vec<String>,

    // Agent hierarchy.
    pub agent_tree: AgentTree,

    // Agent filter: if Some, only show coverage from this agent.
    pub agent_filter: Option<String>,
    /// Selection index in the agent list (0 = All, 1..N = specific agent).
    pub agent_selection_index: usize,

    // Focus.
    pub focus: FocusPanel,

    // Sort mode for tree view.
    pub sort_mode: SortMode,

    // Search.
    pub search_mode: bool,
    pub search_query: String,

    // Session info for display.
    pub session_id: Option<String>,
    pub session_slug: Option<String>,

    // Compaction tracking.
    pub compaction_history: Vec<crate::ingest::CompactionEvent>,
    pub compaction_call_count: usize,
    pub show_compaction_overlay: bool,
    pub compaction_overlay_index: usize,

    // Sub-agent alignment popup (see `tracking::alignment`).
    /// Whether the alignment popup is currently shown.
    pub show_alignment_overlay: bool,
    /// Pairwise alignment scores for the selected agent's sibling group,
    /// computed once when the popup is opened (not recomputed per frame).
    pub agent_alignment: Vec<crate::tracking::alignment::PairAlignment>,
    /// Standalone precomputed `(symbol, agent) -> depth ordinal` cache, kept
    /// in lockstep with `ledger` at the same `mark_file_symbols` /
    /// `mark_targeted_symbols` call sites. See `tracking::alignment` module
    /// docs for why this is a separate structure rather than reading depths
    /// back out of `ledger`.
    pub depth_cache: crate::tracking::alignment::DepthOrdinalCache,

    /// Durable record of which symbols were read and what they looked like at
    /// the time. `None` when journaling is disabled, or when no session id is
    /// known (the journal is keyed by session, and guessing a filename would
    /// silently merge unrelated sessions). See `crate::journal`.
    pub journal: Option<crate::journal::Journal>,
    /// The previous session's journal, kept after a switch so a write still
    /// in the attribution worker at that moment lands in the session it
    /// happened in. Replaced at the next switch.
    retired_journal: Option<crate::journal::Journal>,
    /// How to open a session's journal; `None` when journaling is off. Set
    /// once at startup, so a session switch can reopen without the caller
    /// re-deriving the settings.
    journal_settings: Option<JournalSettings>,
    /// Write events awaiting attribution, each tagged with the session it
    /// happened in. Every source — startup replay, a session switch, the
    /// tailer — queues here, and the TUI drains the queue to its attribution
    /// worker each tick, so parsing never runs on the render thread.
    pending_writes: Vec<(Arc<str>, crate::ingest::WriteEvent)>,
    /// Every tool call as a span, and compactions as instants, for the trace
    /// view. Its own store: the activity feed keeps only the latest calls.
    pub trace: crate::trace::Trace,
    /// This session's writes, for the tree's marks.
    pub writes: crate::write_index::WriteIndex,
    /// The trace view (`t`): layout, zoom and selection.
    pub trace_view: crate::trace::view::TraceView,
    /// Where the trace view last drew its rows, for the mouse. Set while
    /// rendering, which only borrows the app.
    pub trace_geometry: std::cell::Cell<Option<TraceGeometry>>,
    /// The column a drag on the trace started from.
    trace_drag: Option<u16>,
    /// What the right-hand pane shows (`i` switches).
    pub right_pane: RightPane,
    /// Whether the activity feed is shown (`f`).
    pub show_activity: bool,
    /// The row selected in the right-hand panel (the inspector's traces,
    /// the trace panel's rows).
    pub panel_index: usize,
    /// The selected call's content, for the trace panel and `o`.
    pub contents: CallContents,
    /// The call's content full-width, when open (`o`).
    pub content_view: Option<ContentView>,
    /// Each written file's symbols as write statuses compare them, kept
    /// between frames: see [`App::file_contents`].
    file_contents: std::cell::RefCell<HashMap<String, (blake3::Hash, Arc<crate::writes::FileContents>)>>,

    /// Path filter restricting which files are tracked, if any. Shared with
    /// the TUI re-parse paths (file watcher, Serena cache rescan) so that
    /// changes to excluded files don't inject symbols back into the tree
    /// after the initial filtered scan. `None` means no filter — track
    /// everything.
    pub filter: Option<Arc<PathFilter>>,

    /// Resolved "open in editor" command template (see [`crate::editor`]).
    /// `None` means no editor could be resolved from CLI/config/env.
    pub editor_template: Option<String>,
    /// Set by [`App::open_selected_in_editor`] and consumed by the TUI loop,
    /// which owns the terminal handle `App` does not have — this is how the
    /// request to suspend the TUI and launch an editor gets signaled up.
    pub pending_editor_request: Option<(PathBuf, u32)>,
    /// Set when the last "open in editor" attempt failed (no editor
    /// resolved, spawn failure, nonzero exit). Cleared on the next keypress.
    pub last_editor_error: Option<String>,
}

impl App {
    pub fn new(project_tree: ProjectTree, project_root: PathBuf) -> Self {
        let mut app = Self {
            project_tree,
            project_root,
            ledger: ContextLedger::new(),
            should_quit: false,
            tree_rows: Vec::new(),
            selected_index: 0,
            expansion: Expansion::default(),
            activity: Vec::new(),
            activity_scroll_offset: 0,
            agents_seen: Vec::new(),
            agent_tree: AgentTree::new(),
            agent_filter: None,
            agent_selection_index: 0,
            focus: FocusPanel::Left,
            sort_mode: SortMode::Alphabetical,
            search_mode: false,
            search_query: String::new(),
            session_id: None,
            session_slug: None,
            compaction_history: Vec::new(),
            compaction_call_count: 0,
            show_compaction_overlay: false,
            compaction_overlay_index: 0,
            show_alignment_overlay: false,
            agent_alignment: Vec::new(),
            depth_cache: crate::tracking::alignment::DepthOrdinalCache::new(),
            journal: None,
            retired_journal: None,
            journal_settings: None,
            pending_writes: Vec::new(),
            trace: crate::trace::Trace::default(),
            writes: crate::write_index::WriteIndex::default(),
            trace_view: crate::trace::view::TraceView::default(),
            trace_geometry: std::cell::Cell::new(None),
            trace_drag: None,
            right_pane: RightPane::Inspector,
            show_activity: false,
            panel_index: 0,
            contents: CallContents::default(),
            content_view: None,
            file_contents: Default::default(),
            filter: None,
            editor_template: None,
            pending_editor_request: None,
            last_editor_error: None,
        };
        app.rebuild_tree_rows();
        app
    }

    /// Set the resolved "open in editor" command template. Separate from
    /// [`App::new`] rather than a constructor parameter: `App::new` has
    /// several call sites across tests, and threading one more optional
    /// argument through all of them for a setting that's `None` in every
    /// test is unnecessary churn.
    pub fn set_editor_template(&mut self, template: Option<String>) {
        self.editor_template = template;
    }

    /// Start journaling this session's reads to `.ambits/coverage/`.
    ///
    /// Deliberately separate from [`App::set_session_id`], which already does
    /// double duty seeding the agent tree — opening a file is a side effect
    /// callers should ask for explicitly. Returns any non-fatal complaints
    /// (corrupt existing lines, unwritable directory) for the caller to
    /// surface; a journal that cannot be written must never be fatal.
    pub fn enable_journal(&mut self, backend: &str, interval: std::time::Duration) -> Vec<String> {
        let Some(session_id) = self.session_id.clone() else {
            return vec![
                "no session id resolved; coverage journaling disabled".to_string()
            ];
        };
        let journal = crate::journal::Journal::open(
            &self.project_root,
            &session_id,
            interval,
            || {
                crate::journal::EnvironmentManifest::capture(
                    &self.project_tree,
                    backend,
                    // Same display form the coverage report records
                    // (`CoverageReport.filter`), so the two agree on what
                    // "this run was filtered" means.
                    self.filter.as_ref().map(|f| f.display()),
                )
            },
        );
        let warnings = journal.warnings().to_vec();
        self.journal = Some(journal);
        warnings
    }

    /// Journal every session from now on with these settings (`None`: off).
    pub fn set_journal_settings(&mut self, settings: Option<JournalSettings>) {
        self.journal_settings = settings;
    }

    /// Adopt the current session's journal: fold it into the replayed
    /// ledger, open it for appending, and sync what the replay found.
    ///
    /// The one way a session's journal is opened, at startup and on every
    /// session switch, so both keep the order `rehydrate_from_journal`
    /// requires: after the log replay, before opening for writing. A no-op
    /// when journaling is off.
    pub fn attach_journal(&mut self) -> JournalAttach {
        let Some(settings) = self.journal_settings.clone() else {
            return JournalAttach::default();
        };
        let rehydrated = self.rehydrate_from_journal();
        let warnings = self.enable_journal(settings.backend, settings.interval);
        self.sync_journal();
        self.load_writes();
        JournalAttach { rehydrated, warnings }
    }

    /// Fold the session's journaled writes into the index: what earlier runs
    /// attributed, which this run's replay only re-attributes later, if at
    /// all.
    fn load_writes(&mut self) {
        let Some(session) = self.session_id.as_deref() else { return };
        let journaled = crate::write_index::WriteIndex::load(&crate::journal::journal_dir(&self.project_root), session);
        let had = !self.writes.is_empty() || !journaled.is_empty();
        for record in journaled.into_records() {
            self.writes.insert(record);
        }
        if had {
            self.rebuild_tree_rows();
        }
    }

    /// Fold this session's journal into the freshly replayed ledger.
    ///
    /// Must run after the startup log replay and before the journal is opened
    /// for writing. See [`crate::restore::rehydrate_ledger`] for why replaying
    /// the log alone leaves every historical read looking current.
    ///
    /// Also feeds the restored per-agent depths into `depth_cache`, so the
    /// alignment popup sees the same coverage the tree view does rather than
    /// scoring a cold-started session as if no agent had read anything.
    pub fn rehydrate_from_journal(&mut self) -> Option<crate::restore::RehydrateStats> {
        let session_id = self.session_id.clone()?;
        let dir = crate::journal::journal_dir(&self.project_root);
        let contents = crate::journal::read_journal_session(&dir, &session_id);
        if contents.reads.is_empty() {
            return None;
        }

        let stats = crate::restore::rehydrate_ledger(
            &mut self.ledger,
            &contents,
            // v1 journals carry no attribution; the session's own id is the
            // closest true label available, and matches how Claude Code names
            // the root agent's own events.
            &session_id,
            &self.project_tree,
        );

        for ((symbol_id, agent), (_, depth)) in &contents.agent_reads {
            self.depth_cache.record(symbol_id, agent, *depth);
            // The journal outlives the logs it was built from, so it can name
            // an agent this run's replay never produced an event for. Register
            // it: the panel lists agents, not ledger keys, and coverage the
            // panel cannot attribute to a row still lands in "[All]".
            self.register_agent(agent, agent);
        }

        self.rebuild_tree_rows();
        Some(stats)
    }

    /// Bring the journal up to date if its flush interval has elapsed.
    /// Cheap and safe to call every tick.
    pub fn maybe_sync_journal(&mut self) {
        if let Some(journal) = self.journal.as_mut() {
            journal.maybe_sync(&self.ledger);
        }
    }

    /// Bring the journal up to date now, ignoring the interval. Called before
    /// exit so the tail of a session isn't lost.
    pub fn sync_journal(&mut self) {
        if let Some(journal) = self.journal.as_mut() {
            journal.sync(&self.ledger);
        }
    }

    /// Set the resolved session ID and deterministically seed the agent
    /// hierarchy's root node from it, *before* any tool-call events are
    /// processed.
    ///
    /// Without this, `agent_tree.root_id` is only discovered lazily as
    /// events stream in (see `process_agent_event`), which is order-
    /// dependent: if the orchestrator/root session never emits a file-tool
    /// event of its own — common for orchestrator-only sessions that only
    /// dispatch `Task` calls — the first sub-agent event processed would be
    /// mistaken for the root, corrupting every subsequent sibling
    /// relationship (and silently breaking the sub-agent alignment popup,
    /// whose sibling lookup walks `parent_id`). Seeding here guarantees the
    /// *true* root is always known first, regardless of event arrival order.
    pub fn set_session_id(&mut self, session_id: Option<String>) {
        self.session_id = session_id;
        self.seed_agent_tree_root();
    }

    /// Register `self.session_id` as the agent hierarchy's root node, if not
    /// already present. No-op when `session_id` is `None` (falls back to the
    /// existing lazy root-inference in `process_agent_event`).
    fn seed_agent_tree_root(&mut self) {
        if let Some(sid) = self.session_id.clone() {
            if !self.agent_tree.agents.contains_key(&sid) {
                self.agent_tree.add_agent(AgentNode {
                    id: sid,
                    parent_id: None,
                    session_file: PathBuf::new(),
                    label: "main".to_string(),
                });
            }
        }
    }

    /// Reset all live session state (ledger, agents, activity) while preserving
    /// the project tree and UI configuration. Called when a `/clear` is detected
    /// in the session log.
    pub fn reset_session(&mut self) {
        // Journal what the ledger holds before it is discarded.
        self.sync_journal();
        self.ledger = ContextLedger::new();
        self.activity.clear();
        self.agents_seen.clear();
        self.agent_tree = AgentTree::new();
        self.seed_agent_tree_root();
        self.agent_filter = None;
        self.agent_selection_index = 0;
        self.session_slug = None;
        self.compaction_history.clear();
        self.trace.clear();
        self.contents = CallContents::default();
        self.content_view = None;
        self.trace_view.reset();
        self.compaction_call_count = 0;
        self.show_alignment_overlay = false;
        self.agent_alignment.clear();
        self.depth_cache = crate::tracking::alignment::DepthOrdinalCache::new();
        self.rebuild_tree_rows();
    }

    /// Adopt a new session identity, discarding everything tied to the old one.
    ///
    /// Ordering is the whole point of this method. [`Self::reset_session`]
    /// rebuilds `agent_tree` from scratch and re-seeds its root from
    /// `self.session_id`, while [`AgentTree::add_agent`] latches `root_id` on
    /// the first parentless node it ever sees. Resetting *before* adopting the
    /// new id therefore seeds the fresh tree with the session that just ended
    /// and pins `root_id` to it permanently; the live session is then added as
    /// a second parentless node, and the agent panel renders two rows both
    /// labelled `main` — one of them a corpse.
    ///
    /// Callers switching sessions must use this rather than `reset_session` +
    /// [`Self::set_session_id`]. A bare `reset_session` remains correct for a
    /// `/clear` *within* one session, where the identity does not change.
    ///
    /// The outgoing journal is synced before the reset discards the ledger it
    /// syncs from, then retired rather than dropped: writes from the old
    /// session may still be in the attribution worker. The caller replays the
    /// new session and then calls [`Self::attach_journal`].
    pub fn switch_session(&mut self, session_id: Option<String>) {
        self.sync_journal();
        self.retired_journal = self.journal.take();
        self.session_id = session_id;
        // Writes are facts about files, so a `/clear` keeps them; a new
        // session starts without.
        self.writes.clear();
        self.reset_session();
    }

    /// Snapshot the current ledger state, record a compaction event, then
    /// demote every ledger entry to [`Provenance::Restored`].
    ///
    /// This used to wipe the ledger outright, on the grounds that the summary
    /// doesn't reliably describe what the model retained, so pre-compaction
    /// depth claims would over-report coverage. That reasoning was sound while
    /// the alternative was silently presenting stale reads as live — but it
    /// threw away real, verifiable information (which symbols were read, and
    /// whether they have changed since) to avoid a display problem.
    ///
    /// Marking instead of wiping keeps that information and fixes the display
    /// problem directly: restored entries render distinctly and are counted
    /// separately, so coverage survives compaction without ever being claimed
    /// as freshly read. Any genuine re-read flips the entry back to `Live`.
    ///
    /// `compaction_history`, `activity`, and agent-tracking state are
    /// preserved; only the inter-compaction tool-call counter is reset.
    pub fn process_compaction(
        &mut self,
        summary: String,
        timestamp: String,
        agent_id: std::sync::Arc<str>,
        metadata: Option<crate::ingest::CompactionMetadata>,
    ) {
        if let Some(t) = crate::time::parse_rfc3339_millis(&timestamp) {
            self.trace.instant(t, Some(agent_id.clone()), crate::trace::InstantKind::Compaction);
        }
        use std::collections::BTreeSet;
        let files_before: BTreeSet<std::path::PathBuf> = self
            .project_tree
            .files
            .iter()
            .filter(|f| f.symbols.iter().any(|sym| self.ledger.depth_of(&sym.id).is_seen()))
            .map(|f| f.file_path.clone())
            .collect();

        let total = self.project_tree.total_symbols();
        let seen = self.ledger.total_seen();

        let snapshot = crate::ingest::LedgerSnapshot {
            tool_call_count: self.compaction_call_count,
            files_accessed: files_before,
            symbols_seen: seen,
            seen_percent: if total > 0 {
                seen as f64 / total as f64 * 100.0
            } else {
                0.0
            },
        };

        let sequence = self.compaction_history.len() as u32 + 1;
        self.compaction_history.push(crate::ingest::CompactionEvent {
            sequence,
            timestamp,
            agent_id,
            summary,
            ledger_before: snapshot,
            metadata,
        });
        self.compaction_overlay_index = self.compaction_history.len().saturating_sub(1);

        // Demote rather than wipe: the reads are still facts, they just no
        // longer live in the model's context. `depth_cache` is deliberately
        // left intact — it backs the sub-agent alignment popup, which compares
        // what agents *read*, a question compaction doesn't change.
        self.ledger.mark_all_restored();
        self.compaction_call_count = 0;
        self.rebuild_tree_rows();
    }

    /// Rebuild the flattened tree rows from the project tree + expansion state.
    pub fn rebuild_tree_rows(&mut self) {
        let mut rows = Vec::new();
        let agent_filter = self.agent_filter.as_deref();

        // Build iteration order: sorted by coverage status if ByCoverage mode is active.
        let file_indices: Vec<usize> = if self.sort_mode == SortMode::ByCoverage {
            let mut indices: Vec<(FileCoverageStatus, &std::path::Path, usize)> = self
                .project_tree
                .files
                .iter()
                .enumerate()
                .map(|(i, f)| {
                    let (total, seen, full) = count_symbols(&f.symbols, &self.ledger, agent_filter);
                    (
                        coverage_status_from_counts(total, seen, full),
                        f.file_path.as_path(),
                        i,
                    )
                })
                .collect();
            indices.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(b.1)));
            indices.into_iter().map(|(_, _, i)| i).collect()
        } else {
            (0..self.project_tree.files.len()).collect()
        };

        let mut writes = self.writes.by_file(agent_filter);
        for &idx in &file_indices {
            let file = &self.project_tree.files[idx];
            let file_path = file.file_path.to_string_lossy().to_string();
            // Writes name files normalized; only look when there are any.
            let file_writes = (!writes.is_empty())
                .then(|| writes.remove(crate::objects::normalize_path(&file_path).as_str()))
                .flatten()
                .map(|w| crate::write_index::FileWrites::new(w, file));
            let file_id = file_path.clone();
            let is_expanded = self.expansion.is_expanded(&file_id, RowKind::File);

            let (total, seen, full) = count_symbols(&file.symbols, &self.ledger, agent_filter);
            let status = coverage_status_from_counts(total, seen, full);
            let file_read_depth = if status != FileCoverageStatus::NotCovered {
                ReadDepth::NameOnly // Use NameOnly to indicate "has coverage"
            } else {
                ReadDepth::Unseen
            };

            rows.push(TreeRow {
                symbol_id: file_id.clone(),
                display_name: file_path.clone(),
                label: "",
                depth: 0,
                kind: RowKind::File,
                is_expanded,
                has_children: !file.symbols.is_empty(),
                line_range: format!("{} lines", file.total_lines),
                token_count: 0,
                read_depth: file_read_depth,
                // File rows are colored by coverage status, not depth/staleness.
                stale: false,
                restored: false,
                coverage_status: Some(status),
                coverage_seen: seen,
                coverage_total: total,
                coverage_full: full,
                stale_count: count_stale(&file.symbols, &self.ledger),
                write: file_writes.as_ref().and_then(|w| w.file_mark()),
            });

            if is_expanded {
                let context = RowContext { expansion: &self.expansion, ledger: &self.ledger, agent_filter, writes: file_writes.as_ref() };
                for sym in &file.symbols {
                    flatten_symbol(sym, 1, &context, &mut rows);
                }
            }
        }

        self.tree_rows = rows;
    }

    pub fn handle_key(&mut self, key: KeyEvent) {
        self.last_editor_error = None;

        if self.search_mode {
            self.handle_search_key(key);
            return;
        }
        if self.trace_view.typing {
            self.handle_trace_key(key);
            return;
        }
        if self.content_view.is_some() {
            self.handle_content_key(key);
            return;
        }
        // The same in every view.
        let overlay = self.show_compaction_overlay || self.show_alignment_overlay;
        match key.code {
            KeyCode::Char('q') => return self.should_quit = true,
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => return self.should_quit = true,
            KeyCode::Tab => return self.cycle_focus(true),
            KeyCode::BackTab => return self.cycle_focus(false),
            KeyCode::Char('[') if !overlay => return self.cycle_agent_filter_backward(),
            KeyCode::Char(']') if !overlay => return self.cycle_agent_filter(),
            KeyCode::Char('i') => {
                self.right_pane = match self.right_pane {
                    RightPane::Inspector => RightPane::Session,
                    RightPane::Session => RightPane::Inspector,
                };
                return;
            }
            KeyCode::Char('o') if self.trace_view.open && !overlay => return self.open_content_view(),
            KeyCode::Char('f') => {
                self.show_activity = !self.show_activity;
                if !self.show_activity && self.focus == FocusPanel::Feed {
                    self.focus = FocusPanel::Left;
                }
                return;
            }
            KeyCode::Esc if self.focus != FocusPanel::Left && !overlay => return self.focus = FocusPanel::Left,
            _ => {}
        }
        match self.focus {
            FocusPanel::Right => self.handle_right_key(key),
            FocusPanel::Feed => self.handle_feed_key(key),
            FocusPanel::Left if self.trace_view.open => {
                let before = (self.trace_view.selected, self.trace_view.list, self.trace_view.focus);
                self.handle_trace_key(key);
                if (self.trace_view.selected, self.trace_view.list, self.trace_view.focus) != before {
                    self.panel_index = 0;
                }
            }
            FocusPanel::Left => self.handle_tree_key(key),
        }
    }

    /// The right-hand panel's keys, by what it shows.
    fn handle_right_key(&mut self, key: KeyEvent) {
        match (self.right_pane, self.trace_view.open) {
            (RightPane::Session, _) => self.handle_session_key(key),
            (RightPane::Inspector, true) => self.handle_panel_rows_key(key, self.trace_panel_targets().len(), Self::open_panel_target),
            (RightPane::Inspector, false) => self.handle_panel_rows_key(key, usize::MAX, Self::open_inspected_trace),
        }
    }

    /// A panel of rows — the trace panel's, or the inspector's traces —
    /// moved through with `j`/`k` and opened with `Enter`.
    fn handle_panel_rows_key(&mut self, key: KeyEvent, rows: usize, open: fn(&mut Self)) {
        match key.code {
            KeyCode::Char('j') | KeyCode::Down => self.panel_index = (self.panel_index + 1).min(rows.saturating_sub(1)),
            KeyCode::Char('k') | KeyCode::Up => self.panel_index = self.panel_index.saturating_sub(1),
            KeyCode::Enter => open(self),
            _ => {}
        }
    }

    /// The session pane's keys: pick an agent to filter by.
    fn handle_session_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('j') | KeyCode::Down => self.move_agent_selection(1),
            KeyCode::Char('k') | KeyCode::Up => self.move_agent_selection(-1),
            KeyCode::Char('l') | KeyCode::Right | KeyCode::Enter => self.apply_agent_selection(),
            _ => {}
        }
    }

    /// The activity feed's keys: scroll it.
    fn handle_feed_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('k') | KeyCode::Up => self.activity_scroll_offset = self.activity_scroll_offset.saturating_add(1),
            KeyCode::Char('j') | KeyCode::Down => self.activity_scroll_offset = self.activity_scroll_offset.saturating_sub(1),
            KeyCode::Char('G') => self.activity_scroll_offset = 0,
            _ => {}
        }
    }

    /// `Enter` on a trace panel row: a file shows in the tree; a call or a
    /// moment is selected in its trace's timeline — opened, and unfolded
    /// down to it, as needed. Focus stays on the panel, to go on from there.
    fn open_panel_target(&mut self) {
        use crate::trace::summary::Target;
        use crate::trace::view::Item;
        let targets = self.trace_panel_targets();
        let Some(target) = targets.get(self.panel_index.min(targets.len().saturating_sub(1))).cloned() else { return };
        let item = match target {
            Target::File(file) => {
                if self.reveal(&file, None) {
                    self.trace_view.open = false;
                    self.focus = FocusPanel::Left;
                }
                return;
            }
            Target::Symbol(id) => {
                // In the tree, its file and the symbols around it unfolded.
                let (file, _) = crate::symbols::split_id(&id);
                if self.reveal_id(file, Some(id.clone())) {
                    self.trace_view.open = false;
                    self.focus = FocusPanel::Left;
                }
                return;
            }
            Target::Span(i) => Item::Span(i),
            Target::Instant(i) => Item::Instant(i),
        };
        let index = crate::trace::summary::TraceIndex::new(&self.trace);
        if self.trace_view.focus.is_none() {
            let root = match item {
                Item::Span(i) => index.root_of(i),
                Item::Instant(_) => self.trace_view.list,
            };
            let Some(root) = root.or_else(|| self.trace_view.list_index(&self.trace_list()).map(|i| self.trace_list()[i].root)) else { return };
            self.trace_view.open_trace(root);
        }
        if let Item::Span(i) = item {
            for ancestor in index.ancestors(i) {
                self.trace_view.collapsed_spans.remove(&ancestor);
            }
        }
        self.trace_view.selected = Some(item);
        self.follow_selection_to_track();
        self.panel_index = 0;
    }

    /// The file tree's keys.
    fn handle_tree_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Char('j') | KeyCode::Down => self.move_selection(1),
            KeyCode::Char('k') | KeyCode::Up => self.move_selection(-1),
            KeyCode::Char('l') | KeyCode::Right => self.toggle_expand(),
            KeyCode::Enter => {
                if self.selected_expandable().is_some() {
                    self.toggle_expand();
                } else {
                    self.open_selected_in_editor();
                }
            }
            KeyCode::Char('h') | KeyCode::Left => self.collapse_current(),
            KeyCode::Char('G') => self.select_last(),
            KeyCode::Char('g') => self.select_first(),
            KeyCode::Char('/') => {
                self.search_mode = true;
                self.search_query.clear();
            }
            KeyCode::Char('s') => {
                self.sort_mode = match self.sort_mode {
                    SortMode::Alphabetical => SortMode::ByCoverage,
                    SortMode::ByCoverage => SortMode::Alphabetical,
                };
                self.rebuild_tree_rows();
            }
            KeyCode::Char('a') => self.cycle_agent_filter(),
            KeyCode::Char('A') => self.cycle_agent_filter_backward(),
            KeyCode::Char('C') => {
                if !self.compaction_history.is_empty() {
                    self.show_compaction_overlay = !self.show_compaction_overlay;
                }
            }
            KeyCode::Char('[') if self.show_compaction_overlay => {
                if self.compaction_overlay_index > 0 {
                    self.compaction_overlay_index -= 1;
                }
            }
            KeyCode::Char(']') if self.show_compaction_overlay => {
                if self.compaction_overlay_index + 1 < self.compaction_history.len() {
                    self.compaction_overlay_index += 1;
                }
            }
            KeyCode::Char('d') => self.open_alignment_overlay(),
            KeyCode::Char('t') => self.trace_view.open = true,
            KeyCode::Esc if self.show_alignment_overlay => {
                self.show_alignment_overlay = false;
            }
            KeyCode::PageDown => self.move_selection(20),
            KeyCode::PageUp => self.move_selection(-20),
            _ => {}
        }
    }

    pub fn handle_mouse(&mut self, mouse: MouseEvent) {
        // The content view covers the timeline: a click is not meant for it.
        if self.content_view.is_some() {
            return;
        }
        if self.trace_view.open {
            self.handle_trace_mouse(mouse);
            return;
        }
        match mouse.kind {
            MouseEventKind::ScrollUp => match self.focus {
                FocusPanel::Feed => {
                    self.activity_scroll_offset = self.activity_scroll_offset.saturating_add(3);
                }
                FocusPanel::Right => self.move_agent_selection(-1),
                FocusPanel::Left => self.move_selection(-3),
            },
            MouseEventKind::ScrollDown => match self.focus {
                FocusPanel::Feed => {
                    self.activity_scroll_offset = self.activity_scroll_offset.saturating_sub(3);
                }
                FocusPanel::Right => self.move_agent_selection(1),
                FocusPanel::Left => self.move_selection(3),
            },
            _ => {}
        }
    }

    /// What the trace panel shows: the trace chosen on the list, or the
    /// call or moment selected in the open trace (its prompt, the trace).
    pub fn trace_panel_subject(&self) -> PanelSubject {
        use crate::trace::view::Item;
        let tv = &self.trace_view;
        let Some(root) = tv.focus else {
            let traces = self.trace_list();
            return tv.list_index(&traces).map_or(PanelSubject::Nothing, |i| PanelSubject::Trace(traces[i].root));
        };
        match tv.selected {
            Some(Item::Span(i)) if i == root => PanelSubject::Trace(root),
            Some(Item::Span(i)) => PanelSubject::Call(i),
            Some(Item::Instant(i)) => PanelSubject::Instant(i),
            None => PanelSubject::Trace(root),
        }
    }

    /// The call whose content is wanted: the one open full-width, else the
    /// one the trace panel shows.
    fn content_span(&self) -> Option<usize> {
        if let Some(view) = &self.content_view {
            return Some(view.span);
        }
        // Only a trace opened in the trace view has calls to select; this
        // runs on every turn of the event loop, so it asks nothing more.
        if !self.trace_view.open || self.trace_view.focus.is_none() {
            return None;
        }
        match self.trace_panel_subject() {
            PanelSubject::Call(i) => Some(i),
            _ => None,
        }
    }

    /// What content a call has: a read's text, a write's change, anything
    /// else's output — nothing for a prompt.
    pub fn content_kind(&self, span: usize) -> Option<crate::ingest::content::ContentKind> {
        use crate::ingest::content::ContentKind;
        use crate::trace::SpanKind;
        match self.trace.spans().get(span)?.kind {
            SpanKind::Read(_) => Some(ContentKind::Read),
            SpanKind::Write => Some(ContentKind::Write),
            SpanKind::Delegate | SpanKind::Other => Some(ContentKind::Other),
            SpanKind::Prompt => None,
        }
    }

    fn content_key(&self, span: usize) -> Option<ContentKey> {
        let s = self.trace.spans().get(span)?;
        Some(ContentKey { id: s.id.clone()?, ended: s.end.is_some() })
    }

    /// The load to ask the content worker for, if the wanted call's content
    /// is neither loaded nor asked for yet; marks it asked.
    pub fn content_request(&mut self) -> Option<(ContentKey, crate::ingest::content::ContentKind)> {
        let span = self.content_span()?;
        let kind = self.content_kind(span)?;
        let key = self.content_key(span)?;
        let have = self.contents.loaded.as_ref().is_some_and(|(k, _)| *k == key);
        if have || self.contents.asked.as_ref() == Some(&key) {
            return None;
        }
        self.contents.asked = Some(key.clone());
        Some((key, kind))
    }

    /// The content worker's answer.
    pub fn set_call_content(&mut self, key: ContentKey, content: Option<crate::ingest::content::CallDetail>) {
        self.contents.loaded = Some((key, content.map(Arc::new)));
    }

    /// Where `span`'s content stands. A call still running shows what was
    /// loaded while it ran until the load after it ended arrives.
    pub fn content_state(&self, span: usize) -> ContentState<'_> {
        let Some(key) = self.content_key(span).filter(|_| self.content_kind(span).is_some()) else { return ContentState::Missing };
        match &self.contents.loaded {
            Some((k, content)) if k.id == key.id && (*k == key || content.is_some()) => {
                content.as_deref().map_or(ContentState::Missing, ContentState::Loaded)
            }
            _ => ContentState::Loading,
        }
    }

    /// `o`: the selected call's content, full-width.
    fn open_content_view(&mut self) {
        if let PanelSubject::Call(span) = self.trace_panel_subject() {
            if self.content_kind(span).is_some() {
                self.content_view = Some(ContentView { span, scroll: 0, height: std::cell::Cell::new(0), width: std::cell::Cell::new(80) });
            }
        }
    }

    /// The content view's keys: scroll, step between hunks, close.
    fn handle_content_key(&mut self, key: KeyEvent) {
        let Some(view) = &self.content_view else { return };
        let (rows, hunks) = match self.content_state(view.span) {
            ContentState::Loaded(c) => {
                let rows = c.rows(view.width.get());
                (rows.len(), crate::ingest::content::hunk_starts(&rows))
            }
            _ => (0, Vec::new()),
        };
        let page = view.height.get().max(1);
        let last = rows.saturating_sub(page);
        let at = view.scroll.min(last);
        let to = match key.code {
            KeyCode::Char('q') => return self.should_quit = true,
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => return self.should_quit = true,
            KeyCode::Esc | KeyCode::Char('o') => return self.content_view = None,
            KeyCode::Char('j') | KeyCode::Down => at + 1,
            KeyCode::Char('k') | KeyCode::Up => at.saturating_sub(1),
            KeyCode::PageDown | KeyCode::Char(' ') => at + page,
            KeyCode::PageUp => at.saturating_sub(page),
            KeyCode::Char('g') => 0,
            KeyCode::Char('G') => last,
            KeyCode::Char('n') => hunks.iter().copied().find(|&h| h > at).unwrap_or(at),
            KeyCode::Char('N') => hunks.iter().copied().rev().find(|&h| h < at).unwrap_or(0),
            _ => at,
        };
        if let Some(view) = &mut self.content_view {
            view.scroll = to.min(last);
        }
    }

    /// The trace panel's rows, as `Enter` sees them.
    pub fn trace_panel_targets(&self) -> Vec<crate::trace::summary::Target> {
        self.trace_panel_targets_in(&crate::trace::summary::TraceIndex::new(&self.trace))
    }

    /// [`Self::trace_panel_targets`] over an index already worked out.
    pub fn trace_panel_targets_in(&self, index: &crate::trace::summary::TraceIndex) -> Vec<crate::trace::summary::Target> {
        use crate::trace::summary;
        match self.trace_panel_subject() {
            PanelSubject::Trace(root) => summary::detail(&self.trace, index, root).map(|d| d.rows().iter().map(summary::Row::target).collect()).unwrap_or_default(),
            PanelSubject::Call(i) => summary::call_rows(&self.trace, index, i, self.write_of(i)).iter().map(summary::Row::target).collect(),
            PanelSubject::Instant(_) | PanelSubject::Nothing => Vec::new(),
        }
    }

    /// The traces, one per prompt, under the current agent filter.
    pub fn trace_list(&self) -> Vec<crate::trace::view::TraceSummary> {
        crate::trace::view::traces(&self.trace, self.agent_filter.as_deref())
    }

    /// The trace view's rows under the current agent filter.
    pub fn trace_rows(&self) -> Vec<crate::trace::view::Row> {
        let tv = &self.trace_view;
        crate::trace::view::waterfall(&self.trace, self.agent_filter.as_deref(), tv.focus, &tv.collapsed_spans, &tv.query)
    }

    /// The tracks and their screen rows under the current agent filter.
    pub fn trace_tracks(&self) -> (Vec<crate::trace::view::Track>, Vec<crate::trace::view::TrackRow>) {
        let tracks = crate::trace::view::tracks(&self.trace, self.agent_filter.as_deref(), self.trace_view.focus, self.trace_open_end());
        let rows = crate::trace::view::track_rows(&tracks, &self.trace_view.collapsed_agents);
        (tracks, rows)
    }

    /// Where a call still running is drawn to: the last moment seen.
    pub fn trace_open_end(&self) -> u64 {
        self.trace.range().map_or(0, |(_, end)| end)
    }

    /// The trace view's keys: modal, as in Perfetto (`w`/`a`/`s`/`d` zoom
    /// and pan only here).
    fn handle_trace_key(&mut self, key: KeyEvent) {
        use crate::trace::view::Layout;
        let tv = &mut self.trace_view;
        if tv.typing {
            match key.code {
                KeyCode::Esc => {
                    tv.typing = false;
                    tv.query.clear();
                }
                KeyCode::Enter => tv.typing = false,
                KeyCode::Backspace => {
                    tv.query.pop();
                }
                KeyCode::Char(c) => tv.query.push(c),
                _ => {}
            }
            return;
        }
        let waterfall = tv.layout == Layout::Waterfall;
        if tv.focus.is_none() {
            self.handle_trace_list_key(key);
            return;
        }
        match key.code {
            KeyCode::Char('t') => self.trace_view.open = false,
            KeyCode::Esc | KeyCode::Backspace => self.trace_view.close_trace(),
            KeyCode::Char('v') => {
                self.trace_view.layout = if waterfall { Layout::Tracks } else { Layout::Waterfall };
            }
            KeyCode::Char('w') => self.trace_view.zoom(&self.trace, 0.5, None),
            KeyCode::Char('s') => self.trace_view.zoom(&self.trace, 2.0, None),
            KeyCode::Char('a') => self.trace_view.pan(&self.trace, -0.25),
            KeyCode::Char('d') => self.trace_view.pan(&self.trace, 0.25),
            KeyCode::Char('0') => self.trace_view.fit(),
            KeyCode::Char('j') | KeyCode::Down => self.move_trace_row(1),
            KeyCode::Char('k') | KeyCode::Up => self.move_trace_row(-1),
            KeyCode::PageDown => self.move_trace_row(20),
            KeyCode::PageUp => self.move_trace_row(-20),
            KeyCode::Char('g') => self.move_trace_row(isize::MIN / 2),
            KeyCode::Char('G') => self.move_trace_row(isize::MAX / 2),
            KeyCode::Char('h') | KeyCode::Left if waterfall => self.trace_view.toggle_span(Some(false)),
            KeyCode::Char('l') | KeyCode::Right if waterfall => self.trace_view.toggle_span(Some(true)),
            KeyCode::Char('h') | KeyCode::Left => {
                let (tracks, rows) = self.trace_tracks();
                self.trace_view.step_span(&tracks, &rows, -1);
            }
            KeyCode::Char('l') | KeyCode::Right => {
                let (tracks, rows) = self.trace_tracks();
                self.trace_view.step_span(&tracks, &rows, 1);
            }
            KeyCode::Char(' ') if waterfall => self.trace_view.toggle_span(None),
            KeyCode::Char(' ') => {
                let (tracks, rows) = self.trace_tracks();
                if let Some(row) = rows.get(self.trace_view.track_row) {
                    let agent = tracks[row.track].agent.clone();
                    if !self.trace_view.collapsed_agents.remove(&agent) {
                        self.trace_view.collapsed_agents.insert(agent);
                    }
                }
            }
            KeyCode::Char('/') if waterfall => {
                self.trace_view.typing = true;
                self.trace_view.query.clear();
            }
            KeyCode::Char('e') => {
                let rows = self.trace_rows();
                self.trace_view.next_error(&self.trace, &rows);
                self.follow_selection_to_track();
            }
            KeyCode::Enter => self.follow_trace_selection(),
            _ => {}
        }
    }

    /// The list of traces: choose one and `Enter` shows its timeline.
    fn handle_trace_list_key(&mut self, key: KeyEvent) {
        use crate::trace::view::Layout;
        let traces = self.trace_list();
        let tv = &mut self.trace_view;
        match key.code {
            KeyCode::Char('t') | KeyCode::Esc => tv.open = false,
            KeyCode::Char('j') | KeyCode::Down => tv.move_list(&traces, 1),
            KeyCode::Char('k') | KeyCode::Up => tv.move_list(&traces, -1),
            KeyCode::PageDown => tv.move_list(&traces, 20),
            KeyCode::PageUp => tv.move_list(&traces, -20),
            KeyCode::Char('g') => tv.move_list(&traces, isize::MIN / 2),
            KeyCode::Char('G') => tv.move_list(&traces, isize::MAX / 2),
            KeyCode::Enter => {
                if let Some(ix) = tv.list_index(&traces) {
                    tv.open_trace(traces[ix].root);
                }
            }
            KeyCode::Char('v') => tv.layout = if tv.layout == Layout::Waterfall { Layout::Tracks } else { Layout::Waterfall },
            _ => {}
        }
    }

    fn move_trace_row(&mut self, delta: isize) {
        if self.trace_view.layout == crate::trace::view::Layout::Waterfall {
            let rows = self.trace_rows();
            self.trace_view.move_row(&rows, delta);
        } else {
            let (tracks, rows) = self.trace_tracks();
            self.trace_view.move_track_row(&self.trace, &tracks, &rows, delta);
        }
    }

    /// In the tracks layout, put the lane cursor on the selected span's row.
    fn follow_selection_to_track(&mut self) {
        let Some(crate::trace::view::Item::Span(i)) = self.trace_view.selected else { return };
        let (tracks, rows) = self.trace_tracks();
        if let Some(r) = rows.iter().position(|r| r.spans(&tracks).contains(&i)) {
            self.trace_view.track_row = r;
        }
    }

    /// `Enter` on a span: into a delegation's subagent, or out to the
    /// symbol or file it was about, in the tree.
    fn follow_trace_selection(&mut self) {
        use crate::trace::view::{Item, Layout};
        let Some(Item::Span(i)) = self.trace_view.selected else { return };
        let span = self.trace.spans()[i].clone();
        if let Some(child) = &span.child_agent {
            if self.trace_view.layout == Layout::Waterfall {
                self.trace_view.collapsed_spans.remove(&i);
                let rows = self.trace_rows();
                if let Some(pos) = rows.iter().position(|r| r.item == Item::Span(i)) {
                    if let Some(first) = rows.get(pos + 1).filter(|r| r.depth > rows[pos].depth) {
                        self.trace_view.selected = Some(first.item);
                    }
                }
            } else {
                let (tracks, rows) = self.trace_tracks();
                self.trace_view.collapsed_agents.remove(&**child);
                self.trace_view.select_track(&tracks, &rows, child);
            }
            return;
        }
        // To the first symbol it was credited with reading, exactly; else to
        // the symbol it named in its file, by name.
        let revealed = match span.read.first() {
            Some((id, _)) => self.reveal_id(crate::symbols::split_id(id).0, Some(id.clone())),
            None => span.file.as_deref().is_some_and(|file| self.reveal(file, span.symbol.as_deref())),
        };
        if revealed {
            self.trace_view.open = false;
            self.focus = FocusPanel::Left;
        }
    }

    /// Select `file` (project-relative) in the tree, or the symbol in it
    /// named by `symbol`, expanding what hides it. `false` when the file is
    /// not in the tree.
    pub fn reveal(&mut self, file: &str, symbol: Option<&str>) -> bool {
        let Some(tree_file) = self.project_tree.file(file) else { return false };
        let target = symbol.map(normalize_name_path).and_then(|name| {
            let nodes = tree_file.walk();
            let exact = nodes.iter().find(|n| n.name_path() == name);
            let by_tail = || nodes.iter().find(|n| n.name_path().ends_with(&format!("/{name}")) || *n.name == *name);
            exact.or_else(by_tail).map(|n| n.id.clone())
        });
        self.reveal_id(file, target)
    }

    /// Select symbol `id` in the tree, expanding its file and every symbol
    /// it sits in; or, `None`, the file. `false` when the file is not in the
    /// tree. A symbol no longer there leaves the file selected.
    pub fn reveal_id(&mut self, file: &str, target: Option<String>) -> bool {
        let Some(tree_file) = self.project_tree.file(file) else { return false };
        let file_id = tree_file.file_path.to_string_lossy().into_owned();
        self.set_expanded(&file_id, RowKind::File, true);
        if let Some(id) = &target {
            let (path, name) = crate::symbols::split_id(id);
            let segments: Vec<&str> = name.split('/').collect();
            for n in 1..segments.len() {
                self.set_expanded(&format!("{path}::{}", segments[..n].join("/")), RowKind::Symbol, true);
            }
        }
        let want = target.unwrap_or_else(|| file_id.clone());
        let at = self.tree_rows.iter().position(|r| r.symbol_id == want);
        let at = at.or_else(|| self.tree_rows.iter().position(|r| r.symbol_id == file_id));
        if let Some(ix) = at {
            self.selected_index = ix;
        }
        true
    }

    /// `file` (project-relative) as write statuses compare against it, or
    /// `None` when it is not in the tree. Made once and kept until the file
    /// changes — a frame asks for every written file, and making one walks
    /// all its symbols — which its top-level symbols' merkle hashes (each
    /// covering everything beneath it) tell, without a signal from each
    /// place the tree is updated.
    pub fn file_contents(&self, file: &str) -> Option<Arc<crate::writes::FileContents>> {
        let symbols = self.project_tree.file(file)?;
        let mut fingerprint = blake3::Hasher::new();
        for sym in &symbols.symbols {
            fingerprint.update(&sym.merkle_hash);
        }
        let fingerprint = fingerprint.finalize();
        let mut cache = self.file_contents.borrow_mut();
        match cache.get(file) {
            Some((at, contents)) if *at == fingerprint => Some(contents.clone()),
            _ => {
                let contents = Arc::new(crate::writes::FileContents::from_symbols(symbols));
                cache.insert(file.to_string(), (fingerprint, contents.clone()));
                Some(contents)
            }
        }
    }

    /// Every write of the session by op, and whether its version is still
    /// in the tree: read once per written file, for a frame to colour by.
    pub fn write_statuses(&self) -> std::collections::HashMap<&str, (&crate::writes::WriteRecord, crate::writes::Status)> {
        let mut out = std::collections::HashMap::new();
        for (file, writes) in self.writes.by_file(None) {
            let now = self.file_contents(file);
            for w in writes {
                let status = match &now {
                    Some(now) => now.file_status(w),
                    // Out of the tree: gone, unless a path filter just hides it.
                    None if self.filter.is_some() => crate::writes::Status::Unknown,
                    None => crate::writes::Status::Removed,
                };
                out.insert(w.op.as_str(), (w, status));
            }
        }
        out
    }

    /// Wheel zooms at the pointer, a click selects, a drag pans.
    fn handle_trace_mouse(&mut self, mouse: MouseEvent) {
        use crate::trace::view::{Item, Layout};
        let Some(g) = self.trace_geometry.get() else { return };
        if self.trace_view.focus.is_none() {
            let traces = self.trace_list();
            match mouse.kind {
                MouseEventKind::ScrollUp => self.trace_view.move_list(&traces, -3),
                MouseEventKind::ScrollDown => self.trace_view.move_list(&traces, 3),
                MouseEventKind::Down(_) if (g.rows_y..g.rows_y + g.rows).contains(&mouse.row) => {
                    if let Some(t) = traces.get(g.first_row + (mouse.row - g.rows_y) as usize) {
                        self.trace_view.list = Some(t.root);
                    }
                }
                _ => {}
            }
            return;
        }
        let in_bars = mouse.column >= g.bars_x && mouse.column < g.bars_x + g.bars_width;
        let col = mouse.column.saturating_sub(g.bars_x) as usize;
        let vp = self.trace_view.viewport(&self.trace);
        let at = in_bars.then(|| vp.time_at(col, g.bars_width as usize));
        match mouse.kind {
            MouseEventKind::ScrollUp => self.trace_view.zoom(&self.trace, 0.8, at),
            MouseEventKind::ScrollDown => self.trace_view.zoom(&self.trace, 1.25, at),
            MouseEventKind::Down(_) => {
                self.trace_drag = in_bars.then_some(mouse.column);
                if mouse.row < g.rows_y || mouse.row >= g.rows_y + g.rows {
                    return;
                }
                let row = g.first_row + (mouse.row - g.rows_y) as usize;
                if self.trace_view.layout == Layout::Waterfall {
                    if let Some(r) = self.trace_rows().get(row) {
                        self.trace_view.selected = Some(r.item);
                    }
                } else {
                    let (tracks, rows) = self.trace_tracks();
                    if let Some(r) = rows.get(row) {
                        self.trace_view.track_row = row;
                        let open_end = self.trace_open_end();
                        let hit = crate::trace::view::span_at(&self.trace, &r.spans(&tracks), &vp, g.bars_width as usize, col, open_end);
                        if let Some(i) = hit {
                            self.trace_view.selected = Some(Item::Span(i));
                        }
                    }
                }
            }
            MouseEventKind::Drag(_) => {
                if let Some(from) = self.trace_drag {
                    let moved = f64::from(from) - f64::from(mouse.column);
                    self.trace_view.pan(&self.trace, moved / f64::from(g.bars_width.max(1)));
                    self.trace_drag = Some(mouse.column);
                }
            }
            MouseEventKind::Up(_) => self.trace_drag = None,
            _ => {}
        }
    }

    fn handle_search_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc => {
                self.search_mode = false;
                self.search_query.clear();
            }
            KeyCode::Enter => {
                self.search_mode = false;
                self.jump_to_search_match();
            }
            KeyCode::Backspace => {
                self.search_query.pop();
            }
            KeyCode::Char(c) => {
                self.search_query.push(c);
            }
            _ => {}
        }
    }

    fn move_selection(&mut self, delta: i32) {
        if self.tree_rows.is_empty() {
            return;
        }
        let new_idx = self.selected_index as i32 + delta;
        self.selected_index = new_idx.clamp(0, self.tree_rows.len() as i32 - 1) as usize;
        self.panel_index = 0;
    }



    fn select_first(&mut self) {
        self.selected_index = 0;
    }

    fn select_last(&mut self) {
        if !self.tree_rows.is_empty() {
            self.selected_index = self.tree_rows.len() - 1;
        }
    }

    /// The only way to open or close a row. Rebuilds only on an actual change.
    pub fn set_expanded(&mut self, id: &str, kind: RowKind, expanded: bool) {
        if self.expansion.set(id, kind, expanded) {
            self.rebuild_tree_rows();
        }
    }

    /// The highlighted row, if it can be expanded.
    fn selected_expandable(&self) -> Option<(String, RowKind)> {
        self.tree_rows
            .get(self.selected_index)
            .filter(|row| row.has_children)
            .map(|row| (row.symbol_id.clone(), row.kind))
    }

    fn toggle_expand(&mut self) {
        if let Some((id, kind)) = self.selected_expandable() {
            let expanded = self.expansion.is_expanded(&id, kind);
            self.set_expanded(&id, kind, !expanded);
        }
    }

    fn collapse_current(&mut self) {
        if let Some((id, kind)) = self.selected_expandable() {
            self.set_expanded(&id, kind, false);
        }
    }

    /// Resolve the selected row to a `(file, line)` and stash it in
    /// `pending_editor_request` for the TUI loop to act on. `App` owns no
    /// terminal handle, so it can only signal the request upward, not launch
    /// the editor itself.
    ///
    /// A no-op (defense in depth — `handle_key` already gates this) for a row
    /// with children, and a silent no-op if the row no longer resolves to a
    /// file/symbol (e.g. the tree changed since the row was rendered) rather
    /// than requesting a bogus path.
    fn open_selected_in_editor(&mut self) {
        let Some(row) = self.tree_rows.get(self.selected_index) else {
            return;
        };
        if row.has_children {
            return;
        }

        if row.is_file() {
            let Some(file) = self
                .project_tree
                .files
                .iter()
                .find(|f| f.file_path.to_string_lossy() == row.symbol_id)
            else {
                return;
            };
            self.pending_editor_request =
                Some((self.project_root.join(&file.file_path), 1));
            return;
        }

        let id = row.symbol_id.clone();
        self.open_in_editor(&id);
    }

    /// Ask for symbol `id`'s definition in the editor, at its first line —
    /// the first, when ids collide. A no-op for a symbol no longer in the
    /// tree.
    fn open_in_editor(&mut self, id: &str) {
        let (file, _) = crate::symbols::split_id(id);
        let Some(tree_file) = self.project_tree.file(file) else { return };
        if let Some(sym) = tree_file.walk().into_iter().find(|sym| sym.id == id) {
            self.pending_editor_request = Some((self.project_root.join(&tree_file.file_path), sym.line_range.start));
        }
    }

    /// Call `span`'s write record, when it is a write the journal has.
    pub fn write_of(&self, span: usize) -> Option<&crate::writes::WriteRecord> {
        self.trace.spans().get(span)?.id.as_deref().and_then(|op| self.writes.get(op))
    }

    /// The agent ids the stats panel lists, in the order it lists them.
    ///
    /// `agents_seen` is *not* that list: it only holds agents that emitted a
    /// tool call, while the panel renders [`Self::flattened_agents`], which
    /// also carries the root seeded from the session id (see
    /// [`Self::seed_agent_tree_root`]) — an orchestrator that only dispatches
    /// `Task` calls never reads a file of its own and so never lands in
    /// `agents_seen`. Every cursor, bound, and count that has to line up with
    /// what is on screen must come from here.
    fn agent_list(&self) -> Vec<String> {
        self.flattened_agents().into_iter().map(|(id, _)| id).collect()
    }

    fn cycle_agent_filter(&mut self) {
        let agents = self.agent_list();
        if agents.is_empty() {
            self.agent_filter = None;
            self.agent_selection_index = 0;
            return;
        }
        match &self.agent_filter {
            None => {
                self.agent_filter = Some(agents[0].clone());
                self.agent_selection_index = 1;
            }
            Some(current) => {
                let idx = agents.iter().position(|a| a == current);
                match idx {
                    Some(i) if i + 1 < agents.len() => {
                        self.agent_filter = Some(agents[i + 1].clone());
                        self.agent_selection_index = i + 2;
                    }
                    _ => {
                        self.agent_filter = None;
                        self.agent_selection_index = 0;
                    }
                }
            }
        }
        self.rebuild_tree_rows();
    }

    fn cycle_agent_filter_backward(&mut self) {
        let agents = self.agent_list();
        if agents.is_empty() {
            self.agent_filter = None;
            self.agent_selection_index = 0;
            return;
        }
        match &self.agent_filter {
            None => {
                let last = agents.len() - 1;
                self.agent_filter = Some(agents[last].clone());
                self.agent_selection_index = agents.len();
            }
            Some(current) => {
                let idx = agents.iter().position(|a| a == current);
                match idx {
                    Some(0) => {
                        self.agent_filter = None;
                        self.agent_selection_index = 0;
                    }
                    Some(i) => {
                        self.agent_filter = Some(agents[i - 1].clone());
                        self.agent_selection_index = i;
                    }
                    _ => {
                        self.agent_filter = None;
                        self.agent_selection_index = 0;
                    }
                }
            }
        }
        self.rebuild_tree_rows();
    }

    /// Open the sub-agent alignment popup for the currently selected agent's
    /// comparison group: its parent plus all of that parent's children (or,
    /// when the selected agent is itself the root/orchestrator, the root
    /// plus all of its direct children).
    ///
    /// The parent is included deliberately — an orchestrator that spawned a
    /// single sub-agent still has something to compare (root vs. that one
    /// child), and an orchestrator with several sub-agents is itself a
    /// meaningful comparison point against each of them, not just an
    /// excluded coordinator.
    ///
    /// No-op when no agent is selected (`agent_filter` is `None`, meaning
    /// "All"), or when the resulting group has fewer than 2 members —
    /// there is nothing to compare.
    fn open_alignment_overlay(&mut self) {
        let Some(agent_id) = self.agent_filter.clone() else {
            return;
        };
        let Some(node) = self.agent_tree.agents.get(&agent_id) else {
            return;
        };

        // The parent whose children form the sibling half of the group:
        // the selected agent's own parent, or (when the selected agent has
        // no parent, i.e. it *is* the root) the selected agent itself.
        let parent_id = node.parent_id.clone().unwrap_or_else(|| agent_id.clone());

        let mut group_ids: Vec<String> = self
            .agent_tree
            .children_of(&parent_id)
            .into_iter()
            .map(|a| a.id.clone())
            .collect();
        group_ids.push(parent_id);
        group_ids.sort();
        group_ids.dedup();

        if group_ids.len() < 2 {
            return;
        }

        self.agent_alignment = crate::tracking::alignment::compute_group_alignment(
            &self.project_tree,
            &self.depth_cache,
            &group_ids,
        );
        self.show_alignment_overlay = true;
    }

    fn move_agent_selection(&mut self, delta: i32) {
        // Bound the cursor by the rendered list, not by `agents_seen` — see
        // `agent_list`. `apply_agent_selection` resolves the index against
        // the rendered list, so bounding it against a shorter one silently
        // maps rows onto the wrong agents and hides the tail of the list.
        let total = self.agent_list().len() + 1; // +1 for "All"
        if total == 0 {
            return;
        }
        let new_idx = if delta > 0 {
            (self.agent_selection_index + delta as usize) % total
        } else {
            let back = (-delta) as usize;
            (self.agent_selection_index + total - (back % total)) % total
        };
        self.agent_selection_index = new_idx;
    }

    fn apply_agent_selection(&mut self) {
        if self.agent_selection_index == 0 {
            self.agent_filter = None;
        } else {
            let flat = self.flattened_agents();
            if let Some((agent_id, _)) = flat.get(self.agent_selection_index - 1) {
                self.agent_filter = Some(agent_id.clone());
            } else {
                self.agent_filter = None;
            }
        }
        self.rebuild_tree_rows();
    }

    /// Returns agent IDs in hierarchy order (DFS) with indent levels.
    /// Each entry is `(agent_id, indent_level)`.
    /// Agents not reachable from the root are appended at depth 0.
    pub fn flattened_agents(&self) -> Vec<(String, usize)> {
        let mut result = Vec::new();
        if let Some(ref root_id) = self.agent_tree.root_id {
            self.flatten_dfs(root_id, 0, &mut result);
        }
        // Append any agents not reached by DFS (orphans).
        for agent_id in &self.agents_seen {
            if !result.iter().any(|(id, _)| id == agent_id) {
                self.flatten_dfs(agent_id, 0, &mut result);
            }
        }
        result
    }

    fn flatten_dfs(&self, agent_id: &str, depth: usize, out: &mut Vec<(String, usize)>) {
        out.push((agent_id.to_string(), depth));
        let children = self.agent_tree.children_of(agent_id);
        for child in children {
            self.flatten_dfs(&child.id, depth + 1, out);
        }
    }

    /// Tree, the right-hand pane, then the activity feed when it shows.
    /// Move focus forward (`Tab`) or back (`Shift+Tab`) through the left
    /// panel, the right panel and — when shown — the feed.
    fn cycle_focus(&mut self, forward: bool) {
        let order: &[FocusPanel] =
            if self.show_activity { &[FocusPanel::Left, FocusPanel::Right, FocusPanel::Feed] } else { &[FocusPanel::Left, FocusPanel::Right] };
        let at = order.iter().position(|f| *f == self.focus).unwrap_or(0);
        let next = if forward { (at + 1) % order.len() } else { (at + order.len() - 1) % order.len() };
        self.focus = order[next];
    }

    /// The traces that touched the selected row, for the inspector.
    pub fn inspector_touches(&self) -> Vec<crate::trace::touch::Touch> {
        let Some(row) = self.tree_rows.get(self.selected_index) else { return Vec::new() };
        let (file, symbol) = match row.kind {
            RowKind::File => (row.symbol_id.as_str(), None),
            RowKind::Symbol => (crate::symbols::split_id(&row.symbol_id).0, Some(row.symbol_id.as_str())),
        };
        crate::trace::touch::touches(&self.trace, &crate::objects::normalize_path(file), symbol, &self.writes)
    }

    /// Open the trace the inspector has selected, at its first call on the row.
    fn open_inspected_trace(&mut self) {
        let touches = self.inspector_touches();
        let Some(t) = touches.get(self.panel_index.min(touches.len().saturating_sub(1))) else { return };
        self.trace_view.open = true;
        self.trace_view.open_trace(t.root);
        self.trace_view.selected = Some(crate::trace::view::Item::Span(t.span));
        self.focus = FocusPanel::Left;
    }

    fn jump_to_search_match(&mut self) {
        let query = self.search_query.to_lowercase();
        if query.is_empty() {
            return;
        }
        // Search forward from current position.
        let start = (self.selected_index + 1) % self.tree_rows.len();
        for i in 0..self.tree_rows.len() {
            let idx = (start + i) % self.tree_rows.len();
            if self.tree_rows[idx]
                .display_name
                .to_lowercase()
                .contains(&query)
            {
                self.selected_index = idx;
                return;
            }
        }
    }

    /// Record `agent_id` as an agent of this session, placing it in the agent
    /// hierarchy. Idempotent.
    ///
    /// Every source of ledger attribution has to come through here, because
    /// the stats panel's per-agent rows are built from the hierarchy: an
    /// agent that holds coverage but was never registered makes "[All]" count
    /// reads that no row can account for.
    ///
    /// NOTE: sub-agent JSONL *filenames* are prefixed `agent-<hash>`, but the
    /// `agentId` field *inside* each line — which `parse_jsonl_line`
    /// (src/ingest/claude.rs) prefers over the filename/session-derived
    /// fallback — carries no such prefix (e.g. `"a63c858997b4e6124"`, not
    /// `"agent-a63c858997b4e6124"`). A `starts_with("agent-")` check
    /// therefore never matches real sub-agent events; only hand-constructed
    /// test fixtures that bake the prefix into `agent_id` happened to pass.
    /// Don't repeat that mistake in future tests — use realistic unprefixed
    /// ids.
    ///
    /// Now that `self.session_id` is deterministically known before any
    /// events are processed (see `set_session_id` / `seed_agent_tree_root`),
    /// identity is the correct and only check we need: the root's own events
    /// carry `agent_id == session_id` (and must NOT be re-parented to
    /// themselves); every other `agent_id` is a child of the root. Fall back
    /// to the old prefix heuristic only when no session_id is known (e.g.
    /// test paths that skip `set_session_id`), preserving prior behavior
    /// there.
    fn register_agent(&mut self, agent_id: &str, label: &str) {
        if self.agents_seen.iter().any(|a| a == agent_id) {
            return;
        }
        self.agents_seen.push(agent_id.to_string());

        let parent_id = match self.session_id.as_deref() {
            Some(root) if agent_id == root => None,
            Some(root) => Some(root.to_string()),
            None if agent_id.starts_with("agent-") => self.agent_tree.root_id.clone(),
            None => None,
        };
        self.agent_tree.add_agent(AgentNode {
            id: agent_id.to_string(),
            parent_id,
            session_file: PathBuf::new(),
            label: label.to_string(),
        });
    }

    /// A tool call's result arrived: close its span in the trace.
    pub fn process_tool_finished(&mut self, finished: &crate::ingest::ToolFinished) {
        self.trace.finish(finished);
        if !finished.shown.is_empty() {
            let read = apply_shown(&self.project_tree, finished, &mut self.ledger, &mut self.depth_cache);
            self.trace.note_read(&finished.id, read);
            self.rebuild_tree_rows();
        }
    }

    /// A prompt starts a turn: the trace nests the calls that follow under it.
    pub fn process_prompt(&mut self, prompt: &crate::ingest::Prompt) {
        self.trace.prompt(prompt);
    }

    /// Queue a write for attribution, tagged with the current session (see
    /// `pending_writes`). Without a session there is no journal to put it in.
    pub fn queue_write(&mut self, event: crate::ingest::WriteEvent) {
        if let Some(session) = &self.session_id {
            self.pending_writes.push((Arc::from(session.as_str()), event));
        }
    }

    /// Take every queued write, for handing to the attribution worker.
    pub fn take_pending_writes(&mut self) -> Vec<(Arc<str>, crate::ingest::WriteEvent)> {
        std::mem::take(&mut self.pending_writes)
    }

    /// Journal an attributed write into the session it happened in — the
    /// current one, or the one just switched away from — and, for the
    /// current one, mark it on the tree. A write grants no read credit (D9),
    /// so the ledger is untouched; the activity feed already showed the call.
    ///
    /// Never opens a file: a write for any older session is dropped, which
    /// takes two switches while one write is in the worker.
    pub fn record_write(&mut self, session: &str, record: crate::writes::WriteRecord) {
        let journal = [self.journal.as_mut(), self.retired_journal.as_mut()]
            .into_iter()
            .flatten()
            .find(|j| j.session_id() == session);
        match journal {
            Some(journal) => {
                journal.record_write(&record);
            }
            None if self.journal_settings.is_some() => log::warn!(
                target: "ambits::journal",
                session, op = record.op.as_str();
                "write arrived after its session's journal was closed; not journaled"
            ),
            None => {}
        }
        if self.session_id.as_deref() == Some(session) {
            self.writes.insert(record);
            self.rebuild_tree_rows();
        }
    }

    /// `main` for the session's own agent, else the agent id.
    pub fn agent_name<'a>(&self, id: &'a str) -> &'a str {
        if self.session_id.as_deref() == Some(id) { "main" } else { id }
    }

    /// An agent as a reader knows it: `main`, or a subagent by the task
    /// that started it (`Expert review of phase 6`), else by its id.
    pub fn agent_title(&self, id: &str) -> String {
        match self.trace.delegation_of(id).map(|d| self.trace.spans()[d].task()) {
            Some(task) if !task.is_empty() => task,
            _ => self.agent_name(id).to_string(),
        }
    }

    /// Process an agent tool call event and update the ledger.
    pub fn process_agent_event(&mut self, event: AgentToolCall) {
        self.trace.start(&event, &self.project_root);
        self.compaction_call_count += 1;
        self.register_agent(&event.agent_id, &event.label);

        let read = apply_tool_call(
            &self.project_tree,
            &self.project_root,
            &event,
            &mut self.ledger,
            &mut self.depth_cache,
        );
        if let Some(id) = &event.tool_use_id {
            self.trace.note_read(id, read);
        }

        // Only push tracked events to the activity feed: reads, and writes —
        // which carry no read depth (D9) but are exactly what the feed should
        // show.
        if event.read_depth != ReadDepth::Unseen || event.effect == crate::ingest::Effect::Write {
            self.activity.push(event);
            self.activity_scroll_offset = 0; // Auto-scroll to latest
            if self.activity.len() > 200 {
                self.activity.drain(0..100);
            }
        }
        self.rebuild_tree_rows();
    }
}

/// Resolve the `path`/`target` display strings for one activity-log line.
///
/// `file_path`/`target_symbol`/`target_lines` cover most tools, but a
/// selector-driven call (`ambits show <id>`) populates none of those — only
/// `target_selectors` — so without this both columns printed `-` even though
/// real data existed. `target` falls back to the joined selector tokens;
/// `path` falls back to [`resolve_selector_path`], which looks the first
/// selector up in the tree the same way [`mark_selectors`] already
/// does. A tool call with none of the above (e.g. an untracked tool) still
/// prints `-` for both — there is nothing to hydrate from.
fn event_log_path_and_target(tree: &ProjectTree, event: &AgentToolCall) -> (String, String) {
    let target = match (&event.target_symbol, &event.target_lines) {
        (Some(sym), _) => sym.clone(),
        (None, Some(lines)) => format!("L{}-{}", lines.start, lines.end),
        (None, None) if !event.target_selectors.is_empty() => event
            .target_selectors
            .iter()
            .map(|(sel, _)| sel.as_str())
            .collect::<Vec<_>>()
            .join(", "),
        (None, None) => "-".to_string(),
    };

    let path_str = match &event.file_path {
        Some(p) => p.display().to_string(),
        None => resolve_selector_path(tree, event).unwrap_or_else(|| "-".to_string()),
    };

    (path_str, target)
}

/// The file path of the first `target_selectors` entry that resolves to a
/// symbol in `tree`, matched the same way [`mark_selectors`] matches
/// by id or content-hash prefix. `None` if there are no selectors, or none of
/// them resolve.
///
/// The activity log has one `path` column but a selector-driven command can
/// touch several files (`ambits show a.rs::X b.rs::Y`); rather than pick a
/// column-per-file shape for one log line, the first resolved path is shown
/// with `" (+N more)"` appended when others resolve to different files.
fn resolve_selector_path(tree: &ProjectTree, event: &AgentToolCall) -> Option<String> {
    if event.target_selectors.is_empty() {
        return None;
    }

    let symbols = tree.walk();
    let mut paths: Vec<&Path> = Vec::new();
    for (sel, _) in &event.target_selectors {
        let matched = match crate::lookup::parse_selector(sel) {
            crate::lookup::Selector::Id(_) => {
                symbols.iter().find(|(_, sym)| sym.id == *sel).map(|(p, _)| *p)
            }
            crate::lookup::Selector::Hash(h) => symbols
                .iter()
                .find(|(_, sym)| crate::journal::hash_hex(&sym.content_hash).starts_with(h.as_str()))
                .map(|(p, _)| *p),
            crate::lookup::Selector::Unrecognized(_) => None,
        };
        if let Some(p) = matched {
            if !paths.contains(&p) {
                paths.push(p);
            }
        }
    }

    let first = paths.first()?.display().to_string();
    Some(match paths.len() {
        1 => first,
        n => format!("{first} (+{} more)", n - 1),
    })
}

/// Where a call's reads go as they are credited: the ledger and the depth
/// cache, as `agent`'s — and a note of each, so the trace can say which
/// symbols the call read.
pub struct Credit<'a> {
    ledger: &'a mut ContextLedger,
    depth_cache: &'a mut crate::tracking::alignment::DepthOrdinalCache,
    agent: &'a str,
    /// Each credit, and whether it came with a parent read whole — implied,
    /// so not worth listing.
    read: Vec<(String, ReadDepth, bool)>,
}

impl<'a> Credit<'a> {
    pub fn new(ledger: &'a mut ContextLedger, depth_cache: &'a mut crate::tracking::alignment::DepthOrdinalCache, agent: &'a str) -> Self {
        Credit { ledger, depth_cache, agent, read: Vec::new() }
    }

    /// Credit `sym` at `depth`; `implied` when it comes with a parent read
    /// whole.
    fn record(&mut self, sym: &SymbolNode, depth: ReadDepth, implied: bool) {
        self.ledger.record(sym.id.clone(), depth, sym.content_hash, self.agent.to_string(), sym.estimated_tokens as usize);
        self.depth_cache.record(&sym.id, self.agent, depth);
        self.read.push((sym.id.clone(), depth, implied));
    }

    /// The symbols the call read, as the trace lists them: each credited in
    /// its own right — not one that came with a parent read whole (a method
    /// of an impl read in full) — once, at the deepest it was credited, in
    /// the order first credited. A search that matched an impl's header and
    /// a line in one of its methods read both, and lists both.
    pub fn listed(self) -> Vec<(String, ReadDepth)> {
        let mut out: Vec<(String, ReadDepth)> = Vec::new();
        for (id, depth, _) in self.read.iter().filter(|(_, _, implied)| !implied) {
            match out.iter_mut().find(|(seen, _)| seen == id) {
                Some((_, d)) => *d = (*d).max(*depth),
                None => out.push((id.clone(), *depth)),
            }
        }
        out
    }
}

/// Apply one tool call to the ledger, and say which symbols it read (as
/// [`Credit::listed`] lists them).
///
/// The single place that decides how a tool call becomes symbol reads. It had
/// been open-coded at three call sites — the TUI, `coverage::run_report`, and
/// `restore::replay_session_logs` — which is exactly how selector support came
/// to work in the TUI and silently do nothing in the other two.
pub fn apply_tool_call(
    tree: &ProjectTree,
    project_root: &Path,
    event: &AgentToolCall,
    ledger: &mut ContextLedger,
    depth_cache: &mut crate::tracking::alignment::DepthOrdinalCache,
) -> Vec<(String, ReadDepth)> {
    // The one structured line per tool call — the activity record. Replaces
    // both the old hand-rolled `<session>.log` writer and the separate
    // debug-level dispatch logs this function used to carry per branch below;
    // those were redundant with this once it names its own dispatch choice.
    let (path_str, symbol_target) = event_log_path_and_target(tree, event);
    log::info!(
        target: "ambits::activity",
        agent_id = event.agent_id.as_ref(),
        tool = event.tool_name.as_ref(),
        depth:? = event.read_depth,
        path = path_str,
        symbol_target = symbol_target,
        description = event.description;
        "tool call"
    );

    // Only a read that saw something changes read state: a write is not a
    // read (D9), and an Unseen read saw nothing.
    if event.effect == crate::ingest::Effect::Write || !event.read_depth.is_seen() {
        return Vec::new();
    }
    let mut credit = Credit::new(ledger, depth_cache, &event.agent_id);

    if !event.target_selectors.is_empty() {
        mark_selectors(tree, &event.target_selectors, &mut credit);
    }

    if let Some(ref file_path) = event.file_path {
        let tool_rel = normalize_tool_path(file_path, project_root);
        for file in tree.files.iter().filter(|f| f.file_path == tool_rel) {
            if event.target_symbol.is_some() || event.target_lines.is_some() {
                mark_targeted_symbols(&file.symbols, event, &mut credit);
            } else {
                mark_file_symbols(&file.symbols, event.read_depth, &mut credit, false);
            }
        }
    }
    credit.listed()
}

/// Credit the symbols a call's output showed (`ToolFinished::shown`: the
/// matches an `ambits rg` printed), as the agent that made the call, and
/// say which.
pub fn apply_shown(
    tree: &ProjectTree,
    finished: &crate::ingest::ToolFinished,
    ledger: &mut ContextLedger,
    depth_cache: &mut crate::tracking::alignment::DepthOrdinalCache,
) -> Vec<(String, ReadDepth)> {
    if finished.shown.is_empty() {
        return Vec::new();
    }
    let mut credit = Credit::new(ledger, depth_cache, &finished.agent_id);
    mark_selectors(tree, &finished.shown, &mut credit);
    credit.listed()
}

/// Mark every symbol named by `selectors` (ids or content hashes), as read
/// by `agent` at the depth each carries: `ambits show`'s arguments, or the
/// symbols a search printed.
///
/// Ambiguity is credited in full rather than resolved: an id like
/// `src/app.rs::App` names both `struct App` and `impl App`, and a lookup that
/// returned both did in fact show the caller both. That mirrors what
/// `ambits show` actually printed, which is the point — coverage should record
/// what was seen, not what we wish had been asked for.
///
/// A selector matching nothing is silently ignored. The command may have been
/// a miss, or may name a symbol that has since changed; either way there is
/// no read to record.
fn mark_selectors(tree: &ProjectTree, selectors: &[(String, ReadDepth)], credit: &mut Credit<'_>) {
    // An id names its file, so only the files named are walked, each once;
    // a hash can be anywhere, so only a hash selector walks the whole tree
    // (and only then are symbols' hashes spelled out to compare). A search
    // result names up to 200 ids, on the event loop.
    let mut by_file: HashMap<&str, HashMap<&str, ReadDepth>> = HashMap::new();
    let mut hashes: Vec<(String, ReadDepth)> = Vec::new();
    for (sel, depth) in selectors {
        match crate::lookup::parse_selector(sel) {
            crate::lookup::Selector::Hash(h) => hashes.push((h, *depth)),
            crate::lookup::Selector::Id(_) => {
                // A symbol named more than once takes the deepest naming;
                // `record` is upgrade-only anyway, but this keeps the credit
                // independent of order.
                let depths = by_file.entry(crate::symbols::split_id(sel).0).or_default();
                let at = depths.entry(sel.as_str()).or_insert(*depth);
                *at = (*at).max(*depth);
            }
            crate::lookup::Selector::Unrecognized(_) => {}
        }
    }
    // One pass over the files, not a lookup per file named: finding a file
    // by path normalises every path it passes.
    let named = tree.files.iter().filter_map(|file| {
        by_file.get(crate::objects::normalize_path(&file.file_path.to_string_lossy()).as_str()).map(|ids| (file, ids))
    });
    for (file, ids) in named {
        // Every symbol an id names: ids are not unique (`struct App` and
        // `impl App`), and a lookup that returned both showed both.
        for sym in file.walk() {
            if let Some(depth) = ids.get(sym.id.as_str()) {
                credit.record(sym, *depth, false);
            }
        }
    }
    if !hashes.is_empty() {
        for (_, sym) in tree.walk() {
            let hex = crate::journal::hash_hex(&sym.content_hash);
            if let Some(depth) = hashes.iter().filter(|(h, _)| hex.starts_with(h.as_str())).map(|(_, d)| *d).max() {
                credit.record(sym, depth, false);
            }
        }
    }
}

/// How many of `symbols`, at any depth, were read and have changed since.
fn count_stale(symbols: &[SymbolNode], ledger: &ContextLedger) -> usize {
    symbols
        .iter()
        .map(|s| usize::from(ledger.is_stale(&s.id) && ledger.depth_of(&s.id).is_seen()) + count_stale(&s.children, ledger))
        .sum()
}

/// What every symbol row of one file is built from.
struct RowContext<'a> {
    expansion: &'a Expansion,
    ledger: &'a ContextLedger,
    agent_filter: Option<&'a str>,
    writes: Option<&'a crate::write_index::FileWrites<'a>>,
}

fn flatten_symbol(sym: &SymbolNode, depth: usize, cx: &RowContext<'_>, rows: &mut Vec<TreeRow>) {
    let RowContext { expansion, ledger, agent_filter, .. } = *cx;
    let is_expanded = expansion.is_expanded(&sym.id, RowKind::Symbol);
    let read_depth = match agent_filter {
        Some(agent_id) => ledger.depth_of_for_agent(&sym.id, agent_id),
        None => ledger.depth_of(&sym.id),
    };
    // A collapsed symbol summarizes its descendants, as a file row does: reads
    // land on innermost symbols, so without this a method read inside
    // `impl App` leaves every visible row grey under an amber file.
    let (coverage_status, coverage_seen, coverage_total, coverage_full, stale_count) = if !is_expanded && !sym.children.is_empty() {
        let (total, seen, full) = count_symbols(&sym.children, ledger, agent_filter);
        (Some(coverage_status_from_counts(total, seen, full)), seen, total, full, count_stale(&sym.children, ledger))
    } else {
        (None, 0, 0, 0, 0)
    };

    rows.push(TreeRow {
        symbol_id: sym.id.clone(),
        display_name: sym.name.to_string(),
        label: sym.label,
        depth,
        kind: RowKind::Symbol,
        is_expanded,
        has_children: !sym.children.is_empty(),
        line_range: format!("L{}-{}", sym.line_range.start, sym.line_range.end),
        token_count: sym.estimated_tokens as usize,
        read_depth,
        stale: ledger.is_stale(&sym.id),
        restored: ledger.is_restored(&sym.id),
        coverage_status,
        coverage_seen,
        coverage_total,
        coverage_full,
        stale_count,
        write: cx.writes.and_then(|w| w.symbol_mark(&sym.id)),
    });

    if is_expanded {
        for child in &sym.children {
            flatten_symbol(child, depth + 1, cx, rows);
        }
    }
}

/// Convert a tool call file path (usually absolute) to a relative path matching
/// the project tree's convention. Strips the project root prefix if present.
pub fn normalize_tool_path(tool_path: &Path, project_root: &Path) -> PathBuf {
    if tool_path.is_absolute() {
        tool_path
            .strip_prefix(project_root)
            .unwrap_or(tool_path)
            .to_path_buf()
    } else {
        tool_path.to_path_buf()
    }
}

/// Unconditionally record every symbol in `symbols` (and all their descendants)
/// at the event's `read_depth`. Used when the entire file — or the entire body of
/// a named symbol — was present in the tool response.
///
/// Contrast with [`mark_targeted_symbols`], which narrows recording to only the
/// symbols that match the event's `target_symbol` or `target_lines`.
/// `implied` for symbols that come with a parent read whole: listed by the
/// parent, not themselves.
pub fn mark_file_symbols(symbols: &[SymbolNode], depth: ReadDepth, credit: &mut Credit<'_>, implied: bool) {
    for sym in symbols {
        credit.record(sym, depth, implied);
        mark_file_symbols(&sym.children, depth, credit, true);
    }
}

/// Mark only the symbols that match the tool call's targeting info.
///
/// Name-targeted matches (via `target_symbol`) bulk-mark all descendants because
/// the tool response included the entire named symbol's body. Line-range matches
/// (via `target_lines`) recurse precisely so that only overlapping child symbols
/// are promoted — preventing unread siblings from being over-credited.
pub fn mark_targeted_symbols(symbols: &[SymbolNode], event: &AgentToolCall, credit: &mut Credit<'_>) {
    for sym in symbols {
        match classify_symbol_match(sym, event) {
            MatchKind::ByName => {
                credit.record(sym, event.read_depth, false);
                // Full body was in the response — bulk-mark all descendants.
                mark_file_symbols(&sym.children, event.read_depth, credit, true);
            }
            MatchKind::ByLineOverlap => {
                credit.record(sym, event.read_depth, false);
                // Parent container overlaps the read range — recurse precisely so
                // only children whose ranges also overlap get promoted.
                mark_targeted_symbols(&sym.children, event, credit);
            }
            MatchKind::None => {
                // No match at this level — keep searching in children.
                mark_targeted_symbols(&sym.children, event, credit);
            }
        }
    }
}

/// Normalize a `/`-separated name path by stripping leading lowercase-only keyword
/// tokens from each segment.
///
/// Different tools and language servers qualify symbol names with language keywords:
/// Serena uses `"impl App/method"`, `"async def foo"`, `"class Bar/baz"` etc., while
/// ambit's tree-sitter parsers emit just the type/identifier portion: `"App/method"`,
/// `"foo"`, `"Bar/baz"`. Normalising both sides before comparison makes matching
/// language-agnostic without special-casing individual keywords.
///
/// A leading token is considered a keyword if it is one or more ASCII lowercase
/// letters only (no digits, underscores, or colons), followed by a space. The
/// stripping repeats so that multi-word prefixes like `"async def"` are fully
/// removed.
pub fn normalize_name_path(path: &str) -> String {
    path.split('/')
        .map(strip_leading_keywords)
        .collect::<Vec<_>>()
        .join("/")
}

fn strip_leading_keywords(mut s: &str) -> &str {
    loop {
        let keyword_len = s.bytes().take_while(|b| b.is_ascii_lowercase()).count();
        if keyword_len > 0 && s.len() > keyword_len && s.as_bytes()[keyword_len] == b' ' {
            s = &s[keyword_len + 1..];
        } else {
            break;
        }
    }
    s
}

/// Check if a symbol's name path matches the tool call's `target_symbol`.
///
/// Compares both the raw target and the normalised form (see [`normalize_name_path`])
/// so that tool-qualified paths like `"impl App/method"` match ambit's tree path
/// `"App/method"` without special-casing individual languages or keywords.
pub fn symbol_name_matches(sym: &SymbolNode, event: &AgentToolCall) -> bool {
    let Some(ref target_name) = event.target_symbol else { return false };

    let norm_target = normalize_name_path(target_name);

    if let Some(name_part) = sym.id.split("::").last() {
        let norm_name_part = normalize_name_path(name_part);
        // Try both raw and normalised forms: exact match or suffix match.
        for (t, np) in [
            (target_name.as_str(), name_part),
            (norm_target.as_str(), norm_name_part.as_str()),
        ] {
            if np == t || (np.len() > t.len() && np.as_bytes()[np.len() - t.len() - 1] == b'/' && np.ends_with(t)) {
                return true;
            }
        }
    }
    // Plain name match (e.g. target = "handle_key", sym.name = "handle_key").
    let norm_sym_name = normalize_name_path(&sym.name);
    sym.name.as_ref() == target_name.as_str() || norm_sym_name == norm_target
}

/// Check if a symbol's line range overlaps with the tool call's `target_lines`.
pub fn symbol_lines_match(sym: &SymbolNode, event: &AgentToolCall) -> bool {
    let Some(ref target_range) = event.target_lines else { return false };
    sym.line_range.start < target_range.end && target_range.start < sym.line_range.end
}

/// Check if a symbol matches the tool call's target_symbol or target_lines.
pub fn symbol_matches_target(sym: &SymbolNode, event: &AgentToolCall) -> bool {
    symbol_name_matches(sym, event) || symbol_lines_match(sym, event)
}

/// How a symbol matched a tool call's targeting info.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchKind {
    /// The symbol was named explicitly via `target_symbol`.
    /// The full body was in the response — mark this symbol and all descendants.
    ByName,
    /// The symbol's line range overlaps `target_lines`.
    /// Only this symbol and overlapping children should be promoted.
    ByLineOverlap,
    /// No match.
    None,
}

/// Classify how `sym` matches the tool call's targeting info.
///
/// Name targeting takes priority over line-range targeting. Returns [`MatchKind::None`]
/// when the event carries no targeting info or neither predicate fires.
pub fn classify_symbol_match(sym: &SymbolNode, event: &AgentToolCall) -> MatchKind {
    if event.target_symbol.is_some() && symbol_name_matches(sym, event) {
        MatchKind::ByName
    } else if event.target_lines.is_some() && symbol_lines_match(sym, event) {
        MatchKind::ByLineOverlap
    } else {
        MatchKind::None
    }
}

/// Classify a file's coverage as fully covered, all seen, partially covered, or not covered.
/// "Fully covered" means every symbol has been read at FullBody depth.
/// "All seen" means every symbol has been seen (depth > Unseen) but not all at FullBody.
fn coverage_status_from_counts(total: usize, seen: usize, full: usize) -> FileCoverageStatus {
    if total == 0 || full == 0 {
        if seen > 0 && seen == total {
            FileCoverageStatus::AllSeen
        } else if seen > 0 {
            FileCoverageStatus::PartiallyCovered
        } else {
            FileCoverageStatus::NotCovered
        }
    } else if full == total {
        FileCoverageStatus::FullyCovered
    } else if seen == total {
        FileCoverageStatus::AllSeen
    } else {
        FileCoverageStatus::PartiallyCovered
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::*;
    use crate::symbols::FileSymbols;
    use std::path::Path;

    #[test]
    fn normalize_tool_path_absolute() {
        let result = normalize_tool_path(
            Path::new("/project/src/main.rs"),
            Path::new("/project"),
        );
        assert_eq!(result, PathBuf::from("src/main.rs"));
    }

    #[test]
    fn normalize_tool_path_relative() {
        let result = normalize_tool_path(
            Path::new("src/main.rs"),
            Path::new("/project"),
        );
        assert_eq!(result, PathBuf::from("src/main.rs"));
    }

    #[test]
    fn mark_file_symbols_recursive() {
        let child = sym("mock/f.rs::child", "child");
        let parent = sym_with_children("mock/f.rs::parent", "parent", vec![child]);
        let event = tool_call("Read", "mock/f.rs", ReadDepth::FullBody);
        let mut ledger = ContextLedger::new();
        let mut cache = crate::tracking::alignment::DepthOrdinalCache::new();

        mark_file_symbols(&[parent], event.read_depth, &mut Credit::new(&mut ledger, &mut cache, &event.agent_id), false);

        assert_eq!(ledger.depth_of("mock/f.rs::parent"), ReadDepth::FullBody);
        assert_eq!(ledger.depth_of("mock/f.rs::child"), ReadDepth::FullBody);
    }

    #[test]
    fn mark_targeted_by_name() {
        let s1 = sym("mock/f.rs::alpha", "alpha");
        let s2 = sym("mock/f.rs::beta", "beta");
        let event = tool_call_targeted("find_symbol", "mock/f.rs", ReadDepth::FullBody, "beta");
        let mut ledger = ContextLedger::new();
        let mut cache = crate::tracking::alignment::DepthOrdinalCache::new();

        mark_targeted_symbols(&[s1, s2], &event, &mut Credit::new(&mut ledger, &mut cache, &event.agent_id));

        assert_eq!(ledger.depth_of("mock/f.rs::alpha"), ReadDepth::Unseen);
        assert_eq!(ledger.depth_of("mock/f.rs::beta"), ReadDepth::FullBody);
    }

    #[test]
    fn mark_targeted_by_lines() {
        let s1 = sym_with_lines("mock/f.rs::a", "a", 1, 5);
        let s2 = sym_with_lines("mock/f.rs::b", "b", 10, 20);
        let event = tool_call_lines("Read", "mock/f.rs", ReadDepth::FullBody, 12, 18);
        let mut ledger = ContextLedger::new();
        let mut cache = crate::tracking::alignment::DepthOrdinalCache::new();

        mark_targeted_symbols(&[s1, s2], &event, &mut Credit::new(&mut ledger, &mut cache, &event.agent_id));

        assert_eq!(ledger.depth_of("mock/f.rs::a"), ReadDepth::Unseen);
        assert_eq!(ledger.depth_of("mock/f.rs::b"), ReadDepth::FullBody);
    }

    #[test]
    fn coverage_status_from_counts_variants() {
        let mut ledger = ContextLedger::new();
        let syms = vec![sym("s1", "s1"), sym("s2", "s2")];

        // No coverage.
        let (total, seen, full) = count_symbols(&syms, &ledger, None);
        assert_eq!(coverage_status_from_counts(total, seen, full), FileCoverageStatus::NotCovered);

        // Partial: one seen, one unseen → PartiallyCovered.
        ledger.record("s1".into(), ReadDepth::NameOnly, [0; 32], "ag".into(), 10);
        let (total, seen, full) = count_symbols(&syms, &ledger, None);
        assert_eq!(coverage_status_from_counts(total, seen, full), FileCoverageStatus::PartiallyCovered);

        // All seen (both NameOnly) but none FullBody → AllSeen.
        ledger.record("s2".into(), ReadDepth::NameOnly, [0; 32], "ag".into(), 10);
        let (total, seen, full) = count_symbols(&syms, &ledger, None);
        assert_eq!(coverage_status_from_counts(total, seen, full), FileCoverageStatus::AllSeen);

        // One FullBody, one NameOnly → AllSeen (all seen, not all full).
        ledger.record("s1".into(), ReadDepth::FullBody, [0; 32], "ag".into(), 10);
        let (total, seen, full) = count_symbols(&syms, &ledger, None);
        assert_eq!(coverage_status_from_counts(total, seen, full), FileCoverageStatus::AllSeen);

        // Full: both FullBody.
        ledger.record("s2".into(), ReadDepth::FullBody, [0; 32], "ag".into(), 10);
        let (total, seen, full) = count_symbols(&syms, &ledger, None);
        assert_eq!(coverage_status_from_counts(total, seen, full), FileCoverageStatus::FullyCovered);

        // Direct FullBody with unseen siblings → PartiallyCovered (full > 0, seen < total).
        assert_eq!(coverage_status_from_counts(3, 1, 1), FileCoverageStatus::PartiallyCovered);
    }

    #[test]
    fn symbol_matches_target_formats() {
        // Plain name match.
        let s = sym("mock/app.rs::App/handle_key", "handle_key");
        let event = tool_call_targeted("find_symbol", "mock/app.rs", ReadDepth::FullBody, "handle_key");
        assert!(symbol_matches_target(&s, &event));

        // Name path suffix match.
        let event2 = tool_call_targeted("find_symbol", "mock/app.rs", ReadDepth::FullBody, "App/handle_key");
        assert!(symbol_matches_target(&s, &event2));

        // Non-match.
        let event3 = tool_call_targeted("find_symbol", "mock/app.rs", ReadDepth::FullBody, "other_fn");
        assert!(!symbol_matches_target(&s, &event3));
    }

    // --- normalize_name_path ---

    #[test]
    fn normalize_name_path_strips_impl_prefix() {
        assert_eq!(normalize_name_path("impl App"), "App");
        assert_eq!(normalize_name_path("impl App/handle_key"), "App/handle_key");
        assert_eq!(normalize_name_path("impl Trait for App"), "Trait for App");
    }

    #[test]
    fn normalize_name_path_strips_multi_word_prefix() {
        // "async def" prefix (Python-style)
        assert_eq!(normalize_name_path("async def foo"), "foo");
        // "class" prefix
        assert_eq!(normalize_name_path("class Foo/bar"), "Foo/bar");
    }

    #[test]
    fn normalize_name_path_leaves_plain_names_unchanged() {
        assert_eq!(normalize_name_path("App/handle_key"), "App/handle_key");
        assert_eq!(normalize_name_path("handle_key"), "handle_key");
        // Starts with uppercase — not a keyword.
        assert_eq!(normalize_name_path("Display for App"), "Display for App");
        // Has colons — not a simple keyword token.
        assert_eq!(normalize_name_path("std::fmt::Display for App"), "std::fmt::Display for App");
    }

    #[test]
    fn normalize_name_path_per_segment() {
        // Each `/`-separated segment is normalised independently.
        assert_eq!(normalize_name_path("impl App/fn handle_key"), "App/handle_key");
    }

    // --- symbol_name_matches with keyword-qualified paths ---

    #[test]
    fn symbol_name_matches_impl_qualified_method() {
        // Serena emits "impl App/handle_key"; ambit's tree has id "mock/app.rs::App/handle_key".
        let s = sym("mock/app.rs::App/handle_key", "handle_key");
        let event = tool_call_targeted("find_symbol", "mock/app.rs", ReadDepth::FullBody, "impl App/handle_key");
        assert!(symbol_name_matches(&s, &event));
    }

    #[test]
    fn symbol_name_matches_impl_block_itself() {
        // Serena emits "impl App"; ambit's tree has id "mock/app.rs::App" (impl block).
        let s = sym("mock/app.rs::App", "App");
        let event = tool_call_targeted("find_symbol", "mock/app.rs", ReadDepth::Signature, "impl App");
        assert!(symbol_name_matches(&s, &event));
    }

    #[test]
    fn symbol_name_matches_trait_impl() {
        // "impl Display for App" → normalised "Display for App"
        let s = sym("mock/app.rs::Display for App", "Display for App");
        let event = tool_call_targeted("find_symbol", "mock/app.rs", ReadDepth::FullBody, "impl Display for App");
        assert!(symbol_name_matches(&s, &event));
    }

    // --- line-range precision: siblings not over-marked ---

    #[test]
    fn mark_targeted_by_lines_does_not_mark_siblings() {
        // Parent impl block spans lines 1..100; two sibling methods inside.
        let method_a = sym_with_lines("f.rs::Foo/method_a", "method_a", 5, 20);
        let method_b = sym_with_lines("f.rs::Foo/method_b", "method_b", 50, 70);
        let impl_block = sym_with_children_and_lines(
            "f.rs::Foo", "Foo", vec![method_a, method_b], 1, 100,
        );

        // Read only covers method_b's range.
        let event = tool_call_lines("Read", "f.rs", ReadDepth::FullBody, 50, 70);
        let mut ledger = ContextLedger::new();
        let mut cache = crate::tracking::alignment::DepthOrdinalCache::new();

        mark_targeted_symbols(&[impl_block], &event, &mut Credit::new(&mut ledger, &mut cache, &event.agent_id));

        // The parent is marked (it overlaps), but method_a is NOT.
        assert_eq!(ledger.depth_of("f.rs::Foo"), ReadDepth::FullBody);
        assert_eq!(ledger.depth_of("f.rs::Foo/method_b"), ReadDepth::FullBody);
        assert_eq!(ledger.depth_of("f.rs::Foo/method_a"), ReadDepth::Unseen);
    }

    #[test]
    fn mark_targeted_by_name_still_marks_all_children() {
        // Name-based match on the impl block should still bulk-mark all children.
        let method_a = sym_with_lines("f.rs::Foo/method_a", "method_a", 5, 20);
        let method_b = sym_with_lines("f.rs::Foo/method_b", "method_b", 50, 70);
        let impl_block = sym_with_children_and_lines(
            "f.rs::Foo", "Foo", vec![method_a, method_b], 1, 100,
        );

        // find_symbol("impl Foo", include_body=true) — all children embedded in response.
        let event = tool_call_targeted("find_symbol", "f.rs", ReadDepth::FullBody, "impl Foo");
        let mut ledger = ContextLedger::new();
        let mut cache = crate::tracking::alignment::DepthOrdinalCache::new();

        mark_targeted_symbols(&[impl_block], &event, &mut Credit::new(&mut ledger, &mut cache, &event.agent_id));

        assert_eq!(ledger.depth_of("f.rs::Foo"), ReadDepth::FullBody);
        assert_eq!(ledger.depth_of("f.rs::Foo/method_a"), ReadDepth::FullBody);
        assert_eq!(ledger.depth_of("f.rs::Foo/method_b"), ReadDepth::FullBody);
    }

    // --- App method tests ---

    fn test_app(files: Vec<FileSymbols>) -> App {
        let tree = project(files);
        App::new(tree, PathBuf::from("/test/project"))
    }

    // --- open_selected_in_editor ---

    #[test]
    fn open_selected_in_editor_resolves_a_leaf_symbol_row() {
        let leaf = sym_with_lines("mock/f.rs::alpha", "alpha", 42, 50);
        let mut app = test_app(vec![file("mock/f.rs", vec![leaf])]);
        // The file row has children (its one symbol), so expand it first to
        // put the leaf symbol row into `tree_rows`.
        app.set_expanded("mock/f.rs", RowKind::File, true);
        app.selected_index = app
            .tree_rows
            .iter()
            .position(|r| r.symbol_id == "mock/f.rs::alpha")
            .expect("leaf row present");

        app.open_selected_in_editor();

        assert_eq!(
            app.pending_editor_request,
            Some((PathBuf::from("/test/project/mock/f.rs"), 42))
        );
    }

    #[test]
    fn open_selected_in_editor_on_a_file_row_opens_at_line_one() {
        // A file with no symbols renders a leaf (childless) file-header row.
        let mut app = test_app(vec![file("mock/empty.rs", vec![])]);
        app.selected_index = 0;
        assert!(app.tree_rows[0].is_file());
        assert!(!app.tree_rows[0].has_children);

        app.open_selected_in_editor();

        assert_eq!(
            app.pending_editor_request,
            Some((PathBuf::from("/test/project/mock/empty.rs"), 1))
        );
    }

    #[test]
    fn open_selected_in_editor_on_a_row_with_children_is_a_no_op() {
        let child = sym("mock/f.rs::child", "child");
        let parent = sym_with_children("mock/f.rs::parent", "parent", vec![child]);
        let mut app = test_app(vec![file("mock/f.rs", vec![parent])]);
        // Selected row is the file header, which has children (its symbol).
        app.selected_index = 0;
        assert!(app.tree_rows[0].has_children);

        app.open_selected_in_editor();

        assert_eq!(app.pending_editor_request, None);
    }

    #[test]
    fn open_selected_in_editor_on_an_empty_tree_does_not_panic() {
        let mut app = test_app(vec![]);
        app.open_selected_in_editor();
        assert_eq!(app.pending_editor_request, None);
    }

    /// End-to-end wiring for cold-start rehydrate: the journal is found by
    /// session id under the project root, folded in, and the alignment cache
    /// is fed the restored per-agent depths.
    #[test]
    fn rehydrate_from_journal_reads_the_session_file_and_feeds_the_depth_cache() {
        use ambits_journal_test_support::*;

        let dir = tempfile::tempdir().unwrap();
        let syms = vec![sym("mock/f.rs::alpha", "alpha")];
        let tree = project(vec![file("mock/f.rs", syms)]);
        let current = tree.files[0].symbols[0].content_hash;

        let mut app = App::new(tree, dir.path().to_path_buf());
        app.set_session_id(Some("sess-1".into()));
        write_journal(dir.path(), "sess-1", &[("mock/f.rs::alpha", "agent-9", current)]);

        let stats = app.rehydrate_from_journal().expect("journal was found");

        assert_eq!(stats.inserted, 1);
        assert_eq!(stats.drifted, 0);
        assert_eq!(app.ledger.depth_of("mock/f.rs::alpha"), ReadDepth::FullBody);
        assert_eq!(
            app.depth_cache.get("mock/f.rs::alpha", "agent-9"),
            4,
            "the alignment popup sees restored coverage too"
        );
    }

    /// No journal is not an error — it is the normal state for a session whose
    /// TUI has never run.
    #[test]
    fn rehydrate_from_journal_is_a_no_op_without_a_journal() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = App::new(
            project(vec![file("mock/f.rs", vec![sym("mock/f.rs::alpha", "alpha")])]),
            dir.path().to_path_buf(),
        );
        app.set_session_id(Some("sess-none".into()));
        assert!(app.rehydrate_from_journal().is_none());
    }

    /// Without a session id there is no journal to name, and guessing one
    /// would merge unrelated sessions.
    #[test]
    fn rehydrate_from_journal_needs_a_session_id() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = App::new(project(vec![]), dir.path().to_path_buf());
        assert!(app.rehydrate_from_journal().is_none());
    }

    /// Minimal journal writer, so this test exercises the real on-disk format
    /// and path layout rather than a hand-built `JournalContents`.
    mod ambits_journal_test_support {
        use crate::journal::*;
        use std::path::Path;

        pub fn write_journal(root: &Path, session: &str, reads: &[(&str, &str, [u8; 32])]) {
            let dir = root.join(JOURNAL_SUBDIR);
            std::fs::create_dir_all(&dir).unwrap();
            let mut out = String::new();
            for (symbol_id, agent, hash) in reads {
                out.push_str(
                    &serde_json::to_string(&Record::Read {
                        symbol_id: (*symbol_id).into(),
                        hash: encode_hash(hash),
                        depth: DepthDto::FullBody,
                        agent: Some((*agent).into()),
                    })
                    .unwrap(),
                );
                out.push('\n');
            }
            std::fs::write(dir.join(format!("{session}.ndjson")), out).unwrap();
        }
    }

    /// Reading through `ambits show` must count. Without this, using ambit's
    /// own lookup makes coverage *fall* relative to a plain Read, which
    /// inverts the incentive the tool exists to create.
    #[test]
    fn a_show_command_credits_the_symbols_it_names() {
        let syms = vec![sym("mock/f.rs::alpha", "alpha"), sym("mock/f.rs::beta", "beta")];
        let mut app = test_app(vec![file("mock/f.rs", syms)]);

        let mut event = tool_call("Bash", "", ReadDepth::FullBody);
        event.file_path = None;
        event.target_selectors = vec![("mock/f.rs::alpha".into(), ReadDepth::FullBody)];
        app.process_agent_event(event);

        assert_eq!(app.ledger.depth_of("mock/f.rs::alpha"), ReadDepth::FullBody);
        assert_eq!(
            app.ledger.depth_of("mock/f.rs::beta"),
            ReadDepth::Unseen,
            "only the named symbol is credited"
        );
    }

    /// A call lists what it was credited with in its own right: a search
    /// that matched a parent and a symbol inside it lists both; a read of
    /// the whole file, its top-level symbols — their children came with them.
    #[test]
    fn a_call_lists_what_it_read_in_its_own_right() {
        let tree = project(vec![file("mock/f.rs", vec![sym_with_children("mock/f.rs::App", "App", vec![sym("mock/f.rs::App/run", "run")]), sym("mock/f.rs::free", "free")])]);
        let (mut ledger, mut cache) = (ContextLedger::new(), crate::tracking::alignment::DepthOrdinalCache::new());
        let shown = vec![("mock/f.rs::App".to_string(), ReadDepth::NameOnly), ("mock/f.rs::App/run".to_string(), ReadDepth::NameOnly)];
        let search = crate::ingest::ToolFinished { id: Arc::from("s"), agent_id: Arc::from("a"), timestamp: String::new(), error: false, child_agent: None, message: None, shown: shown.clone() };
        assert_eq!(apply_shown(&tree, &search, &mut ledger, &mut cache), shown);
        let whole = tool_call("Read", "mock/f.rs", ReadDepth::FullBody);
        let listed = apply_tool_call(&tree, Path::new(""), &whole, &mut ledger, &mut cache);
        assert_eq!(listed, vec![("mock/f.rs::App".to_string(), ReadDepth::FullBody), ("mock/f.rs::free".to_string(), ReadDepth::FullBody)]);
    }

    /// A file's contents are made once, and made again when the file
    /// changes: its symbols' merkle hashes say so.
    #[test]
    fn file_contents_are_kept_until_the_file_changes() {
        let mut app = test_app(vec![file("mock/f.rs", vec![sym("mock/f.rs::alpha", "alpha")])]);
        let first = app.file_contents("mock/f.rs").unwrap();
        assert!(Arc::ptr_eq(&first, &app.file_contents("mock/f.rs").unwrap()), "kept");
        app.project_tree.files[0].symbols[0].merkle_hash = [9; 32];
        assert!(!Arc::ptr_eq(&first, &app.file_contents("mock/f.rs").unwrap()), "made again");
        assert!(app.file_contents("mock/absent.rs").is_none());
    }

    /// What a search printed is credited when its result arrives — at the
    /// depth it carries, never lowering a deeper read.
    #[test]
    fn a_search_result_credits_what_it_showed() {
        let syms = vec![sym("mock/f.rs::alpha", "alpha"), sym("mock/f.rs::beta", "beta"), sym("mock/f.rs::gamma", "gamma")];
        let mut app = test_app(vec![file("mock/f.rs", syms)]);
        let mut read = tool_call("Bash", "", ReadDepth::FullBody);
        read.file_path = None;
        read.target_selectors = vec![("mock/f.rs::beta".into(), ReadDepth::FullBody)];
        app.process_agent_event(read);

        app.process_tool_finished(&crate::ingest::ToolFinished {
            id: Arc::from("s1"),
            agent_id: Arc::from("main"),
            timestamp: "2026-09-29T10:00:00Z".into(),
            error: false,
            child_agent: None,
            message: None,
            shown: vec![("mock/f.rs::alpha".into(), ReadDepth::NameOnly), ("mock/f.rs::beta".into(), ReadDepth::NameOnly)],
        });
        assert_eq!(app.ledger.depth_of("mock/f.rs::alpha"), ReadDepth::NameOnly, "seen, not read");
        assert_eq!(app.ledger.depth_of("mock/f.rs::beta"), ReadDepth::FullBody, "a full read stays full");
        assert_eq!(app.ledger.depth_of("mock/f.rs::gamma"), ReadDepth::Unseen, "not shown");
    }

    /// Selectors carry their own location, so one command can legitimately
    /// name symbols in different files.
    #[test]
    fn selectors_credit_symbols_across_files() {
        let mut app = test_app(vec![
            file("a.rs", vec![sym("a.rs::one", "one")]),
            file("b.rs", vec![sym("b.rs::two", "two")]),
        ]);

        let mut event = tool_call("Bash", "", ReadDepth::FullBody);
        event.file_path = None;
        event.target_selectors = vec![
            ("a.rs::one".into(), ReadDepth::FullBody),
            ("b.rs::two".into(), ReadDepth::FullBody),
        ];
        app.process_agent_event(event);

        assert_eq!(app.ledger.depth_of("a.rs::one"), ReadDepth::FullBody);
        assert_eq!(app.ledger.depth_of("b.rs::two"), ReadDepth::FullBody);
    }

    /// A hash selector resolves the same way the lookup itself does, including
    /// by prefix.
    #[test]
    fn a_hash_selector_credits_the_symbol_that_owns_it() {
        let node = sym("a.rs::only", "only");
        let hex = crate::journal::encode_hash(&node.content_hash);
        let mut app = test_app(vec![file("a.rs", vec![node])]);

        let mut event = tool_call("Bash", "", ReadDepth::FullBody);
        event.file_path = None;
        event.target_selectors = vec![(hex[3..11].to_string(), ReadDepth::FullBody)];
        app.process_agent_event(event);

        assert_eq!(app.ledger.depth_of("a.rs::only"), ReadDepth::FullBody);
    }

    // --- event_log_path_and_target / resolve_selector_path ---

    /// The regression this exists for: a selector-driven event (`ambits show
    /// <id>`) has neither `file_path` nor `target_symbol`/`target_lines`, so
    /// both columns used to print `-` even though real data existed.
    #[test]
    fn selector_driven_event_hydrates_both_columns() {
        let app = test_app(vec![file("a.rs", vec![sym("a.rs::one", "one")])]);

        let mut event = tool_call("Bash", "", ReadDepth::FullBody);
        event.file_path = None;
        event.target_selectors = vec![("a.rs::one".into(), ReadDepth::FullBody)];

        let (path, target) = event_log_path_and_target(&app.project_tree, &event);
        assert_eq!(path, "a.rs");
        assert_eq!(target, "a.rs::one");
    }

    #[test]
    fn multiple_selectors_join_the_target_and_note_extra_paths() {
        let app = test_app(vec![
            file("a.rs", vec![sym("a.rs::one", "one")]),
            file("b.rs", vec![sym("b.rs::two", "two")]),
        ]);

        let mut event = tool_call("Bash", "", ReadDepth::FullBody);
        event.file_path = None;
        event.target_selectors = vec![
            ("a.rs::one".into(), ReadDepth::FullBody),
            ("b.rs::two".into(), ReadDepth::FullBody),
        ];

        let (path, target) = event_log_path_and_target(&app.project_tree, &event);
        assert_eq!(path, "a.rs (+1 more)");
        assert_eq!(target, "a.rs::one, b.rs::two");
    }

    /// A selector naming a symbol that no longer resolves shouldn't crash the
    /// log line — it just can't hydrate a path from nothing real.
    #[test]
    fn an_unmatched_selector_falls_back_to_dash_in_the_log() {
        let app = test_app(vec![file("a.rs", vec![sym("a.rs::one", "one")])]);

        let mut event = tool_call("Bash", "", ReadDepth::FullBody);
        event.file_path = None;
        event.target_selectors = vec![("a.rs::nope".into(), ReadDepth::FullBody)];

        let (path, target) = event_log_path_and_target(&app.project_tree, &event);
        assert_eq!(path, "-");
        assert_eq!(target, "a.rs::nope");
    }

    /// A tool call with a real `file_path` and `target_symbol` is unaffected
    /// by the selector-hydration fallback — the existing fields still win.
    #[test]
    fn ordinary_file_and_symbol_targeted_events_are_unaffected() {
        let app = test_app(vec![file("a.rs", vec![sym("a.rs::one", "one")])]);
        let event = tool_call_targeted("Read", "/test/project/a.rs", ReadDepth::FullBody, "a.rs::one");

        let (path, target) = event_log_path_and_target(&app.project_tree, &event);
        assert_eq!(path, "/test/project/a.rs");
        assert_eq!(target, "a.rs::one");
    }

    /// A selector naming nothing is not an error and must not disturb the
    /// ledger — the lookup may simply have missed.
    #[test]
    fn an_unmatched_selector_credits_nothing() {
        let mut app = test_app(vec![file("a.rs", vec![sym("a.rs::one", "one")])]);

        let mut event = tool_call("Bash", "", ReadDepth::FullBody);
        event.file_path = None;
        event.target_selectors = vec![("a.rs::nope".into(), ReadDepth::FullBody)];
        app.process_agent_event(event);

        assert_eq!(app.ledger.total_seen(), 0);
    }

    #[test]
    fn process_agent_event_updates_ledger() {
        let syms = vec![sym("mock/f.rs::alpha", "alpha"), sym("mock/f.rs::beta", "beta")];
        let mut app = test_app(vec![file("mock/f.rs", syms)]);

        let event = tool_call("Read", "/test/project/mock/f.rs", ReadDepth::FullBody);
        app.process_agent_event(event);

        assert_eq!(app.ledger.depth_of("mock/f.rs::alpha"), ReadDepth::FullBody);
        assert_eq!(app.ledger.depth_of("mock/f.rs::beta"), ReadDepth::FullBody);
    }

    #[test]
    fn process_agent_event_targeted() {
        let syms = vec![sym("mock/f.rs::alpha", "alpha"), sym("mock/f.rs::beta", "beta")];
        let mut app = test_app(vec![file("mock/f.rs", syms)]);

        let event = tool_call_targeted("find_symbol", "/test/project/mock/f.rs", ReadDepth::FullBody, "beta");
        app.process_agent_event(event);

        assert_eq!(app.ledger.depth_of("mock/f.rs::alpha"), ReadDepth::Unseen);
        assert_eq!(app.ledger.depth_of("mock/f.rs::beta"), ReadDepth::FullBody);
    }

    #[test]
    fn process_agent_event_tracks_agents() {
        let mut app = test_app(vec![file("mock/f.rs", vec![sym("mock/f.rs::a", "a")])]);

        let mut e1 = tool_call("Read", "/test/project/mock/f.rs", ReadDepth::FullBody);
        e1.agent_id = "agent-1".into();
        let mut e2 = tool_call("Read", "/test/project/mock/f.rs", ReadDepth::FullBody);
        e2.agent_id = "agent-2".into();

        app.process_agent_event(e1);
        app.process_agent_event(e2);

        assert_eq!(app.agents_seen.len(), 2);
        assert!(app.agents_seen.contains(&"agent-1".to_string()));
        assert!(app.agents_seen.contains(&"agent-2".to_string()));
    }

    #[test]
    fn cycle_agent_filter_backward_wraps() {
        let mut app = test_app(vec![file("mock/f.rs", vec![sym("mock/f.rs::a", "a")])]);

        let mut e1 = tool_call("Read", "/test/project/mock/f.rs", ReadDepth::FullBody);
        e1.agent_id = "agent-1".into();
        let mut e2 = tool_call("Read", "/test/project/mock/f.rs", ReadDepth::FullBody);
        e2.agent_id = "agent-2".into();

        app.process_agent_event(e1);
        app.process_agent_event(e2);

        // Start at None (All), backward should go to last agent
        assert_eq!(app.agent_filter, None);
        app.cycle_agent_filter_backward();
        assert_eq!(app.agent_filter, Some("agent-2".to_string()));
        app.cycle_agent_filter_backward();
        assert_eq!(app.agent_filter, Some("agent-1".to_string()));
        app.cycle_agent_filter_backward();
        assert_eq!(app.agent_filter, None); // wraps back to All
    }

    #[test]
    fn agent_selection_index_navigates_and_applies() {
        let mut app = test_app(vec![file("mock/f.rs", vec![sym("mock/f.rs::a", "a")])]);

        let mut e1 = tool_call("Read", "/test/project/mock/f.rs", ReadDepth::FullBody);
        e1.agent_id = "agent-1".into();
        let mut e2 = tool_call("Read", "/test/project/mock/f.rs", ReadDepth::FullBody);
        e2.agent_id = "agent-2".into();

        app.process_agent_event(e1);
        app.process_agent_event(e2);
        app.focus = FocusPanel::Right;

        // Start at 0 (All)
        assert_eq!(app.agent_selection_index, 0);
        assert_eq!(app.agent_filter, None);

        // Move down to first agent
        app.move_agent_selection(1);
        assert_eq!(app.agent_selection_index, 1);
        // Not applied yet
        assert_eq!(app.agent_filter, None);

        // Apply selection
        app.apply_agent_selection();
        assert_eq!(app.agent_filter, Some("agent-1".to_string()));
        assert_eq!(app.agent_selection_index, 1);

        // Move down again and apply
        app.move_agent_selection(1);
        assert_eq!(app.agent_selection_index, 2);
        app.apply_agent_selection();
        assert_eq!(app.agent_filter, Some("agent-2".to_string()));

        // Move back to All and apply
        app.move_agent_selection(-2);
        assert_eq!(app.agent_selection_index, 0);
        app.apply_agent_selection();
        assert_eq!(app.agent_filter, None);
    }

    /// The stats panel's agent list and its selection cursor must be indexed
    /// off the *same* list. `flattened_agents` walks the hierarchy — which
    /// includes the seeded root even when the orchestrator never read a file
    /// itself — while `agents_seen` holds only agents that emitted an event.
    /// When those lengths differ, row `n` on screen and the agent that row
    /// `n` selects are two different agents.
    #[test]
    fn agent_selection_indexes_the_list_the_panel_renders() {
        let mut app = test_app(vec![file("mock/f.rs", vec![sym("mock/f.rs::a", "a")])]);
        app.set_session_id(Some("session-main".to_string()));

        // Orchestrator-only session: the root dispatches one sub-agent and
        // never reads a file itself.
        let mut e = tool_call("Read", "/test/project/mock/f.rs", ReadDepth::FullBody);
        e.agent_id = "a63c858997b4e6124".into();
        app.process_agent_event(e);

        let flat = app.flattened_agents();
        assert_eq!(flat.len(), 2, "root plus its one sub-agent");

        // Step the cursor through every agent row and apply it; each row must
        // resolve to the agent rendered on that row.
        app.focus = FocusPanel::Right;
        for (i, (agent_id, _)) in flat.iter().enumerate() {
            app.move_agent_selection(1);
            app.apply_agent_selection();
            assert_eq!(
                app.agent_filter.as_deref(),
                Some(agent_id.as_str()),
                "row {i} renders {agent_id} but selects {:?}",
                app.agent_filter,
            );
        }

        // And one more step wraps back to "[All]".
        app.move_agent_selection(1);
        app.apply_agent_selection();
        assert_eq!(app.agent_selection_index, 0);
        assert_eq!(app.agent_filter, None);
    }

    /// "[All]" is the union of the agent rows beneath it, so every read it
    /// counts has to be reachable by selecting *some* agent in the list — and
    /// when the list holds a single agent, that agent's numbers must equal
    /// "[All]"'s exactly.
    ///
    /// `rehydrate_from_journal` is the path that can break this: it installs
    /// per-agent reads straight into the ledger under whatever agent id the
    /// journal recorded, while the panel lists only agents this run replayed
    /// an event for. A sub-agent whose own log file is gone survives in the
    /// journal but not in the replay, and its coverage then shows up in
    /// "[All]" with no row that can account for it.
    #[test]
    fn all_counts_only_coverage_the_panel_can_attribute() {
        use ambits_journal_test_support::*;

        let dir = tempfile::tempdir().unwrap();
        let syms = vec![
            sym("mock/f.rs::alpha", "alpha"),
            sym("mock/f.rs::beta", "beta"),
        ];
        let tree = project(vec![file("mock/f.rs", syms)]);
        let beta_hash = tree.files[0].symbols[1].content_hash;

        let mut app = App::new(tree, dir.path().to_path_buf());
        app.set_session_id(Some("sess-1".into()));

        // The live replay only produces the root session's own read of alpha.
        let mut e = tool_call("Read", "mock/f.rs", ReadDepth::FullBody);
        e.agent_id = "sess-1".into();
        e.target_symbol = Some("alpha".into());
        app.process_agent_event(e);

        // The journal remembers a sub-agent's read of beta that this run's
        // replay could not reproduce.
        write_journal(
            dir.path(),
            "sess-1",
            &[("mock/f.rs::beta", "a63c858997b4e6124", beta_hash)],
        );
        app.rehydrate_from_journal().expect("journal was found");
        assert_eq!(app.ledger.total_seen(), 2, "[All] sees both reads");

        let listed = app.flattened_agents();
        let attributable = app
            .ledger
            .entries
            .values()
            .filter(|e| {
                listed.iter().any(|(id, _)| {
                    e.agent_depths.get(id).copied().unwrap_or(ReadDepth::Unseen).is_seen()
                })
            })
            .count();
        assert_eq!(
            attributable,
            app.ledger.total_seen(),
            "[All] counts a read no listed agent can account for; listed: {:?}",
            listed,
        );
    }

    #[test]
    fn activity_scroll_offset_resets_on_new_event() {
        let mut app = test_app(vec![file("mock/f.rs", vec![sym("mock/f.rs::a", "a")])]);
        app.activity_scroll_offset = 10;

        let e = tool_call("Read", "/test/project/mock/f.rs", ReadDepth::FullBody);
        app.process_agent_event(e);

        assert_eq!(app.activity_scroll_offset, 0);
    }

    #[test]
    fn handle_mouse_scroll_routes_by_focus() {
        use crossterm::event::{MouseEvent, MouseEventKind};

        let mut app = test_app(vec![file("mock/f.rs", vec![sym("mock/f.rs::a", "a")])]);

        // Add some activity so scroll offset can increase
        for _ in 0..20 {
            let e = tool_call("Read", "/test/project/mock/f.rs", ReadDepth::FullBody);
            app.process_agent_event(e);
        }

        // Focus activity panel and scroll up
        app.focus = FocusPanel::Feed;
        let scroll_up = MouseEvent {
            kind: MouseEventKind::ScrollUp,
            column: 0,
            row: 0,
            modifiers: crossterm::event::KeyModifiers::empty(),
        };
        app.handle_mouse(scroll_up);
        assert_eq!(app.activity_scroll_offset, 3);

        // Scroll down should decrease offset
        let scroll_down = MouseEvent {
            kind: MouseEventKind::ScrollDown,
            column: 0,
            row: 0,
            modifiers: crossterm::event::KeyModifiers::empty(),
        };
        app.handle_mouse(scroll_down);
        assert_eq!(app.activity_scroll_offset, 0);

        // Focus tree panel — scroll should move tree selection, not activity
        app.focus = FocusPanel::Left;
        app.activity_scroll_offset = 5;
        app.handle_mouse(scroll_up);
        assert_eq!(app.activity_scroll_offset, 5); // unchanged
    }

    #[test]
    fn rebuild_tree_rows_alphabetical() {
        let app = test_app(vec![
            file("mock/a.rs", vec![sym("mock/a.rs::a", "a")]),
            file("mock/z.rs", vec![sym("mock/z.rs::z", "z")]),
        ]);
        // Alphabetical mode preserves the file insertion order.
        let file_rows: Vec<&str> = app.tree_rows.iter()
            .filter(|r| r.is_file())
            .map(|r| r.display_name.as_str())
            .collect();
        assert_eq!(file_rows, vec!["mock/a.rs", "mock/z.rs"]);
    }

    #[test]
    fn rebuild_tree_rows_by_coverage() {
        let syms_a = vec![sym("mock/a.rs::x", "x")];
        let syms_b = vec![sym("mock/b.rs::y", "y")];
        let mut app = test_app(vec![
            file("mock/a.rs", syms_a),
            file("mock/b.rs", syms_b),
        ]);

        // Mark mock/a.rs as partially covered.
        app.ledger.record("mock/a.rs::x".into(), ReadDepth::FullBody, [0; 32], "ag".into(), 10);
        app.sort_mode = SortMode::ByCoverage;
        app.rebuild_tree_rows();

        let file_rows: Vec<&str> = app.tree_rows.iter()
            .filter(|r| r.is_file())
            .map(|r| r.display_name.as_str())
            .collect();
        // PartiallyCovered (mock/a.rs) sorts before NotCovered (mock/b.rs).
        assert_eq!(file_rows, vec!["mock/a.rs", "mock/b.rs"]);
    }

    #[test]
    fn process_agent_event_populates_agent_tree() {
        let mut app = test_app(vec![file("mock/f.rs", vec![sym("mock/f.rs::a", "a")])]);

        // Simulate main session event (no "agent-" prefix → becomes root).
        let mut e1 = tool_call("Read", "/test/project/mock/f.rs", ReadDepth::FullBody);
        e1.agent_id = "session-main".into();
        e1.label = "Main session".into();
        app.process_agent_event(e1);

        assert_eq!(app.agent_tree.root_id, Some("session-main".to_string()));
        assert!(app.agent_tree.agents.contains_key("session-main"));
        assert_eq!(app.agent_tree.agents["session-main"].label, "Main session");

        // Simulate subagent event (starts with "agent-" → child of root).
        let mut e2 = tool_call("Read", "/test/project/mock/f.rs", ReadDepth::Overview);
        e2.agent_id = "agent-abc123".into();
        e2.label = "Explore parser module".into();
        app.process_agent_event(e2);

        assert_eq!(app.agent_tree.agents.len(), 2);
        let sub = &app.agent_tree.agents["agent-abc123"];
        assert_eq!(sub.parent_id, Some("session-main".to_string()));
        assert_eq!(sub.label, "Explore parser module");
    }

    /// Regression test for the accidental-root bug: an orchestrator-only
    /// session where the root/main session never emits a file-tool event
    /// itself — every event's `agent_id` is "agent-"-prefixed. Without
    /// seeding the root from `session_id` up front, `AgentTree::add_agent`'s
    /// "first parentless agent becomes root" rule would silently promote
    /// whichever sub-agent event arrives first, corrupting every sibling
    /// relationship and breaking the alignment popup's sibling lookup.
    #[test]
    fn orchestrator_only_stream_roots_correctly() {
        let mut app = test_app(vec![file("mock/f.rs", vec![sym("mock/f.rs::a", "a")])]);
        app.set_session_id(Some("session-main".to_string()));

        // Every event is a sub-agent — the root never appears as an
        // event's agent_id at all.
        for (agent_id, label) in [
            ("agent-1", "Explore parser"),
            ("agent-2", "Explore symbols"),
            ("agent-3", "Explore tracking"),
        ] {
            let mut e = tool_call("Read", "/test/project/mock/f.rs", ReadDepth::FullBody);
            e.agent_id = agent_id.into();
            e.label = label.into();
            app.process_agent_event(e);
        }

        // The root is the seeded session id, never a sub-agent.
        assert_eq!(app.agent_tree.root_id, Some("session-main".to_string()));
        assert_ne!(app.agent_tree.root_id, Some("agent-1".to_string()));
        assert_ne!(app.agent_tree.root_id, Some("agent-2".to_string()));
        assert_ne!(app.agent_tree.root_id, Some("agent-3".to_string()));

        // Every sub-agent parents directly to the true root — none rootless,
        // none parented to another sub-agent.
        for agent_id in ["agent-1", "agent-2", "agent-3"] {
            let node = &app.agent_tree.agents[agent_id];
            assert_eq!(node.parent_id, Some("session-main".to_string()));
        }

        // The sibling lookup `open_alignment_overlay` depends on resolves
        // all three as children of the root.
        let root_id = app.agent_tree.root_id.clone().unwrap();
        let mut children: Vec<String> = app
            .agent_tree
            .children_of(&root_id)
            .into_iter()
            .map(|a| a.id.clone())
            .collect();
        children.sort();
        assert_eq!(children, vec!["agent-1", "agent-2", "agent-3"]);
    }

    /// End-to-end regression test for the alignment popup path itself: given
    /// a correctly-rooted orchestrator-only tree, filtering to any one
    /// sub-agent and opening the alignment overlay must populate
    /// `agent_alignment` with the other siblings *and* the orchestrator
    /// itself — not silently no-op the way it did when the first sub-agent
    /// was mistaken for the root, and not exclude the orchestrator from the
    /// comparison group either.
    #[test]
    fn open_alignment_overlay_resolves_siblings_in_orchestrator_only_session() {
        let mut app = test_app(vec![file("mock/f.rs", vec![sym("mock/f.rs::a", "a")])]);
        app.set_session_id(Some("session-main".to_string()));

        for agent_id in ["agent-1", "agent-2", "agent-3"] {
            let mut e = tool_call("Read", "/test/project/mock/f.rs", ReadDepth::FullBody);
            e.agent_id = agent_id.into();
            app.process_agent_event(e);
        }

        // Filter to the *first* agent seen — exactly what `a` (cycle_agent_filter)
        // does on first press, and exactly the agent that used to be
        // mistaken for the root.
        app.agent_filter = Some("agent-1".to_string());
        app.open_alignment_overlay();

        assert!(app.show_alignment_overlay);
        // Group = {session-main, agent-1, agent-2, agent-3} -> 4 choose 2 = 6
        // pairs, agent-1 involved in 3 of them (one per other group member).
        assert_eq!(app.agent_alignment.len(), 6);
        let involves_agent_1 = app
            .agent_alignment
            .iter()
            .filter(|p| p.agent_a == "agent-1" || p.agent_b == "agent-1")
            .count();
        assert_eq!(involves_agent_1, 3);
        let involves_root = app
            .agent_alignment
            .iter()
            .filter(|p| p.agent_a == "session-main" || p.agent_b == "session-main")
            .count();
        assert_eq!(involves_root, 3);
    }

    /// Regression test for a lone sub-agent: an orchestrator that spawned
    /// exactly one sub-agent has no *siblings* to compare (the sub-agent's
    /// sibling group is itself alone), but there is still a meaningful
    /// comparison to make — the orchestrator against its one child. Before
    /// the parent was folded into the comparison group, this silently
    /// no-op'd because `sibling_ids.len() < 2`.
    #[test]
    fn open_alignment_overlay_compares_root_against_lone_sub_agent() {
        let mut app = test_app(vec![file("mock/f.rs", vec![sym("mock/f.rs::a", "a")])]);
        app.set_session_id(Some("session-main".to_string()));

        let mut e = tool_call("Read", "/test/project/mock/f.rs", ReadDepth::FullBody);
        e.agent_id = "agent-1".into();
        app.process_agent_event(e);

        // Selecting the lone sub-agent must compare it against its parent.
        app.agent_filter = Some("agent-1".to_string());
        app.open_alignment_overlay();
        assert!(app.show_alignment_overlay);
        assert_eq!(app.agent_alignment.len(), 1);
        let pair = &app.agent_alignment[0];
        assert!(
            (pair.agent_a == "session-main" && pair.agent_b == "agent-1")
                || (pair.agent_a == "agent-1" && pair.agent_b == "session-main")
        );

        // Selecting the root/orchestrator itself must produce the same
        // comparison against its one child.
        app.show_alignment_overlay = false;
        app.agent_alignment.clear();
        app.agent_filter = Some("session-main".to_string());
        app.open_alignment_overlay();
        assert!(app.show_alignment_overlay);
        assert_eq!(app.agent_alignment.len(), 1);
    }

    /// Regression test for the identity-vs-prefix bug: real ingestion never
    /// produces `agent_id`s prefixed `"agent-"` — that prefix only exists in
    /// sub-agent JSONL *filenames* (`agent-<hash>.jsonl`); the `agentId`
    /// field `parse_jsonl_line` actually reads (and prefers over the
    /// filename/session fallback) has no prefix at all, e.g.
    /// `"a63c858997b4e6124"`. A `starts_with("agent-")` check silently never
    /// matches real events, so this uses realistic unprefixed ids rather
    /// than the prefixed synthetic ids other tests use — repeating that
    /// mistake is exactly how the bug shipped undetected.
    #[test]
    fn unprefixed_real_world_agent_ids_parent_to_root_by_identity() {
        let mut app = test_app(vec![file("mock/f.rs", vec![sym("mock/f.rs::a", "a")])]);
        let root_id = "0eb2bbd0-fcd7-46a1-84e7-990a6f4734b4".to_string();
        app.set_session_id(Some(root_id.clone()));

        for agent_id in ["a63c858997b4e6124", "b71fa9231c8de55a0"] {
            let mut e = tool_call("Read", "/test/project/mock/f.rs", ReadDepth::FullBody);
            e.agent_id = agent_id.into();
            app.process_agent_event(e);
        }

        for agent_id in ["a63c858997b4e6124", "b71fa9231c8de55a0"] {
            let node = &app.agent_tree.agents[agent_id];
            assert_eq!(node.parent_id, Some(root_id.clone()));
        }
        assert_eq!(app.agent_tree.root_id, Some(root_id.clone()));

        // Sibling lookup resolves correctly for the alignment popup, and the
        // root/orchestrator is included in the comparison group alongside
        // the two sub-agents: group = {root, a63c..., b71fa...} -> 3 choose
        // 2 = 3 pairs.
        app.agent_filter = Some("a63c858997b4e6124".to_string());
        app.open_alignment_overlay();

        assert!(app.show_alignment_overlay);
        assert_eq!(app.agent_alignment.len(), 3);
        let has_pair = |a: &str, b: &str| {
            app.agent_alignment
                .iter()
                .any(|p| (p.agent_a == a && p.agent_b == b) || (p.agent_a == b && p.agent_b == a))
        };
        assert!(has_pair("a63c858997b4e6124", "b71fa9231c8de55a0"));
        assert!(has_pair(&root_id, "a63c858997b4e6124"));
        assert!(has_pair(&root_id, "b71fa9231c8de55a0"));
    }

    #[test]
    fn flattened_agents_dfs_order() {
        let mut app = test_app(vec![file("mock/f.rs", vec![sym("mock/f.rs::a", "a")])]);

        // Register root + two subagents.
        let mut e1 = tool_call("Read", "/test/project/mock/f.rs", ReadDepth::FullBody);
        e1.agent_id = "session-main".into();
        app.process_agent_event(e1);

        let mut e2 = tool_call("Read", "/test/project/mock/f.rs", ReadDepth::Overview);
        e2.agent_id = "agent-aaa".into();
        app.process_agent_event(e2);

        let mut e3 = tool_call("Read", "/test/project/mock/f.rs", ReadDepth::Overview);
        e3.agent_id = "agent-bbb".into();
        app.process_agent_event(e3);

        let flat = app.flattened_agents();
        assert_eq!(flat.len(), 3);
        // Root at depth 0.
        assert_eq!(flat[0].0, "session-main");
        assert_eq!(flat[0].1, 0);
        // Subagents at depth 1.
        assert!(flat.iter().any(|(id, d)| id == "agent-aaa" && *d == 1));
        assert!(flat.iter().any(|(id, d)| id == "agent-bbb" && *d == 1));
    }

    #[test]
    fn flattened_agents_empty_when_no_agents() {
        let app = test_app(vec![]);
        assert!(app.flattened_agents().is_empty());
    }

    #[test]
    fn process_compaction_snapshots_files() {
        // Two files; touch one before triggering the compaction so the snapshot
        // captures only that file.
        let mut app = test_app(vec![
            file("mock/a.rs", vec![sym("mock/a.rs::a1", "a1")]),
            file("mock/b.rs", vec![sym("mock/b.rs::b1", "b1")]),
        ]);

        let event = tool_call("Read", "/test/project/mock/a.rs", ReadDepth::FullBody);
        app.process_agent_event(event);

        app.process_compaction(
            "first compaction".into(),
            "2026-05-11T14:23:00Z".into(),
            "agent-1".into(),
            None,
        );

        assert_eq!(app.compaction_history.len(), 1);
        let snapshot = &app.compaction_history[0];
        assert_eq!(snapshot.sequence, 1);
        assert_eq!(snapshot.summary, "first compaction");
        assert_eq!(snapshot.timestamp, "2026-05-11T14:23:00Z");
        assert_eq!(snapshot.ledger_before.tool_call_count, 1);
        // mock/a.rs was touched → present; mock/b.rs untouched → absent.
        assert!(snapshot
            .ledger_before
            .files_accessed
            .contains(&std::path::PathBuf::from("mock/a.rs")));
        assert!(!snapshot
            .ledger_before
            .files_accessed
            .contains(&std::path::PathBuf::from("mock/b.rs")));
        assert_eq!(snapshot.ledger_before.symbols_seen, 1);
    }

    #[test]
    fn process_compaction_clears_live_ledger() {
        // Two files; read one. After compaction, the live ledger should be
        // empty (symbol back to Unseen) but the snapshot captures the prior state.
        let mut app = test_app(vec![
            file("mock/a.rs", vec![sym("mock/a.rs::a1", "a1")]),
        ]);
        let event = tool_call("Read", "/test/project/mock/a.rs", ReadDepth::FullBody);
        app.process_agent_event(event);

        assert_eq!(app.ledger.depth_of("mock/a.rs::a1"), ReadDepth::FullBody);
        assert_eq!(app.compaction_call_count, 1);

        app.process_compaction(
            "summary".into(),
            "2026-05-11T14:23:00Z".into(),
            "agent-1".into(),
            None,
        );

        // Compaction history records the pre-compaction state.
        assert_eq!(app.compaction_history.len(), 1);
        assert_eq!(app.compaction_history[0].ledger_before.symbols_seen, 1);
        // The read survives with its depth intact — it is a fact about what
        // the agent looked at — but is demoted to Restored.
        assert_eq!(app.ledger.depth_of("mock/a.rs::a1"), ReadDepth::FullBody);
        assert!(app.ledger.is_restored("mock/a.rs::a1"));
        assert_eq!(app.ledger.total_seen(), 1);
        assert_eq!(app.ledger.total_restored(), 1);
        assert_eq!(app.compaction_call_count, 0);
    }

    #[test]
    fn post_compaction_reads_are_live_alongside_restored_ones() {
        // Compaction demotes existing entries; subsequent tool calls add Live
        // ones beside them, without touching compaction_history.
        let mut app = test_app(vec![
            file("mock/a.rs", vec![sym("mock/a.rs::a1", "a1")]),
            file("mock/b.rs", vec![sym("mock/b.rs::b1", "b1")]),
        ]);
        app.process_agent_event(tool_call("Read", "/test/project/mock/a.rs", ReadDepth::FullBody));
        app.process_compaction(
            "first".into(),
            "2026-05-11T14:23:00Z".into(),
            "agent-1".into(),
            None,
        );

        // Read a different file after compaction.
        app.process_agent_event(tool_call("Read", "/test/project/mock/b.rs", ReadDepth::FullBody));

        assert_eq!(app.compaction_history.len(), 1, "compaction history retained");
        assert_eq!(app.ledger.depth_of("mock/a.rs::a1"), ReadDepth::FullBody,
            "pre-compaction read survives");
        assert!(app.ledger.is_restored("mock/a.rs::a1"), "but is marked restored");
        assert_eq!(app.ledger.depth_of("mock/b.rs::b1"), ReadDepth::FullBody,
            "post-compaction read should be reflected");
        assert!(!app.ledger.is_restored("mock/b.rs::b1"), "and is live");
        assert_eq!(app.ledger.total_restored(), 1, "only the pre-compaction read is restored");
        assert_eq!(app.compaction_call_count, 1, "counter restarted from zero after compaction");
    }

    /// Re-reading a symbol after a compaction promotes it back to Live — the
    /// model demonstrably has it in context again.
    #[test]
    fn re_read_after_compaction_promotes_back_to_live() {
        let mut app = test_app(vec![file("mock/a.rs", vec![sym("mock/a.rs::a1", "a1")])]);
        app.process_agent_event(tool_call("Read", "/test/project/mock/a.rs", ReadDepth::FullBody));
        app.process_compaction(
            "s".into(),
            "2026-05-11T14:23:00Z".into(),
            "agent-1".into(),
            None,
        );
        assert!(app.ledger.is_restored("mock/a.rs::a1"));

        app.process_agent_event(tool_call("Read", "/test/project/mock/a.rs", ReadDepth::FullBody));

        assert!(!app.ledger.is_restored("mock/a.rs::a1"));
        assert_eq!(app.ledger.total_restored(), 0);
    }

    #[test]
    fn compaction_history_clears_on_reset_session() {
        let mut app = test_app(vec![file("mock/a.rs", vec![sym("mock/a.rs::a", "a")])]);
        let event = tool_call("Read", "/test/project/mock/a.rs", ReadDepth::FullBody);
        app.process_agent_event(event);
        app.process_compaction(
            "summary".into(),
            "2026-05-11T14:23:00Z".into(),
            "agent-1".into(),
            None,
        );
        assert_eq!(app.compaction_history.len(), 1);

        app.reset_session();

        assert!(app.compaction_history.is_empty());
        assert_eq!(app.compaction_call_count, 0);
    }

    #[test]
    fn reset_session_clears_ledger_and_agents() {
        use crate::ingest::AgentToolCall;
        use crate::tracking::ReadDepth;

        let mut app = test_app(vec![]);

        // Populate session state via a fake event.
        let event = AgentToolCall {
            agent_id: "agent-abc".into(),
            tool_name: "Read".into(),
            file_path: None,
            read_depth: ReadDepth::FullBody,
            description: "Read something".to_string(),
            timestamp_str: "2026-01-01T00:00:00Z".to_string(),
            target_symbol: None,
            target_lines: None,
        target_selectors: Vec::new(),
            label: "agent-abc".into(),
            tool_use_id: None,
            effect: crate::ingest::Effect::Read,
            summary: None,
            result_depth: None,
        };
        app.process_agent_event(event);

        assert!(!app.agents_seen.is_empty());
        assert!(!app.activity.is_empty());

        // reset_session should clear all live state.
        let original_files_count = app.project_tree.files.len();
        app.reset_session();

        assert!(app.ledger.entries.is_empty());
        assert!(app.activity.is_empty());
        assert!(app.agents_seen.is_empty());
        assert!(app.agent_filter.is_none());
        assert_eq!(app.agent_selection_index, 0);
        // Project tree is preserved.
        assert_eq!(app.project_tree.files.len(), original_files_count);
    }
}

/// Writes in the app (spec phase 2): journaled on arrival, shown in the
/// activity feed, never credited as reads (D9).
#[cfg(test)]
mod write_tests {
    use super::*;
    use crate::helpers::*;
    use crate::writes::WriteRecord;

    fn record(op: &str) -> WriteRecord {
        WriteRecord {
            op: op.into(),
            av: crate::writes::ATTRIBUTION_VERSION,
            a: "agent-1".into(),
            t: "2026-09-26T10:00:01Z".into(),
            tool: "Edit".into(),
            file: "src/a.rs".into(),
            ..Default::default()
        }
    }

    /// `src/a.rs` with `S { a, b }`, expanded, in session `sess`.
    fn marked_app(root: &Path) -> App {
        let tree = project(vec![file("src/a.rs", vec![sym_with_children("src/a.rs::S", "S", vec![sym("src/a.rs::S/a", "a"), sym("src/a.rs::S/b", "b")])])]);
        let mut app = App::new(tree, root.to_path_buf());
        app.set_session_id(Some("sess".into()));
        app.set_expanded("src/a.rs", RowKind::File, true);
        app.set_expanded("src/a.rs::S", RowKind::Symbol, true);
        app
    }

    fn mark(app: &App, id: &str) -> Option<crate::write_index::WriteMark> {
        app.tree_rows.iter().find(|r| r.symbol_id == id).expect("row").write.clone()
    }

    fn wrote_a(app: &App) -> WriteRecord {
        let a = &app.project_tree.files[0].symbols[0].children[0];
        WriteRecord {
            level: crate::writes::Level::Symbol,
            syms: vec![(a.id.clone(), crate::journal::encode_hash(&a.content_hash))],
            ..record("toolu_1")
        }
    }

    /// A written symbol, its parent and its file are marked current, and
    /// the symbol turns changed when a re-parse brings a new body.
    #[test]
    fn a_recorded_write_marks_the_tree_until_the_symbol_changes() {
        use crate::writes::Status;
        let dir = tempfile::tempdir().unwrap();
        let mut app = marked_app(dir.path());
        app.record_write("sess", wrote_a(&app));
        let current = Some(crate::write_index::WriteMark { status: Status::Current, latest: "toolu_1".into(), count: 1 });
        assert_eq!(mark(&app, "src/a.rs::S/a"), current);
        assert_eq!(mark(&app, "src/a.rs::S"), current, "a parent rolls up");
        assert_eq!(mark(&app, "src/a.rs::S/b"), None);
        assert_eq!(mark(&app, "src/a.rs"), current);

        // What the file watcher does on a change.
        app.project_tree.files[0].symbols[0].children[0].content_hash = crate::symbols::merkle::content_hash("a2");
        app.rebuild_tree_rows();
        assert_eq!(mark(&app, "src/a.rs::S/a").map(|m| m.status), Some(Status::Changed));
    }

    /// A file-level write marks the file row only.
    #[test]
    fn a_file_level_write_marks_only_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = marked_app(dir.path());
        app.record_write("sess", record("toolu_1"));
        assert_eq!(mark(&app, "src/a.rs").map(|m| m.count), Some(1));
        assert_eq!(mark(&app, "src/a.rs::S"), None);
    }

    /// Only the current session's writes are marked; the agent filter
    /// narrows them to that agent's.
    #[test]
    fn only_this_sessions_writes_are_marked() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = marked_app(dir.path());
        app.record_write("other", wrote_a(&app));
        assert_eq!(mark(&app, "src/a.rs"), None);

        app.record_write("sess", wrote_a(&app));
        app.agent_filter = Some("someone-else".into());
        app.rebuild_tree_rows();
        assert_eq!(mark(&app, "src/a.rs::S/a"), None);

        app.switch_session(Some("next".into()));
        app.agent_filter = None;
        app.rebuild_tree_rows();
        assert_eq!(mark(&app, "src/a.rs"), None, "a new session starts unmarked");
    }

    /// Attaching the journal brings back writes an earlier run recorded.
    #[test]
    fn attaching_the_journal_restores_its_writes() {
        let dir = tempfile::tempdir().unwrap();
        let mut earlier = marked_app(dir.path());
        earlier.enable_journal("tree-sitter", std::time::Duration::ZERO);
        earlier.record_write("sess", wrote_a(&earlier));
        drop(earlier);

        let mut app = marked_app(dir.path());
        app.set_journal_settings(Some(JournalSettings { backend: "tree-sitter", interval: std::time::Duration::ZERO }));
        app.attach_journal();
        assert_eq!(mark(&app, "src/a.rs::S/a").map(|m| m.status), Some(crate::writes::Status::Current));
    }

    #[test]
    fn a_recorded_write_is_journaled() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = App::new(project(vec![]), dir.path().to_path_buf());
        app.set_session_id(Some("sess".into()));
        app.enable_journal("tree-sitter", std::time::Duration::ZERO);

        app.record_write("sess", record("toolu_1"));
        app.record_write("sess", record("toolu_1"));

        let contents = crate::journal::read_journal_session(&crate::journal::journal_dir(dir.path()), "sess");
        assert_eq!(contents.writes.len(), 1, "journaled once");
        assert!(contents.writes.contains_key("toolu_1"));
    }

    #[test]
    fn without_a_journal_a_write_is_simply_not_persisted() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = App::new(project(vec![]), dir.path().to_path_buf());
        app.record_write("sess", record("toolu_1"));
        assert!(!crate::journal::journal_dir(dir.path()).exists());
    }

    /// A write call has no read depth, which used to keep it out of the feed.
    #[test]
    fn a_write_call_is_shown_in_the_activity_feed_without_read_credit() {
        let mut app = App::new(project(vec![file("src/a.rs", vec![sym("src/a.rs::f", "f")])]), "/p".into());
        let mut call = tool_call("Edit", "src/a.rs", ReadDepth::Unseen);
        call.effect = crate::ingest::Effect::Write;
        app.process_agent_event(call);
        assert_eq!(app.activity.len(), 1);
        assert_eq!(app.ledger.depth_of("src/a.rs::f"), ReadDepth::Unseen, "no read credit");
    }

    fn journaled(dir: &Path, session: &str) -> crate::journal::JournalContents {
        crate::journal::read_journal_session(&crate::journal::journal_dir(dir), session)
    }

    /// An app on session `a` with journaling on, as startup leaves it.
    fn journaling_app(dir: &Path, interval: std::time::Duration) -> App {
        let mut app = App::new(
            project(vec![file("src/a.rs", vec![sym("src/a.rs::f", "f")])]),
            dir.to_path_buf(),
        );
        app.set_session_id(Some("a".into()));
        app.set_journal_settings(Some(JournalSettings { backend: "tree-sitter", interval }));
        app.attach_journal();
        app
    }

    fn switch(app: &mut App, session: &str) {
        app.switch_session(Some(session.into()));
        app.attach_journal();
    }

    /// The journal used to stay on the first session's file across a switch,
    /// so the new session's reads and writes landed in the old one.
    #[test]
    fn switching_sessions_journals_the_new_session_into_its_own_file() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = journaling_app(dir.path(), std::time::Duration::ZERO);
        switch(&mut app, "b");
        app.process_agent_event(tool_call("Read", "src/a.rs", ReadDepth::FullBody));
        app.sync_journal();
        app.record_write("b", record("toolu_b"));

        let (a, b) = (journaled(dir.path(), "a"), journaled(dir.path(), "b"));
        assert!(a.reads.is_empty() && a.writes.is_empty(), "nothing leaked into a");
        assert!(b.reads.contains_key("src/a.rs::f"));
        assert!(b.writes.contains_key("toolu_b"));
    }

    /// The reset on a switch discards the ledger; what it held must reach
    /// the outgoing journal first, even inside the flush interval.
    #[test]
    fn switching_flushes_the_outgoing_sessions_unsynced_reads() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = journaling_app(dir.path(), std::time::Duration::from_secs(3600));
        app.process_agent_event(tool_call("Read", "src/a.rs", ReadDepth::FullBody));
        switch(&mut app, "b");
        assert!(journaled(dir.path(), "a").reads.contains_key("src/a.rs::f"));
    }

    /// A write still in the attribution worker at a switch comes back after
    /// it, and belongs to the session it happened in.
    #[test]
    fn a_late_write_lands_in_the_session_it_happened_in() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = journaling_app(dir.path(), std::time::Duration::ZERO);
        switch(&mut app, "b");
        app.record_write("a", record("toolu_a"));
        assert!(journaled(dir.path(), "a").writes.contains_key("toolu_a"));
        assert!(journaled(dir.path(), "b").writes.is_empty());
    }

    #[test]
    fn a_write_two_switches_late_is_dropped() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = journaling_app(dir.path(), std::time::Duration::ZERO);
        switch(&mut app, "b");
        switch(&mut app, "c");
        app.record_write("a", record("toolu_a"));
        for session in ["a", "b", "c"] {
            assert!(journaled(dir.path(), session).writes.is_empty(), "{session}");
        }
    }

    /// Routing a late write must never open a journal of its own.
    #[test]
    fn with_journaling_off_a_late_write_creates_no_file() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = App::new(project(vec![]), dir.path().to_path_buf());
        app.set_session_id(Some("a".into()));
        app.switch_session(Some("b".into()));
        app.attach_journal();
        app.record_write("a", record("toolu_a"));
        assert!(!crate::journal::journal_dir(dir.path()).exists());
    }

    #[test]
    fn queued_writes_carry_their_session_and_need_one() {
        let event = || crate::ingest::WriteEvent {
            op: "toolu_1".into(),
            agent_id: "main".into(),
            tool_name: "Edit".into(),
            path: "src/a.rs".into(),
            timestamp: String::new(),
            source: crate::ingest::WriteSource::Opaque,
        };
        let mut app = App::new(project(vec![]), "/p".into());
        app.queue_write(event());
        assert!(app.take_pending_writes().is_empty(), "no session, no journal");

        app.set_session_id(Some("a".into()));
        app.queue_write(event());
        let queued = app.take_pending_writes();
        assert_eq!(queued.len(), 1);
        assert_eq!(&*queued[0].0, "a");
    }

    fn write_call(mut call: AgentToolCall) -> AgentToolCall {
        call.effect = crate::ingest::Effect::Write;
        call
    }

    /// Read `f`, then let it drift on disk: the entry is stale at its
    /// original hash, which is what every case below must preserve.
    fn app_with_a_stale_read() -> (App, [u8; 32]) {
        let mut app = App::new(project(vec![file("src/a.rs", vec![sym("src/a.rs::f", "f")])]), "/p".into());
        app.process_agent_event(tool_call("Read", "src/a.rs", ReadDepth::FullBody));
        let read_at = app.ledger.entries["src/a.rs::f"].content_hash_at_read;
        app.ledger.mark_stale_if_changed("src/a.rs::f", [7u8; 32]);
        assert!(app.ledger.entries["src/a.rs::f"].stale);
        (app, read_at)
    }

    fn assert_still_stale(app: &App, read_at: [u8; 32], case: &str) {
        let entry = &app.ledger.entries["src/a.rs::f"];
        assert!(entry.stale, "{case}: stale read refreshed");
        assert_eq!(entry.content_hash_at_read, read_at, "{case}: hash replaced");
        assert_eq!(entry.provenance, crate::tracking::Provenance::Live, "{case}");
        assert_eq!(entry.depth, ReadDepth::FullBody, "{case}");
    }

    /// D9: a write refreshes nothing. It used to reach `ledger.record` at
    /// Unseen, which cleared `stale` and adopted the post-edit hash — so an
    /// edit made a stale read look current. One case per marking branch.
    #[test]
    fn a_write_call_leaves_a_stale_read_stale() {
        let cases = [
            ("whole file", write_call(tool_call("Edit", "src/a.rs", ReadDepth::Unseen))),
            ("target symbol", write_call(tool_call_targeted("replace_symbol_body", "src/a.rs", ReadDepth::Unseen, "f"))),
            ("selectors", {
                let mut c = write_call(tool_call("Bash", "src/a.rs", ReadDepth::Unseen));
                c.target_selectors = vec![("src/a.rs::f".into(), ReadDepth::Unseen)];
                c
            }),
        ];
        for (case, call) in cases {
            let (mut app, read_at) = app_with_a_stale_read();
            app.process_agent_event(call);
            assert_still_stale(&app, read_at, case);
        }
    }

    /// The same hole through a *read* stanza whose depth resolves to Unseen.
    #[test]
    fn an_unseen_read_leaves_a_stale_read_stale() {
        let (mut app, read_at) = app_with_a_stale_read();
        app.process_agent_event(tool_call("Glob", "src/a.rs", ReadDepth::Unseen));
        assert_still_stale(&app, read_at, "unseen read");
    }

    #[test]
    fn a_write_call_on_an_unread_file_creates_no_entries() {
        let mut app = App::new(project(vec![file("src/a.rs", vec![sym("src/a.rs::f", "f")])]), "/p".into());
        app.process_agent_event(write_call(tool_call("Edit", "src/a.rs", ReadDepth::Unseen)));
        assert!(app.ledger.entries.is_empty());
    }

    /// End to end through the journal: after an edit to a drifted symbol, the
    /// journal must still hold only the read's original hash. Before the fix
    /// the edit re-established the read and `sync` journaled the new hash as
    /// a fresh FullBody read.
    #[test]
    fn a_write_does_not_journal_a_fresh_read() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = App::new(
            project(vec![file("src/a.rs", vec![sym("src/a.rs::f", "f")])]),
            dir.path().to_path_buf(),
        );
        app.set_session_id(Some("sess".into()));
        app.enable_journal("tree-sitter", std::time::Duration::ZERO);
        app.process_agent_event(tool_call("Read", "src/a.rs", ReadDepth::FullBody));
        app.sync_journal();
        let read_at = app.ledger.entries["src/a.rs::f"].content_hash_at_read;

        let drifted = [7u8; 32];
        app.project_tree.files[0].symbols[0].content_hash = drifted;
        app.ledger.mark_stale_if_changed("src/a.rs::f", drifted);
        app.process_agent_event(write_call(tool_call("Edit", "src/a.rs", ReadDepth::Unseen)));
        app.sync_journal();

        let contents = crate::journal::read_journal_session(&crate::journal::journal_dir(dir.path()), "sess");
        assert_eq!(contents.reads["src/a.rs::f"], (read_at, ReadDepth::FullBody));
    }
}

/// The trace view's keys and mouse (UI-7): modal, following delegations
/// into their agents and spans out to the tree.
#[cfg(test)]
mod trace_view_tests {
    use super::*;
    use crate::helpers::*;
    use crate::ingest::ToolFinished;
    use crate::trace::view::{Item, Layout};

    fn key(app: &mut App, code: KeyCode) {
        app.handle_key(KeyEvent::new(code, KeyModifiers::NONE));
    }

    /// `src/a.rs` with `S { a }`; main reads `S/a`, then delegates to `ax1`,
    /// whose edit fails.
    fn app() -> App {
        let tree = project(vec![file("src/a.rs", vec![sym_with_children("src/a.rs::S", "S", vec![sym("src/a.rs::S/a", "a")])])]);
        let mut app = App::new(tree, PathBuf::from("/test/project"));
        app.set_session_id(Some("sess".into()));
        let root = app.project_root.clone();
        let mut calls = vec![
            ("sess", "r1", "Read", "2026-09-27T10:00:00.000Z", "2026-09-27T10:00:01.000Z", None, false),
            ("sess", "d1", "Agent", "2026-09-27T10:00:02.000Z", "2026-09-27T10:00:02.100Z", Some("ax1"), false),
            ("ax1", "x1", "Edit", "2026-09-27T10:00:03.000Z", "2026-09-27T10:00:04.000Z", None, true),
        ];
        for (agent, id, tool, start, end, child, error) in calls.drain(..) {
            let mut c = tool_call_targeted(tool, "/test/project/src/a.rs", ReadDepth::FullBody, "S/a");
            c.agent_id = Arc::from(agent);
            c.tool_use_id = Some(Arc::from(id));
            c.timestamp_str = start.into();
            app.trace.start(&c, &root);
            app.trace.finish(&ToolFinished {
                id: Arc::from(id),
                agent_id: Arc::from(agent),
                timestamp: end.into(),
                error,
                message: None, child_agent: child.map(Arc::from),
                shown: Vec::new(),
            });
        }
        // Read after its calls, as a tailer poll can deliver it; its time
        // still makes it their parent.
        app.process_prompt(&crate::ingest::Prompt {
            agent_id: Arc::from("sess"),
            timestamp: "2026-09-27T09:59:59.000Z".into(),
            text: "review the code".into(),
        });
        key(&mut app, KeyCode::Char('t'));
        key(&mut app, KeyCode::Enter);
        app
    }

    /// `t` opens on the list of traces, one per prompt; `Enter` opens one,
    /// `Esc` goes back to the list and then to the tree.
    #[test]
    fn the_trace_view_opens_on_a_list_of_prompts() {
        let mut app = app();
        key(&mut app, KeyCode::Esc);
        assert!(app.trace_view.open && app.trace_view.focus.is_none(), "back on the list");
        let traces = app.trace_list();
        assert_eq!(traces.len(), 1);
        assert_eq!((traces[0].root, traces[0].calls, traces[0].failed, traces[0].agents), (3, 3, 1, 1));
        key(&mut app, KeyCode::Enter);
        assert_eq!(app.trace_view.focus, Some(3));
        assert_eq!(app.trace_rows().len(), 4, "the prompt and its three calls");
        key(&mut app, KeyCode::Esc);
        key(&mut app, KeyCode::Esc);
        assert!(!app.trace_view.open);
    }

    #[test]
    fn t_opens_the_trace_and_its_keys_are_modal() {
        let mut app = app();
        assert!(app.trace_view.open);
        let filter = app.agent_filter.clone();
        key(&mut app, KeyCode::Char('w'));
        assert!(app.trace_view.viewport.is_some(), "w zooms");
        key(&mut app, KeyCode::Char('a'));
        assert_eq!(app.agent_filter, filter, "a pans here, not the agent filter");
        key(&mut app, KeyCode::Char('0'));
        assert!(app.trace_view.viewport.is_none(), "0 fits");
        key(&mut app, KeyCode::Char('v'));
        assert_eq!(app.trace_view.layout, Layout::Tracks);
        key(&mut app, KeyCode::Char('t'));
        assert!(!app.trace_view.open);
    }

    #[test]
    fn enter_follows_a_delegation_into_its_agent() {
        let mut app = app();
        key(&mut app, KeyCode::Char('j'));
        key(&mut app, KeyCode::Char('j'));
        key(&mut app, KeyCode::Char('j'));
        assert_eq!(app.trace_view.selected, Some(Item::Span(1)));
        key(&mut app, KeyCode::Char('h'));
        assert!(app.trace_view.collapsed_spans.contains(&1), "h folds");
        key(&mut app, KeyCode::Enter);
        assert_eq!(app.trace_view.selected, Some(Item::Span(2)), "unfolded, into ax1's edit");

        key(&mut app, KeyCode::Char('v'));
        app.trace_view.selected = Some(Item::Span(1));
        key(&mut app, KeyCode::Enter);
        let (tracks, rows) = app.trace_tracks();
        assert_eq!(&*tracks[rows[app.trace_view.track_row].track].agent, "ax1");
    }

    #[test]
    fn enter_on_a_read_shows_its_symbol_in_the_tree() {
        let mut app = app();
        app.trace_view.selected = Some(Item::Span(0));
        key(&mut app, KeyCode::Enter);
        assert!(!app.trace_view.open);
        assert_eq!(app.tree_rows[app.selected_index].symbol_id, "src/a.rs::S/a", "expanded down to it");
    }

    #[test]
    fn e_finds_the_failure_and_brackets_filter_by_agent() {
        let mut app = app();
        key(&mut app, KeyCode::Char('e'));
        assert_eq!(app.trace_view.selected, Some(Item::Span(2)));
        key(&mut app, KeyCode::Char(']'));
        assert!(app.agent_filter.is_some(), "] cycles the agent filter");
        key(&mut app, KeyCode::Char('['));
        assert!(app.agent_filter.is_none(), "[ cycles it back");
        assert!(app.trace_view.open);
    }

    /// One focus model in every view: Tab and Shift+Tab move between the
    /// panels, `[` `]` move the agent filter, Esc brings focus back left.
    #[test]
    fn tab_moves_focus_and_brackets_move_the_filter_in_every_view() {
        let mut app = app();
        for view in ["timeline", "list", "tree"] {
            match view {
                "list" => key(&mut app, KeyCode::Esc),
                "tree" => key(&mut app, KeyCode::Char('t')),
                _ => {}
            }
            assert_eq!(app.focus, FocusPanel::Left, "{view}");
            key(&mut app, KeyCode::Tab);
            assert_eq!(app.focus, FocusPanel::Right, "{view}: tab");
            key(&mut app, KeyCode::Tab);
            assert_eq!(app.focus, FocusPanel::Left, "{view}: no feed shown, so back left");
            key(&mut app, KeyCode::BackTab);
            assert_eq!(app.focus, FocusPanel::Right, "{view}: shift+tab");
            key(&mut app, KeyCode::Esc);
            assert_eq!(app.focus, FocusPanel::Left, "{view}: esc");
            key(&mut app, KeyCode::Char(']'));
            assert!(app.agent_filter.is_some(), "{view}: ]");
            key(&mut app, KeyCode::Char('['));
            assert!(app.agent_filter.is_none(), "{view}: [");
        }
        assert!(!app.trace_view.open, "ended on the tree");
    }

    /// With focus on the right panel, j/k move its rows and leave the
    /// timeline's selection alone.
    #[test]
    fn keys_go_to_the_focused_panel() {
        let mut app = app();
        key(&mut app, KeyCode::Char('j'));
        let selected = app.trace_view.selected;
        key(&mut app, KeyCode::Tab);
        key(&mut app, KeyCode::Char('j'));
        key(&mut app, KeyCode::Char('j'));
        assert_eq!(app.panel_index, 2);
        assert_eq!(app.trace_view.selected, selected);
        key(&mut app, KeyCode::Esc);
        key(&mut app, KeyCode::Char('j'));
        assert_ne!(app.trace_view.selected, selected);
        assert_eq!(app.panel_index, 0, "a new selection starts the panel over");
    }

    #[test]
    fn a_click_selects_and_the_wheel_zooms_at_the_pointer() {
        let mut app = app();
        app.trace_geometry.set(Some(TraceGeometry { bars_x: 10, bars_width: 40, rows_y: 5, rows: 10, first_row: 0 }));
        let mouse = |kind, column, row| MouseEvent { kind, column, row, modifiers: KeyModifiers::NONE };
        app.handle_mouse(mouse(MouseEventKind::Down(crossterm::event::MouseButton::Left), 12, 7));
        assert_eq!(app.trace_view.selected, Some(Item::Span(1)), "the third row: the prompt, the read, the delegation");
        let before = app.trace_view.viewport(&app.trace);
        app.handle_mouse(mouse(MouseEventKind::ScrollUp, 10, 7));
        let after = app.trace_view.viewport.expect("zoomed");
        assert_eq!(after.start, before.start, "zoomed about the left edge, where the pointer is");
        assert!(after.width() < before.width());
    }
}
