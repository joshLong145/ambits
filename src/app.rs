use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind};

use crate::coverage::count_symbols;
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
    pub is_file: bool,        // true for file headers
    pub is_expanded: bool,
    pub has_children: bool,
    pub line_range: String,
    pub token_count: usize,
    pub read_depth: ReadDepth,
    pub coverage_status: Option<FileCoverageStatus>,
    pub file_coverage_seen: usize,
    pub file_coverage_total: usize,
}

/// Which panel is focused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusPanel {
    Tree,
    Stats,
    Activity,
}

pub struct App {
    pub project_tree: ProjectTree,
    pub project_root: PathBuf,
    pub ledger: ContextLedger,
    pub should_quit: bool,

    // Tree view state.
    pub tree_rows: Vec<TreeRow>,
    pub selected_index: usize,
    pub collapsed: std::collections::HashSet<String>,

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

    // Optional event log writer.
    pub event_log: Option<BufWriter<File>>,

    /// Path filter restricting which files are tracked, if any. Shared with
    /// the TUI re-parse paths (file watcher, Serena cache rescan) so that
    /// changes to excluded files don't inject symbols back into the tree
    /// after the initial filtered scan. `None` means no filter — track
    /// everything.
    pub filter: Option<Arc<PathFilter>>,
}

impl App {
    pub fn new(project_tree: ProjectTree, project_root: PathBuf, event_log: Option<BufWriter<File>>) -> Self {
        // Start with all files collapsed.
        let collapsed: std::collections::HashSet<String> = project_tree
            .files
            .iter()
            .map(|f| f.file_path.to_string_lossy().to_string())
            .collect();

        let mut app = Self {
            project_tree,
            project_root,
            ledger: ContextLedger::new(),
            should_quit: false,
            tree_rows: Vec::new(),
            selected_index: 0,
            collapsed,
            activity: Vec::new(),
            activity_scroll_offset: 0,
            agents_seen: Vec::new(),
            agent_tree: AgentTree::new(),
            agent_filter: None,
            agent_selection_index: 0,
            focus: FocusPanel::Tree,
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
            event_log,
            filter: None,
        };
        app.rebuild_tree_rows();
        app
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
        self.ledger = ContextLedger::new();
        self.activity.clear();
        self.agents_seen.clear();
        self.agent_tree = AgentTree::new();
        self.seed_agent_tree_root();
        self.agent_filter = None;
        self.agent_selection_index = 0;
        self.session_slug = None;
        self.compaction_history.clear();
        self.compaction_call_count = 0;
        self.show_alignment_overlay = false;
        self.agent_alignment.clear();
        self.depth_cache = crate::tracking::alignment::DepthOrdinalCache::new();
        self.rebuild_tree_rows();
    }

    /// Snapshot the current ledger state, record a compaction event, then
    /// clear the live ledger. The summary text doesn't reliably describe what
    /// the model retained, so we drop all pre-compaction depth claims rather
    /// than over-report coverage that may no longer reflect the model's
    /// actual context. `compaction_history`, `activity`, and agent-tracking
    /// state are preserved — only the live read-depth ledger and the
    /// inter-compaction tool-call counter are reset.
    pub fn process_compaction(
        &mut self,
        summary: String,
        timestamp: String,
        agent_id: std::sync::Arc<str>,
        metadata: Option<crate::ingest::CompactionMetadata>,
    ) {
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

        // Wipe the live ledger: post-compaction depth tracking starts fresh.
        self.ledger = ContextLedger::new();
        self.depth_cache = crate::tracking::alignment::DepthOrdinalCache::new();
        self.compaction_call_count = 0;
        self.rebuild_tree_rows();
    }

    /// Rebuild the flattened tree rows from the project tree + collapsed state.
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

        for &idx in &file_indices {
            let file = &self.project_tree.files[idx];
            let file_path = file.file_path.to_string_lossy().to_string();
            let file_id = file_path.clone();
            let is_expanded = !self.collapsed.contains(&file_id);

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
                is_file: true,
                is_expanded,
                has_children: !file.symbols.is_empty(),
                line_range: format!("{} lines", file.total_lines),
                token_count: 0,
                read_depth: file_read_depth,
                coverage_status: Some(status),
                file_coverage_seen: seen,
                file_coverage_total: total,
            });

            if is_expanded {
                for sym in &file.symbols {
                    flatten_symbol(sym, 1, &self.collapsed, &self.ledger, agent_filter, &mut rows);
                }
            }
        }

        self.tree_rows = rows;
    }

    pub fn handle_key(&mut self, key: KeyEvent) {
        if self.search_mode {
            self.handle_search_key(key);
            return;
        }

        match key.code {
            KeyCode::Char('q') => self.should_quit = true,
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.should_quit = true;
            }
            KeyCode::Char('j') | KeyCode::Down => {
                if self.focus == FocusPanel::Stats {
                    self.move_agent_selection(1);
                } else {
                    self.move_selection(1);
                }
            }
            KeyCode::Char('k') | KeyCode::Up => {
                if self.focus == FocusPanel::Stats {
                    self.move_agent_selection(-1);
                } else {
                    self.move_selection(-1);
                }
            }
            KeyCode::Char('l') | KeyCode::Right | KeyCode::Enter => {
                if self.focus == FocusPanel::Stats {
                    self.apply_agent_selection();
                } else {
                    self.toggle_expand();
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
            KeyCode::Esc if self.show_alignment_overlay => {
                self.show_alignment_overlay = false;
            }
            KeyCode::Tab => self.cycle_focus(),
            KeyCode::BackTab => self.cycle_agent_filter_backward(),
            KeyCode::PageDown => self.move_selection(20),
            KeyCode::PageUp => self.move_selection(-20),
            _ => {}
        }
    }

    pub fn handle_mouse(&mut self, mouse: MouseEvent) {
        match mouse.kind {
            MouseEventKind::ScrollUp => match self.focus {
                FocusPanel::Activity => {
                    self.activity_scroll_offset = self.activity_scroll_offset.saturating_add(3);
                }
                FocusPanel::Stats => self.move_agent_selection(-1),
                FocusPanel::Tree => self.move_selection(-3),
            },
            MouseEventKind::ScrollDown => match self.focus {
                FocusPanel::Activity => {
                    self.activity_scroll_offset = self.activity_scroll_offset.saturating_sub(3);
                }
                FocusPanel::Stats => self.move_agent_selection(1),
                FocusPanel::Tree => self.move_selection(3),
            },
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
    }

    fn select_first(&mut self) {
        self.selected_index = 0;
    }

    fn select_last(&mut self) {
        if !self.tree_rows.is_empty() {
            self.selected_index = self.tree_rows.len() - 1;
        }
    }

    fn toggle_expand(&mut self) {
        if let Some(row) = self.tree_rows.get(self.selected_index) {
            if row.has_children {
                let id = row.symbol_id.clone();
                if self.collapsed.contains(&id) {
                    self.collapsed.remove(&id);
                } else {
                    self.collapsed.insert(id);
                }
                self.rebuild_tree_rows();
            }
        }
    }

    fn collapse_current(&mut self) {
        if let Some(row) = self.tree_rows.get(self.selected_index) {
            let id = row.symbol_id.clone();
            if row.has_children && !self.collapsed.contains(&id) {
                self.collapsed.insert(id);
                self.rebuild_tree_rows();
            }
        }
    }

    fn cycle_agent_filter(&mut self) {
        if self.agents_seen.is_empty() {
            self.agent_filter = None;
            self.agent_selection_index = 0;
            return;
        }
        match &self.agent_filter {
            None => {
                self.agent_filter = Some(self.agents_seen[0].clone());
                self.agent_selection_index = 1;
            }
            Some(current) => {
                let idx = self.agents_seen.iter().position(|a| a == current);
                match idx {
                    Some(i) if i + 1 < self.agents_seen.len() => {
                        self.agent_filter = Some(self.agents_seen[i + 1].clone());
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
        if self.agents_seen.is_empty() {
            self.agent_filter = None;
            self.agent_selection_index = 0;
            return;
        }
        match &self.agent_filter {
            None => {
                let last = self.agents_seen.len() - 1;
                self.agent_filter = Some(self.agents_seen[last].clone());
                self.agent_selection_index = self.agents_seen.len();
            }
            Some(current) => {
                let idx = self.agents_seen.iter().position(|a| a == current);
                match idx {
                    Some(0) => {
                        self.agent_filter = None;
                        self.agent_selection_index = 0;
                    }
                    Some(i) => {
                        self.agent_filter = Some(self.agents_seen[i - 1].clone());
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
        let total = self.agents_seen.len() + 1; // +1 for "All"
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

    fn cycle_focus(&mut self) {
        self.focus = match self.focus {
            FocusPanel::Tree => FocusPanel::Stats,
            FocusPanel::Stats => FocusPanel::Activity,
            FocusPanel::Activity => FocusPanel::Tree,
        };
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

    /// Process an agent tool call event and update the ledger.
    pub fn process_agent_event(&mut self, event: AgentToolCall) {
        self.compaction_call_count += 1;
        // Track unique agents.
        if !self.agents_seen.iter().any(|a| a.as_str() == &*event.agent_id) {
            self.agents_seen.push(event.agent_id.to_string());

            // Register in the agent hierarchy tree.
            //
            // NOTE: sub-agent JSONL *filenames* are prefixed `agent-<hash>`,
            // but the `agentId` field *inside* each line — which
            // `parse_jsonl_line` (src/ingest/claude.rs) prefers over the
            // filename/session-derived fallback — carries no such prefix
            // (e.g. `"a63c858997b4e6124"`, not `"agent-a63c858997b4e6124"`).
            // A `starts_with("agent-")` check therefore never matches real
            // sub-agent events; only hand-constructed test fixtures that
            // bake the prefix into `agent_id` happened to pass. Don't repeat
            // that mistake in future tests — use realistic unprefixed ids.
            //
            // Now that `self.session_id` is deterministically known before
            // any events are processed (see `set_session_id` /
            // `seed_agent_tree_root`), identity is the correct and only
            // check we need: the root's own events carry `agent_id ==
            // session_id` (and must NOT be re-parented to themselves);
            // every other `agent_id` is a child of the root. Fall back to
            // the old prefix heuristic only when no session_id is known
            // (e.g. test paths that skip `set_session_id`), preserving
            // prior behavior there.
            let parent_id = match self.session_id.as_deref() {
                Some(root) if event.agent_id.as_ref() == root => None,
                Some(root) => Some(root.to_string()),
                None if event.agent_id.starts_with("agent-") => self.agent_tree.root_id.clone(),
                None => None,
            };
            self.agent_tree.add_agent(AgentNode {
                id: event.agent_id.to_string(),
                parent_id,
                session_file: PathBuf::new(),
                label: event.label.to_string(),
            });
        }

        if let Some(ref file_path) = event.file_path {
            // Normalize the tool call path: strip the project root to get a relative path.
            let tool_rel = normalize_tool_path(file_path, &self.project_root);

            for file in &self.project_tree.files {
                if file.file_path == tool_rel {
                    if event.target_symbol.is_some() || event.target_lines.is_some() {
                        mark_targeted_symbols(&file.symbols, &event, &mut self.ledger, &mut self.depth_cache);
                    } else {
                        mark_file_symbols(&file.symbols, &event, &mut self.ledger, &mut self.depth_cache);
                    }
                }
            }
        }
        // Write to event log if configured.
        if let Some(ref mut writer) = self.event_log {
            let path_str = event
                .file_path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "-".to_string());
            let target = if let Some(ref sym) = event.target_symbol {
                sym.clone()
            } else if let Some(ref lines) = event.target_lines {
                format!("L{}-{}", lines.start, lines.end)
            } else {
                "-".to_string()
            };
            let _ = writeln!(
                writer,
                "[{}] agent={} tool={} depth={:?} path={} target={} desc=\"{}\"",
                event.timestamp_str,
                event.agent_id,
                event.tool_name,
                event.read_depth,
                path_str,
                target,
                event.description,
            );
            let _ = writer.flush();
        }

        // Only push tracked events to the activity feed.
        if event.read_depth != ReadDepth::Unseen {
            self.activity.push(event);
            self.activity_scroll_offset = 0; // Auto-scroll to latest
            if self.activity.len() > 200 {
                self.activity.drain(0..100);
            }
        }
        self.rebuild_tree_rows();
    }
}

fn flatten_symbol(
    sym: &SymbolNode,
    depth: usize,
    collapsed: &std::collections::HashSet<String>,
    ledger: &ContextLedger,
    agent_filter: Option<&str>,
    rows: &mut Vec<TreeRow>,
) {
    let is_expanded = !collapsed.contains(&sym.id);
    let read_depth = match agent_filter {
        Some(agent_id) => ledger.depth_of_for_agent(&sym.id, agent_id),
        None => ledger.depth_of(&sym.id),
    };

    rows.push(TreeRow {
        symbol_id: sym.id.clone(),
        display_name: sym.name.to_string(),
        label: sym.label,
        depth,
        is_file: false,
        is_expanded,
        has_children: !sym.children.is_empty(),
        line_range: format!("L{}-{}", sym.line_range.start, sym.line_range.end),
        token_count: sym.estimated_tokens as usize,
        read_depth,
        coverage_status: None,
        file_coverage_seen: 0,
        file_coverage_total: 0,
    });

    if is_expanded {
        for child in &sym.children {
            flatten_symbol(child, depth + 1, collapsed, ledger, agent_filter, rows);
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
pub fn mark_file_symbols(
    symbols: &[SymbolNode],
    event: &AgentToolCall,
    ledger: &mut ContextLedger,
    depth_cache: &mut crate::tracking::alignment::DepthOrdinalCache,
) {
    for sym in symbols {
        ledger.record(
            sym.id.clone(),
            event.read_depth,
            sym.content_hash,
            event.agent_id.to_string(),
            sym.estimated_tokens as usize,
        );
        depth_cache.record(&sym.id, &event.agent_id, event.read_depth);
        mark_file_symbols(&sym.children, event, ledger, depth_cache);
    }
}

/// Mark only the symbols that match the tool call's targeting info.
///
/// Name-targeted matches (via `target_symbol`) bulk-mark all descendants because
/// the tool response included the entire named symbol's body. Line-range matches
/// (via `target_lines`) recurse precisely so that only overlapping child symbols
/// are promoted — preventing unread siblings from being over-credited.
pub fn mark_targeted_symbols(
    symbols: &[SymbolNode],
    event: &AgentToolCall,
    ledger: &mut ContextLedger,
    depth_cache: &mut crate::tracking::alignment::DepthOrdinalCache,
) {
    for sym in symbols {
        match classify_symbol_match(sym, event) {
            MatchKind::ByName => {
                ledger.record(
                    sym.id.clone(),
                    event.read_depth,
                    sym.content_hash,
                    event.agent_id.to_string(),
                    sym.estimated_tokens as usize,
                );
                depth_cache.record(&sym.id, &event.agent_id, event.read_depth);
                // Full body was in the response — bulk-mark all descendants.
                mark_file_symbols(&sym.children, event, ledger, depth_cache);
            }
            MatchKind::ByLineOverlap => {
                ledger.record(
                    sym.id.clone(),
                    event.read_depth,
                    sym.content_hash,
                    event.agent_id.to_string(),
                    sym.estimated_tokens as usize,
                );
                depth_cache.record(&sym.id, &event.agent_id, event.read_depth);
                // Parent container overlaps the read range — recurse precisely so
                // only children whose ranges also overlap get promoted.
                mark_targeted_symbols(&sym.children, event, ledger, depth_cache);
            }
            MatchKind::None => {
                // No match at this level — keep searching in children.
                mark_targeted_symbols(&sym.children, event, ledger, depth_cache);
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
#[path = "../tests/helpers/mod.rs"]
#[allow(dead_code)]
mod helpers;

#[cfg(test)]
mod tests {
    use super::*;
    use super::helpers::*;
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

        mark_file_symbols(&[parent], &event, &mut ledger, &mut cache);

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

        mark_targeted_symbols(&[s1, s2], &event, &mut ledger, &mut cache);

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

        mark_targeted_symbols(&[s1, s2], &event, &mut ledger, &mut cache);

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

        mark_targeted_symbols(&[impl_block], &event, &mut ledger, &mut cache);

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

        mark_targeted_symbols(&[impl_block], &event, &mut ledger, &mut cache);

        assert_eq!(ledger.depth_of("f.rs::Foo"), ReadDepth::FullBody);
        assert_eq!(ledger.depth_of("f.rs::Foo/method_a"), ReadDepth::FullBody);
        assert_eq!(ledger.depth_of("f.rs::Foo/method_b"), ReadDepth::FullBody);
    }

    // --- App method tests ---

    fn test_app(files: Vec<FileSymbols>) -> App {
        let tree = project(files);
        App::new(tree, PathBuf::from("/test/project"), None)
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
        app.focus = FocusPanel::Stats;

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
        use crossterm::event::{MouseEvent, MouseEventKind, MouseButton};

        let mut app = test_app(vec![file("mock/f.rs", vec![sym("mock/f.rs::a", "a")])]);

        // Add some activity so scroll offset can increase
        for _ in 0..20 {
            let e = tool_call("Read", "/test/project/mock/f.rs", ReadDepth::FullBody);
            app.process_agent_event(e);
        }

        // Focus activity panel and scroll up
        app.focus = FocusPanel::Activity;
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
        app.focus = FocusPanel::Tree;
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
            .filter(|r| r.is_file)
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
            .filter(|r| r.is_file)
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
        // Live ledger is wiped — the symbol is Unseen again.
        assert_eq!(app.ledger.depth_of("mock/a.rs::a1"), ReadDepth::Unseen);
        assert_eq!(app.ledger.total_seen(), 0);
        assert_eq!(app.compaction_call_count, 0);
    }

    #[test]
    fn post_compaction_tool_calls_rebuild_ledger() {
        // After compaction wipes the ledger, subsequent tool calls populate
        // a fresh ledger without touching compaction_history.
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
        assert_eq!(app.ledger.depth_of("mock/a.rs::a1"), ReadDepth::Unseen,
            "pre-compaction read should not survive");
        assert_eq!(app.ledger.depth_of("mock/b.rs::b1"), ReadDepth::FullBody,
            "post-compaction read should be reflected");
        assert_eq!(app.compaction_call_count, 1, "counter restarted from zero after compaction");
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
        use std::path::PathBuf;

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
            label: "agent-abc".into(),
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
