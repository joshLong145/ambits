use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind};

use crate::symbols::{ProjectTree, SymbolNode};
use crate::tracking::ReadDepth;
use crate::tracking::ContextLedger;
use crate::ingest::AgentToolCall;

/// A flattened row in the tree view, ready for rendering.
#[derive(Debug, Clone)]
pub struct TreeRow {
    pub symbol_id: String,
    pub display_name: String,
    pub label: String,        // Language-specific label (e.g., "class", "def", "fn")
    pub depth: usize,         // nesting depth for indentation
    pub is_file: bool,        // true for file headers
    pub is_expanded: bool,
    pub has_children: bool,
    pub line_range: String,
    pub token_count: usize,
    pub read_depth: ReadDepth,
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

    // Agents seen.
    pub agents_seen: Vec<String>,

    // Agent filter: if Some, only show coverage from this agent.
    pub agent_filter: Option<String>,

    // Focus.
    pub focus: FocusPanel,

    // Search.
    pub search_mode: bool,
    pub search_query: String,

    // Session info for display.
    pub session_id: Option<String>,

    // Optional event log writer.
    pub event_log: Option<BufWriter<File>>,
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
            agents_seen: Vec::new(),
            agent_filter: None,
            focus: FocusPanel::Tree,
            search_mode: false,
            search_query: String::new(),
            session_id: None,
            event_log,
        };
        app.rebuild_tree_rows();
        app
    }

    /// Rebuild the flattened tree rows from the project tree + collapsed state.
    pub fn rebuild_tree_rows(&mut self) {
        let mut rows = Vec::new();

        for file in &self.project_tree.files {
            let file_path = file.file_path.to_string_lossy().to_string();
            let file_id = file_path.clone();
            let is_expanded = !self.collapsed.contains(&file_id);

            // Compute file-level coverage: if any symbol in the file has been seen, mark file as having coverage
            let file_has_coverage = has_file_coverage(&file.symbols, &self.ledger);
            let file_read_depth = if file_has_coverage {
                ReadDepth::NameOnly // Use NameOnly to indicate "has coverage"
            } else {
                ReadDepth::Unseen
            };

            rows.push(TreeRow {
                symbol_id: file_id.clone(),
                display_name: file_path.clone(),
                label: String::new(),
                depth: 0,
                is_file: true,
                is_expanded,
                has_children: !file.symbols.is_empty(),
                line_range: format!("{} lines", file.total_lines),
                token_count: 0,
                read_depth: file_read_depth,
            });

            if is_expanded {
                for sym in &file.symbols {
                    flatten_symbol(sym, 1, &self.collapsed, &self.ledger, &mut rows);
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
            KeyCode::Char('j') | KeyCode::Down => self.move_selection(1),
            KeyCode::Char('k') | KeyCode::Up => self.move_selection(-1),
            KeyCode::Char('l') | KeyCode::Right | KeyCode::Enter => self.toggle_expand(),
            KeyCode::Char('h') | KeyCode::Left => self.collapse_current(),
            KeyCode::Char('G') => self.select_last(),
            KeyCode::Char('g') => self.select_first(),
            KeyCode::Char('/') => {
                self.search_mode = true;
                self.search_query.clear();
            }
            KeyCode::Char('a') => self.cycle_agent_filter(),
            KeyCode::Tab => self.cycle_focus(),
            KeyCode::PageDown => self.move_selection(20),
            KeyCode::PageUp => self.move_selection(-20),
            _ => {}
        }
    }

    pub fn handle_mouse(&mut self, mouse: MouseEvent) {
        match mouse.kind {
            MouseEventKind::ScrollUp => self.move_selection(-3),
            MouseEventKind::ScrollDown => self.move_selection(3),
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
            return;
        }
        match &self.agent_filter {
            None => {
                self.agent_filter = Some(self.agents_seen[0].clone());
            }
            Some(current) => {
                let idx = self.agents_seen.iter().position(|a| a == current);
                match idx {
                    Some(i) if i + 1 < self.agents_seen.len() => {
                        self.agent_filter = Some(self.agents_seen[i + 1].clone());
                    }
                    _ => {
                        self.agent_filter = None;
                    }
                }
            }
        }
        self.rebuild_tree_rows();
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
        // Track unique agents.
        if !self.agents_seen.contains(&event.agent_id) {
            self.agents_seen.push(event.agent_id.clone());
        }

        if let Some(ref file_path) = event.file_path {
            // Normalize the tool call path: strip the project root to get a relative path.
            let tool_rel = normalize_tool_path(file_path, &self.project_root);

            for file in &self.project_tree.files {
                if file.file_path == tool_rel {
                    if event.target_symbol.is_some() || event.target_lines.is_some() {
                        mark_targeted_symbols(&file.symbols, &event, &mut self.ledger);
                    } else {
                        mark_file_symbols(&file.symbols, &event, &mut self.ledger);
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
    rows: &mut Vec<TreeRow>,
) {
    let is_expanded = !collapsed.contains(&sym.id);
    let read_depth = ledger.depth_of(&sym.id);

    rows.push(TreeRow {
        symbol_id: sym.id.clone(),
        display_name: sym.name.clone(),
        label: sym.label.clone(),
        depth,
        is_file: false,
        is_expanded,
        has_children: !sym.children.is_empty(),
        line_range: format!("L{}-{}", sym.line_range.start, sym.line_range.end),
        token_count: sym.estimated_tokens,
        read_depth,
    });

    if is_expanded {
        for child in &sym.children {
            flatten_symbol(child, depth + 1, collapsed, ledger, rows);
        }
    }
}

/// Convert a tool call file path (usually absolute) to a relative path matching
/// the project tree's convention. Strips the project root prefix if present.
fn normalize_tool_path(tool_path: &Path, project_root: &Path) -> PathBuf {
    if tool_path.is_absolute() {
        tool_path
            .strip_prefix(project_root)
            .unwrap_or(tool_path)
            .to_path_buf()
    } else {
        tool_path.to_path_buf()
    }
}

fn mark_file_symbols(
    symbols: &[SymbolNode],
    event: &AgentToolCall,
    ledger: &mut ContextLedger,
) {
    for sym in symbols {
        ledger.record(
            sym.id.clone(),
            event.read_depth,
            sym.content_hash,
            event.agent_id.clone(),
            sym.estimated_tokens,
        );
        mark_file_symbols(&sym.children, event, ledger);
    }
}

/// Mark only the symbols that match the tool call's targeting info.
fn mark_targeted_symbols(
    symbols: &[SymbolNode],
    event: &AgentToolCall,
    ledger: &mut ContextLedger,
) {
    for sym in symbols {
        let matches = symbol_matches_target(sym, event);
        if matches {
            ledger.record(
                sym.id.clone(),
                event.read_depth,
                sym.content_hash,
                event.agent_id.clone(),
                sym.estimated_tokens,
            );
            // If we matched a parent (e.g. an impl block), also mark children
            mark_file_symbols(&sym.children, event, ledger);
        } else {
            // Recurse — the target might be a child symbol
            mark_targeted_symbols(&sym.children, event, ledger);
        }
    }
}

/// Check if a symbol matches the tool call's target_symbol or target_lines.
fn symbol_matches_target(sym: &SymbolNode, event: &AgentToolCall) -> bool {
    if let Some(ref target_name) = event.target_symbol {
        // Match if the symbol's id ends with the target name path.
        // SymbolId format is "file_path::name_path", e.g. "src/app.rs::impl App/handle_key"
        // target_name is a Serena name_path like "App/handle_key" or just "handle_key"
        if let Some(name_part) = sym.id.split("::").last() {
            if name_part == target_name || name_part.ends_with(&format!("/{target_name}")) {
                return true;
            }
        }
        // Also check plain name match for simple names
        if sym.name == *target_name {
            return true;
        }
    }
    if let Some(ref target_range) = event.target_lines {
        // Check if symbol's line range overlaps with the target line range
        if sym.line_range.start < target_range.end && target_range.start < sym.line_range.end {
            return true;
        }
    }
    false
}

/// Check if any symbol in a file (recursively) has been seen by the agent.
fn has_file_coverage(symbols: &[SymbolNode], ledger: &ContextLedger) -> bool {
    for sym in symbols {
        if ledger.depth_of(&sym.id) != ReadDepth::Unseen {
            return true;
        }
        if has_file_coverage(&sym.children, ledger) {
            return true;
        }
    }
    false
}
