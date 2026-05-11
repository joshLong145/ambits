//! Coverage report generation for symbol visibility metrics.
//!
//! This module provides structures and formatters for generating coverage reports
//! that show how much of a project's symbols have been seen by an LLM agent.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use color_eyre::eyre::Result;
use serde::Serialize;

use crate::ingest::SessionEvent;
use crate::symbols::{ProjectTree, SymbolNode};
use crate::tracking::{ContextLedger, ReadDepth};

/// Per-file coverage metrics.
#[derive(Debug, Clone, Serialize)]
pub struct FileCoverage {
    /// Full relative path to the file.
    pub path: String,
    /// Total number of symbols in the file.
    pub total_symbols: usize,
    /// Symbols with depth > Unseen (NameOnly, Overview, Signature, FullBody).
    pub seen_count: usize,
    /// Symbols with depth == FullBody.
    pub full_count: usize,
}

impl FileCoverage {
    /// Calculate the percentage of symbols that have been seen.
    pub fn seen_percent(&self) -> f64 {
        if self.total_symbols == 0 {
            0.0
        } else {
            (self.seen_count as f64 / self.total_symbols as f64) * 100.0
        }
    }

    /// Calculate the percentage of symbols with full body reads.
    pub fn full_percent(&self) -> f64 {
        if self.total_symbols == 0 {
            0.0
        } else {
            (self.full_count as f64 / self.total_symbols as f64) * 100.0
        }
    }
}

/// Pre-compaction snapshot for a single context compaction event, intended for the
/// coverage report. Files are stored as relative-path strings (sorted) so the
/// summary remains stable in JSON output.
#[derive(Debug, Clone, Serialize)]
pub struct CompactionSummary {
    pub sequence: u32,
    pub timestamp: String,
    pub summary: String,
    pub tool_calls_before: usize,
    pub files_before: Vec<String>,
    pub symbols_seen_before: usize,
    pub seen_percent_before: f64,
}

/// Complete coverage report for a project.
#[derive(Debug, Clone, Serialize)]
pub struct CoverageReport {
    /// Session ID if available.
    pub session_id: Option<String>,
    /// Agent ID if filtering by agent.
    pub agent_id: Option<String>,
    /// Per-file coverage metrics.
    pub files: Vec<FileCoverage>,
    /// Compactions detected in the session, in occurrence order. Empty when none.
    pub compactions: Vec<CompactionSummary>,
}

impl CoverageReport {
    /// Build a coverage report from a project tree and context ledger.
    /// When `agent_filter` is `Some(id)`, only counts coverage from that agent.
    pub fn from_project(
        project_tree: &ProjectTree,
        ledger: &ContextLedger,
        agent_filter: Option<&str>,
    ) -> Self {
        let mut files: Vec<FileCoverage> = project_tree
            .files
            .iter()
            .map(|file| {
                let path = file.file_path.to_string_lossy().to_string();
                let (total, seen, full) = count_symbols(&file.symbols, ledger, agent_filter);
                FileCoverage {
                    path,
                    total_symbols: total,
                    seen_count: seen,
                    full_count: full,
                }
            })
            .collect();

        // Sort by full_percent ascending (lowest coverage first)
        files.sort_by(|a, b| {
            a.full_percent()
                .partial_cmp(&b.full_percent())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Self {
            session_id: None,
            agent_id: agent_filter.map(|s| s.to_string()),
            files,
            compactions: Vec::new(),
        }
    }

    /// Total symbols across all files.
    pub fn total_symbols(&self) -> usize {
        self.files.iter().map(|f| f.total_symbols).sum()
    }

    /// Total seen symbols across all files.
    pub fn total_seen(&self) -> usize {
        self.files.iter().map(|f| f.seen_count).sum()
    }

    /// Total full-body read symbols across all files.
    pub fn total_full(&self) -> usize {
        self.files.iter().map(|f| f.full_count).sum()
    }

    /// Calculate overall seen percentage.
    pub fn total_seen_percent(&self) -> f64 {
        let total = self.total_symbols();
        if total == 0 {
            0.0
        } else {
            (self.total_seen() as f64 / total as f64) * 100.0
        }
    }

    /// Calculate overall full-body percentage.
    pub fn total_full_percent(&self) -> f64 {
        let total = self.total_symbols();
        if total == 0 {
            0.0
        } else {
            (self.total_full() as f64 / total as f64) * 100.0
        }
    }
}

/// Count symbols recursively, returning (total, seen_count, full_count).
/// When `agent_filter` is `Some(id)`, counts use per-agent depths instead of aggregate.
pub fn count_symbols(
    symbols: &[SymbolNode],
    ledger: &ContextLedger,
    agent_filter: Option<&str>,
) -> (usize, usize, usize) {
    let mut total = 0;
    let mut seen = 0;
    let mut full = 0;

    for sym in symbols {
        total += 1;
        let depth = match agent_filter {
            Some(agent_id) => ledger.depth_of_for_agent(&sym.id, agent_id),
            None => ledger.depth_of(&sym.id),
        };

        if depth.is_seen() {
            seen += 1;
        }
        if depth == ReadDepth::FullBody {
            full += 1;
        }

        // Recurse into children
        let (child_total, child_seen, child_full) =
            count_symbols(&sym.children, ledger, agent_filter);
        total += child_total;
        seen += child_seen;
        full += child_full;
    }

    (total, seen, full)
}

/// Trait for formatting coverage reports.
/// Implement this trait to add new output formats (JSON, CSV, etc.).
pub trait CoverageFormatter {
    fn format(&self, report: &CoverageReport) -> String;
}

/// Text table formatter for terminal output.
#[derive(Debug, Clone)]
pub struct TextFormatter {
    /// Minimum width for the path column.
    pub min_path_width: usize,
}

impl Default for TextFormatter {
    fn default() -> Self {
        Self { min_path_width: 40 }
    }
}

impl CoverageFormatter for TextFormatter {
    fn format(&self, report: &CoverageReport) -> String {
        let mut output = String::new();

        // Header
        let session_str = report
            .session_id
            .as_deref()
            .unwrap_or("none");
        let agent_str = report
            .agent_id
            .as_deref()
            .map(|a| format!(", agent: {}", a))
            .unwrap_or_default();
        output.push_str(&format!("Coverage Report (session: {}{})\n", session_str, agent_str));

        // Calculate path width based on longest path
        let max_path_len = report
            .files
            .iter()
            .map(|f| f.path.len())
            .max()
            .unwrap_or(0)
            .max(self.min_path_width)
            .max(5); // "TOTAL" length

        let separator = "─".repeat(max_path_len + 45);
        output.push_str(&separator);
        output.push('\n');

        // Column headers
        output.push_str(&format!(
            "{:<width$} {:>8} {:>7} {:>7} {:>7} {:>7}\n",
            "File",
            "Symbols",
            "Seen",
            "Full",
            "Seen%",
            "Full%",
            width = max_path_len
        ));

        output.push_str(&separator);
        output.push('\n');

        // File rows
        for file in &report.files {
            output.push_str(&format!(
                "{:<width$} {:>8} {:>7} {:>7} {:>6.0}% {:>6.0}%\n",
                file.path,
                file.total_symbols,
                file.seen_count,
                file.full_count,
                file.seen_percent(),
                file.full_percent(),
                width = max_path_len
            ));
        }

        output.push_str(&separator);
        output.push('\n');

        // Total row
        output.push_str(&format!(
            "{:<width$} {:>8} {:>7} {:>7} {:>6.0}% {:>6.0}%\n",
            "TOTAL",
            report.total_symbols(),
            report.total_seen(),
            report.total_full(),
            report.total_seen_percent(),
            report.total_full_percent(),
            width = max_path_len
        ));

        if !report.compactions.is_empty() {
            append_compactions_section(&mut output, &report.compactions);
        }

        output
    }
}

/// Render the "Context Compactions" section after the TOTAL row. Caller must
/// have already confirmed that `compactions` is non-empty.
fn append_compactions_section(output: &mut String, compactions: &[CompactionSummary]) {
    const MAX_FILES_SHOWN: usize = 10;
    const SEP_WIDTH: usize = 63;
    output.push('\n');
    output.push_str(&format!("Context Compactions ({})\n", compactions.len()));
    let separator: String = "─".repeat(SEP_WIDTH);
    output.push_str(&separator);
    output.push('\n');

    for c in compactions {
        output.push_str(&format!(
            "#{}  {}  {} calls · {:.1}% seen\n",
            c.sequence, c.timestamp, c.tool_calls_before, c.seen_percent_before,
        ));
        output.push_str(&format!("    Files ({}):\n", c.files_before.len()));
        for path in c.files_before.iter().take(MAX_FILES_SHOWN) {
            output.push_str(&format!("      {path}\n"));
        }
        if c.files_before.len() > MAX_FILES_SHOWN {
            output.push_str(&format!(
                "      ... ({} more)\n",
                c.files_before.len() - MAX_FILES_SHOWN,
            ));
        }
        append_wrapped_summary(output, &c.summary);
        output.push('\n');
    }
}

/// Append the summary text after a `Summary:` label, wrapping at 80 chars with
/// a hanging indent so continuation lines line up under the first character of
/// the summary text.
fn append_wrapped_summary(output: &mut String, summary: &str) {
    const LINE_WIDTH: usize = 80;
    const INDENT_FIRST: &str = "    Summary: ";
    const INDENT_CONT: &str = "      ";

    let first_budget = LINE_WIDTH.saturating_sub(INDENT_FIRST.len());
    let cont_budget = LINE_WIDTH.saturating_sub(INDENT_CONT.len());

    let mut words = summary.split_whitespace().peekable();
    let mut line = String::new();
    let mut first = true;
    while let Some(word) = words.next() {
        let budget = if first { first_budget } else { cont_budget };
        if line.is_empty() {
            line.push_str(word);
        } else if line.len() + 1 + word.len() > budget {
            // Flush current line.
            output.push_str(if first { INDENT_FIRST } else { INDENT_CONT });
            output.push_str(&line);
            output.push('\n');
            line.clear();
            line.push_str(word);
            first = false;
        } else {
            line.push(' ');
            line.push_str(word);
        }
        if words.peek().is_none() && !line.is_empty() {
            output.push_str(if first { INDENT_FIRST } else { INDENT_CONT });
            output.push_str(&line);
            output.push('\n');
            line.clear();
        }
    }
    if line.is_empty() && first {
        // Empty summary — still print the label so the structure is consistent.
        output.push_str(INDENT_FIRST);
        output.push('\n');
    }
}

/// Compact JSON formatter for machine-readable output.
///
/// Emits a single-line, schema-versioned JSON object terminated by a newline.
/// The wire format is intentionally decoupled from `CoverageReport`'s internal
/// shape via a private DTO so percentages can be precomputed and field
/// ordering (schema_version first) is guaranteed.
#[derive(Debug, Clone, Default)]
pub struct JsonFormatter;

impl CoverageFormatter for JsonFormatter {
    fn format(&self, report: &CoverageReport) -> String {
        #[derive(Serialize)]
        struct Totals {
            symbols: usize,
            seen: usize,
            full: usize,
            seen_percent: f64,
            full_percent: f64,
        }

        #[derive(Serialize)]
        struct FileDto<'a> {
            path: &'a str,
            total_symbols: usize,
            seen_count: usize,
            full_count: usize,
            seen_percent: f64,
            full_percent: f64,
        }

        #[derive(Serialize)]
        struct StateBeforeDto<'a> {
            tool_calls: usize,
            files_accessed: &'a [String],
            symbols_seen: usize,
            seen_percent: f64,
        }

        #[derive(Serialize)]
        struct CompactionDto<'a> {
            sequence: u32,
            timestamp: &'a str,
            summary: &'a str,
            state_before: StateBeforeDto<'a>,
        }

        #[derive(Serialize)]
        struct ReportDto<'a> {
            schema_version: u32,
            session_id: Option<&'a str>,
            agent_id: Option<&'a str>,
            totals: Totals,
            files: Vec<FileDto<'a>>,
            compactions: Vec<CompactionDto<'a>>,
        }

        let dto = ReportDto {
            schema_version: 2,
            session_id: report.session_id.as_deref(),
            agent_id: report.agent_id.as_deref(),
            totals: Totals {
                symbols: report.total_symbols(),
                seen: report.total_seen(),
                full: report.total_full(),
                seen_percent: report.total_seen_percent(),
                full_percent: report.total_full_percent(),
            },
            files: report
                .files
                .iter()
                .map(|f| FileDto {
                    path: &f.path,
                    total_symbols: f.total_symbols,
                    seen_count: f.seen_count,
                    full_count: f.full_count,
                    seen_percent: f.seen_percent(),
                    full_percent: f.full_percent(),
                })
                .collect(),
            compactions: report
                .compactions
                .iter()
                .map(|c| CompactionDto {
                    sequence: c.sequence,
                    timestamp: &c.timestamp,
                    summary: &c.summary,
                    state_before: StateBeforeDto {
                        tool_calls: c.tool_calls_before,
                        files_accessed: &c.files_before,
                        symbols_seen: c.symbols_seen_before,
                        seen_percent: c.seen_percent_before,
                    },
                })
                .collect(),
        };

        let mut out = serde_json::to_string(&dto).expect("CoverageReport DTO must serialize");
        out.push('\n');
        out
    }
}

/// Run a coverage report for a project and print it to stdout.
///
/// Resolves the log directory and session automatically when not provided.
/// Supports optional agent-prefix filtering.
pub fn run_report(
    project_path: &Path,
    project_tree: &ProjectTree,
    log_dir_opt: &Option<PathBuf>,
    session_opt: &Option<String>,
    agent_opt: &Option<String>,
    ingester: &dyn crate::ingest::SessionIngester,
    formatter: &dyn CoverageFormatter,
) -> Result<()> {
    use crate::tracking::ContextLedger;

    // 1. Resolve log directory.
    let log_dir = log_dir_opt
        .clone()
        .or_else(|| ingester.log_dir_for_project(project_path));

    // 2. Find session (auto-detect if not provided).
    let session_id = session_opt.clone().or_else(|| {
        log_dir
            .as_ref()
            .and_then(|d| ingester.find_latest_session(d))
    });

    // 3. Build ledger from session logs.
    let mut ledger = ContextLedger::new();
    let mut known_agents: Vec<String> = Vec::new();
    let mut files_accessed: BTreeSet<PathBuf> = BTreeSet::new();
    let mut tool_call_count: usize = 0;
    let mut compactions: Vec<CompactionSummary> = Vec::new();
    if let (Some(ref log_dir), Some(ref sid)) = (&log_dir, &session_id) {
        let log_files = ingester.session_log_files(log_dir, sid);
        for log_file in &log_files {
            let events = ingester.parse_log_file_with_root(log_file, project_path);
            for event in events {
                match event {
                    SessionEvent::ToolCall(tc) => {
                        tool_call_count += 1;
                        if !known_agents.iter().any(|a: &String| a.as_str() == &*tc.agent_id) {
                            known_agents.push(tc.agent_id.to_string());
                        }
                        if let Some(ref file_path) = tc.file_path {
                            let tool_rel = crate::app::normalize_tool_path(file_path, project_path);
                            files_accessed.insert(tool_rel.clone());
                            for file in &project_tree.files {
                                if file.file_path == tool_rel {
                                    if tc.target_symbol.is_some() || tc.target_lines.is_some() {
                                        crate::app::mark_targeted_symbols(&file.symbols, &tc, &mut ledger);
                                    } else {
                                        crate::app::mark_file_symbols(&file.symbols, &tc, &mut ledger);
                                    }
                                }
                            }
                        }
                    }
                    SessionEvent::Compacted { summary, timestamp, .. } => {
                        let seen = ledger.total_seen();
                        let total = project_tree.total_symbols();
                        compactions.push(CompactionSummary {
                            sequence: compactions.len() as u32 + 1,
                            timestamp,
                            summary,
                            tool_calls_before: tool_call_count,
                            files_before: files_accessed
                                .iter()
                                .map(|p| p.to_string_lossy().into_owned())
                                .collect(),
                            symbols_seen_before: seen,
                            seen_percent_before: if total > 0 {
                                seen as f64 / total as f64 * 100.0
                            } else {
                                0.0
                            },
                        });
                    }
                    SessionEvent::SessionCleared => {
                        ledger = ContextLedger::new();
                        files_accessed.clear();
                        tool_call_count = 0;
                        compactions.clear();
                    }
                }
            }
        }
    }

    // 4. Resolve agent filter (supports prefix matching).
    let resolved_agent = agent_opt.as_ref().map(|prefix| {
        let matches: Vec<&String> = known_agents
            .iter()
            .filter(|id| id.starts_with(prefix.as_str()))
            .collect();
        match matches.len() {
            1 => matches[0].clone(),
            0 => {
                eprintln!("Warning: no agent matching prefix '{}'", prefix);
                prefix.clone()
            }
            _ => {
                eprintln!(
                    "Warning: multiple agents match prefix '{}': {:?}",
                    prefix,
                    matches.iter().take(5).collect::<Vec<_>>()
                );
                matches[0].clone()
            }
        }
    });

    // 5. Generate and print report.
    let mut report = CoverageReport::from_project(project_tree, &ledger, resolved_agent.as_deref());
    report.session_id = session_id;
    report.compactions = compactions;

    print!("{}", formatter.format(&report));

    Ok(())
}

/// Print a project's symbol tree to stdout.
pub fn dump_tree(root: &Path, project_tree: &ProjectTree) {
    println!(
        "Project: {} ({} files, {} symbols)",
        root.display(),
        project_tree.total_files(),
        project_tree.total_symbols(),
    );
    println!();

    for file in &project_tree.files {
        println!("  {} ({} lines)", file.file_path.display(), file.total_lines);
        for sym in &file.symbols {
            print_symbol(sym, 4);
        }
    }
}

/// Print a single symbol and its children recursively with indentation.
pub fn print_symbol(sym: &SymbolNode, indent: usize) {
    let pad = " ".repeat(indent);
    println!(
        "{}{} {} [L{}-{}] (~{} tokens)",
        pad,
        sym.label,
        sym.name,
        sym.line_range.start,
        sym.line_range.end,
        sym.estimated_tokens,
    );
    for child in &sym.children {
        print_symbol(child, indent + 2);
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
    use crate::tracking::ContextLedger;

    #[test]
    fn seen_percent_basic() {
        let fc = FileCoverage { path: "a.rs".into(), total_symbols: 10, seen_count: 3, full_count: 1 };
        assert!((fc.seen_percent() - 30.0).abs() < 0.01);
    }

    #[test]
    fn seen_percent_zero_total() {
        let fc = FileCoverage { path: "a.rs".into(), total_symbols: 0, seen_count: 0, full_count: 0 };
        assert!((fc.seen_percent()).abs() < 0.01);
    }

    #[test]
    fn full_percent_basic() {
        let fc = FileCoverage { path: "a.rs".into(), total_symbols: 4, seen_count: 2, full_count: 2 };
        assert!((fc.full_percent() - 50.0).abs() < 0.01);
    }

    #[test]
    fn count_symbols_empty() {
        let ledger = ContextLedger::new();
        assert_eq!(count_symbols(&[], &ledger, None), (0, 0, 0));
    }

    #[test]
    fn count_symbols_nested() {
        let mut ledger = ContextLedger::new();
        let child1 = sym("c1", "child1");
        let child2 = sym("c2", "child2");
        let parent = sym_with_children("p", "parent", vec![child1, child2]);

        // Mark parent as FullBody, child1 as Overview.
        ledger.record("p".into(), ReadDepth::FullBody, [0; 32], "ag".into(), 10);
        ledger.record("c1".into(), ReadDepth::Overview, [0; 32], "ag".into(), 10);

        let (total, seen, full) = count_symbols(&[parent], &ledger, None);
        assert_eq!(total, 3);
        assert_eq!(seen, 2);
        assert_eq!(full, 1);
    }

    #[test]
    fn from_project_sorts_by_full_percent() {
        let mut ledger = ContextLedger::new();
        let tree = project(vec![
            file("b.rs", vec![sym("b1", "b1")]),
            file("a.rs", vec![sym("a1", "a1")]),
        ]);
        // Mark b1 as FullBody (100%), a1 unseen (0%).
        ledger.record("b1".into(), ReadDepth::FullBody, [0; 32], "ag".into(), 10);

        let report = CoverageReport::from_project(&tree, &ledger, None);
        // Sorted ascending by full_percent: a.rs (0%) first, b.rs (100%) second.
        assert_eq!(report.files[0].path, "a.rs");
        assert_eq!(report.files[1].path, "b.rs");
    }

    #[test]
    fn report_aggregates() {
        let mut ledger = ContextLedger::new();
        let tree = project(vec![
            file("a.rs", vec![sym("a1", "a1"), sym("a2", "a2")]),
            file("b.rs", vec![sym("b1", "b1")]),
        ]);
        ledger.record("a1".into(), ReadDepth::FullBody, [0; 32], "ag".into(), 10);
        ledger.record("b1".into(), ReadDepth::Overview, [0; 32], "ag".into(), 10);

        let report = CoverageReport::from_project(&tree, &ledger, None);
        assert_eq!(report.total_symbols(), 3);
        assert_eq!(report.total_seen(), 2);
        assert_eq!(report.total_full(), 1);
    }

    #[test]
    fn count_symbols_with_agent_filter() {
        let mut ledger = ContextLedger::new();
        let syms = vec![sym("s1", "s1"), sym("s2", "s2")];

        // Agent A reads s1 at FullBody, Agent B reads s2 at Overview.
        ledger.record("s1".into(), ReadDepth::FullBody, [0; 32], "agent_a".into(), 10);
        ledger.record("s2".into(), ReadDepth::Overview, [0; 32], "agent_b".into(), 5);

        // Aggregate: both seen, 1 full.
        let (total, seen, full) = count_symbols(&syms, &ledger, None);
        assert_eq!((total, seen, full), (2, 2, 1));

        // Agent A: only s1 seen (FullBody).
        let (total, seen, full) = count_symbols(&syms, &ledger, Some("agent_a"));
        assert_eq!((total, seen, full), (2, 1, 1));

        // Agent B: only s2 seen (Overview).
        let (total, seen, full) = count_symbols(&syms, &ledger, Some("agent_b"));
        assert_eq!((total, seen, full), (2, 1, 0));

        // Unknown agent: nothing seen.
        let (total, seen, full) = count_symbols(&syms, &ledger, Some("agent_c"));
        assert_eq!((total, seen, full), (2, 0, 0));
    }

    #[test]
    fn from_project_with_agent_filter() {
        let mut ledger = ContextLedger::new();
        let tree = project(vec![
            file("a.rs", vec![sym("a1", "a1")]),
            file("b.rs", vec![sym("b1", "b1")]),
        ]);

        // Agent A reads a1, Agent B reads b1.
        ledger.record("a1".into(), ReadDepth::FullBody, [0; 32], "agent_a".into(), 10);
        ledger.record("b1".into(), ReadDepth::FullBody, [0; 32], "agent_b".into(), 10);

        // Aggregate: both files covered.
        let report = CoverageReport::from_project(&tree, &ledger, None);
        assert_eq!(report.total_seen(), 2);

        // Agent A: only a.rs covered.
        let report = CoverageReport::from_project(&tree, &ledger, Some("agent_a"));
        assert_eq!(report.total_seen(), 1);
        assert_eq!(report.total_full(), 1);
        let fa = report.files.iter().find(|f| f.path == "a.rs").unwrap();
        assert_eq!(fa.seen_count, 1);

        // Agent B: only b.rs covered.
        let report = CoverageReport::from_project(&tree, &ledger, Some("agent_b"));
        assert_eq!(report.total_seen(), 1);
        let fb = report.files.iter().find(|f| f.path == "b.rs").unwrap();
        assert_eq!(fb.seen_count, 1);
    }

    #[test]
    fn text_formatter_output() {
        let report = CoverageReport {
            session_id: Some("abc-123".into()),
            agent_id: None,
            files: vec![FileCoverage {
                path: "src/main.rs".into(),
                total_symbols: 10,
                seen_count: 8,
                full_count: 5,
            }],
            compactions: Vec::new(),
        };
        let formatter = TextFormatter::default();
        let output = formatter.format(&report);
        assert!(output.contains("Coverage Report (session: abc-123)"));
        assert!(output.contains("src/main.rs"));
        assert!(output.contains("TOTAL"));
    }

    #[test]
    fn json_formatter_includes_schema_version() {
        let report = CoverageReport {
            session_id: Some("s".into()),
            agent_id: None,
            files: vec![],
            compactions: Vec::new(),
        };
        let output = JsonFormatter.format(&report);
        let value: serde_json::Value =
            serde_json::from_str(output.trim()).expect("output must be valid JSON");
        assert_eq!(value["schema_version"], 2);
    }

    #[test]
    fn json_formatter_compact_no_internal_newlines() {
        let report = CoverageReport {
            session_id: Some("s".into()),
            agent_id: None,
            files: vec![FileCoverage {
                path: "a.rs".into(),
                total_symbols: 1,
                seen_count: 1,
                full_count: 1,
            }],
            compactions: Vec::new(),
        };
        let output = JsonFormatter.format(&report);
        assert!(output.ends_with('\n'), "output must end with a trailing newline");
        let body = output.trim_end_matches('\n');
        assert!(!body.contains('\n'), "JSON body must be a single line");
    }

    fn sample_compaction() -> CompactionSummary {
        CompactionSummary {
            sequence: 1,
            timestamp: "2026-05-11T14:23:00Z".to_string(),
            summary: "The user is building a CLI parser. Files modified: src/parser/mod.rs, src/cli.rs. Next step: add flag validation.".to_string(),
            tool_calls_before: 47,
            files_before: vec![
                "src/cli.rs".to_string(),
                "src/parser/mod.rs".to_string(),
            ],
            symbols_seen_before: 31,
            seen_percent_before: 31.2,
        }
    }

    #[test]
    fn text_formatter_compaction_section_omitted_when_empty() {
        let report = CoverageReport {
            session_id: Some("s".into()),
            agent_id: None,
            files: vec![],
            compactions: Vec::new(),
        };
        let output = TextFormatter::default().format(&report);
        assert!(!output.contains("Context Compactions"),
            "compaction section must be absent when empty, got:\n{output}");
    }

    #[test]
    fn text_formatter_compaction_section_with_entries() {
        let report = CoverageReport {
            session_id: Some("s".into()),
            agent_id: None,
            files: vec![],
            compactions: vec![sample_compaction()],
        };
        let output = TextFormatter::default().format(&report);
        assert!(output.contains("Context Compactions (1)"));
        assert!(output.contains("#1"));
        assert!(output.contains("47 calls"));
        assert!(output.contains("src/cli.rs"));
        assert!(output.contains("CLI parser"));
    }

    #[test]
    fn json_formatter_compaction_array_empty_when_none() {
        let report = CoverageReport {
            session_id: Some("s".into()),
            agent_id: None,
            files: vec![],
            compactions: Vec::new(),
        };
        let output = JsonFormatter.format(&report);
        let value: serde_json::Value =
            serde_json::from_str(output.trim()).expect("output must be valid JSON");
        assert_eq!(value["schema_version"], 2);
        assert!(value["compactions"].is_array());
        assert_eq!(value["compactions"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn json_formatter_compaction_array_with_entries() {
        let report = CoverageReport {
            session_id: Some("s".into()),
            agent_id: None,
            files: vec![],
            compactions: vec![sample_compaction()],
        };
        let output = JsonFormatter.format(&report);
        let value: serde_json::Value =
            serde_json::from_str(output.trim()).expect("output must be valid JSON");
        let arr = value["compactions"].as_array().unwrap();
        assert_eq!(arr.len(), 1);
        let entry = &arr[0];
        assert_eq!(entry["sequence"], 1);
        assert_eq!(entry["timestamp"], "2026-05-11T14:23:00Z");
        let state = &entry["state_before"];
        assert_eq!(state["tool_calls"], 47);
        assert_eq!(state["symbols_seen"], 31);
        let files = state["files_accessed"].as_array().unwrap();
        assert_eq!(files.len(), 2);
        assert_eq!(files[0], "src/cli.rs");
    }
}
