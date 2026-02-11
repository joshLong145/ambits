//! Coverage report generation for symbol visibility metrics.
//!
//! This module provides structures and formatters for generating coverage reports
//! that show how much of a project's symbols have been seen by an LLM agent.

use crate::symbols::{ProjectTree, SymbolNode};
use crate::tracking::{ContextLedger, ReadDepth};

/// Per-file coverage metrics.
#[derive(Debug, Clone)]
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

/// Complete coverage report for a project.
#[derive(Debug, Clone)]
pub struct CoverageReport {
    /// Session ID if available.
    pub session_id: Option<String>,
    /// Per-file coverage metrics.
    pub files: Vec<FileCoverage>,
}

impl CoverageReport {
    /// Build a coverage report from a project tree and context ledger.
    pub fn from_project(project_tree: &ProjectTree, ledger: &ContextLedger) -> Self {
        let mut files: Vec<FileCoverage> = project_tree
            .files
            .iter()
            .map(|file| {
                let path = file.file_path.to_string_lossy().to_string();
                let (total, seen, full) = count_symbols(&file.symbols, ledger);
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
            files,
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
pub fn count_symbols(symbols: &[SymbolNode], ledger: &ContextLedger) -> (usize, usize, usize) {
    let mut total = 0;
    let mut seen = 0;
    let mut full = 0;

    for sym in symbols {
        total += 1;
        let depth = ledger.depth_of(&sym.id);

        if depth.is_seen() {
            seen += 1;
        }
        if depth == ReadDepth::FullBody {
            full += 1;
        }

        // Recurse into children
        let (child_total, child_seen, child_full) = count_symbols(&sym.children, ledger);
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
            .as_ref()
            .map(|s| s.as_str())
            .unwrap_or("none");
        output.push_str(&format!("Coverage Report (session: {})\n", session_str));

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

        output
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
        assert_eq!(count_symbols(&[], &ledger), (0, 0, 0));
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

        let (total, seen, full) = count_symbols(&[parent], &ledger);
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

        let report = CoverageReport::from_project(&tree, &ledger);
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

        let report = CoverageReport::from_project(&tree, &ledger);
        assert_eq!(report.total_symbols(), 3);
        assert_eq!(report.total_seen(), 2);
        assert_eq!(report.total_full(), 1);
    }

    #[test]
    fn text_formatter_output() {
        let report = CoverageReport { session_id: Some("abc-123".into()), files: vec![
            FileCoverage { path: "src/main.rs".into(), total_symbols: 10, seen_count: 8, full_count: 5 },
        ]};
        let formatter = TextFormatter::default();
        let output = formatter.format(&report);
        assert!(output.contains("Coverage Report (session: abc-123)"));
        assert!(output.contains("src/main.rs"));
        assert!(output.contains("TOTAL"));
    }
}
