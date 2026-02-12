//! End-to-end integration tests for the coverage pipeline.
//!
//! Each test exercises the full path: JSONL → parse → ledger → CoverageReport.

use std::io::Write;
use std::path::PathBuf;

use ambits::app::App;
use ambits::coverage::{CoverageFormatter, CoverageReport, TextFormatter};
use ambits::ingest::claude::parse_log_file;
use ambits::ingest::AgentToolCall;
use ambits::symbols::merkle::content_hash;
use ambits::symbols::{FileSymbols, ProjectTree, SymbolCategory, SymbolNode};
use ambits::tracking::{ContextLedger, ReadDepth};
use tempfile::NamedTempFile;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn sym(id: &str, name: &str) -> SymbolNode {
    let hash = content_hash(name);
    SymbolNode {
        id: id.to_string(),
        name: name.to_string(),
        category: SymbolCategory::Function,
        label: "fn".to_string(),
        file_path: PathBuf::new(),
        byte_range: 0..100,
        line_range: 1..10,
        content_hash: hash,
        merkle_hash: hash,
        children: Vec::new(),
        estimated_tokens: 30,
    }
}

fn file(path: &str, symbols: Vec<SymbolNode>) -> FileSymbols {
    let file_path = PathBuf::from(path);
    let symbols = symbols
        .into_iter()
        .map(|mut s| {
            s.file_path = file_path.clone();
            s
        })
        .collect();
    FileSymbols {
        file_path,
        symbols,
        total_lines: 100,
    }
}

fn project(files: Vec<FileSymbols>) -> ProjectTree {
    ProjectTree {
        root: PathBuf::from("/test/project"),
        files,
    }
}

fn jsonl_read(file_path: &str) -> String {
    format!(
        r#"{{"type":"assistant","sessionId":"s1","timestamp":"2025-01-01T00:00:00Z","message":{{"role":"assistant","content":[{{"type":"tool_use","name":"mcp__acp__Read","input":{{"file_path":"{file_path}"}}}}]}}}}"#
    )
}

fn jsonl_find_symbol(relative_path: &str, name: &str, include_body: bool) -> String {
    format!(
        r#"{{"type":"assistant","sessionId":"s1","timestamp":"2025-01-01T00:00:00Z","message":{{"role":"assistant","content":[{{"type":"tool_use","name":"mcp__serena__find_symbol","input":{{"name_path_pattern":"{name}","relative_path":"{relative_path}","include_body":{include_body}}}}}]}}}}"#
    )
}

fn jsonl_grep(pattern: &str) -> String {
    format!(
        r#"{{"type":"assistant","sessionId":"s1","timestamp":"2025-01-01T00:00:00Z","message":{{"role":"assistant","content":[{{"type":"tool_use","name":"Grep","input":{{"pattern":"{pattern}"}}}}]}}}}"#
    )
}

fn write_jsonl(lines: &[String]) -> NamedTempFile {
    let mut tmp = NamedTempFile::new().unwrap();
    for line in lines {
        writeln!(tmp, "{}", line).unwrap();
    }
    tmp.flush().unwrap();
    tmp
}

fn make_app(files: Vec<FileSymbols>) -> App {
    let tree = project(files);
    App::new(tree, PathBuf::from("/test/project"), None)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Full pipeline: parse Read JSONL → update ledger → report shows 100% for read file, 0% for unread.
#[test]
fn full_pipeline_read() {
    let files = vec![
        file("mock/file_a.rs", vec![sym("mock/file_a.rs::foo", "foo"), sym("mock/file_a.rs::bar", "bar")]),
        file("mock/file_b.rs", vec![sym("mock/file_b.rs::baz", "baz")]),
    ];
    let mut app = make_app(files);

    // Parse a Read event for file_a (absolute path gets normalized).
    let tmp = write_jsonl(&[jsonl_read("/test/project/mock/file_a.rs")]);
    let events = parse_log_file(tmp.path());
    assert_eq!(events.len(), 1);

    for event in events {
        app.process_agent_event(event);
    }

    let report = CoverageReport::from_project(&app.project_tree, &app.ledger);
    // mock/file_a: 2 symbols, all FullBody → 100%. mock/file_b: 0%.
    let fa = report.files.iter().find(|f| f.path == "mock/file_a.rs").unwrap();
    assert_eq!(fa.full_count, 2);
    assert_eq!(fa.full_percent(), 100.0);

    let fb = report.files.iter().find(|f| f.path == "mock/file_b.rs").unwrap();
    assert_eq!(fb.full_count, 0);
    assert_eq!(fb.full_percent(), 0.0);
}

/// find_symbol with target → only the targeted symbol is marked, rest stays Unseen.
#[test]
fn targeted_symbol_partial() {
    let files = vec![
        file("mock/f.rs", vec![sym("mock/f.rs::alpha", "alpha"), sym("mock/f.rs::beta", "beta")]),
    ];
    let mut app = make_app(files);

    let tmp = write_jsonl(&[jsonl_find_symbol("mock/f.rs", "beta", true)]);
    let events = parse_log_file(tmp.path());
    for event in events {
        app.process_agent_event(event);
    }

    assert_eq!(app.ledger.depth_of("mock/f.rs::alpha"), ReadDepth::Unseen);
    assert_eq!(app.ledger.depth_of("mock/f.rs::beta"), ReadDepth::FullBody);

    let report = CoverageReport::from_project(&app.project_tree, &app.ledger);
    let f = report.files.iter().find(|f| f.path == "mock/f.rs").unwrap();
    assert_eq!(f.seen_count, 1);
    assert_eq!(f.full_count, 1);
    assert_eq!(f.total_symbols, 2);
}

/// Grep (NameOnly) then Read (FullBody) then Grep again → final depth stays FullBody.
#[test]
fn depth_upgrade_invariant() {
    let files = vec![
        file("mock/f.rs", vec![sym("mock/f.rs::x", "x")]),
    ];
    let mut app = make_app(files);

    let lines = vec![
        jsonl_grep("pattern"),                      // NameOnly, no file targeting
        jsonl_read("/test/project/mock/f.rs"),       // FullBody
        jsonl_grep("pattern"),                      // NameOnly again — must NOT downgrade
    ];
    let tmp = write_jsonl(&lines);
    let events = parse_log_file(tmp.path());
    for event in events {
        app.process_agent_event(event);
    }

    // Grep doesn't target a file, so only the Read sets depth.
    assert_eq!(app.ledger.depth_of("mock/f.rs::x"), ReadDepth::FullBody);
}

/// Events from two different agents → both tracked in agents_seen.
#[test]
fn multi_agent_session() {
    let files = vec![file("mock/f.rs", vec![sym("mock/f.rs::a", "a")])];
    let mut app = make_app(files);

    // Simulate two agents by modifying the agent_id after parsing.
    let tmp = write_jsonl(&[
        jsonl_read("/test/project/mock/f.rs"),
        jsonl_read("/test/project/mock/f.rs"),
    ]);
    let mut events: Vec<AgentToolCall> = parse_log_file(tmp.path());
    events[0].agent_id = "agent-alpha".into();
    events[1].agent_id = "agent-beta".into();

    for event in events {
        app.process_agent_event(event);
    }

    assert_eq!(app.agents_seen.len(), 2);
    assert!(app.agents_seen.contains(&"agent-alpha".to_string()));
    assert!(app.agents_seen.contains(&"agent-beta".to_string()));
}

/// Write JSONL to a temp file → parse_log_file() returns correct event count.
#[test]
fn parse_log_file_e2e() {
    let lines = vec![
        jsonl_read("/some/file.rs"),
        // A user message line that should be ignored:
        r#"{"type":"user","message":{"role":"user","content":"hello"}}"#.to_string(),
        jsonl_grep("foo"),
        jsonl_find_symbol("bar.rs", "baz", false),
    ];
    let tmp = write_jsonl(&lines);
    let events = parse_log_file(tmp.path());
    // User message is ignored; the other 3 produce events.
    assert_eq!(events.len(), 3);
}

/// Three files at 0%, 50%, 100% → sorted ascending by full_percent in report.
#[test]
fn coverage_sort_order() {
    let files = vec![
        file("mock/a.rs", vec![sym("mock/a.rs::a1", "a1"), sym("mock/a.rs::a2", "a2")]),
        file("mock/b.rs", vec![sym("mock/b.rs::b1", "b1"), sym("mock/b.rs::b2", "b2")]),
        file("mock/c.rs", vec![sym("mock/c.rs::c1", "c1")]),
    ];
    let tree = project(files);
    let mut ledger = ContextLedger::new();

    // mock/b.rs: 50% (1 of 2 full).
    ledger.record("mock/b.rs::b1".into(), ReadDepth::FullBody, [0; 32], "ag".into(), 10);
    // mock/c.rs: 100%.
    ledger.record("mock/c.rs::c1".into(), ReadDepth::FullBody, [0; 32], "ag".into(), 10);
    // mock/a.rs: 0%.

    let report = CoverageReport::from_project(&tree, &ledger);
    let paths: Vec<&str> = report.files.iter().map(|f| f.path.as_str()).collect();
    // Sorted ascending by full_percent: 0%, 50%, 100%.
    assert_eq!(paths, vec!["mock/a.rs", "mock/b.rs", "mock/c.rs"]);
}

/// Record a symbol → change its hash → mark_stale_if_changed → still counts as "seen" but stale.
#[test]
fn stale_detection() {
    let mut ledger = ContextLedger::new();
    let h1 = content_hash("version1");
    let h2 = content_hash("version2");

    ledger.record("s1".into(), ReadDepth::FullBody, h1, "ag".into(), 10);
    assert_eq!(ledger.depth_of("s1"), ReadDepth::FullBody);

    // Content changed — mark stale.
    ledger.mark_stale_if_changed("s1", h2);
    assert_eq!(ledger.depth_of("s1"), ReadDepth::Stale);

    // Stale still counts as "seen" in coverage.
    let sym = sym("s1", "s1");
    let (total, seen, full) = ambits::coverage::count_symbols(&[sym], &ledger);
    assert_eq!(total, 1);
    assert_eq!(seen, 1); // Stale is still "seen"
    assert_eq!(full, 0); // But not "full"
}

/// TextFormatter output contains expected structural elements.
#[test]
fn text_formatter_structure() {
    let files = vec![
        file("mock/main.rs", vec![sym("mock/main.rs::main", "main")]),
    ];
    let tree = project(files);
    let mut ledger = ContextLedger::new();
    ledger.record("mock/main.rs::main".into(), ReadDepth::FullBody, [0; 32], "ag".into(), 10);

    let mut report = CoverageReport::from_project(&tree, &ledger);
    report.session_id = Some("test-session-123".into());

    let formatter = TextFormatter::default();
    let output = formatter.format(&report);

    assert!(output.contains("test-session-123"), "should contain session id");
    assert!(output.contains("File"), "should contain header");
    assert!(output.contains("Symbols"), "should contain header");
    assert!(output.contains("mock/main.rs"), "should contain file path");
    assert!(output.contains("TOTAL"), "should contain total row");
    assert!(output.contains("100%"), "should show 100% for full coverage");
}
