//! End-to-end integration tests for the coverage pipeline.
//!
//! Each test exercises the full path: JSONL → parse → ledger → CoverageReport.

use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

use ambits::app::App;
use ambits::coverage::{CoverageFormatter, CoverageReport, JsonFormatter, TextFormatter};
use ambits::ingest::claude::parse_log_file;
use ambits::ingest::tool_config::ToolMappingConfig;
use ambits::ingest::{AgentToolCall, SessionEvent};
use ambits::symbols::merkle::content_hash;
use ambits::symbols::{FileSymbols, ProjectTree, SymbolCategory, SymbolNode};
use ambits::tracking::{ContextLedger, ReadDepth};
use tempfile::NamedTempFile;

/// Drain `SessionEvent::ToolCall` events into a flat `Vec<AgentToolCall>` so
/// existing assertions that iterate raw tool calls keep working.
fn tool_calls(events: Vec<SessionEvent>) -> Vec<AgentToolCall> {
    events
        .into_iter()
        .filter_map(|e| match e {
            SessionEvent::ToolCall(tc) => Some(tc),
            _ => None,
        })
        .collect()
}

fn builtin_cfg() -> ToolMappingConfig {
    ToolMappingConfig::builtin().expect("built-in config must parse")
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn sym(id: &str, name: &str) -> SymbolNode {
    let hash = content_hash(name);
    SymbolNode {
        id: id.to_string(),
        name: Arc::from(name),
        category: SymbolCategory::Function,
        label: "fn",
        file_path: Arc::new(PathBuf::new()),
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
    let file_path_arc = Arc::new(file_path.clone());
    let symbols = symbols
        .into_iter()
        .map(|mut s| {
            s.file_path = Arc::clone(&file_path_arc);
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
    let events = tool_calls(parse_log_file(tmp.path(), &builtin_cfg()));
    assert_eq!(events.len(), 1);

    for event in events {
        app.process_agent_event(event);
    }

    let report = CoverageReport::from_project(&app.project_tree, &app.ledger, None);
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
    let events = tool_calls(parse_log_file(tmp.path(), &builtin_cfg()));
    for event in events {
        app.process_agent_event(event);
    }

    assert_eq!(app.ledger.depth_of("mock/f.rs::alpha"), ReadDepth::Unseen);
    assert_eq!(app.ledger.depth_of("mock/f.rs::beta"), ReadDepth::FullBody);

    let report = CoverageReport::from_project(&app.project_tree, &app.ledger, None);
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
    let events = tool_calls(parse_log_file(tmp.path(), &builtin_cfg()));
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
    let mut events: Vec<AgentToolCall> = tool_calls(parse_log_file(tmp.path(), &builtin_cfg()));
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
    let events = parse_log_file(tmp.path(), &builtin_cfg());
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

    let report = CoverageReport::from_project(&tree, &ledger, None);
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

    // Content changed — flagged stale, depth retained.
    ledger.mark_stale_if_changed("s1", h2);
    assert!(ledger.is_stale("s1"));
    assert_eq!(ledger.depth_of("s1"), ReadDepth::FullBody);

    // Stale still counts as "seen" in coverage.
    let sym = sym("s1", "s1");
    let (total, seen, full) = ambits::coverage::count_symbols(&[sym], &ledger, None);
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

    let mut report = CoverageReport::from_project(&tree, &ledger, None);
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

/// JsonFormatter emits the documented schema with totals, files, and schema_version.
#[test]
fn json_formatter_structure() {
    let files = vec![
        file("mock/main.rs", vec![sym("mock/main.rs::main", "main")]),
    ];
    let tree = project(files);
    let mut ledger = ContextLedger::new();
    ledger.record(
        "mock/main.rs::main".into(),
        ReadDepth::FullBody,
        [0; 32],
        "ag".into(),
        10,
    );

    let mut report = CoverageReport::from_project(&tree, &ledger, None);
    report.session_id = Some("test-session-456".into());

    let output = JsonFormatter.format(&report);
    let v: serde_json::Value =
        serde_json::from_str(output.trim()).expect("output must be valid JSON");

    assert_eq!(v["schema_version"], 2);
    assert_eq!(v["session_id"], "test-session-456");
    assert!(v["agent_id"].is_null());
    assert_eq!(v["totals"]["symbols"], 1);
    assert_eq!(v["totals"]["seen"], 1);
    assert_eq!(v["totals"]["full"], 1);
    assert_eq!(v["totals"]["seen_percent"], 100.0);
    assert_eq!(v["totals"]["full_percent"], 100.0);

    let files = v["files"].as_array().expect("files must be an array");
    assert_eq!(files.len(), 1);
    assert_eq!(files[0]["path"], "mock/main.rs");
    assert_eq!(files[0]["total_symbols"], 1);
    assert_eq!(files[0]["seen_count"], 1);
    assert_eq!(files[0]["full_count"], 1);
    assert_eq!(files[0]["full_percent"], 100.0);
}

/// agent_id is null without a filter and the resolved id when filtered.
#[test]
fn json_formatter_agent_filter() {
    let files = vec![file("a.rs", vec![sym("a.rs::x", "x")])];
    let tree = project(files);
    let mut ledger = ContextLedger::new();
    ledger.record("a.rs::x".into(), ReadDepth::FullBody, [0; 32], "agent-foo".into(), 10);

    let report_no_filter = CoverageReport::from_project(&tree, &ledger, None);
    let v: serde_json::Value =
        serde_json::from_str(JsonFormatter.format(&report_no_filter).trim()).unwrap();
    assert!(v["agent_id"].is_null());

    let report_filtered = CoverageReport::from_project(&tree, &ledger, Some("agent-foo"));
    let v: serde_json::Value =
        serde_json::from_str(JsonFormatter.format(&report_filtered).trim()).unwrap();
    assert_eq!(v["agent_id"], "agent-foo");
}

/// Empty project serializes with zero totals and an empty files array.
#[test]
fn json_formatter_empty_project() {
    let tree = project(vec![]);
    let ledger = ContextLedger::new();
    let report = CoverageReport::from_project(&tree, &ledger, None);

    let output = JsonFormatter.format(&report);
    let v: serde_json::Value = serde_json::from_str(output.trim()).unwrap();

    assert_eq!(v["schema_version"], 2);
    assert_eq!(v["totals"]["symbols"], 0);
    assert_eq!(v["totals"]["seen"], 0);
    assert_eq!(v["totals"]["full"], 0);
    assert_eq!(v["files"].as_array().unwrap().len(), 0);
}

// ---------------------------------------------------------------------------
// Coverage journal: App -> disk -> back
// ---------------------------------------------------------------------------

/// Drive a real `App` through tool calls and assert the journal on disk
/// reflects exactly the symbols it read.
#[test]
fn journal_records_symbols_read_through_app() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_path_buf();

    let tree = ProjectTree {
        root: root.clone(),
        files: vec![file("a.rs", vec![sym("a.rs::x", "x"), sym("a.rs::y", "y")])],
    };
    let mut app = App::new(tree, root.clone(), None);
    app.set_session_id(Some("sess-1".into()));
    app.enable_journal("tree-sitter", std::time::Duration::from_millis(0));

    let mut call = AgentToolCall {
        agent_id: "ag".into(),
        tool_name: "Read".into(),
        file_path: Some(root.join("a.rs")),
        read_depth: ReadDepth::FullBody,
        description: String::new(),
        timestamp_str: "t".into(),
        target_symbol: None,
        target_lines: None,
        target_selectors: Vec::new(),
        label: "ag".into(),
    };
    app.process_agent_event(call.clone());
    app.sync_journal();

    let path = root.join(ambits::journal::JOURNAL_SUBDIR).join("sess-1.ndjson");
    let contents = ambits::journal::read_journal(&path);
    assert!(contents.warnings.is_empty(), "{:?}", contents.warnings);
    assert!(contents.header.is_some(), "header written");
    let mut ids: Vec<&String> = contents.reads.keys().collect();
    ids.sort();
    assert_eq!(ids, vec!["a.rs::x", "a.rs::y"]);

    // Re-reading unchanged content is not a read-set change.
    let before = std::fs::read_to_string(&path).unwrap();
    call.timestamp_str = "t2".into();
    app.process_agent_event(call);
    app.sync_journal();
    assert_eq!(std::fs::read_to_string(&path).unwrap(), before);
}

/// A symbol read before a compaction stays in the journal: compaction changes
/// what the *model* retains, not whether the file still looks the way it did
/// when it was read.
#[test]
fn journal_survives_compaction() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_path_buf();

    let tree = ProjectTree {
        root: root.clone(),
        files: vec![file("a.rs", vec![sym("a.rs::x", "x")])],
    };
    let mut app = App::new(tree, root.clone(), None);
    app.set_session_id(Some("sess-2".into()));
    app.enable_journal("tree-sitter", std::time::Duration::from_millis(0));

    app.process_agent_event(AgentToolCall {
        agent_id: "ag".into(),
        tool_name: "Read".into(),
        file_path: Some(root.join("a.rs")),
        read_depth: ReadDepth::FullBody,
        description: String::new(),
        timestamp_str: "t".into(),
        target_symbol: None,
        target_lines: None,
        target_selectors: Vec::new(),
        label: "ag".into(),
    });
    app.process_compaction("summary".into(), "ts".into(), "ag".into(), None);
    app.sync_journal();

    let path = root.join(ambits::journal::JOURNAL_SUBDIR).join("sess-2.ndjson");
    let contents = ambits::journal::read_journal(&path);
    assert_eq!(contents.reads.len(), 1, "pre-compaction read is still recorded");
    assert!(app.ledger.is_restored("a.rs::x"));
}

// ---------------------------------------------------------------------------
// Journal -> restore round trip
// ---------------------------------------------------------------------------

fn read_call(root: &std::path::Path, rel: &str) -> AgentToolCall {
    AgentToolCall {
        agent_id: "ag".into(),
        tool_name: "Read".into(),
        file_path: Some(root.join(rel)),
        read_depth: ReadDepth::FullBody,
        description: String::new(),
        timestamp_str: "t".into(),
        target_symbol: None,
        target_lines: None,
        target_selectors: Vec::new(),
        label: "ag".into(),
    }
}

/// The anchor property: with nothing changed on disk, everything the ledger
/// saw comes back through the journal. If this holds, the on-disk format
/// faithfully represents the in-memory state.
#[test]
fn journal_round_trips_to_the_ledgers_seen_set() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_path_buf();

    let tree = ProjectTree {
        root: root.clone(),
        files: vec![
            file("a.rs", vec![sym("a.rs::x", "x"), sym("a.rs::y", "y")]),
            file("b.rs", vec![sym("b.rs::z", "z")]),
            file("untouched.rs", vec![sym("untouched.rs::q", "q")]),
        ],
    };

    let mut app = App::new(tree.clone(), root.clone(), None);
    app.set_session_id(Some("rt".into()));
    app.enable_journal("tree-sitter", std::time::Duration::from_millis(0));
    app.process_agent_event(read_call(&root, "a.rs"));
    app.process_agent_event(read_call(&root, "b.rs"));
    app.sync_journal();

    let path = root.join(ambits::journal::JOURNAL_SUBDIR).join("rt.ndjson");
    let contents = ambits::journal::read_journal(&path);
    assert!(contents.warnings.is_empty(), "{:?}", contents.warnings);

    let outcome = ambits::restore::classify(&contents.reads, &tree);

    let mut restored: Vec<&str> = outcome.restored.iter().map(|s| s.symbol_id.as_str()).collect();
    restored.sort();
    let mut expected: Vec<&str> = app
        .ledger
        .entries
        .values()
        .filter(|e| e.depth.is_seen() && !e.stale)
        .map(|e| e.symbol_id.as_str())
        .collect();
    expected.sort();

    assert_eq!(restored, expected, "journal round-trips the ledger's seen set");
    assert!(outcome.drifted.is_empty() && outcome.removed.is_empty());
    // A file nobody read is simply absent, not "omitted".
    assert!(!restored.contains(&"untouched.rs::q"));
}

/// Editing a symbol after it was read must withhold *that symbol* on restore,
/// while its unedited neighbours survive. This is the behavior the whole
/// design exists to produce.
#[test]
fn edited_symbols_are_withheld_but_neighbours_survive() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_path_buf();

    let tree = ProjectTree {
        root: root.clone(),
        files: vec![file("a.rs", vec![sym("a.rs::kept", "kept"), sym("a.rs::edited", "edited")])],
    };

    let mut app = App::new(tree.clone(), root.clone(), None);
    app.set_session_id(Some("drift".into()));
    app.enable_journal("tree-sitter", std::time::Duration::from_millis(0));
    app.process_agent_event(read_call(&root, "a.rs"));
    app.sync_journal();

    // Simulate re-scanning after one function was edited.
    let mut rescanned = tree.clone();
    rescanned.files[0].symbols[1].content_hash = content_hash("a brand new body");

    let path = root.join(ambits::journal::JOURNAL_SUBDIR).join("drift.ndjson");
    let contents = ambits::journal::read_journal(&path);
    let outcome = ambits::restore::classify(&contents.reads, &rescanned);

    assert_eq!(outcome.restored.len(), 1);
    assert_eq!(outcome.restored[0].symbol_id, "a.rs::kept");
    assert_eq!(outcome.drifted.len(), 1);
    assert_eq!(outcome.drifted[0].symbol_id, "a.rs::edited");
    assert!(outcome.removed.is_empty());
}

/// A deleted symbol is reported as removed rather than silently vanishing, so
/// a digest can tell the agent what it must no longer assume.
#[test]
fn deleted_symbols_are_reported_as_removed() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_path_buf();

    let tree = ProjectTree {
        root: root.clone(),
        files: vec![file("a.rs", vec![sym("a.rs::doomed", "doomed")])],
    };

    let mut app = App::new(tree.clone(), root.clone(), None);
    app.set_session_id(Some("gone".into()));
    app.enable_journal("tree-sitter", std::time::Duration::from_millis(0));
    app.process_agent_event(read_call(&root, "a.rs"));
    app.sync_journal();

    let rescanned = ProjectTree {
        root: root.clone(),
        files: vec![file("a.rs", vec![])],
    };

    let path = root.join(ambits::journal::JOURNAL_SUBDIR).join("gone.ndjson");
    let contents = ambits::journal::read_journal(&path);
    let outcome = ambits::restore::classify(&contents.reads, &rescanned);

    assert!(outcome.restored.is_empty());
    assert_eq!(outcome.removed.len(), 1);
    assert_eq!(outcome.removed[0].symbol_id, "a.rs::doomed");
    assert_eq!(outcome.omitted_files(), vec![std::path::Path::new("a.rs")]);
}
