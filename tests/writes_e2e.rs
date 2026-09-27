//! End to end: a session log's write results → attribution → the journal
//! (spec phase 2), including the §9.6 canary: file contents never persist.

use std::path::Path;

use ambits::app::App;
use ambits::ingest::claude::parse_log_file;
use ambits::ingest::SessionEvent;
use ambits::parser::ParserRegistry;
use ambits::symbols::ProjectTree;

const CANARY: &str = "CANARY_e2e_91d2_must_not_persist";

fn log_lines(root: &Path) -> Vec<String> {
    let lib = root.join("src/lib.rs").display().to_string();
    let new = root.join("src/new.rs").display().to_string();
    let original = format!("fn alpha() {{\n    let a = \"{CANARY}\";\n}}\n");
    vec![
        serde_json::json!({"type":"assistant","sessionId":"sess","timestamp":"2026-09-27T10:00:00Z",
            "message":{"content":[{"type":"tool_use","id":"toolu_01edit","name":"Edit",
                "input":{"file_path":lib,"old_string":format!("\"{CANARY}\""),"new_string":"\"x\""}}]}}),
        serde_json::json!({"type":"user","timestamp":"2026-09-27T10:00:01Z",
            "message":{"content":[{"type":"tool_result","tool_use_id":"toolu_01edit","content":"ok"}]},
            "toolUseResult":{"filePath":lib,"oldString":format!("\"{CANARY}\""),"newString":"\"x\"",
                "originalFile":original,"replaceAll":false,"userModified":false,
                "structuredPatch":[{"oldStart":2,"oldLines":1,"newStart":2,"newLines":1,
                    "lines":[format!("-    let a = \"{CANARY}\";"),"+    let a = \"x\";"]}]}}),
        serde_json::json!({"type":"assistant","sessionId":"sess","timestamp":"2026-09-27T10:01:00Z",
            "message":{"content":[{"type":"tool_use","id":"toolu_02write","name":"Write",
                "input":{"file_path":new,"content":format!("fn beta() {{ \"{CANARY}\"; }}\n")}}]}}),
        serde_json::json!({"type":"user","timestamp":"2026-09-27T10:01:01Z",
            "message":{"content":[{"type":"tool_result","tool_use_id":"toolu_02write","content":"ok"}]},
            "toolUseResult":{"type":"create","filePath":new,"content":format!("fn beta() {{ \"{CANARY}\"; }}\n"),
                "structuredPatch":[],"originalFile":null}}),
    ]
    .into_iter()
    .map(|v| v.to_string())
    .collect()
}

#[test]
fn writes_flow_from_the_log_to_the_journal_without_file_contents() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    let log = root.join("sess.jsonl");
    std::fs::write(&log, log_lines(&root).join("\n") + "\n").unwrap();

    let mut app = App::new(ProjectTree { root: root.clone(), files: vec![] }, root.clone());
    app.set_session_id(Some("sess".into()));
    app.enable_journal("tree-sitter", std::time::Duration::ZERO);

    let registry = ParserRegistry::new();
    let mut calls = 0;
    for event in parse_log_file(&log, &ambits::ingest::tool_config::ToolMappingConfig::builtin().unwrap()) {
        match event {
            SessionEvent::ToolCall(tc) => {
                calls += 1;
                app.process_agent_event(tc);
            }
            SessionEvent::Write(w) => {
                let record = ambits::writes::build_record(&w, &root, &registry, true).expect("inside the project");
                app.record_write("sess", record);
            }
            _ => {}
        }
    }
    assert_eq!(calls, 2);
    assert_eq!(app.activity.len(), 2, "both write calls shown in the feed");

    let journal_dir = ambits::journal::journal_dir(&root);
    let contents = ambits::journal::read_journal_session(&journal_dir, "sess");
    assert_eq!(contents.writes.len(), 2);

    let edit = &contents.writes["toolu_01edit"];
    assert_eq!(edit.level, ambits::writes::Level::Symbol);
    assert_eq!(edit.syms.iter().map(|(s, _)| s.as_str()).collect::<Vec<_>>(), vec!["src/lib.rs::alpha"]);

    let create = &contents.writes["toolu_02write"];
    assert_eq!(create.level, ambits::writes::Level::Symbol);
    assert_eq!(create.syms[0].0, "src/new.rs::beta");
    assert!(create.fh.is_some());

    // Spec §9.6: not a byte of any file's contents reaches the journal.
    for entry in std::fs::read_dir(&journal_dir).unwrap().flatten() {
        let text = std::fs::read_to_string(entry.path()).unwrap();
        assert!(!text.contains(CANARY), "file contents leaked into {}", entry.path().display());
    }
}
