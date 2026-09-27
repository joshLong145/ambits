//! Process-level test for `ambits touched`: dispatch, text and JSON output.

use std::process::Command;

#[test]
fn touched_reports_the_last_write_as_text_and_json() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    std::fs::create_dir_all(root.join(".git")).unwrap();
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(root.join("src/lib.rs"), "fn alpha() {}\n").unwrap();
    std::fs::create_dir_all(root.join(".ambits/coverage")).unwrap();
    std::fs::write(
        root.join(".ambits/coverage/s1.ndjson"),
        r#"{"kind":"write","op":"toolu_1","av":1,"a":"agent-1","t":"2026-09-26T10:00:00Z","tool":"Edit","file":"src/lib.rs","level":"file"}
"#,
    )
    .unwrap();

    let run = |args: &[&str]| {
        let out = Command::new(env!("CARGO_BIN_EXE_ambits"))
            .current_dir(root)
            .env("HOME", root)
            .args(args)
            .output()
            .unwrap();
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        String::from_utf8(out.stdout).unwrap()
    };

    let text = run(&["touched", "src/lib.rs"]);
    assert!(text.contains("last written 2026-09-26T10:00:00Z by agent-1 (Edit)"), "{text}");
    assert!(text.contains("unknown"), "a file-level Edit has no hash: {text}");

    let json: serde_json::Value = serde_json::from_str(&run(&["touched", "--format", "json", "src/lib.rs"])).unwrap();
    assert_eq!(json["last_write"]["op"], "toolu_1");
    assert_eq!(json["last_write"]["session"], "s1");
    assert_eq!(json["last_write"]["status"], "unknown");

    let none = run(&["touched", "src/other.rs"]);
    assert!(none.contains("no agent writes recorded"), "{none}");
}
