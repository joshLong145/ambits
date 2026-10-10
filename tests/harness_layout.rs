//! A session launched by an agent harness (the Claude Agent SDK) from a
//! workspace directory above the clone it works on: tool paths name the
//! clone, the session's `cwd` is its parent, and prompts carry no `origin`.

mod common;

use std::path::{Path, PathBuf};

use common::ambits;
use serde_json::{json, Value};

const SESSION: &str = "1b0cc9c2-14c1-46fa-a36e-8a948e95a1c1";

fn write_log(path: &Path, lines: &[Value]) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, lines.iter().map(|l| format!("{l}\n")).collect::<String>()).unwrap();
}

struct Layout {
    _dir: tempfile::TempDir,
    clone: PathBuf,
    logs: PathBuf,
}

/// `<workspace>/repo/src/a.rs`, and a session whose cwd is `<workspace>` that
/// reads `a.rs` and then edits `alpha`.
fn layout() -> Layout {
    let dir = tempfile::tempdir().unwrap();
    let workspace = dir.path().canonicalize().unwrap().join("workspace");
    let clone = workspace.join("repo");
    std::fs::create_dir_all(clone.join("src")).unwrap();
    std::fs::create_dir_all(clone.join(".git")).unwrap();
    let before = "fn alpha() {}\n\nfn beta() {}\n";
    let after = "fn alpha() { let _x = 1; }\n\nfn beta() {}\n";
    std::fs::write(clone.join("src/a.rs"), after).unwrap();
    let file = clone.join("src/a.rs").to_string_lossy().into_owned();
    let cwd = workspace.to_string_lossy().into_owned();
    let base = |at: &str| json!({"sessionId": SESSION, "cwd": cwd, "entrypoint": "sdk-py", "timestamp": at});
    let merge = |mut a: Value, b: Value| {
        a.as_object_mut().unwrap().extend(b.as_object().unwrap().clone());
        a
    };
    let logs = dir.path().join("logs");
    write_log(
        &logs.join(format!("{SESSION}.jsonl")),
        &[
            merge(base("2026-10-10T10:00:00.000Z"), json!({"type": "user", "message": {"role": "user", "content": "## Runtime Context\n\n- Issue: X-1"}})),
            merge(base("2026-10-10T10:00:01.000Z"), json!({"type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_r", "name": "Read", "input": {"file_path": file}}]}})),
            merge(base("2026-10-10T10:00:01.200Z"), json!({"type": "user", "toolUseResult": {},
                "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_r", "content": "ok"}]}})),
            merge(base("2026-10-10T10:00:02.000Z"), json!({"type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_e", "name": "Edit", "input": {"file_path": file, "old_string": "fn alpha() {}", "new_string": "fn alpha() { let _x = 1; }"}}]}})),
            merge(base("2026-10-10T10:00:02.300Z"), json!({"type": "user",
                "toolUseResult": {"filePath": file, "oldString": "fn alpha() {}", "newString": "fn alpha() { let _x = 1; }",
                    "originalFile": before, "replaceAll": false, "userModified": false,
                    "structuredPatch": [{"oldStart": 1, "oldLines": 1, "newStart": 1, "newLines": 1,
                        "lines": ["-fn alpha() {}", "+fn alpha() { let _x = 1; }"]}]},
                "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_e", "content": "ok"}]}})),
        ],
    );
    Layout { _dir: dir, clone, logs }
}

fn run(l: &Layout, args: &[&str]) -> Value {
    let logs = l.logs.to_string_lossy().into_owned();
    let clone = l.clone.to_string_lossy().into_owned();
    let out = ambits(&l.clone)
        .args([&["-p", clone.as_str(), "--log-dir", logs.as_str(), "--session", SESSION], args].concat())
        .output()
        .unwrap();
    assert!(out.status.success(), "ambits {args:?}: {}", String::from_utf8_lossy(&out.stderr));
    serde_json::from_slice(&out.stdout).expect("JSON output")
}

fn spans(trace: &Value) -> Vec<&Value> {
    trace["resourceSpans"][0]["scopeSpans"][0]["spans"].as_array().unwrap().iter().collect()
}

fn attr<'a>(span: &'a Value, key: &str) -> Option<&'a Value> {
    span["attributes"].as_array()?.iter().find(|a| a["key"] == key).map(|a| &a["value"])
}

#[test]
fn tool_paths_are_relative_to_the_clone_not_the_sessions_cwd() {
    let l = layout();
    let trace = run(&l, &["trace"]);
    let paths: Vec<&str> = spans(&trace)
        .iter()
        .filter_map(|s| attr(s, "code.file.path")?["stringValue"].as_str())
        .collect();
    assert_eq!(paths, ["src/a.rs", "src/a.rs"]);
}

#[test]
fn a_prompt_with_no_origin_still_roots_its_calls() {
    let l = layout();
    let trace = run(&l, &["trace"]);
    let prompt = spans(&trace).into_iter().find(|s| attr(s, "ambits.prompt").is_some()).expect("a prompt span");
    assert!(prompt["name"].as_str().unwrap().starts_with("## Runtime Context"));
}

#[test]
fn reads_in_the_clone_earn_coverage() {
    let l = layout();
    let coverage = run(&l, &["--coverage", "--format", "json"]);
    assert_eq!(coverage["totals"]["full"], 2, "{coverage}");
}
