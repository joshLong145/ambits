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
            merge(base("2026-10-10T10:00:01.200Z"), json!({"type": "user",
                "toolUseResult": {"type": "text", "file": {"filePath": file, "content": before, "startLine": 1}},
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

fn sync(l: &Layout, state: &Path, extra: &[&str]) -> std::process::Output {
    let logs = l.logs.to_string_lossy().into_owned();
    let clone = l.clone.to_string_lossy().into_owned();
    let state = state.to_string_lossy().into_owned();
    ambits(&l.clone)
        .args([&["-p", clone.as_str(), "--log-dir", logs.as_str(), "--session", SESSION, "--state-dir", state.as_str()], extra, &["cache", "sync", "--format", "json"]].concat())
        .output()
        .unwrap()
}

#[test]
fn cache_sync_journals_reads_and_symbol_level_writes_without_a_terminal() {
    let l = layout();
    let state = l.logs.with_file_name("state");
    let out = sync(&l, &state, &[]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let summary: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!((summary["symbols_read"].as_u64(), summary["writes"].as_u64()), (Some(2), Some(1)), "{summary}");

    let journal = std::fs::read_to_string(state.join(format!("coverage/{SESSION}.ndjson"))).unwrap();
    let write: Value = journal.lines().map(|l| serde_json::from_str::<Value>(l).unwrap()).find(|r| r["kind"] == "write").expect("a write record");
    assert_eq!(write["level"], "symbol");
    assert_eq!(write["syms"][0][0], "src/a.rs::alpha", "{write}");
    assert!(!l.clone.join(".ambits").exists());
}

#[test]
fn cache_sync_twice_appends_nothing_the_second_time() {
    let l = layout();
    let state = l.logs.with_file_name("state");
    assert!(sync(&l, &state, &[]).status.success());
    let journal = state.join(format!("coverage/{SESSION}.ndjson"));
    let first = std::fs::read_to_string(&journal).unwrap();
    assert!(sync(&l, &state, &[]).status.success());
    assert_eq!(std::fs::read_to_string(&journal).unwrap(), first);
}

#[test]
fn cache_sync_refuses_when_journaling_is_off() {
    let l = layout();
    let out = sync(&l, &l.logs.with_file_name("state"), &["--no-journal"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("--no-journal"));
}

#[test]
fn cache_sync_refuses_a_session_with_no_logs() {
    let l = layout();
    std::fs::remove_file(l.logs.join(format!("{SESSION}.jsonl"))).unwrap();
    let out = sync(&l, &l.logs.with_file_name("state"), &[]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("no logs for session"));
}

#[test]
fn cache_sync_fails_when_the_journal_cannot_be_written() {
    let l = layout();
    let blocked = l.logs.with_file_name("not-a-dir");
    std::fs::write(&blocked, "a file where the state directory should be").unwrap();
    let out = sync(&l, &blocked, &[]);
    assert!(!out.status.success(), "exited 0 having journaled nothing");
    assert!(String::from_utf8_lossy(&out.stderr).contains("cannot write the journal"));
}

#[test]
fn cache_sync_credits_a_subagent_nested_under_a_workflow() {
    let l = layout();
    let file = l.clone.join("src/a.rs").to_string_lossy().into_owned();
    let line = |at: &str, body: Value| {
        let mut v = json!({"sessionId": SESSION, "agentId": "wf1", "timestamp": at});
        v.as_object_mut().unwrap().extend(body.as_object().unwrap().clone());
        v
    };
    write_log(
        &l.logs.join(SESSION).join("subagents/workflows/run-1/agent-wf1.jsonl"),
        &[
            line("2026-10-10T10:00:03.000Z", json!({"type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_w", "name": "Read", "input": {"file_path": file}}]}})),
            line("2026-10-10T10:00:03.100Z", json!({"type": "user", "toolUseResult": {},
                "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_w", "content": "ok"}]}})),
        ],
    );
    let state = l.logs.with_file_name("state");
    assert!(sync(&l, &state, &[]).status.success());
    let journal = std::fs::read_to_string(state.join(format!("coverage/{SESSION}.ndjson"))).unwrap();
    assert!(
        journal.lines().map(|l| serde_json::from_str::<Value>(l).unwrap()).any(|r| r["kind"] == "read" && r["a"] == "wf1"),
        "no read credited to the nested agent:\n{journal}"
    );
}

fn report(l: &Layout, extra: &[&str]) -> Value {
    run(l, &[&["--state-dir", l.logs.with_file_name("state").to_str().unwrap()], &["report"][..], extra].concat())
}

#[test]
fn the_report_attributes_the_edit_to_its_symbol_and_its_span() {
    let l = layout();
    let r = report(&l, &[]);
    assert_eq!(r["schema_version"], 1);
    assert_eq!(r["project"]["name"], "repo");
    let write = &r["writes"][0];
    assert_eq!(write["symbols"], json!([{"id": "src/a.rs::alpha", "status": "current"}]), "{write}");
    let span = &r["spans"][write["span"].as_u64().unwrap() as usize];
    assert_eq!((span["kind"].as_str(), span["tool"].as_str()), (Some("write"), Some("Edit")));
    let prompt = r["spans"].as_array().unwrap().iter().find(|s| s["kind"] == "prompt").unwrap();
    assert_eq!(span["parent"], prompt["i"], "the edit nests under the prompt that started its turn");
}

#[test]
fn the_report_lists_each_symbol_with_who_read_it_and_which_writes_touched_it() {
    let l = layout();
    let r = report(&l, &[]);
    let file = r["files"].as_array().unwrap().iter().find(|f| f["path"] == "src/a.rs").unwrap();
    let alpha = file["symbols"].as_array().unwrap().iter().find(|s| s["id"] == "src/a.rs::alpha").unwrap();
    assert_eq!(alpha["reads"][SESSION], "full_body");
    assert_eq!(alpha["writes"], json!([r["writes"][0]["op"]]));
    let read = r["spans"].as_array().unwrap().iter().find(|s| s["tool"] == "Read").unwrap();
    assert_eq!(read["read_count"], 2, "{read}");
    assert_eq!(r["totals"]["symbols_written"], 1);
}

#[test]
fn the_report_writes_no_state() {
    let l = layout();
    report(&l, &[]);
    assert!(!l.logs.with_file_name("state").exists());
    assert!(!l.clone.join(".ambits").exists());
}

#[test]
fn the_report_carries_no_absolute_path() {
    let l = layout();
    let text = report(&l, &[]).to_string();
    let root = l.clone.parent().unwrap().to_string_lossy().into_owned();
    // A call's own description is the one field that may name a path; its
    // path arguments are made project-relative.
    let without_details: Value = {
        let mut r: Value = serde_json::from_str(&text).unwrap();
        for s in r["spans"].as_array_mut().unwrap() {
            s["detail"] = Value::Null;
        }
        r
    };
    assert!(!without_details.to_string().contains(&root), "{without_details}");
}

#[test]
fn max_bytes_cuts_in_order_and_says_so() {
    let l = layout();
    let full = report(&l, &[]);
    let r = report(&l, &["--max-bytes", "10"]);
    assert_eq!(r["truncated"], json!(["unwritten_file_symbols", "read_text", "span_reads", "arguments", "diffs", "other_spans"]));
    assert_eq!(r["writes"], full["writes"], "writes are never cut");
    let read = r["spans"].as_array().unwrap().iter().find(|s| s["tool"] == "Read").unwrap();
    assert_eq!((read["read"].clone(), read["read_count"].clone()), (json!([]), json!(2)));
    assert!(report(&l, &["--max-bytes", "100000000"])["truncated"].as_array().unwrap().is_empty());
}

#[test]
fn the_report_folds_in_writes_already_journaled() {
    let l = layout();
    let state = l.logs.with_file_name("state");
    assert!(sync(&l, &state, &[]).status.success());
    std::fs::remove_file(l.logs.join(format!("{SESSION}.jsonl"))).unwrap();
    std::fs::write(l.logs.join(format!("{SESSION}.jsonl")), "{\"type\":\"user\",\"sessionId\":\"x\"}\n").unwrap();
    let r = report(&l, &[]);
    assert_eq!(r["writes"][0]["symbols"][0]["id"], "src/a.rs::alpha", "the journal's write survives a log that lost it");
}

fn embedded_report(page: &str) -> Value {
    let open = "<script id=\"report-data\" type=\"application/json\">";
    let start = page.find(open).expect("the data element") + open.len();
    let end = start + page[start..].find("</script>").unwrap();
    serde_json::from_str(&page[start..end]).expect("the embedded report parses")
}

fn report_page(l: &Layout, args: &[&str]) -> std::process::Output {
    let logs = l.logs.to_string_lossy().into_owned();
    let clone = l.clone.to_string_lossy().into_owned();
    ambits(&l.clone).args([&["-p", clone.as_str(), "--log-dir", logs.as_str(), "--session", SESSION, "report"], args].concat()).output().unwrap()
}

#[test]
fn the_html_report_embeds_the_json_report_under_a_closed_policy() {
    let l = layout();
    let out = report_page(&l, &["--format", "html"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let page = String::from_utf8(out.stdout).unwrap();
    assert!(page.starts_with("<!doctype html>"));
    assert!(page.contains("default-src 'none'"), "the page must load nothing");
    assert!(!page.contains("{{"), "a placeholder was left unfilled");
    let mut embedded = embedded_report(&page);
    let mut json = report(&l, &[]);
    for r in [&mut embedded, &mut json] {
        r["generated_at"] = Value::Null;
    }
    assert_eq!(embedded, json);
}

#[test]
fn from_json_renders_a_screened_report_without_the_session() {
    let l = layout();
    let mut r = report(&l, &[]);
    r["spans"][0]["detail"] = json!("[REDACTED]");
    let file = l.logs.with_file_name("screened.json");
    std::fs::write(&file, r.to_string()).unwrap();
    std::fs::remove_dir_all(&l.logs).unwrap();
    let out = ambits(&l.clone).args(["report", "--format", "html", "--from-json", file.to_str().unwrap()]).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(embedded_report(&String::from_utf8(out.stdout).unwrap()), r, "rendered exactly as given");
}

#[test]
fn from_json_needs_the_html_format_and_takes_no_size_limit() {
    let l = layout();
    let file = l.logs.with_file_name("r.json");
    std::fs::write(&file, report(&l, &[]).to_string()).unwrap();
    let path = file.to_str().unwrap();
    let json_out = ambits(&l.clone).args(["report", "--from-json", path]).output().unwrap();
    assert!(!json_out.status.success());
    assert!(String::from_utf8_lossy(&json_out.stderr).contains("--format html"));
    let capped = ambits(&l.clone).args(["report", "--format", "html", "--from-json", path, "--max-bytes", "10"]).output().unwrap();
    assert!(!capped.status.success(), "--max-bytes cannot apply to a report built elsewhere");
}

#[test]
fn an_edit_span_carries_its_arguments_and_the_tools_own_patch() {
    let l = layout();
    let r = report(&l, &[]);
    let edit = r["spans"].as_array().unwrap().iter().find(|s| s["tool"] == "Edit").unwrap();
    let file_path = edit["args"].as_array().unwrap().iter().find(|a| a["key"] == "file_path").unwrap();
    assert_eq!(file_path["value"], "src/a.rs", "a path argument inside the project is relative");
    let new_string = edit["args"].as_array().unwrap().iter().find(|a| a["key"] == "new_string").unwrap();
    assert_eq!(new_string["bulk"], true, "file contents an argument wrote are summed up, not repeated");
    assert_eq!(
        edit["content"],
        json!({"type": "change", "exact": true, "cut": 0, "hunks": [{"old_start": 1, "new_start": 1,
            "lines": [["-", "fn alpha() {}"], ["+", "fn alpha() { let _x = 1; }"]]}]})
    );
}

#[test]
fn a_read_span_carries_the_text_it_returned() {
    let l = layout();
    let r = report(&l, &[]);
    let read = r["spans"].as_array().unwrap().iter().find(|s| s["tool"] == "Read").unwrap();
    assert_eq!(read["content"], json!({"type": "read", "start": 1, "lines": ["fn alpha() {}", "", "fn beta() {}"], "cut": 0}));
}

#[test]
fn a_prompt_span_carries_no_call_content() {
    let l = layout();
    let r = report(&l, &[]);
    let prompt = r["spans"].as_array().unwrap().iter().find(|s| s["kind"] == "prompt").unwrap();
    assert_eq!((prompt["args"].clone(), prompt["content"].clone()), (json!([]), Value::Null));
}

/// The session's log with every Edit result's `originalFile` taken out, as
/// Claude Code leaves it out for a file over ~10 KB.
fn without_original_file(l: &Layout) {
    let log = l.logs.join(format!("{SESSION}.jsonl"));
    let lines: Vec<String> = std::fs::read_to_string(&log)
        .unwrap()
        .lines()
        .map(|line| {
            let mut v: Value = serde_json::from_str(line).unwrap();
            if let Some(result) = v.get_mut("toolUseResult").and_then(Value::as_object_mut) {
                if result.contains_key("structuredPatch") {
                    result.insert("originalFile".into(), Value::Null);
                }
            }
            v.to_string()
        })
        .collect();
    std::fs::write(&log, lines.join("\n") + "\n").unwrap();
}

fn journaled_write(state: &Path) -> Value {
    let journal = std::fs::read_to_string(state.join(format!("coverage/{SESSION}.ndjson"))).unwrap();
    journal.lines().map(|l| serde_json::from_str::<Value>(l).unwrap()).find(|r| r["kind"] == "write").expect("a write record")
}

#[test]
fn an_edit_without_original_file_is_still_attributed_to_its_symbol() {
    let l = layout();
    without_original_file(&l);
    let state = l.logs.with_file_name("state");
    assert!(sync(&l, &state, &[]).status.success());
    let write = journaled_write(&state);
    assert_eq!((write["level"].as_str(), write["syms"][0][0].as_str()), (Some("symbol"), Some("src/a.rs::alpha")), "{write}");
}

#[test]
fn a_later_sessions_edit_to_the_same_file_is_undone_first() {
    const LATER: &str = "9d1e2f3a-4b5c-4d6e-8f70-112233445566";
    let l = layout();
    without_original_file(&l);
    // A second session, later, edits `alpha` again on the same line: the
    // first edit's added line is no longer on disk until this one is undone.
    let file = l.clone.join("src/a.rs");
    std::fs::write(&file, "fn alpha() { let _x = 2; }\n\nfn beta() {}\n").unwrap();
    let base = |at: &str| json!({"sessionId": LATER, "cwd": l.clone.parent().unwrap(), "timestamp": at});
    let merge = |mut a: Value, b: Value| {
        a.as_object_mut().unwrap().extend(b.as_object().unwrap().clone());
        a
    };
    let later_log = l.logs.join(format!("{LATER}.jsonl"));
    let path = file.to_string_lossy().into_owned();
    write_log(
        &later_log,
        &[
            merge(base("2026-10-10T11:00:00.000Z"), json!({"type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_later", "name": "Edit", "input": {"file_path": path, "old_string": "let _x = 1;", "new_string": "let _x = 2;"}}]}})),
            merge(base("2026-10-10T11:00:00.200Z"), json!({"type": "user",
                "toolUseResult": {"filePath": path, "oldString": "let _x = 1;", "newString": "let _x = 2;", "originalFile": null,
                    "replaceAll": false, "userModified": false,
                    "structuredPatch": [{"oldStart": 1, "oldLines": 1, "newStart": 1, "newLines": 1, "lines": ["-fn alpha() { let _x = 1; }", "+fn alpha() { let _x = 2; }"]}]},
                "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_later", "content": "ok"}]}})),
        ],
    );
    let later = std::time::SystemTime::now() + std::time::Duration::from_secs(60);
    std::fs::File::options().write(true).open(&later_log).unwrap().set_modified(later).unwrap();

    let state = l.logs.with_file_name("state");
    assert!(sync(&l, &state, &[]).status.success());
    let write = journaled_write(&state);
    assert_eq!((write["level"].as_str(), write["syms"][0][0].as_str()), (Some("symbol"), Some("src/a.rs::alpha")), "{write}");
    let r = report(&l, &[]);
    assert_eq!(r["writes"][0]["symbols"][0]["status"], "changed", "the later session changed alpha again");
}
