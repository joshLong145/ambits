//! `ambits trace` end to end: a session's logs out as OTLP/JSON and as
//! Chrome trace events.

mod common;

use std::path::Path;

use common::run_ambits;
use serde_json::{json, Value};

const SESSION: &str = "0b7e9d3a-1c2f-4e5a-9b8c-7d6e5f4a3b2c";
const AGENT: &str = "ax1";

fn tool_use(agent: &str, at: &str, id: &str, name: &str, input: Value) -> Value {
    json!({"type": "assistant", "sessionId": SESSION, "agentId": agent, "timestamp": at,
           "message": {"role": "assistant", "content": [{"type": "tool_use", "id": id, "name": name, "input": input}]}})
}

fn tool_result(at: &str, id: &str, detail: Value) -> Value {
    json!({"type": "user", "sessionId": SESSION, "timestamp": at, "toolUseResult": detail,
           "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": id, "content": "ok"}]}})
}

fn write_log(path: &Path, lines: &[Value]) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, lines.iter().map(|l| format!("{l}\n")).collect::<String>()).unwrap();
}

/// A project with `src/a.rs`, and a session that reads it, then delegates
/// in the background to an agent that reads it too and stops at 10:00:30.
fn project() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(root.join("src/a.rs"), "fn alpha() {}\n").unwrap();
    let file = root.join("src/a.rs").to_string_lossy().into_owned();
    let logs = root.join("logs");
    write_log(
        &logs.join(format!("{SESSION}.jsonl")),
        &[
            tool_use(SESSION, "2026-09-27T10:00:00.000Z", "toolu_r", "Read", json!({"file_path": file})),
            tool_result("2026-09-27T10:00:00.250Z", "toolu_r", json!({})),
            tool_use(SESSION, "2026-09-27T10:00:01.000Z", "toolu_d", "Agent", json!({"description": "look around", "prompt": "p"})),
            tool_result("2026-09-27T10:00:01.100Z", "toolu_d", json!({"agentId": AGENT, "status": "async_launched"})),
            json!({"type": "queue-operation", "operation": "enqueue", "sessionId": SESSION, "timestamp": "2026-09-27T10:00:30.000Z",
                   "content": format!("<task-notification>\n<task-id>{AGENT}</task-id>\n<tool-use-id>toolu_d</tool-use-id>\n<status>completed</status>\n</task-notification>")}),
        ],
    );
    write_log(
        &logs.join(SESSION).join(format!("subagents/agent-{AGENT}.jsonl")),
        &[
            tool_use(AGENT, "2026-09-27T10:00:02.000Z", "toolu_x", "Read", json!({"file_path": file})),
            tool_result("2026-09-27T10:00:02.500Z", "toolu_x", json!({})),
        ],
    );
    (dir, root)
}

fn trace(root: &Path, format: &str) -> Value {
    let logs = root.join("logs").to_string_lossy().into_owned();
    let out = run_ambits(root, &["--log-dir", &logs, "--session", SESSION, "trace", "--format", format]);
    serde_json::from_str(&out).expect("JSON output")
}

fn attr<'a>(span: &'a Value, key: &str) -> Option<&'a Value> {
    span["attributes"].as_array()?.iter().find(|a| a["key"] == key).map(|a| &a["value"])
}

/// One trace: a session root, main's read and the delegation under it, the
/// subagent's read under the delegation, which ends when its agent stopped.
#[test]
fn otlp_nests_the_subagent_under_its_delegation() {
    let (_dir, root) = project();
    let out = trace(&root, "otlp");
    let spans = out["resourceSpans"][0]["scopeSpans"][0]["spans"].as_array().unwrap();
    assert_eq!(spans.len(), 4, "{spans:#?}");
    let named = |pred: &dyn Fn(&Value) -> bool| spans.iter().find(|s| pred(s)).unwrap();
    let session = named(&|s| s.get("parentSpanId").is_none());
    let delegation = named(&|s| attr(s, "gen_ai.operation.name") == Some(&json!({"stringValue": "invoke_agent"})));
    let sub_read = named(&|s| attr(s, "gen_ai.agent.id") == Some(&json!({"stringValue": AGENT})));

    assert_eq!(delegation["parentSpanId"], session["spanId"]);
    assert_eq!(sub_read["parentSpanId"], delegation["spanId"]);
    assert!(spans.iter().all(|s| s["traceId"] == session["traceId"]));
    assert_eq!(attr(sub_read, "code.file.path"), Some(&json!({"stringValue": "src/a.rs"})));
    let nanos = |s: &Value, k: &str| s[k].as_str().unwrap().parse::<u64>().unwrap();
    assert_eq!(nanos(delegation, "endTimeUnixNano") - nanos(delegation, "startTimeUnixNano"), 29_000_000_000);
}

/// Chrome: one thread per agent, complete events, and a flow arrow from the
/// delegation to the subagent's first span.
#[test]
fn chrome_puts_each_agent_on_its_own_thread() {
    let (_dir, root) = project();
    let out = trace(&root, "chrome");
    let events = out["traceEvents"].as_array().unwrap();
    let threads: Vec<&str> =
        events.iter().filter(|e| e["name"] == "thread_name").map(|e| e["args"]["name"].as_str().unwrap()).collect();
    assert_eq!(threads.len(), 2, "{threads:?}");
    assert!(threads[1].contains("look around"), "{threads:?}");
    assert_eq!(events.iter().filter(|e| e["ph"] == "X").count(), 3);
    assert_eq!(events.iter().filter(|e| e["ph"] == "s").count(), 1);
    assert_eq!(events.iter().filter(|e| e["ph"] == "f").count(), 1);
}

/// `--agent` narrows the export to that agent's subtree.
#[test]
fn the_agent_filter_keeps_one_agents_spans() {
    let (_dir, root) = project();
    let logs = root.join("logs").to_string_lossy().into_owned();
    let out = run_ambits(&root, &["--log-dir", &logs, "--session", SESSION, "--agent", AGENT, "trace"]);
    let out: Value = serde_json::from_str(&out).unwrap();
    let spans = out["resourceSpans"][0]["scopeSpans"][0]["spans"].as_array().unwrap();
    let children: Vec<&Value> = spans.iter().filter(|s| s.get("parentSpanId").is_some()).collect();
    assert_eq!(children.len(), 1, "{spans:#?}");
    assert!(children.iter().all(|s| attr(s, "gen_ai.agent.id") == Some(&json!({"stringValue": AGENT}))), "{spans:#?}");
}
