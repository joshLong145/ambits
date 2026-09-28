//! `ambits trace`: a session's trace in formats real tracing tools read.
//!
//! - **OTLP/JSON** — OpenTelemetry's wire format, for Jaeger, Grafana Tempo
//!   or any collector. Spans follow the GenAI semantic conventions where
//!   they fit (`execute_tool`, `invoke_agent`, `gen_ai.tool.*`) and the code
//!   conventions for what a call touched (`code.file.path`,
//!   `code.function.name`). Ids are derived from the session id and each
//!   call's `tool_use_id`, so exporting twice gives the same trace.
//! - **Chrome trace events** — for Perfetto and `chrome://tracing`: a thread
//!   per agent, a complete event per call, instants, and flow arrows from a
//!   delegation to the subagent it started.

use std::collections::HashMap;

use serde_json::{json, Value};

use super::{InstantKind, Node, Span, SpanKind, Trace};

/// Which of an agent's spans to export: `None` is the whole session; an
/// agent keeps that agent's spans and everything it delegated.
pub type AgentFilter<'a> = Option<&'a str>;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn derive(parts: &[&str], len: usize) -> String {
    let hash = crate::objects::hash_framed("ambits-trace v1", parts.iter().map(|p| p.as_bytes()));
    hex(&hash.as_bytes()[..len])
}

fn trace_id(session: &str) -> String {
    derive(&["trace", session], 16)
}

fn span_id(session: &str, span: &Span, index: usize) -> String {
    // A call without a tool_use_id falls back to its position, still stable
    // for a given log.
    let key = span.id.as_deref().map_or_else(|| format!("#{index}"), String::from);
    derive(&["span", session, &key], 8)
}

fn root_span_id(session: &str) -> String {
    derive(&["root", session], 8)
}

fn attr(key: &str, value: Value) -> Value {
    let value = match value {
        Value::Bool(b) => json!({"boolValue": b}),
        Value::Number(n) => json!({"intValue": n.to_string()}),
        other => json!({"stringValue": other.as_str().map_or_else(|| other.to_string(), String::from)}),
    };
    json!({"key": key, "value": value})
}

fn depth_name(depth: crate::tracking::ReadDepth) -> &'static str {
    use crate::tracking::ReadDepth::*;
    match depth {
        Unseen => "unseen",
        NameOnly => "name_only",
        Overview => "overview",
        Signature => "signature",
        FullBody => "full_body",
    }
}

fn span_attributes(span: &Span) -> Vec<Value> {
    // A prompt invokes the session's agent, as a delegation invokes a subagent.
    let operation = if matches!(span.kind, SpanKind::Delegate | SpanKind::Prompt) { "invoke_agent" } else { "execute_tool" };
    let mut attrs = vec![
        attr("gen_ai.operation.name", json!(operation)),
        attr("gen_ai.tool.name", json!(&*span.tool)),
        attr("gen_ai.agent.id", json!(&*span.agent)),
    ];
    if let Some(id) = &span.id {
        attrs.push(attr("gen_ai.tool.call.id", json!(&**id)));
    }
    if let Some(file) = &span.file {
        attrs.push(attr("code.file.path", json!(file)));
    }
    if let Some(symbol) = &span.symbol {
        attrs.push(attr("code.function.name", json!(symbol)));
    }
    match span.kind {
        SpanKind::Read(depth) => attrs.push(attr("ambits.read.depth", json!(depth_name(depth)))),
        SpanKind::Write => attrs.push(attr("ambits.write", json!(true))),
        SpanKind::Prompt => attrs.push(attr("ambits.prompt", json!(true))),
        SpanKind::Delegate | SpanKind::Other => {}
    }
    if let Some(child) = &span.child_agent {
        attrs.push(attr("ambits.subagent.id", json!(&**child)));
    }
    if span.end.is_none() {
        attrs.push(attr("ambits.open", json!(true)));
    }
    attrs
}

fn nanos(millis: u64) -> String {
    (u128::from(millis) * 1_000_000).to_string()
}

/// The trace as OTLP/JSON (`ExportTraceServiceRequest`).
pub fn otlp(trace: &Trace, session: &str, filter: AgentFilter<'_>) -> Value {
    let tree = trace.tree();
    let roots = trace.roots(&tree, filter);
    let tid = trace_id(session);
    let root_id = root_span_id(session);
    let (start, end) = roots
        .iter()
        .map(|n| (trace.spans()[n.span].start, n.end))
        .fold(None, |acc: Option<(u64, u64)>, (s, e)| Some(acc.map_or((s, e), |(a, b)| (a.min(s), b.max(e)))))
        .or_else(|| trace.range())
        .unwrap_or((0, 0));

    let mut spans: Vec<Value> = Vec::new();
    // Each moment is an event on the prompt it happened during, else on
    // the session.
    let mut root_events: Vec<Value> = Vec::new();
    let mut events: std::collections::HashMap<usize, Vec<Value>> = std::collections::HashMap::new();
    for i in trace.instants().iter().filter(|i| filter.is_none_or(|a| i.agent.as_deref().is_none_or(|ia| ia.starts_with(a)))) {
        let mut attrs = Vec::new();
        let name = match &i.kind {
            InstantKind::Compaction => "compaction",
            InstantKind::Snapshot(id) => {
                attrs.push(attr("ambits.snapshot.id", json!(id)));
                "snapshot"
            }
            InstantKind::Commit { sha, subject } => {
                attrs.push(attr("vcs.ref.head.revision", json!(sha)));
                attrs.push(attr("ambits.commit.subject", json!(subject)));
                "commit"
            }
        };
        if let Some(agent) = &i.agent {
            attrs.push(attr("gen_ai.agent.id", json!(&**agent)));
        }
        let event = json!({"timeUnixNano": nanos(i.t), "name": name, "attributes": attrs});
        let during = roots.iter().find(|n| trace.spans()[n.span].kind == SpanKind::Prompt && (trace.spans()[n.span].start..=n.end).contains(&i.t));
        match during {
            Some(n) => events.entry(n.span).or_default().push(event),
            None => root_events.push(event),
        }
    }
    spans.push(json!({
        "traceId": tid,
        "spanId": root_id,
        "name": format!("session {session}"),
        "kind": 1,
        "startTimeUnixNano": nanos(start),
        "endTimeUnixNano": nanos(end),
        "attributes": [attr("session.id", json!(session))],
        "events": root_events,
        "status": {"code": 0},
    }));

    let mut stack: Vec<(&Node, String)> = roots.iter().rev().map(|n| (*n, root_id.clone())).collect();
    while let Some((node, parent)) = stack.pop() {
        let span = &trace.spans()[node.span];
        let id = span_id(session, span, node.span);
        let status = if span.error { json!({"code": 2, "message": "tool call failed"}) } else { json!({"code": 0}) };
        let mut out = json!({
            "traceId": tid,
            "spanId": id,
            "parentSpanId": parent,
            "name": span.name(),
            "kind": 1,
            "startTimeUnixNano": nanos(span.start),
            "endTimeUnixNano": nanos(node.end),
            "attributes": span_attributes(span),
            "status": status,
        });
        if let Some(events) = events.remove(&node.span) {
            out["events"] = json!(events);
        }
        spans.push(out);
        for child in node.children.iter().rev() {
            stack.push((child, id.clone()));
        }
    }

    json!({"resourceSpans": [{
        "resource": {"attributes": [
            attr("service.name", json!("ambits")),
            attr("session.id", json!(session)),
        ]},
        "scopeSpans": [{
            "scope": {"name": "ambits", "version": env!("CARGO_PKG_VERSION")},
            "spans": spans,
        }],
    }]})
}

/// The trace as Chrome trace events (`{"traceEvents": […]}`).
pub fn chrome(trace: &Trace, session: &str, filter: AgentFilter<'_>) -> Value {
    let tree = trace.tree();
    let roots = trace.roots(&tree, filter);
    let mut nodes: Vec<&Node> = Vec::new();
    let mut stack: Vec<&Node> = roots.clone();
    while let Some(n) = stack.pop() {
        nodes.push(n);
        stack.extend(n.children.iter());
    }
    nodes.sort_by_key(|n| (trace.spans()[n.span].start, n.span));

    // One thread per agent, the session's own agent first.
    let mut tids: HashMap<&str, u64> = HashMap::new();
    let mut agents: Vec<&str> = Vec::new();
    for n in &nodes {
        let agent = &*trace.spans()[n.span].agent;
        if !tids.contains_key(agent) {
            agents.push(agent);
            tids.insert(agent, 0);
        }
    }
    agents.sort_by_key(|a| (*a != session, *a));
    for (i, a) in agents.iter().enumerate() {
        tids.insert(a, i as u64 + 1);
    }
    let micros = |ms: u64| ms * 1000;
    let label: HashMap<&str, String> = nodes
        .iter()
        .filter_map(|n| {
            let s = &trace.spans()[n.span];
            Some((s.child_agent.as_deref()?, s.description.clone()))
        })
        .collect();

    let mut events: Vec<Value> = vec![json!({"name": "process_name", "ph": "M", "pid": 1, "args": {"name": format!("session {session}")}})];
    for a in &agents {
        let name = match (*a == session, label.get(a)) {
            (true, _) => "main".to_string(),
            (false, Some(l)) if !l.is_empty() => format!("{a}: {l}"),
            _ => a.to_string(),
        };
        events.push(json!({"name": "thread_name", "ph": "M", "pid": 1, "tid": tids[a], "args": {"name": name}}));
    }
    for (flow, n) in nodes.iter().enumerate() {
        let s = &trace.spans()[n.span];
        let cat = match s.kind {
            SpanKind::Read(_) => "read",
            SpanKind::Write => "write",
            SpanKind::Delegate => "delegate",
            SpanKind::Prompt => "prompt",
            SpanKind::Other => "other",
        };
        let mut args = serde_json::Map::new();
        for a in span_attributes(s) {
            let value = a["value"].as_object().and_then(|v| v.values().next()).cloned().unwrap_or_default();
            args.insert(a["key"].as_str().unwrap_or_default().to_string(), value);
        }
        if s.error {
            args.insert("error".into(), json!(true));
        }
        let tid = tids[&*s.agent];
        events.push(json!({
            "name": s.name(), "cat": cat, "ph": "X", "pid": 1, "tid": tid,
            "ts": micros(s.start), "dur": micros(n.end - s.start), "args": args,
        }));
        // A delegation's flow arrow to the first span of its subagent.
        let first = n.children.iter().map(|c| &trace.spans()[c.span]).filter(|c| s.child_agent.as_ref() == Some(&c.agent)).min_by_key(|c| c.start);
        if let Some(first) = first {
            events.push(json!({"name": "delegate", "cat": "delegate", "ph": "s", "id": flow, "pid": 1, "tid": tid, "ts": micros(s.start)}));
            events.push(json!({"name": "delegate", "cat": "delegate", "ph": "f", "bp": "e", "id": flow, "pid": 1, "tid": tids[&*first.agent], "ts": micros(first.start)}));
        }
    }
    for i in trace.instants() {
        if filter.is_some_and(|a| i.agent.as_deref().is_some_and(|ia| !ia.starts_with(a))) {
            continue;
        }
        let name = i.kind.label();
        let (scope, tid) = match i.agent.as_deref().and_then(|a| tids.get(a)) {
            Some(tid) => ("t", *tid),
            None => ("g", 0),
        };
        events.push(json!({"name": name, "ph": "i", "s": scope, "pid": 1, "tid": tid, "ts": micros(i.t)}));
    }
    json!({"traceEvents": events, "displayTimeUnit": "ms"})
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::{AgentToolCall, ToolFinished};
    use crate::tracking::ReadDepth;
    use std::path::Path;
    use std::sync::Arc;

    const SESSION: &str = "0b7e9d3a-1c2f-4e5a-9b8c-7d6e5f4a3b2c";

    fn call(agent: &str, id: &str, tool: &str, at: &str) -> AgentToolCall {
        let mut c = crate::helpers::tool_call(tool, "/p/src/a.rs", ReadDepth::FullBody);
        c.agent_id = Arc::from(agent);
        c.tool_use_id = Some(Arc::from(id));
        c.timestamp_str = at.to_string();
        if tool == "Edit" {
            c.effect = crate::ingest::Effect::Write;
        }
        c
    }

    fn done(agent: &str, id: &str, at: &str, child: Option<&str>, error: bool) -> ToolFinished {
        ToolFinished { id: Arc::from(id), agent_id: Arc::from(agent), timestamp: at.into(), error, child_agent: child.map(Arc::from) }
    }

    /// main reads, delegates to agent-x (which reads and fails an edit),
    /// and compacts.
    fn sample() -> Trace {
        let mut t = Trace::default();
        let p = Path::new("/p");
        t.start(&call(SESSION, "r1", "Read", "2026-09-27T10:00:00.000Z"), p);
        t.finish(&done(SESSION, "r1", "2026-09-27T10:00:00.400Z", None, false));
        t.start(&call(SESSION, "d1", "Agent", "2026-09-27T10:00:01.000Z"), p);
        t.finish(&done(SESSION, "d1", "2026-09-27T10:00:01.100Z", Some("agent-x"), false));
        t.start(&call("agent-x", "x1", "Read", "2026-09-27T10:00:02.000Z"), p);
        t.finish(&done("agent-x", "x1", "2026-09-27T10:00:02.500Z", None, false));
        t.start(&call("agent-x", "x2", "Edit", "2026-09-27T10:00:03.000Z"), p);
        t.finish(&done("agent-x", "x2", "2026-09-27T10:00:03.100Z", None, true));
        t.instant(1_790_503_204_000, Some(Arc::from(SESSION)), InstantKind::Compaction);
        t
    }

    fn otlp_spans(v: &Value) -> Vec<Value> {
        v["resourceSpans"][0]["scopeSpans"][0]["spans"].as_array().unwrap().clone()
    }

    #[test]
    fn otlp_nests_delegations_and_marks_errors() {
        let v = otlp(&sample(), SESSION, None);
        let spans = otlp_spans(&v);
        assert_eq!(spans.len(), 5, "root + four calls");
        let root = &spans[0];
        assert!(root.get("parentSpanId").is_none());
        let by_name = |n: &str| spans.iter().find(|s| s["name"].as_str().unwrap().starts_with(n)).unwrap().clone();
        let delegation = spans.iter().find(|s| s["attributes"].to_string().contains("invoke_agent")).unwrap();
        let edit = by_name("Edit");
        assert_eq!(edit["parentSpanId"], delegation["spanId"], "the subagent's calls nest under the delegation");
        assert_eq!(edit["status"]["code"], 2);
        assert_eq!(by_name("Read")["parentSpanId"], root["spanId"]);
        // The delegation lasts until its subagent's last call ends.
        assert_eq!(delegation["endTimeUnixNano"], edit["endTimeUnixNano"]);
        assert_eq!(root["events"][0]["name"], "compaction");
        for s in &spans {
            assert_eq!(s["traceId"].as_str().unwrap().len(), 32);
            assert_eq!(s["spanId"].as_str().unwrap().len(), 16);
        }
    }

    #[test]
    fn otlp_ids_are_stable_across_exports() {
        assert_eq!(otlp(&sample(), SESSION, None), otlp(&sample(), SESSION, None));
    }

    #[test]
    fn an_agent_filter_keeps_that_agents_subtrees() {
        let spans = otlp_spans(&otlp(&sample(), SESSION, Some("agent-x")));
        let names: Vec<&str> = spans.iter().skip(1).map(|s| s["name"].as_str().unwrap()).collect();
        assert_eq!(names.len(), 2, "{names:?}");
        assert!(names.iter().all(|n| n.starts_with("Read") || n.starts_with("Edit")));
    }

    #[test]
    fn chrome_has_a_thread_per_agent_complete_events_and_a_flow() {
        let v = chrome(&sample(), SESSION, None);
        let events = v["traceEvents"].as_array().unwrap();
        let threads: Vec<&str> = events.iter().filter(|e| e["name"] == "thread_name").map(|e| e["args"]["name"].as_str().unwrap()).collect();
        assert_eq!(threads[0], "main");
        assert!(threads[1].starts_with("agent-x"));
        assert_eq!(events.iter().filter(|e| e["ph"] == "X").count(), 4);
        assert!(events.iter().any(|e| e["ph"] == "s") && events.iter().any(|e| e["ph"] == "f"));
        assert!(events.iter().any(|e| e["ph"] == "i" && e["name"] == "compaction"));
        let failed = events.iter().find(|e| e["ph"] == "X" && e["cat"] == "write").unwrap();
        assert_eq!(failed["args"]["error"], true);
    }
}
