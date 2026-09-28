//! What a trace, or one call in it, amounts to: the facts the trace view's
//! right-hand panel shows. Pure: computed from the trace alone.

use std::collections::HashMap;
use std::sync::Arc;

use super::view::subtree;
use super::{InstantKind, SpanKind, Trace};

/// One file's share of a trace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileActivity {
    /// Project-relative.
    pub file: String,
    pub reads: usize,
    pub writes: usize,
    /// Its write calls, latest last: their ops name their write records.
    pub write_spans: Vec<usize>,
    /// Its first call in the trace.
    pub first: usize,
}

/// A subagent a trace started.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentRun {
    /// The delegation call.
    pub delegation: usize,
    pub agent: Arc<str>,
    pub description: String,
    /// Milliseconds, the delegation's start to its agent's last moment.
    pub duration: u64,
    pub calls: usize,
    pub failed: usize,
}

/// A trace — a prompt and everything done answering it — summed up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceDetail {
    pub root: usize,
    pub start: u64,
    pub end: u64,
    /// Calls under the root.
    pub calls: usize,
    /// Calls per tool, most first (ties by name).
    pub by_tool: Vec<(Arc<str>, usize)>,
    /// Files touched, most active first.
    pub files: Vec<FileActivity>,
    pub agents: Vec<AgentRun>,
    /// Failed calls, in time order.
    pub failed: Vec<usize>,
    /// Commits made while it ran (instant indices), in time order.
    pub commits: Vec<usize>,
}

/// The trace rooted at span `root`, summed up; `None` for a span that is
/// not a root.
pub fn detail(trace: &Trace, root: usize) -> Option<TraceDetail> {
    let tree = trace.tree();
    let node = tree.iter().find(|n| n.span == root)?;
    let spans = trace.spans();
    let mut under: Vec<usize> = node.spans().into_iter().filter(|&i| i != root).collect();
    under.sort_by_key(|&i| (spans[i].start, i));

    let mut by_tool: HashMap<Arc<str>, usize> = HashMap::new();
    let mut files: Vec<FileActivity> = Vec::new();
    let mut agents = Vec::new();
    let mut failed = Vec::new();
    for &i in &under {
        let s = &spans[i];
        *by_tool.entry(s.tool.clone()).or_default() += 1;
        if s.error {
            failed.push(i);
        }
        if let Some(file) = &s.file {
            let at = match files.iter().position(|f| &f.file == file) {
                Some(at) => at,
                None => {
                    files.push(FileActivity { file: file.clone(), reads: 0, writes: 0, write_spans: Vec::new(), first: i });
                    files.len() - 1
                }
            };
            match s.kind {
                SpanKind::Read(_) => files[at].reads += 1,
                SpanKind::Write => {
                    files[at].writes += 1;
                    files[at].write_spans.push(i);
                }
                _ => {}
            }
        }
        if let (SpanKind::Delegate, Some(agent)) = (s.kind, &s.child_agent) {
            let run = subtree(&tree, i).map(|n| n.spans()).unwrap_or_default();
            let end = subtree(&tree, i).map_or(s.start, |n| n.end);
            agents.push(AgentRun {
                delegation: i,
                agent: agent.clone(),
                description: s.description.clone(),
                duration: end.saturating_sub(s.start),
                calls: run.len().saturating_sub(1),
                failed: run.iter().filter(|&&c| c != i && spans[c].error).count(),
            });
        }
    }
    let mut by_tool: Vec<(Arc<str>, usize)> = by_tool.into_iter().collect();
    by_tool.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    files.sort_by(|a, b| (b.reads + b.writes).cmp(&(a.reads + a.writes)).then_with(|| a.first.cmp(&b.first)));

    let (start, end) = (spans[root].start, node.end);
    let mut commits: Vec<usize> = (0..trace.instants().len())
        .filter(|&i| matches!(trace.instants()[i].kind, InstantKind::Commit { .. }) && (start..=end).contains(&trace.instants()[i].t))
        .collect();
    commits.sort_by_key(|&i| trace.instants()[i].t);

    Some(TraceDetail { root, start, end, calls: under.len(), by_tool, files, agents, failed, commits })
}

/// The other calls in `span`'s trace on the same file, in time order.
pub fn related(trace: &Trace, span: usize) -> Vec<usize> {
    let spans = trace.spans();
    let Some(file) = spans.get(span).and_then(|s| s.file.as_deref()) else { return Vec::new() };
    let root_of = super::roots_by_span(&trace.tree());
    let root = root_of.get(&span).copied();
    let mut out: Vec<usize> = (0..spans.len())
        .filter(|&i| i != span && spans[i].file.as_deref() == Some(file) && root_of.get(&i).copied() == root)
        .collect();
    out.sort_by_key(|&i| (spans[i].start, i));
    out
}

/// The trace `span` belongs to: its root span.
pub fn root_of(trace: &Trace, span: usize) -> Option<usize> {
    super::roots_by_span(&trace.tree()).get(&span).copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::{Effect, Prompt, ToolFinished};
    use std::path::Path;

    /// `(agent, id, tool, file, start, end, failed, subagent)`.
    type Call<'a> = (&'a str, &'a str, &'a str, &'a str, &'a str, &'a str, bool, Option<&'a str>);

    fn add(t: &mut Trace, &(agent, id, tool, file, at, end, error, child): &Call<'_>) {
        let mut c = crate::helpers::tool_call(tool, &format!("/p/{file}"), crate::tracking::ReadDepth::FullBody);
        c.agent_id = Arc::from(agent);
        c.tool_use_id = Some(Arc::from(id));
        c.timestamp_str = at.into();
        if tool == "Edit" {
            c.effect = Effect::Write;
            c.read_depth = crate::tracking::ReadDepth::Unseen;
        }
        t.start(&c, Path::new("/p"));
        t.finish(&ToolFinished { id: Arc::from(id), agent_id: Arc::from(agent), timestamp: end.into(), error, message: None, child_agent: child.map(Arc::from) });
    }

    /// A prompt (0) with: two reads and an edit of a.rs, a failed edit of
    /// b.rs, and a delegation to `x`, whose agent reads a.rs twice.
    fn sample() -> Trace {
        let mut t = Trace::default();
        t.prompt(&Prompt { agent_id: Arc::from("main"), timestamp: "2026-09-27T10:00:00Z".into(), text: "go".into() });
        let calls: [Call<'_>; 7] = [
            ("main", "r1", "Read", "src/a.rs", "2026-09-27T10:00:01Z", "2026-09-27T10:00:02Z", false, None),
            ("main", "r2", "Read", "src/a.rs", "2026-09-27T10:00:03Z", "2026-09-27T10:00:04Z", false, None),
            ("main", "e1", "Edit", "src/a.rs", "2026-09-27T10:00:05Z", "2026-09-27T10:00:06Z", false, None),
            ("main", "e2", "Edit", "src/b.rs", "2026-09-27T10:00:07Z", "2026-09-27T10:00:08Z", true, None),
            ("main", "d1", "Agent", "src/x.md", "2026-09-27T10:00:09Z", "2026-09-27T10:00:10Z", false, Some("x")),
            ("x", "x1", "Read", "src/a.rs", "2026-09-27T10:00:11Z", "2026-09-27T10:00:12Z", false, None),
            ("x", "x2", "Read", "src/a.rs", "2026-09-27T10:00:13Z", "2026-09-27T10:00:20Z", false, None),
        ];
        for c in &calls {
            add(&mut t, c);
        }
        t.set_commits(&[crate::git::Commit { sha: "a".repeat(40), t: 1_790_503_205_500, subject: "c".into() }]);
        t
    }

    #[test]
    fn a_trace_is_summed_up_by_tool_file_agent_failure_and_commit() {
        let t = sample();
        let d = detail(&t, 0).unwrap();
        assert_eq!(d.calls, 7);
        assert_eq!(d.by_tool.iter().map(|(t, n)| (t.as_ref(), *n)).collect::<Vec<_>>(), vec![("Read", 4), ("Edit", 2), ("Agent", 1)]);
        let a = &d.files[0];
        assert_eq!((a.file.as_str(), a.reads, a.writes, a.write_spans.clone()), ("src/a.rs", 4, 1, vec![3]));
        assert_eq!(d.files[1].file, "src/b.rs");
        assert_eq!(d.failed, vec![4]);
        assert_eq!(d.agents.len(), 1);
        let run = &d.agents[0];
        assert_eq!((&*run.agent, run.calls, run.failed, run.duration), ("x", 2, 0, 11_000));
        assert_eq!(d.commits, vec![0]);
        assert!(detail(&t, 1).is_none(), "not a root");
    }

    #[test]
    fn related_calls_share_the_file_and_the_trace_in_time_order() {
        let t = sample();
        assert_eq!(related(&t, 1), vec![2, 3, 6, 7]);
        assert_eq!(related(&t, 4), Vec::<usize>::new(), "nothing else on b.rs");
        assert_eq!(root_of(&t, 7), Some(0));
    }
}
