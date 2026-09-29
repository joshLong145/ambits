//! What a trace, or one call in it, amounts to: the facts the trace view's
//! right-hand panel shows. Pure: computed from the trace alone.

use std::collections::HashMap;
use std::sync::Arc;

use super::view::{effective_ends, subtree};
use super::{InstantKind, Node, SpanKind, Trace};
use crate::writes::WriteRecord;
use crate::tracking::ReadDepth;

/// A trace's structure, worked out once and asked many times (a frame asks
/// dozens of questions of it): the span tree, which trace each span belongs
/// to, and each span's effective end.
#[derive(Debug, Clone)]
pub struct TraceIndex {
    pub tree: Vec<Node>,
    roots: HashMap<usize, usize>,
    ends: HashMap<usize, u64>,
}

impl TraceIndex {
    pub fn new(trace: &Trace) -> Self {
        let tree = trace.tree();
        let roots = super::roots_by_span(&tree);
        let ends = effective_ends(&tree);
        Self { tree, roots, ends }
    }

    /// The trace `span` belongs to: its root span.
    pub fn root_of(&self, span: usize) -> Option<usize> {
        self.roots.get(&span).copied()
    }

    /// When `span` ended — a prompt or delegation, once what it started
    /// did — or `None` while it runs.
    pub fn end_of(&self, trace: &Trace, span: usize) -> Option<u64> {
        trace.spans().get(span)?.end?;
        self.ends.get(&span).copied()
    }

    /// `span`'s effective end, running or not: what bars are drawn to.
    pub fn effective_end(&self, span: usize) -> Option<u64> {
        self.ends.get(&span).copied()
    }

    /// The node for `span`.
    pub fn node(&self, span: usize) -> Option<&Node> {
        subtree(&self.tree, span)
    }

    /// The spans above `span`, its root first: what folds it away.
    pub fn ancestors(&self, span: usize) -> Vec<usize> {
        fn walk(nodes: &[Node], want: usize, path: &mut Vec<usize>) -> bool {
            nodes.iter().any(|n| {
                if n.span == want {
                    return true;
                }
                path.push(n.span);
                let found = walk(&n.children, want, path);
                if !found {
                    path.pop();
                }
                found
            })
        }
        let mut path = Vec::new();
        if walk(&self.tree, span, &mut path) { path } else { Vec::new() }
    }
}

/// One file's share of a trace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileActivity {
    /// Project-relative.
    pub file: String,
    /// Its read calls, in time order.
    pub reads: Vec<usize>,
    /// Its write calls, latest last: their ops name their write records.
    pub writes: Vec<usize>,
    /// Its first call in the trace.
    pub first: usize,
}

impl FileActivity {
    /// What its reads saw: each symbol once, at the deepest it was read,
    /// in the order first read; `None` for a read that failed and saw
    /// nothing (listed only when no read of it succeeded). A `None` symbol
    /// is the whole file.
    pub fn symbols_read(&self, trace: &Trace) -> Vec<(Option<String>, Option<ReadDepth>)> {
        let mut out: Vec<(Option<String>, Option<ReadDepth>)> = Vec::new();
        for &i in &self.reads {
            let s = &trace.spans()[i];
            let SpanKind::Read(depth) = s.kind else { continue };
            let depth = (!s.error).then_some(depth);
            let name = s.symbol_name();
            match out.iter_mut().find(|(n, _)| *n == name) {
                Some((_, d)) => *d = (*d).max(depth),
                None => out.push((name, depth)),
            }
        }
        out
    }
}

/// A subagent a trace started.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentRun {
    /// The delegation call.
    pub delegation: usize,
    pub agent: Arc<str>,
    /// What it was started for.
    pub task: String,
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
pub fn detail(trace: &Trace, index: &TraceIndex, root: usize) -> Option<TraceDetail> {
    let node = index.tree.iter().find(|n| n.span == root)?;
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
                    files.push(FileActivity { file: file.clone(), reads: Vec::new(), writes: Vec::new(), first: i });
                    files.len() - 1
                }
            };
            match s.kind {
                SpanKind::Read(_) => files[at].reads.push(i),
                SpanKind::Write => files[at].writes.push(i),
                _ => {}
            }
        }
        if let (SpanKind::Delegate, Some(agent)) = (s.kind, &s.child_agent) {
            let run = index.node(i).map(|n| n.spans()).unwrap_or_default();
            let end = index.node(i).map_or(s.start, |n| n.end);
            agents.push(AgentRun {
                delegation: i,
                agent: agent.clone(),
                task: s.task(),
                duration: end.saturating_sub(s.start),
                calls: run.len().saturating_sub(1),
                failed: run.iter().filter(|&&c| c != i && spans[c].error).count(),
            });
        }
    }
    let mut by_tool: Vec<(Arc<str>, usize)> = by_tool.into_iter().collect();
    by_tool.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    files.sort_by(|a, b| (b.reads.len() + b.writes.len()).cmp(&(a.reads.len() + a.writes.len())).then_with(|| a.first.cmp(&b.first)));

    let (start, end) = (spans[root].start, node.end);
    let mut commits: Vec<usize> = (0..trace.instants().len())
        .filter(|&i| matches!(trace.instants()[i].kind, InstantKind::Commit { .. }) && (start..=end).contains(&trace.instants()[i].t))
        .collect();
    commits.sort_by_key(|&i| trace.instants()[i].t);

    Some(TraceDetail { root, start, end, calls: under.len(), by_tool, files, agents, failed, commits })
}

/// What a row of the trace panel points at, for `Enter`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Target {
    /// A project file: shown in the tree.
    File(String),
    /// A call (a delegation, a failure, a related call): selected in the
    /// timeline.
    Span(usize),
    /// A commit or compaction: selected in the timeline.
    Instant(usize),
    /// A symbol, by id: its definition opened in the editor.
    Symbol(String),
}

/// What a write did to a symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolChange {
    /// No symbol had its id before.
    Created,
    /// It was there, and the write changed it.
    Edited,
    /// The write took it out.
    Deleted,
}

/// A selectable row of the trace panel: what it shows, and so where
/// `Enter` on it goes. The panel draws these, in this order, and `Enter`
/// indexes the same list — one order for both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Row<'a> {
    /// A file the trace touched.
    File(&'a FileActivity),
    /// A subagent it started.
    Agent(&'a AgentRun),
    /// A call of it that failed.
    Failed(usize),
    /// A commit made while it ran (an instant).
    Commit(usize),
    /// A call's own file.
    ThisFile(&'a str),
    /// Another call on that file, in the same trace, and whether it came
    /// before the call.
    Related { span: usize, before: bool },
    /// A symbol the call read, and how deeply.
    Read { id: &'a str, depth: ReadDepth },
    /// A symbol the call wrote, and what it did to it; `file` is where a
    /// deleted one was.
    Wrote { id: &'a str, change: SymbolChange, file: &'a str },
}

impl Row<'_> {
    /// The heading of the rows it is listed under.
    pub fn section(&self) -> &'static str {
        match self {
            Row::File(_) => "files",
            Row::Agent(_) => "agents",
            Row::Failed(_) => "failed",
            Row::Commit(_) => "commits",
            Row::ThisFile(_) | Row::Related { .. } => "on this file",
            Row::Read { .. } => "symbols read",
            Row::Wrote { .. } => "symbols written",
        }
    }

    pub fn target(&self) -> Target {
        match *self {
            Row::File(f) => Target::File(f.file.clone()),
            Row::ThisFile(file) => Target::File(file.to_string()),
            Row::Agent(a) => Target::Span(a.delegation),
            Row::Failed(i) | Row::Related { span: i, .. } => Target::Span(i),
            Row::Commit(i) => Target::Instant(i),
            // A deleted symbol has no definition left to open: its file.
            Row::Wrote { change: SymbolChange::Deleted, file, .. } => Target::File(file.to_string()),
            Row::Read { id, .. } | Row::Wrote { id, .. } => Target::Symbol(id.to_string()),
        }
    }
}

impl TraceDetail {
    /// The summary's rows: files, agents, failed calls, commits.
    pub fn rows(&self) -> Vec<Row<'_>> {
        let files = self.files.iter().map(Row::File);
        let agents = self.agents.iter().map(Row::Agent);
        let failed = self.failed.iter().map(|&i| Row::Failed(i));
        let commits = self.commits.iter().map(|&i| Row::Commit(i));
        files.chain(agents).chain(failed).chain(commits).collect()
    }
}

/// A call's rows: the symbols it read, or — `write`, its write record —
/// wrote; its file; then the other calls on that file in its trace.
pub fn call_rows<'t>(trace: &'t Trace, index: &TraceIndex, span: usize, write: Option<&'t WriteRecord>) -> Vec<Row<'t>> {
    let Some(s) = trace.spans().get(span) else { return Vec::new() };
    let read = s.read.iter().map(|(id, depth)| Row::Read { id, depth: *depth });
    let wrote = write.map(written).unwrap_or_default();
    let file = s.file.as_deref().map(Row::ThisFile);
    let related = related(trace, index, span).into_iter().map(|j| Row::Related { span: j, before: j < span });
    read.chain(wrote).chain(file).chain(related).collect()
}

/// A write's symbols, each with what it did: edited, then created, then
/// deleted.
fn written(w: &WriteRecord) -> Vec<Row<'_>> {
    let created = |id: &str| w.created.iter().any(|c| c == id);
    let (made, edited): (Vec<&str>, Vec<&str>) = w.syms.iter().map(|(id, _)| id.as_str()).partition(|id| created(id));
    edited
        .into_iter()
        .map(|id| Row::Wrote { id, change: SymbolChange::Edited, file: &w.file })
        .chain(made.into_iter().map(|id| Row::Wrote { id, change: SymbolChange::Created, file: &w.file }))
        .chain(w.removed.iter().map(|id| Row::Wrote { id, change: SymbolChange::Deleted, file: &w.file }))
        .collect()
}

/// The other calls in `span`'s trace on the same file, in time order.
pub fn related(trace: &Trace, index: &TraceIndex, span: usize) -> Vec<usize> {
    let spans = trace.spans();
    let Some(file) = spans.get(span).and_then(|s| s.file.as_deref()) else { return Vec::new() };
    let root = index.root_of(span);
    let mut out: Vec<usize> = (0..spans.len())
        .filter(|&i| i != span && spans[i].file.as_deref() == Some(file) && index.root_of(i) == root)
        .collect();
    out.sort_by_key(|&i| (spans[i].start, i));
    out
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
        t.finish(&ToolFinished { id: Arc::from(id), agent_id: Arc::from(agent), timestamp: end.into(), error, message: None, child_agent: child.map(Arc::from), shown: Vec::new() });
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
        let d = detail(&t, &TraceIndex::new(&t), 0).unwrap();
        assert_eq!(d.calls, 7);
        assert_eq!(d.by_tool.iter().map(|(t, n)| (t.as_ref(), *n)).collect::<Vec<_>>(), vec![("Read", 4), ("Edit", 2), ("Agent", 1)]);
        let a = &d.files[0];
        assert_eq!((a.file.as_str(), a.reads.clone(), a.writes.clone()), ("src/a.rs", vec![1, 2, 6, 7], vec![3]));
        assert_eq!(a.symbols_read(&t), vec![(None, Some(ReadDepth::FullBody))], "four whole-file reads, once");
        assert_eq!(d.files[1].file, "src/b.rs");
        assert_eq!(d.failed, vec![4]);
        assert_eq!(d.agents.len(), 1);
        let run = &d.agents[0];
        assert_eq!((&*run.agent, run.calls, run.failed, run.duration), ("x", 2, 0, 11_000));
        assert_eq!(d.commits, vec![0]);
        assert!(detail(&t, &TraceIndex::new(&t), 1).is_none(), "not a root");
    }

    #[test]
    fn a_files_reads_are_each_symbol_once_at_its_deepest() {
        let mut t = Trace::default();
        t.prompt(&Prompt { agent_id: Arc::from("main"), timestamp: "2026-09-27T10:00:00Z".into(), text: "go".into() });
        let reads = [
            ("r1", Some("impl App/fn run"), ReadDepth::Signature, false),
            ("r2", None, ReadDepth::Overview, false),
            ("r3", Some("App/run"), ReadDepth::FullBody, false),
            ("r4", Some("App/stop"), ReadDepth::FullBody, true),
            ("r5", None, ReadDepth::FullBody, true),
        ];
        for (id, symbol, depth, error) in reads {
            let mut c = crate::helpers::tool_call("Read", "/p/src/a.rs", depth);
            c.agent_id = Arc::from("main");
            c.tool_use_id = Some(Arc::from(id));
            c.timestamp_str = "2026-09-27T10:00:01Z".into();
            c.target_symbol = symbol.map(String::from);
            t.start(&c, Path::new("/p"));
            t.finish(&ToolFinished { id: Arc::from(id), agent_id: Arc::from("main"), timestamp: "2026-09-27T10:00:02Z".into(), error, message: None, child_agent: None, shown: Vec::new() });
        }
        let d = detail(&t, &TraceIndex::new(&t), 0).unwrap();
        assert_eq!(
            d.files[0].symbols_read(&t),
            vec![(Some("App/run".into()), Some(ReadDepth::FullBody)), (None, Some(ReadDepth::Overview)), (Some("App/stop".into()), None)],
            "a failed read saw nothing, and does not deepen one that succeeded"
        );
    }

    #[test]
    fn rows_point_at_files_calls_and_commits_in_panel_order() {
        let t = sample();
        let d = detail(&t, &TraceIndex::new(&t), 0).unwrap();
        assert_eq!(
            d.rows().iter().map(Row::target).collect::<Vec<_>>(),
            vec![
                Target::File("src/a.rs".into()),
                Target::File("src/b.rs".into()),
                Target::File("src/x.md".into()),
                Target::Span(5),
                Target::Span(4),
                Target::Instant(0),
            ]
        );
        let index = TraceIndex::new(&t);
        assert_eq!(index.ancestors(6), vec![0, 5], "the prompt, then the delegation");
        let w = WriteRecord { file: "src/a.rs".into(), syms: vec![("src/a.rs::e".into(), String::new()), ("src/a.rs::n".into(), String::new())], created: vec!["src/a.rs::n".into()], removed: vec!["src/a.rs::d".into()], ..Default::default() };
        let wrote: Vec<(String, SymbolChange)> = call_rows(&t, &index, 3, Some(&w))
            .into_iter()
            .filter_map(|r| match r {
                Row::Wrote { id, change, .. } => Some((id.to_string(), change)),
                _ => None,
            })
            .collect();
        assert_eq!(wrote, vec![("src/a.rs::e".into(), SymbolChange::Edited), ("src/a.rs::n".into(), SymbolChange::Created), ("src/a.rs::d".into(), SymbolChange::Deleted)]);
        assert_eq!(Row::Wrote { id: "src/a.rs::d", change: SymbolChange::Deleted, file: "src/a.rs" }.target(), Target::File("src/a.rs".into()), "a deleted symbol opens its file");
        assert_eq!(call_rows(&t, &index, 1, None).iter().map(Row::target).collect::<Vec<_>>(), vec![Target::File("src/a.rs".into()), Target::Span(2), Target::Span(3), Target::Span(6), Target::Span(7)]);
    }

    #[test]
    fn related_calls_share_the_file_and_the_trace_in_time_order() {
        let t = sample();
        let index = TraceIndex::new(&t);
        assert_eq!(related(&t, &index, 1), vec![2, 3, 6, 7]);
        assert_eq!(related(&t, &index, 4), Vec::<usize>::new(), "nothing else on b.rs");
        assert_eq!(index.root_of(7), Some(0));
        assert_eq!(index.end_of(&t, 5), Some(crate::time::parse_rfc3339_millis("2026-09-27T10:00:20Z").unwrap()), "a delegation ends with its agent");
    }
}
