//! The session as a trace: every tool call a span with a start and an end,
//! plus instants (compactions, snapshots, commits) — the model behind the
//! TUI's trace view and `ambits trace` (OTLP / Chrome trace export).
//!
//! A span starts at its `tool_use` and ends at its `tool_result`
//! ([`crate::ingest::ToolFinished`]); a call without a result yet is open.
//! A delegation (`Agent`, formerly `Task`) names the subagent it started,
//! and every span of that subagent is its child — which is what nests a
//! trace. Delegations are asynchronous, so a delegation's effective end is
//! the later of its own result and its subagent's last span.

pub mod export;
pub mod view;

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::ingest::{AgentToolCall, Effect, Prompt, SessionEvent, ToolFinished};
use crate::tracking::ReadDepth;

/// What a span did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanKind {
    /// A read, at the depth it earned.
    Read(ReadDepth),
    Write,
    /// Started a subagent.
    Delegate,
    /// A prompt the user typed: the parent of the turn's calls.
    Prompt,
    /// Anything else: a shell command, a search that saw nothing, …
    Other,
}

/// One tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Span {
    /// `tool_use_id`, when the log gave one.
    pub id: Option<Arc<str>>,
    pub agent: Arc<str>,
    /// Milliseconds since the epoch.
    pub start: u64,
    /// `None` while the call has no result.
    pub end: Option<u64>,
    pub tool: Arc<str>,
    pub kind: SpanKind,
    /// The file the call was about, project-relative.
    pub file: Option<String>,
    /// The symbol it named, if any.
    pub symbol: Option<String>,
    /// The call's own description (`Read src/app.rs`, a command, …).
    pub description: String,
    pub error: bool,
    /// For a delegation: the subagent it started.
    pub child_agent: Option<Arc<str>>,
}

impl Span {
    /// A short name: the symbol, else the file, else the description; a
    /// prompt by its first line, cut at 80 characters.
    pub fn name(&self) -> String {
        if self.kind == SpanKind::Prompt {
            let line = self.description.lines().next().unwrap_or_default();
            let cut: String = line.chars().take(80).collect();
            return if cut.len() < line.len() || self.description.lines().nth(1).is_some() { format!("{cut}…") } else { cut };
        }
        match (&self.symbol, &self.file) {
            (Some(s), _) => format!("{} {s}", self.tool),
            (None, Some(f)) => format!("{} {f}", self.tool),
            _ if !self.description.is_empty() => self.description.clone(),
            _ => self.tool.to_string(),
        }
    }
}

/// Something that happened at a moment rather than over time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstantKind {
    Compaction,
    Snapshot(String),
    Commit(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Instant {
    /// Milliseconds since the epoch.
    pub t: u64,
    /// The agent it belongs to; `None` for project-wide (commits, snapshots).
    pub agent: Option<Arc<str>>,
    pub kind: InstantKind,
}

/// A session's spans and instants.
#[derive(Debug, Default, Clone)]
pub struct Trace {
    spans: Vec<Span>,
    by_id: HashMap<Arc<str>, usize>,
    instants: Vec<Instant>,
}

impl Trace {
    pub fn spans(&self) -> &[Span] {
        &self.spans
    }

    pub fn instants(&self) -> &[Instant] {
        &self.instants
    }

    pub fn is_empty(&self) -> bool {
        self.spans.is_empty() && self.instants.is_empty()
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// A call started. Calls without a parseable timestamp are skipped: a
    /// span needs a place on the time axis.
    pub fn start(&mut self, call: &AgentToolCall, project_root: &Path) {
        let Some(start) = crate::time::parse_rfc3339_millis(&call.timestamp_str) else { return };
        let kind = if matches!(&*call.tool_name, "Agent" | "Task") {
            SpanKind::Delegate
        } else if call.effect == Effect::Write {
            SpanKind::Write
        } else if call.read_depth.is_seen() {
            SpanKind::Read(call.read_depth)
        } else {
            SpanKind::Other
        };
        let file = call.file_path.as_deref().and_then(|p| {
            let rel = p.strip_prefix(project_root).unwrap_or(p);
            crate::objects::project_rel(rel)
        });
        if let Some(id) = &call.tool_use_id {
            self.by_id.insert(id.clone(), self.spans.len());
        }
        self.spans.push(Span {
            id: call.tool_use_id.clone(),
            agent: call.agent_id.clone(),
            start,
            end: None,
            tool: call.tool_name.clone(),
            kind,
            file,
            symbol: call.target_symbol.clone(),
            description: call.description.clone(),
            error: false,
            child_agent: None,
        });
    }

    /// A call's result arrived. A background delegation finishes twice —
    /// its launch, then its agent stopping — so the later end wins and an
    /// error from either sticks.
    pub fn finish(&mut self, f: &ToolFinished) {
        let Some(&ix) = self.by_id.get(&f.id) else { return };
        let span = &mut self.spans[ix];
        if let Some(t) = crate::time::parse_rfc3339_millis(&f.timestamp) {
            span.end = Some(span.end.unwrap_or(0).max(t).max(span.start));
        }
        span.error |= f.error;
        if f.child_agent.is_some() {
            span.child_agent = f.child_agent.clone();
        }
    }

    /// A turn began. Its span's own end is its start; it lasts, in the
    /// tree, as long as the calls under it.
    pub fn prompt(&mut self, prompt: &Prompt) {
        let Some(t) = crate::time::parse_rfc3339_millis(&prompt.timestamp) else { return };
        self.spans.push(Span {
            id: None,
            agent: prompt.agent_id.clone(),
            start: t,
            end: Some(t),
            tool: Arc::from("prompt"),
            kind: SpanKind::Prompt,
            file: None,
            symbol: None,
            description: prompt.text.clone(),
            error: false,
            child_agent: None,
        });
    }

    pub fn instant(&mut self, t: u64, agent: Option<Arc<str>>, kind: InstantKind) {
        self.instants.push(Instant { t, agent, kind });
    }

    /// Build a trace from a session's events, as the CLI does without a TUI.
    pub fn from_events(events: impl IntoIterator<Item = SessionEvent>, project_root: &Path) -> Self {
        let mut trace = Self::default();
        for event in events {
            trace.apply(&event, project_root);
        }
        trace
    }

    /// Fold one session event in: the single mapping from ingest to trace.
    pub fn apply(&mut self, event: &SessionEvent, project_root: &Path) {
        match event {
            SessionEvent::ToolCall(call) => self.start(call, project_root),
            SessionEvent::ToolFinished(f) => self.finish(f),
            SessionEvent::Compacted { timestamp, agent_id, .. } => {
                if let Some(t) = crate::time::parse_rfc3339_millis(timestamp) {
                    self.instant(t, Some(agent_id.clone()), InstantKind::Compaction);
                }
            }
            SessionEvent::SessionCleared => self.clear(),
            SessionEvent::Prompt(p) => self.prompt(p),
            SessionEvent::Write(_) => {}
        }
    }

    /// First and last moment anything happened, in milliseconds.
    pub fn range(&self) -> Option<(u64, u64)> {
        let starts = self.spans.iter().map(|s| s.start).chain(self.instants.iter().map(|i| i.t));
        let ends = self.spans.iter().map(|s| s.end.unwrap_or(s.start)).chain(self.instants.iter().map(|i| i.t));
        Some((starts.min()?, ends.max()?))
    }

    /// The span tree, as OTel sees a trace: each prompt is a root, the
    /// parent of the calls its agent made from then until the next prompt;
    /// each delegation is the parent of every span its subagent made.
    /// Calls before any prompt are roots, and so is a subagent no
    /// delegation names (a truncated log) — never dropped.
    ///
    /// Parents come from timestamps, not arrival order, so a prompt read
    /// after the calls that follow it still gathers them.
    pub fn tree(&self) -> Vec<Node> {
        let delegated: HashMap<&str, usize> = self
            .spans
            .iter()
            .enumerate()
            .filter_map(|(i, s)| Some((s.child_agent.as_deref()?, i)))
            .collect();
        let mut prompts: HashMap<&str, Vec<(u64, usize)>> = HashMap::new();
        for (i, s) in self.spans.iter().enumerate().filter(|(_, s)| s.kind == SpanKind::Prompt) {
            prompts.entry(&s.agent).or_default().push((s.start, i));
        }
        for list in prompts.values_mut() {
            list.sort_unstable();
        }
        let prompt_of = |span: &Span| {
            let list = prompts.get(&*span.agent)?;
            let before = list.partition_point(|&(t, _)| t <= span.start);
            before.checked_sub(1).map(|p| list[p].1)
        };
        let mut children: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut roots = Vec::new();
        for (i, span) in self.spans.iter().enumerate() {
            let parent = match span.kind {
                SpanKind::Prompt => None,
                _ => delegated.get(&*span.agent).copied().filter(|&p| p != i).or_else(|| prompt_of(span)),
            };
            match parent {
                Some(parent) => children.entry(parent).or_default().push(i),
                None => roots.push(i),
            }
        }
        // Arrival order is not time order across kinds; the tree is.
        let start = |i: &usize| (self.spans[*i].start, *i);
        roots.sort_by_key(start);
        for kids in children.values_mut() {
            kids.sort_by_key(start);
        }
        fn build(ix: usize, trace: &Trace, children: &HashMap<usize, Vec<usize>>) -> Node {
            let kids: Vec<Node> = children.get(&ix).into_iter().flatten().map(|&c| build(c, trace, children)).collect();
            let span = &trace.spans[ix];
            let own_end = span.end.unwrap_or(span.start);
            let end = kids.iter().map(|k| k.end).fold(own_end, u64::max);
            Node { span: ix, end, children: kids }
        }
        roots.into_iter().map(|r| build(r, self, &children)).collect()
    }

    /// The roots under an agent filter: every root, or the subtrees whose
    /// span belongs to that agent (prefix match, as `--agent` is
    /// elsewhere), in start order.
    pub fn roots<'a>(&self, tree: &'a [Node], filter: Option<&str>) -> Vec<&'a Node> {
        let Some(agent) = filter else { return tree.iter().collect() };
        let mut out = Vec::new();
        let mut stack: Vec<&Node> = tree.iter().collect();
        while let Some(node) = stack.pop() {
            if self.spans[node.span].agent.starts_with(agent) {
                out.push(node);
            } else {
                stack.extend(node.children.iter());
            }
        }
        out.sort_by_key(|n| (self.spans[n.span].start, n.span));
        out
    }
}

/// A span in the tree. `end` is the effective end: for a delegation, the
/// later of its own result and its subagent's last span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    pub span: usize,
    pub end: u64,
    pub children: Vec<Node>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(agent: &str, id: &str, tool: &str, at: &str) -> AgentToolCall {
        let mut c = crate::helpers::tool_call(tool, "/p/src/a.rs", ReadDepth::FullBody);
        c.agent_id = Arc::from(agent);
        c.tool_use_id = Some(Arc::from(id));
        c.timestamp_str = at.to_string();
        c
    }

    fn done(agent: &str, id: &str, at: &str, child: Option<&str>) -> ToolFinished {
        ToolFinished {
            id: Arc::from(id),
            agent_id: Arc::from(agent),
            timestamp: at.to_string(),
            error: false,
            child_agent: child.map(Arc::from),
        }
    }

    const T0: &str = "2026-09-27T10:00:00.000Z";
    const T1: &str = "2026-09-27T10:00:01.500Z";
    const T5: &str = "2026-09-27T10:00:05.000Z";
    const T9: &str = "2026-09-27T10:00:09.000Z";

    #[test]
    fn a_call_and_its_result_make_a_span_and_a_call_alone_stays_open() {
        let mut t = Trace::default();
        t.start(&call("main", "t1", "Read", T0), Path::new("/p"));
        t.start(&call("main", "t2", "Read", T1), Path::new("/p"));
        t.finish(&done("main", "t1", T1, None));
        let [a, b] = t.spans() else { panic!() };
        assert_eq!(a.end.unwrap() - a.start, 1500);
        assert_eq!(a.file.as_deref(), Some("src/a.rs"));
        assert_eq!(b.end, None);
    }

    /// A delegation parents its subagent's spans, and lasts until they end.
    #[test]
    fn a_delegation_nests_its_subagent_and_spans_its_work() {
        let mut t = Trace::default();
        t.start(&call("main", "d", "Agent", T0), Path::new("/p"));
        t.finish(&done("main", "d", T0, Some("agent-x")));
        t.start(&call("agent-x", "x1", "Read", T1), Path::new("/p"));
        t.finish(&done("agent-x", "x1", T9, None));
        t.start(&call("main", "m1", "Read", T5), Path::new("/p"));
        let tree = t.tree();
        assert_eq!(tree.len(), 2, "the delegation and main's own read");
        let delegation = &tree[0];
        assert_eq!(t.spans()[delegation.span].kind, SpanKind::Delegate);
        assert_eq!(delegation.children.len(), 1);
        assert_eq!(delegation.end, crate::time::parse_rfc3339_millis(T9).unwrap());
    }

    /// A background delegation finishes at launch and again when its agent
    /// stops: the stop is its end, whatever order the two arrive in.
    #[test]
    fn a_background_delegation_ends_when_its_agent_stops() {
        let mut t = Trace::default();
        t.start(&call("main", "d", "Agent", T0), Path::new("/p"));
        t.finish(&done("main", "d", T1, Some("x")));
        t.finish(&ToolFinished { error: true, ..done("main", "d", T9, Some("x")) });
        t.finish(&done("main", "d", T5, None));
        let span = &t.spans()[0];
        assert_eq!(span.end, crate::time::parse_rfc3339_millis(T9));
        assert!(span.error, "a failed stop marks the delegation");
        assert_eq!(span.child_agent.as_deref(), Some("x"));
    }

    #[test]
    fn a_subagent_no_delegation_names_hangs_off_the_root() {
        let mut t = Trace::default();
        t.start(&call("agent-y", "y1", "Read", T1), Path::new("/p"));
        assert_eq!(t.tree().len(), 1);
    }

    #[test]
    fn a_clear_empties_the_trace() {
        let mut t = Trace::default();
        t.apply(&SessionEvent::ToolCall(call("main", "t1", "Read", T0)), Path::new("/p"));
        t.apply(&SessionEvent::SessionCleared, Path::new("/p"));
        assert!(t.is_empty());
    }
}
