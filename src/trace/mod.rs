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
pub mod summary;
pub mod touch;
pub mod view;

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::ingest::{
    AgentToolCall, BlockCounts, Effect, InferenceBlock, InferenceMeta, Prompt, SessionEvent,
    ToolFinished, Usage,
};
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
    /// A model response: the request, from when it was sent to its last
    /// block.
    Inference,
    /// Anything else: a shell command, a search that saw nothing, …
    Other,
}

impl SpanKind {
    /// A tool call: neither a prompt nor a model response. What a trace's
    /// call and failure counts count.
    pub fn is_call(self) -> bool {
        !matches!(self, SpanKind::Prompt | SpanKind::Inference)
    }
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
    /// Why it failed, in one line, when the result said.
    pub message: Option<String>,
    /// For a delegation: the subagent it started.
    pub child_agent: Option<Arc<str>>,
    /// What the call is for, in the agent's words (Bash's `description`).
    pub summary: Option<String>,
    /// The symbols the call read, as ambits credited them, each at its depth
    /// (see `app::Credit::listed`): what the views of a call's reads are
    /// built from, rather than its file and a name matched loosely.
    pub read: Vec<(String, crate::tracking::ReadDepth)>,
    /// For a model response, what it said of itself.
    pub inference: Option<Box<Inference>>,
    /// For a tool call, the model response that made it, by message id.
    pub issued_by: Option<Arc<str>>,
}

/// A model response, merged from the lines it was logged as.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inference {
    pub meta: InferenceMeta,
    /// The most each count reached over its lines.
    pub usage: Usage,
    /// When its first block was written: the time before it is the wait
    /// for the model, the time after it the rest of the response.
    pub first_block: u64,
    pub blocks: BlockCounts,
    /// The calls it made, by `tool_use` id, in order.
    pub calls: Vec<Arc<str>>,
    pub stop_reason: Option<Arc<str>>,
}

impl Inference {
    /// No line said how it stopped, so its output count is only where the
    /// stream began: a subagent's responses are often logged so.
    pub fn usage_partial(&self) -> bool {
        self.stop_reason.is_none() && !self.meta.synthetic
    }

    /// Its usage counts toward a trace's: a synthetic message's is nobody's.
    pub fn counts(&self) -> bool {
        !self.meta.synthetic
    }

    /// The model, without the family prefix every name shares
    /// (`opus-5-5` for `claude-opus-5-5`).
    pub fn short_model(&self) -> Option<&str> {
        let model = self.meta.model.as_deref()?;
        Some(model.strip_prefix("claude-").unwrap_or(model))
    }

    /// `opus-5-5 · tool_use ×3 · 105k ctx · 419 out`: what it was, how it
    /// ended, how full its context was, and what it wrote.
    pub fn label(&self) -> String {
        if let Some(error) = &self.meta.error {
            return format!("API error · {error}");
        }
        let mut parts = vec![self.short_model().unwrap_or("model").to_string()];
        match (&self.stop_reason, self.blocks.tool_use) {
            (Some(stop), n) if n > 1 => parts.push(format!("{stop} ×{n}")),
            (Some(stop), _) => parts.push(stop.to_string()),
            (None, n) if n > 0 => parts.push(format!("tool_use ×{n}")),
            (None, _) => {}
        }
        if self.counts() {
            if let Some(context) = self.usage.context() {
                parts.push(format!("{} ctx", tokens(context)));
            }
            if let Some(out) = self.usage.output {
                let at_least = if self.usage_partial() { "≥" } else { "" };
                parts.push(format!("{at_least}{} out", tokens(out)));
            }
        }
        parts.join(" · ")
    }
}

/// Model responses summed up: what they cost in tokens, and where the time
/// went between waiting on the model and running tools.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InferenceTotals {
    pub responses: usize,
    /// Responses whose output count is only partial (see
    /// [`Inference::usage_partial`]): `output` is then at least this much.
    pub partial: usize,
    /// Responses that were API errors, by span.
    pub errors: Vec<usize>,
    /// Responses per model, most first (ties by name).
    pub by_model: Vec<(Arc<str>, usize)>,
    /// Responses per stop reason, most first (ties by name).
    pub stop_reasons: Vec<(Arc<str>, usize)>,
    /// Summed over responses: what they were given and what they wrote.
    pub usage: Usage,
    /// The fullest any response's context was.
    pub peak_context: Option<u64>,
    /// Milliseconds some response was under way and no tool was running:
    /// a response that goes on after its tools ran is not counted twice.
    pub model_ms: u64,
    /// Milliseconds some tool call was running, delegations aside: they
    /// wait on an agent, whose own responses and calls are counted.
    pub tool_ms: u64,
}

impl InferenceTotals {
    /// The totals over `spans`, indices into `trace`.
    pub fn of(trace: &Trace, spans: &[usize]) -> Self {
        let mut out = InferenceTotals::default();
        let mut by_model: HashMap<Arc<str>, usize> = HashMap::new();
        let mut stops: HashMap<Arc<str>, usize> = HashMap::new();
        let (mut model, mut tools) = (Vec::new(), Vec::new());
        let add = |sum: &mut Option<u64>, n: Option<u64>| {
            if let Some(n) = n {
                *sum = Some(sum.unwrap_or(0) + n);
            }
        };
        for &i in spans {
            let s = &trace.spans[i];
            let end = s.end.unwrap_or(s.start);
            match &s.inference {
                Some(inf) => {
                    out.responses += 1;
                    model.push((s.start, end));
                    if inf.meta.error.is_some() {
                        out.errors.push(i);
                    }
                    if !inf.counts() {
                        continue;
                    }
                    out.partial += usize::from(inf.usage_partial());
                    if let Some(m) = &inf.meta.model {
                        *by_model.entry(m.clone()).or_default() += 1;
                    }
                    if let Some(stop) = &inf.stop_reason {
                        *stops.entry(stop.clone()).or_default() += 1;
                    }
                    let (u, sum) = (inf.usage, &mut out.usage);
                    add(&mut sum.input, u.input);
                    add(&mut sum.output, u.output);
                    add(&mut sum.thinking, u.thinking);
                    add(&mut sum.cache_read, u.cache_read);
                    add(&mut sum.cache_write, u.cache_write);
                    out.peak_context = out.peak_context.max(u.context());
                }
                None if s.kind.is_call() && s.kind != SpanKind::Delegate => {
                    tools.push((s.start, end))
                }
                None => {}
            }
        }
        let tools = union(tools);
        out.tool_ms = tools.iter().map(|(a, b)| b - a).sum();
        out.model_ms = union(model)
            .into_iter()
            .map(|(a, b)| (b - a) - overlap((a, b), &tools))
            .sum();
        let ranked = |counts: HashMap<Arc<str>, usize>| {
            let mut v: Vec<(Arc<str>, usize)> = counts.into_iter().collect();
            v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            v
        };
        out.by_model = ranked(by_model);
        out.stop_reasons = ranked(stops);
        out
    }
}

/// `spans` as disjoint intervals, in order.
fn union(mut spans: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    spans.sort_unstable();
    let mut out: Vec<(u64, u64)> = Vec::new();
    for (a, b) in spans {
        match out.last_mut() {
            Some(last) if a <= last.1 => last.1 = last.1.max(b),
            _ => out.push((a, b)),
        }
    }
    out
}

/// How much of `(a, b)` the disjoint `spans` cover.
fn overlap((a, b): (u64, u64), spans: &[(u64, u64)]) -> u64 {
    spans
        .iter()
        .map(|&(x, y)| y.min(b).saturating_sub(x.max(a)))
        .sum()
}

/// A token count, short: `419`, `7.1k`, `105k`, `1.2M`.
pub fn tokens(n: u64) -> String {
    match n {
        0..1_000 => n.to_string(),
        1_000..10_000 => format!("{:.1}k", n as f64 / 1e3),
        10_000..1_000_000 => format!("{}k", n / 1_000),
        _ => format!("{:.1}M", n as f64 / 1e6),
    }
}

impl Span {
    /// A short name: what the agent said the call is for (`Bash · Run the
    /// tests`), else the symbol, else the file, else the description; a
    /// prompt by its first line, cut at 80 characters; a model response by
    /// [`Inference::label`].
    pub fn name(&self) -> String {
        if let Some(inference) = &self.inference {
            return inference.label();
        }
        if self.kind == SpanKind::Prompt {
            let line = self.description.lines().next().unwrap_or_default();
            let cut: String = line.chars().take(80).collect();
            return if cut.len() < line.len() || self.description.lines().nth(1).is_some() {
                format!("{cut}…")
            } else {
                cut
            };
        }
        if let Some(summary) = &self.summary {
            return format!("{} · {summary}", self.tool);
        }
        match (&self.symbol, &self.file) {
            (Some(s), _) => format!("{} {s}", self.tool),
            (None, Some(f)) => format!("{} {f}", self.tool),
            _ if !self.description.is_empty() => self.description.clone(),
            _ => self.tool.to_string(),
        }
    }

    /// What a delegation was started for: its own description, else the
    /// one its tool's stanza made (`Agent: …`) without the tool's name.
    pub fn task(&self) -> String {
        self.summary.clone().unwrap_or_else(|| {
            self.description
                .trim_start_matches("Agent: ")
                .trim()
                .to_string()
        })
    }

    /// The files whose symbols the call read — for an `ambits show` or a
    /// search, which name no file of their own, several — in first-read order.
    pub fn read_files(&self) -> Vec<&str> {
        let mut out: Vec<&str> = Vec::new();
        for (id, _) in &self.read {
            let file = crate::symbols::split_id(id).0;
            if !out.contains(&file) {
                out.push(file);
            }
        }
        out
    }

    /// The symbols the call read in `file`, by name path.
    pub fn reads_in<'s>(
        &'s self,
        file: &'s str,
    ) -> impl Iterator<Item = (&'s str, crate::tracking::ReadDepth)> + 's {
        self.read.iter().filter_map(move |(id, depth)| {
            let (f, name) = crate::symbols::split_id(id);
            (f == file).then_some((name, *depth))
        })
    }
}

/// Something that happened at a moment rather than over time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstantKind {
    Compaction,
    Snapshot(String),
    /// A commit made during the session, by its committer time.
    Commit {
        sha: String,
        subject: String,
    },
}

impl InstantKind {
    /// `compaction`, `snapshot 3f9c…`, `commit 1f32c0b Subject`.
    pub fn label(&self) -> String {
        match self {
            InstantKind::Compaction => "compaction".into(),
            InstantKind::Snapshot(id) => format!("snapshot {}", id.get(..12).unwrap_or(id)),
            InstantKind::Commit { sha, subject } => {
                format!("commit {} {subject}", sha.get(..7).unwrap_or(sha))
            }
        }
    }
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
    /// Each model response, by message id: apart from `by_id`, so a tool
    /// result never finishes a response.
    by_message: HashMap<Arc<str>, usize>,
    /// The response that made each call, by `tool_use` id, for a call that
    /// starts after its response was read.
    issued: HashMap<Arc<str>, Arc<str>>,
    /// Each subagent's delegation.
    by_child: HashMap<Arc<str>, usize>,
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
        let Some(start) = crate::time::parse_rfc3339_millis(&call.timestamp_str) else {
            return;
        };
        let kind = if matches!(&*call.tool_name, "Agent" | "Task") {
            SpanKind::Delegate
        } else if call.effect == Effect::Write {
            SpanKind::Write
        } else if call.read_depth.is_seen() {
            SpanKind::Read(call.read_depth)
        } else {
            SpanKind::Other
        };
        let file = call
            .file_path
            .as_deref()
            .and_then(|p| crate::objects::project_file(p, project_root));
        if let Some(id) = &call.tool_use_id {
            self.by_id.insert(id.clone(), self.spans.len());
        }
        let issued_by = call
            .tool_use_id
            .as_ref()
            .and_then(|id| self.issued.get(id).cloned());
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
            message: None,
            child_agent: None,
            summary: call.summary.clone(),
            read: Vec::new(),
            inference: None,
            issued_by,
        });
    }

    /// A call's result arrived. A background delegation finishes twice —
    /// its launch, then its agent stopping — so the later end wins and an
    /// error from either sticks.
    pub fn finish(&mut self, f: &ToolFinished) {
        let Some(&ix) = self.by_id.get(&f.id) else {
            return;
        };
        let span = &mut self.spans[ix];
        if let Some(t) = crate::time::parse_rfc3339_millis(&f.timestamp) {
            span.end = Some(span.end.unwrap_or(0).max(t).max(span.start));
        }
        span.error |= f.error;
        if f.message.is_some() {
            span.message = f.message.clone();
        }
        if let Some(child) = &f.child_agent {
            span.child_agent = Some(child.clone());
            self.by_child.insert(child.clone(), ix);
        }
    }

    /// Call `id` read these symbols — when it was made, or, for a call
    /// credited by its result (a read, a search), when its result arrived.
    pub fn note_read(&mut self, id: &str, read: Vec<(String, crate::tracking::ReadDepth)>) {
        if read.is_empty() {
            return;
        }
        if let Some(&ix) = self.by_id.get(id) {
            self.spans[ix].read.extend(read);
        }
    }

    /// The delegation that started subagent `agent`.
    pub fn delegation_of(&self, agent: &str) -> Option<usize> {
        self.by_child.get(agent).copied()
    }

    /// A turn began. Its span's own end is its start; it lasts, in the
    /// tree, as long as the calls under it.
    pub fn prompt(&mut self, prompt: &Prompt) {
        let Some(t) = crate::time::parse_rfc3339_millis(&prompt.timestamp) else {
            return;
        };
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
            message: None,
            child_agent: None,
            summary: None,
            read: Vec::new(),
            inference: None,
            issued_by: None,
        });
    }

    /// A line of a model response arrived. Its first line opens the span,
    /// from when the request was sent; later ones carry it to their time,
    /// add their blocks, and raise its usage to the most any line gave.
    /// Calls and responses link each other whichever is read first.
    pub fn inference(&mut self, block: &InferenceBlock) {
        let Some(t) = crate::time::parse_rfc3339_millis(&block.timestamp) else {
            return;
        };
        let ix = match self.by_message.get(&block.message_id) {
            Some(&ix) => {
                let span = &mut self.spans[ix];
                span.end = Some(span.end.unwrap_or(t).max(t));
                let inference = span
                    .inference
                    .as_mut()
                    .expect("by_message holds only responses");
                inference.usage = inference.usage.most(block.usage);
                inference.blocks.add(block.blocks);
                inference.calls.extend(block.tool_uses.iter().cloned());
                inference.first_block = inference.first_block.min(t);
                if block.stop_reason.is_some() {
                    inference.stop_reason = block.stop_reason.clone();
                }
                inference.meta.thinking_ms = inference.meta.thinking_ms.max(block.meta.thinking_ms);
                ix
            }
            None => {
                let ix = self.spans.len();
                self.by_message.insert(block.message_id.clone(), ix);
                self.spans.push(Span {
                    id: Some(block.message_id.clone()),
                    agent: block.agent_id.clone(),
                    start: block.requested_at.filter(|&r| r <= t).unwrap_or(t),
                    end: Some(t),
                    tool: Arc::from("inference"),
                    kind: SpanKind::Inference,
                    file: None,
                    symbol: None,
                    description: String::new(),
                    error: block.meta.error.is_some(),
                    message: block.meta.error.clone(),
                    child_agent: None,
                    summary: None,
                    read: Vec::new(),
                    inference: Some(Box::new(Inference {
                        meta: block.meta.clone(),
                        usage: block.usage,
                        first_block: t,
                        blocks: block.blocks,
                        calls: block.tool_uses.clone(),
                        stop_reason: block.stop_reason.clone(),
                    })),
                    issued_by: None,
                });
                ix
            }
        };
        let message = self.spans[ix].id.clone().expect("a response has its id");
        for call in &block.tool_uses {
            self.issued.insert(call.clone(), message.clone());
            if let Some(&c) = self.by_id.get(call) {
                self.spans[c].issued_by = Some(message.clone());
            }
        }
    }

    /// The commits made during the session, replacing any found before:
    /// each scan sees them all.
    pub fn set_commits(&mut self, commits: &[crate::git::Commit]) {
        self.instants
            .retain(|i| !matches!(i.kind, InstantKind::Commit { .. }));
        for c in commits {
            self.instant(
                c.t,
                None,
                InstantKind::Commit {
                    sha: c.sha.clone(),
                    subject: c.subject.clone(),
                },
            );
        }
    }

    pub fn instant(&mut self, t: u64, agent: Option<Arc<str>>, kind: InstantKind) {
        self.instants.push(Instant { t, agent, kind });
    }

    /// Build a trace from a session's events, as the CLI does without a TUI.
    pub fn from_events(
        events: impl IntoIterator<Item = SessionEvent>,
        project_root: &Path,
    ) -> Self {
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
            SessionEvent::Compacted {
                timestamp,
                agent_id,
                ..
            } => {
                if let Some(t) = crate::time::parse_rfc3339_millis(timestamp) {
                    self.instant(t, Some(agent_id.clone()), InstantKind::Compaction);
                }
            }
            SessionEvent::SessionCleared => self.clear(),
            SessionEvent::Prompt(p) => self.prompt(p),
            SessionEvent::Inference(block) => self.inference(block),
            SessionEvent::Write(_) => {}
        }
    }

    /// Each span's number among the spans that are not model responses;
    /// `None` for a response. Exports know a span without an id of its own
    /// by it, so responses, which traces gained later, move no other span.
    pub fn numbering(&self) -> Vec<Option<usize>> {
        let mut next = 0;
        self.spans
            .iter()
            .map(|s| {
                (s.kind != SpanKind::Inference).then(|| {
                    next += 1;
                    next - 1
                })
            })
            .collect()
    }

    /// First and last moment anything happened, in milliseconds.
    pub fn range(&self) -> Option<(u64, u64)> {
        let starts = self
            .spans
            .iter()
            .map(|s| s.start)
            .chain(self.instants.iter().map(|i| i.t));
        let ends = self
            .spans
            .iter()
            .map(|s| s.end.unwrap_or(s.start))
            .chain(self.instants.iter().map(|i| i.t));
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
        for (i, s) in self
            .spans
            .iter()
            .enumerate()
            .filter(|(_, s)| s.kind == SpanKind::Prompt)
        {
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
                _ => delegated
                    .get(&*span.agent)
                    .copied()
                    .filter(|&p| p != i)
                    .or_else(|| prompt_of(span)),
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
            let kids: Vec<Node> = children
                .get(&ix)
                .into_iter()
                .flatten()
                .map(|&c| build(c, trace, children))
                .collect();
            let span = &trace.spans[ix];
            let own_end = span.end.unwrap_or(span.start);
            let end = kids.iter().map(|k| k.end).fold(own_end, u64::max);
            Node {
                span: ix,
                end,
                children: kids,
            }
        }
        roots
            .into_iter()
            .map(|r| build(r, self, &children))
            .collect()
    }

    /// The roots under an agent filter: every root, or the subtrees whose
    /// span belongs to that agent (prefix match, as `--agent` is
    /// elsewhere), in start order.
    pub fn roots<'a>(&self, tree: &'a [Node], filter: Option<&str>) -> Vec<&'a Node> {
        let Some(agent) = filter else {
            return tree.iter().collect();
        };
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

impl Node {
    /// This node's span and every span under it, depth first.
    pub fn spans(&self) -> Vec<usize> {
        let mut out = Vec::new();
        let mut stack = vec![self];
        while let Some(node) = stack.pop() {
            out.push(node.span);
            stack.extend(node.children.iter().rev());
        }
        out
    }
}

/// Span index → the root of its tree: which trace each span belongs to.
pub fn roots_by_span(tree: &[Node]) -> HashMap<usize, usize> {
    tree.iter()
        .flat_map(|root| root.spans().into_iter().map(move |s| (s, root.span)))
        .collect()
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
            message: None,
            child_agent: child.map(Arc::from),
            shown: Vec::new(),
            printed: Vec::new(),
            read: None,
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
        assert_eq!(
            delegation.end,
            crate::time::parse_rfc3339_millis(T9).unwrap()
        );
        let child = t.spans()[delegation.span]
            .child_agent
            .clone()
            .expect("its subagent");
        assert_eq!(
            t.delegation_of(&child),
            Some(delegation.span),
            "a subagent leads back to its delegation"
        );
    }

    /// A background delegation finishes at launch and again when its agent
    /// stops: the stop is its end, whatever order the two arrive in.
    #[test]
    fn a_background_delegation_ends_when_its_agent_stops() {
        let mut t = Trace::default();
        t.start(&call("main", "d", "Agent", T0), Path::new("/p"));
        t.finish(&done("main", "d", T1, Some("x")));
        t.finish(&ToolFinished {
            error: true,
            ..done("main", "d", T9, Some("x"))
        });
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

    fn block(
        message: &str,
        at: &str,
        requested: Option<&str>,
        tools: &[&str],
        stop: Option<&str>,
        output: u64,
    ) -> InferenceBlock {
        crate::helpers::inference_block("main", message, at, requested, tools, stop, output)
    }

    fn ms(ts: &str) -> u64 {
        crate::time::parse_rfc3339_millis(ts).unwrap()
    }

    /// A response logged over three lines is one span: from its request to
    /// its last line, with every line's blocks and the most usage any gave.
    #[test]
    fn the_lines_of_a_response_make_one_span() {
        let mut t = Trace::default();
        t.inference(&block("m1", T1, Some(T0), &[], None, 8));
        t.inference(&block("m1", T5, Some(T0), &["t1"], None, 300));
        t.inference(&block("m1", T9, Some(T0), &["t2"], Some("tool_use"), 120));
        let [span] = t.spans() else {
            panic!("one span")
        };
        assert_eq!(span.kind, SpanKind::Inference);
        assert_eq!((span.start, span.end), (ms(T0), Some(ms(T9))));
        let inference = span.inference.as_deref().unwrap();
        assert_eq!(inference.first_block, ms(T1));
        assert_eq!(inference.usage.output, Some(300), "the most, not the last");
        assert_eq!(inference.blocks.tool_use, 2);
        assert_eq!(inference.calls, [Arc::from("t1"), Arc::from("t2")]);
        assert_eq!(inference.stop_reason.as_deref(), Some("tool_use"));
        assert!(!inference.usage_partial());
    }

    /// A subagent's lines often never say how the response stopped: its
    /// output count is then only where the stream began.
    #[test]
    fn a_response_no_line_says_stopped_has_partial_usage() {
        let mut t = Trace::default();
        t.inference(&block("m1", T1, None, &["t1"], None, 8));
        let inference = t.spans()[0].inference.as_deref().unwrap();
        assert!(inference.usage_partial());
        assert_eq!(
            inference.label(),
            "opus-5-5 · tool_use ×1 · 1.0k ctx · ≥8 out"
        );
        assert_eq!(
            t.spans()[0].start,
            ms(T1),
            "no request time: it starts at its line"
        );
    }

    /// Live polls apply a poll's calls before its responses, a replay the
    /// other way round: the link is made either way.
    #[test]
    fn a_call_and_its_response_link_whichever_is_read_first() {
        let mut t = Trace::default();
        t.start(&call("main", "t1", "Read", T5), Path::new("/p"));
        t.inference(&block(
            "m1",
            T1,
            Some(T0),
            &["t1", "t2"],
            Some("tool_use"),
            9,
        ));
        t.start(&call("main", "t2", "Read", T5), Path::new("/p"));
        let issued: Vec<Option<&str>> = t
            .spans()
            .iter()
            .filter(|s| s.kind.is_call())
            .map(|s| s.issued_by.as_deref())
            .collect();
        assert_eq!(issued, [Some("m1"), Some("m1")]);
    }

    /// A response sits under its prompt beside the calls it made, and is
    /// not one of them.
    #[test]
    fn a_response_is_a_sibling_of_its_calls_and_not_a_call() {
        let mut t = Trace::default();
        t.prompt(&Prompt {
            agent_id: Arc::from("main"),
            timestamp: T0.into(),
            text: "go".into(),
        });
        t.inference(&block("m1", T1, Some(T0), &["t1"], Some("tool_use"), 9));
        t.start(&call("main", "t1", "Read", T1), Path::new("/p"));
        t.finish(&done("main", "t1", T5, None));
        t.inference(&block("m2", T9, Some(T5), &[], Some("end_turn"), 9));
        let tree = t.tree();
        let [prompt] = tree.as_slice() else {
            panic!("one root")
        };
        let kinds: Vec<SpanKind> = prompt
            .children
            .iter()
            .map(|c| t.spans()[c.span].kind)
            .collect();
        assert_eq!(
            kinds,
            [
                SpanKind::Inference,
                SpanKind::Read(ReadDepth::FullBody),
                SpanKind::Inference
            ]
        );
        assert_eq!(prompt.end, ms(T9), "the turn lasts until its last response");
        let summary = &view::traces(&t, None)[0];
        assert_eq!((summary.calls, summary.inference.responses), (1, 2));
    }

    /// Claude Code's own message for a failed request is an error whose
    /// usage counts for nothing.
    #[test]
    fn a_synthetic_api_error_fails_and_counts_no_usage() {
        let mut t = Trace::default();
        let mut error = block("e1", T1, None, &[], None, 0);
        error.meta.synthetic = true;
        error.meta.error = Some("authentication_failed".into());
        t.inference(&error);
        t.inference(&block("m1", T5, None, &[], Some("end_turn"), 40));
        let span = &t.spans()[0];
        assert!(span.error);
        assert_eq!(span.message.as_deref(), Some("authentication_failed"));
        assert_eq!(span.name(), "API error · authentication_failed");
        let totals = InferenceTotals::of(&t, &[0, 1]);
        assert_eq!((totals.responses, totals.errors.as_slice()), (2, &[0][..]));
        assert_eq!(totals.usage.output, Some(40));
        assert_eq!(totals.by_model, [(Arc::from("claude-opus-5-5"), 1)]);
    }

    /// Time under a response that its tools also ran in is the tools'; a
    /// delegation's time is its agent's, counted in that agent's spans.
    #[test]
    fn model_time_leaves_out_what_tools_ran_and_tool_time_leaves_out_delegations() {
        let mut t = Trace::default();
        // A response from 0 s to 9 s whose call ran from 1.5 s to 5 s.
        t.inference(&block("m1", T9, Some(T0), &["t1"], Some("tool_use"), 9));
        t.start(&call("main", "t1", "Read", T1), Path::new("/p"));
        t.finish(&done("main", "t1", T5, None));
        t.start(&call("main", "d", "Agent", T0), Path::new("/p"));
        t.finish(&done("main", "d", T9, Some("x")));
        let totals = InferenceTotals::of(&t, &[0, 1, 2]);
        assert_eq!(totals.tool_ms, 3_500);
        assert_eq!(totals.model_ms, 9_000 - 3_500);
    }

    #[test]
    fn token_counts_are_short() {
        let cases = [
            (419, "419"),
            (7_136, "7.1k"),
            (105_103, "105k"),
            (1_234_567, "1.2M"),
        ];
        for (n, short) in cases {
            assert_eq!(tokens(n), short);
        }
    }

    /// Responses are numbered out: a span without an id keeps its number.
    #[test]
    fn responses_move_no_other_span_in_the_numbering() {
        let mut t = Trace::default();
        t.inference(&block("m1", T0, None, &[], None, 1));
        t.prompt(&Prompt {
            agent_id: Arc::from("main"),
            timestamp: T1.into(),
            text: "go".into(),
        });
        t.inference(&block("m2", T5, None, &[], None, 1));
        t.start(&call("main", "t1", "Read", T9), Path::new("/p"));
        assert_eq!(t.numbering(), [None, Some(0), None, Some(1)]);
    }

    #[test]
    fn a_clear_empties_the_trace() {
        let mut t = Trace::default();
        t.apply(
            &SessionEvent::ToolCall(call("main", "t1", "Read", T0)),
            Path::new("/p"),
        );
        t.apply(&SessionEvent::SessionCleared, Path::new("/p"));
        assert!(t.is_empty());
    }
}
