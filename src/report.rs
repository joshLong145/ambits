//! `ambits report`: one session as a single JSON document — its trace, what
//! each agent read of every symbol, and every write with where it stands and
//! which commit it landed in. Built for a viewer that has neither the logs
//! nor the checkout, so everything it shows is in the document.
//!
//! Like a snapshot (spec §9.6) it carries no file contents and no absolute
//! paths: symbol ids, project-relative paths, the project directory's name.
//! It does carry what the session logs said about each call — a shell
//! command, the first line of a prompt, an error message — which a harness
//! that publishes it should screen for secrets first.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use color_eyre::eyre::Result;
use serde::Serialize;

use crate::journal::DepthDto;
use crate::linkage::{Landed, Resolver};
use crate::parser::ParserRegistry;
use crate::symbols::{split_id, ProjectTree};
use crate::trace::{InstantKind, Node, SpanKind, Trace};
use crate::tracking::agents::AgentTree;
use crate::tracking::ContextLedger;
use crate::writes::{FileContents, Level, Status, WriteRecord};

/// Bumped on any breaking change to the JSON shape.
pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Serialize)]
pub struct Report {
    pub schema_version: u32,
    pub ambits_version: &'static str,
    /// RFC 3339, UTC.
    pub generated_at: String,
    pub session_id: String,
    pub project: Project,
    pub totals: Totals,
    /// What `--max-bytes` cut, in the order it cut it; empty when nothing was.
    pub truncated: Vec<&'static str>,
    pub agents: Vec<Agent>,
    pub spans: Vec<SpanDto>,
    pub instants: Vec<InstantDto>,
    pub files: Vec<FileDto>,
    pub writes: Vec<WriteDto>,
}

#[derive(Debug, Serialize)]
pub struct Project {
    /// The project directory's own name, never its path.
    pub name: String,
    /// `HEAD` when the report was made; `None` outside a repository.
    pub head: Option<String>,
}

#[derive(Debug, Default, Serialize, PartialEq, Eq)]
pub struct Totals {
    pub files: usize,
    pub symbols: usize,
    /// Symbols some agent read at any depth.
    pub symbols_read: usize,
    /// Symbols some agent read in full.
    pub symbols_read_full: usize,
    /// Symbols some write touched, created or removed.
    pub symbols_written: usize,
    pub writes: usize,
    pub spans: usize,
    pub agents: usize,
}

#[derive(Debug, Serialize)]
pub struct Agent {
    pub id: String,
    pub label: String,
    /// `None` for the session's own agent.
    pub parent: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SpanDto {
    /// The span's id, which `parent` and a write's `span` refer to. Ids are
    /// positions in the session's trace, so they are not contiguous once
    /// `truncated` names `other_spans`.
    pub i: usize,
    pub parent: Option<usize>,
    pub agent: String,
    /// `prompt`, `read`, `write`, `delegate` or `other`.
    pub kind: &'static str,
    pub tool: String,
    pub name: String,
    /// The call's own description: a command, a search, the prompt's text.
    pub detail: String,
    pub file: Option<String>,
    pub symbol: Option<String>,
    /// Milliseconds since the epoch.
    pub start_ms: u64,
    /// `None` while the call had no result.
    pub end_ms: Option<u64>,
    pub error: bool,
    pub message: Option<String>,
    /// For a read, the depth it earned.
    pub depth: Option<DepthDto>,
    /// The symbols it was credited with reading, each at its depth.
    pub read: Vec<(String, DepthDto)>,
    /// How many symbols it read, kept when `read` is truncated away.
    pub read_count: usize,
    /// For a delegation, the agent it started.
    pub child_agent: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct InstantDto {
    pub t_ms: u64,
    pub agent: Option<String>,
    /// `compaction`, `commit` or `snapshot`.
    pub kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct FileDto {
    pub path: String,
    pub total_lines: usize,
    pub total_symbols: usize,
    pub symbols_read: usize,
    pub symbols_written: usize,
    /// Every symbol, depth first, for a file something read or wrote; empty
    /// for the rest, whose counts are enough to draw them.
    pub symbols: Vec<SymbolDto>,
}

#[derive(Debug, Serialize)]
pub struct SymbolDto {
    pub id: String,
    pub name: String,
    /// The language's own word for it: `fn`, `class`, `def`, …
    pub kind: &'static str,
    /// Nesting level: 0 for a top-level symbol.
    pub level: usize,
    /// 1-based and inclusive.
    pub lines: [u32; 2],
    /// Each agent's depth, for the agents that read it.
    pub reads: BTreeMap<String, DepthDto>,
    /// The `op`s of the writes that touched, created or removed it.
    pub writes: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct WriteDto {
    pub op: String,
    pub agent: String,
    pub time: String,
    pub tool: String,
    pub file: String,
    pub level: Level,
    pub outside_symbols: bool,
    /// The innermost symbols it changed, each with whether that is still
    /// what is on disk.
    pub symbols: Vec<WrittenSymbol>,
    pub created: Vec<String>,
    pub removed: Vec<String>,
    /// Whether this write's version of the file is still what is on disk.
    pub status: Status,
    pub landed: Landed,
    /// The span of the call that made it, if the trace has it.
    pub span: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct WrittenSymbol {
    pub id: String,
    pub status: Status,
}

/// What a report is made from: a session replayed into its ledger and trace,
/// and the writes attributed for it.
pub struct Inputs<'a> {
    pub project_root: &'a Path,
    pub session_id: &'a str,
    pub tree: &'a ProjectTree,
    pub ledger: &'a ContextLedger,
    pub trace: &'a Trace,
    pub agents: &'a AgentTree,
    pub writes: Vec<WriteRecord>,
    pub registry: &'a ParserRegistry,
}

/// The report for `inputs`, before any size limit.
pub fn build(inputs: Inputs<'_>) -> Result<Report> {
    let Inputs { project_root, session_id, tree, ledger, trace, agents, writes, registry } = inputs;

    let spans = spans(trace);
    let span_of_op: HashMap<&str, usize> =
        trace.spans().iter().enumerate().filter_map(|(i, s)| Some((s.id.as_deref()?, i))).collect();

    let mut writes = writes;
    writes.sort_by(|a, b| a.recency().cmp(&b.recency()));
    let mut resolver = Resolver::new(project_root);
    let mut file_now: HashMap<String, Option<FileContents>> = HashMap::new();
    let mut write_dtos = Vec::with_capacity(writes.len());
    for w in &writes {
        let now = file_now.entry(w.file.clone()).or_insert_with(|| {
            crate::objects::read_regular(&project_root.join(&w.file)).map(|bytes| FileContents::read(&w.file, &bytes, registry))
        });
        let (status, symbols) = match now {
            Some(now) => (
                now.file_status(w),
                w.syms.iter().map(|(id, _)| WrittenSymbol { id: id.clone(), status: now.symbol_status(id, w) }).collect(),
            ),
            None => (Status::Removed, w.syms.iter().map(|(id, _)| WrittenSymbol { id: id.clone(), status: Status::Removed }).collect()),
        };
        write_dtos.push(WriteDto {
            op: w.op.clone(),
            agent: w.a.clone(),
            time: w.t.clone(),
            tool: w.tool.clone(),
            file: w.file.clone(),
            level: w.level,
            outside_symbols: w.outside_symbols,
            symbols,
            created: w.created.clone(),
            removed: w.removed.clone(),
            status,
            landed: crate::linkage::landed(resolver.as_mut(), w, &|_| true)?,
            span: span_of_op.get(w.op.as_str()).copied(),
        });
    }

    let mut written: HashMap<&str, Vec<String>> = HashMap::new();
    for w in &writes {
        for id in w.syms.iter().map(|(id, _)| id).chain(&w.removed) {
            written.entry(id.as_str()).or_default().push(w.op.clone());
        }
    }

    let mut totals = Totals { files: tree.files.len(), spans: spans.len(), writes: write_dtos.len(), ..Totals::default() };
    let mut files = Vec::with_capacity(tree.files.len());
    for file in &tree.files {
        let path = crate::objects::normalize_path(&file.file_path.to_string_lossy());
        let all = file.walk();
        let reads_of = |id: &str| -> BTreeMap<String, DepthDto> {
            ledger
                .entries
                .get(id)
                .map(|e| e.agent_depths.iter().filter(|(_, d)| d.is_seen()).map(|(a, d)| (a.clone(), DepthDto::from(*d))).collect())
                .unwrap_or_default()
        };
        let symbols: Vec<SymbolDto> = all
            .iter()
            .map(|s| SymbolDto {
                id: s.id.clone(),
                name: s.name.to_string(),
                kind: s.label,
                level: split_id(&s.id).1.matches('/').count(),
                lines: [s.line_range.start, s.line_range.end],
                reads: reads_of(&s.id),
                writes: written.get(s.id.as_str()).cloned().unwrap_or_default(),
            })
            .collect();
        let symbols_read = symbols.iter().filter(|s| !s.reads.is_empty()).count();
        let symbols_written = symbols.iter().filter(|s| !s.writes.is_empty()).count();
        totals.symbols += symbols.len();
        totals.symbols_read += symbols_read;
        totals.symbols_read_full += symbols.iter().filter(|s| s.reads.values().any(|d| *d == DepthDto::FullBody)).count();
        totals.symbols_written += symbols_written;
        let active = symbols_read > 0 || symbols_written > 0 || writes.iter().any(|w| w.file == path);
        files.push(FileDto {
            path,
            total_lines: file.total_lines,
            total_symbols: symbols.len(),
            symbols_read,
            symbols_written,
            symbols: if active { symbols } else { Vec::new() },
        });
    }

    let mut agent_list: Vec<Agent> = agents
        .agents
        .values()
        .map(|a| Agent { id: a.id.clone(), label: a.label.clone(), parent: a.parent_id.clone() })
        .collect();
    for span in trace.spans() {
        if !agent_list.iter().any(|a| a.id == *span.agent) {
            agent_list.push(Agent { id: span.agent.to_string(), label: span.agent.to_string(), parent: None });
        }
    }
    agent_list.sort_by(|a, b| (a.parent.is_some(), &a.id).cmp(&(b.parent.is_some(), &b.id)));
    totals.agents = agent_list.len();

    Ok(Report {
        schema_version: SCHEMA_VERSION,
        ambits_version: env!("CARGO_PKG_VERSION"),
        generated_at: crate::time::now_rfc3339(),
        session_id: session_id.to_string(),
        project: Project {
            name: project_root.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
            head: crate::git::git(project_root, &["rev-parse", "HEAD"])
                .and_then(|out| String::from_utf8(out).ok())
                .map(|s| s.trim().to_string())
                .filter(|s| crate::git::is_commit_id(s)),
        },
        totals,
        truncated: Vec::new(),
        agents: agent_list,
        spans,
        instants: trace.instants().iter().map(instant).collect(),
        files,
        writes: write_dtos,
    })
}

fn spans(trace: &Trace) -> Vec<SpanDto> {
    let mut parent: HashMap<usize, usize> = HashMap::new();
    fn walk(node: &Node, parent: &mut HashMap<usize, usize>) {
        for child in &node.children {
            parent.insert(child.span, node.span);
            walk(child, parent);
        }
    }
    for root in trace.tree() {
        walk(&root, &mut parent);
    }
    trace
        .spans()
        .iter()
        .enumerate()
        .map(|(i, s)| SpanDto {
            i,
            parent: parent.get(&i).copied(),
            agent: s.agent.to_string(),
            kind: match s.kind {
                SpanKind::Prompt => "prompt",
                SpanKind::Read(_) => "read",
                SpanKind::Write => "write",
                SpanKind::Delegate => "delegate",
                SpanKind::Other => "other",
            },
            tool: s.tool.to_string(),
            name: s.name(),
            detail: s.description.clone(),
            file: s.file.clone(),
            symbol: s.symbol.clone(),
            start_ms: s.start,
            end_ms: s.end,
            error: s.error,
            message: s.message.clone(),
            depth: match s.kind {
                SpanKind::Read(d) => Some(d.into()),
                _ => None,
            },
            read: s.read.iter().map(|(id, d)| (id.clone(), DepthDto::from(*d))).collect(),
            read_count: s.read.len(),
            child_agent: s.child_agent.as_deref().map(str::to_string),
        })
        .collect()
}

fn instant(i: &crate::trace::Instant) -> InstantDto {
    let (kind, sha, subject) = match &i.kind {
        InstantKind::Compaction => ("compaction", None, None),
        InstantKind::Snapshot(id) => ("snapshot", Some(id.clone()), None),
        InstantKind::Commit { sha, subject } => ("commit", Some(sha.clone()), Some(subject.clone())),
    };
    InstantDto { t_ms: i.t, agent: i.agent.as_deref().map(str::to_string), kind, sha, subject }
}

/// Cut `report` until its JSON fits in `max_bytes`, recording each cut in
/// `truncated`: first the symbol lists of files nothing wrote, then every
/// span's list of symbols read (its `read_count` stays), then the spans that
/// neither read nor wrote. Writes, totals and the remaining spans are never
/// cut, so a report can still exceed the limit; the caller sees `truncated`.
pub fn fit(report: &mut Report, max_bytes: usize) -> Result<()> {
    let size = |r: &Report| serde_json::to_vec(r).map(|v| v.len());
    if size(report)? <= max_bytes {
        return Ok(());
    }
    for file in report.files.iter_mut().filter(|f| f.symbols_written == 0) {
        file.symbols.clear();
    }
    report.truncated.push("unwritten_file_symbols");
    if size(report)? <= max_bytes {
        return Ok(());
    }
    for span in &mut report.spans {
        span.read.clear();
    }
    report.truncated.push("span_reads");
    if size(report)? <= max_bytes {
        return Ok(());
    }
    // Ids stay as they are, so `parent` and a write's `span` still resolve.
    let keep = |s: &SpanDto| matches!(s.kind, "prompt" | "read" | "write" | "delegate");
    report.spans.retain(keep);
    report.truncated.push("other_spans");
    Ok(())
}

