//! `ambits report`: one session as a single JSON document — its trace, what
//! each agent read of every symbol, and every write with where it stands and
//! which commit it landed in. Built for a viewer that has neither the logs
//! nor the checkout, so everything it shows is in the document.
//!
//! It names files by project-relative path, path arguments included, and
//! the project by its directory's name; a command or a call's description
//! still says whatever path it said. Unlike a snapshot (spec §9.6) it
//! does carry file contents, as the session logs recorded them: each call's
//! arguments, every write's diff, the text every read returned and every
//! other call's output, besides what the logs said of each call (a shell
//! command, a prompt's first line, an error). A report is as sensitive as
//! the session's transcript, and a harness that publishes one should screen
//! it for secrets first.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use color_eyre::eyre::Result;
use serde::{Deserialize, Serialize};

use crate::ingest::content::{CallContent, CallDetail, ContentKind, DiffLine};
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
    /// The harness's own timeline, from `--phases`; absent when none was given.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phases: Option<Phases>,
}

/// A timeline the harness that ran the session supplies, drawn as lanes of
/// its own beside the agents': its phases, turns, or whatever it timed.
/// ambits does not interpret the names; it only checks the times.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Phases {
    pub lanes: Vec<PhaseLane>,
    #[serde(default)]
    pub markers: Vec<PhaseMarker>,
    /// The harness could not record all of it, so a missing phase is not
    /// evidence that it did not run.
    #[serde(default)]
    pub incomplete: bool,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhaseLane {
    pub name: String,
    pub spans: Vec<PhaseSpan>,
    /// The page lists each call under the shortest of this lane's spans
    /// that was open when the call started.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub nest: bool,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhaseSpan {
    pub name: String,
    /// Epoch ms, as span times are.
    pub start_ms: u64,
    pub end_ms: u64,
    #[serde(default)]
    pub attrs: Vec<PhaseAttr>,
}

/// A moment in the harness's timeline, such as a failure it signalled.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhaseMarker {
    pub t_ms: u64,
    pub name: String,
    #[serde(default)]
    pub attrs: Vec<PhaseAttr>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhaseAttr {
    pub key: String,
    pub value: String,
}

impl Phases {
    /// The timeline in `json`, refused when a span ends before it starts or a
    /// lane has no name.
    pub fn parse(json: &str) -> Result<Self> {
        let phases: Self = serde_json::from_str(json).map_err(|e| color_eyre::eyre::eyre!("not a phases document: {e}"))?;
        for lane in &phases.lanes {
            if lane.name.trim().is_empty() {
                color_eyre::eyre::bail!("a phases lane has no name");
            }
            if let Some(s) = lane.spans.iter().find(|s| s.end_ms < s.start_ms) {
                color_eyre::eyre::bail!("phase {:?} in lane {:?} ends before it starts", s.name, lane.name);
            }
        }
        Ok(phases)
    }
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
    /// Its arguments as the log recorded them; file contents it wrote are
    /// summed up here, since `content` shows them.
    pub args: Vec<ArgDto>,
    /// What it wrote, read or returned, as the log recorded it.
    pub content: Option<ContentDto>,
}

#[derive(Debug, Serialize)]
pub struct ArgDto {
    pub key: String,
    pub value: String,
    /// Holds file contents the call wrote, summed up in `value`.
    pub bulk: bool,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentDto {
    /// A write's change. `exact` when the hunks are the tool's own patch,
    /// with line numbers; else what the call asked for. `cut` lines were
    /// left out.
    Change { exact: bool, cut: usize, hunks: Vec<HunkDto> },
    /// The text a read returned, its first line numbered `start`; `cut`
    /// lines were left out.
    Read { start: u32, lines: Vec<String>, cut: usize },
    /// What any other call returned: a command's output, a search's matches,
    /// a subagent's answer. `cut` lines were left out.
    Output { lines: Vec<String>, cut: usize },
}

#[derive(Debug, Serialize)]
pub struct HunkDto {
    pub old_start: Option<u32>,
    pub new_start: Option<u32>,
    /// Each line as `[op, text]`, `op` one of ` `, `+`, `-`.
    pub lines: Vec<(&'static str, String)>,
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
    /// Each call's arguments and content, by `tool_use_id`.
    pub details: HashMap<String, CallDetail>,
    pub phases: Option<Phases>,
}

/// What `content` each span kind carries, as the TUI loads it; `None` for a
/// prompt, which has no call.
pub fn content_kind(kind: SpanKind) -> Option<ContentKind> {
    match kind {
        SpanKind::Read(_) => Some(ContentKind::Read),
        SpanKind::Write => Some(ContentKind::Write),
        SpanKind::Delegate | SpanKind::Other => Some(ContentKind::Other),
        SpanKind::Prompt => None,
    }
}

/// The report for `inputs`, before any size limit.
pub fn build(inputs: Inputs<'_>) -> Result<Report> {
    let Inputs { project_root, session_id, tree, ledger, trace, agents, writes, registry, details, phases } = inputs;

    let spans = spans(trace, &details, project_root);
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
        ambits_version: crate::version(),
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
        phases,
    })
}

/// Arguments that name a file; one inside the project is shown relative to
/// it, as every other path in the report is.
const PATH_ARGS: &[&str] = &["file_path", "notebook_path", "path", "relative_path"];

fn arg_value(key: &str, value: &str, project_root: &Path) -> String {
    if !PATH_ARGS.contains(&key) {
        return value.to_string();
    }
    Path::new(value)
        .strip_prefix(project_root)
        .map_or_else(|_| value.to_string(), |rel| crate::objects::normalize_path(&rel.to_string_lossy()))
}

fn spans(trace: &Trace, details: &HashMap<String, CallDetail>, project_root: &Path) -> Vec<SpanDto> {
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
            args: s
                .id
                .as_deref()
                .and_then(|id| details.get(id))
                .map(|d| d.args.iter().map(|a| ArgDto { key: a.key.clone(), value: arg_value(&a.key, &a.value, project_root), bulk: a.bulk }).collect())
                .unwrap_or_default(),
            content: s.id.as_deref().and_then(|id| details.get(id)).and_then(|d| d.content.as_ref()).and_then(content),
        })
        .collect()
}

fn content(c: &CallContent) -> Option<ContentDto> {
    match c {
        CallContent::Change { hunks, exact, cut } => Some(ContentDto::Change {
            exact: *exact,
            cut: *cut,
            hunks: hunks
                .iter()
                .map(|h| HunkDto {
                    old_start: h.old_start,
                    new_start: h.new_start,
                    lines: h
                        .lines
                        .iter()
                        .map(|l| match l {
                            DiffLine::Same(t) => (" ", t.clone()),
                            DiffLine::Added(t) => ("+", t.clone()),
                            DiffLine::Removed(t) => ("-", t.clone()),
                        })
                        .collect(),
                })
                .collect(),
        }),
        CallContent::Read { start, lines, cut } => Some(ContentDto::Read { start: *start, lines: lines.clone(), cut: *cut }),
        CallContent::Output { lines, cut } => Some(ContentDto::Output { lines: lines.clone(), cut: *cut }),
    }
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
/// `truncated`, in this order: the symbol lists of files nothing wrote;
/// the text reads returned; every other call's output; every span's list
/// of symbols read (its
/// `read_count` stays); every call's arguments; the attributes of every
/// phase and marker; every write's diff; the spans that neither read, wrote
/// nor delegated. Writes, totals, phase times and the remaining spans are
/// never cut, so a report can still exceed the limit; the caller sees
/// `truncated`.
pub fn fit(report: &mut Report, max_bytes: usize) -> Result<()> {
    type Cut = fn(&mut Report);
    let cuts: [(&'static str, Cut); 8] = [
        ("unwritten_file_symbols", |r| r.files.iter_mut().filter(|f| f.symbols_written == 0).for_each(|f| f.symbols.clear())),
        ("read_text", |r| r.spans.iter_mut().filter(|s| matches!(s.content, Some(ContentDto::Read { .. }))).for_each(|s| s.content = None)),
        ("output", |r| r.spans.iter_mut().filter(|s| matches!(s.content, Some(ContentDto::Output { .. }))).for_each(|s| s.content = None)),
        ("span_reads", |r| r.spans.iter_mut().for_each(|s| s.read.clear())),
        ("arguments", |r| r.spans.iter_mut().for_each(|s| s.args.clear())),
        ("phase_attrs", |r| {
            if let Some(p) = r.phases.as_mut() {
                p.lanes.iter_mut().flat_map(|l| l.spans.iter_mut()).for_each(|s| s.attrs.clear());
                p.markers.iter_mut().for_each(|m| m.attrs.clear());
            }
        }),
        ("diffs", |r| r.spans.iter_mut().for_each(|s| s.content = None)),
        // Ids stay as they are, so `parent` and a write's `span` still resolve.
        ("other_spans", |r| r.spans.retain(|s| matches!(s.kind, "prompt" | "read" | "write" | "delegate"))),
    ];
    let size = |r: &Report| serde_json::to_vec(r).map(|v| v.len());
    for (name, cut) in cuts {
        if size(report)? <= max_bytes {
            break;
        }
        cut(report);
        report.truncated.push(name);
    }
    Ok(())
}

const VIEW: &str = include_str!("report_view.html");

/// The report page for `json`, a document `report` printed: one HTML file
/// that loads nothing and runs only its own script, the document embedded.
pub fn render_html(json: &str) -> Result<String> {
    let value: serde_json::Value = serde_json::from_str(json).map_err(|e| color_eyre::eyre::eyre!("not a JSON report: {e}"))?;
    match value.get("schema_version").and_then(serde_json::Value::as_u64) {
        Some(v) if v == u64::from(SCHEMA_VERSION) => {}
        other => color_eyre::eyre::bail!("not an ambits report of schema version {SCHEMA_VERSION} (found {other:?})"),
    }
    let csp = format!(
        "default-src 'none'; script-src '{}'; style-src '{}'; img-src data:; base-uri 'none'; form-action 'none'",
        csp_hash(inline(VIEW, "<script>")),
        csp_hash(inline(VIEW, "<style>")),
    );
    let data = embeddable(&serde_json::to_string(&value)?);
    Ok(VIEW.replacen("{{CSP}}", &csp, 1).replacen("{{DATA}}", &data, 1))
}

/// The text between `open` and the next closing tag of the same element.
fn inline<'a>(page: &'a str, open: &str) -> &'a str {
    let name: String = open[1..].chars().take_while(|c| c.is_ascii_alphanumeric()).collect();
    let close = format!("</{name}>");
    let start = page.find(open).expect("the template has the element") + open.len();
    let end = start + page[start..].find(&close).expect("the element is closed");
    &page[start..end]
}

fn csp_hash(text: &str) -> String {
    use sha2::Digest as _;
    format!("sha256-{}", base64(&sha2::Sha256::digest(text.as_bytes())))
}

/// JSON safe inside a `<script>` element: nothing in it can close the
/// element or open a comment, and JavaScript reads it unchanged. Outside
/// strings JSON has none of these characters, so escaping is always valid.
fn embeddable(json: &str) -> String {
    let mut out = String::with_capacity(json.len());
    for c in json.chars() {
        match c {
            '<' => out.push_str("\\u003c"),
            '>' => out.push_str("\\u003e"),
            '&' => out.push_str("\\u0026"),
            '\u{2028}' => out.push_str("\\u2028"),
            '\u{2029}' => out.push_str("\\u2029"),
            c => out.push(c),
        }
    }
    out
}

fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = (u32::from(chunk[0]) << 16) | (u32::from(*chunk.get(1).unwrap_or(&0)) << 8) | u32::from(*chunk.get(2).unwrap_or(&0));
        for (i, shift) in [18, 12, 6, 0].into_iter().enumerate() {
            out.push(if i <= chunk.len() { ALPHABET[(n >> shift) as usize & 63] as char } else { '=' });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal(detail: &str) -> String {
        serde_json::json!({"schema_version": SCHEMA_VERSION, "session_id": "s", "detail": detail}).to_string()
    }

    #[test]
    fn base64_matches_the_standard_padding() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn the_policy_allows_exactly_the_pages_own_script_and_style() {
        let page = render_html(&minimal("x")).unwrap();
        let script = csp_hash(inline(&page, "<script>"));
        let style = csp_hash(inline(&page, "<style>"));
        assert!(page.contains(&format!("script-src '{script}'")), "the script hash is stale");
        assert!(page.contains(&format!("style-src '{style}'")), "the style hash is stale");
        assert!(page.contains("default-src 'none'"));
    }

    #[test]
    fn data_cannot_close_its_script_element() {
        let hostile = "</script><script>alert(1)</script><!-- \u{2028} & >";
        let page = render_html(&minimal(hostile)).unwrap();
        assert_eq!(page.matches("</script>").count(), 2, "only the template's own two elements close");
        assert!(!page.contains("<!--"));
        let embedded = inline(&page, "<script id=\"report-data\" type=\"application/json\">");
        let back: serde_json::Value = serde_json::from_str(embedded).unwrap();
        assert_eq!(back["detail"], hostile, "the escaped JSON parses back to the same text");
    }

    #[test]
    fn only_a_report_of_this_schema_is_rendered() {
        assert!(render_html("not json").is_err());
        assert!(render_html(r#"{"schema_version": 99}"#).is_err());
        assert!(render_html(r#"{"session_id": "s"}"#).is_err());
    }
}
