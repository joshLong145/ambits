//! Content search with symbol attribution, for `ambits find`.
//!
//! ## Why this is shaped like ripgrep
//!
//! The interface is not ours to invent. Claude Code's `Grep` tool is
//! ripgrep-backed, so every agent that reaches for this already knows `-g`,
//! `-t`, `-i`, `-A/-B/-C`, `-l`, `-c`. A bespoke grammar — this command had one
//! — is a second dialect to learn for no gain. Where `grep(1)` and `rg`
//! disagree, `rg` wins; where `rg` and ambit's own conventions disagree, `rg`
//! still wins.
//!
//! We deviate in four places, each on purpose: output is always sorted by
//! `(path, line, column)`, because determinism is worth more to an agent than
//! the microseconds; `--head-limit` and `--max-columns` carry non-zero defaults,
//! because this output lands in a context window rather than a terminal; and
//! `--column` is on, because it is what disambiguates two matches on one line.
//!
//! ## What the symbol column buys
//!
//! A grep hit is a coordinate. `src/app.rs:1847` tells an agent where to look
//! but not what it is looking at, and nothing about whether it has been there
//! before. Every symbol carries a `byte_range`, so the innermost symbol
//! containing a match is a containment search over that file's symbols
//! ([`FileSymbols::enclosing`]), and the coverage journal turns the resulting id
//! into a read depth. `src/app.rs:1847` becomes
//! `src/app.rs::App/process_agent_event`, already read in full — an id `show`
//! accepts and a reason not to spend a `Read` on it.
//!
//! Files no parser handles are still searched. A hit in `Cargo.toml` is a real
//! hit; it simply has no symbol, and says so rather than being hidden.
//!
//! ## Prefilter first, parse second
//!
//! Every other command scans the project up front: walk, then parse every file,
//! then answer. For a search that is backwards. Most files do not match, and a
//! file that does not match need never be parsed — so the pipeline is walk,
//! read, reject on the raw bytes, and only then parse the survivors for
//! attribution. Against this repo a typical query parses 7 files instead of 40.
//! The regex crate's own literal prefilters do the rejecting, which is the same
//! machinery ripgrep relies on.

use std::io::Write;
use std::path::{Path, PathBuf};

use color_eyre::eyre::{Result, WrapErr};
use regex::bytes::{Regex, RegexBuilder};

use crate::parser::ParserRegistry;
use crate::restore::CoverageIndex;
use crate::symbols::FileSymbols;
use crate::tracking::ReadDepth;

/// Matches reported before truncating, across all files.
///
/// ripgrep has no such cap, and for a terminal it should not. This output goes
/// into an agent's context window, where an unbounded grep is a hazard rather
/// than a scroll. `0` lifts it.
pub const DEFAULT_HEAD_LIMIT: usize = 200;

/// Columns of a matching line shown before truncating.
///
/// Also not an rg default. A minified bundle or an embedded blob is one line of
/// tens of thousands of bytes, and printing it teaches the reader nothing.
pub const DEFAULT_MAX_COLUMNS: usize = 300;

/// How much of a file to sniff for NUL before calling it binary.
const BINARY_SNIFF_BYTES: usize = 8 * 1024;

/// What to print for a symbol that has never been read. An empty column would
/// read as "unread" when the truth may be "unknown"; see [`Options::no_symbol`]
/// and the journal-absent case in [`render_symbol`].
const UNREAD: &str = "—";

/// What to print where a symbol should be but none exists — a match at file
/// scope, or in a file no parser handles.
const NO_SYMBOL: &str = "-";

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// How results are reported, mirroring ripgrep's mutually exclusive modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputMode {
    /// Matching lines. The default, and the only mode that shows source.
    #[default]
    Content,
    /// `-l`: one path per file with a match.
    FilesWithMatches,
    /// `-c`: matching lines per file.
    Count,
    /// `--count-matches`: total matches per file.
    CountMatches,
    /// `-q`: nothing at all; the exit code is the answer.
    Quiet,
}

impl OutputMode {
    /// Whether this mode puts source text in front of the caller.
    ///
    /// Load-bearing beyond formatting: only a mode that shows source can
    /// justify recording a read, so this is what the journal keys off.
    pub fn shows_source(&self) -> bool {
        matches!(self, OutputMode::Content)
    }
}

/// When to colorize.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorChoice {
    /// Color when stdout is a terminal.
    #[default]
    Auto,
    Always,
    Never,
}

/// Everything `find` needs, mapped from the CLI in `main`.
///
/// A bag rather than a dozen parameters: the flag set is ripgrep's, so it is
/// large by definition, and threading it as arguments would put this function
/// well past any reasonable arity.
#[derive(Debug, Default)]
pub struct Options {
    /// Patterns, combined as an alternation. The positional `PATTERN` plus
    /// every `-e`.
    pub patterns: Vec<String>,
    /// `-F`: treat patterns as literal text.
    pub fixed_strings: bool,
    /// `-i`
    pub ignore_case: bool,
    /// `-w`
    pub word_regexp: bool,
    /// `-x`. Takes precedence over `-w`, as in rg.
    pub line_regexp: bool,
    /// `-U`: patterns may match across line boundaries.
    pub multiline: bool,
    /// `-v`
    pub invert_match: bool,

    pub mode: OutputMode,
    /// `--json`: ripgrep's JSON Lines event stream, plus `symbol` and
    /// `coverage`.
    pub json: bool,
    /// `--heading` / `--no-heading`. `None` follows the terminal.
    pub heading: Option<bool>,
    /// `-n` / `-N`
    pub line_number: bool,
    /// `--column` / `--no-column`
    pub column: bool,
    /// `-o`
    pub only_matching: bool,
    /// `-B`
    pub before_context: usize,
    /// `-A`
    pub after_context: usize,
    /// `-M`, `0` for unlimited.
    pub max_columns: usize,
    /// `-m`, per file.
    pub max_count: Option<usize>,
    /// `--head-limit`, `0` for unlimited.
    pub head_limit: usize,
    /// Drop the attribution field entirely, for byte-identical rg output.
    pub no_symbol: bool,
    pub color: ColorChoice,
}

impl Options {
    /// Options for `patterns`, with every default in place.
    ///
    /// The defaults are the CLI's defaults, so tests exercise the same
    /// configuration users get rather than an all-false strawman.
    pub fn new(patterns: Vec<String>) -> Self {
        Self {
            patterns,
            line_number: true,
            column: true,
            max_columns: DEFAULT_MAX_COLUMNS,
            head_limit: DEFAULT_HEAD_LIMIT,
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Matcher
// ---------------------------------------------------------------------------

/// A compiled pattern set.
///
/// `regex::bytes` rather than `regex`: a source tree contains files that are not
/// valid UTF-8, and refusing to search them — or lossily rewriting them before
/// the match, which moves every byte offset after the first bad byte — are both
/// worse than matching the bytes as they are.
#[derive(Debug)]
pub struct Matcher {
    re: Regex,
    invert: bool,
    multiline: bool,
}

impl Matcher {
    /// Compile `opts.patterns` into one alternation.
    ///
    /// Alternation rather than a `RegexSet` because the match *positions* are
    /// the whole point; `RegexSet` reports which patterns matched but not where.
    pub fn new(opts: &Options) -> Result<Self> {
        if opts.patterns.is_empty() {
            return Err(color_eyre::eyre::eyre!("no pattern given"));
        }
        let alternation = opts
            .patterns
            .iter()
            .map(|p| {
                let atom = if opts.fixed_strings {
                    regex::escape(p)
                } else {
                    p.clone()
                };
                // `-x` subsumes `-w`: a pattern anchored to the whole line is
                // already at word boundaries, and applying both nests the
                // anchors wrongly.
                if opts.line_regexp {
                    format!("^(?:{atom})$")
                } else if opts.word_regexp {
                    format!(r"\b(?:{atom})\b")
                } else {
                    format!("(?:{atom})")
                }
            })
            .collect::<Vec<_>>()
            .join("|");

        let re = RegexBuilder::new(&alternation)
            .case_insensitive(opts.ignore_case)
            // Only meaningful in multiline mode: the per-line path matches
            // against one line at a time, where `^`/`$` already mean what the
            // caller expects.
            .multi_line(opts.multiline)
            .build()
            .wrap_err_with(|| format!("invalid pattern: {alternation}"))?;

        Ok(Self {
            re,
            invert: opts.invert_match,
            multiline: opts.multiline,
        })
    }

    /// Whether the file is worth opening further.
    ///
    /// Under `-v` every file qualifies — a file with no match is nothing but
    /// inverted matches — so the prefilter is skipped rather than inverted.
    fn worth_searching(&self, buf: &[u8]) -> bool {
        self.invert || self.re.is_match(buf)
    }
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

/// The symbol a match landed in.
///
/// Owned rather than borrowed from a `FileSymbols`: files are parsed one at a
/// time and dropped, so there is no tree outliving the search to borrow from.
/// At `--head-limit` scale the allocations are noise.
#[derive(Debug, Clone)]
pub struct SymbolHit {
    /// `<path>::<name-path>`, accepted as-is by `show`.
    pub id: String,
    /// The nesting path alone, which is what the text output shows.
    pub name_path: String,
    pub label: &'static str,
    /// 1-based inclusive line range of the whole symbol.
    pub lines: [u32; 2],
    /// Depth this symbol was read at, when a journal says so.
    pub depth: Option<ReadDepth>,
    /// Current content hash, for the journal to record against.
    pub content_hash: [u8; 32],
    pub estimated_tokens: u32,
}

/// One matching line.
#[derive(Debug, Clone)]
pub struct Hit {
    /// 1-based.
    pub line: u32,
    /// 1-based **byte** column of the match start, as rg reports it.
    pub column: u32,
    /// Absolute byte offset of the match start, which is what attribution
    /// searches.
    pub byte: u32,
    /// The whole line, or every line the match spans under `-U`.
    pub text: Vec<u8>,
    /// The match, relative to `text`.
    pub span: (usize, usize),
    pub symbol: Option<SymbolHit>,
}

/// One file's matches.
#[derive(Debug)]
pub struct FileHits {
    /// Project-relative.
    pub path: PathBuf,
    pub hits: Vec<Hit>,
    /// Matches before `-m` truncated them.
    pub total: usize,
    /// `-A`/`-B`/`-C` lines, by 1-based line number, never overlapping `hits`.
    /// Captured during the search because that is the only point at which the
    /// file's bytes are in hand.
    pub context: Vec<(u32, Vec<u8>)>,
}

// ---------------------------------------------------------------------------
// Searching
// ---------------------------------------------------------------------------

/// Whether `buf` looks like something a reader would want printed.
///
/// rg's heuristic: a NUL byte near the start. Cheap, and wrong only for text
/// files that open with a NUL, which are not text files.
fn is_binary(buf: &[u8]) -> bool {
    buf.iter().take(BINARY_SNIFF_BYTES).any(|&b| b == 0)
}

/// Every match in one already-read buffer, before attribution.
fn hits_in(matcher: &Matcher, buf: &[u8], max_count: Option<usize>) -> (Vec<Hit>, usize) {
    let mut hits = Vec::new();
    let mut total = 0usize;

    if matcher.multiline {
        // One pass over the whole buffer: the match may cross line boundaries,
        // so lines are derived from the match rather than the other way around.
        for m in matcher.re.find_iter(buf) {
            total += 1;
            if max_count.is_some_and(|c| hits.len() >= c) {
                continue;
            }
            let line_start = buf[..m.start()]
                .iter()
                .rposition(|&b| b == b'\n')
                .map_or(0, |i| i + 1);
            let line_end = buf[m.end()..]
                .iter()
                .position(|&b| b == b'\n')
                .map_or(buf.len(), |i| m.end() + i);
            hits.push(Hit {
                line: buf[..m.start()].iter().filter(|&&b| b == b'\n').count() as u32 + 1,
                column: (m.start() - line_start) as u32 + 1,
                byte: m.start() as u32,
                text: buf[line_start..line_end].to_vec(),
                span: (m.start() - line_start, m.end() - line_start),
                symbol: None,
            });
        }
        return (hits, total);
    }

    let mut offset = 0usize;
    for (i, raw) in buf.split_inclusive(|&b| b == b'\n').enumerate() {
        let line_no = i as u32 + 1;
        let line = strip_newline(raw);

        if matcher.invert {
            if !matcher.re.is_match(line) {
                total += 1;
                if !max_count.is_some_and(|c| hits.len() >= c) {
                    hits.push(Hit {
                        line: line_no,
                        column: 1,
                        byte: offset as u32,
                        text: line.to_vec(),
                        span: (0, 0),
                        symbol: None,
                    });
                }
            }
        } else {
            for m in matcher.re.find_iter(line) {
                total += 1;
                if max_count.is_some_and(|c| hits.len() >= c) {
                    continue;
                }
                hits.push(Hit {
                    line: line_no,
                    column: m.start() as u32 + 1,
                    byte: (offset + m.start()) as u32,
                    text: line.to_vec(),
                    span: (m.start(), m.end()),
                    symbol: None,
                });
            }
        }
        offset += raw.len();
    }

    (hits, total)
}

/// A line without its terminator, CRLF included.
fn strip_newline(raw: &[u8]) -> &[u8] {
    let mut line = raw;
    if line.last() == Some(&b'\n') {
        line = &line[..line.len() - 1];
        if line.last() == Some(&b'\r') {
            line = &line[..line.len() - 1];
        }
    }
    line
}

/// Search one file, parsing it only if it matched and only if attribution is
/// wanted.
///
/// Returns `None` for a file that is unreadable, binary, or has no match —
/// three outcomes a caller treats identically, and none of which is an error:
/// one unreadable file should narrow the answer, not fail the command.
pub fn search_file(
    matcher: &Matcher,
    registry: &ParserRegistry,
    abs: &Path,
    rel: &Path,
    opts: &Options,
    coverage: Option<&CoverageIndex>,
) -> Option<FileHits> {
    let buf = std::fs::read(abs).ok()?;
    if is_binary(&buf) || !matcher.worth_searching(&buf) {
        return None;
    }

    let (mut hits, total) = hits_in(matcher, &buf, opts.max_count);
    if hits.is_empty() {
        return None;
    }

    // Attribution is the only reason to parse, so the modes that show no source
    // skip it outright — as does a file whose bytes are not valid UTF-8, since
    // the parsers take `&str` and a lossy rewrite would move every offset after
    // the first bad byte.
    if opts.mode.shows_source() && !opts.no_symbol {
        if let (Some(parser), Ok(source)) = (registry.parser_for(abs), std::str::from_utf8(&buf)) {
            if let Ok(symbols) = parser.parse_file(rel, source) {
                attribute(&mut hits, &symbols, coverage);
            }
        }
    }

    let context = context_lines(&buf, &hits, opts);

    Some(FileHits {
        path: rel.to_path_buf(),
        hits,
        total,
        context,
    })
}

/// The `-A`/`-B` lines around `hits`, excluding the matching lines themselves.
///
/// One pass over the buffer rather than a ring buffer during matching: context
/// is off by default, and keeping the match loop free of it is worth more than
/// the second pass costs on the few files that matched.
fn context_lines(buf: &[u8], hits: &[Hit], opts: &Options) -> Vec<(u32, Vec<u8>)> {
    if opts.before_context == 0 && opts.after_context == 0 || hits.is_empty() {
        return Vec::new();
    }
    let matched: std::collections::HashSet<u32> = hits.iter().map(|h| h.line).collect();

    let mut out = Vec::new();
    for (i, raw) in buf.split_inclusive(|&b| b == b'\n').enumerate() {
        let line_no = i as u32 + 1;
        if matched.contains(&line_no) {
            continue;
        }
        let wanted = hits.iter().any(|h| {
            let lo = h.line.saturating_sub(opts.before_context as u32);
            let hi = h.line + opts.after_context as u32;
            line_no >= lo && line_no <= hi
        });
        if wanted {
            out.push((line_no, strip_newline(raw).to_vec()));
        }
    }
    out
}

/// Fill in each hit's enclosing symbol and read depth.
fn attribute(hits: &mut [Hit], symbols: &FileSymbols, coverage: Option<&CoverageIndex>) {
    for hit in hits {
        let Some(node) = symbols.enclosing(hit.byte) else {
            continue;
        };
        hit.symbol = Some(SymbolHit {
            id: node.id.clone(),
            name_path: node.name_path().to_string(),
            label: node.label,
            lines: [node.line_range.start, node.line_range.end],
            depth: coverage.and_then(|c| c.depth_of(&node.id)),
            content_hash: node.content_hash,
            estimated_tokens: node.estimated_tokens,
        });
    }
}

/// Search every file the walk yields.
///
/// Files are searched in parallel and the results sorted afterwards: threads
/// finish out of order, and ripgrep's own nondeterminism under `-j` is a thing
/// callers work around rather than want.
pub fn search(
    matcher: &Matcher,
    registry: &ParserRegistry,
    targets: &[(PathBuf, PathBuf)],
    opts: &Options,
    coverage: Option<&CoverageIndex>,
) -> Vec<FileHits> {
    if targets.is_empty() {
        return Vec::new();
    }

    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(targets.len());
    let chunk = targets.len().div_ceil(threads);

    let mut out: Vec<FileHits> = std::thread::scope(|scope| {
        let handles: Vec<_> = targets
            .chunks(chunk)
            .map(|batch| {
                scope.spawn(move || {
                    batch
                        .iter()
                        .filter_map(|(abs, rel)| {
                            search_file(matcher, registry, abs, rel, opts, coverage)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .filter_map(|h| h.join().ok())
            .flatten()
            .collect()
    });

    out.sort_by(|a, b| a.path.cmp(&b.path));
    out
}

/// Drop everything past `head_limit` matches, reporting how many were dropped.
///
/// Applied after sorting so the kept prefix is stable across runs, and counted
/// in matches rather than files so the cap means what it says.
pub fn apply_head_limit(files: &mut Vec<FileHits>, head_limit: usize) -> usize {
    if head_limit == 0 {
        return 0;
    }
    let mut kept = 0usize;
    let mut withheld = 0usize;
    for file in files.iter_mut() {
        let room = head_limit.saturating_sub(kept);
        if file.hits.len() > room {
            withheld += file.hits.len() - room;
            file.hits.truncate(room);
        }
        kept += file.hits.len();
    }
    files.retain(|f| !f.hits.is_empty());
    withheld
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

/// Whether to group by file with a heading, as rg does on a terminal.
fn use_heading(opts: &Options) -> bool {
    opts.heading
        .unwrap_or_else(|| std::io::IsTerminal::is_terminal(&std::io::stdout()))
}

fn use_color(opts: &Options) -> bool {
    match opts.color {
        ColorChoice::Always => true,
        ColorChoice::Never => false,
        ColorChoice::Auto => std::io::IsTerminal::is_terminal(&std::io::stdout()),
    }
}

/// The attribution field.
///
/// Three cases, and the difference between the last two is load-bearing: with
/// no journal loaded, an unread symbol and an unknown one are indistinguishable,
/// so the depth is omitted entirely rather than shown as unread. A caller told a
/// symbol is unread will go read it; one told nothing should not.
fn render_symbol(symbol: Option<&SymbolHit>, have_journal: bool) -> String {
    let Some(sym) = symbol else {
        return format!("[{NO_SYMBOL}]");
    };
    if !have_journal {
        return format!("[{}]", sym.name_path);
    }
    match sym.depth {
        Some(d) => format!("[{d} {}]", sym.name_path),
        None => format!("[{UNREAD} {}]", sym.name_path),
    }
}

/// Clip to `max` bytes on a character boundary, reporting whether it cut.
fn clip(text: &str, max: usize) -> (String, bool) {
    if max == 0 || text.len() <= max {
        return (text.to_string(), false);
    }
    let mut end = max;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    (format!("{}…", &text[..end]), true)
}

/// The text of one hit: the whole line, or just the match under `-o`.
fn body(hit: &Hit, opts: &Options, color: bool) -> (String, bool) {
    let bytes = if opts.only_matching {
        &hit.text[hit.span.0.min(hit.text.len())..hit.span.1.min(hit.text.len())]
    } else {
        &hit.text[..]
    };
    let raw = String::from_utf8_lossy(bytes).into_owned();
    let (shown, truncated) = clip(&raw, opts.max_columns);

    let (start, end) = hit.span;
    let highlightable = color
        && !opts.only_matching
        && end > start
        && end <= shown.len()
        && shown.is_char_boundary(start)
        && shown.is_char_boundary(end);
    if highlightable {
        return (
            format!(
                "{}\x1b[1;31m{}\x1b[0m{}",
                &shown[..start],
                &shown[start..end],
                &shown[end..]
            ),
            truncated,
        );
    }
    (shown, truncated)
}

/// `path:line:col:` — or the `-` separated form context lines use.
fn locate(path: &Path, line: u32, column: Option<u32>, opts: &Options, sep: char) -> String {
    let mut out = format!("{}{sep}", path.display());
    if opts.line_number {
        out.push_str(&format!("{line}{sep}"));
        if let Some(c) = column {
            out.push_str(&format!("{c}{sep}"));
        }
    }
    out
}

/// One printable line: a match, or a context line around one.
///
/// The two arrive as separate sequences and print interleaved by line number,
/// which is the whole reason they need a common type.
enum Row<'a> {
    Match(&'a Hit),
    Context(u32, &'a [u8]),
}

impl Row<'_> {
    /// Sort key. Context lines sort before any match on the same line, which
    /// cannot happen — a line is one or the other — but keeps the order total.
    fn order(&self) -> (u32, u32) {
        match self {
            Row::Match(hit) => (hit.line, hit.column),
            Row::Context(line, _) => (*line, 0),
        }
    }
}

/// Print matching lines, either flat or grouped under file headings.
fn print_content(
    w: &mut impl Write,
    files: &[FileHits],
    opts: &Options,
    have_journal: bool,
) -> std::io::Result<()> {
    let heading = use_heading(opts);
    let color = use_color(opts);

    for (i, file) in files.iter().enumerate() {
        if heading {
            if i > 0 {
                writeln!(w)?;
            }
            writeln!(w, "{}", file.path.display())?;
        }

        // Context lines interleave by line number, so the two sequences are
        // merged rather than printed in turn.
        let mut rows: Vec<Row> = file
            .hits
            .iter()
            .map(Row::Match)
            .chain(
                file.context
                    .iter()
                    .map(|(line, text)| Row::Context(*line, text.as_slice())),
            )
            .collect();
        rows.sort_by_key(Row::order);

        for row in rows {
            match row {
                Row::Match(hit) => {
                    let (text, _) = body(hit, opts, color);
                    let symbol = if opts.no_symbol {
                        String::new()
                    } else {
                        format!("{} ", render_symbol(hit.symbol.as_ref(), have_journal))
                    };
                    if heading {
                        let col = if opts.column {
                            format!(":{}", hit.column)
                        } else {
                            String::new()
                        };
                        writeln!(w, "  {}{col}  {symbol}{text}", hit.line)?;
                    } else {
                        let prefix = locate(
                            &file.path,
                            hit.line,
                            opts.column.then_some(hit.column),
                            opts,
                            ':',
                        );
                        writeln!(w, "{prefix}{symbol}{text}")?;
                    }
                }
                Row::Context(line, text) => {
                    let (shown, _) = clip(&String::from_utf8_lossy(text), opts.max_columns);
                    if heading {
                        writeln!(w, "  {line}-  {shown}")?;
                    } else {
                        writeln!(w, "{}{shown}", locate(&file.path, line, None, opts, '-'))?;
                    }
                }
            }
        }
    }
    Ok(())
}

/// ripgrep's JSON Lines stream, plus `symbol` on each match and `coverage` on
/// the summary.
///
/// The event shape is rg's so existing consumers keep working; `stats` carries
/// the subset we can answer honestly rather than inventing timings.
fn print_json(
    w: &mut impl Write,
    files: &[FileHits],
    opts: &Options,
    coverage: Option<&CoverageIndex>,
    withheld: usize,
) -> std::io::Result<()> {
    use serde_json::json;

    let mut matched_lines = 0usize;
    for file in files {
        let path = file.path.display().to_string();
        writeln!(w, "{}", json!({"type": "begin", "data": {"path": {"text": path}}}))?;

        if opts.mode.shows_source() {
            for hit in &file.hits {
                let (mut text, truncated) = body(hit, opts, false);
                // rg's `lines.text` carries the line terminator. A truncated
                // line has had its tail removed, so it gets none — the marker
                // already says the line did not end there.
                if !truncated && !opts.only_matching {
                    text.push('\n');
                }
                let matched = String::from_utf8_lossy(
                    &hit.text[hit.span.0.min(hit.text.len())..hit.span.1.min(hit.text.len())],
                )
                .into_owned();
                let symbol = hit.symbol.as_ref().map(|s| {
                    json!({
                        "id": s.id,
                        "label": s.label,
                        "lines": s.lines,
                        "read_depth": s.depth.map(|d| d.to_string()),
                    })
                });
                writeln!(
                    w,
                    "{}",
                    json!({
                        "type": "match",
                        "data": {
                            "path": {"text": path},
                            "lines": {"text": text},
                            "line_number": hit.line,
                            "absolute_offset": hit.byte,
                            "submatches": [{
                                "match": {"text": matched},
                                "start": hit.span.0,
                                "end": hit.span.1,
                            }],
                            "symbol": symbol,
                            "truncated": truncated,
                        }
                    })
                )?;
            }
        }

        matched_lines += file.hits.len();
        writeln!(
            w,
            "{}",
            json!({
                "type": "end",
                "data": {
                    "path": {"text": path},
                    "binary_offset": serde_json::Value::Null,
                    "stats": {"matched_lines": file.hits.len(), "matches": file.total},
                }
            })
        )?;
    }

    let coverage = coverage.map(|c| json!({"session_id": c.session_id(), "symbols_read": c.len()}));
    writeln!(
        w,
        "{}",
        json!({
            "type": "summary",
            "data": {
                "stats": {"matched_lines": matched_lines, "searched_files": files.len()},
                "coverage": coverage,
                "withheld": withheld,
            }
        })
    )
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// What a search produced, beyond what it printed.
#[derive(Debug, Default)]
pub struct Outcome {
    /// Whether anything matched, which is the exit code.
    pub matched: bool,
    /// The symbols whose matches were actually **shown**, deduplicated.
    ///
    /// This is what may be journaled as read, and why it is "shown" rather than
    /// "found": `-l`, `-c` and `-q` put no source in front of the caller, and
    /// neither do matches cut off by `--head-limit`. The journal has to record
    /// what the agent saw, not what the process computed.
    pub shown: Vec<SymbolHit>,
}

/// The symbols whose matches are about to be printed, deduplicated.
///
/// Called after `--head-limit` has already truncated, so a symbol cut from the
/// output is absent here too. Modes that print no source contribute nothing at
/// all: the caller cannot have read what it was never shown.
fn shown_symbols(files: &[FileHits], mode: OutputMode) -> Vec<SymbolHit> {
    if !mode.shows_source() {
        return Vec::new();
    }
    let mut seen = std::collections::HashSet::new();
    let mut shown = Vec::new();
    for hit in files.iter().flat_map(|f| f.hits.iter()) {
        if let Some(sym) = &hit.symbol {
            if seen.insert(sym.id.clone()) {
                shown.push(sym.clone());
            }
        }
    }
    shown
}

/// Search `targets`, print the results, and report what was shown.
///
/// Walking is the caller's job: `main` builds the [`crate::parser::WalkOptions`]
/// from the CLI, which keeps this function testable against a synthetic file
/// list and keeps glob/type handling in one place.
pub fn run(
    registry: &ParserRegistry,
    targets: &[(PathBuf, PathBuf)],
    opts: &Options,
    coverage: Option<&CoverageIndex>,
) -> Result<Outcome> {
    let matcher = Matcher::new(opts)?;
    let mut files = search(&matcher, registry, targets, opts, coverage);
    let withheld = apply_head_limit(&mut files, opts.head_limit);

    let matched = !files.is_empty();
    let shown = shown_symbols(&files, opts.mode);

    if opts.mode == OutputMode::Quiet {
        return Ok(Outcome { matched, shown });
    }

    let stdout = std::io::stdout();
    let mut w = std::io::BufWriter::new(stdout.lock());

    if opts.json {
        print_json(&mut w, &files, opts, coverage, withheld)?;
        w.flush()?;
        return Ok(Outcome { matched, shown });
    }

    match opts.mode {
        OutputMode::Content => print_content(&mut w, &files, opts, coverage.is_some())?,
        OutputMode::FilesWithMatches => {
            for file in &files {
                writeln!(w, "{}", file.path.display())?;
            }
        }
        OutputMode::Count => {
            for file in &files {
                writeln!(w, "{}:{}", file.path.display(), file.hits.len())?;
            }
        }
        OutputMode::CountMatches => {
            for file in &files {
                writeln!(w, "{}:{}", file.path.display(), file.total)?;
            }
        }
        OutputMode::Quiet => unreachable!("returned above"),
    }
    w.flush()?;

    // stderr, so stdout stays exactly what a grep consumer expects to parse.
    if withheld > 0 {
        eprintln!("… {withheld} more matches withheld (--head-limit 0 for all)");
    }

    Ok(Outcome { matched, shown })
}

// ---------------------------------------------------------------------------
// Journaling
// ---------------------------------------------------------------------------

/// Record the symbols this search showed as read, in the session's journal.
///
/// ## Why a search records reads at all
///
/// Because it puts source in front of an agent. A `Grep` tool call is credited
/// at `Overview` against a path; this can do better, because it knows exactly
/// which symbols it printed and what their contents hash to right now.
///
/// ## Why `FullBody`
///
/// `find` searched the symbol's entire body, and the hash recorded alongside is
/// the body it searched. The agent saw the matching lines rather than all of
/// them, so this is generous — a deliberate product decision, made knowing the
/// exposure is over-crediting at restore time rather than unsound drift
/// detection.
///
/// ## What is *not* recorded
///
/// Anything the caller did not see: `shown` excludes `-q`, `-l` and `-c`, which
/// print no source, and excludes matches cut off by `--head-limit`. A journal
/// that records what the process computed rather than what the agent read is
/// worse than no journal, because it claims knowledge nobody has.
///
/// Attribution goes to the session id, which is the agent id Claude Code's own
/// records use for a session's main agent (`agentId` falls back to `sessionId`
/// in `ingest::claude`). The `--agent` flag deliberately does not steer this: it
/// is a *filter*, matched by prefix, and a prefix is not an id to write down.
pub fn journal_reads(
    project_root: &Path,
    session_id: &str,
    shown: &[SymbolHit],
    manifest: impl FnOnce() -> crate::journal::EnvironmentManifest,
) -> usize {
    if shown.is_empty() {
        return 0;
    }

    // The ledger starts empty rather than rehydrated: `Journal::open` seeds its
    // dedup map from the file, so a symbol already recorded at this hash and
    // depth is skipped without needing the whole history in memory. `record`
    // refreshes `content_hash_at_read` and clears `stale`, which is exactly the
    // "a symbol that matched is current again" rule, for free.
    let mut ledger = crate::tracking::ContextLedger::new();
    for sym in shown {
        ledger.record(
            sym.id.clone(),
            ReadDepth::FullBody,
            sym.content_hash,
            session_id.to_string(),
            sym.estimated_tokens as usize,
        );
    }

    let mut journal = crate::journal::Journal::open(
        project_root,
        session_id,
        // No interval: a CLI process syncs once and exits, where the TUI
        // spreads its writes over a long-lived run.
        std::time::Duration::ZERO,
        manifest,
    );
    journal.sync(&ledger)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::{file, sym_with_bytes, sym_with_children};
    use crate::tracking::ReadDepth;

    // ─── matching ────────────────────────────────────────────────────────────

    const SRC: &str = "fn alpha() {}\nfn beta() {}\nlet alphabet = 1;\n";

    fn hits(src: &str, opts: &Options) -> Vec<Hit> {
        let matcher = Matcher::new(opts).expect("pattern must compile");
        hits_in(&matcher, src.as_bytes(), opts.max_count).0
    }

    fn lines_of(hits: &[Hit]) -> Vec<u32> {
        hits.iter().map(|h| h.line).collect()
    }

    /// The headline change: the pattern searches file *content*, not symbol
    /// names, so a use of a name matches as readily as its definition.
    #[test]
    fn a_bare_pattern_is_a_regex_over_content() {
        let found = hits(SRC, &Options::new(vec!["fn \\w+".into()]));
        assert_eq!(lines_of(&found), vec![1, 2]);
        assert_eq!(found[0].column, 1, "columns are 1-based, as in ripgrep");
    }

    #[test]
    fn fixed_strings_disables_metacharacters() {
        let mut opts = Options::new(vec!["alpha()".into()]);
        assert_eq!(
            lines_of(&hits(SRC, &opts)),
            vec![1, 3],
            "as a regex `()` is an empty group, so this is really just `alpha` \
             and drags in `alphabet`"
        );

        opts.fixed_strings = true;
        assert_eq!(
            lines_of(&hits(SRC, &opts)),
            vec![1],
            "taken literally, `alpha()` appears only where it is called"
        );
    }

    /// `-w` must hold the boundary that `-i` would otherwise widen: `alpha`
    /// matches `alphabet` without it, and must not with it.
    #[test]
    fn ignore_case_and_word_boundaries_compose() {
        let mut opts = Options::new(vec!["ALPHA".into()]);
        opts.ignore_case = true;
        assert_eq!(lines_of(&hits(SRC, &opts)), vec![1, 3]);

        opts.word_regexp = true;
        assert_eq!(
            lines_of(&hits(SRC, &opts)),
            vec![1],
            "`alphabet` is not the word `alpha`"
        );
    }

    #[test]
    fn line_regexp_anchors_the_whole_line() {
        let mut opts = Options::new(vec!["fn beta\\(\\) \\{\\}".into()]);
        opts.line_regexp = true;
        assert_eq!(lines_of(&hits(SRC, &opts)), vec![2]);

        let mut partial = Options::new(vec!["fn beta".into()]);
        partial.line_regexp = true;
        assert!(
            hits(SRC, &partial).is_empty(),
            "a partial line must not match under -x"
        );
    }

    #[test]
    fn invert_match_reports_non_matching_lines() {
        let mut opts = Options::new(vec!["fn ".into()]);
        opts.invert_match = true;
        assert_eq!(lines_of(&hits(SRC, &opts)), vec![3]);
    }

    /// Under `-U` a match owns several lines; the one reported is where it
    /// starts, and the text carries every line it touched.
    #[test]
    fn multiline_reports_the_line_of_the_match_start() {
        let mut opts = Options::new(vec!["alpha[\\s\\S]*beta".into()]);
        opts.multiline = true;
        let found = hits(SRC, &opts);
        assert_eq!(lines_of(&found), vec![1]);
        assert!(
            String::from_utf8_lossy(&found[0].text).contains("fn beta"),
            "the text spans to the end of the matched region"
        );
    }

    #[test]
    fn max_count_caps_per_file_and_head_limit_caps_globally() {
        let mut opts = Options::new(vec!["fn".into()]);
        opts.max_count = Some(1);
        let matcher = Matcher::new(&opts).unwrap();
        let (kept, total) = hits_in(&matcher, SRC.as_bytes(), opts.max_count);
        assert_eq!(kept.len(), 1);
        assert_eq!(total, 2, "the total still counts what -m withheld");

        let mut files = vec![
            FileHits {
                path: PathBuf::from("a.rs"),
                hits: hits(SRC, &Options::new(vec!["fn".into()])),
                total: 2,
                context: Vec::new(),
            },
            FileHits {
                path: PathBuf::from("b.rs"),
                hits: hits(SRC, &Options::new(vec!["fn".into()])),
                total: 2,
                context: Vec::new(),
            },
        ];
        assert_eq!(apply_head_limit(&mut files, 3), 1);
        assert_eq!(files.iter().map(|f| f.hits.len()).sum::<usize>(), 3);
    }

    #[test]
    fn max_columns_truncates_and_marks() {
        let long = format!("let x = \"{}\";\n", "a".repeat(500));
        let mut opts = Options::new(vec!["let x".into()]);
        opts.max_columns = 20;
        let found = hits(&long, &opts);
        let (text, truncated) = body(&found[0], &opts, false);
        assert!(truncated);
        assert!(text.ends_with('…'));
        assert!(text.chars().count() <= 21, "20 columns plus the marker");
    }

    /// A pattern that cannot compile is a usage error. Reporting it as "no
    /// matches" would tell the caller their search succeeded and found nothing.
    #[test]
    fn an_invalid_regex_is_an_error_not_an_empty_result() {
        let err = Matcher::new(&Options::new(vec!["fn (".into()])).unwrap_err();
        assert!(
            format!("{err}").contains("invalid pattern"),
            "the message must name the pattern as the problem"
        );
    }

    // ─── attribution ─────────────────────────────────────────────────────────

    fn hit_at(byte: u32) -> Hit {
        Hit {
            line: 1,
            column: 1,
            byte,
            text: b"x".to_vec(),
            span: (0, 1),
            symbol: None,
        }
    }

    fn thing() -> crate::symbols::FileSymbols {
        file(
            "a.rs",
            vec![sym_with_children(
                "a.rs::Thing",
                "Thing",
                vec![sym_with_bytes("a.rs::Thing/method", "method", 40, 60)],
            )],
        )
    }

    #[test]
    fn a_hit_is_attributed_to_the_innermost_symbol() {
        let mut found = vec![hit_at(50)];
        attribute(&mut found, &thing(), None);
        assert_eq!(found[0].symbol.as_ref().unwrap().id, "a.rs::Thing/method");
    }

    /// A match in a `use` line or a file-level comment belongs to no symbol,
    /// and must say so rather than be attributed to whatever is nearest.
    #[test]
    fn a_hit_between_symbols_has_no_symbol() {
        let mut found = vec![hit_at(500)];
        attribute(&mut found, &thing(), None);
        assert!(found[0].symbol.is_none());
    }

    /// Symbol ids are not unique — `struct Foo` and `impl Foo` in one file both
    /// yield `a.rs::Foo` — so hits are keyed by position, never by id. Two hits
    /// in two different symbols that share an id stay two hits.
    #[test]
    fn colliding_ids_do_not_merge_hits() {
        let symbols = file(
            "a.rs",
            vec![
                sym_with_bytes("a.rs::Foo", "Foo", 0, 10),
                sym_with_bytes("a.rs::Foo", "Foo", 20, 30),
            ],
        );
        let mut found = vec![hit_at(5), hit_at(25)];
        attribute(&mut found, &symbols, None);
        assert_eq!(found.len(), 2, "position, not id, is the key");
        assert_eq!(found[0].symbol.as_ref().unwrap().lines, [1, 10]);
        assert_eq!(found[1].symbol.as_ref().unwrap().lines, [1, 10]);
    }

    // ─── coverage column ─────────────────────────────────────────────────────

    fn symbol_hit(depth: Option<ReadDepth>) -> SymbolHit {
        SymbolHit {
            id: "a.rs::render".into(),
            name_path: "render".into(),
            label: "fn",
            lines: [1, 10],
            depth,
            content_hash: [0u8; 32],
            estimated_tokens: 30,
        }
    }

    #[test]
    fn a_read_symbol_reports_the_depth_it_was_read_at() {
        let sym = symbol_hit(Some(ReadDepth::FullBody));
        assert_eq!(render_symbol(Some(&sym), true), "[full render]");
    }

    #[test]
    fn an_unread_symbol_renders_an_em_dash() {
        let sym = symbol_hit(None);
        assert_eq!(render_symbol(Some(&sym), true), "[— render]");
    }

    /// Without a journal there is no coverage context at all, and that must not
    /// be rendered the same way as "read nothing" — an agent told a symbol is
    /// unread will go read it; one told nothing is known should not.
    #[test]
    fn no_journal_omits_the_depth_entirely() {
        let sym = symbol_hit(None);
        assert_eq!(render_symbol(Some(&sym), false), "[render]");
        assert_eq!(render_symbol(None, true), "[-]", "no symbol is its own case");
    }

    // ─── output ──────────────────────────────────────────────────────────────

    fn one_file(path: &str, opts: &Options) -> Vec<FileHits> {
        let mut found = hits(SRC, opts);
        attribute(&mut found, &thing(), None);
        vec![FileHits {
            path: PathBuf::from(path),
            hits: found,
            total: 2,
            context: Vec::new(),
        }]
    }

    fn rendered(files: &[FileHits], opts: &Options, journal: bool) -> String {
        let mut buf = Vec::new();
        print_content(&mut buf, files, opts, journal).unwrap();
        String::from_utf8(buf).unwrap()
    }

    /// Piped output is what agents parse, and the `file:line:col:` prefix is
    /// what every existing grep consumer expects. Ambit's field goes after it.
    #[test]
    fn piped_output_keeps_the_file_line_col_prefix() {
        let mut opts = Options::new(vec!["fn alpha".into()]);
        opts.heading = Some(false);
        let out = rendered(&one_file("src/a.rs", &opts), &opts, true);
        assert!(
            out.starts_with("src/a.rs:1:1:"),
            "expected a file:line:col prefix, got {out:?}"
        );
    }

    #[test]
    fn no_symbol_produces_rg_identical_output() {
        let mut opts = Options::new(vec!["fn alpha".into()]);
        opts.heading = Some(false);
        opts.no_symbol = true;
        let out = rendered(&one_file("src/a.rs", &opts), &opts, true);
        assert_eq!(out, "src/a.rs:1:1:fn alpha() {}\n");
    }

    #[test]
    fn json_match_events_carry_the_symbol() {
        let mut opts = Options::new(vec!["fn alpha".into()]);
        opts.json = true;
        let mut buf = Vec::new();
        print_json(&mut buf, &one_file("src/a.rs", &opts), &opts, None, 0).unwrap();
        let events: Vec<serde_json::Value> = String::from_utf8(buf)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).expect("every line is one JSON event"))
            .collect();

        assert_eq!(events[0]["type"], "begin");
        assert_eq!(events[1]["type"], "match");
        assert_eq!(events[1]["data"]["line_number"], 1);
        assert_eq!(events[1]["data"]["submatches"][0]["match"]["text"], "fn alpha");
        assert!(
            events[1]["data"]["symbol"].is_null() || events[1]["data"]["symbol"]["id"].is_string(),
            "symbol is an object or null, never absent"
        );
        assert_eq!(events.last().unwrap()["type"], "summary");
    }

    /// The envelope's `coverage` is what distinguishes "no journal" from
    /// "nothing read", exactly as the old `find` envelope did.
    #[test]
    fn json_summary_carries_coverage_and_withheld() {
        let opts = Options::new(vec!["fn alpha".into()]);
        let reads = [("a.rs::render".to_string(), ([0u8; 32], ReadDepth::FullBody))]
            .into_iter()
            .collect();
        let index = crate::restore::CoverageIndex::from_read_set(reads, "sess-1");

        let mut buf = Vec::new();
        print_json(&mut buf, &one_file("src/a.rs", &opts), &opts, Some(&index), 7).unwrap();
        let summary: serde_json::Value = serde_json::from_str(
            String::from_utf8(buf).unwrap().lines().last().unwrap(),
        )
        .unwrap();
        assert_eq!(summary["data"]["coverage"]["session_id"], "sess-1");
        assert_eq!(summary["data"]["withheld"], 7);

        let mut without = Vec::new();
        print_json(&mut without, &one_file("src/a.rs", &opts), &opts, None, 0).unwrap();
        let summary: serde_json::Value = serde_json::from_str(
            String::from_utf8(without).unwrap().lines().last().unwrap(),
        )
        .unwrap();
        assert!(
            summary["data"]["coverage"].is_null(),
            "no journal must be null, not an empty object"
        );
    }

    // ─── searching real files ────────────────────────────────────────────────

    fn write(dir: &tempfile::TempDir, name: &str, bytes: &[u8]) -> (PathBuf, PathBuf) {
        let abs = dir.path().join(name);
        std::fs::write(&abs, bytes).unwrap();
        (abs, PathBuf::from(name))
    }

    #[test]
    fn results_are_sorted_by_path_line_col() {
        let dir = tempfile::tempdir().unwrap();
        let targets = vec![
            write(&dir, "z.rs", b"fn zeta() {}\n"),
            write(&dir, "a.rs", b"fn alpha() {}\nfn alpha2() {}\n"),
        ];
        let opts = Options::new(vec!["fn".into()]);
        let matcher = Matcher::new(&opts).unwrap();
        let files = search(&matcher, &ParserRegistry::new(), &targets, &opts, None);

        let paths: Vec<String> = files.iter().map(|f| f.path.display().to_string()).collect();
        assert_eq!(paths, vec!["a.rs", "z.rs"], "threads finish out of order");
        assert_eq!(lines_of(&files[0].hits), vec![1, 2]);
    }

    /// grep searches every text file; only attribution is limited to what a
    /// parser understands. A hit in a config file is still a hit.
    #[test]
    fn a_hit_in_an_unparseable_file_has_no_symbol() {
        let dir = tempfile::tempdir().unwrap();
        let (abs, rel) = write(&dir, "Cargo.toml", b"name = \"ambits\"\n");
        let opts = Options::new(vec!["ambits".into()]);
        let matcher = Matcher::new(&opts).unwrap();

        let found =
            search_file(&matcher, &ParserRegistry::new(), &abs, &rel, &opts, None).unwrap();
        assert_eq!(found.hits.len(), 1);
        assert!(found.hits[0].symbol.is_none());
    }

    #[test]
    fn binary_files_are_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let (abs, rel) = write(&dir, "blob.bin", b"fn alpha\x00\x00 more");
        let opts = Options::new(vec!["fn alpha".into()]);
        let matcher = Matcher::new(&opts).unwrap();
        assert!(
            search_file(&matcher, &ParserRegistry::new(), &abs, &rel, &opts, None).is_none(),
            "a NUL byte near the start means it is not text"
        );
    }

    /// Context lines are a property of the file, not of a match, so they carry
    /// no attribution — the symbol around a context line may not be the symbol
    /// the match landed in.
    #[test]
    fn context_lines_carry_no_attribution() {
        let dir = tempfile::tempdir().unwrap();
        let (abs, rel) = write(&dir, "a.rs", b"fn alpha() {}\nfn beta() {}\nfn gamma() {}\n");
        let mut opts = Options::new(vec!["beta".into()]);
        opts.before_context = 1;
        opts.after_context = 1;
        let matcher = Matcher::new(&opts).unwrap();

        let found =
            search_file(&matcher, &ParserRegistry::new(), &abs, &rel, &opts, None).unwrap();
        assert_eq!(found.context.len(), 2, "one line either side");
        assert_eq!(found.context[0].0, 1);
        assert_eq!(found.context[1].0, 3);

        opts.heading = Some(false);
        let out = rendered(&[found], &opts, true);
        assert!(
            out.contains("a.rs-1-fn alpha"),
            "context uses grep's `-` separator and no symbol field, got {out:?}"
        );
    }

    /// The withheld count is reported to the caller so it can go to stderr;
    /// stdout stays exactly what a grep consumer expects to parse.
    #[test]
    fn the_withheld_count_never_reaches_stdout() {
        let mut opts = Options::new(vec!["fn".into()]);
        opts.heading = Some(false);
        let mut files = one_file("src/a.rs", &opts);
        let withheld = apply_head_limit(&mut files, 1);
        assert_eq!(withheld, 1);
        assert!(!rendered(&files, &opts, true).contains("withheld"));
    }
}

#[cfg(test)]
mod journaling_tests {
    use super::*;
    use crate::journal::{read_journal, EnvironmentManifest};
    use crate::tracking::ReadDepth;

    fn manifest() -> EnvironmentManifest {
        EnvironmentManifest {
            project_root: "/p".into(),
            tree_fingerprint: crate::journal::encode_hash(&[7u8; 32]),
            ambit_version: "test".into(),
            backend: "tree-sitter".into(),
            parsers: vec![],
            tool_config_version: Some(1),
            filter: None,
            os: "testos".into(),
            arch: "testarch".into(),
            host: "testhost".into(),
        }
    }

    fn symbol(id: &str, hash: u8) -> SymbolHit {
        SymbolHit {
            id: id.into(),
            name_path: id.rsplit("::").next().unwrap().into(),
            label: "fn",
            lines: [1, 10],
            depth: None,
            content_hash: [hash; 32],
            estimated_tokens: 30,
        }
    }

    fn hit_with(symbol: Option<SymbolHit>) -> Hit {
        Hit {
            line: 1,
            column: 1,
            byte: 0,
            text: b"x".to_vec(),
            span: (0, 1),
            symbol,
        }
    }

    fn file_with(path: &str, symbols: Vec<Option<SymbolHit>>) -> FileHits {
        FileHits {
            path: PathBuf::from(path),
            hits: symbols.into_iter().map(hit_with).collect(),
            total: 1,
            context: Vec::new(),
        }
    }

    /// A search shows source, so it records what it showed — at `FullBody`,
    /// because it searched the whole body and recorded the hash it searched.
    #[test]
    fn a_printed_hit_records_full_body_for_its_symbol() {
        let dir = tempfile::tempdir().unwrap();
        let written = journal_reads(
            dir.path(),
            "sess",
            &[symbol("a.rs::alpha", 1)],
            manifest,
        );
        assert_eq!(written, 1);

        let contents = read_journal(&crate::cache::journal_dir(dir.path()).join("sess.ndjson"));
        let (_, depth) = contents.reads.get("a.rs::alpha").expect("recorded");
        assert_eq!(*depth, ReadDepth::FullBody);
    }

    #[test]
    fn a_symbol_with_no_hit_is_not_recorded() {
        let dir = tempfile::tempdir().unwrap();
        journal_reads(dir.path(), "sess", &[symbol("a.rs::alpha", 1)], manifest);

        let contents = read_journal(&crate::cache::journal_dir(dir.path()).join("sess.ndjson"));
        assert!(
            !contents.reads.contains_key("a.rs::beta"),
            "a symbol the search never showed must not appear"
        );
    }

    /// The staleness rule, and the reason it needs no machinery of its own:
    /// `ContextLedger::record` refreshes `content_hash_at_read` on every read,
    /// so a symbol that matched again is current again.
    #[test]
    fn a_matched_symbol_that_drifted_gets_a_fresh_hash() {
        let dir = tempfile::tempdir().unwrap();
        let path = crate::cache::journal_dir(dir.path()).join("sess.ndjson");

        journal_reads(dir.path(), "sess", &[symbol("a.rs::alpha", 1)], manifest);
        let before = read_journal(&path).reads["a.rs::alpha"].0;

        // Same symbol, different content: the file changed and it matched again.
        journal_reads(dir.path(), "sess", &[symbol("a.rs::alpha", 2)], manifest);
        let after = read_journal(&path).reads["a.rs::alpha"].0;

        assert_ne!(before, after, "the newer hash supersedes the older");
        assert_eq!(after, [2u8; 32]);
    }

    /// …and the converse. A symbol that did not match keeps the hash it was
    /// read at, so a later restore still sees it as drifted.
    #[test]
    fn an_unmatched_drifted_symbol_stays_stale() {
        let dir = tempfile::tempdir().unwrap();
        let path = crate::cache::journal_dir(dir.path()).join("sess.ndjson");

        journal_reads(
            dir.path(),
            "sess",
            &[symbol("a.rs::alpha", 1), symbol("a.rs::beta", 1)],
            manifest,
        );
        // Only alpha matches the second search.
        journal_reads(dir.path(), "sess", &[symbol("a.rs::alpha", 2)], manifest);

        let reads = read_journal(&path).reads;
        assert_eq!(reads["a.rs::alpha"].0, [2u8; 32], "refreshed");
        assert_eq!(reads["a.rs::beta"].0, [1u8; 32], "untouched, so still stale");
    }

    /// The credit rule: modes that print no source put nothing in front of the
    /// caller, so they credit nothing.
    #[test]
    fn modes_that_print_no_source_record_nothing() {
        let files = vec![file_with("a.rs", vec![Some(symbol("a.rs::alpha", 1))])];
        assert_eq!(shown_symbols(&files, OutputMode::Content).len(), 1);
        for silent in [
            OutputMode::Quiet,
            OutputMode::FilesWithMatches,
            OutputMode::Count,
            OutputMode::CountMatches,
        ] {
            assert!(
                shown_symbols(&files, silent).is_empty(),
                "{silent:?} shows no source, so it can credit no read"
            );
        }
    }

    /// A match cut off by `--head-limit` was never shown, so it is never
    /// credited — the journal records what the agent saw.
    #[test]
    fn symbols_past_head_limit_record_nothing() {
        let mut files = vec![
            file_with("a.rs", vec![Some(symbol("a.rs::alpha", 1))]),
            file_with("b.rs", vec![Some(symbol("b.rs::beta", 1))]),
        ];
        assert_eq!(apply_head_limit(&mut files, 1), 1);

        let shown = shown_symbols(&files, OutputMode::Content);
        assert_eq!(shown.len(), 1);
        assert_eq!(shown[0].id, "a.rs::alpha", "only the printed one");
    }

    /// One symbol matched five times is one read, not five records.
    #[test]
    fn repeated_hits_in_one_symbol_are_one_read() {
        let files = vec![file_with(
            "a.rs",
            vec![
                Some(symbol("a.rs::alpha", 1)),
                Some(symbol("a.rs::alpha", 1)),
                None,
            ],
        )];
        assert_eq!(shown_symbols(&files, OutputMode::Content).len(), 1);
    }
}
