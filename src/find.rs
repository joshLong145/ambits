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
    /// `--files-without-match`: one path per file with *no* match — the
    /// complement of `-l`, not of `-v`. `-v` reports non-matching *lines*
    /// within files that were searched; this reports files that had zero
    /// matching lines at all.
    FilesWithoutMatch,
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
    /// `--vimgrep`: one row per match, even where several share a line —
    /// the exception to the default line-grouped output, same as `-o`. See
    /// `match_groups`, which both text and JSON printing call, so this
    /// affects `--json` too, splitting a shared-line event's `submatches`
    /// back into one event per match — exactly `-o`'s existing precedent
    /// there, not a special case invented for this flag.
    pub vimgrep: bool,
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
    /// Same pattern as `re`, always compiled `multi_line(true)`.
    ///
    /// `worth_searching` runs against the *whole file's* bytes, not one line
    /// at a time, so a `^`/`$` in `re` needs multi-line semantics there even
    /// when the caller never asked for `-U` — otherwise the anchor binds to
    /// the true start/end of the buffer, the prefilter rejects any file
    /// whose first line does not itself satisfy the pattern, and every later
    /// line's real match is silently dropped. Kept as a second `Regex`
    /// rather than always compiling `re` multi-line, because the per-line
    /// path must not gain multi-line semantics on its own — see `re`'s use
    /// in `hits_in`, one line at a time, where `^`/`$` already mean what the
    /// caller expects.
    prefilter_re: Regex,
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

        // Always multi-line, regardless of `opts.multiline` — see the field
        // doc on `prefilter_re`. Same alternation, so it stays in lockstep
        // with `re` (line-regexp/word-regexp/case-insensitive all already
        // baked into the string) without re-deriving any of that here.
        let prefilter_re = RegexBuilder::new(&alternation)
            .case_insensitive(opts.ignore_case)
            .multi_line(true)
            .build()
            .wrap_err_with(|| format!("invalid pattern: {alternation}"))?;

        Ok(Self {
            re,
            prefilter_re,
            invert: opts.invert_match,
            multiline: opts.multiline,
        })
    }

    /// Whether the file is worth opening further.
    ///
    /// Under `-v` every file qualifies — a file with no match is nothing but
    /// inverted matches — so the prefilter is skipped rather than inverted.
    /// Uses `prefilter_re`, not `re`: this runs against the whole file's
    /// bytes in one shot, so an anchor needs multi-line semantics here even
    /// when the caller is not searching with `-U`.
    fn worth_searching(&self, buf: &[u8]) -> bool {
        self.invert || self.prefilter_re.is_match(buf)
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

impl FileHits {
    /// Distinct lines with at least one match, which is what `-c` reports.
    ///
    /// Hits are per match, not per line, so this is not `hits.len()`: a line
    /// holding two matches is one matching line.
    pub fn matched_lines(&self) -> usize {
        self.hits
            .iter()
            .map(|h| h.line)
            .collect::<std::collections::HashSet<_>>()
            .len()
    }
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

/// One printable line: a group of same-line matches, or a context line.
///
/// `Match` holds every [`Hit`] on that line, not just one — ripgrep prints a
/// line once regardless of how many matches it holds, and `--json` folds them
/// into one event's `submatches`. Printing one row per `Hit` (the bug this
/// grouping fixes) double-counted a two-match line as two hits, in both plain
/// text and `--json`. `-o` is the deliberate exception: each match *is* its
/// own line there, so [`match_groups`] leaves it ungrouped.
///
/// The two variants arrive as separate sequences and print interleaved by
/// line number, which is the whole reason they need a common type.
enum Row<'a> {
    Match(&'a [Hit]),
    Context(u32, &'a [u8]),
}

impl Row<'_> {
    /// Sort key. Context lines sort before any match on the same line, which
    /// cannot happen — a line is one or the other — but keeps the order total.
    fn order(&self) -> (u32, u32) {
        match self {
            Row::Match(group) => (group[0].line, group[0].column),
            Row::Context(line, _) => (*line, 0),
        }
    }
}

/// Group `hits` into same-line runs for printing, except under `-o` or
/// `--vimgrep`, where each match prints as its own line and must stay
/// ungrouped — `-o` because there is no shared line text left to group by,
/// `--vimgrep` because quickfix format is one entry per match by definition.
///
/// `hits` is already sorted by `(line, column)` — the search walks each
/// file's bytes once, left to right, so two matches on one line are always
/// adjacent — which is what makes a simple [`slice::chunk_by`] correct here
/// without a re-sort.
fn match_groups<'a>(hits: &'a [Hit], opts: &Options) -> Vec<&'a [Hit]> {
    if opts.only_matching || opts.vimgrep {
        hits.iter().map(std::slice::from_ref).collect()
    } else {
        hits.chunk_by(|a, b| a.line == b.line).collect()
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
        let mut rows: Vec<Row> = match_groups(&file.hits, opts)
            .into_iter()
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
                Row::Match(group) => {
                    // Every hit in the group shares a line, so the text and
                    // (per rg's own convention) the reported column both come
                    // from the first — the leftmost match on that line.
                    let hit = &group[0];
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
            // One event per line, not per match — real rg folds same-line
            // matches into one event's `submatches`; see `match_groups`.
            for group in match_groups(&file.hits, opts) {
                let hit = &group[0];
                let (mut text, truncated) = body(hit, opts, false);
                // rg's `lines.text` carries the line terminator. A truncated
                // line has had its tail removed, so it gets none — the marker
                // already says the line did not end there.
                if !truncated && !opts.only_matching {
                    text.push('\n');
                }
                let submatches: Vec<_> = group
                    .iter()
                    .map(|h| {
                        let matched = String::from_utf8_lossy(
                            &h.text[h.span.0.min(h.text.len())..h.span.1.min(h.text.len())],
                        )
                        .into_owned();
                        json!({
                            "match": {"text": matched},
                            "start": h.span.0,
                            "end": h.span.1,
                        })
                    })
                    .collect();
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
                            "submatches": submatches,
                            "symbol": symbol,
                            "truncated": truncated,
                        }
                    })
                )?;
            }
        }

        matched_lines += file.matched_lines();
        writeln!(
            w,
            "{}",
            json!({
                "type": "end",
                "data": {
                    "path": {"text": path},
                    "binary_offset": serde_json::Value::Null,
                    "stats": {"matched_lines": file.matched_lines(), "matches": file.hits.len()},
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
    let matched = !files.is_empty();

    // `--head-limit` caps what is *printed*, so it applies only where matches
    // are printed. Letting it truncate `-l` or `-c` would silently drop files
    // from a listing and cap a count at the limit — answers that look complete
    // and are not.
    let withheld = if opts.mode.shows_source() {
        apply_head_limit(&mut files, opts.head_limit)
    } else {
        0
    };

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
        // The complement of `-l`: `files` only ever holds files with at
        // least one hit (search_file returns None for the rest, filtered out
        // in `search`), so this is targets minus that set, not a second walk.
        OutputMode::FilesWithoutMatch => {
            let matched: std::collections::HashSet<&Path> =
                files.iter().map(|f| f.path.as_path()).collect();
            for (_, rel) in targets {
                if !matched.contains(rel.as_path()) {
                    writeln!(w, "{}", rel.display())?;
                }
            }
        }
        // grep and rg both count matching *lines* for `-c`; only
        // `--count-matches` counts the matches themselves. A line with two
        // matches is one line and two matches.
        OutputMode::Count => {
            for file in &files {
                writeln!(w, "{}:{}", file.path.display(), file.matched_lines())?;
            }
        }
        OutputMode::CountMatches => {
            for file in &files {
                writeln!(w, "{}:{}", file.path.display(), file.hits.len())?;
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

    /// Two matches sharing a line print as one row, not two — printing one
    /// row per `Hit` used to double the line, both here and in `rg`'s own
    /// default text mode, which never repeats a line for extra matches on it.
    #[test]
    fn two_matches_on_one_line_print_as_one_row() {
        let two_on_one_line = "fn alpha() { fn nested() {} }\nfn beta() {}\n";
        let mut opts = Options::new(vec!["fn".into()]);
        opts.heading = Some(false);
        let found = hits(two_on_one_line, &opts);
        assert_eq!(lines_of(&found), vec![1, 1, 2], "fixture: two matches on line 1");

        let files = vec![FileHits {
            path: PathBuf::from("a.rs"),
            hits: found,
            total: 3,
            context: Vec::new(),
        }];
        let out = rendered(&files, &opts, false);
        let rows: Vec<&str> = out.lines().collect();
        assert_eq!(
            rows.len(),
            2,
            "one row per line, not per match: {rows:?}"
        );
        assert!(
            rows[0].starts_with("a.rs:1:1:"),
            "reports the leftmost match's column: {:?}",
            rows[0]
        );
    }

    /// Same grouping, under `--heading`'s separate rendering branch.
    #[test]
    fn two_matches_on_one_line_print_as_one_row_under_heading() {
        let two_on_one_line = "fn alpha() { fn nested() {} }\nfn beta() {}\n";
        let mut opts = Options::new(vec!["fn".into()]);
        opts.heading = Some(true);
        let files = vec![FileHits {
            path: PathBuf::from("a.rs"),
            hits: hits(two_on_one_line, &opts),
            total: 3,
            context: Vec::new(),
        }];
        let out = rendered(&files, &opts, false);
        let match_rows = out
            .lines()
            .filter(|l| l.trim_start().starts_with("1:") || l.trim_start().starts_with("2:"))
            .count();
        assert_eq!(match_rows, 2, "grouped under heading mode too: {out:?}");
    }

    /// `-o` is the deliberate exception: each match is its own line even when
    /// several share a source line, so it must stay ungrouped.
    #[test]
    fn only_matching_keeps_each_match_as_its_own_group() {
        let two_on_one_line = "fn alpha() { fn nested() {} }\nfn beta() {}\n";
        let mut opts = Options::new(vec!["fn".into()]);
        opts.only_matching = true;
        let found = hits(two_on_one_line, &opts);
        let groups = match_groups(&found, &opts);
        assert_eq!(
            groups.len(),
            3,
            "-o prints one line per match, even sharing a source line"
        );
        assert!(groups.iter().all(|g| g.len() == 1));
    }

    /// `--vimgrep` is the same exception as `-o`, for a different reason —
    /// quickfix format, not "nothing left to group by".
    #[test]
    fn vimgrep_keeps_each_match_as_its_own_group() {
        let two_on_one_line = "fn alpha() { fn nested() {} }\nfn beta() {}\n";
        let mut opts = Options::new(vec!["fn".into()]);
        opts.vimgrep = true;
        let found = hits(two_on_one_line, &opts);
        let groups = match_groups(&found, &opts);
        assert_eq!(groups.len(), 3, "one line per match, even sharing a source line");
        assert!(groups.iter().all(|g| g.len() == 1));
    }

    /// `--vimgrep` reaches `--json` too, through the same `match_groups`
    /// call `-o` already goes through there — not a special case, the same
    /// rule applied uniformly.
    #[test]
    fn vimgrep_splits_json_events_back_to_one_per_match() {
        let two_on_one_line = "fn alpha() { fn nested() {} }\nfn beta() {}\n";
        let mut opts = Options::new(vec!["fn".into()]);
        opts.json = true;
        opts.vimgrep = true;
        let files = vec![FileHits {
            path: PathBuf::from("a.rs"),
            hits: hits(two_on_one_line, &opts),
            total: 3,
            context: Vec::new(),
        }];
        let mut buf = Vec::new();
        print_json(&mut buf, &files, &opts, None, 0).unwrap();
        let events: Vec<serde_json::Value> = String::from_utf8(buf)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        let matches: Vec<&serde_json::Value> =
            events.iter().filter(|e| e["type"] == "match").collect();
        assert_eq!(matches.len(), 3, "one event per match, not per line: {matches:?}");
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

    /// Real `rg --json` folds same-line matches into one `match` event with a
    /// multi-entry `submatches` array, rather than one event per match. This
    /// pins that shape down — it is the whole point of speaking rg's dialect.
    #[test]
    fn two_matches_on_one_line_are_one_json_event_with_two_submatches() {
        let two_on_one_line = "fn alpha() { fn nested() {} }\nfn beta() {}\n";
        let mut opts = Options::new(vec!["fn".into()]);
        opts.json = true;
        let files = vec![FileHits {
            path: PathBuf::from("a.rs"),
            hits: hits(two_on_one_line, &opts),
            total: 3,
            context: Vec::new(),
        }];
        let mut buf = Vec::new();
        print_json(&mut buf, &files, &opts, None, 0).unwrap();
        let events: Vec<serde_json::Value> = String::from_utf8(buf)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).expect("every line is one JSON event"))
            .collect();
        let matches: Vec<&serde_json::Value> =
            events.iter().filter(|e| e["type"] == "match").collect();

        assert_eq!(matches.len(), 2, "one event per line, not per match: {matches:?}");
        assert_eq!(matches[0]["data"]["line_number"], 1);
        assert_eq!(
            matches[0]["data"]["submatches"].as_array().unwrap().len(),
            2,
            "both of line 1's matches ride in one event"
        );
        assert_eq!(matches[1]["data"]["line_number"], 2);
        assert_eq!(matches[1]["data"]["submatches"].as_array().unwrap().len(), 1);
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

    /// `-c` counts lines and `--count-matches` counts matches, which differ the
    /// moment a line holds two of them.
    #[test]
    fn counting_distinguishes_lines_from_matches() {
        let opts = Options::new(vec!["fn".into()]);
        let file = FileHits {
            path: PathBuf::from("a.rs"),
            hits: hits("fn alpha() { fn nested() {} }\nfn beta() {}\n", &opts),
            total: 3,
            context: Vec::new(),
        };
        assert_eq!(file.hits.len(), 3, "three matches");
        assert_eq!(file.matched_lines(), 2, "on two lines");
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

/// `shown_symbols` is what an `Outcome.shown` caller would journal, if one
/// exists — `find` itself no longer does; see the module doc's "Why `find`
/// does not journal its own reads" and `journal.rs`'s "one writer" doc.
/// These tests pin down the credit rule that computation still has to get
/// right regardless of who, if anyone, consumes `Outcome.shown`.
#[cfg(test)]
mod shown_symbols_tests {
    use super::*;

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

#[cfg(test)]
mod head_limit_tests {
    use super::*;

    fn file_of(path: &str, hits: usize) -> FileHits {
        FileHits {
            path: PathBuf::from(path),
            hits: (0..hits)
                .map(|i| Hit {
                    line: i as u32 + 1,
                    column: 1,
                    byte: i as u32,
                    text: b"x".to_vec(),
                    span: (0, 1),
                    symbol: None,
                })
                .collect(),
            total: hits,
            context: Vec::new(),
        }
    }

    /// `--head-limit` caps what is printed. Applying it to `-l` would drop
    /// files from a listing that claims to be every file with a match, and
    /// applying it to `-c` would report a count that is really the limit.
    #[test]
    fn counting_modes_are_not_subject_to_head_limit() {
        for mode in [
            OutputMode::FilesWithMatches,
            OutputMode::Count,
            OutputMode::CountMatches,
            OutputMode::Quiet,
        ] {
            assert!(
                !mode.shows_source(),
                "{mode:?} prints no matches, so the cap must not reach it"
            );
        }

        // The cap itself still works where matches are printed.
        let mut files = vec![file_of("a.rs", 300), file_of("b.rs", 5)];
        assert_eq!(apply_head_limit(&mut files, 200), 105);
        assert_eq!(files.len(), 1, "b.rs had no room left");
        assert_eq!(files[0].hits.len(), 200);
    }
}

/// The invariant the whole pipeline rests on: a file that cannot match is never
/// parsed. Pinned with a parser that panics rather than by timing, which would
/// be both flaky and unable to tell "fast" from "skipped".
#[cfg(test)]
mod prefilter_tests {
    use super::*;
    use crate::parser::LanguageParser;
    use crate::symbols::FileSymbols;

    /// Claims `.probe` files and explodes if anything asks it to parse one.
    struct ExplodingParser;

    impl LanguageParser for ExplodingParser {
        fn extensions(&self) -> &[&str] {
            &["probe"]
        }

        fn parse_file(&self, _path: &Path, _source: &str) -> color_eyre::Result<FileSymbols> {
            panic!("parsed a file it did not need to parse");
        }

        fn language(&self) -> tree_sitter::Language {
            tree_sitter_rust::LANGUAGE.into()
        }

        fn tags_query(&self) -> &'static str {
            ""
        }
    }

    fn registry() -> ParserRegistry {
        let mut registry = ParserRegistry::new();
        registry.register(Box::new(ExplodingParser));
        registry
    }

    fn probe(body: &str) -> (tempfile::TempDir, PathBuf, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let abs = dir.path().join("a.probe");
        std::fs::write(&abs, body).unwrap();
        (dir, abs, PathBuf::from("a.probe"))
    }

    #[test]
    fn a_file_with_no_match_is_never_parsed() {
        let (_dir, abs, rel) = probe("nothing of interest here\n");
        let opts = Options::new(vec!["needle".into()]);
        let matcher = Matcher::new(&opts).unwrap();

        assert!(search_file(&matcher, &registry(), &abs, &rel, &opts, None).is_none());
    }

    /// The other half, without which the first proves nothing: a file that
    /// *does* match is parsed, so the parser really was reachable.
    #[test]
    #[should_panic(expected = "parsed a file it did not need to parse")]
    fn a_file_with_a_match_is_parsed_for_attribution() {
        let (_dir, abs, rel) = probe("here is a needle\n");
        let opts = Options::new(vec!["needle".into()]);
        let matcher = Matcher::new(&opts).unwrap();

        search_file(&matcher, &registry(), &abs, &rel, &opts, None);
    }

    /// …and modes that show no source skip the parse even when it matched,
    /// because attribution is the only thing the parse was for.
    #[test]
    fn counting_modes_do_not_parse_at_all() {
        let (_dir, abs, rel) = probe("here is a needle\n");
        let mut opts = Options::new(vec!["needle".into()]);
        opts.mode = OutputMode::FilesWithMatches;
        let matcher = Matcher::new(&opts).unwrap();

        let found = search_file(&matcher, &registry(), &abs, &rel, &opts, None).unwrap();
        assert_eq!(found.hits.len(), 1, "it matched, it just was not parsed");
    }

    /// The whole-file prefilter used to run `re` — compiled multi-line only
    /// under `-U` — against the entire buffer at once, so `^` bound to the
    /// true start of the *file*, not of a line. A file whose first line did
    /// not itself satisfy an anchored pattern was rejected outright, silently
    /// dropping every real match on the lines below. `prefilter_re` is always
    /// multi-line, specifically so this file is not skipped.
    #[test]
    fn an_anchored_pattern_still_finds_a_match_past_the_first_line() {
        let (_dir, abs, rel) =
            probe("first line has nothing to do with it\nneedle shows up here\n");
        let mut opts = Options::new(vec!["^needle".into()]);
        // This suite's registry explodes on parse; attribution is not what
        // this test is about, so skip it the same way the tests above do.
        opts.mode = OutputMode::FilesWithMatches;
        let matcher = Matcher::new(&opts).unwrap();

        let found = search_file(&matcher, &registry(), &abs, &rel, &opts, None)
            .expect("the prefilter must not reject this file");
        assert_eq!(found.hits.len(), 1);
        assert_eq!(found.hits[0].line, 2);
    }

    /// Same shape, for `$`.
    #[test]
    fn an_end_anchored_pattern_still_finds_a_match_past_the_first_line() {
        let (_dir, abs, rel) = probe("first line\nthis one ends with needle\n");
        let mut opts = Options::new(vec!["needle$".into()]);
        opts.mode = OutputMode::FilesWithMatches;
        let matcher = Matcher::new(&opts).unwrap();

        let found = search_file(&matcher, &registry(), &abs, &rel, &opts, None)
            .expect("the prefilter must not reject this file");
        assert_eq!(found.hits.len(), 1);
        assert_eq!(found.hits[0].line, 2);
    }
}
