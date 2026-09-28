//! Markdown symbol extractor: headings only.
//!
//! Unlike the other language parsers, this one never walks a tree-sitter CST —
//! it hand-scans `source` line by line for ATX headings (`#` through `######`)
//! and builds nesting from a simple heading-level stack. Everything else in
//! Markdown (lists, links, emphasis, tables, code spans) is left unmodeled;
//! trading completeness for a parser this small is the whole point.
//!
//! A [`tree_sitter::Language`] is still required to satisfy [`LanguageParser`]
//! — `callers::call_sites_filtered` calls `language()`/`tags_query()`
//! unconditionally on any file the registry claims, to build a reference-
//! finding query — so `tree-sitter-md`'s grammar is used to answer that, with
//! an empty `tags_query()` since Markdown has no calls to find.
//!
//! ## Heading body ranges
//!
//! A heading's body — and therefore its `content_hash`/`estimated_tokens` —
//! runs from its own line through everything nested beneath it, up to (but
//! not including) the next heading of equal or shallower depth. This mirrors
//! how a Rust `mod`/`impl` block's range covers its members.
//!
//! Content before the first heading (or an entire file with none) is not
//! ignored: it becomes a synthesized top-level `preamble` symbol so it stays
//! addressable via `ambits show` like everything else, provided it is not
//! empty/all-whitespace.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use color_eyre::eyre::Result;

use crate::symbols::merkle::{compute_merkle_hash, content_hash, estimate_tokens};
use crate::symbols::{FileSymbols, NameInterner, SymbolCategory, SymbolNode};

use super::{LanguageParser, SymbolMeta};

pub struct MarkdownParser {
    _private: (),
}

impl MarkdownParser {
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Default for MarkdownParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Heading metadata by depth (`HEADINGS[0]` is `h1`), mirroring the
/// `SymbolMeta` const-table pattern the other parsers use for node kinds.
const HEADINGS: [SymbolMeta; 6] = [
    SymbolMeta { category: SymbolCategory::Module, label: "h1" },
    SymbolMeta { category: SymbolCategory::Module, label: "h2" },
    SymbolMeta { category: SymbolCategory::Module, label: "h3" },
    SymbolMeta { category: SymbolCategory::Module, label: "h4" },
    SymbolMeta { category: SymbolCategory::Module, label: "h5" },
    SymbolMeta { category: SymbolCategory::Module, label: "h6" },
];

const PREAMBLE: SymbolMeta = SymbolMeta { category: SymbolCategory::Module, label: "preamble" };

impl LanguageParser for MarkdownParser {
    fn extensions(&self) -> &[&str] {
        &["md"]
    }

    fn language(&self) -> tree_sitter::Language {
        tree_sitter_md::LANGUAGE.into()
    }

    fn tags_query(&self) -> &'static str {
        // Markdown has no "calls" to extract. `call_sites_filtered` degrades
        // an empty query to "no call sites" rather than erroring.
        ""
    }

    fn parse_file(&self, path: &Path, source: &str) -> Result<FileSymbols> {
        let path_prefix = path.to_string_lossy();
        let file_path_arc = Arc::new(path.to_path_buf());
        let names = NameInterner::new();

        let headings = scan_headings(source);
        let mut symbols = build_tree(&headings, source, &file_path_arc, &names, &path_prefix);

        for sym in symbols.iter_mut() {
            compute_merkle_hash(sym);
        }

        Ok(FileSymbols {
            file_path: path.to_path_buf(),
            symbols,
            total_lines: source.lines().count(),
        })
    }
}

/// A single detected ATX heading.
struct Heading {
    /// 1-6.
    level: usize,
    /// Heading text with the leading `#`s and one separating space stripped.
    text: String,
    /// Byte offset of the heading line's first character.
    start_byte: usize,
    /// 1-based line number of the heading line.
    start_line: usize,
}

/// Scan `source` for ATX headings, skipping lines inside fenced code blocks
/// so a `# comment` in a shell/Python example isn't mistaken for one.
///
/// Single pass over `source.split_inclusive('\n')`, tracking a running byte
/// offset — no tree-sitter parse involved.
fn scan_headings(source: &str) -> Vec<Heading> {
    let mut headings = Vec::new();
    let mut in_fence = false;
    let mut offset = 0usize;

    for (line_no, raw_line) in source.split_inclusive('\n').enumerate() {
        let line_no = line_no + 1;
        let content = raw_line.strip_suffix('\n').unwrap_or(raw_line);
        let content = content.strip_suffix('\r').unwrap_or(content);
        let trimmed = content.trim_start();

        if is_fence_line(trimmed) {
            in_fence = !in_fence;
        } else if !in_fence {
            if let Some((level, text)) = parse_atx_heading(trimmed) {
                headings.push(Heading { level, text, start_byte: offset, start_line: line_no });
            }
        }

        offset += raw_line.len();
    }

    headings
}

/// A fenced code block delimiter — opens or closes a fence depending on
/// whether one is already open. The closing marker isn't checked against the
/// opening one (` ``` ` vs `~~~`); toggling on either is simpler and the
/// mismatch case doesn't occur in practice.
fn is_fence_line(trimmed: &str) -> bool {
    trimmed.starts_with("```") || trimmed.starts_with("~~~")
}

/// Parse a trimmed line as an ATX heading: 1-6 `#` characters followed by a
/// space or end of line. `"#tag"` is not a heading (no separating space);
/// `"#######"` (7+) is not a heading either.
///
/// CommonMark also strips a closing run of `#`s and trailing whitespace from
/// the heading text; skipped here for simplicity since the untrimmed text is
/// still a perfectly usable display name.
fn parse_atx_heading(trimmed: &str) -> Option<(usize, String)> {
    let hashes = trimmed.chars().take_while(|&c| c == '#').count();
    if hashes == 0 || hashes > 6 {
        return None;
    }
    let rest = &trimmed[hashes..];
    if !rest.is_empty() && !rest.starts_with(' ') && !rest.starts_with('\t') {
        return None;
    }
    Some((hashes, rest.trim().to_string()))
}

/// One heading whose body is still being accumulated: its own metadata plus
/// the children collected so far.
struct OpenHeading {
    level: usize,
    text: String,
    start_byte: usize,
    start_line: usize,
    name_path: String,
    children: Vec<SymbolNode>,
}

/// Build the nested symbol tree from a flat heading list via a level stack:
/// push each heading; before pushing, pop (and close) every open heading
/// whose level is `>=` the new one, since a heading only continues nesting
/// under a *shallower* one. Remaining open headings close at EOF.
fn build_tree(
    headings: &[Heading],
    source: &str,
    file_path: &Arc<PathBuf>,
    names: &NameInterner,
    path_prefix: &str,
) -> Vec<SymbolNode> {
    let total_len = source.len();
    let total_lines = source.lines().count();
    let mut roots: Vec<SymbolNode> = Vec::new();
    let mut stack: Vec<OpenHeading> = Vec::new();

    let first_start = headings.first().map(|h| h.start_byte).unwrap_or(total_len);
    let first_end_line = headings.first().map(|h| h.start_line - 1).unwrap_or(total_lines);
    if let Some(sym) = make_preamble(source, first_start, first_end_line, file_path, names, path_prefix) {
        roots.push(sym);
    }

    for heading in headings {
        while stack.last().is_some_and(|open| open.level >= heading.level) {
            let open = stack.pop().unwrap();
            let sym = finish_heading(
                open,
                heading.start_byte,
                heading.start_line - 1,
                source,
                file_path,
                names,
                path_prefix,
            );
            push_child(&mut stack, &mut roots, sym);
        }

        let name_path = match stack.last() {
            Some(parent) => format!("{}/{}", parent.name_path, heading.text),
            None => heading.text.clone(),
        };

        stack.push(OpenHeading {
            level: heading.level,
            text: heading.text.clone(),
            start_byte: heading.start_byte,
            start_line: heading.start_line,
            name_path,
            children: Vec::new(),
        });
    }

    while let Some(open) = stack.pop() {
        let sym = finish_heading(open, total_len, total_lines, source, file_path, names, path_prefix);
        push_child(&mut stack, &mut roots, sym);
    }

    roots
}

fn push_child(stack: &mut [OpenHeading], roots: &mut Vec<SymbolNode>, sym: SymbolNode) {
    match stack.last_mut() {
        Some(parent) => parent.children.push(sym),
        None => roots.push(sym),
    }
}

fn finish_heading(
    open: OpenHeading,
    end_byte: usize,
    end_line: usize,
    source: &str,
    file_path: &Arc<PathBuf>,
    names: &NameInterner,
    path_prefix: &str,
) -> SymbolNode {
    let text = &source[open.start_byte..end_byte];
    let meta = &HEADINGS[open.level - 1];

    SymbolNode {
        id: format!("{path_prefix}::{}", open.name_path),
        name: names.intern(&open.text),
        category: meta.category,
        label: meta.label,
        file_path: Arc::clone(file_path),
        byte_range: open.start_byte as u32..end_byte as u32,
        line_range: open.start_line as u32..end_line.max(open.start_line) as u32,
        content_hash: content_hash(text),
        merkle_hash: [0u8; 32],
        children: open.children,
        estimated_tokens: estimate_tokens(text) as u32,
    }
}

/// The synthesized top-level symbol for content before the first heading (or
/// the whole file, if there are no headings). `None` if that span is empty or
/// all whitespace — same as today's behavior for an empty/headingless file.
fn make_preamble(
    source: &str,
    end_byte: usize,
    end_line: usize,
    file_path: &Arc<PathBuf>,
    names: &NameInterner,
    path_prefix: &str,
) -> Option<SymbolNode> {
    let text = &source[..end_byte];
    if text.trim().is_empty() {
        return None;
    }

    Some(SymbolNode {
        id: format!("{path_prefix}::preamble"),
        name: names.intern("preamble"),
        category: PREAMBLE.category,
        label: PREAMBLE.label,
        file_path: Arc::clone(file_path),
        byte_range: 0..end_byte as u32,
        line_range: 1..end_line.max(1) as u32,
        content_hash: content_hash(text),
        merkle_hash: [0u8; 32],
        children: Vec::new(),
        estimated_tokens: estimate_tokens(text) as u32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(src: &str) -> Vec<SymbolNode> {
        MarkdownParser::new()
            .parse_file(Path::new("test.md"), src)
            .unwrap()
            .symbols
    }

    #[test]
    fn extensions_is_md() {
        assert_eq!(MarkdownParser::new().extensions(), &["md"]);
    }

    #[test]
    fn empty_file_has_no_symbols() {
        assert!(parse("").is_empty());
    }

    #[test]
    fn flat_sequence_of_same_level_headings() {
        let syms = parse("# One\nbody one\n# Two\nbody two\n");
        assert_eq!(syms.len(), 2);
        assert_eq!(syms[0].name.as_ref(), "One");
        assert_eq!(syms[0].label, "h1");
        assert_eq!(syms[0].id, "test.md::One");
        assert!(syms[0].children.is_empty());
        assert_eq!(syms[1].name.as_ref(), "Two");
        assert_eq!(syms[1].id, "test.md::Two");
    }

    #[test]
    fn nested_headings_by_level() {
        let syms = parse("# A\n## B\n### C\n# D\n");
        assert_eq!(syms.len(), 2, "two top-level headings: A and D");

        let a = &syms[0];
        assert_eq!(a.id, "test.md::A");
        assert_eq!(a.children.len(), 1);

        let b = &a.children[0];
        assert_eq!(b.id, "test.md::A/B");
        assert_eq!(b.label, "h2");
        assert_eq!(b.children.len(), 1);

        let c = &b.children[0];
        assert_eq!(c.id, "test.md::A/B/C");
        assert_eq!(c.label, "h3");
        assert!(c.children.is_empty());

        let d = &syms[1];
        assert_eq!(d.id, "test.md::D");
        assert!(d.children.is_empty(), "D is a sibling of A, not nested under it");
    }

    #[test]
    fn a_shallower_heading_closes_a_deeper_open_one() {
        // ## under # under nothing, then a bare # again should close both.
        let syms = parse("# A\n## B\n# C\n");
        assert_eq!(syms.len(), 2);
        assert_eq!(syms[0].children.len(), 1);
        assert_eq!(syms[0].children[0].id, "test.md::A/B");
        assert!(syms[1].children.is_empty());
    }

    #[test]
    fn content_before_the_first_heading_becomes_preamble() {
        let syms = parse("Some intro text.\n\n# Heading\nbody\n");
        assert_eq!(syms.len(), 2);
        assert_eq!(syms[0].label, "preamble");
        assert_eq!(syms[0].id, "test.md::preamble");
        assert_eq!(syms[1].label, "h1");
    }

    #[test]
    fn a_file_with_no_headings_is_entirely_preamble() {
        let syms = parse("Just prose.\nNo headings anywhere.\n");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].label, "preamble");
        assert_eq!(syms[0].line_range, 1..2);
    }

    #[test]
    fn whitespace_only_file_has_no_preamble() {
        assert!(parse("   \n\n\t\n").is_empty());
    }

    #[test]
    fn a_hash_inside_a_fenced_code_block_is_not_a_heading() {
        let src = "# Real\n```bash\n# not a heading\necho hi\n```\nmore body\n";
        let syms = parse(src);
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].id, "test.md::Real");
        // The fenced block content is still part of the heading's own body.
        assert!(syms[0].byte_range.end as usize >= src.len() - 1);
    }

    #[test]
    fn a_line_of_only_hashes_needs_a_separating_space() {
        // Not a heading, so it's just body text — falls into the preamble
        // ahead of the real heading rather than vanishing.
        let syms = parse("#no-space-not-a-heading\n# Real Heading\n");
        assert_eq!(syms.len(), 2);
        assert_eq!(syms[0].label, "preamble");
        assert_eq!(syms[1].name.as_ref(), "Real Heading");
    }

    #[test]
    fn seven_hashes_is_not_a_heading() {
        // Not a heading, so the whole file is just body text (preamble).
        let syms = parse("####### Too Deep\n");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].label, "preamble");
    }

    #[test]
    fn line_ranges_cover_through_the_next_heading() {
        let syms = parse("# A\nline2\nline3\n# B\nline5\n");
        assert_eq!(syms[0].line_range, 1..3);
        assert_eq!(syms[1].line_range, 4..5);
    }
}
