//! Which symbols an agent's write touched.
//!
//! Spec: `docs/spikes/coverage-snapshots.md` §2. Two layers, both free of
//! I/O: [`attribute`] is the core — given the file before and after a write,
//! the write's diff hunks, and the file's parser, it answers which symbols
//! the write changed or removed; [`build_record`] turns an ingested
//! [`WriteEvent`] into the [`WriteRecord`] the journal stores, choosing
//! symbol- or file-level attribution per spec §2.2.
//!
//! ## Changed lines, not hunk ranges
//!
//! Claude Code's `structuredPatch` hunks carry three lines of unchanged
//! context on each side, so overlapping a hunk's *range* with symbol spans
//! over-attributes the neighbours of a small edit. Instead each hunk's
//! `lines` are walked, tracking old and new line numbers, and only `+` and
//! `-` lines attribute:
//!
//! - a `+` line at new line *n* touches the innermost symbol in *after*
//!   containing *n*;
//! - a `-` line at old line *n* names the innermost symbol in *before*
//!   containing *n*: touched if a symbol with that id exists in *after*
//!   (ids are not unique, so any match counts), removed otherwise.
//!
//! A changed line inside no symbol — a `use` line, a gap, a detached
//! comment — sets [`Attribution::outside_symbols`] rather than being
//! dropped.
//!
//! ## No change to the symbol representation
//!
//! Everything here is a function of existing [`FileSymbols`]/[`SymbolNode`]
//! values (spec §2.4). `SymbolNode::line_range` holds an **inclusive** end
//! despite being a `Range`, so containment is `start <= n && n <= end`,
//! never `Range::contains`.

use std::collections::BTreeSet;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::ingest::{WriteEvent, WriteSource};
use crate::parser::{LanguageParser, ParserRegistry};
use crate::symbols::{nested_in, split_id, FileSymbols, SymbolId, SymbolNode};

/// Version of the attribution rules. Recorded on every write (`av`) so a
/// write re-attributed after the rules change replaces the old record rather
/// than duplicating it (spec §2.6).
///
/// 2: a write is symbol-level only when its hunks reproduce the file exactly;
/// a missing or malformed patch is file-level.
pub const ATTRIBUTION_VERSION: u32 = 2;

/// How precisely a write was attributed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    /// Changed symbols named, from the session log (spec D15).
    Symbol,
    /// Only the file is known: the log lacked the text to attribute, the
    /// file has no parser, or the reconstruction did not verify.
    #[default]
    File,
}

/// One agent write, as journaled (spec §2.6). One record per `(session,
/// op)`; the fold keeps the highest `av`. `Default` is for building
/// fixtures field by field.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WriteRecord {
    /// The tool call's `tool_use_id`.
    pub op: String,
    /// [`ATTRIBUTION_VERSION`] that produced this record.
    pub av: u32,
    /// Agent id.
    pub a: String,
    /// Timestamp of the tool result.
    pub t: String,
    pub tool: String,
    /// Project-relative, `/`-separated.
    pub file: String,
    pub level: Level,
    /// Some changed line fell inside no symbol.
    #[serde(default)]
    pub outside_symbols: bool,
    /// Innermost touched symbols with their post-write hash (`b3:…`). Hashes
    /// travel with ids because ids are not unique.
    #[serde(default)]
    pub syms: Vec<(String, String)>,
    /// Innermost symbols the write deleted.
    #[serde(default)]
    pub removed: Vec<String>,
    /// For a `Write`, BLAKE3 of the new content, so even a file-level `Write`
    /// can later be matched to the commit it landed in. Hashing is allowed;
    /// storing contents is not (spec §9.6).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fh: Option<String>,
}

/// Whether a write's version of a file or symbol is still what is there.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    /// Unchanged since the agent wrote it.
    Current,
    /// Changed since.
    Changed,
    /// No longer exists.
    Removed,
    /// A file-level write with no hash to compare (spec §3.1).
    Unknown,
}

impl WriteRecord {
    /// Whether this write changed symbol `id`, through itself or anything
    /// nested in it: writes record innermost symbols only (D11), so an edit
    /// inside `App/handle_key` is an edit to `App`. Only symbol-level
    /// writes count — a file-level write cannot say which symbols it
    /// changed, and a symbol-level one names every symbol it did (§2.3).
    pub fn touches_symbol(&self, id: &str) -> bool {
        self.syms.iter().any(|(s, _)| nested_in(s, id)) || self.removed.iter().any(|s| nested_in(s, id))
    }

    /// Latest first: by timestamp (RFC 3339 UTC, so lexicographic order is
    /// chronological; none sorts oldest), then by `op`, so ties resolve the
    /// same way every time.
    pub fn recency(&self) -> (&str, &str) {
        (&self.t, &self.op)
    }
}

/// What one version of a file holds, as writes are checked against it:
/// the hash of its raw bytes, and each symbol's content hash by name path.
/// The one reading of "is the agent's version still there" — for the file
/// on disk (`touched`) and for a committed blob (linkage) alike.
#[derive(Debug, Default)]
pub struct FileContents {
    /// [`crate::objects::file_hash`] of the bytes.
    pub hash: String,
    /// Name path → content hashes (ids are not unique, so possibly several).
    symbols: std::collections::HashMap<String, Vec<String>>,
    /// Whether the bytes parsed; without that, only `hash` means anything.
    pub parsed: bool,
}

impl FileContents {
    /// Read `bytes` as project file `rel`. The bytes are not kept (§9.6).
    pub fn read(rel: &str, bytes: &[u8], registry: &ParserRegistry) -> Self {
        let path = Path::new(rel);
        let parsed = std::str::from_utf8(bytes)
            .ok()
            .zip(registry.parser_for(path))
            .and_then(|(source, parser)| parser.parse_file(path, source).ok());
        let mut out = parsed.as_ref().map(Self::from_symbols).unwrap_or_default();
        out.hash = crate::objects::file_hash(bytes);
        out
    }

    /// An already-parsed file, as the TUI holds it. No `hash`: the bytes are
    /// not at hand, so a write known only by its file hash is
    /// [`Status::Unknown`] here.
    pub fn from_symbols(file: &FileSymbols) -> Self {
        let mut out = Self { parsed: true, ..Default::default() };
        for sym in file.walk() {
            out.symbols.entry(sym.name_path().to_string()).or_default().push(crate::journal::encode_hash(&sym.content_hash));
        }
        out
    }

    /// Whether `write`'s version of symbol `id` is what this file holds.
    /// Everything the write left under `id` must be as it left it: written
    /// symbols at their hash, deleted ones still gone. That includes `id`
    /// itself, which is back if the write deleted it.
    pub fn symbol_status(&self, id: &str, write: &WriteRecord) -> Status {
        if !self.parsed {
            return Status::Unknown;
        }
        let name = |id: &str| split_id(id).1.to_string();
        // Present if anything by that name path is: an inherent impl is
        // `impl App`, but its methods are `App/…`, so `App` can have
        // children with no symbol of its own.
        if !self.any_within(&name(id)) {
            return Status::Removed;
        }
        let written_intact = write.syms.iter().filter(|(s, _)| nested_in(s, id)).all(|(s, h)| self.has(&name(s), h));
        let deleted_gone = write.removed.iter().filter(|s| nested_in(s, id)).all(|s| self.hashes(&name(s)).is_empty());
        if written_intact && deleted_gone { Status::Current } else { Status::Changed }
    }

    /// Whether `write`'s version of the whole file is what this holds: by
    /// the file hash when both sides have one, else by every symbol it
    /// wrote. A file-level write with no hash cannot say (spec §3.1).
    pub fn file_status(&self, write: &WriteRecord) -> Status {
        if let (Some(fh), false) = (&write.fh, self.hash.is_empty()) {
            return if *fh == self.hash { Status::Current } else { Status::Changed };
        }
        if write.level == Level::File || write.syms.is_empty() || !self.parsed {
            return Status::Unknown;
        }
        let unchanged = write.syms.iter().all(|(s, h)| self.has(split_id(s).1, h));
        if unchanged { Status::Current } else { Status::Changed }
    }

    /// Whether a symbol at `name_path` has content hash `hash`.
    pub fn has(&self, name_path: &str, hash: &str) -> bool {
        self.hashes(name_path).iter().any(|h| h == hash)
    }

    /// Content hashes of the symbols at `name_path`; empty when none is.
    pub fn hashes(&self, name_path: &str) -> &[String] {
        self.symbols.get(name_path).map_or(&[], Vec::as_slice)
    }

    /// Whether any symbol is at `name_path` or nested in it.
    pub fn any_within(&self, name_path: &str) -> bool {
        self.symbols.keys().any(|k| crate::symbols::nested_in(k, name_path))
    }
}

/// Turn a write event into its journal record (spec §2.2).
///
/// `None` for a write outside `project_root` (spec §2.6): neither portable
/// nor the project's. `symbol_level` is `false` in Serena mode, whose tree
/// ids need not match a tree-sitter parse (spec §2.5).
pub fn build_record(
    event: &WriteEvent,
    project_root: &Path,
    registry: &ParserRegistry,
    symbol_level: bool,
) -> Option<WriteRecord> {
    let rel = if event.path.is_absolute() {
        event.path.strip_prefix(project_root).ok()?
    } else {
        event.path.as_path()
    };
    // Normalized, and parsed under the normalized name, so the file and its
    // symbol ids match what `touched`, snapshots and linkage look up.
    let file = crate::objects::project_rel(rel)?;
    let rel = Path::new(&file);

    let fh = match &event.source {
        WriteSource::Write { content, .. } => Some(crate::objects::file_hash(content.as_bytes())),
        _ => None,
    };

    let attribution = symbol_level
        .then(|| registry.parser_for(rel))
        .flatten()
        .and_then(|parser| match &event.source {
            WriteSource::Edit {
                original: Some(before),
                old,
                new,
                replace_all,
                hunks: Some(hunks),
                user_modified: false,
            } => {
                let after = apply_edit(before, old, new, *replace_all)?;
                attribute(parser, rel, before, &after, Diff::Hunks(hunks))
            }
            WriteSource::Write { create: true, content, user_modified: false, .. } => {
                attribute(parser, rel, "", content, Diff::Created)
            }
            WriteSource::Write {
                original: Some(before),
                content,
                hunks: Some(hunks),
                user_modified: false,
                ..
            } => attribute(parser, rel, before, content, Diff::Hunks(hunks)),
            _ => None,
        });

    let (level, outside_symbols, syms, removed) = match attribution {
        Some(a) => (
            Level::Symbol,
            a.outside_symbols,
            a.touched
                .iter()
                .map(|(id, h)| (id.clone(), crate::journal::encode_hash(h)))
                .collect(),
            a.removed,
        ),
        None => (Level::File, false, Vec::new(), Vec::new()),
    };

    Some(WriteRecord {
        op: event.op.to_string(),
        av: ATTRIBUTION_VERSION,
        a: event.agent_id.to_string(),
        t: event.timestamp.clone(),
        tool: event.tool_name.to_string(),
        file,
        level,
        outside_symbols,
        syms,
        removed,
        fh,
    })
}

/// One diff hunk, as Claude Code records it in `structuredPatch`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hunk {
    /// 1-based first line of the hunk in the file before the write.
    pub old_start: u32,
    /// 1-based first line of the hunk in the file after the write.
    pub new_start: u32,
    /// Hunk body: each line prefixed with `' '`, `'-'` or `'+'`.
    pub lines: Vec<String>,
}

/// What a write changed, at symbol level.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Attribution {
    /// Innermost symbols the write changed, with their hash *after* the
    /// write. Sorted and deduplicated, so equal inputs give equal output.
    pub touched: Vec<(SymbolId, [u8; 32])>,
    /// Innermost symbols the write deleted: present before, no symbol with
    /// that id after. Sorted and deduplicated.
    pub removed: Vec<SymbolId>,
    /// Some changed line fell inside no symbol.
    pub outside_symbols: bool,
}

/// Apply an `Edit` to `before`: replace `old` with `new` — every occurrence
/// if `replace_all`, else the first.
///
/// `None` when `old` does not occur (or is empty), which the caller turns
/// into a file-level write rather than guessing. An `Edit` that succeeded in
/// Claude Code had a unique match unless `replace_all` was set, so "first"
/// and "only" coincide there.
pub fn apply_edit(before: &str, old: &str, new: &str, replace_all: bool) -> Option<String> {
    if old.is_empty() || !before.contains(old) {
        return None;
    }
    Some(if replace_all {
        before.replace(old, new)
    } else {
        before.replacen(old, new, 1)
    })
}

/// How a write changed its file, for [`attribute`].
#[derive(Debug, Clone, Copy)]
pub enum Diff<'a> {
    /// The file was created: there is no diff to walk, and every innermost
    /// symbol in `after` is touched (spec §2.2).
    Created,
    /// The write's hunks, which must account for every changed line.
    Hunks(&'a [Hunk]),
}

/// Attribute a write of `path` from `before` to `after`.
///
/// `None` when either side fails to parse, or when the hunks do not turn
/// `before` into exactly `after`: a context or `-` line that is not the
/// `before` line at its position, or a change the hunks leave out. The caller
/// then records a file-level write. The check makes attribution
/// self-verifying, and complete — so a symbol a symbol-level write does not
/// name really was left alone (spec §2.3). `after` is usually reconstructed
/// from `oldString`/`newString`, and anything else that changed the file in
/// the same write (Claude Code stamps memory files' frontmatter, for one)
/// would otherwise be attributed to the wrong lines, or missed. Against this
/// project's logs, 80 of 82 reconstructions matched; the two that did not
/// were such stamped memory files.
pub fn attribute(
    parser: &dyn LanguageParser,
    path: &Path,
    before: &str,
    after: &str,
    diff: Diff<'_>,
) -> Option<Attribution> {
    let after_syms = parser.parse_file(path, after).ok()?.symbols;
    let after_lines = lines(after);

    let hunks = match diff {
        Diff::Hunks(hunks) => hunks,
        Diff::Created => {
            let touched: BTreeSet<_> = crate::symbols::walk_symbols(&after_syms)
                .into_iter()
                .filter(|s| s.children.is_empty())
                .map(|s| (s.id.clone(), s.content_hash))
                .collect();
            // A created line in no symbol is as much a change outside
            // symbols as an edited one; without this, a file with no
            // symbols would record a write that changed nothing.
            let outside_symbols = (1..=after_lines.len() as u32).any(|line| {
                !after_lines[line as usize - 1].trim().is_empty()
                    && innermost_at(&after_syms, line).is_none()
            });
            return Some(Attribution {
                touched: touched.into_iter().collect(),
                outside_symbols,
                ..Attribution::default()
            });
        }
    };

    let before_lines = lines(before);
    if apply_hunks(&before_lines, hunks)? != after_lines {
        return None;
    }
    let before_syms = parser.parse_file(path, before).ok()?.symbols;

    let mut touched = BTreeSet::new();
    let mut removed = BTreeSet::new();
    let mut outside_symbols = false;

    for hunk in hunks {
        let mut old_line = hunk.old_start;
        let mut new_line = hunk.new_start;
        for line in &hunk.lines {
            match line.as_bytes().first() {
                Some(b'+') => {
                    match innermost_at(&after_syms, new_line) {
                        Some(s) => {
                            touched.insert((s.id.clone(), s.content_hash));
                        }
                        None => outside_symbols = true,
                    }
                    new_line += 1;
                }
                Some(b'-') => {
                    match innermost_at(&before_syms, old_line) {
                        Some(s) => {
                            let survivors = with_id(&after_syms, &s.id);
                            if survivors.is_empty() {
                                removed.insert(s.id.clone());
                            } else {
                                for kept in survivors {
                                    touched.insert((kept.id.clone(), kept.content_hash));
                                }
                            }
                        }
                        None => outside_symbols = true,
                    }
                    old_line += 1;
                }
                // `\ No newline at end of file` annotates the previous line
                // and occupies no line of its own.
                Some(b'\\') => {}
                // Context: present on both sides.
                _ => {
                    old_line += 1;
                    new_line += 1;
                }
            }
        }
    }

    Some(Attribution {
        touched: touched.into_iter().collect(),
        removed: removed.into_iter().collect(),
        outside_symbols,
    })
}

/// A text's lines, without the empty "line" after a final newline — so a
/// file and the same file with its trailing newline added or dropped (`\ No
/// newline at end of file`) differ only where the hunks say they do.
fn lines(text: &str) -> Vec<&str> {
    if text.is_empty() {
        return Vec::new();
    }
    text.strip_suffix('\n').unwrap_or(text).split('\n').collect()
}

/// Apply `hunks` to `before`, checking every context and `-` line against it.
/// `None` when a hunk does not fit `before` — out of order, out of range, or
/// naming a line that is not there.
fn apply_hunks<'a>(before: &[&'a str], hunks: &'a [Hunk]) -> Option<Vec<&'a str>> {
    let mut ordered: Vec<&Hunk> = hunks.iter().collect();
    ordered.sort_by_key(|h| h.old_start);
    let mut out = Vec::with_capacity(before.len());
    let mut next = 0; // index of the first `before` line not yet consumed
    for hunk in ordered {
        let body = |l: &'a String| l.get(1..).unwrap_or("");
        let consumes = hunk.lines.iter().any(|l| matches!(l.as_bytes().first(), Some(b' ' | b'-')));
        // Unified-diff convention: a hunk with no old lines inserts *after*
        // line `old_start`; any other starts *at* it.
        let start = if consumes { (hunk.old_start as usize).checked_sub(1)? } else { hunk.old_start as usize };
        if start < next || start > before.len() {
            return None;
        }
        out.extend_from_slice(&before[next..start]);
        next = start;
        for line in &hunk.lines {
            match line.as_bytes().first() {
                Some(b'+') => out.push(body(line)),
                Some(b'\\') => {}
                Some(b'-') | Some(b' ') => {
                    if before.get(next) != Some(&body(line)) {
                        return None;
                    }
                    if line.starts_with(' ') {
                        out.push(body(line));
                    }
                    next += 1;
                }
                _ => return None,
            }
        }
    }
    out.extend_from_slice(before.get(next..)?);
    Some(out)
}

/// The deepest symbol whose inclusive line range contains `line`.
fn innermost_at(symbols: &[SymbolNode], line: u32) -> Option<&SymbolNode> {
    crate::symbols::innermost(symbols, &|s| s.line_range.start <= line && line <= s.line_range.end)
}

/// Every symbol, at any depth, whose id is `id`.
fn with_id<'a>(symbols: &'a [SymbolNode], id: &str) -> Vec<&'a SymbolNode> {
    crate::symbols::walk_symbols(symbols).into_iter().filter(|s| &*s.id == id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::rust::RustParser;

    const PATH: &str = "src/lib.rs";

    const BEFORE: &str = "\
fn alpha() {
    let a = 1;
}

fn beta() {
    let b = 2;
}

fn gamma() {
    let c = 3;
}
";

    fn hunk(old_start: u32, new_start: u32, lines: &[&str]) -> Hunk {
        Hunk { old_start, new_start, lines: lines.iter().map(|l| l.to_string()).collect() }
    }

    fn run(before: &str, after: &str, diff: Diff<'_>) -> Attribution {
        attribute(&RustParser::new(), Path::new(PATH), before, after, diff).expect("parses")
    }

    fn touched_ids(a: &Attribution) -> Vec<&str> {
        a.touched.iter().map(|(id, _)| id.as_str()).collect()
    }

    /// Claude Code's hunks carry context lines around the change. Only the
    /// changed line attributes, even though the context reaches into the
    /// neighbouring functions.
    #[test]
    fn only_changed_lines_attribute_not_hunk_context() {
        let after = BEFORE.replace("let b = 2;", "let b = 20;");
        let hunks = [hunk(3, 3, &[
            " }",
            " ",
            " fn beta() {",
            "-    let b = 2;",
            "+    let b = 20;",
            " }",
            " ",
            " fn gamma() {",
        ])];
        let a = run(BEFORE, &after, Diff::Hunks(&hunks));
        assert_eq!(touched_ids(&a), vec!["src/lib.rs::beta"]);
        assert!(a.removed.is_empty());
        assert!(!a.outside_symbols);
    }

    /// The recorded hash is the symbol's hash *after* the write.
    #[test]
    fn the_touched_hash_is_the_post_write_hash() {
        let after = BEFORE.replace("let b = 2;", "let b = 20;");
        let hunks = [hunk(6, 6, &["-    let b = 2;", "+    let b = 20;"])];
        let a = run(BEFORE, &after, Diff::Hunks(&hunks));
        let after_beta = RustParser::new()
            .parse_file(Path::new(PATH), &after)
            .unwrap()
            .symbols
            .into_iter()
            .find(|s| &*s.name == "beta")
            .unwrap();
        assert_eq!(a.touched, vec![("src/lib.rs::beta".to_string(), after_beta.content_hash)]);
    }

    /// A pure deletion inside a function has no `+` lines; the `-` line still
    /// touches the function, which survives.
    #[test]
    fn a_pure_deletion_inside_a_function_touches_it() {
        let before = "fn alpha() {\n    let a = 1;\n    let z = 9;\n}\n";
        let after = "fn alpha() {\n    let a = 1;\n}\n";
        let hunks = [hunk(1, 1, &[" fn alpha() {", "     let a = 1;", "-    let z = 9;", " }"])];
        let a = run(before, after, Diff::Hunks(&hunks));
        assert_eq!(touched_ids(&a), vec!["src/lib.rs::alpha"]);
        assert!(a.removed.is_empty());
    }

    /// Deleting a whole function removes it.
    #[test]
    fn deleting_a_function_removes_it() {
        let after = "fn alpha() {\n    let a = 1;\n}\n\nfn gamma() {\n    let c = 3;\n}\n";
        let hunks = [hunk(4, 4, &[
            " ",
            "-fn beta() {",
            "-    let b = 2;",
            "-}",
            "-",
            " fn gamma() {",
        ])];
        let a = run(BEFORE, after, Diff::Hunks(&hunks));
        assert_eq!(a.removed, vec!["src/lib.rs::beta".to_string()]);
        assert!(a.touched.is_empty(), "{:?}", a.touched);
        assert!(a.outside_symbols, "the removed blank line sat between symbols");
    }

    /// A method edit touches the method, not the impl block containing it.
    #[test]
    fn the_innermost_symbol_is_touched() {
        let before = "struct S;\nimpl S {\n    fn m(&self) {\n        let x = 1;\n    }\n}\n";
        let after = before.replace("let x = 1;", "let x = 2;");
        let hunks = [hunk(4, 4, &["-        let x = 1;", "+        let x = 2;"])];
        let a = run(before, &after, Diff::Hunks(&hunks));
        assert_eq!(touched_ids(&a), vec!["src/lib.rs::S/m"]);
    }

    /// Editing a function's doc comment touches the function: leading docs
    /// are part of its span.
    #[test]
    fn editing_a_doc_comment_touches_its_function() {
        let before = "/// Old docs.\nfn alpha() {}\n";
        let after = "/// New docs.\nfn alpha() {}\n";
        let hunks = [hunk(1, 1, &["-/// Old docs.", "+/// New docs."])];
        let a = run(before, after, Diff::Hunks(&hunks));
        assert_eq!(touched_ids(&a), vec!["src/lib.rs::alpha"]);
    }

    /// A changed line in no symbol is flagged, not silently dropped.
    #[test]
    fn a_change_outside_any_symbol_is_flagged() {
        let before = "use std::fmt;\n\nfn alpha() {}\n";
        let after = "use std::io;\n\nfn alpha() {}\n";
        let hunks = [hunk(1, 1, &["-use std::fmt;", "+use std::io;"])];
        let a = run(before, after, Diff::Hunks(&hunks));
        assert!(a.touched.is_empty() && a.removed.is_empty());
        assert!(a.outside_symbols);
    }

    /// Adding a new function touches it.
    #[test]
    fn an_added_function_is_touched() {
        let before = "fn alpha() {}\n";
        let after = "fn alpha() {}\n\nfn delta() {}\n";
        let hunks = [hunk(1, 1, &[" fn alpha() {}", "+", "+fn delta() {}"])];
        let a = run(before, after, Diff::Hunks(&hunks));
        assert_eq!(touched_ids(&a), vec!["src/lib.rs::delta"]);
        assert!(a.outside_symbols, "the added blank line is in no symbol");
    }

    /// Two hunks attribute independently.
    #[test]
    fn several_hunks_attribute_independently() {
        let after = BEFORE.replace("let a = 1;", "let a = 10;").replace("let c = 3;", "let c = 30;");
        let hunks = [
            hunk(2, 2, &["-    let a = 1;", "+    let a = 10;"]),
            hunk(10, 10, &["-    let c = 3;", "+    let c = 30;"]),
        ];
        let a = run(BEFORE, &after, Diff::Hunks(&hunks));
        assert_eq!(touched_ids(&a), vec!["src/lib.rs::alpha", "src/lib.rs::gamma"]);
    }

    /// A create has no hunks: every innermost symbol is touched.
    #[test]
    fn a_create_touches_every_innermost_symbol() {
        // No `struct S`: it would be a leaf sharing the id `S` with the impl
        // (ids are not unique), which would muddy the assertion below.
        let after = "impl S {\n    fn m(&self) {}\n}\nfn free() {}\n";
        let a = run("", after, Diff::Created);
        let ids = touched_ids(&a);
        assert!(ids.contains(&"src/lib.rs::S/m"), "{ids:?}");
        assert!(ids.contains(&"src/lib.rs::free"), "{ids:?}");
        assert!(!ids.contains(&"src/lib.rs::S"), "the impl has children; only innermost: {ids:?}");
    }

    /// `\ No newline at end of file` occupies no line.
    #[test]
    fn a_no_newline_marker_does_not_shift_lines() {
        let before = "fn alpha() {}\nfn beta() {}";
        let after = "fn alpha() {}\nfn beta() { 1; }";
        let hunks = [hunk(2, 2, &[
            "-fn beta() {}",
            "\\ No newline at end of file",
            "+fn beta() { 1; }",
            "\\ No newline at end of file",
        ])];
        let a = run(before, after, Diff::Hunks(&hunks));
        assert_eq!(touched_ids(&a), vec!["src/lib.rs::beta"]);
    }

    /// Equal inputs give byte-equal output, whatever order hunks touch
    /// symbols in.
    #[test]
    fn output_is_sorted_and_deduplicated() {
        let after = BEFORE.replace("let c = 3;", "let c = 30;").replace("let a = 1;", "let a = 10;");
        let hunks = [
            hunk(10, 10, &["-    let c = 3;", "+    let c = 30;"]),
            hunk(2, 2, &["-    let a = 1;", "+    let a = 10;"]),
        ];
        let a = run(BEFORE, &after, Diff::Hunks(&hunks));
        assert_eq!(touched_ids(&a), vec!["src/lib.rs::alpha", "src/lib.rs::gamma"]);
    }

    /// Hunks must account for every change: one the patch leaves out means
    /// a symbol the record would not name — a false "untouched" (spec §2.3).
    #[test]
    fn hunks_that_leave_a_change_out_refuse_attribution() {
        let after = BEFORE.replace("let c = 3;", "let c = 30;").replace("let a = 1;", "let a = 10;");
        let hunks = [hunk(2, 2, &["-    let a = 1;", "+    let a = 10;"])];
        let refused = attribute(&RustParser::new(), Path::new(PATH), BEFORE, &after, Diff::Hunks(&hunks));
        assert_eq!(refused, None);
    }

    #[test]
    fn a_wrong_context_line_refuses_attribution() {
        let after = BEFORE.replace("let b = 2;", "let b = 20;");
        let hunks = [hunk(5, 5, &[" fn not_beta() {", "-    let b = 2;", "+    let b = 20;"])];
        let refused = attribute(&RustParser::new(), Path::new(PATH), BEFORE, &after, Diff::Hunks(&hunks));
        assert_eq!(refused, None);
    }

    /// A hunk with no old lines inserts after `old_start`, as in unified diff.
    #[test]
    fn a_pure_insertion_into_an_empty_file_attributes() {
        let after = "fn a() {}\n";
        let a = run("", after, Diff::Hunks(&[hunk(0, 1, &["+fn a() {}"])]));
        assert_eq!(touched_ids(&a), vec!["src/lib.rs::a"]);
    }

    /// A create with no symbols still changed something: every line is
    /// outside any symbol.
    #[test]
    fn a_create_with_no_symbols_is_outside_symbols() {
        let a = run("", "// just a comment\n", Diff::Created);
        assert!(a.touched.is_empty());
        assert!(a.outside_symbols);
    }

    /// A hunk that disagrees with the reconstructed text — here, a change the
    /// edit itself did not make, as when Claude Code stamps a memory file's
    /// frontmatter — refuses attribution rather than attributing the wrong
    /// lines. The caller falls back to a file-level write.
    #[test]
    fn a_hunk_that_disagrees_with_the_texts_refuses_attribution() {
        let after = BEFORE.replace("let b = 2;", "let b = 20;");
        // Claims line 2 changed, but the reconstructed `after` did not.
        let hunks = [
            hunk(2, 2, &["-    let a = 1;", "+    let a = 99;"]),
            hunk(6, 6, &["-    let b = 2;", "+    let b = 20;"]),
        ];
        assert_eq!(
            attribute(&RustParser::new(), Path::new(PATH), BEFORE, &after, Diff::Hunks(&hunks)),
            None
        );
    }

    #[test]
    fn apply_edit_replaces_first_or_all_and_refuses_a_missing_match() {
        assert_eq!(apply_edit("a b a", "a", "x", false).as_deref(), Some("x b a"));
        assert_eq!(apply_edit("a b a", "a", "x", true).as_deref(), Some("x b x"));
        assert_eq!(apply_edit("a b a", "zz", "x", false), None);
        assert_eq!(apply_edit("a b a", "", "x", false), None);
    }
}

/// Journal records from ingested writes (spec §2.2, §2.6, §9.6).
#[cfg(test)]
mod record_tests {
    use super::*;
    use crate::ingest::{Hunk, WriteEvent, WriteSource};
    use std::path::PathBuf;
    use std::sync::Arc;

    const ROOT: &str = "/proj";
    const CANARY: &str = "CANARY_7f3c_never_persist";

    fn event(path: &str, source: WriteSource) -> WriteEvent {
        WriteEvent {
            op: Arc::from("toolu_01abc"),
            agent_id: Arc::from("agent-1"),
            tool_name: Arc::from("Edit"),
            path: PathBuf::from(path),
            timestamp: "2026-09-26T10:00:01Z".into(),
            source,
        }
    }

    fn edit(original: Option<&str>, user_modified: bool) -> WriteSource {
        WriteSource::Edit {
            original: original.map(String::from),
            old: format!("let b = 2; // {CANARY}"),
            new: "let b = 20;".into(),
            replace_all: false,
            hunks: Some(vec![Hunk {
                old_start: 2,
                new_start: 2,
                lines: vec![format!("-    let b = 2; // {CANARY}"), "+    let b = 20;".into()],
            }]),
            user_modified,
        }
    }

    const BEFORE: &str = "fn beta() {\n    let b = 2; // CANARY_7f3c_never_persist\n}\n";

    fn build(ev: &WriteEvent) -> Option<WriteRecord> {
        build_record(ev, Path::new(ROOT), &ParserRegistry::new(), true)
    }

    #[test]
    fn an_edit_with_original_file_is_symbol_level() {
        let rec = build(&event("/proj/src/lib.rs", edit(Some(BEFORE), false))).unwrap();
        assert_eq!(rec.level, Level::Symbol);
        assert_eq!(rec.file, "src/lib.rs");
        assert_eq!(rec.syms.len(), 1);
        assert_eq!(rec.syms[0].0, "src/lib.rs::beta");
        assert!(rec.syms[0].1.starts_with("b3:"));
        assert_eq!((rec.op.as_str(), rec.av, rec.a.as_str()), ("toolu_01abc", ATTRIBUTION_VERSION, "agent-1"));
        assert_eq!(rec.fh, None, "only a Write carries a file hash");
    }

    #[test]
    fn missing_original_file_or_user_modified_is_file_level() {
        for source in [edit(None, false), edit(Some(BEFORE), true)] {
            let rec = build(&event("/proj/src/lib.rs", source)).unwrap();
            assert_eq!(rec.level, Level::File);
            assert!(rec.syms.is_empty() && rec.removed.is_empty());
        }
    }

    /// Without a usable patch there is no knowing which lines changed, so no
    /// symbol-level claim — least of all "no symbol changed".
    #[test]
    fn an_edit_without_a_structured_patch_is_file_level() {
        let WriteSource::Edit { original, old, new, replace_all, user_modified, .. } = edit(Some(BEFORE), false) else {
            unreachable!()
        };
        let source = WriteSource::Edit { original, old, new, replace_all, hunks: None, user_modified };
        let rec = build(&event("/proj/src/lib.rs", source)).unwrap();
        assert_eq!(rec.level, Level::File);
        assert!(rec.syms.is_empty() && rec.removed.is_empty());
    }

    /// An empty patch for an edit that changed text leaves a change
    /// unaccounted for.
    #[test]
    fn an_edit_whose_patch_is_empty_is_file_level() {
        let WriteSource::Edit { original, old, new, replace_all, user_modified, .. } = edit(Some(BEFORE), false) else {
            unreachable!()
        };
        let source = WriteSource::Edit { original, old, new, replace_all, hunks: Some(vec![]), user_modified };
        let rec = build(&event("/proj/src/lib.rs", source)).unwrap();
        assert_eq!(rec.level, Level::File);
    }

    /// A decomposed (NFD) file name is recorded in NFC, symbol ids included,
    /// so `touched`, snapshots and linkage — which all normalize — find it.
    #[test]
    fn paths_and_symbol_ids_are_recorded_normalized() {
        let ev = event("/proj/src/cafe\u{301}.rs", edit(Some(BEFORE), false));
        let rec = build(&ev).unwrap();
        assert_eq!(rec.file, "src/caf\u{e9}.rs");
        assert_eq!(rec.syms[0].0, "src/caf\u{e9}.rs::beta");
        assert!(crate::objects::valid_record_path(&rec.file));
    }

    #[test]
    fn serena_mode_is_file_level() {
        let ev = event("/proj/src/lib.rs", edit(Some(BEFORE), false));
        let rec = build_record(&ev, Path::new(ROOT), &ParserRegistry::new(), false).unwrap();
        assert_eq!(rec.level, Level::File);
    }

    #[test]
    fn a_create_is_symbol_level_with_a_file_hash() {
        let ev = event("/proj/src/new.rs", WriteSource::Write {
            original: None,
            content: "fn a() {}\nfn b() {}\n".into(),
            create: true,
            hunks: None,
            user_modified: false,
        });
        let rec = build(&ev).unwrap();
        assert_eq!(rec.level, Level::Symbol);
        let ids: Vec<&str> = rec.syms.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ids, vec!["src/new.rs::a", "src/new.rs::b"]);
        assert_eq!(
            rec.fh.as_deref(),
            Some(crate::journal::encode_hash(blake3::hash(b"fn a() {}\nfn b() {}\n").as_bytes()).as_str())
        );
    }

    #[test]
    fn a_file_with_no_parser_is_file_level() {
        let ev = event("/proj/notes.txt", WriteSource::Write {
            original: None, content: "hello".into(), create: true, hunks: None, user_modified: false,
        });
        let rec = build(&ev).unwrap();
        assert_eq!((rec.level, rec.file.as_str()), (Level::File, "notes.txt"));
        assert!(rec.fh.is_some());
    }

    /// Writes outside the project are neither portable nor the project's.
    #[test]
    fn a_write_outside_the_project_is_dropped() {
        assert!(build(&event("/elsewhere/.aws/config", WriteSource::Opaque)).is_none());
        assert!(build(&event("../escape.rs", WriteSource::Opaque)).is_none());
        assert!(build(&event("/proj", WriteSource::Opaque)).is_none(), "the root itself is not a file");
    }

    /// File contents never reach a record (spec §9.6): not the original, not
    /// the edit strings, not written content.
    #[test]
    fn file_contents_never_reach_the_record() {
        let write = WriteSource::Write {
            original: Some(BEFORE.into()),
            content: format!("fn beta() {{ /* {CANARY} */ }}\n"),
            create: false,
            hunks: None,
            user_modified: false,
        };
        for source in [edit(Some(BEFORE), false), edit(None, false), write] {
            let rec = build(&event("/proj/src/lib.rs", source)).unwrap();
            let json = serde_json::to_string(&rec).unwrap();
            assert!(!json.contains(CANARY), "leaked into {json}");
        }
    }
}
