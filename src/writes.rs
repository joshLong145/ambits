//! Which symbols an agent's write touched.
//!
//! Spec: `docs/spikes/coverage-snapshots.md` §2. This module is the pure core
//! of write attribution — no I/O, no ingestion, no journal. Given the file
//! before and after a write, the write's diff hunks, and the file's parser,
//! it answers which symbols the write changed or removed.
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

use crate::parser::LanguageParser;
use crate::symbols::{SymbolId, SymbolNode};

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

/// Attribute a write of `path` from `before` to `after`.
///
/// `hunks` = `None` means a file creation: there is no diff to walk, and
/// every innermost symbol in `after` is touched (spec §2.2).
///
/// `None` when either side fails to parse, or when a hunk disagrees with the
/// texts — a `+` line that is not the `after` line at its position, or a `-`
/// line that is not the `before` line. The caller then records a file-level
/// write. The check makes attribution self-verifying: `after` is usually
/// reconstructed from `oldString`/`newString`, and anything else that
/// changed the file in the same write (Claude Code stamps memory files'
/// frontmatter, for one) would otherwise be attributed to the wrong lines.
/// Against this project's logs, 80 of 82 reconstructions matched; the two
/// that did not were such stamped memory files.
pub fn attribute(
    parser: &dyn LanguageParser,
    path: &Path,
    before: &str,
    after: &str,
    hunks: Option<&[Hunk]>,
) -> Option<Attribution> {
    let after_syms = parser.parse_file(path, after).ok()?.symbols;

    let Some(hunks) = hunks else {
        let mut touched = BTreeSet::new();
        for_each_leaf(&after_syms, &mut |s| {
            touched.insert((s.id.clone(), s.content_hash));
        });
        return Some(Attribution {
            touched: touched.into_iter().collect(),
            ..Attribution::default()
        });
    };

    let before_syms = parser.parse_file(path, before).ok()?.symbols;
    let before_lines: Vec<&str> = before.split('\n').collect();
    let after_lines: Vec<&str> = after.split('\n').collect();
    // 1-based `line` of `lines` equals `body`.
    let line_is = |lines: &[&str], line: u32, body: &str| {
        (line as usize).checked_sub(1).and_then(|i| lines.get(i)) == Some(&body)
    };

    let mut touched = BTreeSet::new();
    let mut removed = BTreeSet::new();
    let mut outside_symbols = false;

    for hunk in hunks {
        let mut old_line = hunk.old_start;
        let mut new_line = hunk.new_start;
        for line in &hunk.lines {
            match line.as_bytes().first() {
                Some(b'+') => {
                    if !line_is(&after_lines, new_line, &line[1..]) {
                        return None;
                    }
                    match innermost_at(&after_syms, new_line) {
                        Some(s) => {
                            touched.insert((s.id.clone(), s.content_hash));
                        }
                        None => outside_symbols = true,
                    }
                    new_line += 1;
                }
                Some(b'-') => {
                    if !line_is(&before_lines, old_line, &line[1..]) {
                        return None;
                    }
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

/// The deepest symbol whose inclusive line range contains `line`.
fn innermost_at(symbols: &[SymbolNode], line: u32) -> Option<&SymbolNode> {
    let hit = symbols
        .iter()
        .find(|s| s.line_range.start <= line && line <= s.line_range.end)?;
    innermost_at(&hit.children, line).or(Some(hit))
}

/// Every symbol, at any depth, whose id is `id`.
fn with_id<'a>(symbols: &'a [SymbolNode], id: &str) -> Vec<&'a SymbolNode> {
    let mut out = Vec::new();
    let mut stack: Vec<&SymbolNode> = symbols.iter().collect();
    while let Some(s) = stack.pop() {
        if &*s.id == id {
            out.push(s);
        }
        stack.extend(s.children.iter());
    }
    out
}

/// Visit every symbol with no children.
fn for_each_leaf<'a>(symbols: &'a [SymbolNode], f: &mut impl FnMut(&'a SymbolNode)) {
    for s in symbols {
        if s.children.is_empty() {
            f(s);
        } else {
            for_each_leaf(&s.children, f);
        }
    }
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

    fn run(before: &str, after: &str, hunks: Option<&[Hunk]>) -> Attribution {
        attribute(&RustParser::new(), Path::new(PATH), before, after, hunks).expect("parses")
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
        let a = run(BEFORE, &after, Some(&hunks));
        assert_eq!(touched_ids(&a), vec!["src/lib.rs::beta"]);
        assert!(a.removed.is_empty());
        assert!(!a.outside_symbols);
    }

    /// The recorded hash is the symbol's hash *after* the write.
    #[test]
    fn the_touched_hash_is_the_post_write_hash() {
        let after = BEFORE.replace("let b = 2;", "let b = 20;");
        let hunks = [hunk(6, 6, &["-    let b = 2;", "+    let b = 20;"])];
        let a = run(BEFORE, &after, Some(&hunks));
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
        let a = run(before, after, Some(&hunks));
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
        let a = run(BEFORE, after, Some(&hunks));
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
        let a = run(before, &after, Some(&hunks));
        assert_eq!(touched_ids(&a), vec!["src/lib.rs::S/m"]);
    }

    /// Editing a function's doc comment touches the function: leading docs
    /// are part of its span.
    #[test]
    fn editing_a_doc_comment_touches_its_function() {
        let before = "/// Old docs.\nfn alpha() {}\n";
        let after = "/// New docs.\nfn alpha() {}\n";
        let hunks = [hunk(1, 1, &["-/// Old docs.", "+/// New docs."])];
        let a = run(before, after, Some(&hunks));
        assert_eq!(touched_ids(&a), vec!["src/lib.rs::alpha"]);
    }

    /// A changed line in no symbol is flagged, not silently dropped.
    #[test]
    fn a_change_outside_any_symbol_is_flagged() {
        let before = "use std::fmt;\n\nfn alpha() {}\n";
        let after = "use std::io;\n\nfn alpha() {}\n";
        let hunks = [hunk(1, 1, &["-use std::fmt;", "+use std::io;"])];
        let a = run(before, after, Some(&hunks));
        assert!(a.touched.is_empty() && a.removed.is_empty());
        assert!(a.outside_symbols);
    }

    /// Adding a new function touches it.
    #[test]
    fn an_added_function_is_touched() {
        let before = "fn alpha() {}\n";
        let after = "fn alpha() {}\n\nfn delta() {}\n";
        let hunks = [hunk(1, 1, &[" fn alpha() {}", "+", "+fn delta() {}"])];
        let a = run(before, after, Some(&hunks));
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
        let a = run(BEFORE, &after, Some(&hunks));
        assert_eq!(touched_ids(&a), vec!["src/lib.rs::alpha", "src/lib.rs::gamma"]);
    }

    /// A create has no hunks: every innermost symbol is touched.
    #[test]
    fn a_create_touches_every_innermost_symbol() {
        // No `struct S`: it would be a leaf sharing the id `S` with the impl
        // (ids are not unique), which would muddy the assertion below.
        let after = "impl S {\n    fn m(&self) {}\n}\nfn free() {}\n";
        let a = run("", after, None);
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
        let a = run(before, after, Some(&hunks));
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
        let a = run(BEFORE, &after, Some(&hunks));
        assert_eq!(touched_ids(&a), vec!["src/lib.rs::alpha", "src/lib.rs::gamma"]);
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
            attribute(&RustParser::new(), Path::new(PATH), BEFORE, &after, Some(&hunks)),
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
