//! Decide which journaled reads still describe the code as it is now.
//!
//! The journal ([`crate::journal`]) records what each symbol looked like when
//! it was read. Restoring is the act of comparing that against a freshly
//! scanned tree and keeping only the reads that still hold.
//!
//! ## Omission is the point
//!
//! Every journaled read lands in exactly one of three buckets:
//!
//! | Journal says | Tree says | Outcome |
//! |---|---|---|
//! | read at hash H | symbol exists, hash H | **restored** |
//! | read at hash H | symbol exists, hash H' | **drifted** — omitted |
//! | read at hash H | symbol is gone | **removed** — omitted |
//!
//! Only the first bucket is handed onward. Dropping the other two is a
//! feature, not a shortfall: a stale entry is strictly worse than a missing
//! one, because it tells an agent it already knows something it no longer
//! knows. Leaving it out produces the behavior we actually want — the agent
//! re-reads that file the next time it needs it.
//!
//! Depth is carried through but never used to filter or rank. Any depth of
//! read is worth restoring; a symbol seen at `NameOnly` is still a symbol the
//! agent has some purchase on.
//!
//! The omitted buckets are retained rather than discarded so callers can say
//! *what is deliberately not covered* — a digest that silently skips drifted
//! symbols is indistinguishable from one that never saw them.

use std::collections::HashMap;
use std::ops::Range;
use std::path::{Path, PathBuf};

use crate::symbols::{ProjectTree, SymbolNode};
use crate::tracking::ReadDepth;

/// A journaled read that still describes the current code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestoredSymbol {
    pub symbol_id: String,
    /// Project-relative path, so this is meaningful on any machine.
    pub file_path: PathBuf,
    /// Nesting path within the file, e.g. `App/process_compaction`.
    pub name_path: String,
    pub depth: ReadDepth,
    /// 1-based inclusive line range, for pointing an agent at the source.
    pub line_range: Range<u32>,
    /// Cost of re-reading this symbol, if the agent decides to.
    pub estimated_tokens: u32,
}

/// Why a journaled read was withheld.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OmissionReason {
    /// The symbol still exists but its content changed since it was read.
    Drifted,
    /// The symbol is no longer in the tree — renamed, moved, or deleted.
    Removed,
}

/// A journaled read that no longer holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OmittedSymbol {
    pub symbol_id: String,
    /// Recovered from the symbol id, so it is available even for symbols that
    /// no longer exist in the tree.
    pub file_path: PathBuf,
    pub name_path: String,
    pub reason: OmissionReason,
}

/// The result of checking a journal against the current tree.
#[derive(Debug, Clone, Default)]
pub struct RestoreOutcome {
    /// Sorted by file, then by position in the file, so output is stable.
    pub restored: Vec<RestoredSymbol>,
    /// Symbols whose content changed. Sorted by file, then name.
    pub drifted: Vec<OmittedSymbol>,
    /// Symbols that vanished from the tree. Sorted by file, then name.
    pub removed: Vec<OmittedSymbol>,
}

impl RestoreOutcome {
    /// Total journaled reads considered.
    pub fn total(&self) -> usize {
        self.restored.len() + self.drifted.len() + self.removed.len()
    }

    /// Restored symbols grouped by file, preserving the sorted order.
    /// This is the shape a per-file digest wants.
    pub fn restored_by_file(&self) -> Vec<(&Path, Vec<&RestoredSymbol>)> {
        let mut out: Vec<(&Path, Vec<&RestoredSymbol>)> = Vec::new();
        for sym in &self.restored {
            match out.last_mut() {
                Some((path, group)) if *path == sym.file_path.as_path() => group.push(sym),
                _ => out.push((sym.file_path.as_path(), vec![sym])),
            }
        }
        out
    }

    /// Distinct files containing at least one omitted symbol, deduplicated and
    /// sorted. Useful for telling an agent which files to treat as unknown.
    pub fn omitted_files(&self) -> Vec<&Path> {
        let mut paths: Vec<&Path> = self
            .drifted
            .iter()
            .chain(self.removed.iter())
            .map(|s| s.file_path.as_path())
            .collect();
        paths.sort_unstable();
        paths.dedup();
        paths
    }
}

/// Split `"<relative-path>::<Name/Path>"` into its two halves.
///
/// Symbol ids are built as `{path_prefix}::{name_path}` by every parser, with
/// `/` separating nesting levels inside the name path — so the first `::` is
/// unambiguously the boundary.
pub fn split_symbol_id(id: &str) -> Option<(&str, &str)> {
    id.split_once("::")
}

/// Recursively index every symbol in the tree by id, including nested children.
fn index_symbols<'a>(symbols: &'a [SymbolNode], out: &mut HashMap<&'a str, &'a SymbolNode>) {
    for sym in symbols {
        out.insert(sym.id.as_str(), sym);
        index_symbols(&sym.children, out);
    }
}

/// Build an id → symbol index over the whole project tree.
pub fn index_tree(tree: &ProjectTree) -> HashMap<&str, &SymbolNode> {
    let mut out = HashMap::new();
    for file in &tree.files {
        index_symbols(&file.symbols, &mut out);
    }
    out
}

/// Partition journaled reads against the current tree.
///
/// `reads` is the last-write-wins fold produced by
/// [`crate::journal::read_journal`].
pub fn classify(
    reads: &HashMap<String, ([u8; 32], ReadDepth)>,
    tree: &ProjectTree,
) -> RestoreOutcome {
    let index = index_tree(tree);
    let mut outcome = RestoreOutcome::default();

    for (id, (hash_at_read, depth)) in reads {
        // Paths come from the symbol id in every branch, not from
        // `SymbolNode.file_path`. The id's prefix *is* the project-relative
        // path, it is the only source available for symbols that no longer
        // exist, and using it uniformly means one rule instead of two.
        let (file_path, name_path) = split_id(id);

        match index.get(id.as_str()) {
            Some(sym) if sym.content_hash == *hash_at_read => {
                outcome.restored.push(RestoredSymbol {
                    symbol_id: id.clone(),
                    file_path,
                    name_path,
                    depth: *depth,
                    line_range: sym.line_range.clone(),
                    estimated_tokens: sym.estimated_tokens,
                });
            }
            Some(_) => outcome.drifted.push(OmittedSymbol {
                symbol_id: id.clone(),
                file_path,
                name_path,
                reason: OmissionReason::Drifted,
            }),
            None => outcome.removed.push(OmittedSymbol {
                symbol_id: id.clone(),
                file_path,
                name_path,
                reason: OmissionReason::Removed,
            }),
        }
    }

    // Deterministic ordering: reads come from a HashMap, whose iteration order
    // is deliberately randomized between runs.
    outcome
        .restored
        .sort_by(|a, b| {
            a.file_path
                .cmp(&b.file_path)
                .then_with(|| a.line_range.start.cmp(&b.line_range.start))
                .then_with(|| a.name_path.cmp(&b.name_path))
        });
    for bucket in [&mut outcome.drifted, &mut outcome.removed] {
        bucket.sort_by(|a, b| {
            a.file_path
                .cmp(&b.file_path)
                .then_with(|| a.name_path.cmp(&b.name_path))
        });
    }

    outcome
}

/// Owned `(file_path, name_path)` for a symbol id. A malformed id degrades to
/// an empty path and the whole id as the name, rather than losing the record.
fn split_id(id: &str) -> (PathBuf, String) {
    match split_symbol_id(id) {
        Some((path, name)) => (PathBuf::from(path), name.to_string()),
        None => (PathBuf::new(), id.to_string()),
    }
}

#[cfg(test)]
#[path = "../tests/helpers/mod.rs"]
#[allow(dead_code)]
mod helpers;

#[cfg(test)]
mod tests {
    use super::helpers::*;
    use super::*;
    use crate::symbols::merkle::content_hash;

    fn reads(
        entries: &[(&str, [u8; 32], ReadDepth)],
    ) -> HashMap<String, ([u8; 32], ReadDepth)> {
        entries
            .iter()
            .map(|(id, h, d)| (id.to_string(), (*h, *d)))
            .collect()
    }

    /// Give a symbol a known content hash so drift can be simulated.
    fn sym_hashed(id: &str, name: &str, source: &str) -> crate::symbols::SymbolNode {
        let mut s = sym(id, name);
        s.content_hash = content_hash(source);
        s
    }

    #[test]
    fn matching_hash_restores() {
        let tree = project(vec![file("a.rs", vec![sym_hashed("a.rs::x", "x", "body")])]);
        let out = classify(
            &reads(&[("a.rs::x", content_hash("body"), ReadDepth::FullBody)]),
            &tree,
        );
        assert_eq!(out.restored.len(), 1);
        assert_eq!(out.restored[0].name_path, "x");
        assert_eq!(out.restored[0].depth, ReadDepth::FullBody);
        assert!(out.drifted.is_empty() && out.removed.is_empty());
    }

    #[test]
    fn changed_hash_is_withheld_as_drifted() {
        let tree = project(vec![file("a.rs", vec![sym_hashed("a.rs::x", "x", "new body")])]);
        let out = classify(
            &reads(&[("a.rs::x", content_hash("old body"), ReadDepth::FullBody)]),
            &tree,
        );
        assert!(out.restored.is_empty(), "a drifted read must not be restored");
        assert_eq!(out.drifted.len(), 1);
        assert_eq!(out.drifted[0].reason, OmissionReason::Drifted);
        assert_eq!(out.drifted[0].file_path, PathBuf::from("a.rs"));
    }

    #[test]
    fn vanished_symbol_is_withheld_as_removed() {
        let tree = project(vec![file("a.rs", vec![sym_hashed("a.rs::x", "x", "body")])]);
        let out = classify(
            &reads(&[("a.rs::gone", content_hash("body"), ReadDepth::FullBody)]),
            &tree,
        );
        assert_eq!(out.removed.len(), 1);
        assert_eq!(out.removed[0].reason, OmissionReason::Removed);
        // The file path is recovered from the id even though the tree can't
        // supply it.
        assert_eq!(out.removed[0].file_path, PathBuf::from("a.rs"));
        assert_eq!(out.removed[0].name_path, "gone");
    }

    /// Editing one function must not evict the other symbols in its file —
    /// this per-symbol precision is the whole reason for tracking symbols
    /// rather than files.
    #[test]
    fn drift_is_per_symbol_not_per_file() {
        let tree = project(vec![file(
            "a.rs",
            vec![
                sym_hashed("a.rs::kept", "kept", "unchanged"),
                sym_hashed("a.rs::edited", "edited", "v2"),
            ],
        )]);
        let out = classify(
            &reads(&[
                ("a.rs::kept", content_hash("unchanged"), ReadDepth::FullBody),
                ("a.rs::edited", content_hash("v1"), ReadDepth::FullBody),
            ]),
            &tree,
        );
        assert_eq!(out.restored.len(), 1);
        assert_eq!(out.restored[0].name_path, "kept");
        assert_eq!(out.drifted.len(), 1);
        assert_eq!(out.drifted[0].name_path, "edited");
    }

    #[test]
    fn nested_children_are_indexed() {
        let child = sym_hashed("a.rs::Outer/inner", "inner", "child body");
        let mut parent = sym_with_children("a.rs::Outer", "Outer", vec![child]);
        parent.content_hash = content_hash("parent body");
        let tree = project(vec![file("a.rs", vec![parent])]);

        let out = classify(
            &reads(&[(
                "a.rs::Outer/inner",
                content_hash("child body"),
                ReadDepth::Signature,
            )]),
            &tree,
        );
        assert_eq!(out.restored.len(), 1, "nested symbols must be reachable");
        assert_eq!(out.restored[0].name_path, "Outer/inner");
    }

    #[test]
    fn any_depth_is_restored() {
        let tree = project(vec![
            file("a.rs", vec![sym_hashed("a.rs::x", "x", "b1")]),
            file("b.rs", vec![sym_hashed("b.rs::y", "y", "b2")]),
        ]);
        let out = classify(
            &reads(&[
                ("a.rs::x", content_hash("b1"), ReadDepth::NameOnly),
                ("b.rs::y", content_hash("b2"), ReadDepth::FullBody),
            ]),
            &tree,
        );
        assert_eq!(out.restored.len(), 2, "shallow reads are still worth restoring");
    }

    #[test]
    fn output_is_deterministically_ordered() {
        let tree = project(vec![
            file("z.rs", vec![sym_hashed("z.rs::a", "a", "1")]),
            file("a.rs", vec![sym_hashed("a.rs::b", "b", "2")]),
        ]);
        let r = reads(&[
            ("z.rs::a", content_hash("1"), ReadDepth::FullBody),
            ("a.rs::b", content_hash("2"), ReadDepth::FullBody),
        ]);
        // HashMap iteration order is randomized per process, so a stable
        // result has to come from the sort, not from luck.
        let first = classify(&r, &tree);
        let second = classify(&r, &tree);
        assert_eq!(first.restored, second.restored);
        assert_eq!(
            first.restored.iter().map(|s| s.file_path.as_path()).collect::<Vec<_>>(),
            vec![Path::new("a.rs"), Path::new("z.rs")]
        );
    }

    #[test]
    fn restored_by_file_groups_without_reordering() {
        let tree = project(vec![
            file(
                "a.rs",
                vec![sym_hashed("a.rs::x", "x", "1"), sym_hashed("a.rs::y", "y", "2")],
            ),
            file("b.rs", vec![sym_hashed("b.rs::z", "z", "3")]),
        ]);
        let out = classify(
            &reads(&[
                ("a.rs::x", content_hash("1"), ReadDepth::FullBody),
                ("a.rs::y", content_hash("2"), ReadDepth::FullBody),
                ("b.rs::z", content_hash("3"), ReadDepth::FullBody),
            ]),
            &tree,
        );
        let grouped = out.restored_by_file();
        assert_eq!(grouped.len(), 2);
        assert_eq!(grouped[0].0, Path::new("a.rs"));
        assert_eq!(grouped[0].1.len(), 2);
        assert_eq!(grouped[1].0, Path::new("b.rs"));
    }

    #[test]
    fn omitted_files_dedupes_across_both_buckets() {
        let tree = project(vec![file("a.rs", vec![sym_hashed("a.rs::x", "x", "new")])]);
        let out = classify(
            &reads(&[
                ("a.rs::x", content_hash("old"), ReadDepth::FullBody),
                ("a.rs::gone", content_hash("old"), ReadDepth::FullBody),
            ]),
            &tree,
        );
        assert_eq!(out.drifted.len(), 1);
        assert_eq!(out.removed.len(), 1);
        assert_eq!(out.omitted_files(), vec![Path::new("a.rs")], "one file, not two");
        assert_eq!(out.total(), 2);
    }

    #[test]
    fn empty_journal_yields_empty_outcome() {
        let tree = project(vec![file("a.rs", vec![sym_hashed("a.rs::x", "x", "b")])]);
        let out = classify(&HashMap::new(), &tree);
        assert_eq!(out.total(), 0);
        assert!(out.omitted_files().is_empty());
    }

    #[test]
    fn split_symbol_id_handles_both_shapes() {
        assert_eq!(split_symbol_id("src/a.rs::App/run"), Some(("src/a.rs", "App/run")));
        assert_eq!(split_symbol_id("no-separator"), None);
        // A malformed id degrades to using the whole thing as the name rather
        // than losing the record.
        assert_eq!(
            split_id("no-separator"),
            (PathBuf::new(), "no-separator".to_string())
        );
    }

    /// Nested symbols must report the file they live in. Paths are derived
    /// from the id rather than `SymbolNode.file_path`, so this holds however
    /// deep the nesting goes.
    #[test]
    fn nested_symbols_report_their_file() {
        let child = sym_hashed("src/a.rs::Outer/inner", "inner", "child");
        let mut parent = sym_with_children("src/a.rs::Outer", "Outer", vec![child]);
        parent.content_hash = content_hash("parent");
        let tree = project(vec![file("src/a.rs", vec![parent])]);

        let out = classify(
            &reads(&[(
                "src/a.rs::Outer/inner",
                content_hash("child"),
                ReadDepth::FullBody,
            )]),
            &tree,
        );
        assert_eq!(out.restored[0].file_path, PathBuf::from("src/a.rs"));
        assert_eq!(out.restored[0].name_path, "Outer/inner");
    }
}
