//! Sub-agent alignment scoring: pairwise comparison of how similarly two
//! sibling agents have explored the codebase.
//!
//! ## Depth-cache design
//!
//! Per the feature spec, this module maintains its own standalone
//! precomputed cache — [`DepthOrdinalCache`], effectively a
//! `HashMap<(SymbolId, AgentId), u8>` (ordinal `0=Unseen .. 4=FullBody`,
//! `5=Stale`) — rather than reading depths back out of
//! [`crate::tracking::ContextLedger`] at popup-render time.
//!
//! The cache is populated at the same call sites that invoke
//! [`ContextLedger::record`] during log ingestion: `mark_file_symbols` and
//! `mark_targeted_symbols` in `src/app.rs` each also call
//! `App::depth_cache.record(..)` alongside `ledger.record(..)`, so the two
//! stay in lockstep as events stream in. Alignment computation
//! (`pair_alignment` / `compute_group_alignment`) then reads exclusively
//! from `DepthOrdinalCache`, giving O(1) lookups with no dependency on
//! `ContextLedger`'s internals and no re-derivation work when the popup
//! opens.
//!
//! What *is* computed fresh on each `d` keypress (not cached across frames)
//! is the pairwise file classification below — that's cheap (bounded by
//! project size × sibling-pair count) and only runs when the popup is
//! opened, not on every render.

use std::collections::HashMap;

use crate::symbols::{ProjectTree, SymbolNode};
use crate::tracking::ReadDepth;

/// Ordinal encoding of [`ReadDepth`] used for alignment comparisons.
/// `Unseen=0 .. FullBody=4`, with `Stale=5` sorted above `FullBody` so it
/// never spuriously compares equal to a non-stale depth.
fn depth_ordinal(depth: ReadDepth) -> u8 {
    match depth {
        ReadDepth::Unseen => 0,
        ReadDepth::NameOnly => 1,
        ReadDepth::Overview => 2,
        ReadDepth::Signature => 3,
        ReadDepth::FullBody => 4,
        ReadDepth::Stale => 5,
    }
}

/// Standalone precomputed cache of `(symbol_id, agent_id) -> depth ordinal`.
///
/// This is the actual FR2 cache: populated incrementally at ingestion time
/// (see module docs), not derived from `ContextLedger` on demand. Depths are
/// upgrade-only per `(symbol, agent)` pair — mirroring
/// `ContextLedger::record`'s own upgrade-only semantics for per-agent
/// depth — except `Stale`, which always overwrites (matching
/// `ContextLedger`'s "Stale overrides everything" rule).
#[derive(Debug, Clone, Default)]
pub struct DepthOrdinalCache {
    map: HashMap<(String, String), u8>,
}

impl DepthOrdinalCache {
    pub fn new() -> Self {
        Self { map: HashMap::new() }
    }

    /// Record that `agent_id` read `symbol_id` at `depth`. Only upgrades the
    /// cached ordinal (never downgrades), except `Stale` which always wins.
    pub fn record(&mut self, symbol_id: &str, agent_id: &str, depth: ReadDepth) {
        let ordinal = depth_ordinal(depth);
        let key = (symbol_id.to_string(), agent_id.to_string());
        let entry = self.map.entry(key).or_insert(0);
        if depth == ReadDepth::Stale || ordinal > *entry {
            *entry = ordinal;
        }
    }

    /// Look up the cached depth ordinal for `(symbol_id, agent_id)`,
    /// defaulting to `0` (Unseen) when absent.
    pub fn get(&self, symbol_id: &str, agent_id: &str) -> u8 {
        self.map
            .get(&(symbol_id.to_string(), agent_id.to_string()))
            .copied()
            .unwrap_or(0)
    }
}

/// Per-file classification of how two agents' coverage of that file relate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileAlignment {
    /// One agent touched (depth > Unseen for some symbol in the file) the
    /// file; the other did not touch it at all.
    Unshared,
    /// Both agents touched the file, but at least one symbol's depth ordinal
    /// differs between them.
    DepthMismatch,
    /// Both agents touched the file and every symbol's depth ordinal is
    /// identical between them.
    Aligned,
}

/// Per-file detail backing a [`PairAlignment`]'s file list, so the popup can
/// show exactly which files drifted (and how badly) rather than just totals.
#[derive(Debug, Clone)]
pub struct FileAlignmentEntry {
    pub path: String,
    pub status: FileAlignment,
    /// Only meaningful for `DepthMismatch`: the fraction of symbols *touched
    /// by either agent* in this file whose depth ordinal matches between
    /// them (`matched / touched_by_either`). `None` for `Aligned` (trivially
    /// `1.0` — the status already says everything) and `Unshared` (the
    /// notion of "matching depth" doesn't apply when one agent never looked
    /// at the file at all).
    ///
    /// Denominator choice: "touched by either agent" rather than "all
    /// symbols in the file" — a file's untouched symbols (by both agents)
    /// carry no signal about *their* alignment and would dilute the score
    /// with irrelevant noise, especially in large files where only a small
    /// corner was actually read by either agent.
    pub matched_fraction: Option<f64>,
}

/// Pairwise alignment result for a single sibling pair.
#[derive(Debug, Clone)]
pub struct PairAlignment {
    pub agent_a: String,
    pub agent_b: String,
    pub unshared_count: usize,
    pub mismatch_count: usize,
    pub aligned_count: usize,
    /// `aligned_count / (unshared_count + mismatch_count + aligned_count)`.
    /// When the union of touched files is empty (neither agent touched
    /// anything), the score defaults to `0.0` — there is no evidence of
    /// shared, aligned work between the two agents, so we don't credit them
    /// with alignment they haven't demonstrated.
    pub score: f64,
    /// Per-file breakdown, sorted worst-to-best so the files needing
    /// attention surface first without scrolling: `DepthMismatch` files
    /// (ascending `matched_fraction`, worst drift first), then `Unshared`
    /// files, then `Aligned` files last. `DepthMismatch` is ranked ahead of
    /// `Unshared` because it represents *active disagreement* on work both
    /// agents actually engaged with — typically the more actionable signal
    /// — whereas `Unshared` is simply one agent not having looked yet.
    /// Ties within a group break alphabetically by path for stable output.
    pub files: Vec<FileAlignmentEntry>,
}

/// Recursively collect every symbol id in a symbol subtree (including
/// nested children) into `out`.
fn collect_symbol_ids<'a>(symbols: &'a [SymbolNode], out: &mut Vec<&'a str>) {
    for sym in symbols {
        out.push(&sym.id);
        collect_symbol_ids(&sym.children, out);
    }
}

/// Result of analyzing a single file's alignment between two agents.
struct FileAnalysis {
    status: FileAlignment,
    matched_fraction: Option<f64>,
}

/// Analyze a single file's alignment between two agents. Returns `None`
/// when neither agent touched the file (excluded from the union entirely).
fn analyze_file(
    symbols: &[SymbolNode],
    cache: &DepthOrdinalCache,
    agent_a: &str,
    agent_b: &str,
) -> Option<FileAnalysis> {
    let mut ids = Vec::new();
    collect_symbol_ids(symbols, &mut ids);

    let mut a_touched = false;
    let mut b_touched = false;
    let mut touched_union = 0usize;
    let mut matched = 0usize;
    let mut mismatch = false;

    for id in ids {
        let da = cache.get(id, agent_a);
        let db = cache.get(id, agent_b);
        if da > 0 {
            a_touched = true;
        }
        if db > 0 {
            b_touched = true;
        }
        if da > 0 || db > 0 {
            touched_union += 1;
            if da == db {
                matched += 1;
            } else {
                mismatch = true;
            }
        }
    }

    match (a_touched, b_touched) {
        (false, false) => None,
        (true, false) | (false, true) => Some(FileAnalysis {
            status: FileAlignment::Unshared,
            matched_fraction: None,
        }),
        (true, true) => Some(if mismatch {
            let fraction = if touched_union == 0 {
                0.0
            } else {
                matched as f64 / touched_union as f64
            };
            FileAnalysis {
                status: FileAlignment::DepthMismatch,
                matched_fraction: Some(fraction),
            }
        } else {
            FileAnalysis {
                status: FileAlignment::Aligned,
                matched_fraction: None,
            }
        }),
    }
}

/// Sort rank for worst-to-best file ordering (lower sorts first).
/// See [`PairAlignment::files`] doc for the rationale.
fn file_rank(status: FileAlignment) -> u8 {
    match status {
        FileAlignment::DepthMismatch => 0,
        FileAlignment::Unshared => 1,
        FileAlignment::Aligned => 2,
    }
}

/// Compute the pairwise alignment for a single pair of agents across the
/// whole project tree.
pub fn pair_alignment(
    project_tree: &ProjectTree,
    cache: &DepthOrdinalCache,
    agent_a: &str,
    agent_b: &str,
) -> PairAlignment {
    let mut unshared_count = 0usize;
    let mut mismatch_count = 0usize;
    let mut aligned_count = 0usize;
    let mut files: Vec<FileAlignmentEntry> = Vec::new();

    for file in &project_tree.files {
        if let Some(analysis) = analyze_file(&file.symbols, cache, agent_a, agent_b) {
            match analysis.status {
                FileAlignment::Unshared => unshared_count += 1,
                FileAlignment::DepthMismatch => mismatch_count += 1,
                FileAlignment::Aligned => aligned_count += 1,
            }
            files.push(FileAlignmentEntry {
                path: file.file_path.to_string_lossy().to_string(),
                status: analysis.status,
                matched_fraction: analysis.matched_fraction,
            });
        }
    }

    files.sort_by(|a, b| {
        file_rank(a.status)
            .cmp(&file_rank(b.status))
            .then_with(|| {
                let fa = a.matched_fraction.unwrap_or(0.0);
                let fb = b.matched_fraction.unwrap_or(0.0);
                fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.path.cmp(&b.path))
    });

    let union = unshared_count + mismatch_count + aligned_count;
    let score = if union == 0 {
        0.0
    } else {
        aligned_count as f64 / union as f64
    };

    PairAlignment {
        agent_a: agent_a.to_string(),
        agent_b: agent_b.to_string(),
        unshared_count,
        mismatch_count,
        aligned_count,
        score,
        files,
    }
}

/// Compute alignment for every unordered pair (`N choose 2`) within a
/// sibling group, in stable order (all pairs for `siblings[0]`, then
/// `siblings[1]`, etc., mirroring the input order).
pub fn compute_group_alignment(
    project_tree: &ProjectTree,
    cache: &DepthOrdinalCache,
    sibling_ids: &[String],
) -> Vec<PairAlignment> {
    let mut pairs = Vec::new();
    for i in 0..sibling_ids.len() {
        for j in (i + 1)..sibling_ids.len() {
            pairs.push(pair_alignment(
                project_tree,
                cache,
                &sibling_ids[i],
                &sibling_ids[j],
            ));
        }
    }
    pairs
}

#[cfg(test)]
#[path = "../../tests/helpers/mod.rs"]
#[allow(dead_code)]
mod helpers;

#[cfg(test)]
mod tests {
    use super::*;
    use super::helpers::*;

    #[test]
    fn empty_union_defaults_to_score_zero() {
        let cache = DepthOrdinalCache::new();
        let tree = project(vec![file("a.rs", vec![sym("a1", "a1")])]);
        let result = pair_alignment(&tree, &cache, "agent_a", "agent_b");
        assert_eq!(result.unshared_count, 0);
        assert_eq!(result.mismatch_count, 0);
        assert_eq!(result.aligned_count, 0);
        assert!((result.score - 0.0).abs() < 0.0001);
    }

    #[test]
    fn unshared_file_when_only_one_agent_touched_it() {
        let mut cache = DepthOrdinalCache::new();
        let tree = project(vec![file("a.rs", vec![sym("a1", "a1")])]);
        cache.record("a1", "agent_a", ReadDepth::FullBody);

        let result = pair_alignment(&tree, &cache, "agent_a", "agent_b");
        assert_eq!(result.unshared_count, 1);
        assert_eq!(result.mismatch_count, 0);
        assert_eq!(result.aligned_count, 0);
        assert!((result.score - 0.0).abs() < 0.0001);
    }

    #[test]
    fn depth_mismatch_when_both_touched_but_differ() {
        let mut cache = DepthOrdinalCache::new();
        let tree = project(vec![file("a.rs", vec![sym("a1", "a1")])]);
        cache.record("a1", "agent_a", ReadDepth::FullBody);
        cache.record("a1", "agent_b", ReadDepth::Overview);

        let result = pair_alignment(&tree, &cache, "agent_a", "agent_b");
        assert_eq!(result.unshared_count, 0);
        assert_eq!(result.mismatch_count, 1);
        assert_eq!(result.aligned_count, 0);
        assert!((result.score - 0.0).abs() < 0.0001);
    }

    #[test]
    fn aligned_when_both_touched_with_identical_depths() {
        let mut cache = DepthOrdinalCache::new();
        let tree = project(vec![file("a.rs", vec![sym("a1", "a1")])]);
        cache.record("a1", "agent_a", ReadDepth::FullBody);
        cache.record("a1", "agent_b", ReadDepth::FullBody);

        let result = pair_alignment(&tree, &cache, "agent_a", "agent_b");
        assert_eq!(result.unshared_count, 0);
        assert_eq!(result.mismatch_count, 0);
        assert_eq!(result.aligned_count, 1);
        assert!((result.score - 1.0).abs() < 0.0001);
    }

    #[test]
    fn mixed_files_produce_partial_score() {
        let mut cache = DepthOrdinalCache::new();
        let tree = project(vec![
            file("aligned.rs", vec![sym("s1", "s1")]),
            file("mismatch.rs", vec![sym("s2", "s2")]),
            file("unshared.rs", vec![sym("s3", "s3")]),
            file("untouched.rs", vec![sym("s4", "s4")]),
        ]);
        // aligned.rs: both FullBody.
        cache.record("s1", "agent_a", ReadDepth::FullBody);
        cache.record("s1", "agent_b", ReadDepth::FullBody);
        // mismatch.rs: differing depths.
        cache.record("s2", "agent_a", ReadDepth::FullBody);
        cache.record("s2", "agent_b", ReadDepth::NameOnly);
        // unshared.rs: only agent_a touched.
        cache.record("s3", "agent_a", ReadDepth::Overview);
        // untouched.rs: neither touched -> excluded from union.

        let result = pair_alignment(&tree, &cache, "agent_a", "agent_b");
        assert_eq!(result.aligned_count, 1);
        assert_eq!(result.mismatch_count, 1);
        assert_eq!(result.unshared_count, 1);
        // union = 3, aligned = 1
        assert!((result.score - (1.0 / 3.0)).abs() < 0.0001);

        // untouched.rs is excluded from `files` entirely (union-only).
        assert_eq!(result.files.len(), 3);
        // Worst-to-best: DepthMismatch, then Unshared, then Aligned.
        assert_eq!(result.files[0].path, "mismatch.rs");
        assert_eq!(result.files[0].status, FileAlignment::DepthMismatch);
        assert_eq!(result.files[1].path, "unshared.rs");
        assert_eq!(result.files[1].status, FileAlignment::Unshared);
        assert_eq!(result.files[2].path, "aligned.rs");
        assert_eq!(result.files[2].status, FileAlignment::Aligned);
    }

    #[test]
    fn partial_symbol_mismatch_produces_sensible_matched_fraction() {
        let mut cache = DepthOrdinalCache::new();
        // 4 symbols in one file: 3 match, 1 differs, all touched by at least
        // one agent -> matched_fraction = 3/4.
        let tree = project(vec![file(
            "a.rs",
            vec![
                sym("s1", "s1"),
                sym("s2", "s2"),
                sym("s3", "s3"),
                sym("s4", "s4"),
            ],
        )]);
        cache.record("s1", "agent_a", ReadDepth::FullBody);
        cache.record("s1", "agent_b", ReadDepth::FullBody);
        cache.record("s2", "agent_a", ReadDepth::Overview);
        cache.record("s2", "agent_b", ReadDepth::Overview);
        cache.record("s3", "agent_a", ReadDepth::Signature);
        cache.record("s3", "agent_b", ReadDepth::Signature);
        // s4 differs -> mismatch.
        cache.record("s4", "agent_a", ReadDepth::FullBody);
        cache.record("s4", "agent_b", ReadDepth::NameOnly);

        let result = pair_alignment(&tree, &cache, "agent_a", "agent_b");
        assert_eq!(result.files.len(), 1);
        let entry = &result.files[0];
        assert_eq!(entry.status, FileAlignment::DepthMismatch);
        assert_eq!(entry.matched_fraction, Some(0.75));
    }

    #[test]
    fn matched_fraction_excludes_symbols_untouched_by_either_agent() {
        let mut cache = DepthOrdinalCache::new();
        // 3 symbols: 1 matched, 1 mismatched, 1 untouched by both. The
        // untouched symbol must not dilute the denominator.
        let tree = project(vec![file(
            "a.rs",
            vec![sym("s1", "s1"), sym("s2", "s2"), sym("s3", "s3")],
        )]);
        cache.record("s1", "agent_a", ReadDepth::FullBody);
        cache.record("s1", "agent_b", ReadDepth::FullBody);
        cache.record("s2", "agent_a", ReadDepth::FullBody);
        cache.record("s2", "agent_b", ReadDepth::Overview);
        // s3: neither agent touched it.

        let result = pair_alignment(&tree, &cache, "agent_a", "agent_b");
        let entry = &result.files[0];
        assert_eq!(entry.status, FileAlignment::DepthMismatch);
        // touched_union = {s1, s2} = 2, matched = {s1} = 1 -> 0.5, not 1/3.
        assert_eq!(entry.matched_fraction, Some(0.5));
    }

    #[test]
    fn files_sort_worst_to_best_with_mismatch_severity_and_path_tiebreak() {
        let mut cache = DepthOrdinalCache::new();
        let tree = project(vec![
            file("z_aligned.rs", vec![sym("a1", "a1")]),
            file("b_aligned.rs", vec![sym("a2", "a2")]),
            file("mild_mismatch.rs", vec![sym("m1", "m1"), sym("m2", "m2")]),
            file("severe_mismatch.rs", vec![sym("s1", "s1"), sym("s2", "s2")]),
            file("unshared.rs", vec![sym("u1", "u1")]),
        ]);
        // Two aligned files (order should tiebreak alphabetically).
        cache.record("a1", "agent_a", ReadDepth::FullBody);
        cache.record("a1", "agent_b", ReadDepth::FullBody);
        cache.record("a2", "agent_a", ReadDepth::FullBody);
        cache.record("a2", "agent_b", ReadDepth::FullBody);
        // mild_mismatch.rs: 1/2 symbols match (matched_fraction = 0.5).
        cache.record("m1", "agent_a", ReadDepth::FullBody);
        cache.record("m1", "agent_b", ReadDepth::FullBody);
        cache.record("m2", "agent_a", ReadDepth::FullBody);
        cache.record("m2", "agent_b", ReadDepth::NameOnly);
        // severe_mismatch.rs: 0/2 symbols match (matched_fraction = 0.0).
        cache.record("s1", "agent_a", ReadDepth::FullBody);
        cache.record("s1", "agent_b", ReadDepth::NameOnly);
        cache.record("s2", "agent_a", ReadDepth::Signature);
        cache.record("s2", "agent_b", ReadDepth::Overview);
        // unshared.rs: only agent_a touched.
        cache.record("u1", "agent_a", ReadDepth::FullBody);

        let result = pair_alignment(&tree, &cache, "agent_a", "agent_b");
        let paths: Vec<&str> = result.files.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(
            paths,
            vec![
                "severe_mismatch.rs", // DepthMismatch, worst (0.0) first
                "mild_mismatch.rs",   // DepthMismatch, better (0.5) second
                "unshared.rs",        // Unshared next
                "b_aligned.rs",       // Aligned, alphabetical tiebreak
                "z_aligned.rs",
            ]
        );
    }

    #[test]
    fn nested_children_are_considered() {
        let mut cache = DepthOrdinalCache::new();
        let child = sym("c1", "child");
        let parent = sym_with_children("p1", "parent", vec![child]);
        let tree = project(vec![file("a.rs", vec![parent])]);

        cache.record("p1", "agent_a", ReadDepth::FullBody);
        cache.record("p1", "agent_b", ReadDepth::FullBody);
        // Child differs -> mismatch, even though the parent matches.
        cache.record("c1", "agent_a", ReadDepth::FullBody);
        cache.record("c1", "agent_b", ReadDepth::Overview);

        let result = pair_alignment(&tree, &cache, "agent_a", "agent_b");
        assert_eq!(result.mismatch_count, 1);
        assert_eq!(result.aligned_count, 0);
    }

    #[test]
    fn compute_group_alignment_produces_all_pairs() {
        let cache = DepthOrdinalCache::new();
        let tree = project(vec![file("a.rs", vec![sym("a1", "a1")])]);
        let siblings = vec![
            "agent_a".to_string(),
            "agent_b".to_string(),
            "agent_c".to_string(),
        ];
        let pairs = compute_group_alignment(&tree, &cache, &siblings);
        // 3 choose 2 = 3 pairs.
        assert_eq!(pairs.len(), 3);
        let pair_names: Vec<(String, String)> = pairs
            .iter()
            .map(|p| (p.agent_a.clone(), p.agent_b.clone()))
            .collect();
        assert!(pair_names.contains(&("agent_a".to_string(), "agent_b".to_string())));
        assert!(pair_names.contains(&("agent_a".to_string(), "agent_c".to_string())));
        assert!(pair_names.contains(&("agent_b".to_string(), "agent_c".to_string())));
    }

    #[test]
    fn cache_record_upgrades_but_never_downgrades() {
        let mut cache = DepthOrdinalCache::new();
        cache.record("s1", "agent_a", ReadDepth::FullBody);
        cache.record("s1", "agent_a", ReadDepth::Overview);
        assert_eq!(cache.get("s1", "agent_a"), depth_ordinal(ReadDepth::FullBody));
    }

    #[test]
    fn cache_stale_always_overrides() {
        let mut cache = DepthOrdinalCache::new();
        cache.record("s1", "agent_a", ReadDepth::FullBody);
        cache.record("s1", "agent_a", ReadDepth::Stale);
        assert_eq!(cache.get("s1", "agent_a"), depth_ordinal(ReadDepth::Stale));
    }
}
