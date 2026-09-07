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
//! | read at hash H | id is gone, hash H found elsewhere | **restored**, `moved_from` set |
//! | read at hash H | symbol is gone | **removed** — omitted |
//!
//! The third row is why the hash is worth more than the id. An id encodes a
//! location (`<path>::<name-path>`), so hoisting a helper into another module
//! destroys it; the body is unchanged and the agent's knowledge of it is
//! entirely intact. See [`classify`] for the uniqueness rules that keep this
//! from guessing.
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
    /// Hash of the matched node. Unlike an id this is unique to the body, so
    /// it is the selector to hand `ambits show` when an id is ambiguous —
    /// `<path>::Foo` names both `struct Foo` and `impl Foo`, this does not.
    pub content_hash: [u8; 32],
    /// The id this symbol had when it was read, when that differs from where
    /// it lives now — i.e. it was located by content hash rather than by id.
    ///
    /// `None` for the ordinary case. When set, every other field on this
    /// struct describes the symbol's *current* home; this is the stale address
    /// the agent is still holding, and consumers must surface it. An agent
    /// told only the new location has no way to connect it to what it
    /// remembers; one told only the old location will read the wrong file.
    pub moved_from: Option<String>,
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

    /// Restored symbols that were found somewhere other than where they were
    /// read. A subset of `restored` — a moved symbol is recovered knowledge,
    /// not a separate outcome.
    pub fn moved(&self) -> impl Iterator<Item = &RestoredSymbol> {
        self.restored.iter().filter(|s| s.moved_from.is_some())
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

/// Symbol id → (content hash as of the read, depth read at).
///
/// The common currency between the journal, the session-log fallback, and
/// [`classify`]. Deliberately not keyed by agent: who read a symbol does not
/// change whether that read is still accurate.
pub type ReadSet = HashMap<String, ([u8; 32], ReadDepth)>;

/// Where a set of prior reads came from, and therefore how much it can be
/// trusted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestoreSource {
    /// The coverage journal — records the hash as of each read, so drift is
    /// detectable and the result is trustworthy.
    Journal,
    /// Replayed from Claude Code's session logs, which do **not** record what
    /// a file looked like when it was read. Drift is therefore undetectable:
    /// a symbol edited after being read still compares equal, because both
    /// sides of the comparison come from the current tree. Everything is
    /// reported as restored, some of it wrongly.
    ///
    /// Kept because the alternative is returning nothing at all for every
    /// session that predates journaling, or any session where the TUI was
    /// never run. Callers must label it.
    SessionLogs,
}

impl RestoreSource {
    /// Whether drift could actually be detected. `false` means "reads are
    /// listed, but freshness is unverified".
    pub fn verifies_drift(&self) -> bool {
        matches!(self, RestoreSource::Journal)
    }
}

/// A classification plus the provenance needed to interpret it.
#[derive(Debug, Clone)]
pub struct RestoreReport {
    pub outcome: RestoreOutcome,
    pub source: RestoreSource,
    pub session_id: Option<String>,
    /// Non-fatal complaints from reading the journal.
    pub warnings: Vec<String>,
}

/// Read the journal for `session_id`, if one exists and is non-empty.
pub fn load_from_journal(
    project_root: &Path,
    session_id: &str,
) -> Option<(ReadSet, Vec<String>)> {
    let path = project_root
        .join(crate::journal::JOURNAL_SUBDIR)
        .join(format!("{session_id}.ndjson"));
    let contents = crate::journal::read_journal(&path);
    if contents.reads.is_empty() {
        return None;
    }
    Some((contents.reads, contents.warnings))
}

/// Rebuild a ledger by replaying a session's logs.
///
/// The fallback path for sessions with no journal. Compaction is deliberately
/// **not** replayed as a wipe: the point of restoring is to recover what a
/// compaction cost, so honoring it here would return exactly the state the
/// agent already has — nothing.
pub fn replay_session_logs(
    project_root: &Path,
    project_tree: &ProjectTree,
    log_dir: &Path,
    session_id: &str,
    ingester: &dyn crate::ingest::SessionIngester,
) -> crate::tracking::ContextLedger {
    use crate::ingest::SessionEvent;

    let mut ledger = crate::tracking::ContextLedger::new();
    // Required by the shared marking signature; the alignment popup is not
    // involved here, so nothing reads it back.
    let mut depth_cache = crate::tracking::alignment::DepthOrdinalCache::new();

    for log_file in ingester.session_log_files(log_dir, session_id) {
        for event in ingester.parse_log_file_with_root(&log_file, project_root) {
            match event {
                SessionEvent::ToolCall(tc) => crate::app::apply_tool_call(
                    project_tree,
                    project_root,
                    &tc,
                    &mut ledger,
                    &mut depth_cache,
                ),
                // A `/clear` is a voluntary discard — resurrecting that
                // context would mislead, so it is the one boundary we honor.
                SessionEvent::SessionCleared => {
                    ledger = crate::tracking::ContextLedger::new();
                    depth_cache = crate::tracking::alignment::DepthOrdinalCache::new();
                }
                SessionEvent::Compacted { .. } => {}
            }
        }
    }
    ledger
}

/// Recover prior reads from a ledger built by replaying session logs.
///
/// Every hash here is taken from the *current* tree, because that is all the
/// session log can tell us — see [`RestoreSource::SessionLogs`]. The result
/// therefore classifies as fully restored by construction.
pub fn reads_from_ledger(
    ledger: &crate::tracking::ContextLedger,
) -> ReadSet {
    ledger
        .entries
        .values()
        .filter(|e| e.depth.is_seen() && !e.stale)
        .map(|e| {
            (
                e.symbol_id.clone(),
                (e.content_hash_at_read, e.depth),
            )
        })
        .collect()
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
/// Build an id → symbols index over the whole project tree.
///
/// The value is a `Vec` because **symbol ids are not unique**. Ids are
/// `{relative_path}::{name_path}`, so in Rust a `struct Foo` and its
/// `impl Foo` in the same file both produce `path::Foo` — with different
/// content hashes, since they are different spans of source.
///
/// That collision predates this module and affects `ContextLedger` (keyed by
/// id) and the coverage counts too. Here it would be actively harmful:
/// keeping one arbitrary node per id means the one we compare against may not
/// be the one whose hash was journaled, reporting drift on a file nobody
/// touched. Since almost every Rust type has an impl block, that would be the
/// common case rather than an edge case.
pub fn index_tree(tree: &ProjectTree) -> HashMap<&str, Vec<&SymbolNode>> {
    let mut out: HashMap<&str, Vec<&SymbolNode>> = HashMap::new();
    for (_, sym) in tree.walk() {
        out.entry(sym.id.as_str()).or_default().push(sym);
    }
    out
}

/// Index every symbol in the tree by content hash.
///
/// A `Vec` because bodies collide: identical trivial symbols (`mod helpers;`,
/// a unit struct, a two-line helper duplicated across modules) hash the same.
/// Measured on this repo, ~1% of hashes name more than one symbol, which is
/// exactly why the move rescue demands a unique match.
pub fn index_tree_by_hash(tree: &ProjectTree) -> HashMap<[u8; 32], Vec<&SymbolNode>> {
    let mut out: HashMap<[u8; 32], Vec<&SymbolNode>> = HashMap::new();
    for (_, sym) in tree.walk() {
        out.entry(sym.content_hash).or_default().push(sym);
    }
    out
}

/// Partition journaled reads against the current tree.
///
/// `reads` is the last-write-wins fold produced by
/// [`crate::journal::read_journal`].
///
/// ## Following moves
///
/// A symbol whose id has vanished is not necessarily gone: hoisting a helper
/// into a shared module changes its id (the id is `<path>::<name-path>`) while
/// leaving its body byte-identical. A second pass therefore looks the orphans
/// up by content hash and, on a unique match, restores them at their new
/// address with `moved_from` set.
///
/// Renames are deliberately *not* followed. `content_hash` covers the
/// symbol's whole source span including its signature, so renaming changes the
/// hash — and a renamed symbol genuinely is something the agent's memory now
/// mislabels.
///
/// The rescue requires uniqueness on **both** sides: one unclaimed tree node
/// with that hash, and one journal entry claiming it. That is not
/// conservatism for its own sake — `index` is a `HashMap` with randomized
/// iteration order, so a "first claimant wins" rule would rescue different
/// symbols on different runs over identical data.
pub fn classify(
    reads: &ReadSet,
    tree: &ProjectTree,
) -> RestoreOutcome {
    let index = index_tree(tree);
    let mut outcome = RestoreOutcome::default();
    // Orphans, held back for the hash pass below rather than being written
    // off as removed immediately.
    let mut orphans: Vec<(&String, [u8; 32], ReadDepth)> = Vec::new();

    for (id, (hash_at_read, depth)) in reads {
        // Paths come from the symbol id in every branch, not from
        // `SymbolNode.file_path`. The id's prefix *is* the project-relative
        // path, it is the only source available for symbols that no longer
        // exist, and using it uniformly means one rule instead of two.
        let (file_path, name_path) = split_id(id);

        // A read is still good if *any* symbol under this id still hashes to
        // what was recorded — see `index_tree` on why one id can name several
        // symbols.
        let candidates = index.get(id.as_str());
        let matched = candidates
            .and_then(|syms| syms.iter().find(|s| s.content_hash == *hash_at_read));

        match (candidates, matched) {
            (_, Some(sym)) => {
                outcome.restored.push(RestoredSymbol {
                    symbol_id: id.clone(),
                    file_path,
                    name_path,
                    depth: *depth,
                    line_range: sym.line_range.clone(),
                    estimated_tokens: sym.estimated_tokens,
                    content_hash: sym.content_hash,
                    moved_from: None,
                });
            }
            (Some(_), None) => outcome.drifted.push(OmittedSymbol {
                symbol_id: id.clone(),
                file_path,
                name_path,
                reason: OmissionReason::Drifted,
            }),
            (None, _) => orphans.push((id, *hash_at_read, *depth)),
        }
    }

    follow_moves(&mut outcome, orphans, tree);

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

/// Second classification pass: rescue orphans that merely moved.
///
/// Everything here is decided by uniqueness, never by order. Any orphan that
/// is not rescued falls through to `removed`, which is where it would have
/// landed without this pass — so the worst case is the previous behavior.
fn follow_moves(
    outcome: &mut RestoreOutcome,
    orphans: Vec<(&String, [u8; 32], ReadDepth)>,
    tree: &ProjectTree,
) {
    // Nothing to do in the overwhelmingly common case, and building the hash
    // index over the whole tree is not free.
    if orphans.is_empty() {
        return;
    }

    let by_hash = index_tree_by_hash(tree);

    // Nodes already restored under their own id are spoken for. Collected up
    // front so the check below is a lookup rather than a scan per orphan.
    let claimed: std::collections::HashSet<(String, u32)> = outcome
        .restored
        .iter()
        .map(|r| (r.symbol_id.clone(), r.line_range.start))
        .collect();

    // Count how many orphans want each hash, so a body duplicated across two
    // files — one copy since deleted — cannot be rescued to the survivor. We
    // could not tell which of the two the agent read, and they were identical
    // anyway.
    let mut claims: HashMap<[u8; 32], usize> = HashMap::new();
    for (_, hash, depth) in &orphans {
        if *depth == ReadDepth::FullBody {
            *claims.entry(*hash).or_insert(0) += 1;
        }
    }

    for (old_id, hash, depth) in orphans {
        // Only full-body reads follow a move. At shallower depths what the
        // agent retained is essentially the symbol's name and location — and
        // the location is precisely what changed, so there is nothing to
        // carry over. A full read means it knows the body, and the body is
        // provably identical.
        let rescued = (depth == ReadDepth::FullBody && claims.get(&hash) == Some(&1))
            .then(|| by_hash.get(&hash))
            .flatten()
            .filter(|nodes| nodes.len() == 1)
            .map(|nodes| nodes[0])
            // Without this an unrelated orphan sharing a node's body would
            // surface that same node twice in one digest.
            .filter(|node| !claimed.contains(&(node.id.clone(), node.line_range.start)));

        let (file_path, name_path) = split_id(old_id);
        match rescued {
            Some(node) => {
                let (new_file, new_name) = split_id(&node.id);
                outcome.restored.push(RestoredSymbol {
                    symbol_id: node.id.clone(),
                    file_path: new_file,
                    name_path: new_name,
                    depth,
                    line_range: node.line_range.clone(),
                    estimated_tokens: node.estimated_tokens,
                    content_hash: node.content_hash,
                    moved_from: Some(old_id.clone()),
                });
            }
            None => outcome.removed.push(OmittedSymbol {
                symbol_id: old_id.clone(),
                file_path,
                name_path,
                reason: OmissionReason::Removed,
            }),
        }
    }
}

/// Owned `(file_path, name_path)` for a symbol id. A malformed id degrades to
/// an empty path and the whole id as the name, rather than losing the record.
fn split_id(id: &str) -> (PathBuf, String) {
    match split_symbol_id(id) {
        Some((path, name)) => (PathBuf::from(path), name.to_string()),
        None => (PathBuf::new(), id.to_string()),
    }
}

/// What a cold-start rehydrate changed about the ledger.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RehydrateStats {
    /// Entries the session log had already rebuilt, whose recorded hash the
    /// journal replaced with the true hash-at-read.
    pub corrected: usize,
    /// Entries the journal knew about that the session log did not account
    /// for.
    pub inserted: usize,
    /// Entries the post-overlay comparison found no longer match the tree.
    /// Without the overlay these would all have looked fresh.
    pub drifted: usize,
    /// Symbols re-keyed to a new id because they moved. Counted separately
    /// from `corrected`: nothing about the read changed, only its address.
    pub moved: usize,
}

/// Fold the journal into a freshly replayed ledger, then re-derive staleness.
///
/// ## Why this is not redundant with replaying the session log
///
/// Startup already replays the whole session JSONL, which rebuilds essentially
/// the same set of entries. What it cannot rebuild is *drift*: replay stamps
/// each entry with the hash the symbol has right now, so the later comparison
/// is against itself and always passes. `check_staleness` can't help either —
/// it diffs two trees, and at cold start there is no earlier tree to diff
/// against. The upshot is that a freshly launched TUI reports every historical
/// read as current, including reads of files that have since been rewritten.
///
/// The journal is the only record of the hash at read time, so overlaying it
/// and re-comparing is what converts that silent over-report into an accurate
/// one.
///
/// Where the two sources disagree the journal wins, which is the conservative
/// direction: if a symbol was re-read while ambit was not running, the journal
/// holds the older hash, the entry is marked stale, and the agent re-reads
/// something it arguably still knew. The opposite error — declaring a drifted
/// symbol fresh — is the one that actually misleads.
///
/// `fallback_agent` attributes reads from v1 journals, which predate per-agent
/// records and so carry no attribution of their own.
pub fn rehydrate_ledger(
    ledger: &mut crate::tracking::ContextLedger,
    contents: &crate::journal::JournalContents,
    fallback_agent: &str,
    tree: &ProjectTree,
) -> RehydrateStats {
    let mut stats = RehydrateStats::default();

    // Resolve moves through `classify`, so the ledger and the digest agree on
    // where a symbol lives. Doing it any other way would let the TUI and
    // `restore-context` disagree about the same journal.
    //
    // A journaled id that no longer exists is dead weight in the ledger: the
    // tree view renders from `project_tree`, so an entry keyed to a vanished
    // id is invisible, and `refresh_staleness` would mark it stale for good
    // measure. Re-keying puts the coverage back where the code actually is.
    // Owned rather than borrowed: the new id exists only in the tree and the
    // classification, never in the journal we are reading from.
    let remap: HashMap<String, String> = classify(&contents.reads, tree)
        .restored
        .into_iter()
        .filter_map(|s| s.moved_from.map(|old| (old, s.symbol_id)))
        .collect();
    stats.moved = remap.len();

    let resolve = |id: &str| -> String {
        remap.get(id).cloned().unwrap_or_else(|| id.to_string())
    };

    // The hash needs no adjustment when an id is re-keyed: a move is *defined*
    // by the body being byte-identical, which is how it was found. So the
    // journaled hash still matches the node at the new address, and
    // `refresh_staleness` below correctly leaves it fresh.
    for ((symbol_id, agent), (hash, depth)) in &contents.agent_reads {
        if ledger.rehydrate(resolve(symbol_id), *depth, *hash, agent.clone()) {
            stats.inserted += 1;
        } else {
            stats.corrected += 1;
        }
    }

    // v1 journals have no attribution to recover, so their reads are only
    // visible in the symbol-level fold. Applying them under `fallback_agent`
    // keeps the coverage rather than discarding it for want of a label.
    let attributed: std::collections::HashSet<&str> = contents
        .agent_reads
        .keys()
        .map(|(sym, _)| sym.as_str())
        .collect();
    for (symbol_id, (hash, depth)) in &contents.reads {
        if attributed.contains(symbol_id.as_str()) {
            continue;
        }
        if ledger.rehydrate(resolve(symbol_id), *depth, *hash, fallback_agent.to_string()) {
            stats.inserted += 1;
        } else {
            stats.corrected += 1;
        }
    }

    stats.drifted = refresh_staleness(ledger, tree);
    stats
}

/// Re-derive `stale` for every ledger entry by comparing its recorded
/// hash-at-read against the current tree. Returns how many are stale.
///
/// Unlike `tracking::check_staleness` this needs no previous tree — the
/// entry's own `content_hash_at_read` is the earlier side of the comparison.
/// That makes it usable at startup, where no earlier tree exists.
pub fn refresh_staleness(ledger: &mut crate::tracking::ContextLedger, tree: &ProjectTree) -> usize {
    let index = index_tree(tree);

    // One id can name several symbols (`struct Foo` and `impl Foo` collide), so
    // an entry is only stale when *none* of its candidates match — the same
    // rule `classify` applies.
    let stale: Vec<String> = ledger
        .entries
        .iter()
        .filter(|(_, entry)| entry.depth.is_seen())
        .filter(|(id, entry)| {
            !index
                .get(id.as_str())
                .is_some_and(|syms| syms.iter().any(|s| s.content_hash == entry.content_hash_at_read))
        })
        .map(|(id, _)| id.clone())
        .collect();

    for id in &stale {
        ledger.mark_stale(id);
    }
    stale.len()
}

#[cfg(test)]
mod tests {
    use crate::helpers::*;
    use super::*;
    use crate::symbols::merkle::content_hash;

    fn reads(
        entries: &[(&str, [u8; 32], ReadDepth)],
    ) -> ReadSet {
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
        // A body that survives nowhere in the tree — otherwise this is a move,
        // not a deletion, and `follow_moves` would rightly rescue it.
        let out = classify(
            &reads(&[("a.rs::gone", content_hash("deleted body"), ReadDepth::FullBody)]),
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

    /// Symbol ids are not unique — a Rust `struct Foo` and its `impl Foo` in
    /// the same file share the id `path::Foo` with different content hashes.
    /// A read matching either one must restore, or nearly every Rust type
    /// would report spurious drift.
    #[test]
    fn colliding_ids_restore_when_any_candidate_matches() {
        let tree = project(vec![file(
            "a.rs",
            vec![
                sym_hashed("a.rs::Foo", "Foo", "struct Foo { x: u8 }"),
                sym_hashed("a.rs::Foo", "Foo", "impl Foo { fn new() {} }"),
            ],
        )]);

        // Read recorded the impl block's hash; the struct is indexed first.
        let out = classify(
            &reads(&[(
                "a.rs::Foo",
                content_hash("impl Foo { fn new() {} }"),
                ReadDepth::FullBody,
            )]),
            &tree,
        );
        assert_eq!(out.restored.len(), 1, "matching either candidate is enough");
        assert!(out.drifted.is_empty());

        // A hash matching neither is still genuine drift.
        let out = classify(
            &reads(&[("a.rs::Foo", content_hash("something else"), ReadDepth::FullBody)]),
            &tree,
        );
        assert!(out.restored.is_empty());
        assert_eq!(out.drifted.len(), 1);
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

    // -----------------------------------------------------------------------
    // Cold-start rehydrate
    // -----------------------------------------------------------------------

    use crate::journal::JournalContents;
    use crate::tracking::{ContextLedger, Provenance};

    fn journal_of(entries: &[(&str, &str, [u8; 32], ReadDepth)]) -> JournalContents {
        let mut c = JournalContents::default();
        for (symbol, agent, hash, depth) in entries {
            c.reads.insert(symbol.to_string(), (*hash, *depth));
            c.agent_reads
                .insert((symbol.to_string(), agent.to_string()), (*hash, *depth));
        }
        c
    }

    /// The whole reason increment 6 exists. Replaying the session log stamps
    /// every entry with the hash the symbol has *now*, so a symbol rewritten
    /// since it was read still looks fresh. Only the journal knows better.
    #[test]
    fn rehydrate_detects_drift_that_replay_alone_cannot() {
        let tree = project(vec![file("a.rs", vec![sym("a.rs::x", "current")])]);
        let current = tree.files[0].symbols[0].content_hash;
        let at_read = content_hash("what it looked like when read");

        // What startup replay produces: the read is known, but its hash was
        // taken from the tree as it is today.
        let mut ledger = ContextLedger::new();
        ledger.record("a.rs::x".into(), ReadDepth::FullBody, current, "ag".into(), 10);
        assert!(!ledger.is_stale("a.rs::x"), "replay alone sees nothing wrong");

        let contents = journal_of(&[("a.rs::x", "ag", at_read, ReadDepth::FullBody)]);
        let stats = rehydrate_ledger(&mut ledger, &contents, "sess", &tree);

        assert!(ledger.is_stale("a.rs::x"), "the journal exposes the drift");
        assert_eq!(stats.drifted, 1);
        assert_eq!(stats.corrected, 1);
        assert_eq!(stats.inserted, 0);
    }

    /// A symbol that moved must land in the ledger at its *new* id. Keyed to
    /// the dead id it is invisible — the tree view renders from the project
    /// tree, so nothing would show at either address.
    #[test]
    fn rehydrate_re_keys_a_moved_symbol_to_its_current_id() {
        let mut moved = sym_hashed("src/util.rs::helper", "helper", "fn body");
        moved.line_range = 12..18;
        let tree = project(vec![file("src/util.rs", vec![moved])]);
        let hash = content_hash("fn body");

        let mut contents = JournalContents::default();
        contents
            .reads
            .insert("src/old.rs::helper".into(), (hash, ReadDepth::FullBody));
        contents.agent_reads.insert(
            ("src/old.rs::helper".into(), "agent-1".into()),
            (hash, ReadDepth::FullBody),
        );

        let mut ledger = ContextLedger::new();
        let stats = rehydrate_ledger(&mut ledger, &contents, "sess", &tree);

        assert_eq!(stats.moved, 1);
        assert_eq!(
            ledger.depth_of("src/util.rs::helper"),
            ReadDepth::FullBody,
            "coverage lands where the code now lives"
        );
        assert!(
            !ledger.is_stale("src/util.rs::helper"),
            "a move is byte-identical by definition, so nothing drifted"
        );
        assert_eq!(
            ledger.depth_of("src/old.rs::helper"),
            ReadDepth::Unseen,
            "and nothing is stranded at the dead address"
        );
        assert_eq!(
            ledger.depth_of_for_agent("src/util.rs::helper", "agent-1"),
            ReadDepth::FullBody,
            "attribution follows the move"
        );
    }

    /// The unattributed v1 path has to re-key too, or old journals lose moved
    /// symbols that the digest would happily report.
    #[test]
    fn rehydrate_re_keys_unattributed_reads_as_well() {
        let moved = sym_hashed("src/util.rs::helper", "helper", "fn body");
        let tree = project(vec![file("src/util.rs", vec![moved])]);

        let mut contents = JournalContents::default();
        contents.reads.insert(
            "src/old.rs::helper".into(),
            (content_hash("fn body"), ReadDepth::FullBody),
        );

        let mut ledger = ContextLedger::new();
        let stats = rehydrate_ledger(&mut ledger, &contents, "sess-9", &tree);

        assert_eq!(stats.moved, 1);
        assert_eq!(
            ledger.depth_of_for_agent("src/util.rs::helper", "sess-9"),
            ReadDepth::FullBody
        );
    }

    /// The ledger and the digest must not disagree about the same journal.
    #[test]
    fn rehydrate_and_classify_agree_on_where_a_symbol_lives() {
        let moved = sym_hashed("src/util.rs::helper", "helper", "fn body");
        let tree = project(vec![file("src/util.rs", vec![moved])]);
        let hash = content_hash("fn body");

        let mut contents = JournalContents::default();
        contents
            .reads
            .insert("src/old.rs::helper".into(), (hash, ReadDepth::FullBody));

        let mut ledger = ContextLedger::new();
        rehydrate_ledger(&mut ledger, &contents, "sess", &tree);
        let outcome = classify(&contents.reads, &tree);

        assert_eq!(outcome.restored.len(), 1);
        let id = &outcome.restored[0].symbol_id;
        assert!(
            ledger.depth_of(id).is_seen(),
            "the digest points at {id}, so the ledger must have it there too"
        );
    }

    /// Precision: correcting one symbol must not evict its neighbours.
    #[test]
    fn rehydrate_leaves_unchanged_symbols_alone() {
        let tree = project(vec![file(
            "a.rs",
            vec![sym("a.rs::x", "x"), sym("a.rs::y", "y")],
        )]);
        let x_now = tree.files[0].symbols[0].content_hash;
        let y_now = tree.files[0].symbols[1].content_hash;

        let mut ledger = ContextLedger::new();
        ledger.record("a.rs::x".into(), ReadDepth::FullBody, x_now, "ag".into(), 10);
        ledger.record("a.rs::y".into(), ReadDepth::FullBody, y_now, "ag".into(), 10);

        let contents = journal_of(&[
            ("a.rs::x", "ag", content_hash("stale"), ReadDepth::FullBody),
            ("a.rs::y", "ag", y_now, ReadDepth::FullBody),
        ]);
        let stats = rehydrate_ledger(&mut ledger, &contents, "sess", &tree);

        assert!(ledger.is_stale("a.rs::x"));
        assert!(!ledger.is_stale("a.rs::y"), "its neighbour is untouched");
        assert_eq!(stats.drifted, 1);
    }

    /// A journal entry the session log cannot account for — the log was
    /// rotated, or the journal came from another machine.
    #[test]
    fn rehydrate_recovers_reads_the_log_no_longer_has() {
        let tree = project(vec![file("a.rs", vec![sym("a.rs::x", "x")])]);
        let current = tree.files[0].symbols[0].content_hash;

        let mut ledger = ContextLedger::new();
        let contents = journal_of(&[("a.rs::x", "agent-7", current, ReadDepth::Signature)]);
        let stats = rehydrate_ledger(&mut ledger, &contents, "sess", &tree);

        assert_eq!(stats.inserted, 1);
        assert_eq!(ledger.depth_of("a.rs::x"), ReadDepth::Signature);
        assert!(!ledger.is_stale("a.rs::x"), "it still matches the tree");
        assert!(
            ledger.is_restored("a.rs::x"),
            "a journaled read predates this process, so it is not live context"
        );
        assert_eq!(
            ledger.depth_of_for_agent("a.rs::x", "agent-7"),
            ReadDepth::Signature,
            "attribution survives the round trip"
        );
    }

    /// The aggregate depth is a max over `agent_depths`, so a rehydrated entry
    /// with no agent recorded would be silently downgraded by the next real
    /// read. Seeding the agent is what prevents that.
    #[test]
    fn a_later_shallower_read_cannot_downgrade_a_rehydrated_entry() {
        let tree = project(vec![file("a.rs", vec![sym("a.rs::x", "x")])]);
        let current = tree.files[0].symbols[0].content_hash;

        let mut ledger = ContextLedger::new();
        let contents = journal_of(&[("a.rs::x", "old-agent", current, ReadDepth::FullBody)]);
        rehydrate_ledger(&mut ledger, &contents, "sess", &tree);

        ledger.record("a.rs::x".into(), ReadDepth::NameOnly, current, "new-agent".into(), 1);

        assert_eq!(ledger.depth_of("a.rs::x"), ReadDepth::FullBody);
        assert_eq!(ledger.provenance_of("a.rs::x"), Some(Provenance::Live));
    }

    /// v1 journals carry no attribution. Their reads must still land, under
    /// the session id, rather than being dropped for want of a label.
    #[test]
    fn unattributed_reads_fall_back_to_the_session_id() {
        let tree = project(vec![file("a.rs", vec![sym("a.rs::x", "x")])]);
        let current = tree.files[0].symbols[0].content_hash;

        let mut contents = JournalContents::default();
        contents
            .reads
            .insert("a.rs::x".into(), (current, ReadDepth::Overview));

        let mut ledger = ContextLedger::new();
        let stats = rehydrate_ledger(&mut ledger, &contents, "sess-42", &tree);

        assert_eq!(stats.inserted, 1);
        assert_eq!(
            ledger.depth_of_for_agent("a.rs::x", "sess-42"),
            ReadDepth::Overview
        );
    }

    /// Symbol ids are not unique (`struct Foo` and `impl Foo` collide), so a
    /// match against any candidate has to count — otherwise every ambiguous id
    /// is reported as drifted.
    #[test]
    fn refresh_staleness_accepts_any_matching_candidate() {
        let mut struct_node = sym("a.rs::Foo", "struct Foo");
        let impl_node = sym("a.rs::Foo", "impl Foo");
        struct_node.content_hash = content_hash("struct body");
        let tree = project(vec![file("a.rs", vec![struct_node, impl_node.clone()])]);

        let mut ledger = ContextLedger::new();
        ledger.record(
            "a.rs::Foo".into(),
            ReadDepth::FullBody,
            impl_node.content_hash,
            "ag".into(),
            10,
        );

        assert_eq!(refresh_staleness(&mut ledger, &tree), 0);
        assert!(!ledger.is_stale("a.rs::Foo"));
    }

    /// A symbol that no longer exists cannot be verified, so it is stale.
    #[test]
    fn refresh_staleness_marks_vanished_symbols() {
        let tree = project(vec![file("a.rs", vec![sym("a.rs::x", "x")])]);

        let mut ledger = ContextLedger::new();
        ledger.record(
            "a.rs::gone".into(),
            ReadDepth::FullBody,
            content_hash("gone"),
            "ag".into(),
            10,
        );

        assert_eq!(refresh_staleness(&mut ledger, &tree), 1);
        assert!(ledger.is_stale("a.rs::gone"));
    }

    // -----------------------------------------------------------------------
    // Following moves
    // -----------------------------------------------------------------------

    /// The case this exists for: a helper hoisted into another module. Its id
    /// changes because the id embeds the path, but the body is byte-identical.
    #[test]
    fn a_moved_symbol_is_restored_at_its_new_home() {
        let mut moved = sym_hashed("src/util.rs::format_tokens", "format_tokens", "fn body");
        moved.line_range = 12..18;
        let tree = project(vec![
            file("src/util.rs", vec![moved]),
            file("src/stats.rs", vec![sym_hashed("src/stats.rs::other", "other", "unrelated")]),
        ]);

        let out = classify(
            &reads(&[(
                "src/stats.rs::format_tokens",
                content_hash("fn body"),
                ReadDepth::FullBody,
            )]),
            &tree,
        );

        assert!(out.removed.is_empty(), "it was found, not lost");
        assert_eq!(out.restored.len(), 1);
        let r = &out.restored[0];
        assert_eq!(r.symbol_id, "src/util.rs::format_tokens", "the current address");
        assert_eq!(r.file_path, PathBuf::from("src/util.rs"));
        assert_eq!(r.line_range, 12..18, "line numbers describe where it is now");
        assert_eq!(
            r.moved_from.as_deref(),
            Some("src/stats.rs::format_tokens"),
            "the stale address the agent is still holding"
        );
        assert_eq!(out.moved().count(), 1);
    }

    /// ~1% of hashes name more than one symbol — duplicated trivial bodies
    /// like `mod helpers;`. Picking one would invent a location.
    #[test]
    fn an_ambiguous_target_is_not_followed() {
        let tree = project(vec![
            file("a.rs", vec![sym_hashed("a.rs::helpers", "helpers", "same")]),
            file("b.rs", vec![sym_hashed("b.rs::helpers", "helpers", "same")]),
        ]);

        let out = classify(
            &reads(&[("c.rs::helpers", content_hash("same"), ReadDepth::FullBody)]),
            &tree,
        );

        assert!(out.restored.is_empty());
        assert_eq!(out.removed.len(), 1, "ambiguity falls back to removed");
    }

    /// The mirror case: one body duplicated in two files, one copy deleted.
    /// We cannot tell which the agent read, so neither is rescued.
    #[test]
    fn two_orphans_claiming_one_node_are_both_withheld() {
        let tree = project(vec![file("c.rs", vec![sym_hashed("c.rs::dup", "dup", "same")])]);

        let out = classify(
            &reads(&[
                ("a.rs::dup", content_hash("same"), ReadDepth::FullBody),
                ("b.rs::dup", content_hash("same"), ReadDepth::FullBody),
            ]),
            &tree,
        );

        assert!(out.restored.is_empty());
        assert_eq!(out.removed.len(), 2);
    }

    /// A node restored under its own id is spoken for. Without this guard an
    /// unrelated orphan sharing its body would list the same node twice.
    #[test]
    fn a_node_already_restored_is_not_claimed_again() {
        let tree = project(vec![file("a.rs", vec![sym_hashed("a.rs::x", "x", "same")])]);

        let out = classify(
            &reads(&[
                ("a.rs::x", content_hash("same"), ReadDepth::FullBody),
                ("gone.rs::x", content_hash("same"), ReadDepth::FullBody),
            ]),
            &tree,
        );

        assert_eq!(out.restored.len(), 1, "the node appears once");
        assert_eq!(out.restored[0].moved_from, None, "claimed by its own id");
        assert_eq!(out.removed.len(), 1);
    }

    /// At shallower depths what the agent retained is the name and location —
    /// and the location is exactly what changed, so there is nothing to carry.
    #[test]
    fn only_full_body_reads_follow_a_move() {
        let tree = project(vec![file("b.rs", vec![sym_hashed("b.rs::x", "x", "body")])]);

        for depth in [ReadDepth::NameOnly, ReadDepth::Overview, ReadDepth::Signature] {
            let out = classify(&reads(&[("a.rs::x", content_hash("body"), depth)]), &tree);
            assert!(out.restored.is_empty(), "{depth:?} must not follow a move");
            assert_eq!(out.removed.len(), 1);
        }

        let out = classify(
            &reads(&[("a.rs::x", content_hash("body"), ReadDepth::FullBody)]),
            &tree,
        );
        assert_eq!(out.restored.len(), 1, "a full read does");
    }
}
