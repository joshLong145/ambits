pub mod agents;
pub mod alignment;

use std::collections::HashMap;
use std::time::Instant;

use crate::symbols::{SymbolId, SymbolNode};

/// How much of a symbol an agent has actually read.
///
/// This is a strict total order (`Unseen < NameOnly < .. < FullBody`) and is
/// only ever *upgraded* by [`ContextLedger::record`]. Staleness is
/// deliberately **not** a variant here: "the content changed since we read it"
/// is orthogonal to "how much of it we read", and modelling it as the maximum
/// depth made it absorbing — a stale symbol could never recover, because a
/// later `FullBody` read failed the `depth > current` upgrade test. See
/// [`ContextEntry::stale`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ReadDepth {
    Unseen,
    NameOnly,
    Overview,
    Signature,
    FullBody,
}

impl ReadDepth {
    /// Whether this depth indicates the symbol has been seen at all.
    pub fn is_seen(&self) -> bool {
        !matches!(self, ReadDepth::Unseen)
    }
}

impl std::fmt::Display for ReadDepth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadDepth::Unseen => write!(f, "unseen"),
            ReadDepth::NameOnly => write!(f, "name"),
            ReadDepth::Overview => write!(f, "overview"),
            ReadDepth::Signature => write!(f, "signature"),
            ReadDepth::FullBody => write!(f, "full"),
        }
    }
}

/// Where a ledger entry's knowledge came from.
///
/// Compaction does not destroy the fact that a symbol was read — it destroys
/// the model's *retention* of it. Marking those entries `Restored` keeps the
/// coverage visible without claiming it as freshly read, which is what the old
/// wipe-on-compaction behavior was guarding against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Provenance {
    /// Read during the current context window.
    Live,
    /// Read before a compaction, or rehydrated from the coverage journal.
    Restored,
}

/// One agent's read of one symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgentRead {
    pub depth: ReadDepth,
    /// The symbol's content hash when this agent last read it. Each agent
    /// keeps its own: agent B having read an earlier version says nothing
    /// about whether B has seen the one agent A just read.
    pub hash_at_read: [u8; 32],
    /// Whether the symbol has changed since this agent read it.
    pub stale: bool,
}

#[derive(Debug, Clone)]
pub struct ContextEntry {
    pub symbol_id: SymbolId,
    /// Aggregate depth: the deepest read of the current content, by any
    /// agent; when every agent's read is stale, the deepest read at all, so
    /// a stale entry still says how much was read before it drifted.
    pub depth: ReadDepth,
    /// Whether this read survives in the model's context, or predates a
    /// compaction. Reset to [`Provenance::Live`] by any subsequent read.
    pub provenance: Provenance,
    /// Whether every agent's read of this symbol predates its current
    /// content.
    ///
    /// Orthogonal to `depth` — a stale symbol retains the depth it was read
    /// at, so we can still say *how much* was read before it drifted. Derived
    /// from the agents' own flags (see [`AgentRead::stale`]), which
    /// [`ContextLedger::mark_stale_if_changed`] and
    /// [`ContextLedger::mark_stale_unless`] set and a read clears.
    pub stale: bool,
    /// The content hash the current reads describe: that of the deepest
    /// agent whose read is not stale, or, when all are, of the latest read.
    pub content_hash_at_read: [u8; 32],
    pub timestamp: Instant,
    /// The last agent that touched this symbol.
    pub agent_id: String,
    pub token_count: usize,
    /// Each agent's own read of this symbol.
    pub agent_reads: HashMap<String, AgentRead>,
}

impl ContextEntry {
    fn new(symbol_id: SymbolId, provenance: Provenance, hash: [u8; 32], agent_id: String) -> Self {
        Self {
            symbol_id,
            depth: ReadDepth::Unseen,
            provenance,
            stale: false,
            content_hash_at_read: hash,
            timestamp: Instant::now(),
            agent_id,
            token_count: 0,
            agent_reads: HashMap::new(),
        }
    }

    /// Re-derive the aggregate fields from the agents' reads.
    fn settle(&mut self) {
        let fresh = self
            .agent_reads
            .values()
            .filter(|r| !r.stale && r.depth.is_seen())
            // Ties on depth happen only when an id names several symbols;
            // the hash breaks them so the result never depends on map order.
            .max_by_key(|r| (r.depth, r.hash_at_read));
        match fresh {
            Some(read) => {
                self.depth = read.depth;
                self.content_hash_at_read = read.hash_at_read;
                self.stale = false;
            }
            None => {
                self.depth = self
                    .agent_reads
                    .values()
                    .map(|r| r.depth)
                    .max()
                    .unwrap_or(ReadDepth::Unseen);
                self.stale = self.depth.is_seen();
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ContextLedger {
    pub entries: HashMap<SymbolId, ContextEntry>,
}

impl ContextLedger {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Record that a symbol was seen at the given depth by a specific agent.
    ///
    /// Each agent's read is tracked independently — agent A can have
    /// `FullBody` while agent B has `Overview` for the same symbol. Re-reading
    /// the same content only ever deepens an agent's read; reading content
    /// that changed since — the agent's read is stale — replaces it, because
    /// what the agent saw of the old version says nothing about the new one.
    /// The journal's fold applies the same rule.
    ///
    /// Only the reader's own staleness is settled: another agent's is the
    /// tree's to decide ([`Self::mark_stale_if_changed`]), not a hash this
    /// read happened to see — one id can name several symbols.
    ///
    /// Provenance (`stale`, `timestamp`, `agent_id`, `token_count`) is
    /// refreshed on *every* read, not just on a depth upgrade: a re-read at an
    /// unchanged depth still proves the symbol was seen in its current form.
    pub fn record(
        &mut self,
        symbol_id: SymbolId,
        depth: ReadDepth,
        content_hash: [u8; 32],
        agent_id: String,
        token_count: usize,
    ) {
        // Recording nothing seen must not re-establish provenance below: that
        // would clear `stale` and adopt the current hash for a symbol nobody
        // looked at (spec §2.7).
        if !depth.is_seen() {
            return;
        }
        let entry = self.entries.entry(symbol_id.clone()).or_insert_with(|| {
            ContextEntry::new(
                symbol_id.clone(),
                Provenance::Live,
                content_hash,
                String::new(),
            )
        });

        let read = entry
            .agent_reads
            .entry(agent_id.clone())
            .or_insert(AgentRead {
                depth: ReadDepth::Unseen,
                hash_at_read: content_hash,
                stale: false,
            });
        let from = read.depth;
        // Changed content is what made the agent's read stale. A different
        // hash on a read that is not stale is another symbol with the same
        // id (two `impl Foo` in one file), whose content says nothing about
        // whether this one changed.
        if read.stale && read.hash_at_read != content_hash {
            read.depth = depth;
        } else {
            read.depth = read.depth.max(depth);
        }
        read.hash_at_read = content_hash;
        read.stale = false;
        log::debug!(
            target: "ambits::symbol_update",
            symbol_id = symbol_id.as_str(),
            agent_id = agent_id.as_str(),
            from:? = from,
            to:? = read.depth;
            "recorded"
        );
        entry.settle();

        // A read always re-establishes provenance: we have just seen this
        // symbol at its current content, so it is by definition not merely
        // restored.
        entry.provenance = Provenance::Live;
        entry.timestamp = Instant::now();
        entry.agent_id = agent_id;
        entry.token_count = token_count;
    }

    /// Get the read depth for a symbol, defaulting to Unseen.
    pub fn depth_of(&self, symbol_id: &str) -> ReadDepth {
        self.entries
            .get(symbol_id)
            .map(|e| e.depth)
            .unwrap_or(ReadDepth::Unseen)
    }

    /// Flag each agent's read of a symbol stale whose hash no longer matches
    /// `current_hash`, and fresh again when it does.
    ///
    /// Depth is left untouched — we still know how much each agent read, we
    /// just know whether it describes the current content.
    pub fn mark_stale_if_changed(&mut self, symbol_id: &str, current_hash: [u8; 32]) {
        self.mark_stale_unless(symbol_id, |hash| *hash == current_hash);
    }

    /// Flag each agent's read of a symbol stale unless `current` accepts the
    /// hash it was read at, and fresh again when it does.
    ///
    /// For callers that check against several candidates at once — one id can
    /// name several symbols, so a read is current when any of them matches —
    /// or, with a predicate that accepts nothing, for an entry no symbol in
    /// the tree can verify.
    pub fn mark_stale_unless(&mut self, symbol_id: &str, current: impl Fn(&[u8; 32]) -> bool) {
        if let Some(entry) = self.entries.get_mut(symbol_id) {
            for read in entry.agent_reads.values_mut() {
                read.stale = !current(&read.hash_at_read);
            }
            entry.settle();
        }
    }

    /// Flag a seen entry as stale outright.
    ///
    /// For callers that have already decided the entry cannot be verified —
    /// because no symbol with that id survives in the tree.
    pub fn mark_stale(&mut self, symbol_id: &str) {
        self.mark_stale_unless(symbol_id, |_| false);
    }

    /// Install a read recovered from the durable journal.
    ///
    /// Cold-start only: this must run before any live ingestion, because it
    /// treats the journal as authoritative over whatever is already in the
    /// entry.
    ///
    /// The point of the overwrite is `hash_at_read`. Replaying a session log
    /// can only stamp the hash a symbol has *now*, which makes the later drift
    /// comparison vacuous — every restored symbol looks fresh. The journal
    /// holds the hash the symbol actually had when `agent_id` read it, so
    /// installing it is what makes drift detectable at all.
    ///
    /// Depth is merged upgrade-only per agent, so a rehydrated entry can never
    /// lose depth to a later read. Staleness is deliberately *not* decided
    /// here — the caller runs one comparison pass against the tree afterwards.
    ///
    /// Returns `true` when this created a new entry, i.e. the journal knew
    /// about a read the session log could not account for.
    pub fn rehydrate(
        &mut self,
        symbol_id: SymbolId,
        depth: ReadDepth,
        hash_at_read: [u8; 32],
        agent_id: String,
    ) -> bool {
        let inserted = !self.entries.contains_key(&symbol_id);
        let entry = self.entries.entry(symbol_id.clone()).or_insert_with(|| {
            // A journaled read is by definition from before this process
            // started, so no live context holds it.
            ContextEntry::new(
                symbol_id,
                Provenance::Restored,
                hash_at_read,
                agent_id.clone(),
            )
        });

        let read = entry.agent_reads.entry(agent_id).or_insert(AgentRead {
            depth: ReadDepth::Unseen,
            hash_at_read,
            stale: false,
        });
        read.depth = read.depth.max(depth);
        read.hash_at_read = hash_at_read;
        entry.settle();

        inserted
    }

    /// Whether the symbol has been read and has since drifted. `false` for
    /// untracked symbols.
    pub fn is_stale(&self, symbol_id: &str) -> bool {
        self.entries
            .get(symbol_id)
            .map(|e| e.stale)
            .unwrap_or(false)
    }

    /// Provenance of a tracked symbol, or `None` if it is untracked.
    pub fn provenance_of(&self, symbol_id: &str) -> Option<Provenance> {
        self.entries.get(symbol_id).map(|e| e.provenance)
    }

    /// Whether the symbol's read predates a compaction. `false` for untracked
    /// symbols.
    pub fn is_restored(&self, symbol_id: &str) -> bool {
        self.entries
            .get(symbol_id)
            .map(|e| e.provenance == Provenance::Restored)
            .unwrap_or(false)
    }

    /// Demote every entry to [`Provenance::Restored`].
    ///
    /// Called at a compaction boundary: the reads still happened and the
    /// content is still current, but the model has lost the context, so the
    /// coverage should no longer be presented as live.
    pub fn mark_all_restored(&mut self) {
        for entry in self.entries.values_mut() {
            entry.provenance = Provenance::Restored;
        }
    }

    /// Count of seen entries whose reads predate a compaction.
    pub fn total_restored(&self) -> usize {
        self.entries
            .values()
            .filter(|e| e.provenance == Provenance::Restored && e.depth.is_seen())
            .count()
    }

    /// Count of seen entries currently flagged stale.
    pub fn total_stale(&self) -> usize {
        self.entries
            .values()
            .filter(|e| e.stale && e.depth.is_seen())
            .count()
    }

    pub fn total_seen(&self) -> usize {
        self.entries.values().filter(|e| e.depth.is_seen()).count()
    }

    pub fn count_by_depth(&self) -> HashMap<ReadDepth, usize> {
        let mut counts = HashMap::new();
        for entry in self.entries.values() {
            *counts.entry(entry.depth).or_insert(0) += 1;
        }
        counts
    }

    /// Returns the read depth of a specific symbol for a specific agent.
    /// Returns `ReadDepth::Unseen` if the symbol or agent is not tracked.
    pub fn depth_of_for_agent(&self, symbol_id: &str, agent_id: &str) -> ReadDepth {
        self.entries
            .get(symbol_id)
            .and_then(|e| e.agent_reads.get(agent_id))
            .map_or(ReadDepth::Unseen, |r| r.depth)
    }

    /// Whether `agent_id` read the symbol and it has changed since. `false`
    /// when that agent never read it.
    pub fn is_stale_for_agent(&self, symbol_id: &str, agent_id: &str) -> bool {
        self.entries
            .get(symbol_id)
            .and_then(|e| e.agent_reads.get(agent_id))
            .is_some_and(|r| r.stale && r.depth.is_seen())
    }

    /// Returns the list of agent IDs that have interacted with a given symbol.
    pub fn agents_for_symbol(&self, symbol_id: &str) -> Vec<&str> {
        self.entries
            .get(symbol_id)
            .map(|e| e.agent_reads.keys().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// Count entries by depth for a specific agent.
    pub fn count_by_depth_for_agent(&self, agent_id: &str) -> HashMap<ReadDepth, usize> {
        let mut counts = HashMap::new();
        for entry in self.entries.values() {
            let depth = entry
                .agent_reads
                .get(agent_id)
                .map_or(ReadDepth::Unseen, |r| r.depth);
            *counts.entry(depth).or_insert(0) += 1;
        }
        counts
    }

    /// Count total seen symbols for a specific agent.
    pub fn total_seen_for_agent(&self, agent_id: &str) -> usize {
        self.entries
            .values()
            .filter(|e| {
                e.agent_reads
                    .get(agent_id)
                    .is_some_and(|r| r.depth.is_seen())
            })
            .count()
    }
}

/// Collect all symbol IDs and their content hashes into `map` (recursive).
pub fn collect_symbol_hashes(symbols: &[SymbolNode], map: &mut HashMap<String, [u8; 32]>) {
    for sym in symbols {
        map.insert(sym.id.clone(), sym.content_hash);
        collect_symbol_hashes(&sym.children, map);
    }
}

/// Mark symbols whose content hash changed as stale in `ledger` (recursive).
pub fn check_staleness(
    symbols: &[SymbolNode],
    old_map: &HashMap<String, [u8; 32]>,
    ledger: &mut ContextLedger,
) {
    for sym in symbols {
        if let Some(old_hash) = old_map.get(&sym.id) {
            if *old_hash != sym.content_hash {
                ledger.mark_stale_if_changed(&sym.id, sym.content_hash);
            }
        }
        check_staleness(&sym.children, old_map, ledger);
    }
}

/// Compare old and new symbol trees; mark changed symbols as stale in `ledger`.
pub fn mark_stale_symbols(
    old_symbols: &[SymbolNode],
    new_symbols: &[SymbolNode],
    ledger: &mut ContextLedger,
) {
    let mut old_map = HashMap::new();
    collect_symbol_hashes(old_symbols, &mut old_map);
    check_staleness(new_symbols, &old_map, ledger);
}

impl Default for ContextLedger {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(s: &str) -> [u8; 32] {
        crate::symbols::merkle::content_hash(s)
    }

    #[test]
    fn record_upgrades_depth() {
        let mut ledger = ContextLedger::new();
        ledger.record("s1".into(), ReadDepth::NameOnly, hash("a"), "ag".into(), 10);
        ledger.record("s1".into(), ReadDepth::FullBody, hash("a"), "ag".into(), 10);
        assert_eq!(ledger.depth_of("s1"), ReadDepth::FullBody);
    }

    #[test]
    fn record_never_downgrades() {
        let mut ledger = ContextLedger::new();
        ledger.record("s1".into(), ReadDepth::FullBody, hash("a"), "ag".into(), 10);
        ledger.record("s1".into(), ReadDepth::NameOnly, hash("a"), "ag".into(), 10);
        assert_eq!(ledger.depth_of("s1"), ReadDepth::FullBody);
    }

    #[test]
    fn mark_stale_if_changed() {
        let mut ledger = ContextLedger::new();
        let h1 = hash("v1");
        let h2 = hash("v2");
        ledger.record("s1".into(), ReadDepth::FullBody, h1, "ag".into(), 10);

        // Same hash — no change.
        ledger.mark_stale_if_changed("s1", h1);
        assert!(!ledger.is_stale("s1"));
        assert_eq!(ledger.depth_of("s1"), ReadDepth::FullBody);

        // Different hash — flagged stale, but the depth is retained: we still
        // know how much was read, just not that it's current.
        ledger.mark_stale_if_changed("s1", h2);
        assert!(ledger.is_stale("s1"));
        assert_eq!(ledger.depth_of("s1"), ReadDepth::FullBody);
    }

    /// Regression: staleness used to be the maximum `ReadDepth` variant, which
    /// made it absorbing — `FullBody > Stale` is false, so a genuine re-read
    /// could never clear it and the symbol stayed stale forever.
    #[test]
    fn re_read_after_drift_clears_stale() {
        let mut ledger = ContextLedger::new();
        let h1 = hash("v1");
        let h2 = hash("v2");

        ledger.record("s1".into(), ReadDepth::FullBody, h1, "ag".into(), 10);
        ledger.mark_stale_if_changed("s1", h2);
        assert!(ledger.is_stale("s1"));

        // The agent re-reads the changed symbol.
        ledger.record("s1".into(), ReadDepth::FullBody, h2, "ag".into(), 10);
        assert!(!ledger.is_stale("s1"));
        assert_eq!(ledger.depth_of("s1"), ReadDepth::FullBody);
        assert_eq!(ledger.depth_of_for_agent("s1", "ag"), ReadDepth::FullBody);
    }

    /// Regression: `content_hash_at_read` used to refresh only on a strict
    /// depth increase, so re-reading at an unchanged depth left the old hash
    /// behind and the entry looked drifted when it wasn't.
    #[test]
    fn same_depth_re_read_refreshes_content_hash() {
        let mut ledger = ContextLedger::new();
        let h1 = hash("v1");
        let h2 = hash("v2");

        ledger.record("s1".into(), ReadDepth::Overview, h1, "ag".into(), 10);
        // File edited and re-read at the *same* depth.
        ledger.record("s1".into(), ReadDepth::Overview, h2, "ag".into(), 10);

        assert_eq!(ledger.entries["s1"].content_hash_at_read, h2);
        // Comparing against the current content must now agree.
        ledger.mark_stale_if_changed("s1", h2);
        assert!(!ledger.is_stale("s1"));
    }

    /// What an agent read of one version says nothing about the next: a
    /// re-read of changed content replaces the agent's depth, even a
    /// shallower one.
    #[test]
    fn a_re_read_of_changed_content_replaces_the_depth() {
        let mut ledger = ContextLedger::new();
        let h1 = hash("v1");
        let h2 = hash("v2");

        ledger.record("s1".into(), ReadDepth::FullBody, h1, "ag".into(), 10);
        ledger.mark_stale_if_changed("s1", h2);
        ledger.record("s1".into(), ReadDepth::NameOnly, h2, "ag".into(), 10);

        assert_eq!(ledger.depth_of("s1"), ReadDepth::NameOnly);
        assert_eq!(ledger.entries["s1"].content_hash_at_read, h2);
    }

    #[test]
    fn unseen_not_marked_stale() {
        let mut ledger = ContextLedger::new();
        ledger.mark_stale_if_changed("never_seen", hash("x"));
        assert_eq!(ledger.depth_of("never_seen"), ReadDepth::Unseen);
    }

    #[test]
    fn depth_of_defaults_unseen() {
        let ledger = ContextLedger::new();
        assert_eq!(ledger.depth_of("nonexistent"), ReadDepth::Unseen);
    }

    #[test]
    fn per_agent_depth_tracking() {
        let mut ledger = ContextLedger::new();
        let h = hash("v1");

        // Agent A reads at Overview, Agent B reads at FullBody.
        ledger.record("s1".into(), ReadDepth::Overview, h, "agent_a".into(), 5);
        ledger.record("s1".into(), ReadDepth::FullBody, h, "agent_b".into(), 10);

        // Per-agent depths are independent.
        assert_eq!(
            ledger.depth_of_for_agent("s1", "agent_a"),
            ReadDepth::Overview
        );
        assert_eq!(
            ledger.depth_of_for_agent("s1", "agent_b"),
            ReadDepth::FullBody
        );

        // Aggregate is max across agents.
        assert_eq!(ledger.depth_of("s1"), ReadDepth::FullBody);
    }

    #[test]
    fn per_agent_depth_never_downgrades() {
        let mut ledger = ContextLedger::new();
        let h = hash("v1");

        ledger.record("s1".into(), ReadDepth::FullBody, h, "agent_a".into(), 10);
        ledger.record("s1".into(), ReadDepth::Overview, h, "agent_a".into(), 5);

        // Agent depth should not downgrade.
        assert_eq!(
            ledger.depth_of_for_agent("s1", "agent_a"),
            ReadDepth::FullBody
        );
    }

    #[test]
    fn depth_of_for_agent_defaults_unseen() {
        let ledger = ContextLedger::new();
        assert_eq!(
            ledger.depth_of_for_agent("nonexistent", "agent_a"),
            ReadDepth::Unseen
        );

        // Symbol exists but agent doesn't.
        let mut ledger = ContextLedger::new();
        ledger.record(
            "s1".into(),
            ReadDepth::Overview,
            hash("v1"),
            "agent_a".into(),
            5,
        );
        assert_eq!(
            ledger.depth_of_for_agent("s1", "agent_b"),
            ReadDepth::Unseen
        );
    }

    #[test]
    fn agents_for_symbol_returns_all_agents() {
        let mut ledger = ContextLedger::new();
        let h = hash("v1");

        ledger.record("s1".into(), ReadDepth::Overview, h, "agent_a".into(), 5);
        ledger.record("s1".into(), ReadDepth::FullBody, h, "agent_b".into(), 10);

        let mut agents = ledger.agents_for_symbol("s1");
        agents.sort();
        assert_eq!(agents, vec!["agent_a", "agent_b"]);
    }

    #[test]
    fn agents_for_symbol_empty_when_not_tracked() {
        let ledger = ContextLedger::new();
        assert!(ledger.agents_for_symbol("nonexistent").is_empty());
    }

    /// When the content changes, every agent that read it is stale, and each
    /// keeps its own depth.
    #[test]
    fn a_change_makes_every_agent_stale_and_preserves_their_depths() {
        let mut ledger = ContextLedger::new();
        let h1 = hash("v1");
        let h2 = hash("v2");

        ledger.record("s1".into(), ReadDepth::FullBody, h1, "agent_a".into(), 10);
        ledger.record("s1".into(), ReadDepth::Overview, h1, "agent_b".into(), 5);

        ledger.mark_stale_if_changed("s1", h2);

        assert!(ledger.is_stale("s1"));
        assert!(ledger.is_stale_for_agent("s1", "agent_a"));
        assert!(ledger.is_stale_for_agent("s1", "agent_b"));
        assert_eq!(ledger.total_stale(), 1);
        assert_eq!(ledger.depth_of("s1"), ReadDepth::FullBody);
        assert_eq!(
            ledger.depth_of_for_agent("s1", "agent_a"),
            ReadDepth::FullBody
        );
        assert_eq!(
            ledger.depth_of_for_agent("s1", "agent_b"),
            ReadDepth::Overview
        );
    }

    /// Agent A read the old version in full; agent B then read the new one
    /// by name. B knows the current symbol, A does not, and the symbol as a
    /// whole is known by name only — not in full, as one shared hash said.
    #[test]
    fn each_agent_keeps_the_version_it_read() {
        let mut ledger = ContextLedger::new();
        let h1 = hash("v1");
        let h2 = hash("v2");

        ledger.record("s1".into(), ReadDepth::FullBody, h1, "a".into(), 10);
        ledger.mark_stale_if_changed("s1", h2);
        ledger.record("s1".into(), ReadDepth::NameOnly, h2, "b".into(), 10);

        let reads = &ledger.entries["s1"].agent_reads;
        assert_eq!(reads["a"].hash_at_read, h1);
        assert_eq!(reads["b"].hash_at_read, h2);
        assert!(ledger.is_stale_for_agent("s1", "a"));
        assert!(!ledger.is_stale_for_agent("s1", "b"));
        assert!(!ledger.is_stale("s1"), "b's read is current");
        assert_eq!(ledger.depth_of("s1"), ReadDepth::NameOnly);
        assert_eq!(ledger.entries["s1"].content_hash_at_read, h2);
    }

    /// Staleness is decided per agent against the current hash, so a symbol
    /// reverted to the version an agent read is current for that agent again.
    #[test]
    fn a_reverted_symbol_is_current_again_for_the_agent_that_read_that_version() {
        let mut ledger = ContextLedger::new();
        let h1 = hash("v1");
        let h2 = hash("v2");

        ledger.record("s1".into(), ReadDepth::FullBody, h1, "a".into(), 10);
        ledger.mark_stale_if_changed("s1", h2);
        ledger.record("s1".into(), ReadDepth::Overview, h2, "b".into(), 10);
        assert!(ledger.is_stale_for_agent("s1", "a"));

        // Reverted: v1 is back on disk.
        ledger.mark_stale_if_changed("s1", h1);
        assert!(!ledger.is_stale_for_agent("s1", "a"));
        assert!(ledger.is_stale_for_agent("s1", "b"));
        assert_eq!(ledger.depth_of("s1"), ReadDepth::FullBody);
        assert!(!ledger.is_stale("s1"));
    }

    /// Two `impl Foo` in one file share an id, so reads of it alternate
    /// between their hashes with nothing changed. A full read must survive a
    /// later name-only one, and no agent turns stale.
    #[test]
    fn symbols_sharing_an_id_are_not_changed_content() {
        let mut ledger = ContextLedger::new();
        let (first, second) = (hash("impl Foo { a }"), hash("impl Foo { b }"));

        // A whole-file read by b credits both, then a reads the first alone.
        ledger.record("f::Foo".into(), ReadDepth::FullBody, first, "b".into(), 1);
        ledger.record("f::Foo".into(), ReadDepth::FullBody, second, "b".into(), 1);
        ledger.record("f::Foo".into(), ReadDepth::FullBody, first, "a".into(), 1);
        // A grep hit in the second, by b.
        ledger.record("f::Foo".into(), ReadDepth::NameOnly, second, "b".into(), 1);

        assert_eq!(
            ledger.depth_of_for_agent("f::Foo", "b"),
            ReadDepth::FullBody
        );
        assert!(!ledger.is_stale_for_agent("f::Foo", "a"));
        assert!(!ledger.is_stale_for_agent("f::Foo", "b"));
        assert!(!ledger.is_stale("f::Foo"));
    }

    /// An agent that never read a symbol is not stale for it.
    #[test]
    fn an_agent_that_never_read_a_symbol_is_not_stale_for_it() {
        let mut ledger = ContextLedger::new();
        ledger.record("s1".into(), ReadDepth::FullBody, hash("v1"), "a".into(), 10);
        ledger.mark_stale_if_changed("s1", hash("v2"));
        assert!(!ledger.is_stale_for_agent("s1", "b"));
    }

    #[test]
    fn aggregate_depth_reflects_max_across_agents() {
        let mut ledger = ContextLedger::new();
        let h = hash("v1");

        // Three agents with increasing depths.
        ledger.record("s1".into(), ReadDepth::NameOnly, h, "a".into(), 1);
        assert_eq!(ledger.depth_of("s1"), ReadDepth::NameOnly);

        ledger.record("s1".into(), ReadDepth::Signature, h, "b".into(), 3);
        assert_eq!(ledger.depth_of("s1"), ReadDepth::Signature);

        ledger.record("s1".into(), ReadDepth::FullBody, h, "c".into(), 10);
        assert_eq!(ledger.depth_of("s1"), ReadDepth::FullBody);

        // Each agent retains its own depth.
        assert_eq!(ledger.depth_of_for_agent("s1", "a"), ReadDepth::NameOnly);
        assert_eq!(ledger.depth_of_for_agent("s1", "b"), ReadDepth::Signature);
        assert_eq!(ledger.depth_of_for_agent("s1", "c"), ReadDepth::FullBody);
    }
}
