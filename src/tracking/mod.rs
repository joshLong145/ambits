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

#[derive(Debug, Clone)]
pub struct ContextEntry {
    pub symbol_id: SymbolId,
    /// Aggregate depth: the maximum depth across all agents.
    pub depth: ReadDepth,
    /// Whether the symbol's content has changed since it was last read.
    ///
    /// Orthogonal to `depth` — a stale symbol retains the depth it was read
    /// at, so we can still say *how much* was read before it drifted. Set by
    /// [`ContextLedger::mark_stale_if_changed`], cleared by any subsequent
    /// read. Staleness is a property of the content, so it is shared by every
    /// agent rather than tracked per-agent.
    pub stale: bool,
    /// Content hash as of the most recent read. Always refreshed on every
    /// read, so it can be compared against the current tree to detect drift.
    pub content_hash_at_read: [u8; 32],
    pub timestamp: Instant,
    /// The last agent that touched this symbol.
    pub agent_id: String,
    pub token_count: usize,
    /// Per-agent depth tracking: each agent's independent read depth for this symbol.
    pub agent_depths: HashMap<String, ReadDepth>,
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
    /// Per-agent depths are tracked independently — agent A can have `FullBody`
    /// while agent B has `Overview` for the same symbol — and are upgrade-only.
    /// The aggregate `depth` field is always the maximum across all agents.
    ///
    /// Provenance (`stale`, `content_hash_at_read`, `timestamp`, `agent_id`,
    /// `token_count`) is refreshed on *every* read, not just on a depth
    /// upgrade. That matters: a re-read at an unchanged depth still proves the
    /// symbol was seen in its current form, so gating the hash refresh on a
    /// depth increase would leave a stale hash behind and make the entry look
    /// drifted when it isn't.
    pub fn record(
        &mut self,
        symbol_id: SymbolId,
        depth: ReadDepth,
        content_hash: [u8; 32],
        agent_id: String,
        token_count: usize,
    ) {
        let entry = self.entries.entry(symbol_id.clone()).or_insert_with(|| ContextEntry {
            symbol_id: symbol_id.clone(),
            depth: ReadDepth::Unseen,
            stale: false,
            content_hash_at_read: [0u8; 32],
            timestamp: Instant::now(),
            agent_id: String::new(),
            token_count: 0,
            agent_depths: HashMap::new(),
        });

        // Update per-agent depth (only upgrade, never downgrade).
        let agent_depth = entry.agent_depths.entry(agent_id.clone()).or_insert(ReadDepth::Unseen);
        if depth > *agent_depth {
            *agent_depth = depth;
        }

        // Recompute aggregate depth as max across all agents.
        entry.depth = entry
            .agent_depths
            .values()
            .copied()
            .max()
            .unwrap_or(ReadDepth::Unseen);

        // A read always re-establishes provenance: we have just seen this
        // symbol at its current content, so it is by definition not stale.
        entry.stale = false;
        entry.content_hash_at_read = content_hash;
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

    /// Flag a seen entry as stale when its content hash no longer matches.
    ///
    /// Depth is left untouched — we still know how much was read, we just know
    /// it no longer describes the current content. The flag is per-entry
    /// rather than per-agent because content drift affects every agent that
    /// read it equally.
    pub fn mark_stale_if_changed(&mut self, symbol_id: &str, current_hash: [u8; 32]) {
        if let Some(entry) = self.entries.get_mut(symbol_id) {
            if entry.depth.is_seen() && entry.content_hash_at_read != current_hash {
                entry.stale = true;
            }
        }
    }

    /// Whether the symbol has been read and has since drifted. `false` for
    /// untracked symbols.
    pub fn is_stale(&self, symbol_id: &str) -> bool {
        self.entries.get(symbol_id).map(|e| e.stale).unwrap_or(false)
    }

    /// Count of seen entries currently flagged stale.
    pub fn total_stale(&self) -> usize {
        self.entries.values().filter(|e| e.stale && e.depth.is_seen()).count()
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
            .and_then(|e| e.agent_depths.get(agent_id))
            .copied()
            .unwrap_or(ReadDepth::Unseen)
    }

    /// Returns the list of agent IDs that have interacted with a given symbol.
    pub fn agents_for_symbol(&self, symbol_id: &str) -> Vec<&str> {
        self.entries
            .get(symbol_id)
            .map(|e| e.agent_depths.keys().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// Count entries by depth for a specific agent.
    pub fn count_by_depth_for_agent(&self, agent_id: &str) -> HashMap<ReadDepth, usize> {
        let mut counts = HashMap::new();
        for entry in self.entries.values() {
            let depth = entry
                .agent_depths
                .get(agent_id)
                .copied()
                .unwrap_or(ReadDepth::Unseen);
            *counts.entry(depth).or_insert(0) += 1;
        }
        counts
    }

    /// Count total seen symbols for a specific agent.
    pub fn total_seen_for_agent(&self, agent_id: &str) -> usize {
        self.entries
            .values()
            .filter(|e| {
                e.agent_depths
                    .get(agent_id)
                    .copied()
                    .unwrap_or(ReadDepth::Unseen)
                    .is_seen()
            })
            .count()
    }
}

/// Collect all symbol IDs and their content hashes into `map` (recursive).
pub fn collect_symbol_hashes(
    symbols: &[SymbolNode],
    map: &mut HashMap<String, [u8; 32]>,
) {
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

    /// A lower-depth re-read must not downgrade the recorded depth, but must
    /// still refresh provenance.
    #[test]
    fn lower_depth_re_read_refreshes_hash_without_downgrading() {
        let mut ledger = ContextLedger::new();
        let h1 = hash("v1");
        let h2 = hash("v2");

        ledger.record("s1".into(), ReadDepth::FullBody, h1, "ag".into(), 10);
        ledger.record("s1".into(), ReadDepth::NameOnly, h2, "ag".into(), 10);

        assert_eq!(ledger.depth_of("s1"), ReadDepth::FullBody);
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
        assert_eq!(ledger.depth_of_for_agent("s1", "agent_a"), ReadDepth::Overview);
        assert_eq!(ledger.depth_of_for_agent("s1", "agent_b"), ReadDepth::FullBody);

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
        assert_eq!(ledger.depth_of_for_agent("s1", "agent_a"), ReadDepth::FullBody);
    }

    #[test]
    fn depth_of_for_agent_defaults_unseen() {
        let ledger = ContextLedger::new();
        assert_eq!(ledger.depth_of_for_agent("nonexistent", "agent_a"), ReadDepth::Unseen);

        // Symbol exists but agent doesn't.
        let mut ledger = ContextLedger::new();
        ledger.record("s1".into(), ReadDepth::Overview, hash("v1"), "agent_a".into(), 5);
        assert_eq!(ledger.depth_of_for_agent("s1", "agent_b"), ReadDepth::Unseen);
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

    /// Staleness is a property of the content, so it is recorded once per
    /// entry and applies to every agent that read it — while each agent's own
    /// depth is left intact.
    #[test]
    fn staleness_is_shared_across_agents_and_preserves_their_depths() {
        let mut ledger = ContextLedger::new();
        let h1 = hash("v1");
        let h2 = hash("v2");

        ledger.record("s1".into(), ReadDepth::FullBody, h1, "agent_a".into(), 10);
        ledger.record("s1".into(), ReadDepth::Overview, h1, "agent_b".into(), 5);

        ledger.mark_stale_if_changed("s1", h2);

        assert!(ledger.is_stale("s1"));
        assert_eq!(ledger.total_stale(), 1);
        assert_eq!(ledger.depth_of("s1"), ReadDepth::FullBody);
        assert_eq!(ledger.depth_of_for_agent("s1", "agent_a"), ReadDepth::FullBody);
        assert_eq!(ledger.depth_of_for_agent("s1", "agent_b"), ReadDepth::Overview);
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
