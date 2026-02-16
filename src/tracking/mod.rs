pub mod agents;

use std::collections::HashMap;
use std::time::Instant;

use crate::symbols::SymbolId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ReadDepth {
    Unseen,
    NameOnly,
    Overview,
    Signature,
    FullBody,
    Stale,
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
            ReadDepth::Stale => write!(f, "stale"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ContextEntry {
    pub symbol_id: SymbolId,
    /// Aggregate depth: the maximum depth across all agents.
    pub depth: ReadDepth,
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
    /// while agent B has `Overview` for the same symbol. The aggregate `depth`
    /// field is always the maximum across all agents.
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
            content_hash_at_read: [0u8; 32],
            timestamp: Instant::now(),
            agent_id: String::new(),
            token_count: 0,
            agent_depths: HashMap::new(),
        });

        // Update per-agent depth (only upgrade, never downgrade).
        let agent_depth = entry.agent_depths.entry(agent_id.clone()).or_insert(ReadDepth::Unseen);
        if depth == ReadDepth::Stale || depth > *agent_depth {
            *agent_depth = depth;
        }

        // Recompute aggregate depth as max across all agents.
        let max_depth = entry
            .agent_depths
            .values()
            .copied()
            .max()
            .unwrap_or(ReadDepth::Unseen);

        if max_depth == ReadDepth::Stale || max_depth > entry.depth {
            entry.depth = max_depth;
            entry.content_hash_at_read = content_hash;
            entry.timestamp = Instant::now();
            entry.agent_id = agent_id;
            entry.token_count = token_count;
        }
    }

    /// Get the read depth for a symbol, defaulting to Unseen.
    pub fn depth_of(&self, symbol_id: &str) -> ReadDepth {
        self.entries
            .get(symbol_id)
            .map(|e| e.depth)
            .unwrap_or(ReadDepth::Unseen)
    }

    /// Mark all entries whose content hash no longer matches as Stale.
    /// Propagates the stale marker to all per-agent depth entries as well.
    pub fn mark_stale_if_changed(&mut self, symbol_id: &str, current_hash: [u8; 32]) {
        if let Some(entry) = self.entries.get_mut(symbol_id) {
            if entry.depth != ReadDepth::Unseen && entry.content_hash_at_read != current_hash {
                entry.depth = ReadDepth::Stale;
                for depth in entry.agent_depths.values_mut() {
                    *depth = ReadDepth::Stale;
                }
            }
        }
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
    fn stale_overrides_everything() {
        let mut ledger = ContextLedger::new();
        ledger.record("s1".into(), ReadDepth::FullBody, hash("a"), "ag".into(), 10);
        ledger.record("s1".into(), ReadDepth::Stale, hash("b"), "ag".into(), 10);
        assert_eq!(ledger.depth_of("s1"), ReadDepth::Stale);
    }

    #[test]
    fn mark_stale_if_changed() {
        let mut ledger = ContextLedger::new();
        let h1 = hash("v1");
        let h2 = hash("v2");
        ledger.record("s1".into(), ReadDepth::FullBody, h1, "ag".into(), 10);

        // Same hash — no change.
        ledger.mark_stale_if_changed("s1", h1);
        assert_eq!(ledger.depth_of("s1"), ReadDepth::FullBody);

        // Different hash — becomes stale.
        ledger.mark_stale_if_changed("s1", h2);
        assert_eq!(ledger.depth_of("s1"), ReadDepth::Stale);
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

    #[test]
    fn mark_stale_propagates_to_agent_depths() {
        let mut ledger = ContextLedger::new();
        let h1 = hash("v1");
        let h2 = hash("v2");

        ledger.record("s1".into(), ReadDepth::FullBody, h1, "agent_a".into(), 10);
        ledger.record("s1".into(), ReadDepth::Overview, h1, "agent_b".into(), 5);

        // Mark stale with changed hash.
        ledger.mark_stale_if_changed("s1", h2);

        // Both aggregate and per-agent depths should be stale.
        assert_eq!(ledger.depth_of("s1"), ReadDepth::Stale);
        assert_eq!(ledger.depth_of_for_agent("s1", "agent_a"), ReadDepth::Stale);
        assert_eq!(ledger.depth_of_for_agent("s1", "agent_b"), ReadDepth::Stale);
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
