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
    pub depth: ReadDepth,
    pub content_hash_at_read: [u8; 32],
    pub timestamp: Instant,
    pub agent_id: String,
    pub token_count: usize,
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

    /// Record that a symbol was seen at the given depth.
    /// Only upgrades depth (never downgrades, except to Stale).
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
        });

        // Only upgrade, never downgrade (except Stale overrides everything).
        if depth == ReadDepth::Stale || depth > entry.depth {
            entry.depth = depth;
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
    pub fn mark_stale_if_changed(&mut self, symbol_id: &str, current_hash: [u8; 32]) {
        if let Some(entry) = self.entries.get_mut(symbol_id) {
            if entry.depth != ReadDepth::Unseen && entry.content_hash_at_read != current_hash {
                entry.depth = ReadDepth::Stale;
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
}
