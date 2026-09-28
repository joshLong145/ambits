//! Snapshot ancestry, cached: which snapshots descend from which, without
//! loading and re-verifying every snapshot object on every question.
//!
//! A line of `.ambits/cache/graph.ndjson` is `{id, inputs, parents}`. A
//! snapshot's id *is* `BLAKE3(inputs digest, sorted parents)` (D17), so each
//! line proves itself: one whose id does not match its fields is ignored,
//! and a tampered parent list cannot be passed off. Such facts hold in every
//! store, so what one learned walking a remote answers questions about it
//! later. Only *ancestry* comes from here; whether a store holds an object,
//! and what a snapshot references, is always read from the store.

use std::collections::{HashMap, HashSet};

use color_eyre::eyre::{bail, Result};
use serde::{Deserialize, Serialize};

use super::flat::{worth_compacting, FlatLog};
use super::inputs::snapshot_id;
use super::snapshot::{Snapshot, MAX_HISTORY};
use super::store::{Durability, Store};
use super::ObjectId;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Line {
    id: String,
    inputs: String,
    parents: Vec<String>,
}

impl Line {
    /// The parents, if the line is what its id says it is.
    fn verified(&self) -> Option<(ObjectId, Vec<ObjectId>)> {
        let id = ObjectId::parse(&self.id).ok()?;
        let inputs = crate::journal::decode_hash(&self.inputs)?;
        let parents = self.parents.iter().map(|p| ObjectId::parse(p)).collect::<Result<Vec<_>>>().ok()?;
        (snapshot_id(&inputs, &parents) == id).then_some((id, parents))
    }
}

fn graph_file(store: &Store) -> FlatLog {
    FlatLog::at(store.root().join(crate::state_dir::CACHE).join(crate::state_dir::GRAPH))
}

/// The ancestry this store knows, and what it learned since loading.
pub struct Graph {
    file: FlatLog,
    parents: HashMap<ObjectId, Vec<ObjectId>>,
    learned: Vec<Line>,
}

impl Graph {
    /// The cache of the project store `local` (best-effort: an unreadable
    /// cache is an empty one).
    pub fn load(local: &Store) -> Self {
        let file = graph_file(local);
        let parents = file.read::<Line>().unwrap_or_default().iter().filter_map(Line::verified).collect();
        Self { file, parents, learned: Vec::new() }
    }

    /// Record `snap`'s parents, as loaded (so verified) from some store.
    pub fn learn(&mut self, snap: &Snapshot) {
        if self.parents.contains_key(&snap.id) {
            return;
        }
        self.parents.insert(snap.id, snap.parents.clone());
        self.learned.push(Line {
            id: snap.id.hex(),
            inputs: crate::journal::encode_hash(&snap.inputs_digest),
            parents: snap.parents.iter().map(ObjectId::hex).collect(),
        });
    }

    /// `id`'s parents: known, or read (and verified) from `store`.
    pub fn parents(&mut self, store: &Store, id: ObjectId) -> Result<Vec<ObjectId>> {
        if let Some(p) = self.parents.get(&id) {
            return Ok(p.clone());
        }
        let snap = Snapshot::load(store, id)?;
        self.learn(&snap);
        Ok(snap.parents)
    }

    /// `tip` and everything it descends from.
    pub fn ancestors(&mut self, store: &Store, tip: ObjectId) -> Result<HashSet<ObjectId>> {
        let mut seen = HashSet::new();
        self.walk(store, tip, |id| {
            seen.insert(id);
            false
        })?;
        Ok(seen)
    }

    /// Whether `descendant` is `ancestor` or descends from it; the walk stops
    /// as soon as it is found.
    pub fn is_ancestor(&mut self, store: &Store, ancestor: ObjectId, descendant: ObjectId) -> Result<bool> {
        self.walk(store, descendant, |id| id == ancestor)
    }

    /// Visit `tip` and its ancestors, each once, until `stop` says so;
    /// whether it did.
    fn walk(&mut self, store: &Store, tip: ObjectId, mut stop: impl FnMut(ObjectId) -> bool) -> Result<bool> {
        let mut seen = HashSet::new();
        let mut stack = vec![tip];
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue;
            }
            if stop(id) {
                return Ok(true);
            }
            if seen.len() > MAX_HISTORY {
                bail!("history of {} is longer than {MAX_HISTORY} snapshots", tip.short());
            }
            stack.extend(self.parents(store, id)?);
        }
        Ok(false)
    }

    /// Write what was learned; rewrite the cache once it is mostly
    /// repeats. Best-effort: the cache only saves work.
    pub fn save(&mut self) -> Result<()> {
        if self.learned.is_empty() {
            return Ok(());
        }
        let lock = self.file.lock()?;
        let lines = self.file.line_count() + self.learned.len();
        if worth_compacting(lines, self.parents.len()) {
            // The file's lines that still prove themselves, and what was
            // learned: one line per snapshot.
            let mut kept: Vec<Line> = self.file.read::<Line>()?.into_iter().filter(|l| l.verified().is_some()).collect();
            kept.extend(self.learned.iter().cloned());
            kept.sort_by(|a, b| a.id.cmp(&b.id));
            kept.dedup_by(|a, b| a.id == b.id);
            self.file.rewrite(&lock, &kept, Durability::NoSync)?;
        } else {
            self.file.append(&lock, &self.learned, Durability::NoSync)?;
        }
        self.learned.clear();
        Ok(())
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        if let Err(e) = self.save() {
            log::warn!(target: "ambits::objects", "ancestry cache not written: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(id: ObjectId, inputs: [u8; 32], parents: &[ObjectId]) -> Line {
        Line { id: id.hex(), inputs: crate::journal::encode_hash(&inputs), parents: parents.iter().map(ObjectId::hex).collect() }
    }

    /// A line proves itself: forge its parents and it is ignored.
    #[test]
    fn a_line_whose_id_does_not_match_is_ignored() {
        let (inputs, parent) = ([7u8; 32], ObjectId([1; 32]));
        let id = snapshot_id(&inputs, &[parent]);
        assert_eq!(line(id, inputs, &[parent]).verified(), Some((id, vec![parent])));
        assert_eq!(line(id, inputs, &[]).verified(), None, "a parent dropped");
        assert_eq!(line(id, inputs, &[ObjectId([2; 32])]).verified(), None, "a parent swapped");
    }

    /// Answered from the cache: the store holds none of these objects.
    #[test]
    fn ancestry_is_answered_from_verified_lines() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::at(dir.path());
        let a = snapshot_id(&[1; 32], &[]);
        let b = snapshot_id(&[2; 32], &[a]);
        let c = snapshot_id(&[3; 32], &[b]);
        let file = graph_file(&store);
        let lock = file.lock().unwrap();
        file.append(&lock, &[line(a, [1; 32], &[]), line(b, [2; 32], &[a]), line(c, [3; 32], &[b])], Durability::NoSync).unwrap();
        drop(lock);
        let mut graph = Graph::load(&store);
        assert!(graph.is_ancestor(&store, a, c).unwrap());
        assert!(!graph.is_ancestor(&store, c, a).unwrap_or(false));
        assert_eq!(graph.ancestors(&store, c).unwrap(), HashSet::from([a, b, c]));
    }
}
