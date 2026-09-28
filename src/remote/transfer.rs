//! Copying a snapshot's history between stores (spec §12.2, §9.4), for
//! push and fetch alike.
//!
//! Everything is read through [`Store::get`] and [`Snapshot::load`], which
//! verify: regular files only, a size and entry cap, content ids re-hashed,
//! snapshot ids and state digests recomputed. What is written lands
//! children first, fsynced, each snapshot after everything it references.

use std::collections::HashSet;

use color_eyre::eyre::{bail, Result, WrapErr};

use crate::objects::graph::Graph;
use crate::objects::snapshot::{Snapshot, MAX_HISTORY};
use crate::objects::store::{refresh_age, Store};
use crate::objects::{Kind, ObjectId};

/// `tip` and its ancestors, loaded and verified from `store`, parents before
/// children. Each snapshot is loaded once.
pub fn closure(store: &Store, tip: ObjectId) -> Result<Vec<Snapshot>> {
    let mut out = Vec::new();
    let mut queued: HashSet<ObjectId> = HashSet::from([tip]);
    // (snapshot, its parents already queued)
    let mut stack: Vec<(Snapshot, bool)> = vec![(Snapshot::load(store, tip)?, false)];
    while let Some((snap, expanded)) = stack.pop() {
        if expanded {
            out.push(snap);
            continue;
        }
        let parents: Vec<ObjectId> = snap.parents.iter().filter(|p| queued.insert(**p)).copied().collect();
        if queued.len() > MAX_HISTORY {
            bail!("history of {} is longer than {MAX_HISTORY} snapshots", tip.short());
        }
        stack.push((snap, true));
        for p in parents {
            stack.push((Snapshot::load(store, p).wrap_err("history is incomplete")?, false));
        }
    }
    Ok(out)
}

/// How much of a history a transfer re-checks.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Verify {
    /// Walk back only to what the destination already holds whole: the
    /// store writes children before parents and gc deletes parents first,
    /// so a snapshot held whole has all it descends from (§8).
    #[default]
    Frontier,
    /// Walk and re-check the whole history, repairing anything that no
    /// longer verifies (`--verify-all`).
    Everything,
}

/// `id`, if `store` holds it whole: the snapshot and the objects it
/// references all load and verify.
fn held_whole(store: &Store, id: ObjectId) -> Option<Snapshot> {
    if !store.contains(&id) {
        return None;
    }
    let snap = Snapshot::load(store, id).ok()?;
    (store.get(&snap.coverage, Kind::Coverage).is_ok() && store.get(&snap.writes, Kind::Writes).is_ok()).then_some(snap)
}

/// The part of `tip`'s history `to` needs from `from`, loaded and verified
/// from `from`, parents before children. Under [`Verify::Frontier`] the walk
/// stops at each snapshot `to` holds whole — after checking it is the same
/// snapshot there, not another under its id — so a push or fetch costs what
/// is new, not the whole history.
pub fn missing(graph: &mut Graph, from: &Store, to: &Store, tip: ObjectId, verify: Verify) -> Result<Vec<Snapshot>> {
    let mut out = Vec::new();
    let mut queued: HashSet<ObjectId> = HashSet::from([tip]);
    let mut stack: Vec<(ObjectId, Option<Snapshot>)> = vec![(tip, None)];
    while let Some((id, loaded)) = stack.pop() {
        if let Some(snap) = loaded {
            out.push(snap);
            continue;
        }
        if verify == Verify::Frontier {
            if let Some(there) = held_whole(to, id) {
                if Snapshot::load(from, id)?.state_digest != there.state_digest {
                    bail!("snapshot {} already exists with different contents; refusing it", id.short());
                }
                graph.learn(&there);
                continue;
            }
        }
        let snap = Snapshot::load(from, id).wrap_err("history is incomplete")?;
        graph.learn(&snap);
        let parents: Vec<ObjectId> = snap.parents.iter().filter(|p| queued.insert(**p)).copied().collect();
        if queued.len() > MAX_HISTORY {
            bail!("history of {} is longer than {MAX_HISTORY} snapshots", tip.short());
        }
        stack.push((id, Some(snap)));
        stack.extend(parents.into_iter().map(|p| (p, None)));
    }
    Ok(out)
}

/// What a transfer did.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Stats {
    /// Written: missing, or there but no longer verifying.
    pub copied: usize,
    pub present: usize,
}

/// Copy `history` (from [`missing`] or [`closure`] over `from`) to `to`,
/// repairing any object there that no longer verifies, then check `to`
/// holds all of it — including what was skipped as already there.
pub fn transfer(from: &Store, to: &Store, history: &[Snapshot]) -> Result<Stats> {
    let mut stats = Stats::default();
    for snap in history {
        for (id, kind) in [(snap.coverage, Kind::Coverage), (snap.writes, Kind::Writes)] {
            if to.contains(&id) && to.get(&id, kind).is_ok() {
                refresh_age(&to.path_of(&id));
                stats.present += 1;
            } else {
                to.replace_at(&id, kind, &from.get(&id, kind)?)?;
                stats.copied += 1;
            }
        }
        match to.contains(&snap.id).then(|| Snapshot::load(to, snap.id)) {
            // A snapshot id is derived, not a content hash: the same id with
            // other contents is a forgery or corruption, never a skip (§6.3).
            Some(Ok(there)) if there.state_digest != snap.state_digest => {
                bail!("snapshot {} already exists with different contents; refusing it", snap.id.short());
            }
            Some(Ok(_)) => {
                refresh_age(&to.path_of(&snap.id));
                stats.present += 1;
            }
            // Missing, or there but unreadable: write the verified copy.
            _ => {
                to.sync_dirs()?;
                to.replace_at(&snap.id, Kind::Snapshot, &snap.to_value())?;
                stats.copied += 1;
            }
        }
    }
    to.sync_dirs()?;
    for snap in history {
        Snapshot::load(to, snap.id)?;
        to.get(&snap.coverage, Kind::Coverage)?;
        to.get(&snap.writes, Kind::Writes)?;
    }
    Ok(stats)
}
