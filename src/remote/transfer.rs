//! Copying a snapshot's history between stores (spec §12.2, §9.4), for
//! push and fetch alike.
//!
//! Everything is read through [`Store::get`] and [`Snapshot::load`], which
//! verify: regular files only, a size and entry cap, content ids re-hashed,
//! snapshot ids and state digests recomputed. What is written lands
//! children first, fsynced, each snapshot after everything it references.

use std::collections::HashSet;

use color_eyre::eyre::{bail, Result, WrapErr};

use crate::objects::snapshot::{Snapshot, MAX_HISTORY};
use crate::objects::store::{refresh_age, Store};
use crate::objects::{Kind, ObjectId};

/// `tip` and its ancestors, loaded and verified from `store`, parents before
/// children.
pub fn closure(store: &Store, tip: ObjectId) -> Result<Vec<Snapshot>> {
    let mut out = Vec::new();
    let mut done: HashSet<ObjectId> = HashSet::new();
    // (snapshot, its parents already queued)
    let mut stack: Vec<(Snapshot, bool)> = vec![(Snapshot::load(store, tip)?, false)];
    while let Some((snap, expanded)) = stack.pop() {
        if done.contains(&snap.id) {
            continue;
        }
        if expanded {
            done.insert(snap.id);
            out.push(snap);
            if out.len() > MAX_HISTORY {
                bail!("history of {} is longer than {MAX_HISTORY} snapshots", tip.short());
            }
            continue;
        }
        let parents: Vec<ObjectId> = snap.parents.iter().filter(|p| !done.contains(p)).copied().collect();
        stack.push((snap, true));
        for p in parents {
            stack.push((Snapshot::load(store, p).wrap_err("history is incomplete")?, false));
        }
    }
    Ok(out)
}

/// What a transfer did.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Stats {
    pub copied: usize,
    pub present: usize,
}

/// Copy `tip`'s history from `from` to `to`, then check `to` holds all of
/// it — including what was skipped as already there.
pub fn transfer(from: &Store, to: &Store, tip: ObjectId) -> Result<Stats> {
    let history = closure(from, tip)?;
    let mut stats = Stats::default();
    for snap in &history {
        for (id, kind) in [(snap.coverage, Kind::Coverage), (snap.writes, Kind::Writes)] {
            if to.contains(&id) {
                refresh_age(&to.path_of(&id));
                stats.present += 1;
            } else {
                to.put_at(&id, kind, &from.get(&id, kind)?)?;
                stats.copied += 1;
            }
        }
        if to.contains(&snap.id) {
            // A snapshot id is derived, not a content hash: the same id with
            // other contents is a forgery or corruption, never a skip (§6.3).
            let there = Snapshot::load(to, snap.id)?;
            if there.state_digest != snap.state_digest {
                bail!("snapshot {} already exists with different contents; refusing it", snap.id.short());
            }
            refresh_age(&to.path_of(&snap.id));
            stats.present += 1;
        } else {
            to.sync_dirs()?;
            to.put_at(&snap.id, Kind::Snapshot, &snap.to_value())?;
            stats.copied += 1;
        }
    }
    to.sync_dirs()?;
    verify(to, &history)?;
    Ok(stats)
}

/// Every snapshot of `history`, and what it references, readable and
/// verified in `store`.
fn verify(store: &Store, history: &[Snapshot]) -> Result<()> {
    for snap in history {
        Snapshot::load(store, snap.id)?;
        store.get(&snap.coverage, Kind::Coverage)?;
        store.get(&snap.writes, Kind::Writes)?;
    }
    Ok(())
}
