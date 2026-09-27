//! Garbage collection (spec §8).
//!
//! Safety rests on three rules the formal model found necessary (§16):
//!
//! 1. **The gc lock.** `gc` holds `.ambits/gc.lock` exclusively; `snapshot`
//!    holds it shared. It is an OS advisory lock, so a crashed holder
//!    releases it.
//! 2. **A grace period, re-checked at deletion.** Only unreachable objects
//!    older than the grace period go, and age is read again immediately
//!    before each deletion. Writers refresh the age of objects they find
//!    present, so an object a snapshot just reused cannot be collected.
//! 3. **Parents first.** An object is deleted only after everything
//!    unreachable that references it, so a present object always has its
//!    children — even if gc is interrupted.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use color_eyre::eyre::{Result, WrapErr};
use serde_json::Value;

use super::refs;
use super::snapshot::Snapshot;
use super::store::{create_private_dir, is_temp, walk_files, Store};
use super::{canonical, tree, Kind, ObjectId};

/// Unreachable objects younger than this many days are kept (§8).
pub const DEFAULT_GRACE_DAYS: u64 = 14;
/// [`DEFAULT_GRACE_DAYS`] as a duration.
pub const DEFAULT_GRACE: Duration = Duration::from_secs(DEFAULT_GRACE_DAYS * crate::time::SECS_PER_DAY);

/// `.ambits/gc.lock`, held for as long as this value lives.
pub struct GcLock(#[allow(dead_code)] fs::File);

impl GcLock {
    fn open(store: &Store) -> Result<fs::File> {
        create_private_dir(store.root())?;
        let path = store.root().join("gc.lock");
        fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&path)
            .wrap_err_with(|| format!("opening {}", path.display()))
    }

    /// Held by `snapshot` (and later `fetch` and `pull`) for its duration.
    pub fn shared(store: &Store) -> Result<Self> {
        let file = Self::open(store)?;
        fs4::FileExt::lock_shared(&file).wrap_err("waiting for the gc lock")?;
        Ok(Self(file))
    }

    /// Held by `gc`.
    pub fn exclusive(store: &Store) -> Result<Self> {
        let file = Self::open(store)?;
        fs4::FileExt::lock(&file).wrap_err("waiting for the gc lock")?;
        Ok(Self(file))
    }
}

/// What a gc run did.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct GcStats {
    pub reachable: usize,
    /// Deleted, in deletion order.
    pub deleted: Vec<ObjectId>,
    /// Unreachable but kept: too young, or referenced by something kept.
    pub kept: usize,
    pub reflog_entries_expired: usize,
    pub notes_removed: usize,
    pub temp_files_removed: usize,
}

/// Collect `store`'s unreachable objects older than `grace`, after dropping
/// reflog entries older than `reflog_expiry` (§8; 90 days by default).
pub fn gc(store: &Store, grace: Duration, reflog_expiry: Duration) -> Result<GcStats> {
    let _lock = GcLock::exclusive(store)?;
    let mut stats = GcStats { reflog_entries_expired: refs::expire_reflogs(store, reflog_expiry)?, ..Default::default() };

    let reachable = mark(store)?;
    stats.reachable = reachable.len();

    // The unreachable subgraph, with who references whom inside it.
    let unreachable: HashMap<ObjectId, PathBuf> =
        store.list().into_iter().filter(|(id, _)| !reachable.contains(id)).collect();
    let mut children: HashMap<ObjectId, Vec<ObjectId>> = HashMap::new();
    let mut referrers: HashMap<ObjectId, usize> = unreachable.keys().map(|id| (*id, 0)).collect();
    for id in unreachable.keys() {
        let refs: Vec<ObjectId> = references_of(store, id).into_iter().filter(|c| unreachable.contains_key(c)).collect();
        for c in &refs {
            *referrers.get_mut(c).expect("present") += 1;
        }
        children.insert(*id, refs);
    }

    // Kahn's order: an object becomes deletable only once every unreachable
    // object referencing it is gone. A kept object never releases its
    // children, so they are kept too.
    let mut ready: Vec<ObjectId> = referrers.iter().filter(|(_, n)| **n == 0).map(|(id, _)| *id).collect();
    ready.sort();
    while let Some(id) = ready.pop() {
        let path = &unreachable[&id];
        if !older_than(path, grace) {
            continue;
        }
        fs::remove_file(path).wrap_err_with(|| format!("removing {}", path.display()))?;
        stats.deleted.push(id);
        for c in &children[&id] {
            let n = referrers.get_mut(c).expect("present");
            *n -= 1;
            if *n == 0 {
                ready.push(*c);
            }
        }
    }
    stats.kept = unreachable.len() - stats.deleted.len();

    // Notes of snapshots that no longer exist.
    for path in walk_files(&store.root().join(crate::state_dir::NOTES)) {
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let orphan = name.strip_suffix(".json").and_then(|n| ObjectId::parse(n).ok()).is_some_and(|id| !store.contains(&id));
        if orphan && fs::remove_file(&path).is_ok() {
            stats.notes_removed += 1;
        }
    }
    // Temp files a crash left, wherever ambits writes atomically.
    for dir in crate::state_dir::STORE_DIRS {
        for path in walk_files(&store.root().join(dir)) {
            if is_temp(&path) && older_than(&path, grace) && fs::remove_file(&path).is_ok() {
                stats.temp_files_removed += 1;
            }
        }
    }
    Ok(stats)
}

/// Every object reachable from a root: every ref (local and
/// remote-tracking) and every unexpired reflog entry (§8).
fn mark(store: &Store) -> Result<HashSet<ObjectId>> {
    let mut roots: Vec<ObjectId> = refs::all(store).into_iter().map(|(_, id)| id).collect();
    for (_, entries) in refs::reflogs(store) {
        for e in entries {
            roots.extend(ObjectId::parse(&e.new));
            roots.extend(e.old.as_deref().and_then(|o| ObjectId::parse(o).ok()));
        }
    }

    let mut reachable = HashSet::new();
    let mut snapshots = roots;
    while let Some(id) = snapshots.pop() {
        if !store.contains(&id) || !reachable.insert(id) {
            continue;
        }
        let snapshot = Snapshot::load(store, id).wrap_err("gc refuses to run on a store it cannot read")?;
        for (child, kind) in snapshot.references() {
            match kind {
                Kind::Snapshot => snapshots.push(child),
                Kind::Dir => tree::walk(store, child, &mut |id, _| {
                    reachable.insert(id);
                })?,
                _ => {
                    reachable.insert(child);
                }
            }
        }
    }
    Ok(reachable)
}

/// What an object references, read without trusting it: an object that
/// cannot be parsed references nothing, and is simply collected.
fn references_of(store: &Store, id: &ObjectId) -> Vec<ObjectId> {
    let Some((kind, payload)) = fs::read(store.path_of(id)).ok().and_then(|b| canonical::from_bytes(&b).ok()).and_then(|v| {
        let kind = Kind::from_name(v.get("type")?.as_str()?)?;
        Some((kind, v.get("payload")?.clone()))
    }) else {
        return Vec::new();
    };
    match kind {
        Kind::Snapshot => ["root", "coverage", "writes"]
            .iter()
            .filter_map(|k| payload.get(*k).and_then(Value::as_str))
            .chain(payload.get("parents").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str))
            .filter_map(|s| ObjectId::parse(s).ok())
            .collect(),
        _ => tree::references(kind, &payload).map(|r| r.into_iter().map(|(id, _)| id).collect()).unwrap_or_default(),
    }
}

fn older_than(path: &Path, grace: Duration) -> bool {
    fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| SystemTime::now().duration_since(t).ok())
        .is_some_and(|age| age >= grace)
}
