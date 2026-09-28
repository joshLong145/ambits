//! Session refs, the reflog and notes (spec §8), each one flat file:
//!
//! ```text
//! .ambits/reflog.ndjson   one line per ref move; a ref's tip is its last
//! .ambits/notes.ndjson    one line per snapshot: time, message, version
//! ```
//!
//! The reflog is the refs: moving a ref appends its entry under the file's
//! lock, after checking the ref still points where the caller saw it.

use std::collections::HashMap;
use std::time::Duration;

use color_eyre::eyre::{bail, Result};
use serde::{Deserialize, Serialize};

use super::flat::FlatLog;
use super::store::{Durability, Store};
use super::ObjectId;
use crate::time::{now_rfc3339, now_secs};

/// Default days after which reflog entries stop protecting objects from gc
/// (§8); `ambits gc --reflog-expiry-days` overrides it.
pub const REFLOG_EXPIRY_DAYS: u64 = 90;
/// [`REFLOG_EXPIRY_DAYS`] as a duration.
pub const REFLOG_EXPIRY: Duration = Duration::from_secs(REFLOG_EXPIRY_DAYS * crate::time::SECS_PER_DAY);

/// A validated ref name, e.g. `refs/sessions/<id>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefName(String);

impl RefName {
    /// The ref of a session. Session ids are untrusted names (§9.1) and must
    /// be UUIDs, which is what Claude Code uses.
    pub fn session(session: &str) -> Result<Self> {
        if !crate::ingest::claude::is_uuid(session) {
            bail!("not a session id: {session:?}");
        }
        Ok(Self(format!("refs/sessions/{session}")))
    }

    /// This store's copy of `remote`'s ref for `session`, as last fetched
    /// or pushed: `refs/remotes/<remote>/sessions/<session>`.
    pub fn tracking(remote: &str, session: &str) -> Result<Self> {
        if !valid_remote_name(remote) {
            bail!("not a remote name: {remote:?}");
        }
        let session = Self::session(session)?;
        Ok(Self(format!("refs/remotes/{remote}/sessions/{}", session.session_id())))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The session this ref belongs to.
    pub fn session_id(&self) -> &str {
        self.0.rsplit('/').next().unwrap_or(&self.0)
    }
}

/// A remote's name: letters, digits, `.`, `_` and `-`, not starting with
/// `.`, at most 64 — one path component and one ref component (§9.1).
pub fn valid_remote_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && !name.starts_with('.')
        && name.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

/// Directories an older ambits kept refs, reflogs and notes in.
const OLD_LAYOUT: [&str; 3] = ["refs", "logs", "notes"];

/// Refuse a store laid out by an older ambits: pre-1.0, it is not
/// converted.
fn check_layout(store: &Store) -> Result<()> {
    let old: Vec<&str> = OLD_LAYOUT.into_iter().filter(|d| store.root().join(d).is_dir()).collect();
    if !old.is_empty() {
        let dirs: Vec<String> = old.iter().map(|d| format!(".ambits/{d}")).collect();
        bail!(
            "this snapshot store was written by an older ambits (a file per ref, reflog and note) and is not converted; \
             delete {} and .ambits/objects to start a new history",
            dirs.join(", ")
        );
    }
    Ok(())
}

fn reflog_file(store: &Store) -> FlatLog {
    FlatLog::at(store.root().join(crate::state_dir::REFLOG))
}

fn notes_file(store: &Store) -> FlatLog {
    FlatLog::at(store.root().join(crate::state_dir::NOTES))
}

/// One move of one ref.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReflogEntry {
    #[serde(rename = "ref")]
    pub name: String,
    pub old: Option<String>,
    pub new: String,
    /// Seconds since the epoch, for expiry.
    pub secs: u64,
    pub time: String,
    pub action: String,
}

/// Every ref's tip: the `new` of its last entry.
fn tips(entries: &[ReflogEntry]) -> HashMap<&str, &str> {
    entries.iter().map(|e| (e.name.as_str(), e.new.as_str())).collect()
}

/// The snapshot `name` points at, if any.
pub fn read(store: &Store, name: &RefName) -> Result<Option<ObjectId>> {
    check_layout(store)?;
    let entries = reflog(store)?;
    tips(&entries).get(name.as_str()).map(|t| ObjectId::parse(t)).transpose()
}

/// Every ref, local and remote-tracking, with its tip.
pub fn all(store: &Store) -> Result<Vec<(String, ObjectId)>> {
    let entries = reflog(store)?;
    let mut out: Vec<(String, ObjectId)> =
        tips(&entries).into_iter().filter_map(|(name, tip)| Some((name.to_string(), ObjectId::parse(tip).ok()?))).collect();
    out.sort();
    Ok(out)
}

/// Move `name` from `expected` to `new`: one reflog entry, appended under
/// the reflog's lock once the ref is seen to hold `expected` still — if it
/// does not, another process moved it and nothing is written.
pub fn update(store: &Store, name: &RefName, expected: Option<ObjectId>, new: ObjectId, action: &str) -> Result<()> {
    check_layout(store)?;
    let file = reflog_file(store);
    let lock = file.lock()?;
    let entries = reflog(store)?;
    let current = tips(&entries).get(name.as_str()).and_then(|t| ObjectId::parse(t).ok());
    if current != expected {
        bail!("{} moved meanwhile (another snapshot, restore or push); run it again", name.as_str());
    }
    let entry = ReflogEntry {
        name: name.as_str().to_string(),
        old: current.map(|c| c.hex()),
        new: new.hex(),
        secs: now_secs(),
        time: now_rfc3339(),
        action: action.to_string(),
    };
    file.append(&lock, &[entry], Durability::Fsync)
}

/// Every reflog entry, oldest first. Unparseable lines are skipped.
pub fn reflog(store: &Store) -> Result<Vec<ReflogEntry>> {
    reflog_file(store).read()
}

/// Drop reflog entries older than `expiry` — but never a ref's latest,
/// which is where it points.
pub fn expire_reflogs(store: &Store, expiry: Duration) -> Result<usize> {
    // gc runs this first: on an old layout it would see no refs, and
    // collect every object.
    check_layout(store)?;
    let cutoff = now_secs().saturating_sub(expiry.as_secs());
    let file = reflog_file(store);
    let lock = file.lock()?;
    let entries = reflog(store)?;
    let last: HashMap<&str, usize> = entries.iter().enumerate().map(|(i, e)| (e.name.as_str(), i)).collect();
    let kept: Vec<&ReflogEntry> = entries.iter().enumerate().filter(|(i, e)| e.secs >= cutoff || last[e.name.as_str()] == *i).map(|(_, e)| e).collect();
    let dropped = entries.len() - kept.len();
    if dropped > 0 {
        file.rewrite(&lock, &kept, Durability::Fsync)?;
    }
    Ok(dropped)
}

/// A snapshot's note: when it was made, the message, the ambits version.
/// Never host, branch or project root (D16).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Note {
    pub time: String,
    #[serde(default)]
    pub message: Option<String>,
    pub version: String,
}

/// A note as stored: with the snapshot it belongs to.
#[derive(Serialize, Deserialize)]
struct NoteLine {
    id: String,
    #[serde(flatten)]
    note: Note,
}

pub fn write_note(store: &Store, id: &ObjectId, message: Option<&str>) -> Result<()> {
    check_layout(store)?;
    let note = Note { time: now_rfc3339(), message: message.map(String::from), version: env!("CARGO_PKG_VERSION").to_string() };
    let file = notes_file(store);
    let lock = file.lock()?;
    file.append(&lock, &[NoteLine { id: id.hex(), note }], Durability::Fsync)
}

/// Every snapshot's note; the last written wins.
pub fn notes(store: &Store) -> Result<HashMap<ObjectId, Note>> {
    Ok(notes_file(store).read::<NoteLine>()?.into_iter().filter_map(|l| Some((ObjectId::parse(&l.id).ok()?, l.note))).collect())
}

pub fn read_note(store: &Store, id: &ObjectId) -> Option<Note> {
    notes(store).ok()?.remove(id)
}

/// Copy `from`'s notes for `ids` that `to` lacks; how many.
pub fn copy_notes(from: &Store, to: &Store, ids: &std::collections::HashSet<ObjectId>) -> Result<usize> {
    let have = notes(to)?;
    let missing: Vec<NoteLine> = notes(from)?
        .into_iter()
        .filter(|(id, _)| ids.contains(id) && !have.contains_key(id))
        .map(|(id, note)| NoteLine { id: id.hex(), note })
        .collect();
    if !missing.is_empty() {
        let file = notes_file(to);
        let lock = file.lock()?;
        file.append(&lock, &missing, Durability::Fsync)?;
    }
    Ok(missing.len())
}

/// Drop the notes of snapshots `keep` rejects; how many went.
pub fn prune_notes(store: &Store, keep: impl Fn(&ObjectId) -> bool) -> Result<usize> {
    let file = notes_file(store);
    if !file.path().exists() {
        return Ok(0);
    }
    let lock = file.lock()?;
    let lines: Vec<NoteLine> = file.read()?;
    let kept: Vec<&NoteLine> = lines.iter().filter(|l| ObjectId::parse(&l.id).is_ok_and(|id| keep(&id))).collect();
    let dropped = lines.len() - kept.len();
    if dropped > 0 {
        file.rewrite(&lock, &kept, Durability::Fsync)?;
    }
    Ok(dropped)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SESSION: &str = "0b7e9d3a-1c2f-4e5a-9b8c-7d6e5f4a3b2c";

    #[test]
    fn session_refs_need_a_uuid() {
        assert!(RefName::session(SESSION).is_ok());
        for bad in ["", "../../etc", "s", "0b7e9d3a-1c2f-4e5a-9b8c-7d6e5f4a3b2c/../x"] {
            assert!(RefName::session(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn a_ref_moves_only_from_where_the_caller_saw_it_and_logs_each_move() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::at(dir.path());
        let name = RefName::session(SESSION).unwrap();
        let (a, b) = (ObjectId([1; 32]), ObjectId([2; 32]));

        update(&store, &name, None, a, "snapshot").unwrap();
        assert!(update(&store, &name, None, b, "snapshot").is_err(), "stale expectation");
        update(&store, &name, Some(a), b, "snapshot").unwrap();
        assert_eq!(read(&store, &name).unwrap(), Some(b));
        assert_eq!(all(&store).unwrap(), vec![(name.as_str().to_string(), b)]);

        let news: Vec<String> = reflog(&store).unwrap().into_iter().map(|e| e.new).collect();
        assert_eq!(news, vec![a.hex(), b.hex()]);
        assert_eq!(std::fs::read_dir(dir.path().join(".ambits")).unwrap().count(), 2, "one reflog and its lock");
    }

    /// Two writers racing from the same tip: exactly one moves the ref.
    #[test]
    fn racing_writers_move_a_ref_once() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        let wins: usize = (1..=6u8)
            .map(|n| {
                let root = root.clone();
                std::thread::spawn(move || {
                    let store = Store::at(&root);
                    update(&store, &RefName::session(SESSION).unwrap(), None, ObjectId([n; 32]), "snapshot").is_ok()
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|t| usize::from(t.join().unwrap()))
            .sum();
        assert_eq!(wins, 1);
        assert_eq!(reflog(&Store::at(&root)).unwrap().len(), 1);
    }

    /// Expiry drops old moves but never where a ref points.
    #[test]
    fn expiry_keeps_every_refs_tip() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::at(dir.path());
        let name = RefName::session(SESSION).unwrap();
        let entry = |old: Option<u8>, new: u8| ReflogEntry {
            name: name.as_str().to_string(),
            old: old.map(|o| ObjectId([o; 32]).hex()),
            new: ObjectId([new; 32]).hex(),
            secs: 1_000,
            time: "1970-01-01T00:16:40Z".into(),
            action: "snapshot".into(),
        };
        let file = reflog_file(&store);
        let lock = file.lock().unwrap();
        file.append(&lock, &[entry(None, 1), entry(Some(1), 2)], Durability::NoSync).unwrap();
        drop(lock);
        assert_eq!(expire_reflogs(&store, REFLOG_EXPIRY).unwrap(), 1, "both are old; the tip's stays");
        assert_eq!(read(&store, &name).unwrap(), Some(ObjectId([2; 32])));
    }

    #[test]
    fn notes_fold_by_snapshot_and_prune() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::at(dir.path());
        let (a, b) = (ObjectId([1; 32]), ObjectId([2; 32]));
        write_note(&store, &a, Some("first")).unwrap();
        write_note(&store, &b, None).unwrap();
        assert_eq!(read_note(&store, &a).unwrap().message.as_deref(), Some("first"));
        assert_eq!(prune_notes(&store, |id| *id == b).unwrap(), 1);
        assert!(read_note(&store, &a).is_none());
        assert!(read_note(&store, &b).is_some());
    }

    /// A store an older ambits laid out is refused, not misread.
    #[test]
    fn the_old_layout_is_refused_with_what_to_delete() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join(".ambits/refs/sessions")).unwrap();
        let store = Store::at(dir.path());
        let err = read(&store, &RefName::session(SESSION).unwrap()).unwrap_err().to_string();
        assert!(err.contains("older ambits") && err.contains(".ambits/refs"), "{err}");
        let objects = dir.path().join(".ambits/objects/ab");
        std::fs::create_dir_all(&objects).unwrap();
        std::fs::write(objects.join("cd.json"), "{}").unwrap();
        assert!(crate::objects::gc::gc(&store, Duration::ZERO, REFLOG_EXPIRY).is_err(), "gc must not see no refs and collect everything");
        assert!(objects.join("cd.json").exists());
    }
}
