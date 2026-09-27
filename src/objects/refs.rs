//! Session refs, their lock, the reflog and notes (spec §8).
//!
//! ```text
//! .ambits/refs/sessions/<session-id>       tip snapshot id
//! .ambits/logs/refs/sessions/<session-id>  reflog, one JSON line per move
//! .ambits/notes/<snapshot-id>.json         time, message, version
//! ```

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use color_eyre::eyre::{bail, Result, WrapErr};
use serde::{Deserialize, Serialize};

use super::store::{create_private_dir, random_token, write_atomic, Store};
use super::ObjectId;

/// Default age after which reflog entries stop protecting objects from gc
/// (§8); `ambits gc --reflog-expiry-days` overrides it.
pub const REFLOG_EXPIRY: Duration = Duration::from_secs(90 * 24 * 60 * 60);

/// A validated ref name, relative to `.ambits/`, e.g. `refs/sessions/<id>`.
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

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The session this ref belongs to.
    pub fn session_id(&self) -> &str {
        self.0.rsplit('/').next().unwrap_or(&self.0)
    }
}

fn ref_path(store: &Store, name: &RefName) -> PathBuf {
    store.root().join(name.as_str())
}

fn reflog_path(store: &Store, name: &RefName) -> PathBuf {
    store.root().join("logs").join(name.as_str())
}

/// The snapshot `name` points at, if any.
pub fn read(store: &Store, name: &RefName) -> Result<Option<ObjectId>> {
    match fs::read_to_string(ref_path(store, name)) {
        Ok(s) => Ok(Some(ObjectId::parse(s.trim())?)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).wrap_err_with(|| format!("reading {}", name.as_str())),
    }
}

/// Every ref under `.ambits/refs/`, local and remote-tracking, with its tip.
pub fn all(store: &Store) -> Vec<(String, ObjectId)> {
    let mut out = Vec::new();
    let mut stack = vec![store.root().join("refs")];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).into_iter().flatten().flatten() {
            let path = entry.path();
            let Ok(meta) = fs::symlink_metadata(&path) else { continue };
            if meta.is_dir() {
                stack.push(path);
            } else if meta.is_file()
                && path.extension().is_none_or(|e| e != "lock")
                && !entry.file_name().to_string_lossy().starts_with('.')
            {
                if let Some(id) = fs::read_to_string(&path).ok().and_then(|s| ObjectId::parse(s.trim()).ok()) {
                    let name = path.strip_prefix(store.root()).unwrap_or(&path).to_string_lossy().replace('\\', "/");
                    out.push((name, id));
                }
            }
        }
    }
    out.sort();
    out
}

/// Held while a ref is being moved: `<ref>.lock`, created exclusively and
/// holding a pid, start time and random token — no host (D16). Removed on
/// drop.
struct RefLock(PathBuf);

impl RefLock {
    fn acquire(path: &Path) -> Result<Self> {
        let lock = path.with_extension("lock");
        create_private_dir(lock.parent().unwrap_or(path))?;
        let mut options = fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = match options.open(&lock) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                let holder = fs::read_to_string(&lock).unwrap_or_default();
                bail!("{} is locked by another ambits process ({}); if none is running, delete the lock file", lock.display(), holder.trim());
            }
            Err(e) => return Err(e).wrap_err_with(|| format!("locking {}", lock.display())),
        };
        let body = serde_json::json!({"pid": std::process::id(), "start": now_secs(), "token": random_token()});
        file.write_all(body.to_string().as_bytes())?;
        Ok(Self(lock))
    }
}

impl Drop for RefLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

/// Move `name` from `expected` to `new`, and record the move in the reflog.
///
/// The ref is re-read after locking: if it no longer holds `expected`,
/// another process moved it and nothing is written.
pub fn update(store: &Store, name: &RefName, expected: Option<ObjectId>, new: ObjectId, action: &str) -> Result<()> {
    let path = ref_path(store, name);
    let _lock = RefLock::acquire(&path)?;
    let current = read(store, name)?;
    if current != expected {
        bail!("{} moved while this snapshot was being made; run it again", name.as_str());
    }
    write_atomic(&path, format!("{new}\n").as_bytes())?;
    super::store::fsync_dir(path.parent().unwrap_or(&path))?;
    append_reflog(store, name, &ReflogEntry { old: current.map(|c| c.hex()), new: new.hex(), secs: now_secs(), time: now_rfc3339(), action: action.to_string() })
}

/// One reflog line.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReflogEntry {
    pub old: Option<String>,
    pub new: String,
    /// Seconds since the epoch, for expiry.
    pub secs: u64,
    pub time: String,
    pub action: String,
}

fn append_reflog(store: &Store, name: &RefName, entry: &ReflogEntry) -> Result<()> {
    let path = reflog_path(store, name);
    create_private_dir(path.parent().unwrap_or(&path))?;
    let mut options = fs::OpenOptions::new();
    options.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&path).wrap_err_with(|| format!("opening {}", path.display()))?;
    file.write_all(format!("{}\n", serde_json::to_string(entry)?).as_bytes())?;
    file.sync_all()?;
    Ok(())
}

/// Every reflog, as `(file, entries)`. Unparseable lines are skipped.
pub fn reflogs(store: &Store) -> Vec<(PathBuf, Vec<ReflogEntry>)> {
    let mut out = Vec::new();
    let mut stack = vec![store.root().join("logs")];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).into_iter().flatten().flatten() {
            let path = entry.path();
            match fs::symlink_metadata(&path) {
                Ok(m) if m.is_dir() => stack.push(path),
                Ok(m) if m.is_file() && !entry.file_name().to_string_lossy().starts_with('.') => {
                    let entries = fs::read_to_string(&path)
                        .unwrap_or_default()
                        .lines()
                        .filter_map(|l| serde_json::from_str(l).ok())
                        .collect();
                    out.push((path, entries));
                }
                _ => {}
            }
        }
    }
    out
}

/// Drop reflog entries older than `expiry`, rewriting each log atomically.
pub fn expire_reflogs(store: &Store, expiry: Duration) -> Result<usize> {
    let cutoff = now_secs().saturating_sub(expiry.as_secs());
    let mut dropped = 0;
    for (path, entries) in reflogs(store) {
        let kept: Vec<&ReflogEntry> = entries.iter().filter(|e| e.secs >= cutoff).collect();
        if kept.len() == entries.len() {
            continue;
        }
        dropped += entries.len() - kept.len();
        let body: String = kept.iter().map(|e| format!("{}\n", serde_json::to_string(e).unwrap_or_default())).collect();
        write_atomic(&path, body.as_bytes())?;
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

pub fn note_path(store: &Store, id: &ObjectId) -> PathBuf {
    store.root().join("notes").join(format!("{id}.json"))
}

pub fn write_note(store: &Store, id: &ObjectId, message: Option<&str>) -> Result<()> {
    let note = Note { time: now_rfc3339(), message: message.map(String::from), version: env!("CARGO_PKG_VERSION").to_string() };
    write_atomic(&note_path(store, id), serde_json::to_string(&note)?.as_bytes())
}

pub fn read_note(store: &Store, id: &ObjectId) -> Option<Note> {
    serde_json::from_str(&fs::read_to_string(note_path(store, id)).ok()?).ok()
}

pub fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// Now, as RFC 3339 UTC to the second.
pub fn now_rfc3339() -> String {
    rfc3339(now_secs())
}

/// `secs` since the epoch as `YYYY-MM-DDTHH:MM:SSZ` (Howard Hinnant's
/// civil-from-days, so no date dependency).
pub fn rfc3339(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z", rem / 3600, rem % 3600 / 60, rem % 60)
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
        assert_eq!(all(&store), vec![(name.as_str().to_string(), b)]);

        let logs = reflogs(&store);
        assert_eq!(logs.len(), 1);
        let news: Vec<&str> = logs[0].1.iter().map(|e| e.new.as_str()).collect();
        assert_eq!(news, vec![a.hex(), b.hex()]);
    }

    #[test]
    fn a_held_lock_refuses_a_second_writer() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::at(dir.path());
        let name = RefName::session(SESSION).unwrap();
        let _held = RefLock::acquire(&ref_path(&store, &name)).unwrap();
        assert!(update(&store, &name, None, ObjectId([1; 32]), "snapshot").is_err());
    }

    #[test]
    fn rfc3339_formats_known_instants() {
        assert_eq!(rfc3339(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(rfc3339(1_790_000_000), "2026-09-21T14:13:20Z");
    }
}
