//! Loose objects under `.ambits/objects/` (spec §8), and the filesystem
//! primitives every part of the store shares: private permissions, atomic
//! writes, and age refreshes for gc.
//!
//! An object file is the canonical JSON envelope `{"payload":…,"type":…}`.
//! A content-addressed object's id is computed from its type and canonical
//! payload bytes (§5.2), so reading one verifies it; a snapshot's id is
//! derived from its inputs instead (D17) and is verified by recomputing that.

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use color_eyre::eyre::{bail, eyre, Result, WrapErr};
use serde_json::{json, Value};

use super::{canonical, content_id, Kind, ObjectId};

/// Largest object a reader will load (§9.5), checked before reading.
pub const MAX_OBJECT_BYTES: u64 = 16 * 1024 * 1024;

/// The object store of one project.
#[derive(Debug, Clone)]
pub struct Store {
    /// `.ambits/`
    root: PathBuf,
    /// Directories that gained an entry since the last [`Store::sync_dirs`].
    /// A rename is durable only once its directory is synced, so these are
    /// flushed before anything that references the new objects is written.
    written_dirs: std::sync::Arc<std::sync::Mutex<std::collections::BTreeSet<PathBuf>>>,
}

impl Store {
    /// The store of `project_root`. Nothing is created until something is
    /// written.
    pub fn at(project_root: &Path) -> Self {
        Self { root: project_root.join(crate::state_dir::STATE_DIR), written_dirs: Default::default() }
    }

    /// Make every object written so far durable: fsync each directory that
    /// gained one (§8). Called before writing what references them — the
    /// snapshot object, then the ref — so a power loss can never keep a
    /// parent whose children's renames were lost.
    pub fn sync_dirs(&self) -> Result<()> {
        let dirs = std::mem::take(&mut *self.written_dirs.lock().expect("not poisoned"));
        for dir in dirs.iter().chain(std::iter::once(&self.objects())) {
            fsync_dir(dir)?;
        }
        Ok(())
    }

    /// `.ambits/`, under which refs, logs and notes also live.
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn objects(&self) -> PathBuf {
        self.root.join("objects")
    }

    /// Where the object `id` lives: `objects/ab/cdef….json`.
    pub fn path_of(&self, id: &ObjectId) -> PathBuf {
        let hex = id.hex();
        self.objects().join(&hex[..2]).join(format!("{}.json", &hex[2..]))
    }

    pub fn contains(&self, id: &ObjectId) -> bool {
        self.path_of(id).is_file()
    }

    /// Store `payload` as a content-addressed object and return its id.
    ///
    /// An object already present is left as it is — equal ids mean equal
    /// bytes — but its age is refreshed, so a gc running alongside cannot
    /// collect what this snapshot is about to reference (§8).
    pub fn put(&self, kind: Kind, payload: &Value) -> Result<ObjectId> {
        debug_assert!(kind != Kind::Snapshot, "snapshots are stored with their derived id");
        let bytes = canonical::to_bytes(payload)?;
        let id = content_id(kind, &bytes);
        self.put_at(&id, kind, payload)?;
        Ok(id)
    }

    /// Store `payload` under `id`, which the caller derived. A present
    /// object is only refreshed; the caller compares contents where that
    /// matters (a snapshot's `state_digest`, §6.3).
    pub fn put_at(&self, id: &ObjectId, kind: Kind, payload: &Value) -> Result<()> {
        let path = self.path_of(id);
        if path.is_file() {
            refresh_age(&path);
            return Ok(());
        }
        let envelope = canonical::to_bytes(&json!({"payload": payload, "type": kind.name()}))?;
        write_atomic(&path, &envelope)?;
        if let Some(dir) = path.parent() {
            self.written_dirs.lock().expect("not poisoned").insert(dir.to_path_buf());
        }
        Ok(())
    }

    /// Load and verify object `id`, which must be of type `kind`.
    pub fn get(&self, id: &ObjectId, kind: Kind) -> Result<Value> {
        let path = self.path_of(id);
        let meta = fs::symlink_metadata(&path).wrap_err_with(|| format!("object {} is missing", id.short()))?;
        if !meta.is_file() {
            bail!("object {} is not a regular file", id.short());
        }
        if meta.len() > MAX_OBJECT_BYTES {
            bail!("object {} is {} bytes, over the {MAX_OBJECT_BYTES}-byte limit", id.short(), meta.len());
        }
        let bytes = fs::read(&path)?;
        let envelope = canonical::from_bytes(&bytes).wrap_err_with(|| format!("object {}", id.short()))?;
        let found = envelope.get("type").and_then(Value::as_str).and_then(Kind::from_name);
        if found != Some(kind) {
            bail!("object {} is not a {} object", id.short(), kind.name());
        }
        let payload = envelope.get("payload").cloned().ok_or_else(|| eyre!("object {} has no payload", id.short()))?;
        if kind != Kind::Snapshot && content_id(kind, &canonical::to_bytes(&payload)?) != *id {
            bail!("object {} does not match its id", id.short());
        }
        Ok(payload)
    }

    /// Every object on disk, with its path. Unparseable file names are
    /// skipped: they are not ours.
    pub fn list(&self) -> Vec<(ObjectId, PathBuf)> {
        let mut out = Vec::new();
        for fan in fs::read_dir(self.objects()).into_iter().flatten().flatten() {
            let prefix = fan.file_name().to_string_lossy().into_owned();
            for entry in fs::read_dir(fan.path()).into_iter().flatten().flatten() {
                let name = entry.file_name().to_string_lossy().into_owned();
                let Some(rest) = name.strip_suffix(".json") else { continue };
                if let Ok(id) = ObjectId::parse(&format!("{prefix}{rest}")) {
                    out.push((id, entry.path()));
                }
            }
        }
        out
    }
}

/// Create `dir` and any missing parents, private to the user (`0700`, §8).
pub fn create_private_dir(dir: &Path) -> Result<()> {
    if dir.is_dir() {
        return Ok(());
    }
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(dir).wrap_err_with(|| format!("creating {}", dir.display()))
}

/// Write `bytes` to `path` atomically: a private temp file beside it, fsync,
/// rename. A reader sees the old file or the new one, never a torn one.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let dir = path.parent().ok_or_else(|| eyre!("{} has no parent", path.display()))?;
    create_private_dir(dir)?;
    let tmp = dir.join(format!(".tmp-{}", random_token()));
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let result = (|| -> Result<()> {
        let mut file = options.open(&tmp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::rename(&tmp, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result.wrap_err_with(|| format!("writing {}", path.display()))
}

/// Flush a directory's entries to disk, so renames into it survive a power
/// loss. A no-op where directories cannot be opened as files (Windows).
pub fn fsync_dir(dir: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        fs::File::open(dir)
            .and_then(|d| d.sync_all())
            .wrap_err_with(|| format!("syncing {}", dir.display()))?;
    }
    #[cfg(not(unix))]
    let _ = dir;
    Ok(())
}

/// Mark `path` as recently used, for gc's grace period (§8). Best-effort: a
/// failure only makes the object look older than it is.
pub fn refresh_age(path: &Path) {
    if let Ok(file) = fs::OpenOptions::new().write(true).open(path) {
        let _ = file.set_modified(std::time::SystemTime::now());
    }
}

/// 64 unpredictable bits, from the per-process random keys std already
/// seeds for `HashMap` — enough for temp names and lock tokens without a
/// dependency.
pub fn random_token() -> String {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0));
    h.write_u32(std::process::id());
    format!("{:016x}", h.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_then_get_round_trips_and_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::at(dir.path());
        let payload = json!({"entries": [], "n": 3});
        let id = store.put(Kind::Dir, &payload).unwrap();
        assert_eq!(store.put(Kind::Dir, &payload).unwrap(), id);
        assert_eq!(store.get(&id, Kind::Dir).unwrap(), payload);
        assert_eq!(store.list().len(), 1);
    }

    #[test]
    fn a_tampered_or_mistyped_object_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::at(dir.path());
        let id = store.put(Kind::Dir, &json!({"n": 1})).unwrap();
        assert!(store.get(&id, Kind::File).is_err(), "wrong type");

        fs::write(store.path_of(&id), br#"{"payload":{"n":2},"type":"dir"}"#).unwrap();
        assert!(store.get(&id, Kind::Dir).is_err(), "content no longer matches the id");
    }

    #[cfg(unix)]
    #[test]
    fn the_store_is_private() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let store = Store::at(dir.path());
        let id = store.put(Kind::Dir, &json!({})).unwrap();
        let mode = |p: &Path| fs::metadata(p).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode(&store.path_of(&id)), 0o600);
        assert_eq!(mode(store.path_of(&id).parent().unwrap()), 0o700);
    }
}
