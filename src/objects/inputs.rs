//! What a snapshot's id is derived from (spec §6.1, D8, D17), and the
//! dirty state it records (§7).
//!
//! Inputs are gathered without reading the objects they determine, so the
//! no-op rule (§6.4) can decide "nothing changed" before writing anything.
//!
//! A snapshot stores coverage and writes, not the symbol tree: restore
//! checks reads against the tree as it is now, and the tree at snapshot time
//! is the pinned commit's for committed files. So only what shapes the
//! stored objects is an input — the session's journal, the ignore patterns
//! that filter it — plus the git pin and dirty list that say what the
//! working tree was.

use std::path::Path;

use color_eyre::eyre::Result;
use serde_json::json;

use super::sync_ignore::SyncIgnore;
use super::{b3, canonical, normalize_path, ObjectId};
use crate::git::Repo;
use crate::symbols::ProjectTree;

/// Version of the object encodings (`record`, the snapshot payload). An
/// input, so a change to how objects are written makes new snapshots rather
/// than colliding with ids made under the old encoding. 2: snapshots no
/// longer carry a symbol tree.
pub const OBJECT_FORMAT: &str = "ambits-objects@2";

/// A snapshot's inputs (§6.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inputs {
    pub session: String,
    pub journal: [u8; 32],
    /// `HEAD`, or `None` outside a repository or before its first commit.
    pub git: Option<String>,
    /// [`OBJECT_FORMAT`] when the snapshot was made.
    pub format: String,
    pub ignore: [u8; 32],
    /// `(path, BLAKE3 of raw bytes)` for every dirty file, sorted by path.
    /// A deleted file's hash is `"deleted"`.
    pub dirty: Vec<(String, String)>,
}

impl Inputs {
    pub fn to_value(&self) -> serde_json::Value {
        json!({
            "dirty": self.dirty.iter().map(|(p, h)| json!([p, h])).collect::<Vec<_>>(),
            "format": self.format,
            "git": self.git.as_deref().unwrap_or("none"),
            "ignore": b3(&self.ignore),
            "journal": b3(&self.journal),
            "session": self.session,
        })
    }

    /// `inputs_digest` (§6.1).
    pub fn digest(&self) -> Result<[u8; 32]> {
        let mut h = blake3::Hasher::new();
        h.update(b"ambits-inputs v1\0");
        h.update(&canonical::to_bytes(&self.to_value())?);
        Ok(*h.finalize().as_bytes())
    }
}

/// `snapshot_id = BLAKE3("ambits-snapshot v1\0" ‖ inputs_digest ‖ sorted parents)` (D17).
pub fn snapshot_id(inputs_digest: &[u8; 32], parents: &[ObjectId]) -> ObjectId {
    let mut sorted = parents.to_vec();
    sorted.sort();
    let mut h = blake3::Hasher::new();
    h.update(b"ambits-snapshot v1\0");
    h.update(inputs_digest);
    for p in &sorted {
        h.update(&p.0);
    }
    ObjectId(*h.finalize().as_bytes())
}

/// The working tree's state, as a snapshot records it.
pub struct Environment {
    pub git: Option<String>,
    /// `(normalized path, BLAKE3 of raw bytes)`, sorted: the `dirty` input.
    pub dirty: Vec<(String, String)>,
}

/// Inspect the working tree of `project_root`, whose scan produced `tree`.
///
/// In a repository, the dirty files are the tracked files git reports
/// changed plus the untracked files the scan picked up. Outside one — or
/// before the first commit — every scanned file is dirty (§7).
pub fn environment(project_root: &Path, tree: &ProjectTree, ignore: &SyncIgnore) -> Result<Environment> {
    // Scanned files by normalized path, keeping the raw path to open them by
    // (git on Linux hands back names in whatever form they were created).
    let scanned: std::collections::HashMap<String, std::path::PathBuf> = tree
        .files
        .iter()
        .map(|f| (normalize_path(&f.file_path.to_string_lossy()), f.file_path.clone()))
        .collect();
    let repo = Repo::discover(project_root);
    let status = repo.as_ref().filter(|r| r.head.is_some()).and_then(Repo::status);

    let mut candidates: Vec<(String, std::path::PathBuf)> = match &status {
        Some(entries) => entries
            .iter()
            .map(|e| (normalize_path(&e.path), e))
            .filter(|(path, e)| !e.untracked || scanned.contains_key(path))
            .map(|(path, e)| (path, std::path::PathBuf::from(&e.path)))
            .collect(),
        None => scanned.into_iter().collect(),
    };
    candidates.retain(|(path, _)| !ignore.is_ignored(path));
    candidates.sort();
    candidates.dedup_by(|a, b| a.0 == b.0);

    let dirty = candidates.into_iter().map(|(path, raw)| (path, fingerprint(&project_root.join(raw)))).collect();
    Ok(Environment { git: repo.and_then(|r| r.head), dirty })
}

/// BLAKE3 of a file's raw bytes (§7): `"deleted"` when it is gone, and
/// `"not-a-file"` for anything that is not a regular file, which is never
/// followed (§9.1).
fn fingerprint(path: &Path) -> String {
    match super::read_regular(path) {
        Some(bytes) => super::file_hash(&bytes),
        None if std::fs::symlink_metadata(path).is_ok() => "not-a-file".into(),
        None => "deleted".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs() -> Inputs {
        Inputs {
            session: "s".into(),
            journal: [1; 32],
            git: None,
            format: OBJECT_FORMAT.into(),
            ignore: [3; 32],
            dirty: vec![("a.rs".into(), "b3:00".into())],
        }
    }

    #[test]
    fn every_input_moves_the_digest() {
        let base = inputs().digest().unwrap();
        let variants: [fn(&mut Inputs); 6] = [
            |i| i.session = "t".into(),
            |i| i.journal = [9; 32],
            |i| i.git = Some("a".repeat(40)),
            |i| i.format = "ambits-objects@3".into(),
            |i| i.ignore = [9; 32],
            |i| i.dirty[0].1 = "b3:01".into(),
        ];
        for change in variants {
            let mut i = inputs();
            change(&mut i);
            assert_ne!(i.digest().unwrap(), base);
        }
    }

    #[test]
    fn parents_are_order_independent_but_part_of_the_id() {
        let d = inputs().digest().unwrap();
        let (a, b) = (ObjectId([1; 32]), ObjectId([2; 32]));
        assert_eq!(snapshot_id(&d, &[a, b]), snapshot_id(&d, &[b, a]));
        assert_ne!(snapshot_id(&d, &[a]), snapshot_id(&d, &[]));
    }
}
