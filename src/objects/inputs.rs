//! What a snapshot's id is derived from (spec §6.1, D8, D17), and the
//! dirty state it records (§7).
//!
//! Inputs are gathered without reading the objects they determine, so the
//! no-op rule (§6.4) can decide "nothing changed" before writing anything.

use std::path::Path;

use color_eyre::eyre::Result;
use serde_json::{json, Value};

use super::sync_ignore::SyncIgnore;
use super::{b3, canonical, normalize_path, ObjectId};
use crate::git::Repo;
use crate::symbols::ProjectTree;

/// Which backend produced the tree.
pub enum Backend<'a> {
    TreeSitter(&'a crate::parser::ParserRegistry),
    /// Serena's cached symbols; the fingerprint identifies the cache. Such
    /// snapshots are not reproducible from the commit (§6.4).
    Serena { fingerprint: String },
}

impl Backend<'_> {
    /// The `parsers` input: backend, grammar versions and symbol schemas.
    pub fn parsers(&self) -> Vec<String> {
        match self {
            Backend::TreeSitter(registry) => {
                let mut out = vec![format!("tree-sitter@{}", crate::parser::grammar_version("tree-sitter"))];
                out.extend(registry.identities());
                out.sort();
                out
            }
            Backend::Serena { fingerprint } => vec![format!("serena@{fingerprint}")],
        }
    }
}

/// A snapshot's inputs (§6.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inputs {
    pub session: String,
    pub journal: [u8; 32],
    /// `HEAD`, or `None` outside a repository or before its first commit.
    pub git: Option<String>,
    pub parsers: Vec<String>,
    pub scan: [u8; 32],
    pub ignore: [u8; 32],
    /// `(path, BLAKE3 of raw bytes)` for every dirty file, sorted by path.
    /// A deleted file's hash is `"deleted"`.
    pub dirty: Vec<(String, String)>,
}

impl Inputs {
    pub fn to_value(&self) -> Value {
        json!({
            "dirty": self.dirty.iter().map(|(p, h)| json!([p, h])).collect::<Vec<_>>(),
            "git": self.git.as_deref().unwrap_or("none"),
            "ignore": b3(&self.ignore),
            "journal": b3(&self.journal),
            "parsers": self.parsers,
            "scan": b3(&self.scan),
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

/// Everything about the environment that goes into [`Inputs`], other than
/// the session and journal.
pub struct Environment {
    pub git: Option<String>,
    pub scan: [u8; 32],
    pub dirty: Vec<(String, String)>,
}

/// Inspect the working tree of `project_root`, whose scan produced `tree`.
///
/// In a repository, the dirty files are the tracked files git reports
/// changed plus the untracked files the scan picked up. Outside one — or
/// before the first commit — every scanned file is dirty (§7).
pub fn environment(project_root: &Path, tree: &ProjectTree, filter: Option<&str>, ignore: &SyncIgnore) -> Result<Environment> {
    let scanned: std::collections::HashSet<String> =
        tree.files.iter().map(|f| normalize_path(&f.file_path.to_string_lossy())).collect();
    let repo = Repo::discover(project_root);
    let status = repo.as_ref().filter(|r| r.head.is_some()).and_then(Repo::status);

    let mut dirty_paths: Vec<String> = Vec::new();
    let mut untracked_ignores: Vec<(String, String)> = Vec::new();
    match &status {
        Some(entries) => {
            for entry in entries {
                let path = normalize_path(&entry.path);
                let name = path.rsplit('/').next().unwrap_or(&path);
                if entry.untracked && matches!(name, ".ignore" | ".gitignore") {
                    untracked_ignores.push((path.clone(), fingerprint(&project_root.join(&entry.path))));
                }
                if !entry.untracked || scanned.contains(&path) {
                    dirty_paths.push(path);
                }
            }
        }
        None => dirty_paths.extend(scanned.iter().cloned()),
    }

    let mut dirty: Vec<(String, String)> = dirty_paths
        .into_iter()
        .filter(|p| !ignore.is_ignored(p))
        .map(|p| {
            let hash = fingerprint(&project_root.join(&p));
            (p, hash)
        })
        .collect();
    dirty.sort();
    dirty.dedup();
    untracked_ignores.sort();

    let file_hash = |p: Option<std::path::PathBuf>| -> Value {
        match p.and_then(|p| std::fs::read(p).ok()) {
            Some(bytes) => json!(b3(blake3::hash(&bytes).as_bytes())),
            None => Value::Null,
        }
    };
    let scan_value = json!({
        "filter": filter,
        "global_excludes": file_hash(repo.as_ref().and_then(Repo::global_excludes)),
        "info_exclude": file_hash(repo.as_ref().and_then(|r| r.git_path("info/exclude"))),
        "untracked_ignores": untracked_ignores.iter().map(|(p, h)| json!([p, h])).collect::<Vec<_>>(),
        // The walker's own settings (`parser::walk_files` defaults).
        "walker": "hidden=skip ignore=on git_ignore=on git_global=on git_exclude=on",
    });
    let mut h = blake3::Hasher::new();
    h.update(b"ambits-scan v1\0");
    h.update(&canonical::to_bytes(&scan_value)?);

    Ok(Environment {
        git: repo.and_then(|r| r.head),
        scan: *h.finalize().as_bytes(),
        dirty,
    })
}

/// BLAKE3 of a file's raw bytes (§7): `"deleted"` when it is gone, and for
/// anything that is not a regular file, which is never followed (§9.1).
fn fingerprint(path: &Path) -> String {
    match std::fs::symlink_metadata(path) {
        Ok(m) if m.is_file() => match std::fs::read(path) {
            Ok(bytes) => b3(blake3::hash(&bytes).as_bytes()),
            Err(_) => "unreadable".into(),
        },
        Ok(_) => "not-a-file".into(),
        Err(_) => "deleted".into(),
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
            parsers: vec!["rust:x@1:schema=1".into()],
            scan: [2; 32],
            ignore: [3; 32],
            dirty: vec![("a.rs".into(), "b3:00".into())],
        }
    }

    #[test]
    fn every_input_moves_the_digest() {
        let base = inputs().digest().unwrap();
        let variants: [fn(&mut Inputs); 7] = [
            |i| i.session = "t".into(),
            |i| i.journal = [9; 32],
            |i| i.git = Some("a".repeat(40)),
            |i| i.parsers[0] = "rust:x@1:schema=2".into(),
            |i| i.scan = [9; 32],
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
