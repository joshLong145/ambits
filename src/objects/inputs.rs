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

/// Version of the object encodings (`tree`, `record`). Part of every
/// snapshot's `parsers` input, so a change to how objects are written makes
/// new snapshots rather than colliding with ids made under the old encoding.
pub const OBJECT_FORMAT: &str = "ambits-objects@1";

impl Backend<'_> {
    /// The `parsers` input: object format, backend, grammar versions and
    /// symbol schemas.
    pub fn parsers(&self) -> Vec<String> {
        let mut out = vec![OBJECT_FORMAT.to_string()];
        match self {
            Backend::TreeSitter(registry) => {
                out.push(format!("tree-sitter@{}", crate::parser::grammar_version("tree-sitter")));
                out.extend(registry.identities());
            }
            Backend::Serena { fingerprint } => out.push(format!("serena@{fingerprint}")),
        }
        out.sort();
        out
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
    /// `(normalized path, BLAKE3 of raw bytes)`, sorted: the `dirty` input.
    pub dirty: Vec<(String, String)>,
    /// The bytes each dirty file was fingerprinted from, by its raw
    /// project-relative path (`None` when it is gone or not a regular file).
    /// A snapshot parses dirty files from exactly these bytes, so the tree
    /// and the fingerprints cannot describe two different moments.
    pub contents: Vec<(std::path::PathBuf, Option<Vec<u8>>)>,
}

/// Inspect the working tree of `project_root`, whose scan produced `tree`.
///
/// In a repository, the dirty files are the tracked files git reports
/// changed plus the untracked files the scan picked up. Outside one — or
/// before the first commit — every scanned file is dirty (§7).
///
/// Call it *after* the scan: a file changed during the scan then shows up
/// dirty here, and its symbols are re-parsed from the bytes read now.
pub fn environment(project_root: &Path, tree: &ProjectTree, filter: Option<&str>, ignore: &SyncIgnore) -> Result<Environment> {
    // Scanned files by normalized path, keeping the raw path to open them by
    // (git on Linux hands back names in whatever form they were created).
    let scanned: std::collections::HashMap<String, std::path::PathBuf> = tree
        .files
        .iter()
        .map(|f| (normalize_path(&f.file_path.to_string_lossy()), f.file_path.clone()))
        .collect();
    let repo = Repo::discover(project_root);
    let status = repo.as_ref().filter(|r| r.head.is_some()).and_then(Repo::status);

    let mut candidates: Vec<(String, std::path::PathBuf)> = Vec::new();
    let mut untracked_ignores: Vec<(String, Value)> = Vec::new();
    match &status {
        Some(entries) => {
            for entry in entries {
                let path = normalize_path(&entry.path);
                let name = path.rsplit('/').next().unwrap_or(&path);
                if entry.untracked && matches!(name, ".ignore" | ".gitignore") {
                    untracked_ignores.push((path.clone(), stream_hash(&project_root.join(&entry.path))));
                }
                if !entry.untracked || scanned.contains_key(&path) {
                    candidates.push((path, std::path::PathBuf::from(&entry.path)));
                }
            }
        }
        None => candidates.extend(scanned.iter().map(|(n, raw)| (n.clone(), raw.clone()))),
    }
    candidates.retain(|(path, _)| !ignore.is_ignored(path));
    candidates.sort();
    candidates.dedup_by(|a, b| a.0 == b.0);

    let mut dirty = Vec::with_capacity(candidates.len());
    let mut contents = Vec::with_capacity(candidates.len());
    for (path, raw) in candidates {
        let bytes = super::read_regular(&project_root.join(&raw));
        let hash = match (&bytes, std::fs::symlink_metadata(project_root.join(&raw))) {
            (Some(b), _) => super::file_hash(b),
            (None, Ok(m)) if !m.is_file() => "not-a-file".into(),
            (None, Ok(_)) => "unreadable".into(),
            (None, Err(_)) => "deleted".into(),
        };
        dirty.push((path, hash));
        contents.push((raw, bytes));
    }
    untracked_ignores.sort_by(|a, b| a.0.cmp(&b.0));

    // Ignore files the walker also reads above a project that is a
    // subdirectory of its repository: `git status -- .` does not see them.
    let mut parent_ignores: Vec<Value> = Vec::new();
    if let Some(repo) = repo.as_ref().filter(|r| !r.prefix.is_empty()) {
        let top = repo.top.canonicalize().unwrap_or_else(|_| repo.top.clone());
        for dir in project_root.ancestors().skip(1) {
            let rel = dir.strip_prefix(&top).map(|p| normalize_path(&p.to_string_lossy())).unwrap_or_default();
            for name in [".gitignore", ".ignore"] {
                parent_ignores.push(json!([format!("{rel}/{name}"), stream_hash(&dir.join(name))]));
            }
            if dir == top {
                break;
            }
        }
    }

    let scan_value = json!({
        "filter": filter,
        "global_excludes": repo.as_ref().and_then(Repo::global_excludes).map_or(Value::Null, |p| stream_hash(&p)),
        "info_exclude": repo.as_ref().and_then(|r| r.git_path("info/exclude")).map_or(Value::Null, |p| stream_hash(&p)),
        "parent_ignores": parent_ignores,
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
        contents,
    })
}

/// `b3:` hash of a regular file, streamed, or `null` when there is none.
fn stream_hash(path: &Path) -> Value {
    let Some(file) = std::fs::symlink_metadata(path)
        .ok()
        .filter(|m| m.is_file())
        .and_then(|_| std::fs::File::open(path).ok())
    else {
        return Value::Null;
    };
    let mut h = blake3::Hasher::new();
    match h.update_reader(file) {
        Ok(_) => json!(b3(h.finalize().as_bytes())),
        Err(_) => Value::Null,
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
