//! Which git commit an agent's write landed in (spec §3.2).
//!
//! Resolved lazily, per *unit* of a write — each `(op, symbol, hash)` of a
//! symbol-level write, or the file of a file-level one — because one write's
//! symbols can land in different commits (`git add -p`). Results are kept in
//! the **links index** (`.ambits/links/`), re-checked for reachability before
//! use so amends and rebases re-resolve; misses go to the local-only
//! **never-landed** cache, valid until any branch moves.
//!
//! Nothing here reads file contents into anything persisted: blobs are
//! parsed or hashed and dropped (§9.6).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use color_eyre::eyre::Result;
use serde::{Deserialize, Serialize};

use crate::git::{git, is_commit_id, Repo};
use crate::objects::store::{create_private_dir, write_atomic};
use crate::objects::{b3, valid_record_path};
use crate::parser::ParserRegistry;
use crate::writes::{Level, WriteRecord};

/// What landed: one part of one write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unit {
    /// The symbol id, or the file path for a file-level write.
    pub target: String,
    pub proof: Proof,
}

/// How a commit is recognized as containing the unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Proof {
    /// A symbol with this name path and content hash.
    Symbol { name_path: String, hash: String },
    /// A file whose raw bytes hash to this (`fh`).
    FileHash(String),
    /// Nothing to compare: the first commit touching the file after the
    /// write, labelled unverified.
    None,
}

impl Unit {
    fn hash(&self) -> Option<&str> {
        match &self.proof {
            Proof::Symbol { hash, .. } | Proof::FileHash(hash) => Some(hash),
            Proof::None => None,
        }
    }
}

/// The units of `write`: its written symbols, else its file.
///
/// Removed symbols are not units — there is no content to find — and
/// neither is anything the write did outside symbols; the file-level
/// fallback covers a write that names no symbol at all.
pub fn units(write: &WriteRecord) -> Vec<Unit> {
    let prefix = format!("{}::", write.file);
    if write.level == Level::Symbol && !write.syms.is_empty() {
        return write
            .syms
            .iter()
            .map(|(id, hash)| Unit {
                target: id.clone(),
                proof: Proof::Symbol {
                    name_path: id.strip_prefix(&prefix).unwrap_or(id).to_string(),
                    hash: hash.clone(),
                },
            })
            .collect();
    }
    let proof = match &write.fh {
        Some(fh) => Proof::FileHash(fh.clone()),
        None => Proof::None,
    };
    vec![Unit { target: write.file.clone(), proof }]
}

/// Where a unit landed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Link {
    pub op: String,
    pub target: String,
    #[serde(default)]
    pub hash: Option<String>,
    pub commit: String,
    /// `false` when nothing could be compared (a file-level `Edit`): the
    /// commit is only the first to touch the file after the write.
    pub verified: bool,
    /// The file's path in that commit, relative to the project; differs
    /// from the write's when it was renamed first.
    pub path: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Landing {
    Landed(Link),
    /// In no commit reachable from a branch or `HEAD`.
    Uncommitted,
}

/// One commit of the log, with the paths it touched (repo-relative).
#[derive(Debug, Clone)]
struct Commit {
    sha: String,
    time: u64,
    changes: Vec<Change>,
}

#[derive(Debug, Clone)]
enum Change {
    /// Added, modified or type-changed.
    Touched(String),
    Deleted(String),
    Renamed { from: String, to: String },
}

/// Resolves units against one repository, caching the log per start time.
pub struct Resolver {
    repo: Repo,
    state: PathBuf,
    registry: ParserRegistry,
    logs: HashMap<u64, Vec<Commit>>,
    tips: String,
    /// What each `(commit, path)` contains, computed once: many units share
    /// a file, and a refresh walks the same commits for all of them.
    contents: HashMap<(String, String), Contents>,
}

/// A committed file's identity, as units are matched against it.
#[derive(Default)]
struct Contents {
    file_hash: Option<String>,
    /// `(name_path, content hash)` of every symbol.
    symbols: std::collections::HashSet<(String, String)>,
}

impl Resolver {
    /// `None` outside a git repository or before its first commit.
    pub fn new(project_root: &Path) -> Option<Self> {
        let repo = Repo::discover(project_root).filter(|r| r.head.is_some())?;
        let tips = tips_digest(project_root);
        Some(Self {
            repo,
            state: project_root.join(crate::state_dir::STATE_DIR),
            registry: ParserRegistry::new(),
            logs: HashMap::new(),
            tips,
            contents: HashMap::new(),
        })
    }

    fn dir(&self) -> &Path {
        self.repo.dir()
    }

    /// Where `unit` of `write` landed: from the links index when the commit
    /// is still reachable, from the never-landed cache while no branch has
    /// moved, else resolved now and cached.
    pub fn landing(&mut self, write: &WriteRecord, unit: &Unit) -> Result<Landing> {
        let key = link_key(&write.op, &unit.target, unit.hash());
        let link_path = self.state.join("links").join(format!("{key}.json"));
        let never_path = self.state.join("cache").join("never-landed").join(format!("{key}.json"));

        if let Some(link) = std::fs::read_to_string(&link_path).ok().and_then(|s| serde_json::from_str::<Link>(&s).ok()) {
            if is_commit_id(&link.commit) && self.reachable(&link.commit) {
                return Ok(Landing::Landed(link));
            }
            // Amended or rebased away: resolve again.
            let _ = std::fs::remove_file(&link_path);
        }
        if std::fs::read_to_string(&never_path).is_ok_and(|tips| tips.trim() == self.tips) {
            return Ok(Landing::Uncommitted);
        }

        let landing = self.resolve(write, unit);
        match &landing {
            Landing::Landed(link) => {
                create_private_dir(link_path.parent().expect("has parent"))?;
                write_atomic(&link_path, serde_json::to_string(link)?.as_bytes())?;
                let _ = std::fs::remove_file(&never_path);
            }
            Landing::Uncommitted => {
                create_private_dir(never_path.parent().expect("has parent"))?;
                write_atomic(&never_path, format!("{}\n", self.tips).as_bytes())?;
            }
        }
        Ok(landing)
    }

    /// Resolve without caches (§3.2): walk the commits since a day before
    /// the write, oldest first, following the file through renames, and
    /// take the first whose version of it contains the unit.
    fn resolve(&mut self, write: &WriteRecord, unit: &Unit) -> Landing {
        let Some(written_at) = crate::objects::refs::parse_rfc3339(&write.t) else {
            return Landing::Uncommitted;
        };
        if !valid_record_path(&write.file) {
            return Landing::Uncommitted;
        }
        let since = written_at.saturating_sub(24 * 60 * 60);
        let commits = self.log(since);
        let prefix = self.repo.prefix.clone();
        let mut path = format!("{prefix}{}", write.file);

        for commit in commits {
            let mut touched = false;
            let mut deleted = false;
            for change in &commit.changes {
                match change {
                    Change::Touched(p) if *p == path => touched = true,
                    Change::Renamed { from, to } if *from == path => {
                        path = to.clone();
                        touched = true;
                    }
                    Change::Deleted(p) if *p == path => deleted = true,
                    _ => {}
                }
            }
            // A rename that also changed most of the file is no rename to
            // git: it reports a deletion and an addition. Look for the unit
            // in what the commit added, and follow it there.
            if deleted && !touched && !matches!(unit.proof, Proof::None) {
                let added: Vec<String> = commit
                    .changes
                    .iter()
                    .filter_map(|c| match c {
                        Change::Touched(p) => Some(p.clone()),
                        _ => None,
                    })
                    .collect();
                for candidate in added {
                    let Some(rel) = candidate.strip_prefix(&prefix).map(str::to_string) else { continue };
                    if self.contains(&commit.sha, &candidate, &rel, unit) {
                        return Landing::Landed(self.link(write, unit, &commit.sha, rel));
                    }
                }
                continue;
            }
            if !touched {
                continue;
            }
            let Some(rel) = path.strip_prefix(&prefix).map(str::to_string) else {
                // Renamed out of the project: nothing further to find.
                return Landing::Uncommitted;
            };
            let found = match &unit.proof {
                Proof::None => commit.time >= written_at,
                _ => self.contains(&commit.sha, &path, &rel, unit),
            };
            if found {
                return Landing::Landed(self.link(write, unit, &commit.sha, rel));
            }
        }
        Landing::Uncommitted
    }

    /// Whether commit `sha`'s version of `path` (repo-relative; `rel` is
    /// the same file relative to the project) contains `unit`.
    fn contains(&mut self, sha: &str, path: &str, rel: &str, unit: &Unit) -> bool {
        let key = (sha.to_string(), path.to_string());
        if !self.contents.contains_key(&key) {
            let contents = self.blob(sha, path).map(|b| self.read_contents(rel, &b)).unwrap_or_default();
            self.contents.insert(key.clone(), contents);
        }
        let contents = &self.contents[&key];
        match &unit.proof {
            Proof::FileHash(fh) => contents.file_hash.as_deref() == Some(fh.as_str()),
            Proof::Symbol { name_path, hash } => contents.symbols.contains(&(name_path.clone(), hash.clone())),
            Proof::None => false,
        }
    }

    fn link(&self, write: &WriteRecord, unit: &Unit, sha: &str, rel: String) -> Link {
        Link {
            op: write.op.clone(),
            target: unit.target.clone(),
            hash: unit.hash().map(String::from),
            commit: sha.to_string(),
            verified: !matches!(unit.proof, Proof::None),
            path: rel,
        }
    }

    /// The file hash of `blob`, and every symbol in it when parsed as
    /// project file `rel`. The blob itself is dropped (§9.6).
    fn read_contents(&self, rel: &str, blob: &[u8]) -> Contents {
        let mut out = Contents { file_hash: Some(b3(blake3::hash(blob).as_bytes())), ..Default::default() };
        let path = Path::new(rel);
        let parsed = std::str::from_utf8(blob)
            .ok()
            .zip(self.registry.parser_for(path))
            .and_then(|(source, parser)| parser.parse_file(path, source).ok());
        if let Some(parsed) = parsed {
            let prefix = format!("{rel}::");
            let mut stack: Vec<&crate::symbols::SymbolNode> = parsed.symbols.iter().collect();
            while let Some(s) = stack.pop() {
                if let Some(name_path) = s.id.strip_prefix(&prefix) {
                    out.symbols.insert((name_path.to_string(), b3(&s.content_hash)));
                }
                stack.extend(s.children.iter());
            }
        }
        out
    }

    /// `path` (repo-relative) as committed in `sha`.
    fn blob(&self, sha: &str, path: &str) -> Option<Vec<u8>> {
        if !is_commit_id(sha) || !valid_record_path(path) {
            return None;
        }
        git(self.dir(), &["cat-file", "blob", "--end-of-options", &format!("{sha}:{path}")])
    }

    /// Whether `sha` is reachable from any local branch or `HEAD`.
    fn reachable(&self, sha: &str) -> bool {
        let on_branch = git(self.dir(), &["for-each-ref", "--count=1", "--format=%(refname)", "--contains", sha, "refs/heads"])
            .is_some_and(|out| !out.is_empty());
        on_branch || git(self.dir(), &["merge-base", "--is-ancestor", "--end-of-options", sha, "HEAD"]).is_some()
    }

    /// Commits reachable from any branch or `HEAD` since `since`, oldest
    /// first, with the paths each touched and renames detected. Not limited
    /// to the write's path: a pathspec would hide the rename that moved it
    /// (spec §3.2, amended).
    fn log(&mut self, since: u64) -> Vec<Commit> {
        if let Some(cached) = self.logs.get(&since) {
            return cached.clone();
        }
        let since_arg = format!("--since={}", crate::objects::refs::rfc3339(since));
        let out = git(
            self.dir(),
            &[
                "log", "--branches", "HEAD", "--full-history", "-M", "--name-status", "-z",
                "--format=commit %H %ct", "--reverse", &since_arg,
            ],
        )
        .unwrap_or_default();
        let commits = parse_log(&out);
        self.logs.insert(since, commits.clone());
        commits
    }
}

/// Parse `git log --name-status -z --format='commit %H %ct'`.
fn parse_log(out: &[u8]) -> Vec<Commit> {
    let mut commits: Vec<Commit> = Vec::new();
    let mut tokens = out.split(|&b| b == 0).map(|t| String::from_utf8_lossy(t).trim_start_matches('\n').to_string());
    while let Some(token) = tokens.next() {
        if let Some(rest) = token.strip_prefix("commit ") {
            let mut parts = rest.split(' ');
            let sha = parts.next().unwrap_or_default().to_string();
            let time = parts.next().and_then(|t| t.parse().ok()).unwrap_or(0);
            if is_commit_id(&sha) {
                commits.push(Commit { sha, time, changes: Vec::new() });
            }
            continue;
        }
        let Some(commit) = commits.last_mut() else { continue };
        let change = match token.as_bytes().first() {
            Some(b'R') | Some(b'C') => {
                let (from, to) = (tokens.next().unwrap_or_default(), tokens.next().unwrap_or_default());
                if token.starts_with('R') {
                    Change::Renamed { from, to }
                } else {
                    Change::Touched(to)
                }
            }
            Some(b'D') => Change::Deleted(tokens.next().unwrap_or_default()),
            Some(b'A' | b'M' | b'T') => Change::Touched(tokens.next().unwrap_or_default()),
            _ => continue,
        };
        commit.changes.push(change);
    }
    commits
}

/// Identifies the state of every branch and `HEAD`: the never-landed cache
/// is valid only while this is unchanged.
fn tips_digest(dir: &Path) -> String {
    let heads = git(dir, &["for-each-ref", "--format=%(refname) %(objectname)", "refs/heads"]).unwrap_or_default();
    let head = git(dir, &["rev-parse", "--verify", "--quiet", "HEAD"]).unwrap_or_default();
    let mut h = blake3::Hasher::new();
    h.update(&heads);
    h.update(b"\0");
    h.update(&head);
    h.finalize().to_hex().to_string()
}

/// The file name of a unit's entry: a hash of `(op, target, hash)`, so
/// untrusted strings never become paths.
fn link_key(op: &str, target: &str, hash: Option<&str>) -> String {
    let mut h = blake3::Hasher::new();
    for part in [op, target, hash.unwrap_or("")] {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part.as_bytes());
    }
    h.finalize().to_hex().to_string()
}

/// What [`refresh`] did.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RefreshStats {
    pub landed: usize,
    pub uncommitted: usize,
    /// The time budget ran out before every write was looked at.
    pub stopped_early: bool,
}

/// Resolve every unit of every write from the last `window`, across all
/// sessions, within `budget` — for the post-commit hook, so later
/// `touched` calls answer from the links index.
pub fn refresh(project_root: &Path, window: std::time::Duration, budget: std::time::Duration) -> Result<RefreshStats> {
    let started = std::time::Instant::now();
    let mut stats = RefreshStats::default();
    let Some(mut resolver) = Resolver::new(project_root) else { return Ok(stats) };
    let cutoff = crate::objects::refs::now_secs().saturating_sub(window.as_secs());
    let dir = crate::journal::journal_dir(project_root);
    for session in crate::cache::session_ids(&dir) {
        for write in crate::journal::read_session_writes(&dir, &session).into_values() {
            if crate::objects::refs::parse_rfc3339(&write.t).is_none_or(|t| t < cutoff) {
                continue;
            }
            for unit in units(&write) {
                if started.elapsed() > budget {
                    stats.stopped_early = true;
                    return Ok(stats);
                }
                match resolver.landing(&write, &unit)? {
                    Landing::Landed(_) => stats.landed += 1,
                    Landing::Uncommitted => stats.uncommitted += 1,
                }
            }
        }
    }
    Ok(stats)
}

/// Where the units of a write landed, for display.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "state")]
pub enum Landed {
    /// Every unit is in a commit it was verified against.
    Verified { commits: Vec<String> },
    /// The first commit touching the file after the write; nothing could be
    /// compared (a file-level `Edit`).
    Unverified { commits: Vec<String> },
    /// Some units landed, others are in no commit yet.
    Partial { commits: Vec<String> },
    Uncommitted,
    /// Not a git repository, or no commits yet.
    NoRepository,
}

/// Resolve every unit of `write` (those `keep` selects) and summarize.
pub fn landed(resolver: Option<&mut Resolver>, write: &WriteRecord, keep: &dyn Fn(&Unit) -> bool) -> Result<Landed> {
    let Some(resolver) = resolver else { return Ok(Landed::NoRepository) };
    let mut commits: Vec<String> = Vec::new();
    let (mut missing, mut unverified) = (0, false);
    for unit in units(write).iter().filter(|u| keep(u)) {
        match resolver.landing(write, unit)? {
            Landing::Landed(link) => {
                unverified |= !link.verified;
                if !commits.contains(&link.commit) {
                    commits.push(link.commit);
                }
            }
            Landing::Uncommitted => missing += 1,
        }
    }
    Ok(match (commits.is_empty(), missing > 0, unverified) {
        (true, _, _) => Landed::Uncommitted,
        (false, true, _) => Landed::Partial { commits },
        (false, false, true) => Landed::Unverified { commits },
        (false, false, false) => Landed::Verified { commits },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_log_parses_changes_and_renames() {
        let sha = |c: char| c.to_string().repeat(40);
        let out = format!(
            "commit {} 100\0\0\nA\0a.rs\0commit {} 200\0\0\nR100\0a.rs\0b.rs\0M\0c.rs\0commit {} 300\0\0\nD\0b.rs\0",
            sha('a'),
            sha('b'),
            sha('c')
        );
        let commits = parse_log(out.as_bytes());
        assert_eq!(commits.len(), 3);
        assert_eq!(commits[1].time, 200);
        assert!(matches!(&commits[1].changes[0], Change::Renamed { from, to } if from == "a.rs" && to == "b.rs"));
        assert!(matches!(&commits[1].changes[1], Change::Touched(p) if p == "c.rs"));
        assert!(matches!(&commits[2].changes[0], Change::Deleted(p) if p == "b.rs"));
    }

    #[test]
    fn units_are_symbols_else_the_file() {
        let mut w = WriteRecord {
            op: "toolu_1".into(),
            av: 2,
            a: "a".into(),
            t: "2026-09-26T10:00:00Z".into(),
            tool: "Edit".into(),
            file: "src/a.rs".into(),
            level: Level::Symbol,
            outside_symbols: false,
            syms: vec![("src/a.rs::A/f".into(), "b3:00".into())],
            removed: vec![],
            fh: None,
        };
        assert_eq!(units(&w)[0].proof, Proof::Symbol { name_path: "A/f".into(), hash: "b3:00".into() });
        w.level = Level::File;
        w.syms.clear();
        assert_eq!(units(&w)[0].proof, Proof::None);
        w.fh = Some("b3:11".into());
        assert_eq!(units(&w), vec![Unit { target: "src/a.rs".into(), proof: Proof::FileHash("b3:11".into()) }]);
    }
}
