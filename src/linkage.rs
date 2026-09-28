//! Which git commit an agent's write landed in (spec §3.2).
//!
//! Resolved lazily, per *unit* of a write — each `(op, symbol, hash)` of a
//! symbol-level write, or the file of a file-level one — because one write's
//! symbols can land in different commits (`git add -p`). Results are kept in
//! the **links index** (`.ambits/links.ndjson`), re-checked for reachability
//! before use so amends and rebases re-resolve; misses go to the local-only
//! **never-landed** cache (`.ambits/cache/never-landed.ndjson`), valid until
//! any branch moves. Both are flat files ([`crate::objects::flat`]): read
//! once per [`Resolver`], written when it is dropped.
//!
//! Nothing here reads file contents into anything persisted: blobs are
//! parsed or hashed and dropped (§9.6).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use color_eyre::eyre::Result;
use serde::{Deserialize, Serialize};

use crate::git::{git, is_commit_id, Repo};
use crate::objects::valid_record_path;
use crate::parser::ParserRegistry;
use crate::writes::{FileContents, Level, WriteRecord};

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
    if write.level == Level::Symbol && !write.syms.is_empty() {
        return write
            .syms
            .iter()
            .map(|(id, hash)| Unit {
                target: id.clone(),
                proof: Proof::Symbol {
                    name_path: crate::symbols::split_id(id).1.to_string(),
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

/// How far back `links refresh` (and so the hook) looks by default.
pub const REFRESH_WINDOW_DAYS: u64 = 14;

/// Clock skew allowed between the session log's timestamps and commit times
/// on the same machine: commits this long before a write are still searched.
pub const SKEW_SECS: u64 = 10 * 60;

/// One commit of the log, with the paths it touched (repo-relative).
#[derive(Debug, Clone)]
struct Commit {
    sha: String,
    time: u64,
    changes: Vec<Change>,
}

#[derive(Debug, Clone)]
enum Change {
    Added(String),
    /// Modified or type-changed.
    Modified(String),
    Deleted(String),
    Renamed { from: String, to: String },
}

/// The never-landed cache entry: the branch tips a unit was last searched
/// up to, and the names its file had by then.
#[derive(Debug, Clone)]
struct NeverLanded {
    tips: Vec<String>,
    names: Vec<String>,
}

/// A line of the links index: where a unit landed, or `null` once that
/// commit is gone (amended, rebased away).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LinkLine {
    k: String,
    link: Option<Link>,
}

/// A line of the never-landed cache: a set of branch tips, stored once
/// (`tips` + `list`); a unit searched up to them (`k` + `tips` + `names`);
/// or a unit that has since landed (`k` alone).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct NeverLine {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    k: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tips: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    list: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    names: Option<Vec<String>>,
}

/// Both caches, folded (last line per key wins), and what this resolver
/// has learned since.
#[derive(Debug, Default)]
struct Caches {
    loaded: bool,
    links: HashMap<String, Option<Link>>,
    /// Unit → (tips digest, names); `None` once it landed.
    never: HashMap<String, Option<(String, Vec<String>)>>,
    tip_lists: HashMap<String, Vec<String>>,
    new_links: Vec<LinkLine>,
    new_never: Vec<NeverLine>,
}

fn fold_links(lines: Vec<LinkLine>, into: &mut HashMap<String, Option<Link>>) {
    for l in lines {
        into.insert(l.k, l.link);
    }
}

fn fold_never(lines: Vec<NeverLine>, never: &mut HashMap<String, Option<(String, Vec<String>)>>, tip_lists: &mut HashMap<String, Vec<String>>) {
    for l in lines {
        match l {
            NeverLine { k: None, tips: Some(d), list: Some(list), .. } => {
                tip_lists.insert(d, list);
            }
            NeverLine { k: Some(k), tips: Some(d), names, .. } => {
                never.insert(k, Some((d, names.unwrap_or_default())));
            }
            NeverLine { k: Some(k), tips: None, .. } => {
                never.insert(k, None);
            }
            _ => {}
        }
    }
}

/// A set of branch tips, by content.
fn tips_digest(tips: &[String]) -> String {
    crate::objects::hash_framed("ambits-tips v1", tips.iter().map(|t| t.as_bytes())).to_hex().to_string()
}

/// Resolves units against one repository.
pub struct Resolver {
    repo: Repo,
    state: PathBuf,
    registry: ParserRegistry,
    /// The log from `log_since` on, oldest first: one search serves every
    /// write that starts at or after it.
    log: Vec<Commit>,
    log_since: Option<u64>,
    /// Every branch tip and `HEAD`, sorted and deduplicated.
    tips: Vec<String>,
    /// What each `(commit, path)` contains, computed once: many units share
    /// a file, and a refresh walks the same commits for all of them.
    contents: HashMap<(String, String), FileContents>,
    /// Whether each commit is reachable, asked once: many links share one.
    reachable: HashMap<String, bool>,
    caches: Caches,
}

/// Arguments every `git log` here runs with, overriding user config that
/// would change its output: signatures printed before each commit
/// (`log.showSignature`), paths relative to the current directory
/// (`diff.relative`), colour, external diff drivers.
const LOG_ARGS: &[&str] = &[
    "log", "--no-show-signature", "--no-relative", "--no-color", "--no-ext-diff", "-M", "--name-status", "-z",
    "--format=commit %H %ct", "--reverse",
];

impl Resolver {
    /// `None` outside a git repository or before its first commit.
    pub fn new(project_root: &Path) -> Option<Self> {
        let repo = Repo::discover(project_root).filter(|r| r.head.is_some())?;
        let tips = tips(&repo);
        let state = project_root.join(crate::state_dir::STATE_DIR);
        // A file per entry, as older ambits kept them: caches, so dropped.
        let _ = std::fs::remove_dir_all(state.join("links"));
        let _ = std::fs::remove_dir_all(state.join(crate::state_dir::CACHE).join("never-landed"));
        Some(Self {
            repo,
            state,
            registry: ParserRegistry::new(),
            log: Vec::new(),
            log_since: None,
            tips,
            contents: HashMap::new(),
            reachable: HashMap::new(),
            caches: Caches::default(),
        })
    }

    fn links_file(&self) -> crate::objects::flat::FlatLog {
        crate::objects::flat::FlatLog::at(self.state.join(crate::state_dir::LINKS))
    }

    fn never_file(&self) -> crate::objects::flat::FlatLog {
        crate::objects::flat::FlatLog::at(self.state.join(crate::state_dir::CACHE).join(crate::state_dir::NEVER_LANDED))
    }

    /// Read both caches, once.
    fn load(&mut self) {
        if self.caches.loaded {
            return;
        }
        self.caches.loaded = true;
        fold_links(self.links_file().read(), &mut self.caches.links);
        let lines = self.never_file().read();
        fold_never(lines, &mut self.caches.never, &mut self.caches.tip_lists);
    }

    /// Write what was learned: appended under each file's lock, or, once
    /// most of a file is dead lines, the file rewritten with what is live —
    /// merged with whatever other writers appended meanwhile. Best-effort:
    /// a cache that cannot be written costs speed, never the answer.
    pub fn flush(&mut self) -> Result<()> {
        use crate::objects::flat::worth_compacting;
        use crate::objects::store::Durability;
        let new_links = std::mem::take(&mut self.caches.new_links);
        if !new_links.is_empty() {
            let file = self.links_file();
            let lock = file.lock()?;
            let on_disk: Vec<LinkLine> = file.read();
            let lines = file.line_count() + new_links.len();
            let mut folded = HashMap::new();
            fold_links(on_disk, &mut folded);
            fold_links(new_links.clone(), &mut folded);
            let live: Vec<LinkLine> = folded.into_iter().filter(|(_, l)| l.is_some()).map(|(k, link)| LinkLine { k, link }).collect();
            if worth_compacting(lines, live.len()) {
                file.rewrite(&lock, &live, Durability::NoSync)?;
            } else {
                file.append(&lock, &new_links, Durability::NoSync)?;
            }
        }
        let new_never = std::mem::take(&mut self.caches.new_never);
        if !new_never.is_empty() {
            let file = self.never_file();
            let lock = file.lock()?;
            let lines = file.line_count() + new_never.len();
            let (mut never, mut lists) = (HashMap::new(), HashMap::new());
            fold_never(file.read(), &mut never, &mut lists);
            fold_never(new_never.clone(), &mut never, &mut lists);
            let live: Vec<(String, (String, Vec<String>))> = never.into_iter().filter_map(|(k, v)| Some((k, v?))).collect();
            let used: std::collections::HashSet<&String> = live.iter().map(|(_, (d, _))| d).collect();
            if worth_compacting(lines, live.len() + used.len()) {
                let mut out: Vec<NeverLine> =
                    lists.iter().filter(|(d, _)| used.contains(d)).map(|(d, l)| NeverLine { tips: Some(d.clone()), list: Some(l.clone()), ..Default::default() }).collect();
                out.extend(live.iter().map(|(k, (d, names))| NeverLine { k: Some(k.clone()), tips: Some(d.clone()), names: Some(names.clone()), ..Default::default() }));
                file.rewrite(&lock, &out, Durability::NoSync)?;
            } else {
                file.append(&lock, &new_never, Durability::NoSync)?;
            }
        }
        Ok(())
    }

    fn dir(&self) -> &Path {
        self.repo.dir()
    }

    /// Load the log from `since` on, if what is loaded starts later.
    /// [`refresh`] calls it once with its earliest write, so every write
    /// shares one search.
    pub fn prime(&mut self, since: u64) {
        if self.log_since.is_some_and(|s| s <= since) {
            return;
        }
        let since_arg = format!("--since={}", crate::time::rfc3339(since));
        let mut args: Vec<&str> = LOG_ARGS.to_vec();
        args.extend(["--branches", "HEAD", &since_arg]);
        self.log = parse_log(&git(self.dir(), &args).unwrap_or_default());
        self.log_since = Some(since);
    }

    /// Where `unit` of `write` landed: from the links index when the commit
    /// is still reachable, else resolved and cached. Caching is best-effort:
    /// a store that cannot be written costs speed, never the answer.
    pub fn landing(&mut self, write: &WriteRecord, unit: &Unit) -> Result<Landing> {
        self.load();
        let key = link_key(&write.op, &unit.target, unit.hash());

        if let Some(link) = self.caches.links.get(&key).cloned().flatten() {
            if is_commit_id(&link.commit) && self.reachable(&link.commit) {
                return Ok(Landing::Landed(link));
            }
            // Amended or rebased away: resolve again.
            self.caches.links.insert(key.clone(), None);
            self.caches.new_links.push(LinkLine { k: key.clone(), link: None });
        }

        let cached: Option<NeverLanded> = self.caches.never.get(&key).cloned().flatten().and_then(|(digest, names)| {
            Some(NeverLanded { tips: self.caches.tip_lists.get(&digest)?.clone(), names })
        });
        let (landing, names) = match cached {
            // Nothing has moved since it was last searched.
            Some(c) if c.tips == self.tips => return Ok(Landing::Uncommitted),
            // Search only what is new since then, from the names the file
            // had by then.
            Some(c) => match self.new_commits(&c.tips) {
                Some(commits) => self.search(write, unit, &commits, c.names),
                None => self.resolve(write, unit),
            },
            None => self.resolve(write, unit),
        };
        match &landing {
            Landing::Landed(link) => {
                self.caches.links.insert(key.clone(), Some(link.clone()));
                self.caches.new_links.push(LinkLine { k: key.clone(), link: Some(link.clone()) });
                if matches!(self.caches.never.get(&key), Some(Some(_))) {
                    self.caches.never.insert(key.clone(), None);
                    self.caches.new_never.push(NeverLine { k: Some(key), ..Default::default() });
                }
            }
            Landing::Uncommitted => {
                let digest = tips_digest(&self.tips);
                if !self.caches.tip_lists.contains_key(&digest) {
                    self.caches.tip_lists.insert(digest.clone(), self.tips.clone());
                    self.caches.new_never.push(NeverLine { tips: Some(digest.clone()), list: Some(self.tips.clone()), ..Default::default() });
                }
                self.caches.never.insert(key.clone(), Some((digest.clone(), names.clone())));
                self.caches.new_never.push(NeverLine { k: Some(key), tips: Some(digest), names: Some(names), ..Default::default() });
            }
        }
        Ok(landing)
    }

    /// Resolve from scratch (§3.2): the commits since just before the write.
    fn resolve(&mut self, write: &WriteRecord, unit: &Unit) -> (Landing, Vec<String>) {
        let Some(written_at) = crate::time::parse_rfc3339(&write.t) else {
            return (Landing::Uncommitted, Vec::new());
        };
        let since = written_at.saturating_sub(SKEW_SECS);
        self.prime(since);
        let commits: Vec<Commit> = self.log.iter().filter(|c| c.time >= since).cloned().collect();
        let names = vec![format!("{}{}", self.repo.prefix, write.file)];
        self.search(write, unit, &commits, names)
    }

    /// Commits reachable from the current tips but not from `old`, oldest
    /// first; `None` when an old tip no longer exists.
    fn new_commits(&self, old: &[String]) -> Option<Vec<Commit>> {
        if old.iter().any(|t| !is_commit_id(t)) {
            return None;
        }
        let mut args: Vec<&str> = LOG_ARGS.to_vec();
        args.extend(self.tips.iter().map(String::as_str));
        args.push("--not");
        args.extend(old.iter().map(String::as_str));
        git(self.dir(), &args).map(|out| parse_log(&out))
    }

    /// Walk `commits` oldest first and take the first whose version of the
    /// file contains `unit`. `names` are every name the file is known by
    /// (repo-relative); a rename adds its new name and keeps the old, since
    /// the log mixes branches and a rename on one says nothing about the
    /// others. Returns the names as they stand at the end, for the cache.
    fn search(&mut self, write: &WriteRecord, unit: &Unit, commits: &[Commit], mut names: Vec<String>) -> (Landing, Vec<String>) {
        let written_at = crate::time::parse_rfc3339(&write.t).unwrap_or(0);
        if !valid_record_path(&write.file) {
            return (Landing::Uncommitted, names);
        }
        let prefix = self.repo.prefix.clone();
        // With nothing to compare, following renames could only guess:
        // unverified results stay on the file's own name.
        let follow = !matches!(unit.proof, Proof::None);

        for commit in commits {
            let mut candidates: Vec<String> = Vec::new();
            let mut deleted = false;
            for change in &commit.changes {
                match change {
                    Change::Added(p) | Change::Modified(p) if names.contains(p) => candidates.push(p.clone()),
                    Change::Renamed { from, to } if names.contains(from) || names.contains(to) => {
                        if follow && !names.contains(to) {
                            names.push(to.clone());
                        }
                        candidates.push(to.clone());
                    }
                    Change::Deleted(p) if names.contains(p) => deleted = true,
                    _ => {}
                }
            }
            // A rename that also changed most of the file is no rename to
            // git: it reports a deletion and an addition. The unit may be in
            // what the commit added — only added files, never modified ones.
            if follow && deleted && candidates.is_empty() {
                for change in &commit.changes {
                    if let Change::Added(p) = change {
                        if !names.contains(p) {
                            names.push(p.clone());
                        }
                        candidates.push(p.clone());
                    }
                }
            }
            for path in candidates {
                let Some(rel) = path.strip_prefix(&prefix).map(str::to_string) else { continue };
                let found = match &unit.proof {
                    Proof::None => commit.time >= written_at,
                    _ => self.contains(&commit.sha, &path, &rel, unit),
                };
                if found {
                    return (Landing::Landed(self.link(write, unit, &commit.sha, rel)), names);
                }
            }
        }
        (Landing::Uncommitted, names)
    }

    /// Whether commit `sha`'s version of `path` (repo-relative; `rel` is
    /// the same file relative to the project) contains `unit`.
    fn contains(&mut self, sha: &str, path: &str, rel: &str, unit: &Unit) -> bool {
        let key = (sha.to_string(), path.to_string());
        if !self.contents.contains_key(&key) {
            let contents = self.blob(sha, path).map(|b| FileContents::read(rel, &b, &self.registry)).unwrap_or_default();
            self.contents.insert(key.clone(), contents);
        }
        let contents = &self.contents[&key];
        match &unit.proof {
            // A missing blob reads as empty contents, whose hash is empty.
            Proof::FileHash(fh) => contents.hash == *fh,
            Proof::Symbol { name_path, hash } => contents.has(name_path, hash),
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

    /// `path` (repo-relative) as committed in `sha`.
    fn blob(&self, sha: &str, path: &str) -> Option<Vec<u8>> {
        if !is_commit_id(sha) || !valid_record_path(path) {
            return None;
        }
        git(self.dir(), &["cat-file", "blob", "--end-of-options", &format!("{sha}:{path}")])
    }

    /// Whether `sha` is reachable from any local branch or `HEAD`.
    fn reachable(&mut self, sha: &str) -> bool {
        if let Some(&known) = self.reachable.get(sha) {
            return known;
        }
        let answer = self.ask_reachable(sha);
        self.reachable.insert(sha.to_string(), answer);
        answer
    }

    fn ask_reachable(&self, sha: &str) -> bool {
        let on_branch = git(self.dir(), &["for-each-ref", "--count=1", "--format=%(refname)", "--contains", sha, "refs/heads"])
            .is_some_and(|out| !out.is_empty());
        on_branch || git(self.dir(), &["merge-base", "--is-ancestor", "--end-of-options", sha, "HEAD"]).is_some()
    }
}

/// Parse `git log --name-status -z --format='commit %H %ct'`.
///
/// Tokens are NUL-separated. A commit header is the *last line* of its
/// token, so text git prints before it (a signature, should config ask for
/// one anyway) cannot hide the commit. Paths are taken exactly as given:
/// only the status tokens carry the newline that separates a commit's
/// header from its changes.
fn parse_log(out: &[u8]) -> Vec<Commit> {
    let mut commits: Vec<Commit> = Vec::new();
    let mut tokens = out.split(|&b| b == 0).map(|t| String::from_utf8_lossy(t).into_owned());
    while let Some(token) = tokens.next() {
        let token = token.strip_prefix('\n').unwrap_or(&token).to_string();
        if let Some(rest) = token.rsplit('\n').next().and_then(|line| line.strip_prefix("commit ")) {
            let mut parts = rest.split(' ');
            let sha = parts.next().unwrap_or_default().to_string();
            let time = parts.next().and_then(|t| t.parse().ok());
            if let (true, Some(time)) = (is_commit_id(&sha), time) {
                commits.push(Commit { sha, time, changes: Vec::new() });
                continue;
            }
        }
        let Some(commit) = commits.last_mut() else { continue };
        let mut path = || tokens.next().unwrap_or_default();
        let change = match token.as_bytes().first() {
            Some(b'R') => Change::Renamed { from: path(), to: path() },
            Some(b'C') => {
                let _source = path();
                Change::Added(path())
            }
            Some(b'D') => Change::Deleted(path()),
            Some(b'A') => Change::Added(path()),
            Some(b'M' | b'T') => Change::Modified(path()),
            _ => continue,
        };
        commit.changes.push(change);
    }
    commits
}

/// Every local branch tip and `HEAD`, sorted and deduplicated. The
/// never-landed cache is exact while these are unchanged, and otherwise
/// only commits not reachable from the old tips need searching.
fn tips(repo: &Repo) -> Vec<String> {
    let heads = git(repo.dir(), &["for-each-ref", "--format=%(objectname)", "refs/heads"]).unwrap_or_default();
    let mut tips: Vec<String> = String::from_utf8_lossy(&heads)
        .lines()
        .map(str::trim)
        .filter(|t| is_commit_id(t))
        .map(String::from)
        .chain(repo.head.clone())
        .collect();
    tips.sort();
    tips.dedup();
    tips
}

impl Drop for Resolver {
    fn drop(&mut self) {
        if let Err(e) = self.flush() {
            log::warn!(target: "ambits::linkage", "links cache not written: {e}");
        }
    }
}

/// A unit's key in both caches: a hash of `(op, target, hash)`.
fn link_key(op: &str, target: &str, hash: Option<&str>) -> String {
    let parts = [op, target, hash.unwrap_or("")];
    crate::objects::hash_framed("ambits-link v1", parts.iter().map(|p| p.as_bytes())).to_hex().to_string()
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
    let cutoff = crate::time::now_secs().saturating_sub(window.as_secs());
    let dir = crate::journal::journal_dir(project_root);

    // Newest first: if the budget runs out, what goes unresolved is the
    // oldest, least likely to be asked about — and not the same tail
    // starved on every run in session order.
    let mut writes: Vec<(u64, WriteRecord)> = crate::cache::session_ids(&dir)
        .into_iter()
        .flat_map(|session| crate::journal::read_session_writes(&dir, &session).into_values())
        .filter_map(|w| Some((crate::time::parse_rfc3339(&w.t).filter(|t| *t >= cutoff)?, w)))
        .collect();
    writes.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.op.cmp(&b.1.op)));
    // One history search for all of them.
    if let Some(earliest) = writes.iter().map(|(t, _)| *t).min() {
        resolver.prime(earliest.saturating_sub(SKEW_SECS));
    }

    for (_, write) in &writes {
        for unit in units(write) {
            if started.elapsed() > budget {
                stats.stopped_early = true;
                return Ok(stats);
            }
            match resolver.landing(write, &unit)? {
                Landing::Landed(_) => stats.landed += 1,
                Landing::Uncommitted => stats.uncommitted += 1,
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
    /// Some units landed, others are in no commit yet. `unverified` when
    /// any that landed could not be compared.
    Partial { commits: Vec<String>, unverified: bool },
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
        (false, true, unverified) => Landed::Partial { commits, unverified },
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
            "commit {} 100\0\0\nA\0a.rs\0No signature\ncommit {} 200\0\0\nR100\0a.rs\0b.rs\0M\0c.rs\0commit {} 300\0\0\nD\0\nodd.rs\0",
            sha('a'),
            sha('b'),
            sha('c')
        );
        let commits = parse_log(out.as_bytes());
        assert_eq!(commits.len(), 3);
        assert_eq!(commits[1].time, 200);
        assert!(matches!(&commits[1].changes[0], Change::Renamed { from, to } if from == "a.rs" && to == "b.rs"));
        assert!(matches!(&commits[1].changes[1], Change::Modified(p) if p == "c.rs"));
        assert!(matches!(&commits[0].changes[0], Change::Added(p) if p == "a.rs"));
        // A path may begin with a newline; only status tokens lose theirs.
        assert!(matches!(&commits[2].changes[0], Change::Deleted(p) if p == "\nodd.rs"));
    }

    /// The never-landed cache is *read*: seeded with the current tips, it
    /// answers "uncommitted" even for a unit that is committed.
    #[test]
    fn the_never_landed_cache_is_consulted_while_the_tips_stand() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let run = |args: &[&str]| {
            crate::git::test_git(&root, args, &[]);
        };
        std::fs::write(root.join("a.txt"), "x").unwrap();
        run(&["init", "-q"]);
        run(&["add", "."]);
        run(&["commit", "-qm", "c"]);

        let write = WriteRecord {
            op: "toolu_1".into(),
            av: 2,
            a: "a".into(),
            t: crate::time::rfc3339(crate::time::now_secs() - 60),
            tool: "Edit".into(),
            file: "a.txt".into(),
            fh: Some(crate::objects::file_hash(b"x")),
            ..Default::default()
        };
        let unit = &units(&write)[0];
        let mut resolver = Resolver::new(&root).unwrap();
        let key = link_key(&write.op, &unit.target, unit.hash());
        let digest = tips_digest(&resolver.tips);
        let file = resolver.never_file();
        let lock = file.lock().unwrap();
        let lines = [
            NeverLine { tips: Some(digest.clone()), list: Some(resolver.tips.clone()), ..Default::default() },
            NeverLine { k: Some(key), tips: Some(digest), names: Some(vec!["a.txt".into()]), ..Default::default() },
        ];
        file.append(&lock, &lines, crate::objects::store::Durability::NoSync).unwrap();
        drop(lock);
        assert_eq!(resolver.landing(&write, unit).unwrap(), Landing::Uncommitted);
    }

    /// A repo with one commit of `a.txt`, and `n` file-level writes to it
    /// that never landed (their file hash matches nothing).
    fn unlanded(n: usize) -> (tempfile::TempDir, PathBuf, Vec<WriteRecord>) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        std::fs::write(root.join("a.txt"), "x").unwrap();
        for args in [&["init", "-q"][..], &["add", "."], &["commit", "-qm", "c"]] {
            crate::git::test_git(&root, args, &[]);
        }
        let writes = (0..n)
            .map(|i| WriteRecord {
                op: format!("toolu_{i}"),
                av: 2,
                t: crate::time::rfc3339(crate::time::now_secs() - 60),
                tool: "Write".into(),
                file: "a.txt".into(),
                fh: Some(crate::objects::file_hash(format!("never {i}").as_bytes())),
                ..Default::default()
            })
            .collect();
        (dir, root, writes)
    }

    /// What one resolver learns, the next reads from one flat file: the
    /// branch tips stored once, however many units share them.
    #[test]
    fn the_caches_are_two_flat_files_read_back_by_the_next_resolver() {
        let (_dir, root, writes) = unlanded(3);
        std::fs::create_dir_all(root.join(".ambits/links")).unwrap();
        std::fs::create_dir_all(root.join(".ambits/cache/never-landed")).unwrap();
        {
            let mut r = Resolver::new(&root).unwrap();
            assert!(!root.join(".ambits/links").exists(), "the per-entry cache is dropped");
            for w in &writes {
                assert_eq!(r.landing(w, &units(w)[0]).unwrap(), Landing::Uncommitted);
            }
        }
        let file = crate::objects::flat::FlatLog::at(root.join(".ambits/cache/never-landed.ndjson"));
        let lines: Vec<NeverLine> = file.read();
        assert_eq!(lines.iter().filter(|l| l.list.is_some()).count(), 1, "one tip list");
        assert_eq!(lines.iter().filter(|l| l.k.is_some()).count(), 3);
        assert_eq!(std::fs::read_dir(root.join(".ambits/cache")).unwrap().count(), 2, "the file and its lock");

        let mut r = Resolver::new(&root).unwrap();
        r.load();
        assert_eq!(r.caches.never.values().flatten().count(), 3);
    }

    /// Once most lines are dead, a flush rewrites the file with what is live.
    #[test]
    fn a_mostly_dead_cache_is_compacted() {
        let (_dir, root, writes) = unlanded(1);
        let (w, unit) = (&writes[0], &units(&writes[0])[0]);
        for _ in 0..80 {
            // A branch moving each time makes every search a new entry.
            let mut r = Resolver::new(&root).unwrap();
            r.tips.push(format!("{:040x}", r.tips.len() + crate::time::now_secs() as usize));
            r.tips.sort();
            r.landing(w, unit).unwrap();
        }
        let lines = crate::objects::flat::FlatLog::at(root.join(".ambits/cache/never-landed.ndjson")).line_count();
        assert!(lines <= 70, "compacted, not 160 lines: {lines}");
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
            syms: vec![("src/a.rs::A/f".into(), "b3:00".into())],
            ..Default::default()
        };
        assert_eq!(units(&w)[0].proof, Proof::Symbol { name_path: "A/f".into(), hash: "b3:00".into() });
        w.level = Level::File;
        w.syms.clear();
        assert_eq!(units(&w)[0].proof, Proof::None);
        w.fh = Some("b3:11".into());
        assert_eq!(units(&w), vec![Unit { target: "src/a.rs".into(), proof: Proof::FileHash("b3:11".into()) }]);
    }
}
