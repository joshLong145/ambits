//! `ambits snapshot` and `ambits log` (spec §6, §11).

use std::path::Path;

use color_eyre::eyre::{bail, eyre, Result};
use serde_json::{json, Value};

use super::inputs::{self, Backend, Inputs};
use super::refs::{self, RefName};
use super::store::Store;
use super::sync_ignore::SyncIgnore;
use super::{b3, gc, record, tree, Kind, ObjectId};
use crate::ingest::tool_config::SyncConfig;
use crate::symbols::{FileSymbols, ProjectTree};

/// A snapshot object, parsed and verified.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snapshot {
    pub id: ObjectId,
    pub inputs: Inputs,
    pub inputs_digest: [u8; 32],
    pub root: ObjectId,
    pub coverage: ObjectId,
    pub writes: ObjectId,
    pub parents: Vec<ObjectId>,
    pub state_digest: [u8; 32],
}

/// `BLAKE3` over `root`, `coverage`, `writes` and the sorted parents (§6.3).
fn state_digest(root: &ObjectId, coverage: &ObjectId, writes: &ObjectId, parents: &[ObjectId]) -> [u8; 32] {
    let mut sorted = parents.to_vec();
    sorted.sort();
    let mut h = blake3::Hasher::new();
    h.update(b"ambits-state v1\0");
    for id in [root, coverage, writes].into_iter().chain(sorted.iter()) {
        h.update(&id.0);
    }
    *h.finalize().as_bytes()
}

impl Snapshot {
    fn to_value(&self) -> Value {
        let mut v = self.inputs.to_value();
        let obj = v.as_object_mut().expect("inputs are an object");
        obj.insert("inputs".into(), json!(b3(&self.inputs_digest)));
        obj.insert("root".into(), json!(self.root.hex()));
        obj.insert("coverage".into(), json!(self.coverage.hex()));
        obj.insert("writes".into(), json!(self.writes.hex()));
        let mut parents = self.parents.clone();
        parents.sort();
        obj.insert("parents".into(), json!(parents.iter().map(ObjectId::hex).collect::<Vec<_>>()));
        obj.insert("state_digest".into(), json!(b3(&self.state_digest)));
        v
    }

    /// Load snapshot `id` and check it against itself: its inputs hash to
    /// its `inputs`, those and its parents to its id (D17), and its contents
    /// to its `state_digest` (§6.3, §9.4).
    pub fn load(store: &Store, id: ObjectId) -> Result<Self> {
        let v = store.get(&id, Kind::Snapshot)?;
        let text = |k: &str| v.get(k).and_then(Value::as_str).ok_or_else(|| eyre!("snapshot {}: missing {k}", id.short()));
        let hash = |k: &str| -> Result<[u8; 32]> {
            crate::journal::decode_hash(text(k)?).ok_or_else(|| eyre!("snapshot {}: bad {k}", id.short()))
        };
        let oid = |k: &str| ObjectId::parse(text(k)?);
        let list = |k: &str| v.get(k).and_then(Value::as_array).cloned().unwrap_or_default();

        let git = text("git")?;
        let inputs = Inputs {
            session: text("session")?.to_string(),
            journal: hash("journal")?,
            git: (git != "none").then(|| git.to_string()),
            parsers: list("parsers").iter().filter_map(|p| p.as_str().map(String::from)).collect(),
            scan: hash("scan")?,
            ignore: hash("ignore")?,
            dirty: list("dirty")
                .iter()
                .filter_map(|d| Some((d.get(0)?.as_str()?.to_string(), d.get(1)?.as_str()?.to_string())))
                .collect(),
        };
        let parents = list("parents").iter().map(|p| ObjectId::parse(p.as_str().unwrap_or_default())).collect::<Result<Vec<_>>>()?;
        let snapshot = Self {
            id,
            inputs_digest: hash("inputs")?,
            root: oid("root")?,
            coverage: oid("coverage")?,
            writes: oid("writes")?,
            state_digest: hash("state_digest")?,
            inputs,
            parents,
        };
        // Fields are validated as untrusted input (§9.1), and the payload
        // must be exactly what these values encode to: an extra key would
        // ride along covered by no digest.
        let git_ok = snapshot.inputs.git.as_deref().is_none_or(crate::git::is_commit_id);
        let dirty_ok = snapshot.inputs.dirty.iter().all(|(p, _)| super::valid_record_path(p));
        if !git_ok || !dirty_ok || !crate::ingest::claude::is_uuid(&snapshot.inputs.session) {
            bail!("snapshot {}: malformed fields", id.short());
        }
        if super::canonical::to_bytes(&snapshot.to_value())? != super::canonical::to_bytes(&v)? {
            bail!("snapshot {}: unexpected or malformed fields", id.short());
        }
        if snapshot.inputs.digest()? != snapshot.inputs_digest {
            bail!("snapshot {}: its inputs do not match its inputs digest", id.short());
        }
        if inputs::snapshot_id(&snapshot.inputs_digest, &snapshot.parents) != id {
            bail!("snapshot {}: does not match its id", id.short());
        }
        if state_digest(&snapshot.root, &snapshot.coverage, &snapshot.writes, &snapshot.parents) != snapshot.state_digest {
            bail!("snapshot {}: its contents do not match its state digest", id.short());
        }
        Ok(snapshot)
    }

    /// The objects a snapshot references directly.
    pub fn references(&self) -> Vec<(ObjectId, Kind)> {
        let mut out = vec![(self.root, Kind::Dir), (self.coverage, Kind::Coverage), (self.writes, Kind::Writes)];
        out.extend(self.parents.iter().map(|p| (*p, Kind::Snapshot)));
        out
    }
}

/// Everything `ambits snapshot` needs.
pub struct Request<'a> {
    pub project_root: &'a Path,
    pub session: &'a str,
    pub tree: &'a ProjectTree,
    pub backend: Backend<'a>,
    /// `--filter`/`--filter-regex` as the scan applied it.
    pub filter: Option<String>,
    pub sync: &'a SyncConfig,
    pub message: Option<&'a str>,
    pub require_clean: bool,
}

/// What `ambits snapshot` did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// Inputs equal the tip's and nothing is pending: nothing written (§6.4).
    NothingChanged(ObjectId),
    Created { id: ObjectId, parents: Vec<ObjectId>, files: usize, reads: usize, writes: usize, dirty: usize },
}

/// Make a snapshot of `req.session` (§6.4).
pub fn snapshot(req: &Request<'_>) -> Result<Outcome> {
    let store = Store::at(req.project_root);
    let name = RefName::session(req.session)?;
    let ignore = SyncIgnore::new(req.sync)?;
    // Shared: gc waits until this snapshot's objects are reachable (§8).
    let _gc = gc::GcLock::shared(&store)?;

    let prefix = record::read_prefix(req.project_root, req.session)?;
    // After the scan (§7): see `inputs::environment`.
    let env = inputs::environment(req.project_root, req.tree, req.filter.as_deref(), &ignore)?;
    let scanned = match &req.backend {
        Backend::TreeSitter(registry) => reparse_dirty(req.tree, &env.contents, registry),
        // Serena's symbols come from its cache, not from the files.
        Backend::Serena { .. } => req.tree.clone(),
    };
    if req.require_clean && !env.dirty.is_empty() {
        bail!("working tree has {} dirty file(s); commit or stash them, or drop --require-clean", env.dirty.len());
    }
    let inputs = Inputs {
        session: req.session.to_string(),
        journal: prefix.digest,
        git: env.git,
        parsers: req.backend.parsers(),
        scan: env.scan,
        ignore: ignore.digest(),
        dirty: env.dirty,
    };
    let inputs_digest = inputs.digest()?;

    // Parents: the session's tip. Pending merge tips join them once `pull`
    // records merges (phase 6); until then there are none.
    let tip = refs::read(&store, &name)?;
    let mut repair: Option<Snapshot> = None;
    if let Some(tip) = tip {
        let current = Snapshot::load(&store, tip)?;
        if current.inputs_digest == inputs_digest {
            if [current.root, current.coverage, current.writes].iter().all(|id| store.contains(id)) {
                return Ok(Outcome::NothingChanged(tip));
            }
            // The tip survived a crash its objects did not. Same inputs,
            // same objects: rebuild them below and leave the tip as it is.
            repair = Some(current);
        }
    }
    let parents: Vec<ObjectId> = tip.into_iter().collect();
    let id = inputs::snapshot_id(&inputs_digest, &parents);

    // Children before parents; the snapshot object last (§8).
    let parser_of = |file: &FileSymbols| -> String {
        match &req.backend {
            Backend::TreeSitter(registry) => registry.parser_for(&file.file_path).map_or_else(|| "none".into(), |p| p.identity()),
            Backend::Serena { .. } => req.backend.parsers().join(","),
        }
    };
    let tree = tree::write_tree(&store, &scanned, &parser_of, &ignore)?;
    let records = record::write_records(&store, req.session, &prefix.contents, &ignore)?;
    if let Some(tip) = repair {
        if (tip.root, tip.coverage, tip.writes) != (tree.root, records.coverage, records.writes) {
            bail!("snapshot {} is missing objects that cannot be rebuilt from the current state", tip.id.short());
        }
        store.sync_dirs()?;
        return Ok(Outcome::NothingChanged(tip.id));
    }
    let dirty = inputs.dirty.len();
    let snapshot = Snapshot {
        id,
        inputs,
        inputs_digest,
        root: tree.root,
        coverage: records.coverage,
        writes: records.writes,
        state_digest: state_digest(&tree.root, &records.coverage, &records.writes, &parents),
        parents: parents.clone(),
    };
    if store.contains(&id) {
        // Same inputs and parents must mean the same state. Anything else is
        // corruption or a forged object, never something to skip (§6.3).
        let existing = Snapshot::load(&store, id)?;
        if existing.state_digest != snapshot.state_digest {
            bail!("snapshot {} already exists with different contents; the store may be corrupt", id.short());
        }
    }
    // Every object durable before the snapshot that references them, and
    // the snapshot before the ref (§8).
    store.sync_dirs()?;
    store.put_at(&id, Kind::Snapshot, &snapshot.to_value())?;
    store.sync_dirs()?;
    refs::write_note(&store, &id, req.message)?;
    refs::update(&store, &name, tip, id, "snapshot")?;

    Ok(Outcome::Created { id, parents, files: tree.files, reads: records.reads, writes: records.write_count, dirty })
}

/// `tree` with every dirty file parsed again from the bytes its fingerprint
/// was taken from — or dropped when it is gone or no longer parses — so the
/// tree and the `dirty` input describe the same moment even if a file
/// changed while the project was being scanned (§7).
fn reparse_dirty(
    tree: &ProjectTree,
    contents: &[(std::path::PathBuf, Option<Vec<u8>>)],
    registry: &crate::parser::ParserRegistry,
) -> ProjectTree {
    let mut tree = tree.clone();
    for (raw, bytes) in contents {
        let key = super::normalize_path(&raw.to_string_lossy());
        let Some(pos) = tree.files.iter().position(|f| super::normalize_path(&f.file_path.to_string_lossy()) == key) else {
            continue;
        };
        let path = tree.files[pos].file_path.clone();
        let parsed = bytes
            .as_deref()
            .and_then(|b| std::str::from_utf8(b).ok())
            .zip(registry.parser_for(&path))
            .and_then(|(source, parser)| parser.parse_file(&path, source).ok());
        match parsed {
            Some(file) => tree.files[pos] = file,
            None => {
                tree.files.remove(pos);
            }
        }
    }
    tree
}

/// Resolve what `ambits log` was given: a session id (its ref), a full
/// snapshot id, or an unambiguous prefix of one (at least 7 digits).
pub fn resolve(store: &Store, arg: &str) -> Result<ObjectId> {
    if let Ok(name) = RefName::session(arg) {
        return refs::read(store, &name)?.ok_or_else(|| eyre!("session {arg} has no snapshots"));
    }
    if let Ok(id) = ObjectId::parse(arg) {
        return Ok(id);
    }
    if arg.len() < 7 || !arg.bytes().all(|b| b.is_ascii_hexdigit()) {
        bail!("{arg:?} is neither a session id nor a snapshot id");
    }
    let arg = arg.to_ascii_lowercase();
    let matches: Vec<ObjectId> = store
        .list()
        .into_iter()
        .map(|(id, _)| id)
        .filter(|id| id.hex().starts_with(&arg) && store.get(id, Kind::Snapshot).is_ok())
        .collect();
    match matches.as_slice() {
        [one] => Ok(*one),
        [] => Err(eyre!("no snapshot starts with {arg}")),
        _ => Err(eyre!("{arg} is ambiguous: {} snapshots start with it", matches.len())),
    }
}

/// One line of history.
#[derive(Debug, Clone)]
pub struct LogEntry {
    pub snapshot: Snapshot,
    pub note: Option<refs::Note>,
    pub reads: usize,
    pub writes: usize,
}

/// `start` and its ancestors, newest first by note time.
pub fn history(store: &Store, start: ObjectId) -> Result<Vec<LogEntry>> {
    let count = |id: &ObjectId, kind: Kind, key: &str| -> Result<usize> {
        Ok(store.get(id, kind)?.get(key).and_then(Value::as_array).map_or(0, Vec::len))
    };
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![start];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        let snapshot = Snapshot::load(store, id)?;
        stack.extend(snapshot.parents.iter().copied());
        out.push(LogEntry {
            note: refs::read_note(store, &id),
            reads: count(&snapshot.coverage, Kind::Coverage, "reads")?,
            writes: count(&snapshot.writes, Kind::Writes, "writes")?,
            snapshot,
        });
    }
    out.sort_by(|a, b| {
        let t = |e: &LogEntry| e.note.as_ref().map(|n| n.time.clone()).unwrap_or_default();
        t(b).cmp(&t(a)).then(a.snapshot.id.cmp(&b.snapshot.id))
    });
    Ok(out)
}

/// Print `ambits log`.
pub fn print_log(out: &mut impl std::io::Write, entries: &[LogEntry]) -> std::io::Result<()> {
    for (i, e) in entries.iter().enumerate() {
        if i > 0 {
            writeln!(out)?;
        }
        let s = &e.snapshot;
        let time = e.note.as_ref().map_or("unknown time", |n| n.time.as_str());
        writeln!(out, "snapshot {}  {time}", s.id.short())?;
        let git = s.inputs.git.as_deref().map_or("none", |g| g.get(..7).unwrap_or(g));
        let dirty = match s.inputs.dirty.len() {
            0 => String::new(),
            n => format!(" (dirty: {n} file{})", if n == 1 { "" } else { "s" }),
        };
        writeln!(out, "  git {git}{dirty}   reads {}   writes {}", e.reads, e.writes)?;
        if !s.parents.is_empty() {
            let parents: Vec<String> = s.parents.iter().map(ObjectId::short).collect();
            writeln!(out, "  parents {}", parents.join(" "))?;
        }
        if s.inputs.parsers.iter().any(|p| p.starts_with("serena@")) {
            writeln!(out, "  non-reproducible (Serena backend)")?;
        }
        if let Some(message) = e.note.as_ref().and_then(|n| n.message.as_deref()) {
            writeln!(out, "  {message}")?;
        }
    }
    Ok(())
}

/// Report a snapshot's outcome.
pub fn print_outcome(out: &mut impl std::io::Write, outcome: &Outcome) -> std::io::Result<()> {
    match outcome {
        Outcome::NothingChanged(tip) => writeln!(out, "nothing changed: {}", tip.short()),
        Outcome::Created { id, parents, files, reads, writes, dirty } => {
            let parent = parents.first().map_or("none".to_string(), ObjectId::short);
            writeln!(
                out,
                "snapshot {}  ({files} files, {reads} reads, {writes} writes, {dirty} dirty; parent {parent})",
                id.short()
            )
        }
    }
}
