//! `ambits restore` (spec §12.1, D19): a snapshot's coverage and writes,
//! restored into a session on this machine.
//!
//! Reads are checked against the project as it is now — the same
//! classification `restore-context` uses, moves included — and only those
//! still valid are restored. Writes are restored as history: facts about
//! what an agent did, not writes of the target session. Everything goes to
//! the target's own `<session>.restore.ndjson` shard, never to a file the
//! TUI holds open, and nothing the target's journal already has is
//! appended again, so restoring twice changes nothing.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use color_eyre::eyre::{bail, eyre, Result};
use serde_json::Value;

use super::refs::{self, RefName};
use super::snapshot::{self, Snapshot};
use super::store::Store;
use super::{Kind, ObjectId};
use crate::journal::{self, DepthDto, EnvironmentManifest, Journal, Record};
use crate::restore::{classify, ReadSet};
use crate::symbols::ProjectTree;
use crate::tracking::ReadDepth;

/// Everything `ambits restore` needs.
pub struct Request<'a> {
    pub project_root: &'a Path,
    /// A session id, snapshot id or unique prefix (see [`snapshot::resolve`]).
    pub reference: &'a str,
    /// The session to restore into; a new one is minted when `None`.
    pub into: Option<&'a str>,
    /// The project as scanned now.
    pub tree: &'a ProjectTree,
    /// For the restore shard's header, as the journal records it.
    pub backend: &'a str,
    pub filter: Option<String>,
}

/// How one file's reads came through.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct FileReport {
    /// Still what was read — restored.
    pub verified: usize,
    /// Changed since — not restored.
    pub drifted: usize,
    /// Not restored because the file was dirty when snapshotted: what was
    /// read was never committed, and this machine does not have it.
    pub unverifiable: usize,
    /// Of `verified`, found at a new address (moved between files).
    pub moved: usize,
}

/// What a restore did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Report {
    pub snapshot: ObjectId,
    pub session: String,
    /// The snapshot's git pin, when it differs from `HEAD` (which is `None`
    /// outside a repository). Never checked out — only reported.
    pub pinned_elsewhere: Option<(Option<String>, Option<String>)>,
    /// Per file, by the path the reads were recorded under.
    pub files: BTreeMap<String, FileReport>,
    pub reads_appended: usize,
    pub writes_appended: usize,
    /// Whether the target's ref moved (it already pointed here otherwise).
    pub ref_moved: bool,
}

/// One entry of a `coverage` object: `[symbol, hash, depth, agent]`.
struct CoverageRead {
    symbol: String,
    hash: [u8; 32],
    depth: ReadDepth,
    agent: String,
}

fn coverage_reads(payload: &Value) -> Result<Vec<CoverageRead>> {
    let entries = payload.get("reads").and_then(Value::as_array).ok_or_else(|| eyre!("coverage object has no reads"))?;
    entries
        .iter()
        .map(|e| {
            let field = |i: usize| e.get(i).and_then(Value::as_str).ok_or_else(|| eyre!("malformed coverage entry {e}"));
            let depth: DepthDto = serde_json::from_value(e.get(2).cloned().unwrap_or_default())?;
            Ok(CoverageRead {
                symbol: field(0)?.to_string(),
                hash: journal::decode_hash(field(1)?).ok_or_else(|| eyre!("malformed hash in {e}"))?,
                depth: depth.into(),
                agent: field(3)?.to_string(),
            })
        })
        .collect()
}

/// A new session id: a random (version 4) UUID, as Claude Code mints them.
pub fn mint_session_id() -> String {
    let hex = format!("{}{}", super::store::random_token(), super::store::random_token());
    let variant = ["8", "9", "a", "b"][usize::from_str_radix(&hex[16..17], 16).unwrap_or(0) % 4];
    format!("{}-{}-4{}-{}{}-{}", &hex[..8], &hex[8..12], &hex[13..16], variant, &hex[17..20], &hex[20..32])
}

/// Restore `req.reference` into a session (§12.1).
pub fn restore(req: &Request<'_>) -> Result<Report> {
    let store = Store::at(req.project_root);
    let id = snapshot::resolve(&store, req.reference)?;
    let snap = Snapshot::load(&store, id)?;
    let session = match req.into {
        Some(s) => s.to_string(),
        None => mint_session_id(),
    };
    let target = RefName::session(&session)?;

    // 1. Say so when the snapshot sits on another commit; never check out.
    let head = crate::git::Repo::discover(req.project_root).and_then(|r| r.head);
    let pinned_elsewhere = (snap.inputs.git != head).then(|| (snap.inputs.git.clone(), head));

    // 2. Classify the snapshot's reads against the project as it is now.
    let reads = coverage_reads(&store.get(&snap.coverage, Kind::Coverage)?)?;
    let verdicts = classify_reads(&reads, req.tree);
    let dirty: std::collections::HashSet<&str> = snap.inputs.dirty.iter().map(|(p, _)| p.as_str()).collect();

    // 3. Append what the target does not already have, to its own shard.
    let dir = journal::journal_dir(req.project_root);
    let existing = journal::read_journal_session(&dir, &session);
    let mut files: BTreeMap<String, FileReport> = BTreeMap::new();
    let mut new_records: Vec<Record> = Vec::new();
    for read in &reads {
        let (file, _) = crate::symbols::split_id(&read.symbol);
        let report = files.entry(file.to_string()).or_default();
        match verdicts.get(&(read.symbol.as_str(), read.hash)) {
            Some(Verdict::Valid { symbol }) => {
                report.verified += 1;
                report.moved += usize::from(symbol != &read.symbol);
                let key = (symbol.clone(), read.agent.clone());
                let known = existing.agent_reads.get(&key).is_some_and(|(h, d)| *h == read.hash && *d >= read.depth);
                if !known {
                    new_records.push(Record::Read {
                        symbol_id: symbol.clone(),
                        hash: journal::encode_hash(&read.hash),
                        depth: read.depth.into(),
                        agent: Some(read.agent.clone()),
                    });
                }
            }
            _ if dirty.contains(file) => report.unverifiable += 1,
            _ => report.drifted += 1,
        }
    }
    let reads_appended = new_records.len();
    let writes = store.get(&snap.writes, Kind::Writes)?;
    for write in writes.get("writes").and_then(Value::as_array).into_iter().flatten() {
        let Some(rest) = write.as_object() else { continue };
        let op = rest.get("op").and_then(Value::as_str).unwrap_or_default();
        if existing.history_writes.contains(op) {
            continue;
        }
        new_records.push(Record::History { of: "write".into(), rest: rest.clone() });
    }
    let writes_appended = new_records.len() - reads_appended;

    if !new_records.is_empty() {
        let mut shard = Journal::open_shard(req.project_root, &session, "restore", std::time::Duration::ZERO, || {
            EnvironmentManifest::capture(req.tree, req.backend, req.filter.clone())
        });
        for record in &new_records {
            if !shard.append(record) {
                bail!("cannot write the restore shard {}", shard.path().display());
            }
        }
    }

    // 4. The target's next snapshot descends from the restored one.
    let current = refs::read(&store, &target)?;
    let ref_moved = current != Some(id);
    if ref_moved {
        refs::update(&store, &target, current, id, "restore")?;
    }

    Ok(Report { snapshot: id, session, pinned_elsewhere, files, reads_appended, writes_appended, ref_moved })
}

/// Whether a read still holds, and under which id.
enum Verdict {
    Valid { symbol: String },
    Invalid,
}

/// Classify every `(symbol, hash)` the coverage holds. `classify` takes one
/// hash per symbol, but two agents may have read two versions of one
/// symbol, so reads are split into layers with at most one hash per symbol
/// and each layer classified — almost always there is just one.
fn classify_reads<'a>(reads: &'a [CoverageRead], tree: &ProjectTree) -> HashMap<(&'a str, [u8; 32]), Verdict> {
    let mut layers: Vec<ReadSet> = Vec::new();
    for read in reads {
        let slot = layers.iter_mut().find(|l| l.get(&read.symbol).is_none_or(|(h, _)| *h == read.hash));
        let layer = match slot {
            Some(layer) => layer,
            None => {
                layers.push(ReadSet::new());
                layers.last_mut().expect("just pushed")
            }
        };
        let entry = layer.entry(read.symbol.clone()).or_insert((read.hash, read.depth));
        entry.1 = entry.1.max(read.depth);
    }

    let mut verdicts: HashMap<(&str, [u8; 32]), Verdict> = HashMap::new();
    for layer in &layers {
        let outcome = classify(layer, tree);
        let mut valid: HashMap<&str, String> = HashMap::new();
        for r in &outcome.restored {
            let original = r.moved_from.as_deref().unwrap_or(&r.symbol_id);
            valid.insert(original, r.symbol_id.clone());
        }
        for read in reads {
            let Some((hash, _)) = layer.get(&read.symbol) else { continue };
            if *hash != read.hash {
                continue;
            }
            let verdict = match valid.get(read.symbol.as_str()) {
                Some(symbol) => Verdict::Valid { symbol: symbol.clone() },
                None => Verdict::Invalid,
            };
            verdicts.insert((read.symbol.as_str(), read.hash), verdict);
        }
    }
    verdicts
}

/// Print a restore's report.
pub fn print_report(out: &mut impl std::io::Write, report: &Report) -> std::io::Result<()> {
    writeln!(out, "restored snapshot {} into session {}", report.snapshot.short(), report.session)?;
    if let Some((pinned, head)) = &report.pinned_elsewhere {
        let short = |c: &Option<String>| c.as_deref().map_or("none".to_string(), |c| c.get(..7).unwrap_or(c).to_string());
        writeln!(
            out,
            "  warning: the snapshot is pinned to {}, but HEAD is {}; nothing was checked out",
            short(pinned),
            short(head)
        )?;
    }
    for (file, r) in &report.files {
        let mut parts = vec![format!("{} verified", r.verified)];
        if r.moved > 0 {
            parts.push(format!("{} moved", r.moved));
        }
        if r.drifted > 0 {
            parts.push(format!("{} drifted", r.drifted));
        }
        if r.unverifiable > 0 {
            parts.push(format!("{} unverifiable here (dirty when snapshotted)", r.unverifiable));
        }
        writeln!(out, "  {file}: {}", parts.join(", "))?;
    }
    writeln!(
        out,
        "  {} read(s) and {} write(s) appended{}",
        report.reads_appended,
        report.writes_appended,
        if report.ref_moved { "" } else { "; the session already pointed at this snapshot" }
    )
}
