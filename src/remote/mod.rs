//! A dumb remote: a directory, on a local disk or a mount, laid out as
//! `.ambits/` is (spec §12.2, §12.3). `push` copies a session's history
//! there and moves its ref; `fetch` copies every session's back and mirrors
//! the remote's refs as `refs/remotes/<remote>/…`; `pull` fetches and merges
//! one session's remote history into the same session here.
//!
//! A remote serves **one user**: its files are private to whoever created
//! it, as `.ambits/` is, and one owned by someone else is refused.
//!
//! Nothing is trusted: every object read from either side is verified
//! ([`transfer`]), ref names and records are validated, links are proven
//! before they are kept, and a ref moves only after the whole history it
//! points at is in place.

pub mod config;
pub mod lock;
pub mod transfer;

use std::collections::HashSet;
use std::path::Path;

use color_eyre::eyre::{bail, eyre, Result, WrapErr};
use serde_json::Value;

use crate::journal::{self, EnvironmentManifest, Journal, Record};
use crate::objects::refs::{self, RefName};
use crate::objects::snapshot::{ancestors, Snapshot};
use crate::objects::store::{Durability, Store};
use crate::objects::sync_ignore::SyncIgnore;
use crate::objects::{gc, valid_hash, valid_label, valid_op, valid_record_path, valid_symbol_id, Kind, ObjectId};
use crate::symbols::ProjectTree;
use crate::writes::WriteRecord;

/// Refuse a remote someone else owns: its files would be private to them
/// (or ours to them), and a remote is single-user for now.
fn check_owner(remote: &Store) -> Result<()> {
    #[cfg(unix)]
    if let Ok(meta) = std::fs::metadata(remote.root()) {
        use std::os::unix::fs::MetadataExt;
        // SAFETY: getuid has no preconditions and cannot fail.
        let me = unsafe { libc::getuid() };
        if meta.uid() != me {
            bail!(
                "{} belongs to another user; a remote serves one user for now (its files are private to whoever created it)",
                remote.root().display()
            );
        }
    }
    #[cfg(not(unix))]
    let _ = remote;
    Ok(())
}

/// What `push` should do.
pub struct PushRequest<'a> {
    pub project_root: &'a Path,
    pub remote: Option<&'a str>,
    pub session: &'a str,
    /// Overwrite the remote's ref if it still holds what was last fetched.
    pub force_with_lease: bool,
    pub dry_run: bool,
    pub ignore: &'a SyncIgnore,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PushOutcome {
    /// The remote already holds this tip, or descends from it.
    UpToDate { remote: String, tip: ObjectId },
    Pushed {
        remote: String,
        from: Option<ObjectId>,
        to: ObjectId,
        objects: usize,
        links: usize,
        forced: bool,
        /// Snapshots sent that were made under other `[sync] ignore` rules
        /// than today's: what they hold is what those rules let in.
        other_ignore: usize,
    },
    /// What a push would do.
    DryRun { remote: String, from: Option<ObjectId>, to: ObjectId, objects: usize, links: usize, other_ignore: usize },
}

/// Push `req.session`'s history to a remote (§12.2).
pub fn push(req: &PushRequest<'_>) -> Result<PushOutcome> {
    let local = Store::at(req.project_root);
    let (remote_name, remote) = config::resolve(req.project_root, req.remote)?;
    check_owner(&remote)?;
    let name = RefName::session(req.session)?;
    let tracking = RefName::tracking(&remote_name, req.session)?;
    let Some(tip) = refs::read(&local, &name)? else {
        bail!("session {} has no snapshots to push; run `ambits snapshot` first", req.session);
    };
    let remote_tip = refs::read(&remote, &name)?;
    let lease = refs::read(&local, &tracking)?;

    // Already there, or the remote is ahead of us: nothing to send. The
    // lease stays where the last fetch left it — a tip never fetched must
    // not become what --force-with-lease trusts.
    if let Some(r) = remote_tip {
        if r == tip || ancestors(&remote, r)?.contains(&tip) {
            return Ok(PushOutcome::UpToDate { remote: remote_name, tip: r });
        }
    }
    let history = transfer::closure(&local, tip)?;
    let ours: HashSet<ObjectId> = history.iter().map(|s| s.id).collect();
    let fast_forward = remote_tip.is_none_or(|r| ours.contains(&r));
    if !fast_forward {
        if !req.force_with_lease {
            bail!(
                "{remote_name} has snapshots of this session you do not; run `ambits pull {remote_name}` and snapshot, \
                 or overwrite them with --force-with-lease"
            );
        }
        if remote_tip != lease {
            bail!("{remote_name} moved since you last fetched it; run `ambits fetch {remote_name}` and look before forcing");
        }
    }

    let links = pushable_links(req.project_root, &local, &history, req.ignore)?;
    let other_ignore = history.iter().filter(|s| s.inputs.ignore != req.ignore.digest()).count();
    if req.dry_run {
        let missing = history
            .iter()
            .flat_map(|s| [s.id, s.coverage, s.writes])
            .collect::<HashSet<_>>()
            .into_iter()
            .filter(|id| !remote.contains(id))
            .count();
        return Ok(PushOutcome::DryRun { remote: remote_name, from: remote_tip, to: tip, objects: missing, links: links.len(), other_ignore });
    }

    // Objects first, verified in place; then, under the remote's lock, its
    // notes, links and — last — the ref.
    let stats = transfer::transfer(&local, &remote, &history)?;
    let links = {
        let _lock = lock::RemoteLock::acquire(&remote, &config::Config::store_id(req.project_root)?)?;
        refs::copy_notes(&local, &remote, &ours)?;
        let links = crate::linkage::add_links(remote.root(), links, Durability::Fsync)?;
        refs::update(&remote, &name, remote_tip, tip, if fast_forward { "push" } else { "push (forced)" })
            .wrap_err_with(|| format!("someone else pushed to {remote_name} meanwhile; fetch and try again"))?;
        links
    };
    // Whatever a concurrent fetch left it at, it is now the tip we pushed.
    refs::update(&local, &tracking, refs::read(&local, &tracking)?, tip, "push")?;
    Ok(PushOutcome::Pushed { remote: remote_name, from: remote_tip, to: tip, objects: stats.copied, links, forced: !fast_forward, other_ignore })
}

/// The links of the writes in `history` whose file and target the ignore
/// filter lets leave the machine (§3.2, §10, D18) — not the whole index,
/// which holds sessions never pushed.
fn pushable_links(project_root: &Path, store: &Store, history: &[Snapshot], ignore: &SyncIgnore) -> Result<Vec<(String, crate::linkage::Link)>> {
    let mut ops = HashSet::new();
    for snap in history {
        let writes = store.get(&snap.writes, Kind::Writes)?;
        ops.extend(writes.get("writes").and_then(Value::as_array).into_iter().flatten().filter_map(|w| w.get("op")?.as_str().map(String::from)));
    }
    Ok(crate::linkage::links_of(&project_root.join(crate::state_dir::STATE_DIR))?
        .into_iter()
        .filter(|(_, l)| ops.contains(&l.op))
        .filter(|(_, l)| !ignore.is_ignored(&l.path) && !ignore.ignores_symbol(&l.target) && !ignore.is_ignored(&l.target))
        .collect())
}

/// One session ref a fetch moved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fetched {
    pub session: String,
    pub from: Option<ObjectId>,
    pub to: ObjectId,
    /// Not a fast-forward: someone forced the remote.
    pub forced: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FetchReport {
    pub remote: String,
    pub moved: Vec<Fetched>,
    pub objects: usize,
    /// Links the remote offered that were proven here and kept.
    pub links: usize,
    /// Sessions whose history did not verify, and why: nothing of theirs
    /// was recorded.
    pub failed: Vec<(String, String)>,
}

/// Fetch every session's history from a remote (§12.2). Each session on its
/// own: its objects verified, then `refs/remotes/<remote>/sessions/<id>`
/// set to the remote's ref — always, even when that is not a fast-forward,
/// so one clone's force-with-lease cannot wedge every other's fetch (§16).
/// A session that fails verification records nothing and is reported; the
/// others still come.
pub fn fetch(project_root: &Path, remote: Option<&str>) -> Result<FetchReport> {
    let local = Store::at(project_root);
    let (remote_name, remote) = config::resolve(project_root, remote)?;
    if !remote.root().is_dir() {
        bail!("remote {remote_name} ({}) does not exist yet; push to it first", remote.root().display());
    }
    check_owner(&remote)?;
    let _gc = gc::GcLock::shared(&local)?;
    let mut report = FetchReport { remote: remote_name.clone(), ..Default::default() };
    let mut ids = HashSet::new();
    for (name, tip) in refs::all(&remote)? {
        // Only well-formed session refs; anything else is not ours to take.
        let Some(session) = name.strip_prefix("refs/sessions/") else { continue };
        let Ok(tracking) = RefName::tracking(&remote_name, session) else { continue };
        let fetched = (|| -> Result<Option<Fetched>> {
            let history = transfer::closure(&remote, tip)?;
            report.objects += transfer::transfer(&remote, &local, &history)?.copied;
            let theirs: HashSet<ObjectId> = history.iter().map(|s| s.id).collect();
            let old = refs::read(&local, &tracking)?;
            ids.extend(theirs.iter().copied());
            if old == Some(tip) {
                return Ok(None);
            }
            let forced = old.is_some_and(|o| !theirs.contains(&o));
            refs::update(&local, &tracking, old, tip, if forced { "fetch (forced)" } else { "fetch" })?;
            Ok(Some(Fetched { session: session.to_string(), from: old, to: tip, forced }))
        })();
        match fetched {
            Ok(Some(moved)) => report.moved.push(moved),
            Ok(None) => {}
            Err(e) => report.failed.push((session.to_string(), format!("{e:#}"))),
        }
    }
    refs::copy_notes(&remote, &local, &ids)?;
    let offered = crate::linkage::links_of(remote.root())?.into_iter().map(|(_, l)| l).collect();
    report.links = crate::linkage::import_links(project_root, offered)?;
    Ok(report)
}

/// What `pull` should do.
pub struct PullRequest<'a> {
    pub project_root: &'a Path,
    pub remote: Option<&'a str>,
    pub session: &'a str,
    /// The project as it is now, to check remote reads against.
    pub tree: &'a ProjectTree,
    pub backend: &'static str,
    pub filter: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PullOutcome {
    /// The remote has no snapshots of this session.
    NotOnRemote,
    /// The remote's tip is ours or behind it.
    UpToDate,
    /// We are behind and the remote adds nothing we lack: nothing written,
    /// so pull → snapshot → push rounds settle (§12.3, §16).
    Behind,
    Merged {
        tip: ObjectId,
        /// Remote reads valid here, appended.
        reads: usize,
        /// Remote reads not valid here, kept as history.
        stale: usize,
        writes: usize,
        /// Same write, same attribution version, different contents: ours
        /// kept, theirs history.
        conflicts: usize,
        /// Remote records that failed validation, and were left out.
        rejected: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PullReport {
    pub fetch: FetchReport,
    pub outcome: PullOutcome,
}

/// A remote write fit to enter this journal (§9.1).
fn valid_write(w: &WriteRecord) -> bool {
    valid_op(&w.op)
        && valid_record_path(&w.file)
        && valid_label(&w.a)
        && valid_label(&w.tool)
        && (w.t.is_empty() || crate::time::parse_rfc3339(&w.t).is_some())
        && w.syms.iter().all(|(id, h)| valid_symbol_id(id) && valid_hash(h))
        && w.removed.iter().all(|id| valid_symbol_id(id))
        && w.fh.as_deref().is_none_or(valid_hash)
}

/// Fetch, then merge the remote's history of `req.session` into the same
/// session here (§12.3), appending to its `.pull` shard.
pub fn pull(req: &PullRequest<'_>) -> Result<PullReport> {
    let fetch = fetch(req.project_root, req.remote)?;
    if let Some((_, why)) = fetch.failed.iter().find(|(s, _)| s == req.session) {
        bail!("{}'s history of session {} did not verify: {why}", fetch.remote, req.session);
    }
    let store = Store::at(req.project_root);
    let remote_name = fetch.remote.clone();
    let Some(theirs) = refs::read(&store, &RefName::tracking(&remote_name, req.session)?)? else {
        return Ok(PullReport { fetch, outcome: PullOutcome::NotOnRemote });
    };
    let ours = refs::read(&store, &RefName::session(req.session)?)?;
    if let Some(ours) = ours {
        if ours == theirs || ancestors(&store, ours)?.contains(&theirs) {
            return Ok(PullReport { fetch, outcome: PullOutcome::UpToDate });
        }
    }
    let snap = Snapshot::load(&store, theirs)?;
    let dir = journal::journal_dir(req.project_root);
    let existing = journal::read_journal_session(&dir, req.session);
    let mut records: Vec<Record> = Vec::new();
    let mut rejected = 0;

    // Reads: appended when valid here and new to us; otherwise history, once.
    let reads: Vec<_> = crate::objects::restore::coverage_reads(&store.get(&snap.coverage, Kind::Coverage)?)?
        .into_iter()
        .filter(|r| {
            let ok = valid_symbol_id(&r.symbol) && valid_label(&r.agent);
            rejected += usize::from(!ok);
            ok
        })
        .collect();
    let verdicts = crate::objects::restore::classify_reads(&reads, req.tree);
    let (mut appended_reads, mut stale) = (0, 0);
    for read in &reads {
        let hash = journal::encode_hash(&read.hash);
        match verdicts.get(&(read.symbol.as_str(), read.hash)) {
            Some(crate::objects::restore::Verdict::Valid { symbol }) => {
                let known = existing.agent_reads.get(&(symbol.clone(), read.agent.clone())).is_some_and(|(h, d)| *h == read.hash && *d >= read.depth);
                if !known {
                    appended_reads += 1;
                    records.push(Record::Read { symbol_id: symbol.clone(), hash, depth: read.depth.into(), agent: Some(read.agent.clone()) });
                }
            }
            _ => {
                if existing.history_reads.contains(&(read.symbol.clone(), hash.clone())) {
                    continue;
                }
                stale += 1;
                let mut rest = serde_json::Map::new();
                rest.insert("sym".into(), read.symbol.clone().into());
                rest.insert("h".into(), hash.into());
                rest.insert("d".into(), serde_json::to_value(journal::DepthDto::from(read.depth))?);
                rest.insert("a".into(), read.agent.clone().into());
                rest.insert("origin".into(), remote_name.clone().into());
                records.push(Record::History { of: "read".into(), rest });
            }
        }
    }

    // Writes: a newer attribution wins; the same one with other contents is
    // a conflict — ours stays, theirs is kept as history, once.
    let (mut appended_writes, mut conflicts) = (0, 0);
    let remote_writes = store.get(&snap.writes, Kind::Writes)?;
    for value in remote_writes.get("writes").and_then(Value::as_array).into_iter().flatten() {
        let theirs_w = match serde_json::from_value::<WriteRecord>(value.clone()) {
            Ok(w) if valid_write(&w) => WriteRecord { origin: None, ..w },
            _ => {
                rejected += 1;
                continue;
            }
        };
        match existing.writes.get(&theirs_w.op) {
            Some(mine) if mine.av > theirs_w.av => {}
            Some(mine) if mine.av == theirs_w.av => {
                let same = WriteRecord { origin: None, ..mine.clone() } == theirs_w;
                if same || existing.conflict_writes.contains(&theirs_w.op) {
                    continue;
                }
                conflicts += 1;
                let Value::Object(mut rest) = serde_json::to_value(&theirs_w)? else { continue };
                rest.insert("conflict".into(), true.into());
                rest.insert("origin".into(), remote_name.clone().into());
                records.push(Record::History { of: "write".into(), rest });
            }
            _ => {
                appended_writes += 1;
                records.push(Record::Write(Box::new(WriteRecord { origin: Some(remote_name.clone()), ..theirs_w })));
            }
        }
    }

    // A merge record when anything was written, or the histories diverged —
    // once per remote tip; never for a clone merely behind with nothing new
    // (§12.3), so rounds settle.
    let diverged = match ours {
        None => true,
        Some(o) => !ancestors(&store, theirs)?.contains(&o),
    };
    if records.is_empty() && !diverged {
        return Ok(PullReport { fetch, outcome: PullOutcome::Behind });
    }
    if !existing.merges.contains(&theirs.hex()) {
        records.push(Record::Merge { remote: remote_name.clone(), tip: theirs.hex() });
    }
    if !records.is_empty() {
        let mut shard = Journal::open_shard(req.project_root, req.session, "pull", std::time::Duration::ZERO, || {
            EnvironmentManifest::capture(req.tree, req.backend, req.filter.clone())
        });
        for record in &records {
            if !shard.append(record) {
                return Err(eyre!("cannot write the pull shard {}", shard.path().display()));
            }
        }
    }
    Ok(PullReport {
        fetch,
        outcome: PullOutcome::Merged { tip: theirs, reads: appended_reads, stale, writes: appended_writes, conflicts, rejected },
    })
}
