//! Phase 6 end to end: two clones sharing a dumb remote (spec §12.2,
//! §12.3; the key tests of §14).

mod common;

use std::path::{Path, PathBuf};

use ambits::ingest::tool_config::SyncConfig;
use ambits::objects::refs::{self, RefName};
use ambits::objects::snapshot::{snapshot, Outcome, Snapshot};
use ambits::objects::store::Store;
use ambits::objects::sync_ignore::SyncIgnore;
use ambits::objects::ObjectId;
use ambits::parser::ParserRegistry;
use ambits::remote::{self, PullOutcome, PushOutcome};
use common::{git, run_ambits};

const SESSION: &str = "0b7e9d3a-1c2f-4e5a-9b8c-7d6e5f4a3b2c";
const SOURCE: &str = "fn alpha() {}\nfn beta() {}\n";

/// A clone of the project: a git repo with `src/a.rs`, and a remote.
struct Clone {
    _dir: tempfile::TempDir,
    root: PathBuf,
    registry: ParserRegistry,
}

impl Clone {
    fn new(remote: &Path) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/a.rs"), SOURCE).unwrap();
        std::fs::write(root.join(".gitignore"), ".ambits/\n").unwrap();
        git(&root, &["init", "-q"]);
        git(&root, &["add", "."]);
        git(&root, &["commit", "-qm", "init"]);
        remote::config::add(&root, "origin", remote).unwrap();
        Self { _dir: dir, root, registry: ParserRegistry::new() }
    }

    fn store(&self) -> Store {
        Store::at(&self.root)
    }

    /// Append a full read of `name` by `agent`, at the hash it has here.
    fn read(&self, name: &str, agent: &str) {
        self.read_in(SESSION, name, agent)
    }

    fn read_in(&self, session: &str, name: &str, agent: &str) {
        let tree = self.registry.scan_project(&self.root, None).unwrap();
        let (_, sym) = tree.walk().into_iter().find(|(_, s)| &*s.name == name).unwrap();
        let record = ambits::journal::Record::Read {
            symbol_id: sym.id.clone(),
            hash: ambits::journal::encode_hash(&sym.content_hash),
            depth: ambits::journal::DepthDto::FullBody,
            agent: Some(agent.into()),
        };
        let dir = self.root.join(".ambits/coverage");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{session}.ndjson"));
        let mut text = std::fs::read_to_string(&path).unwrap_or_default();
        text.push_str(&serde_json::to_string(&record).unwrap());
        text.push('\n');
        std::fs::write(path, text).unwrap();
    }

    fn snap(&self) -> Outcome {
        self.snap_in(SESSION)
    }

    fn snap_in(&self, session: &str) -> Outcome {
        let tree = self.registry.scan_project(&self.root, None).unwrap();
        snapshot(&ambits::objects::snapshot::Request {
            project_root: &self.root,
            session,
            tree: &tree,
            sync: &SyncConfig::default(),
            message: None,
            require_clean: false,
        })
        .unwrap()
    }

    fn push(&self, force: bool) -> color_eyre::Result<PushOutcome> {
        push_from(&self.root, SESSION, force)
    }

    fn pull(&self) -> PullOutcome {
        let tree = self.registry.scan_project(&self.root, None).unwrap();
        remote::pull(&remote::PullRequest {
            project_root: &self.root,
            remote: None,
            session: SESSION,
            tree: &tree,
            backend: "tree-sitter",
            filter: None,
            verify: Default::default(),
        })
        .unwrap()
        .outcome
    }

    fn reads(&self) -> Vec<String> {
        let contents = ambits::journal::read_journal_session(&self.root.join(".ambits/coverage"), SESSION);
        let mut ids: Vec<String> = contents.reads.into_keys().collect();
        ids.sort();
        ids
    }
}

fn push_from(root: &Path, session: &str, force: bool) -> color_eyre::Result<PushOutcome> {
    remote::push(&remote::PushRequest {
        project_root: root,
        remote: None,
        session,
        force_with_lease: force,
        dry_run: false,
        ignore: &SyncIgnore::none(),
        verify: Default::default(),
    })
}

fn created(o: Outcome) -> ObjectId {
    match o {
        Outcome::Created { id, .. } => id,
        Outcome::NothingChanged(id) => panic!("expected a new snapshot, tip {id:?}"),
    }
}

fn pair() -> (tempfile::TempDir, PathBuf, Clone, Clone) {
    let dir = tempfile::tempdir().unwrap();
    let remote = dir.path().canonicalize().unwrap().join("remote");
    let (a, b) = (Clone::new(&remote), Clone::new(&remote));
    (dir, remote, a, b)
}

fn remote_tip(remote: &Path) -> Option<ObjectId> {
    refs::read(&Store::at_root(remote), &RefName::session(SESSION).unwrap()).unwrap()
}

/// A pushes; B pulls, snapshots and pushes; then every further round of
/// pull → snapshot → push on either side changes nothing (§14, §16).
#[test]
fn two_clones_converge_and_then_settle() {
    let (_dir, remote, a, b) = pair();
    a.read("alpha", "agent-1");
    let s1 = created(a.snap());
    assert!(matches!(a.push(false).unwrap(), PushOutcome::Pushed { from: None, .. }));
    assert_eq!(remote_tip(&remote), Some(s1));

    assert!(matches!(b.pull(), PullOutcome::Merged { reads: 1, .. }));
    assert_eq!(b.reads(), vec!["src/a.rs::alpha"]);
    let s2 = created(b.snap());
    assert_eq!(Snapshot::load(&b.store(), s2).unwrap().parents, vec![s1], "descends from A's");
    assert!(matches!(b.push(false).unwrap(), PushOutcome::Pushed { forced: false, .. }));

    for _ in 0..2 {
        assert_eq!(a.pull(), PullOutcome::Behind, "behind, and B added nothing A lacks");
        assert!(matches!(a.snap(), Outcome::NothingChanged(_)), "nothing new to snapshot");
        assert!(matches!(a.push(false).unwrap(), PushOutcome::UpToDate { .. }));
        assert!(matches!(b.pull(), PullOutcome::UpToDate));
        assert!(matches!(b.snap(), Outcome::NothingChanged(_)));
        assert!(matches!(b.push(false).unwrap(), PushOutcome::UpToDate { .. }));
    }
    assert_eq!(remote_tip(&remote), Some(s2));
}

/// Diverged with nothing to append: the pull still records the merge, the
/// snapshot takes both parents, and the push is a fast-forward.
#[test]
fn a_diverged_pull_with_nothing_new_makes_a_two_parent_snapshot() {
    let (_dir, remote, a, b) = pair();
    a.read("alpha", "agent-1");
    let theirs = created(a.snap());
    a.push(false).unwrap();
    // B knows everything A pushed, and more: its history differs, but the
    // remote's adds nothing it lacks.
    b.read("alpha", "agent-1");
    b.read("beta", "agent-2");
    let ours = created(b.snap());
    assert!(b.push(false).is_err(), "non-fast-forward");

    assert!(matches!(b.pull(), PullOutcome::Merged { reads: 0, writes: 0, .. }));
    let merge = created(b.snap());
    let mut parents = Snapshot::load(&b.store(), merge).unwrap().parents;
    parents.sort();
    let mut want = vec![ours, theirs];
    want.sort();
    assert_eq!(parents, want);
    assert!(matches!(b.push(false).unwrap(), PushOutcome::Pushed { forced: false, .. }));
    assert_eq!(remote_tip(&remote), Some(merge));
}

/// After B forces the remote, A's fetch mirrors it (warned, the old tip in
/// the reflog) and A's pull still works (§16).
#[test]
fn after_a_force_with_lease_the_other_clone_still_fetches_and_pulls() {
    let (_dir, remote, a, b) = pair();
    a.read("alpha", "agent-1");
    let s1 = created(a.snap());
    a.push(false).unwrap();
    b.read("beta", "agent-2");
    let forced = created(b.snap());
    assert!(b.push(true).is_err(), "no lease yet: B never fetched");
    remote::fetch(&b.root, None, Default::default()).unwrap();
    assert!(matches!(b.push(true).unwrap(), PushOutcome::Pushed { forced: true, .. }));
    assert_eq!(remote_tip(&remote), Some(forced));

    let report = remote::fetch(&a.root, None, Default::default()).unwrap();
    assert!(report.moved.iter().any(|m| m.forced && m.from == Some(s1) && m.to == forced));
    assert!(matches!(a.pull(), PullOutcome::Merged { reads: 1, .. }));
    assert_eq!(a.reads(), vec!["src/a.rs::alpha", "src/a.rs::beta"]);
    let merge = created(a.snap());
    assert!(matches!(a.push(false).unwrap(), PushOutcome::Pushed { forced: false, .. }));
    assert_eq!(remote_tip(&remote), Some(merge));
}

/// A remote read of code that is different here is kept as history, once;
/// it never regresses what we know.
#[test]
fn a_stale_remote_read_becomes_history_once() {
    let (_dir, _remote, a, b) = pair();
    a.read("alpha", "agent-1");
    a.snap();
    a.push(false).unwrap();
    std::fs::write(b.root.join("src/a.rs"), "fn alpha() { 1; }\nfn beta() {}\n").unwrap();
    assert!(matches!(b.pull(), PullOutcome::Merged { reads: 0, stale: 1, .. }));
    assert!(b.reads().is_empty());
    assert!(matches!(b.pull(), PullOutcome::Merged { stale: 0, .. }), "not again");
    let shard = std::fs::read_to_string(b.root.join(format!(".ambits/coverage/{SESSION}.pull.ndjson"))).unwrap();
    assert_eq!(shard.matches(r#""of":"read""#).count(), 1);
}

/// A push that cannot take the remote's lock moves nothing — no dangling
/// ref — and the lock's owner can still push.
#[test]
fn a_locked_remote_refuses_a_push_and_leaves_no_ref() {
    let (_dir, remote, a, _b) = pair();
    a.read("alpha", "agent-1");
    a.snap();
    let held = remote::lock::RemoteLock::acquire(&Store::at_root(&remote), "someone").unwrap();
    assert!(a.push(false).unwrap_err().to_string().contains("locked"));
    assert_eq!(remote_tip(&remote), None);
    drop(held);
    assert!(matches!(a.push(false).unwrap(), PushOutcome::Pushed { .. }));
}

/// Fetch verifies: a tampered object, a symlink or an oversized file on the
/// remote is refused — each for its own reason, not only a bad hash — and
/// nothing of that session is recorded.
#[test]
fn fetch_refuses_what_does_not_verify() {
    let (_dir, remote, a, b) = pair();
    a.read("alpha", "agent-1");
    let tip = created(a.snap());
    a.push(false).unwrap();
    let remote_store = Store::at_root(&remote);
    let snap = Snapshot::load(&remote_store, tip).unwrap();
    let coverage = remote_store.path_of(&snap.coverage);
    let original = std::fs::read(&coverage).unwrap();
    let tracking = RefName::tracking("origin", SESSION).unwrap();
    let refused = |why: &str| {
        let report = remote::fetch(&b.root, None, Default::default()).unwrap();
        assert_eq!(report.failed.len(), 1, "{why}: {report:?}");
        assert_eq!(refs::read(&b.store(), &tracking).unwrap(), None, "{why}: nothing recorded");
        report.failed[0].1.clone()
    };

    std::fs::write(&coverage, String::from_utf8(original.clone()).unwrap().replace("full_body", "overview")).unwrap();
    assert!(refused("tampered").contains("does not match its id"));

    #[cfg(unix)]
    {
        // A symlink to the right bytes: only the symlink check can refuse it.
        let elsewhere = remote.join("elsewhere.json");
        std::fs::write(&elsewhere, &original).unwrap();
        std::fs::remove_file(&coverage).unwrap();
        std::os::unix::fs::symlink(&elsewhere, &coverage).unwrap();
        let why = refused("symlink");
        assert!(why.contains("symbolic link") || why.contains("regular file"), "{why}");
        std::fs::remove_file(&coverage).unwrap();
    }
    // The right bytes padded past the cap: only the size check can refuse it.
    let mut padded = original.clone();
    padded.resize(17 * 1024 * 1024, b' ');
    std::fs::write(&coverage, padded).unwrap();
    assert!(refused("oversized").contains("limit"));

    std::fs::write(&coverage, original).unwrap();
    assert!(remote::fetch(&b.root, None, Default::default()).unwrap().failed.is_empty());
    assert_eq!(refs::read(&b.store(), &tracking).unwrap(), Some(tip));
}

/// One bad session is reported and skipped; the others still come, and
/// the CLI says so and fails.
#[test]
fn a_bad_session_does_not_stop_the_others() {
    const OTHER: &str = "1c8f0e4b-2d3a-4f6b-8c9d-8e7f6a5b4c3d";
    let (_dir, remote, a, b) = pair();
    a.read("alpha", "agent-1");
    let good = created(a.snap());
    a.push(false).unwrap();
    a.read_in(OTHER, "beta", "agent-1");
    let bad = created(a.snap_in(OTHER));
    push_from(&a.root, OTHER, false).unwrap();
    let remote_store = Store::at_root(&remote);
    let coverage = remote_store.path_of(&Snapshot::load(&remote_store, bad).unwrap().coverage);
    std::fs::write(&coverage, "{}").unwrap();

    let report = remote::fetch(&b.root, None, Default::default()).unwrap();
    assert_eq!(report.failed.iter().map(|(s, _)| s.as_str()).collect::<Vec<_>>(), vec![OTHER]);
    assert_eq!(refs::read(&b.store(), &RefName::tracking("origin", SESSION).unwrap()).unwrap(), Some(good));
    assert_eq!(refs::read(&b.store(), &RefName::tracking("origin", OTHER).unwrap()).unwrap(), None);
    assert!(matches!(b.pull(), PullOutcome::Merged { reads: 1, .. }), "the healthy session still pulls");

    let out = common::ambits(&b.root).args(["fetch"]).output().unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains(&format!("origin/{OTHER}: not fetched")));
}

/// Hostile ref names in the remote's reflog are ignored, never followed.
#[test]
fn hostile_ref_names_on_the_remote_are_ignored() {
    let (_dir, remote, a, b) = pair();
    a.read("alpha", "agent-1");
    a.snap();
    a.push(false).unwrap();
    let mut reflog = std::fs::read_to_string(remote.join("reflog.ndjson")).unwrap();
    for name in ["refs/sessions/../../etc/passwd", "refs/other/x", "refs/sessions/not-a-uuid"] {
        reflog.push_str(&format!(r#"{{"ref":"{name}","old":null,"new":"{}","secs":1,"time":"t","action":"x"}}"#, "0".repeat(64)));
        reflog.push('\n');
    }
    std::fs::write(remote.join("reflog.ndjson"), reflog).unwrap();
    let report = remote::fetch(&b.root, None, Default::default()).unwrap();
    assert!(report.failed.is_empty(), "{report:?}");
    assert_eq!(report.moved.len(), 1);
}

/// A remote's links are kept only when this machine can prove them; a
/// forged one — a reachable commit that does not hold the write — is not.
#[test]
fn fetched_links_are_kept_only_when_proven() {
    let (_dir, remote, a, b) = pair();
    a.read("alpha", "agent-1");
    a.snap();
    a.push(false).unwrap();
    let head = git(&b.root, &["rev-parse", "HEAD"]);
    let tree = ParserRegistry::new().scan_project(&b.root, None).unwrap();
    let (_, alpha) = tree.walk().into_iter().find(|(_, s)| &*s.name == "alpha").unwrap();
    let real = ambits::journal::encode_hash(&alpha.content_hash);
    let link = |op: &str, hash: &str| {
        serde_json::json!({"k": op, "link": {"op": op, "target": "src/a.rs::alpha", "hash": hash, "commit": head, "verified": true, "path": "src/a.rs"}})
    };
    let lines = [link("toolu_real", &real), link("toolu_forged", &format!("b3:{}", "f".repeat(64)))];
    let text: String = lines.iter().map(|l| format!("{l}\n")).collect();
    std::fs::write(remote.join("links.ndjson"), text).unwrap();

    let report = remote::fetch(&b.root, None, Default::default()).unwrap();
    assert_eq!(report.links, 1);
    let kept = ambits::linkage::links_of(&b.root.join(".ambits")).unwrap();
    assert_eq!(kept.iter().map(|(_, l)| l.op.as_str()).collect::<Vec<_>>(), vec!["toolu_real"]);
}

/// Links resolved after a push (the post-commit hook runs after it) still
/// go with the next push, though nothing else is new.
#[test]
fn links_resolved_later_are_pushed_without_a_new_snapshot() {
    let (_dir, remote, a, _b) = pair();
    let write = ambits::writes::WriteRecord {
        op: "toolu_late".into(),
        av: ambits::writes::ATTRIBUTION_VERSION,
        a: "agent-1".into(),
        t: "2026-09-27T10:00:00Z".into(),
        tool: "Edit".into(),
        file: "src/a.rs".into(),
        ..Default::default()
    };
    let dir = a.root.join(".ambits/coverage");
    std::fs::create_dir_all(&dir).unwrap();
    let line = serde_json::to_string(&ambits::journal::Record::Write(Box::new(write))).unwrap();
    std::fs::write(dir.join(format!("{SESSION}.ndjson")), format!("{line}\n")).unwrap();
    a.snap();
    a.push(false).unwrap();
    let link = serde_json::json!({"k": "toolu_late", "link": {"op": "toolu_late", "target": "src/a.rs", "hash": null, "commit": "0".repeat(40), "verified": false, "path": "src/a.rs"}});
    std::fs::write(a.root.join(".ambits/links.ndjson"), format!("{link}\n")).unwrap();
    assert!(matches!(a.push(false).unwrap(), PushOutcome::UpToDate { links: 1, .. }));
    assert!(std::fs::read_to_string(remote.join("links.ndjson")).unwrap().contains("toolu_late"));
    assert!(matches!(a.push(false).unwrap(), PushOutcome::UpToDate { links: 0, .. }), "once");
}

/// A push sends only the links of writes it pushes, not the whole index.
#[test]
fn a_push_sends_only_its_own_links() {
    let (_dir, remote, a, _b) = pair();
    let write = |op: &str| ambits::writes::WriteRecord {
        op: op.into(),
        av: ambits::writes::ATTRIBUTION_VERSION,
        a: "agent-1".into(),
        t: "2026-09-27T10:00:00Z".into(),
        tool: "Edit".into(),
        file: "src/a.rs".into(),
        ..Default::default()
    };
    let dir = a.root.join(".ambits/coverage");
    std::fs::create_dir_all(&dir).unwrap();
    let line = serde_json::to_string(&ambits::journal::Record::Write(Box::new(write("toolu_mine")))).unwrap();
    std::fs::write(dir.join(format!("{SESSION}.ndjson")), format!("{line}\n")).unwrap();
    a.snap();
    let link = |op: &str| serde_json::json!({"k": op, "link": {"op": op, "target": "src/a.rs", "hash": null, "commit": "0".repeat(40), "verified": false, "path": "src/a.rs"}});
    std::fs::write(a.root.join(".ambits/links.ndjson"), format!("{}\n{}\n", link("toolu_mine"), link("toolu_other_session"))).unwrap();
    a.push(false).unwrap();
    let sent = std::fs::read_to_string(remote.join("links.ndjson")).unwrap();
    assert!(sent.contains("toolu_mine") && !sent.contains("toolu_other_session"), "{sent}");
}

/// Two clones pushing diverged histories at once: one wins, the other is
/// refused, and the remote's history is whole.
#[test]
fn racing_pushes_leave_one_winner_and_a_whole_history() {
    let (_dir, remote, a, b) = pair();
    a.read("alpha", "agent-1");
    a.snap();
    b.read("beta", "agent-2");
    b.snap();
    let (ra, rb) = (a.root.clone(), b.root.clone());
    let ta = std::thread::spawn(move || push_from(&ra, SESSION, false).is_ok());
    let tb = std::thread::spawn(move || push_from(&rb, SESSION, false).is_ok());
    let wins = usize::from(ta.join().unwrap()) + usize::from(tb.join().unwrap());
    assert_eq!(wins, 1);
    let tip = remote_tip(&remote).expect("the winner's tip");
    assert!(remote::transfer::closure(&Store::at_root(&remote), tip).is_ok());
}

/// Objects copied but the ref never moved (a push interrupted before its
/// lock): nothing points at them, a fetch finds nothing, the next push works.
#[test]
fn an_interrupted_push_leaves_no_ref() {
    let (_dir, remote, a, b) = pair();
    a.read("alpha", "agent-1");
    let tip = created(a.snap());
    let history = remote::transfer::closure(&a.store(), tip).unwrap();
    remote::transfer::transfer(&a.store(), &Store::at_root(&remote), &history).unwrap();
    assert_eq!(remote_tip(&remote), None);
    assert!(remote::fetch(&b.root, None, Default::default()).unwrap().moved.is_empty());
    assert!(matches!(a.push(false).unwrap(), PushOutcome::Pushed { objects: 0, .. }), "the objects are already there");
}

/// A corrupt object already on the remote is repaired by the next push,
/// not skipped as present.
#[test]
fn a_push_repairs_a_corrupt_object_it_would_have_skipped() {
    let (_dir, remote, a, _b) = pair();
    a.read("alpha", "agent-1");
    let first = created(a.snap());
    a.push(false).unwrap();
    let remote_store = Store::at_root(&remote);
    let coverage = Snapshot::load(&remote_store, first).unwrap().coverage;
    std::fs::write(remote_store.path_of(&coverage), "rot").unwrap();
    a.read("beta", "agent-1");
    a.snap();
    assert!(matches!(a.push(false).unwrap(), PushOutcome::Pushed { .. }));
    assert!(remote_store.get(&coverage, ambits::objects::Kind::Coverage).is_ok());
}

/// Damage behind the frontier — an old snapshot's object, under a newer
/// one the remote holds whole — is not looked for by a normal push (that
/// is the point), and `--verify-all` finds and repairs it.
#[test]
fn verify_all_repairs_what_the_frontier_does_not_look_at() {
    let (_dir, remote, a, _b) = pair();
    a.read("alpha", "agent-1");
    let old = created(a.snap());
    a.read("beta", "agent-1");
    created(a.snap());
    a.push(false).unwrap();
    let remote_store = Store::at_root(&remote);
    let coverage = Snapshot::load(&remote_store, old).unwrap().coverage;
    std::fs::write(remote_store.path_of(&coverage), "rot").unwrap();

    a.read("alpha", "agent-2");
    a.snap();
    a.push(false).unwrap();
    assert!(remote_store.get(&coverage, ambits::objects::Kind::Coverage).is_err(), "not looked at");

    a.read("beta", "agent-2");
    a.snap();
    remote::push(&remote::PushRequest {
        project_root: &a.root,
        remote: None,
        session: SESSION,
        force_with_lease: false,
        dry_run: false,
        ignore: &SyncIgnore::none(),
        verify: remote::transfer::Verify::Everything,
    })
    .unwrap();
    assert!(remote_store.get(&coverage, ambits::objects::Kind::Coverage).is_ok(), "repaired");

    // Damaged again with nothing new to push: --verify-all still repairs.
    std::fs::write(remote_store.path_of(&coverage), "rot").unwrap();
    let outcome = remote::push(&remote::PushRequest {
        project_root: &a.root,
        remote: None,
        session: SESSION,
        force_with_lease: false,
        dry_run: false,
        ignore: &SyncIgnore::none(),
        verify: remote::transfer::Verify::Everything,
    })
    .unwrap();
    assert!(matches!(outcome, PushOutcome::UpToDate { repaired: 1, .. }), "{outcome:?}");
    assert!(remote_store.get(&coverage, ambits::objects::Kind::Coverage).is_ok());
}

/// Two versions of one write at one attribution version: ours stays, theirs
/// is a conflict kept once.
#[test]
fn a_write_conflict_keeps_ours_and_records_theirs_once() {
    let (_dir, _remote, a, b) = pair();
    let write = |file: &str| ambits::writes::WriteRecord {
        op: "toolu_1".into(),
        av: ambits::writes::ATTRIBUTION_VERSION,
        a: "agent-1".into(),
        t: "2026-09-27T10:00:00Z".into(),
        tool: "Edit".into(),
        file: file.into(),
        ..Default::default()
    };
    for (clone, file) in [(&a, "src/a.rs"), (&b, "src/b.rs")] {
        let dir = clone.root.join(".ambits/coverage");
        std::fs::create_dir_all(&dir).unwrap();
        let line = serde_json::to_string(&ambits::journal::Record::Write(Box::new(write(file)))).unwrap();
        std::fs::write(dir.join(format!("{SESSION}.ndjson")), format!("{line}\n")).unwrap();
    }
    a.snap();
    a.push(false).unwrap();
    b.snap();
    assert!(matches!(b.pull(), PullOutcome::Merged { writes: 0, conflicts: 1, .. }));
    assert!(matches!(b.pull(), PullOutcome::Merged { conflicts: 0, .. }), "kept once");
    let contents = ambits::journal::read_journal_session(&b.root.join(".ambits/coverage"), SESSION);
    assert_eq!(contents.writes["toolu_1"].file, "src/b.rs", "ours stays");
}

/// A pull that only keeps history while behind still records the merge, so
/// the next snapshot descends from the remote and pushes as a fast-forward.
#[test]
fn history_alone_while_behind_still_merges() {
    let (_dir, remote, a, b) = pair();
    a.read("alpha", "agent-1");
    a.snap();
    a.push(false).unwrap();
    assert!(matches!(b.pull(), PullOutcome::Merged { reads: 1, .. }));
    let behind = created(b.snap());
    // A moves on reading code B has changed since.
    a.read("beta", "agent-1");
    a.snap();
    a.push(false).unwrap();
    std::fs::write(b.root.join("src/a.rs"), "fn alpha() {}\nfn beta() { 2; }\n").unwrap();
    assert!(matches!(b.pull(), PullOutcome::Merged { reads: 0, stale: 1, .. }));
    let next = created(b.snap());
    assert_ne!(next, behind);
    assert!(matches!(b.push(false).unwrap(), PushOutcome::Pushed { forced: false, .. }));
    assert_eq!(remote_tip(&remote), Some(next));
}

/// An object lost locally, and gc run over the store, then a fetch: the
/// object is back.
#[test]
fn crash_then_gc_then_fetch_recovers() {
    let (_dir, _remote, a, b) = pair();
    a.read("alpha", "agent-1");
    let tip = created(a.snap());
    a.push(false).unwrap();
    remote::fetch(&b.root, None, Default::default()).unwrap();
    let snap = Snapshot::load(&b.store(), tip).unwrap();
    std::fs::remove_file(b.store().path_of(&snap.writes)).unwrap();
    ambits::objects::gc::gc(&b.store(), std::time::Duration::ZERO, ambits::objects::refs::REFLOG_EXPIRY).unwrap();
    let report = remote::fetch(&b.root, None, Default::default()).unwrap();
    assert_eq!(report.objects, 1);
    assert!(b.store().get(&snap.writes, ambits::objects::Kind::Writes).is_ok());
}

/// A write pulled from a remote says where it came from.
#[test]
fn touched_names_the_remote_a_write_came_from() {
    let (_dir, _remote, a, b) = pair();
    let write = ambits::writes::WriteRecord {
        op: "toolu_1".into(),
        av: ambits::writes::ATTRIBUTION_VERSION,
        a: "agent-1".into(),
        t: "2026-09-27T10:00:00Z".into(),
        tool: "Edit".into(),
        file: "src/a.rs".into(),
        ..Default::default()
    };
    let dir = a.root.join(".ambits/coverage");
    std::fs::create_dir_all(&dir).unwrap();
    let line = serde_json::to_string(&ambits::journal::Record::Write(Box::new(write))).unwrap();
    std::fs::write(dir.join(format!("{SESSION}.ndjson")), format!("{line}\n")).unwrap();
    a.snap();
    a.push(false).unwrap();
    assert!(matches!(b.pull(), PullOutcome::Merged { writes: 1, .. }));
    let out = run_ambits(&b.root, &["touched", "src/a.rs"]);
    assert!(out.contains("pulled from origin"), "{out}");
    let json = run_ambits(&b.root, &["touched", "--format", "json", "src/a.rs"]);
    assert!(json.contains(r#""origin":"origin""#), "{json}");
    assert!(!run_ambits(&a.root, &["touched", "src/a.rs"]).contains("pulled from"), "A's own");
}

/// The CLI: add a remote, push, and on the other clone fetch and restore
/// the remote's session into a new one — cross-machine restore.
#[test]
fn the_cli_pushes_fetches_and_restores_across_clones() {
    let (_dir, remote, a, b) = pair();
    a.read("alpha", "agent-1");
    run_ambits(&a.root, &["--session", SESSION, "snapshot"]);
    let out = run_ambits(&a.root, &["--session", SESSION, "push", "--dry-run"]);
    assert!(out.contains("would push to origin"), "{out}");
    assert!(remote_tip(&remote).is_none(), "a dry run sends nothing");
    let out = run_ambits(&a.root, &["--session", SESSION, "push"]);
    assert!(out.contains("origin: (none) →"), "{out}");
    assert!(run_ambits(&a.root, &["remote", "list"]).starts_with("origin\t"));

    let out = run_ambits(&b.root, &["fetch"]);
    assert!(out.contains(&format!("origin/{SESSION}: (none) →")), "{out}");
    let out = run_ambits(&b.root, &["restore", &format!("origin/{SESSION}")]);
    assert!(out.contains("src/a.rs: 1 verified"), "{out}");
}

/// Not a check but a measurement, run by hand:
/// `cargo test --release --test remote history_walk_costs -- --ignored --nocapture`.
/// A long history, then what the common operations cost once it is shared.
#[test]
#[ignore]
fn history_walk_costs() {
    let n: usize = std::env::var("SNAPSHOTS").ok().and_then(|s| s.parse().ok()).unwrap_or(200);
    let (_dir, _remote, a, b) = pair();
    for i in 0..n {
        a.read(if i % 2 == 0 { "alpha" } else { "beta" }, &format!("agent-{i}"));
        a.snap();
    }
    let time = |what: &str, f: &mut dyn FnMut()| {
        let t = std::time::Instant::now();
        f();
        eprintln!("{what:<32} {:>8.1} ms", t.elapsed().as_secs_f64() * 1000.0);
    };
    time("first push", &mut || { let _ = a.push(false).unwrap(); });
    time("push, nothing new", &mut || { let _ = a.push(false).unwrap(); });
    a.read("alpha", "agent-last");
    time("snapshot, one new read", &mut || { let _ = a.snap(); });
    time("push, one new snapshot", &mut || { let _ = a.push(false).unwrap(); });
    time("first fetch", &mut || { let _ = remote::fetch(&b.root, None, Default::default()).unwrap(); });
    time("fetch, nothing new", &mut || { let _ = remote::fetch(&b.root, None, Default::default()).unwrap(); });
    time("pull (merge)", &mut || { let _ = b.pull(); });
    time("snapshot after pull", &mut || { let _ = b.snap(); });
    time("pull, up to date", &mut || { let _ = b.pull(); });
    time("snapshot, nothing changed", &mut || { let _ = b.snap(); });
}
