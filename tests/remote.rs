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
        let path = dir.join(format!("{SESSION}.ndjson"));
        let mut text = std::fs::read_to_string(&path).unwrap_or_default();
        text.push_str(&serde_json::to_string(&record).unwrap());
        text.push('\n');
        std::fs::write(path, text).unwrap();
    }

    fn snap(&self) -> Outcome {
        let tree = self.registry.scan_project(&self.root, None).unwrap();
        snapshot(&ambits::objects::snapshot::Request {
            project_root: &self.root,
            session: SESSION,
            tree: &tree,
            sync: &SyncConfig::default(),
            message: None,
            require_clean: false,
        })
        .unwrap()
    }

    fn push(&self, force: bool) -> color_eyre::Result<PushOutcome> {
        remote::push(&remote::PushRequest {
            project_root: &self.root,
            remote: None,
            session: SESSION,
            force_with_lease: force,
            dry_run: false,
            ignore: &SyncIgnore::none(),
        })
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

    assert!(matches!(b.pull(), PullOutcome::Merged { reads: 1, merged: true, .. }));
    assert_eq!(b.reads(), vec!["src/a.rs::alpha"]);
    let s2 = created(b.snap());
    assert_eq!(Snapshot::load(&b.store(), s2).unwrap().parents, vec![s1], "descends from A's");
    assert!(matches!(b.push(false).unwrap(), PushOutcome::Pushed { forced: false, .. }));

    for _ in 0..2 {
        assert!(matches!(a.pull(), PullOutcome::Behind | PullOutcome::UpToDate));
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

    assert!(matches!(b.pull(), PullOutcome::Merged { reads: 0, writes: 0, merged: true, .. }));
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
    remote::fetch(&b.root, None).unwrap();
    assert!(matches!(b.push(true).unwrap(), PushOutcome::Pushed { forced: true, .. }));
    assert_eq!(remote_tip(&remote), Some(forced));

    let report = remote::fetch(&a.root, None).unwrap();
    assert!(report.moved.iter().any(|m| m.forced && m.from == Some(s1) && m.to == forced));
    assert!(matches!(a.pull(), PullOutcome::Merged { reads: 1, merged: true, .. }));
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

/// Fetch verifies: a tampered object, a symlink, or an oversized file on
/// the remote is refused and nothing is recorded.
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
    let mut tampered = String::from_utf8(original.clone()).unwrap();
    tampered = tampered.replace("full_body", "overview");
    std::fs::write(&coverage, tampered).unwrap();
    assert!(remote::fetch(&b.root, None).is_err(), "tampered");
    assert_eq!(refs::read(&b.store(), &tracking).unwrap(), None);

    std::fs::remove_file(&coverage).unwrap();
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink("/etc/hosts", &coverage).unwrap();
        assert!(remote::fetch(&b.root, None).is_err(), "symlink");
        std::fs::remove_file(&coverage).unwrap();
    }
    std::fs::File::create(&coverage).unwrap().set_len(17 * 1024 * 1024).unwrap();
    assert!(remote::fetch(&b.root, None).is_err(), "oversized");
    assert_eq!(refs::read(&b.store(), &tracking).unwrap(), None);

    std::fs::write(&coverage, original).unwrap();
    remote::fetch(&b.root, None).unwrap();
    assert_eq!(refs::read(&b.store(), &tracking).unwrap(), Some(tip));
}

/// An object lost locally (a crash, then gc) comes back with the next fetch.
#[test]
fn a_fetch_recovers_objects_lost_locally() {
    let (_dir, _remote, a, b) = pair();
    a.read("alpha", "agent-1");
    let tip = created(a.snap());
    a.push(false).unwrap();
    remote::fetch(&b.root, None).unwrap();
    let snap = Snapshot::load(&b.store(), tip).unwrap();
    std::fs::remove_file(b.store().path_of(&snap.writes)).unwrap();
    let report = remote::fetch(&b.root, None).unwrap();
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
