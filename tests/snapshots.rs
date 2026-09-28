//! Snapshots end to end (spec §6–§8; the phase-3 key tests of §14), against
//! a real git repository in a temp directory.

use std::path::PathBuf;
use std::time::{Duration, SystemTime};

mod common;

use ambits::ingest::tool_config::SyncConfig;
use common::{git, run_ambits};
use ambits::objects::gc;
use ambits::objects::snapshot::{snapshot, Outcome, Request, Snapshot};
use ambits::objects::store::Store;
use ambits::objects::{Kind, ObjectId};
use ambits::parser::ParserRegistry;

const SESSION: &str = "0b7e9d3a-1c2f-4e5a-9b8c-7d6e5f4a3b2c";
const READ: &str = r#"{"kind":"read","sym":"src/a.rs::alpha","h":"b3:0000000000000000000000000000000000000000000000000000000000000000","d":"full_body","a":"agent-1"}"#;

struct Project {
    _dir: tempfile::TempDir,
    root: PathBuf,
    registry: ParserRegistry,
}

impl Project {
    /// A committed repo with `src/a.rs`, `src/b.rs` and a journal of one read.
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/a.rs"), "fn alpha() {}\n").unwrap();
        std::fs::write(root.join("src/b.rs"), "fn beta() {}\n").unwrap();
        std::fs::write(root.join(".gitignore"), ".ambits/\n").unwrap();
        git(&root, &["init", "-q"]);
        git(&root, &["add", "."]);
        git(&root, &["commit", "-qm", "init"]);
        let project = Self { _dir: dir, root, registry: ParserRegistry::new() };
        project.append_journal(&format!("{READ}\n"));
        project
    }

    fn append_journal(&self, text: &str) {
        use std::io::Write as _;
        let dir = self.root.join(".ambits/coverage");
        std::fs::create_dir_all(&dir).unwrap();
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(dir.join(format!("{SESSION}.ndjson"))).unwrap();
        f.write_all(text.as_bytes()).unwrap();
    }

    fn snapshot_with(&self, sync: &SyncConfig) -> Outcome {
        let tree = self.registry.scan_project(&self.root, None).unwrap();
        snapshot(&Request {
            project_root: &self.root,
            session: SESSION,
            tree: &tree,
            sync,
            message: None,
            require_clean: false,
        })
        .unwrap()
    }

    fn snap(&self) -> Outcome {
        self.snapshot_with(&SyncConfig::default())
    }

    fn store(&self) -> Store {
        Store::at(&self.root)
    }
}

fn created(outcome: Outcome) -> ObjectId {
    match outcome {
        Outcome::Created { id, .. } => id,
        Outcome::NothingChanged(tip) => panic!("expected a new snapshot, got nothing changed at {tip:?}"),
    }
}

fn unchanged(outcome: Outcome) -> ObjectId {
    match outcome {
        Outcome::NothingChanged(tip) => tip,
        Outcome::Created { id, .. } => panic!("expected nothing changed, got new snapshot {id:?}"),
    }
}

#[test]
fn snapshotting_twice_is_a_no_op() {
    let p = Project::new();
    let first = created(p.snap());
    assert_eq!(unchanged(p.snap()), first);
    assert_eq!(p.store().list().iter().filter(|(id, _)| p.store().get(id, Kind::Snapshot).is_ok()).count(), 1);
}

/// Only content counts: a touched file is the same state.
#[test]
fn touching_a_dirty_file_changes_nothing() {
    let p = Project::new();
    std::fs::write(p.root.join("src/a.rs"), "fn alpha() { 1; }\n").unwrap();
    let first = created(p.snap());
    let file = std::fs::File::options().write(true).open(p.root.join("src/a.rs")).unwrap();
    file.set_modified(SystemTime::now() + Duration::from_secs(60)).unwrap();
    assert_eq!(unchanged(p.snap()), first);
}

/// Raw bytes, not symbol hashes: whitespace and same-length edits count (§7).
#[test]
fn whitespace_and_same_length_edits_make_a_new_snapshot() {
    let p = Project::new();
    std::fs::write(p.root.join("src/a.rs"), "fn alpha() { 1; }\n").unwrap();
    let mut tip = created(p.snap());
    for edit in ["fn alpha() { 1;  }\n", "fn alpha() { 2;  }\n"] {
        std::fs::write(p.root.join("src/a.rs"), edit).unwrap();
        let next = created(p.snap());
        assert_eq!(Snapshot::load(&p.store(), next).unwrap().parents, vec![tip]);
        tip = next;
    }
}

/// History never rewinds: returning to an earlier state is a new snapshot
/// whose parent is the tip, not the earlier id (§6.4).
#[test]
fn reverting_makes_a_new_snapshot_on_top_of_the_tip() {
    let p = Project::new();
    let clean = created(p.snap());
    std::fs::write(p.root.join("src/a.rs"), "fn alpha() { 1; }\n").unwrap();
    let dirty = created(p.snap());
    std::fs::write(p.root.join("src/a.rs"), "fn alpha() {}\n").unwrap();
    let reverted = created(p.snap());
    assert_ne!(reverted, clean);
    let s = Snapshot::load(&p.store(), reverted).unwrap();
    assert_eq!(s.parents, vec![dirty]);
    assert_eq!(s.coverage, Snapshot::load(&p.store(), clean).unwrap().coverage, "same coverage as before");
}

/// The journal is an input: a new record is a new snapshot; a torn,
/// half-written line is not part of it yet (§6.2).
#[test]
fn a_journal_append_counts_and_a_torn_line_does_not() {
    let p = Project::new();
    let first = created(p.snap());
    let second_read = READ.replace("src/a.rs::alpha", "src/b.rs::beta");
    p.append_journal(&second_read[..30]);
    assert_eq!(unchanged(p.snap()), first);
    p.append_journal(&format!("{}\n", &second_read[30..]));
    let second = created(p.snap());
    let s = Snapshot::load(&p.store(), second).unwrap();
    let coverage = p.store().get(&s.coverage, Kind::Coverage).unwrap();
    assert_eq!(coverage["reads"].as_array().unwrap().len(), 2);
}

/// A journal schema upgrade appends a new header: the id changes once.
#[test]
fn a_journal_schema_upgrade_changes_the_id_once() {
    let p = Project::new();
    let v2 = format!(
        r#"{{"kind":"header","schema_version":2,"created_at":"0","session_id":"{SESSION}","project_root":"/p","tree_fingerprint":"b3:00","ambit_version":"0","backend":"tree-sitter","os":"x","arch":"y"}}"#
    );
    std::fs::write(p.root.join(format!(".ambits/coverage/{SESSION}.ndjson")), format!("{v2}\n{READ}\n")).unwrap();
    let before = created(p.snap());

    let open = || {
        drop(ambits::journal::Journal::open(&p.root, SESSION, Duration::ZERO, || {
            ambits::journal::EnvironmentManifest::capture(&p.registry.scan_project(&p.root, None).unwrap(), "tree-sitter", None)
        }))
    };
    open();
    let upgraded = created(p.snap());
    assert_ne!(upgraded, before);
    open();
    assert_eq!(unchanged(p.snap()), upgraded, "a second open appends nothing");
}

/// `[sync] ignore` covers every record type: tree, reads, writes and the
/// dirty list (§4). Nothing names the ignored directory.
#[test]
fn ignored_paths_appear_in_no_object() {
    let p = Project::new();
    std::fs::create_dir_all(p.root.join("secret")).unwrap();
    std::fs::write(p.root.join("secret/key.rs"), "fn topsecret() {}\n").unwrap();
    p.append_journal(&format!(
        "{}\n{}\n",
        READ.replace("src/a.rs::alpha", "secret/key.rs::topsecret"),
        r#"{"kind":"write","op":"toolu_1","av":2,"a":"agent-1","t":"2026-09-26T10:00:00Z","tool":"Edit","file":"secret/key.rs","level":"file"}"#
    ));
    let sync = SyncConfig { ignore: Some(vec!["secret/".into()]), global_ignore: vec![] };
    let id = created(p.snapshot_with(&sync));

    for (_, path) in p.store().list() {
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(!text.contains("secret") && !text.contains("topsecret"), "{} leaks: {text}", path.display());
    }
    let s = Snapshot::load(&p.store(), id).unwrap();
    assert!(s.inputs.dirty.is_empty(), "the untracked ignored file is not dirty");
}

#[test]
fn require_clean_refuses_a_dirty_tree() {
    let p = Project::new();
    std::fs::write(p.root.join("src/a.rs"), "fn alpha() { 1; }\n").unwrap();
    let tree = p.registry.scan_project(&p.root, None).unwrap();
    let sync = SyncConfig::default();
    let result = snapshot(&Request {
        project_root: &p.root,
        session: SESSION,
        tree: &tree,
        sync: &sync,
        message: None,
        require_clean: true,
    });
    assert!(result.is_err());
}

/// Every object a present object references is present too.
fn assert_closed(store: &Store) {
    for (id, path) in store.list() {
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let text = v["payload"].to_string();
        for hex in text.split('"').filter(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())) {
            assert!(store.contains(&ObjectId::parse(hex).unwrap()), "{id:?} references missing {hex}");
        }
    }
}

/// A crash between writing objects and moving the ref leaves unreachable
/// objects. gc must never leave a present object without its children, and
/// the next snapshot must succeed whatever gc removed (§8, §16).
#[test]
fn crash_then_gc_then_snapshot_never_loses_an_object() {
    let p = Project::new();
    let kept = created(p.snap());
    // A "crashed" snapshot: its objects are written, the ref was not moved.
    std::fs::write(p.root.join("src/a.rs"), "fn alpha() { 9; }\n").unwrap();
    let orphan = created(p.snap());
    let ref_path = p.root.join(format!(".ambits/refs/sessions/{SESSION}"));
    std::fs::write(&ref_path, format!("{kept}\n")).unwrap();
    let _ = std::fs::remove_dir_all(p.root.join(".ambits/logs"));

    // Within the grace period nothing unreachable goes.
    let young = gc::gc(&p.store(), gc::DEFAULT_GRACE, ambits::objects::refs::REFLOG_EXPIRY).unwrap();
    assert!(young.deleted.is_empty());
    assert!(p.store().contains(&orphan));

    let collected = gc::gc(&p.store(), Duration::ZERO, ambits::objects::refs::REFLOG_EXPIRY).unwrap();
    assert!(collected.deleted.contains(&orphan));
    assert_closed(&p.store());
    assert!(Snapshot::load(&p.store(), kept).is_ok());

    let again = created(p.snap());
    assert_eq!(again, orphan, "same inputs and parent: the same id, rebuilt");
    assert_closed(&p.store());
}

/// Parents before children: each deleted object's unreachable referrers
/// were deleted before it.
#[test]
fn gc_deletes_parents_before_children() {
    let p = Project::new();
    let _ = created(p.snap());
    std::fs::write(p.root.join("src/a.rs"), "fn alpha() { 1; }\n").unwrap();
    let _ = created(p.snap());
    std::fs::remove_dir_all(p.root.join(".ambits/refs")).unwrap();
    std::fs::remove_dir_all(p.root.join(".ambits/logs")).unwrap();

    // Record who references whom before gc removes it all.
    let mut references: Vec<(ObjectId, Vec<ObjectId>)> = Vec::new();
    for (id, path) in p.store().list() {
        let text = std::fs::read_to_string(path).unwrap();
        let refs = text
            .split('"')
            .filter(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()))
            .filter_map(|h| ObjectId::parse(h).ok())
            .collect();
        references.push((id, refs));
    }
    let stats = gc::gc(&p.store(), Duration::ZERO, ambits::objects::refs::REFLOG_EXPIRY).unwrap();
    assert!(p.store().list().is_empty(), "everything was unreachable");
    let position = |id: &ObjectId| stats.deleted.iter().position(|d| d == id).unwrap();
    for (parent, children) in &references {
        for child in children {
            assert!(position(parent) < position(child), "{parent:?} deleted after its child {child:?}");
        }
    }
}

/// Reusing an object refreshes its age, so gc's grace period protects what
/// a snapshot just referenced even when it was written long ago (§8).
#[test]
fn reusing_an_object_refreshes_its_age() {
    let p = Project::new();
    let first = created(p.snap());
    // A new read changes the coverage object, not the writes object, so
    // the next snapshot reuses the latter.
    let writes = Snapshot::load(&p.store(), first).unwrap().writes;
    let path = p.store().path_of(&writes);
    let old = SystemTime::now() - Duration::from_secs(30 * 24 * 60 * 60);
    std::fs::File::options().write(true).open(&path).unwrap().set_modified(old).unwrap();

    p.append_journal(&format!("{}\n", READ.replace("src/a.rs::alpha", "src/b.rs::beta")));
    created(p.snap());
    let age = SystemTime::now().duration_since(std::fs::metadata(&path).unwrap().modified().unwrap()).unwrap();
    assert!(age < Duration::from_secs(60), "age not refreshed: {age:?}");
}

/// The CLI: snapshot, a no-op, and a log that shows both the message and
/// the git pin.
#[test]
fn the_cli_snapshots_and_logs() {
    let p = Project::new();
    let run = |args: &[&str]| run_ambits(&p.root, &[&["--session", SESSION], args].concat());
    let made = run(&["snapshot", "-m", "first"]);
    assert!(made.starts_with("snapshot "), "{made}");
    assert!(run(&["snapshot"]).starts_with("nothing changed: "));
    let log = run(&["log"]);
    assert!(log.contains("first") && log.contains("reads 1"), "{log}");
    let gc = run(&["gc"]);
    assert!(gc.contains("0 deleted"), "{gc}");
}

/// File contents are never persisted (§9.6): a marker inside a function body
/// — in a committed file and in a dirty one — appears in no object or note.
#[test]
fn file_contents_never_reach_objects_or_notes() {
    const CANARY: &str = "CANARY_5e1d_never_persist";
    let p = Project::new();
    std::fs::write(p.root.join("src/a.rs"), format!("fn alpha() {{ let _ = \"{CANARY}\"; }}\n")).unwrap();
    git(&p.root, &["commit", "-qam", "canary"]);
    std::fs::write(p.root.join("src/b.rs"), format!("fn beta() {{ /* {CANARY} */ }}\n")).unwrap();
    created(p.snap());

    let mut files = vec![];
    for dir in ["objects", "notes", "refs", "logs"] {
        let mut stack = vec![p.root.join(".ambits").join(dir)];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
                if e.path().is_dir() { stack.push(e.path()) } else { files.push(e.path()) }
            }
        }
    }
    assert!(!files.is_empty());
    for f in files {
        let text = std::fs::read_to_string(&f).unwrap_or_default();
        assert!(!text.contains(CANARY), "{} leaks file contents", f.display());
    }
}

/// A tip whose objects were lost (a power cut after the ref moved) is
/// repaired from the unchanged state rather than reported as fine.
#[test]
fn a_tip_with_missing_objects_is_repaired() {
    let p = Project::new();
    let id = created(p.snap());
    let coverage = Snapshot::load(&p.store(), id).unwrap().coverage;
    std::fs::remove_file(p.store().path_of(&coverage)).unwrap();
    assert_eq!(unchanged(p.snap()), id);
    assert!(p.store().contains(&coverage));
}

/// A field no digest covers — a forged `host`, say — makes the snapshot
/// unreadable rather than silently carried.
#[test]
fn an_extra_field_in_a_snapshot_is_refused() {
    let p = Project::new();
    let id = created(p.snap());
    let path = p.store().path_of(&id);
    let mut envelope: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    envelope["payload"]["host"] = serde_json::json!("laptop");
    std::fs::write(&path, ambits::objects::canonical::to_bytes(&envelope).unwrap()).unwrap();
    assert!(Snapshot::load(&p.store(), id).is_err());
}

/// Temp files a crash left are collected, and a leftover one in `refs/`
/// is never mistaken for a ref.
#[test]
fn gc_collects_leftover_temp_files() {
    let p = Project::new();
    created(p.snap());
    let stray = p.root.join(".ambits/refs/sessions/.tmp-deadbeef");
    std::fs::write(&stray, format!("{}\n", "0".repeat(64))).unwrap();
    assert!(ambits::objects::refs::all(&p.store()).iter().all(|(name, _)| !name.contains(".tmp-")));
    let stats = gc::gc(&p.store(), Duration::ZERO, ambits::objects::refs::REFLOG_EXPIRY).unwrap();
    assert_eq!(stats.temp_files_removed, 1);
    assert!(!stray.exists());
}

/// A snapshot from before the tree was dropped (object format 1) is refused
/// with a message saying what to do, not reported as corrupt.
#[test]
fn an_older_format_snapshot_is_refused_with_advice() {
    let p = Project::new();
    let id = created(p.snap());
    let path = p.store().path_of(&id);
    let mut envelope: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    envelope["payload"].as_object_mut().unwrap().remove("format");
    std::fs::write(&path, ambits::objects::canonical::to_bytes(&envelope).unwrap()).unwrap();
    let err = Snapshot::load(&p.store(), id).unwrap_err().to_string();
    assert!(err.contains("older ambits") && err.contains("delete .ambits/objects"), "{err}");
}
