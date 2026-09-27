//! `ambits restore` end to end (spec §12.1; the phase-5 key tests of §14).

mod common;

use std::path::PathBuf;

use ambits::ingest::tool_config::SyncConfig;
use ambits::objects::inputs::Backend;
use ambits::objects::restore::{restore, Report, Request};
use ambits::objects::snapshot::{snapshot, Outcome, Snapshot};
use ambits::objects::store::Store;
use ambits::objects::ObjectId;
use ambits::parser::ParserRegistry;
use common::{git, run_ambits};

const SOURCE: &str = "0b7e9d3a-1c2f-4e5a-9b8c-7d6e5f4a3b2c";
const TARGET: &str = "1c8f0e4b-2d3a-4f6b-8c9d-8e7f6a5b4c3d";

struct Project {
    _dir: tempfile::TempDir,
    root: PathBuf,
    registry: ParserRegistry,
}

impl Project {
    /// A committed repo with `src/a.rs` (`alpha`, `beta`) and `src/b.rs`
    /// (`gamma`), and a source session that read all three in full and
    /// wrote `alpha`.
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/a.rs"), "fn alpha() {}\nfn beta() {}\n").unwrap();
        std::fs::write(root.join("src/b.rs"), "fn gamma() {}\n").unwrap();
        std::fs::write(root.join(".gitignore"), ".ambits/\n").unwrap();
        git(&root, &["init", "-q"]);
        git(&root, &["add", "."]);
        git(&root, &["commit", "-qm", "init"]);
        let p = Self { _dir: dir, root, registry: ParserRegistry::new() };

        let tree = p.registry.scan_project(&p.root, None).unwrap();
        let mut lines = String::new();
        for (_, sym) in tree.walk() {
            let read = ambits::journal::Record::Read {
                symbol_id: sym.id.clone(),
                hash: ambits::journal::encode_hash(&sym.content_hash),
                depth: ambits::journal::DepthDto::FullBody,
                agent: Some("agent-1".into()),
            };
            lines.push_str(&serde_json::to_string(&read).unwrap());
            lines.push('\n');
        }
        let write = ambits::writes::WriteRecord {
            op: "toolu_1".into(),
            a: "agent-1".into(),
            t: "2026-09-26T10:00:00Z".into(),
            tool: "Edit".into(),
            file: "src/a.rs".into(),
            ..Default::default()
        };
        lines.push_str(&serde_json::to_string(&ambits::journal::Record::Write(Box::new(write))).unwrap());
        lines.push('\n');
        let dir = p.root.join(".ambits/coverage");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(format!("{SOURCE}.ndjson")), lines).unwrap();
        p
    }

    fn snapshot(&self, session: &str) -> Outcome {
        let tree = self.registry.scan_project(&self.root, None).unwrap();
        snapshot(&ambits::objects::snapshot::Request {
            project_root: &self.root,
            session,
            tree: &tree,
            backend: Backend::TreeSitter(&self.registry),
            filter: None,
            sync: &SyncConfig::default(),
            message: None,
            require_clean: false,
        })
        .unwrap()
    }

    fn restore(&self, reference: &str, into: Option<&str>) -> Report {
        let tree = self.registry.scan_project(&self.root, None).unwrap();
        restore(&Request { project_root: &self.root, reference, into, tree: &tree, backend: "tree-sitter", filter: None }).unwrap()
    }

    fn shard(&self, session: &str) -> String {
        std::fs::read_to_string(self.root.join(format!(".ambits/coverage/{session}.restore.ndjson"))).unwrap_or_default()
    }

    fn target_reads(&self, session: &str) -> Vec<String> {
        let contents = ambits::journal::read_journal_session(&self.root.join(".ambits/coverage"), session);
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

/// Into a new session: a minted id, every valid read, the write as history,
/// and the new session's ref at the snapshot.
#[test]
fn restoring_into_a_new_session_mints_it_and_fills_it() {
    let p = Project::new();
    let id = created(p.snapshot(SOURCE));
    let report = p.restore(SOURCE, None);
    assert!(ambits::ingest::claude::is_uuid(&report.session), "{}", report.session);
    assert_eq!((report.reads_appended, report.writes_appended), (3, 1));
    assert_eq!(p.target_reads(&report.session), vec!["src/a.rs::alpha", "src/a.rs::beta", "src/b.rs::gamma"]);
    assert!(p.shard(&report.session).contains(r#""kind":"history","of":"write""#));
    let store = Store::at(&p.root);
    let tip = ambits::objects::refs::read(&store, &ambits::objects::refs::RefName::session(&report.session).unwrap()).unwrap();
    assert_eq!(tip, Some(id));
    assert!(report.pinned_elsewhere.is_none(), "same commit: no warning");
}

/// Restoring twice changes nothing: no record, no ref move.
#[test]
fn restoring_twice_changes_nothing() {
    let p = Project::new();
    created(p.snapshot(SOURCE));
    p.restore(SOURCE, Some(TARGET));
    let (shard, reflog) = (p.shard(TARGET), std::fs::read_to_string(p.root.join(format!(".ambits/logs/refs/sessions/{TARGET}"))).unwrap());
    let again = p.restore(SOURCE, Some(TARGET));
    assert_eq!((again.reads_appended, again.writes_appended, again.ref_moved), (0, 0, false));
    assert_eq!(p.shard(TARGET), shard);
    assert_eq!(std::fs::read_to_string(p.root.join(format!(".ambits/logs/refs/sessions/{TARGET}"))).unwrap(), reflog);
}

/// The target's next snapshot descends from the restored one.
#[test]
fn the_targets_next_snapshot_descends_from_the_restored_one() {
    let p = Project::new();
    let restored = created(p.snapshot(SOURCE));
    p.restore(SOURCE, Some(TARGET));
    let next = created(p.snapshot(TARGET));
    assert_eq!(Snapshot::load(&Store::at(&p.root), next).unwrap().parents, vec![restored]);
}

/// A symbol moved to another file since is restored at its new address.
#[test]
fn a_moved_symbol_is_restored_where_it_lives_now() {
    let p = Project::new();
    created(p.snapshot(SOURCE));
    std::fs::write(p.root.join("src/a.rs"), "fn alpha() {}\n").unwrap();
    std::fs::write(p.root.join("src/b.rs"), "fn gamma() {}\nfn beta() {}\n").unwrap();
    let report = p.restore(SOURCE, Some(TARGET));
    assert_eq!(report.files["src/a.rs"].moved, 1);
    assert!(p.target_reads(TARGET).contains(&"src/b.rs::beta".to_string()));
    assert!(!p.target_reads(TARGET).contains(&"src/a.rs::beta".to_string()));
}

/// A symbol changed since is reported drifted and not restored.
#[test]
fn a_drifted_symbol_is_reported_and_left_out() {
    let p = Project::new();
    created(p.snapshot(SOURCE));
    std::fs::write(p.root.join("src/a.rs"), "fn alpha() { 1; }\nfn beta() {}\n").unwrap();
    git(&p.root, &["commit", "-qam", "edit"]);
    let report = p.restore(SOURCE, Some(TARGET));
    assert_eq!(report.files["src/a.rs"].drifted, 1);
    assert!(!p.target_reads(TARGET).contains(&"src/a.rs::alpha".to_string()));
}

/// A snapshot on another commit is restored with a warning, never a checkout.
#[test]
fn a_snapshot_on_another_commit_warns() {
    let p = Project::new();
    created(p.snapshot(SOURCE));
    let before = git(&p.root, &["rev-parse", "HEAD"]);
    std::fs::write(p.root.join("README.md"), "readme\n").unwrap();
    git(&p.root, &["add", "."]);
    git(&p.root, &["commit", "-qm", "later"]);
    let report = p.restore(SOURCE, Some(TARGET));
    let (pinned, head) = report.pinned_elsewhere.clone().expect("warned");
    assert_eq!(pinned.as_deref(), Some(before.as_str()));
    assert_ne!(head, pinned);
    assert_ne!(git(&p.root, &["rev-parse", "HEAD"]), before, "nothing was checked out");
}

/// What was read in a file that was dirty when snapshotted, and has since
/// changed, cannot be verified on this machine.
#[test]
fn reads_of_a_since_changed_dirty_file_are_unverifiable() {
    let p = Project::new();
    std::fs::write(p.root.join("src/b.rs"), "fn gamma() { 1; }\n").unwrap();
    // The session read the dirty version.
    let tree = p.registry.scan_project(&p.root, None).unwrap();
    let gamma = tree.walk().into_iter().find(|(_, s)| &*s.name == "gamma").unwrap().1.clone();
    let read = ambits::journal::Record::Read {
        symbol_id: gamma.id.clone(),
        hash: ambits::journal::encode_hash(&gamma.content_hash),
        depth: ambits::journal::DepthDto::FullBody,
        agent: Some("agent-1".into()),
    };
    std::fs::write(
        p.root.join(format!(".ambits/coverage/{SOURCE}.ndjson")),
        format!("{}\n", serde_json::to_string(&read).unwrap()),
    )
    .unwrap();
    created(p.snapshot(SOURCE));
    git(&p.root, &["checkout", "-q", "--", "src/b.rs"]);
    let report = p.restore(SOURCE, Some(TARGET));
    assert_eq!(report.files["src/b.rs"].unverifiable, 1);
    assert_eq!(report.files["src/b.rs"].drifted, 0);
}

/// The CLI mints, reports per file, and a second run says nothing moved.
#[test]
fn the_cli_restores_and_reports() {
    let p = Project::new();
    run_ambits(&p.root, &["--session", SOURCE, "snapshot"]);
    let out = run_ambits(&p.root, &["restore", SOURCE, "--into", TARGET]);
    assert!(out.contains(&format!("into session {TARGET}")), "{out}");
    assert!(out.contains("src/a.rs: 2 verified"), "{out}");
    let again = run_ambits(&p.root, &["restore", SOURCE, "--into", TARGET]);
    assert!(again.contains("0 read(s) and 0 write(s) appended; the session already pointed at this snapshot"), "{again}");
}
