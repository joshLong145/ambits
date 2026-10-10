//! `--state-dir` and `AMBITS_STATE_DIR` move every state file out of the
//! project, so a checkout an agent commits from never gains a `.ambits/`.

use std::path::{Path, PathBuf};

mod common;

use common::{ambits, git};

const SESSION: &str = "0b7e9d3a-1c2f-4e5a-9b8c-7d6e5f4a3b2c";
const READ: &str = r#"{"kind":"read","sym":"src/a.rs::alpha","h":"b3:0000000000000000000000000000000000000000000000000000000000000000","d":"full_body","a":"agent-1"}"#;

struct Fixture {
    _dir: tempfile::TempDir,
    root: PathBuf,
    state: PathBuf,
}

/// A committed repo with no `.gitignore` for `.ambits/`, and a journal of one
/// read already in the external state directory.
fn fixture() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().canonicalize().unwrap();
    let root = base.join("repo");
    let state = base.join("state");
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(root.join("src/a.rs"), "fn alpha() {}\n").unwrap();
    git(&root, &["init", "-q"]);
    git(&root, &["add", "."]);
    git(&root, &["commit", "-qm", "init"]);
    std::fs::create_dir_all(state.join("coverage")).unwrap();
    std::fs::write(state.join("coverage").join(format!("{SESSION}.ndjson")), format!("{READ}\n")).unwrap();
    Fixture { _dir: dir, root, state }
}

fn assert_project_untouched(root: &Path) {
    assert!(!root.join(".ambits").exists(), "state leaked into the project");
    assert_eq!(git(root, &["status", "--porcelain"]), "", "the work tree changed");
}

#[test]
fn the_flag_keeps_snapshots_out_of_the_project() {
    let f = fixture();
    let state = f.state.to_str().unwrap();
    let out = ambits(&f.root).args(["--state-dir", state, "--session", SESSION, "snapshot"]).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));

    assert!(f.state.join("objects").is_dir(), "the snapshot went elsewhere");
    assert!(f.state.join("reflog.ndjson").is_file());
    assert_project_untouched(&f.root);
}

#[test]
fn the_environment_variable_does_the_same() {
    let f = fixture();
    let run = |args: &[&str]| {
        let out = ambits(&f.root).env("AMBITS_STATE_DIR", &f.state).args(args).output().unwrap();
        assert!(out.status.success(), "ambits {args:?}: {}", String::from_utf8_lossy(&out.stderr));
        String::from_utf8(out.stdout).unwrap()
    };
    assert!(run(&["--session", SESSION, "snapshot", "-m", "first"]).starts_with("snapshot "));
    let log = run(&["--session", SESSION, "log"]);
    assert!(log.contains("first") && log.contains("reads 1"), "{log}");
    assert_project_untouched(&f.root);
}

#[test]
fn a_relative_state_dir_resolves_against_the_working_directory() {
    let f = fixture();
    let out = ambits(&f.root).args(["--state-dir", "../state", "--session", SESSION, "snapshot"]).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(f.state.join("objects").is_dir());
    assert_project_untouched(&f.root);
}

#[test]
fn a_remote_inside_the_external_store_is_refused() {
    let f = fixture();
    let inside = f.state.join("objects");
    std::fs::create_dir_all(&inside).unwrap();
    let out = ambits(&f.root)
        .args(["--state-dir", f.state.to_str().unwrap(), "remote", "add", "self", inside.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!out.status.success(), "a remote inside the store was accepted");
    assert!(String::from_utf8_lossy(&out.stderr).contains("own store"), "{}", String::from_utf8_lossy(&out.stderr));
}
