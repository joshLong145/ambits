//! Git linkage end to end (spec §3.2, §9.3; the phase-4 key tests of §14):
//! which commit an agent's write landed in.

use std::path::{Path, PathBuf};

mod common;

use ambits::linkage::{landed, Landed, Resolver};
use common::{git, git_env, run_ambits};
use ambits::parser::ParserRegistry;
use ambits::writes::{Level, WriteRecord};

const SESSION: &str = "0b7e9d3a-1c2f-4e5a-9b8c-7d6e5f4a3b2c";

struct Repo {
    _dir: tempfile::TempDir,
    root: PathBuf,
}

impl Repo {
    /// A repository whose first commit has `src/a.rs` with `alpha` and `beta`.
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().canonicalize().unwrap();
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/a.rs"), "fn alpha() {}\nfn beta() {}\n").unwrap();
        std::fs::write(root.join(".gitignore"), ".ambits/\n").unwrap();
        git(&root, &["init", "-q", "-b", "main"]);
        git(&root, &["add", "."]);
        // Two hours back, well before any write a test makes.
        let past = format!("@{} +0000", ambits::time::now_secs() - 7200);
        git_env(&root, &["commit", "-qm", "init"], &[("GIT_AUTHOR_DATE", &past), ("GIT_COMMITTER_DATE", &past)]);
        Self { _dir: dir, root }
    }

    fn write_file(&self, rel: &str, content: &str) {
        std::fs::write(self.root.join(rel), content).unwrap();
    }

    fn commit_all(&self, message: &str) -> String {
        git(&self.root, &["add", "-A"]);
        git(&self.root, &["commit", "-qm", message]);
        git(&self.root, &["rev-parse", "HEAD"])
    }

    fn landed(&self, write: &WriteRecord) -> Landed {
        landed(Resolver::new(&self.root).as_mut(), write, &|_| true).unwrap()
    }
}

/// The content hash `content`'s symbol `name` has, as a write records it.
fn hash_of(content: &str, name: &str) -> String {
    ambits::writes::FileContents::read("src/a.rs", content.as_bytes(), &ParserRegistry::new()).hashes(name)[0].clone()
}

/// A write made a minute ago.
fn write(op: &str, level: Level, syms: Vec<(&str, String)>, fh: Option<String>) -> WriteRecord {
    WriteRecord {
        op: op.into(),
        av: ambits::writes::ATTRIBUTION_VERSION,
        a: "agent-1".into(),
        t: ambits::time::rfc3339(ambits::time::now_secs() - 60),
        tool: "Edit".into(),
        file: "src/a.rs".into(),
        level,
        syms: syms.into_iter().map(|(s, h)| (s.to_string(), h)).collect(),
        fh,
        ..Default::default()
    }
}

const EDITED: &str = "fn alpha() { 1; }\nfn beta() {}\n";

#[test]
fn a_symbol_write_lands_in_the_commit_that_contains_it() {
    let repo = Repo::new();
    repo.write_file("src/a.rs", EDITED);
    let w = write("toolu_1", Level::Symbol, vec![("src/a.rs::alpha", hash_of(EDITED, "alpha"))], None);
    assert_eq!(repo.landed(&w), Landed::Uncommitted);

    let commit = repo.commit_all("edit alpha");
    assert_eq!(repo.landed(&w), Landed::Verified { commits: vec![commit] });
}

/// `git add -p`: one write's symbols land in two commits.
#[test]
fn one_writes_symbols_can_land_in_different_commits() {
    let repo = Repo::new();
    let both = "fn alpha() { 1; }\nfn beta() { 2; }\n";
    let w = write(
        "toolu_1",
        Level::Symbol,
        vec![("src/a.rs::alpha", hash_of(both, "alpha")), ("src/a.rs::beta", hash_of(both, "beta"))],
        None,
    );
    repo.write_file("src/a.rs", "fn alpha() { 1; }\nfn beta() {}\n");
    let first = repo.commit_all("alpha only");
    assert!(matches!(repo.landed(&w), Landed::Partial { commits, unverified: false } if commits == vec![first.clone()]));

    repo.write_file("src/a.rs", both);
    let second = repo.commit_all("then beta");
    assert_eq!(repo.landed(&w), Landed::Verified { commits: vec![first, second] });
}

/// A file-level `Write` is matched by the hash of its bytes.
#[test]
fn a_file_level_write_lands_by_its_content_hash() {
    let repo = Repo::new();
    repo.write_file("src/a.rs", EDITED);
    let fh = ambits::journal::encode_hash(blake3::hash(EDITED.as_bytes()).as_bytes());
    let w = write("toolu_1", Level::File, vec![], Some(fh));
    let commit = repo.commit_all("write");
    assert_eq!(repo.landed(&w), Landed::Verified { commits: vec![commit] });
}

/// With nothing to compare, the first commit touching the file after the
/// write is reported, and labelled unverified.
#[test]
fn a_file_level_edit_lands_unverified() {
    let repo = Repo::new();
    repo.write_file("src/a.rs", EDITED);
    let w = write("toolu_1", Level::File, vec![], None);
    let commit = repo.commit_all("edit");
    assert_eq!(repo.landed(&w), Landed::Unverified { commits: vec![commit] });
}

/// A cached link to a commit amended away is re-resolved to its successor.
#[test]
fn an_amended_commit_is_re_resolved() {
    let repo = Repo::new();
    repo.write_file("src/a.rs", EDITED);
    let w = write("toolu_1", Level::Symbol, vec![("src/a.rs::alpha", hash_of(EDITED, "alpha"))], None);
    let original = repo.commit_all("edit alpha");
    assert_eq!(repo.landed(&w), Landed::Verified { commits: vec![original.clone()] });

    git(&repo.root, &["commit", "-q", "--amend", "-m", "edit alpha, reworded"]);
    git(&repo.root, &["reflog", "expire", "--expire=now", "--all"]);
    let amended = git(&repo.root, &["rev-parse", "HEAD"]);
    assert_ne!(amended, original);
    assert_eq!(repo.landed(&w), Landed::Verified { commits: vec![amended] });
}

/// A rebase onto a moved base re-resolves too.
#[test]
fn a_rebased_commit_is_re_resolved() {
    let repo = Repo::new();
    git(&repo.root, &["checkout", "-q", "-b", "feature"]);
    repo.write_file("src/a.rs", EDITED);
    let w = write("toolu_1", Level::Symbol, vec![("src/a.rs::alpha", hash_of(EDITED, "alpha"))], None);
    let before = repo.commit_all("edit alpha");
    assert_eq!(repo.landed(&w), Landed::Verified { commits: vec![before.clone()] });

    git(&repo.root, &["checkout", "-q", "main"]);
    repo.write_file("README.md", "readme\n");
    repo.commit_all("unrelated");
    git(&repo.root, &["checkout", "-q", "feature"]);
    git(&repo.root, &["rebase", "-q", "main"]);
    let after = git(&repo.root, &["rev-parse", "HEAD"]);
    assert_ne!(after, before);
    assert_eq!(repo.landed(&w), Landed::Verified { commits: vec![after] });
}

/// A commit on another branch counts, whatever is checked out.
#[test]
fn a_write_on_another_branch_is_found() {
    let repo = Repo::new();
    git(&repo.root, &["checkout", "-q", "-b", "feature"]);
    repo.write_file("src/a.rs", EDITED);
    let w = write("toolu_1", Level::Symbol, vec![("src/a.rs::alpha", hash_of(EDITED, "alpha"))], None);
    let commit = repo.commit_all("on feature");
    git(&repo.root, &["checkout", "-q", "main"]);
    assert_eq!(repo.landed(&w), Landed::Verified { commits: vec![commit] });
}

/// The file was renamed before the write was committed: it is found under
/// its new name — here with so much of the small file changed that git
/// sees a deletion and an addition, not a rename.
#[test]
fn a_write_committed_under_a_new_name_is_found() {
    let repo = Repo::new();
    repo.write_file("src/a.rs", EDITED);
    let w = write("toolu_1", Level::Symbol, vec![("src/a.rs::alpha", hash_of(EDITED, "alpha"))], None);
    git(&repo.root, &["mv", "src/a.rs", "src/b.rs"]);
    let commit = repo.commit_all("rename with the edit");
    assert_eq!(repo.landed(&w), Landed::Verified { commits: vec![commit] });
}

/// Live entries of a flat cache: the last line per key, unless it is a
/// tombstone (`link: null`, or a never-landed key without tips).
fn live(root: &Path, file: &str) -> usize {
    let text = std::fs::read_to_string(root.join(".ambits").join(file)).unwrap_or_default();
    let mut last: std::collections::HashMap<String, bool> = std::collections::HashMap::new();
    for line in text.lines().filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok()) {
        let Some(k) = line.get("k").and_then(|k| k.as_str()) else { continue };
        let alive = line.get("link").map_or_else(|| line.get("tips").is_some(), |l| !l.is_null());
        last.insert(k.to_string(), alive);
    }
    last.values().filter(|a| **a).count()
}

/// "Not landed" is cached until a branch moves; a commit then finds it.
#[test]
fn never_landed_is_cached_until_a_branch_moves() {
    let repo = Repo::new();
    repo.write_file("src/a.rs", EDITED);
    let w = write("toolu_1", Level::Symbol, vec![("src/a.rs::alpha", hash_of(EDITED, "alpha"))], None);
    assert_eq!(repo.landed(&w), Landed::Uncommitted);
    assert_eq!(live(&repo.root, "cache/never-landed.ndjson"), 1, "cached");

    let commit = repo.commit_all("edit alpha");
    assert_eq!(repo.landed(&w), Landed::Verified { commits: vec![commit] });
    assert_eq!(live(&repo.root, "cache/never-landed.ndjson"), 0, "cleared once it landed");
    assert_eq!(live(&repo.root, "links.ndjson"), 1);
}

/// `touched` reports the landing commit, as text and JSON.
#[test]
fn touched_shows_the_landing_commit() {
    let repo = Repo::new();
    repo.write_file("src/a.rs", EDITED);
    let w = write("toolu_1", Level::Symbol, vec![("src/a.rs::alpha", hash_of(EDITED, "alpha"))], None);
    common::journal_write(&repo.root, SESSION, w);
    let commit = repo.commit_all("edit alpha");

    let run = |args: &[&str]| run_ambits(&repo.root, args);
    let text = run(&["touched", "src/a.rs::alpha"]);
    assert!(text.contains(&format!("landed in {} (verified)", &commit[..7])), "{text}");
    let json: serde_json::Value = serde_json::from_str(&run(&["touched", "--format", "json", "src/a.rs::alpha"])).unwrap();
    assert_eq!(json["last_write"]["landed"]["state"], "verified");
    assert_eq!(json["last_write"]["landed"]["commits"][0], commit);
}

/// The hook chains to an existing one, and never fails a commit — even
/// when ambits itself cannot run.
#[test]
fn the_git_hook_chains_and_never_fails_a_commit() {
    let repo = Repo::new();
    std::fs::create_dir_all(repo.root.join(".ambits")).unwrap();
    let hooks = PathBuf::from(git(&repo.root, &["rev-parse", "--git-path", "hooks"]));
    let hooks = if hooks.is_absolute() { hooks } else { repo.root.join(hooks) };
    std::fs::create_dir_all(&hooks).unwrap();
    let marker = repo.root.join("chained-ran");
    let existing = hooks.join("post-commit");
    std::fs::write(&existing, format!("#!/bin/sh\ntouch '{}'\n", marker.display())).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&existing, std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    // An ambits that cannot run.
    ambits::git_hook::install(&repo.root, Path::new("/nonexistent/ambits")).unwrap();
    repo.write_file("src/a.rs", EDITED);
    repo.commit_all("with the hook");
    assert!(marker.exists(), "the existing hook still runs");

    assert!(ambits::git_hook::uninstall(&repo.root).unwrap());
    let restored = std::fs::read_to_string(&existing).unwrap();
    assert!(restored.contains("chained-ran"), "the original hook is back");
}

/// A file long enough that git detects the rename, carrying the edit with it.
#[test]
fn a_write_committed_through_a_detected_rename_is_found() {
    let repo = Repo::new();
    let long: String = (0..40).map(|i| format!("fn f{i}() {{}}\n")).collect();
    repo.write_file("src/a.rs", &long);
    repo.commit_all("long file");
    let edited = long.replace("fn f0() {}", "fn f0() { 1; }");
    repo.write_file("src/a.rs", &edited);
    let w = write("toolu_1", Level::Symbol, vec![("src/a.rs::f0", hash_of(&edited, "f0"))], None);
    git(&repo.root, &["mv", "src/a.rs", "src/b.rs"]);
    let commit = repo.commit_all("rename with the edit");
    let log = git(&repo.root, &["show", "--name-status", "--format=", "HEAD"]);
    assert!(log.starts_with('R'), "git saw a rename: {log}");
    assert_eq!(repo.landed(&w), Landed::Verified { commits: vec![commit] });
}

/// A rename on one branch must not stop the search following the file on
/// another: the log mixes branches.
#[test]
fn a_rename_on_another_branch_does_not_hide_this_one() {
    let repo = Repo::new();
    repo.write_file("src/a.rs", EDITED);
    let w = write("toolu_1", Level::Symbol, vec![("src/a.rs::alpha", hash_of(EDITED, "alpha"))], None);
    git(&repo.root, &["stash", "-q"]);
    git(&repo.root, &["checkout", "-q", "-b", "feature"]);
    git(&repo.root, &["mv", "src/a.rs", "src/b.rs"]);
    repo.commit_all("rename on feature");
    git(&repo.root, &["checkout", "-q", "main"]);
    git(&repo.root, &["stash", "pop", "-q"]);
    let commit = repo.commit_all("the write, on main");
    assert_eq!(repo.landed(&w), Landed::Verified { commits: vec![commit] });
}

/// A project in a subdirectory of its repository, with `diff.relative` set:
/// git would print project-relative paths unless told not to.
#[test]
fn a_subdirectory_project_resolves_despite_diff_relative() {
    let dir = tempfile::tempdir().unwrap();
    let top = dir.path().canonicalize().unwrap();
    let root = top.join("proj");
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(root.join("src/a.rs"), "fn alpha() {}\nfn beta() {}\n").unwrap();
    git(&top, &["init", "-q"]);
    git(&top, &["config", "diff.relative", "true"]);
    git(&top, &["add", "."]);
    git(&top, &["commit", "-qm", "init"]);
    std::fs::write(root.join("src/a.rs"), EDITED).unwrap();
    let w = write("toolu_1", Level::Symbol, vec![("src/a.rs::alpha", hash_of(EDITED, "alpha"))], None);
    git(&top, &["commit", "-qam", "edit"]);
    let commit = git(&top, &["rev-parse", "HEAD"]);
    let result = landed(Resolver::new(&root).as_mut(), &w, &|_| true).unwrap();
    assert_eq!(result, Landed::Verified { commits: vec![commit] });
}

fn hooks_dir(root: &Path) -> PathBuf {
    let hooks = PathBuf::from(git(root, &["rev-parse", "--git-path", "hooks"]));
    if hooks.is_absolute() { hooks } else { root.join(hooks) }
}

/// A commit whose output is captured (`$(git commit)`, an IDE, an agent's
/// shell tool) must not wait for the hook's background work.
#[test]
fn the_git_hook_never_delays_a_captured_commit() {
    let repo = Repo::new();
    std::fs::create_dir_all(repo.root.join(".ambits")).unwrap();
    let slow = repo.root.join("slow-ambits");
    std::fs::write(&slow, "#!/bin/sh\nsleep 5\n").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&slow, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    ambits::git_hook::install(&repo.root, &slow).unwrap();
    repo.write_file("src/a.rs", EDITED);
    git(&repo.root, &["add", "-A"]);
    let started = std::time::Instant::now();
    git(&repo.root, &["commit", "-qm", "captured"]);
    assert!(started.elapsed() < std::time::Duration::from_secs(3), "commit waited {:?}", started.elapsed());
}

/// Installed for real, the hook records the landing commit in the links
/// index without being asked.
#[test]
fn the_git_hook_records_links() {
    let repo = Repo::new();
    repo.write_file("src/a.rs", EDITED);
    let w = write("toolu_1", Level::Symbol, vec![("src/a.rs::alpha", hash_of(EDITED, "alpha"))], None);
    common::journal_write(&repo.root, SESSION, w);
    ambits::git_hook::install(&repo.root, Path::new(env!("CARGO_BIN_EXE_ambits"))).unwrap();
    assert!(hooks_dir(&repo.root).join("post-commit").exists());

    repo.commit_all("edit alpha");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    while live(&repo.root, "links.ndjson") == 0 {
        assert!(std::time::Instant::now() < deadline, "the hook wrote no link");
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

/// The links index is part of the private store: `0600`, and gc sweeps
/// the temp files an interrupted rewrite leaves beside the flat files.
#[cfg(unix)]
#[test]
fn links_are_private_and_their_temp_files_are_swept() {
    use std::os::unix::fs::PermissionsExt;
    let repo = Repo::new();
    repo.write_file("src/a.rs", EDITED);
    let w = write("toolu_1", Level::Symbol, vec![("src/a.rs::alpha", hash_of(EDITED, "alpha"))], None);
    repo.commit_all("edit alpha");
    repo.landed(&w);
    let links = repo.root.join(".ambits/links.ndjson");
    assert_eq!(std::fs::metadata(&links).unwrap().permissions().mode() & 0o777, 0o600);

    let stray = [repo.root.join(".ambits/.tmp-1"), repo.root.join(".ambits/cache/.tmp-2")];
    for s in &stray {
        std::fs::create_dir_all(s.parent().unwrap()).unwrap();
        std::fs::write(s, "x").unwrap();
    }
    let store = ambits::objects::store::Store::at(&repo.root);
    let stats = ambits::objects::gc::gc(&store, std::time::Duration::ZERO, ambits::objects::refs::REFLOG_EXPIRY).unwrap();
    assert_eq!(stats.temp_files_removed, 2);
    assert!(stray.iter().all(|s| !s.exists()));
}
