//! Git linkage end to end (spec §3.2, §9.3; the phase-4 key tests of §14):
//! which commit an agent's write landed in.

use std::path::{Path, PathBuf};
use std::process::Command;

use ambits::linkage::{landed, Landed, Resolver};
use ambits::parser::ParserRegistry;
use ambits::writes::{Level, WriteRecord};

const SESSION: &str = "0b7e9d3a-1c2f-4e5a-9b8c-7d6e5f4a3b2c";

fn git(root: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_AUTHOR_NAME", "t")
        .env("GIT_AUTHOR_EMAIL", "t@t")
        .env("GIT_COMMITTER_NAME", "t")
        .env("GIT_COMMITTER_EMAIL", "t@t")
        .output()
        .unwrap();
    assert!(out.status.success(), "git {args:?}: {}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8(out.stdout).unwrap().trim().to_string()
}

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
        let past = format!("@{} +0000", ambits::objects::refs::now_secs() - 7200);
        let ok = Command::new("git")
            .arg("-C")
            .arg(&root)
            .args(["commit", "-qm", "init"])
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("GIT_AUTHOR_NAME", "t")
            .env("GIT_AUTHOR_EMAIL", "t@t")
            .env("GIT_COMMITTER_NAME", "t")
            .env("GIT_COMMITTER_EMAIL", "t@t")
            .env("GIT_AUTHOR_DATE", &past)
            .env("GIT_COMMITTER_DATE", &past)
            .status()
            .unwrap()
            .success();
        assert!(ok);
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
    let parsed = ParserRegistry::new()
        .parser_for(Path::new("src/a.rs"))
        .unwrap()
        .parse_file(Path::new("src/a.rs"), content)
        .unwrap();
    let sym = parsed.symbols.iter().find(|s| &*s.name == name).unwrap();
    ambits::journal::encode_hash(&sym.content_hash)
}

/// A write made a minute ago.
fn write(op: &str, level: Level, syms: Vec<(&str, String)>, fh: Option<String>) -> WriteRecord {
    WriteRecord {
        op: op.into(),
        av: ambits::writes::ATTRIBUTION_VERSION,
        a: "agent-1".into(),
        t: ambits::objects::refs::rfc3339(ambits::objects::refs::now_secs() - 60),
        tool: "Edit".into(),
        file: "src/a.rs".into(),
        level,
        outside_symbols: false,
        syms: syms.into_iter().map(|(s, h)| (s.to_string(), h)).collect(),
        removed: vec![],
        fh,
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
    assert!(matches!(repo.landed(&w), Landed::Partial { commits } if commits == vec![first.clone()]));

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

/// "Not landed" is cached until a branch moves; a commit then finds it.
#[test]
fn never_landed_is_cached_until_a_branch_moves() {
    let repo = Repo::new();
    repo.write_file("src/a.rs", EDITED);
    let w = write("toolu_1", Level::Symbol, vec![("src/a.rs::alpha", hash_of(EDITED, "alpha"))], None);
    assert_eq!(repo.landed(&w), Landed::Uncommitted);
    let cache = repo.root.join(".ambits/cache/never-landed");
    assert_eq!(std::fs::read_dir(&cache).unwrap().count(), 1, "cached");

    let commit = repo.commit_all("edit alpha");
    assert_eq!(repo.landed(&w), Landed::Verified { commits: vec![commit] });
    assert_eq!(std::fs::read_dir(&cache).unwrap().count(), 0, "cleared once it landed");
    assert_eq!(std::fs::read_dir(repo.root.join(".ambits/links")).unwrap().count(), 1);
}

/// `touched` reports the landing commit, as text and JSON.
#[test]
fn touched_shows_the_landing_commit() {
    let repo = Repo::new();
    repo.write_file("src/a.rs", EDITED);
    let w = write("toolu_1", Level::Symbol, vec![("src/a.rs::alpha", hash_of(EDITED, "alpha"))], None);
    let dir = repo.root.join(".ambits/coverage");
    std::fs::create_dir_all(&dir).unwrap();
    let line = serde_json::to_string(&ambits::journal::Record::Write(Box::new(w))).unwrap();
    std::fs::write(dir.join(format!("{SESSION}.ndjson")), format!("{line}\n")).unwrap();
    let commit = repo.commit_all("edit alpha");

    let run = |args: &[&str]| {
        let out = Command::new(env!("CARGO_BIN_EXE_ambits")).current_dir(&repo.root).env("HOME", &repo.root).args(args).output().unwrap();
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        String::from_utf8(out.stdout).unwrap()
    };
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
