//! Process-level tests for `ambits grep` — the GNU grep dialect.
//!
//! Everything here is about the *dialect*, not the search: the engine is shared
//! with `ambits rg` and covered by `tests/rg_cli.rs` and `search.rs`'s own unit
//! tests. What needs pinning is the handful of letters that mean one thing in
//! grep and something else in ripgrep, plus the flags that cannot be honoured.
//!
//! **Do not compare against this machine's `grep`.** It resolves to ugrep here,
//! not GNU grep, so using it as an oracle would quietly validate against the
//! wrong specification. Expectations below are written from documented GNU grep
//! behaviour instead.

use std::path::Path;
use std::process::Command;

/// A project with one matching file, one not, and one under an excluded
/// directory.
fn fixture() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();

    std::fs::create_dir_all(root.join(".git")).unwrap();
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::create_dir_all(root.join("docs")).unwrap();
    std::fs::create_dir_all(root.join("vendor")).unwrap();

    std::fs::write(
        root.join("src/lib.rs"),
        "pub fn needle() -> u32 {\n    42\n}\n",
    )
    .unwrap();
    std::fs::write(root.join("src/util.rs"), "pub fn unrelated() {}\n").unwrap();
    std::fs::write(root.join("docs/notes.md"), "the needle is documented\n").unwrap();
    std::fs::write(root.join("vendor/copy.rs"), "fn needle() {}\n").unwrap();

    dir
}

struct Output {
    code: i32,
    stdout: String,
    stderr: String,
}

fn run(root: &Path, args: &[&str]) -> Output {
    let out = Command::new(env!("CARGO_BIN_EXE_ambits"))
        .current_dir(root)
        .arg("-p")
        .arg(root)
        .args(args)
        .output()
        .expect("the binary must run");
    Output {
        code: out.status.code().expect("no signal"),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

fn grep(root: &Path, args: &[&str]) -> Output {
    let mut all = vec!["-s", "sess", "--no-journal", "grep"];
    all.extend_from_slice(args);
    run(root, &all)
}

// ---------------------------------------------------------------------------
// The letters that collide with ripgrep
// ---------------------------------------------------------------------------

/// `-L` is the whole reason two dialects exist: files-*without*-match in grep,
/// follow-symlinks in ripgrep. Getting this wrong silently inverts an answer.
#[test]
fn dash_l_means_files_without_match() {
    let dir = fixture();
    let out = grep(dir.path(), &["-L", "needle"]);

    assert_eq!(out.code, 0);
    assert!(out.stdout.contains("src/util.rs"), "got {:?}", out.stdout);
    assert!(
        !out.stdout.contains("src/lib.rs"),
        "a file that matched must not be listed: {:?}",
        out.stdout
    );
}

/// `-Z` NUL-terminates each path so the listing survives `xargs -0`. In
/// ripgrep the same letter means "search compressed files".
#[test]
fn dash_z_nul_terminates_paths() {
    let dir = fixture();
    let out = grep(dir.path(), &["-lZ", "needle"]);

    assert_eq!(out.code, 0);
    assert!(out.stdout.contains('\0'), "expected NUL separators");
    assert!(
        !out.stdout.contains('\n'),
        "NUL replaces the newline, it does not join it: {:?}",
        out.stdout
    );
}

/// `-r`/`-R` mean "recurse" in grep and "replace" in ripgrep. The search is
/// already recursive, so they are accepted and change nothing — a habitual
/// `grep -r` must not become an unknown-flag error.
#[test]
fn recursive_flags_are_accepted_no_ops() {
    let dir = fixture();
    let plain = grep(dir.path(), &["needle"]);
    for flag in ["-r", "-R"] {
        let with = grep(dir.path(), &[flag, "needle"]);
        assert_eq!(with.code, 0, "{flag} should be accepted");
        assert_eq!(
            with.stdout, plain.stdout,
            "{flag} must not change the result"
        );
    }
}

/// `-h` is `--no-filename` in grep, where clap would otherwise claim it for
/// help. Help stays reachable through `--help`.
#[test]
fn dash_h_is_no_filename_not_help() {
    let dir = fixture();
    let out = grep(dir.path(), &["-h", "needle"]);

    assert_eq!(out.code, 0);
    assert!(
        !out.stdout.contains("src/lib.rs"),
        "-h suppresses the path: {:?}",
        out.stdout
    );
    assert!(
        out.stdout.contains("needle"),
        "but still prints the match: {:?}",
        out.stdout
    );

    let help = grep(dir.path(), &["--help"]);
    assert_eq!(help.code, 0);
    assert!(help.stdout.contains("--no-filename"));
}

// ---------------------------------------------------------------------------
// grep's own defaults
// ---------------------------------------------------------------------------

/// `grep -n` is opt-in and grep has no column at all, where the rg dialect
/// prints both by default. Same engine, different defaults.
#[test]
fn line_numbers_are_opt_in_unlike_the_rg_dialect() {
    let dir = fixture();

    let bare = grep(dir.path(), &["needle", "src"]);
    let line = bare.stdout.lines().next().unwrap();
    let after_path = line.split_once(':').unwrap().1;
    assert!(
        !after_path.starts_with(|c: char| c.is_ascii_digit()),
        "no line number by default: {line:?}"
    );

    let numbered = grep(dir.path(), &["-n", "needle", "src"]);
    let line = numbered.stdout.lines().next().unwrap();
    let after_path = line.split_once(':').unwrap().1;
    assert!(
        after_path.starts_with(|c: char| c.is_ascii_digit()),
        "-n turns them on: {line:?}"
    );
}

// ---------------------------------------------------------------------------
// Flags that cannot be honoured
// ---------------------------------------------------------------------------

/// Accepting `-P` would be the anchor bug again: a lookaround pattern that
/// works under real grep would quietly match something else here, because the
/// regex crate cannot express it. Refusing is the honest answer.
#[test]
fn perl_regexp_is_rejected_rather_than_silently_wrong() {
    let dir = fixture();
    let out = grep(dir.path(), &["-P", "needle"]);

    assert_eq!(out.code, 2);
    assert!(
        out.stderr.contains("not supported") && out.stderr.contains("lookaround"),
        "the message must say why: {:?}",
        out.stderr
    );
    assert!(out.stdout.is_empty());
}

#[test]
fn null_data_is_rejected() {
    let dir = fixture();
    let out = grep(dir.path(), &["-z", "needle"]);

    assert_eq!(out.code, 2);
    assert!(out.stderr.contains("not supported"), "{:?}", out.stderr);
}

/// `-E` and `-G` select a regex flavour grep has and ambits does not. One
/// engine handles both portably, so they are accepted and inert rather than
/// rejected — unlike `-P`, neither promises syntax we cannot deliver.
#[test]
fn regex_flavour_flags_are_accepted_inert() {
    let dir = fixture();
    let plain = grep(dir.path(), &["needle"]);
    for flag in ["-E", "-G"] {
        let with = grep(dir.path(), &[flag, "needle"]);
        assert_eq!(with.code, 0, "{flag} should be accepted");
        assert_eq!(with.stdout, plain.stdout);
    }
}

// ---------------------------------------------------------------------------
// grep's own file selection
// ---------------------------------------------------------------------------

#[test]
fn include_and_exclude_select_files() {
    let dir = fixture();

    let included = grep(dir.path(), &["-l", "--include", "*.rs", "needle"]);
    assert!(included.stdout.contains("src/lib.rs"));
    assert!(
        !included.stdout.contains("notes.md"),
        "--include is exclusive: {:?}",
        included.stdout
    );

    let excluded = grep(dir.path(), &["-l", "--exclude", "*.md", "needle"]);
    assert!(excluded.stdout.contains("src/lib.rs"));
    assert!(!excluded.stdout.contains("notes.md"));
}

/// A directory exclusion has to cover everything beneath it, not just an entry
/// named for the directory itself.
#[test]
fn exclude_dir_skips_everything_beneath_it() {
    let dir = fixture();

    let all = grep(dir.path(), &["-l", "needle"]);
    assert!(all.stdout.contains("vendor/copy.rs"), "baseline");

    let out = grep(dir.path(), &["-l", "--exclude-dir", "vendor", "needle"]);
    assert!(
        !out.stdout.contains("vendor/"),
        "nothing under vendor/ survives: {:?}",
        out.stdout
    );
    assert!(out.stdout.contains("src/lib.rs"), "the rest is untouched");
}

// ---------------------------------------------------------------------------
// Shared contracts, confirmed for this dialect too
// ---------------------------------------------------------------------------

#[test]
fn exit_codes_follow_grep() {
    let dir = fixture();
    assert_eq!(grep(dir.path(), &["needle"]).code, 0, "matched");
    assert_eq!(grep(dir.path(), &["nosuchtext"]).code, 1, "no match");
    assert_eq!(grep(dir.path(), &["fn ("]).code, 2, "bad pattern");
}

/// The symbol column is ambit's addition to both dialects, not an rg-only
/// feature.
#[test]
fn the_symbol_column_is_present_here_too() {
    let dir = fixture();
    let out = grep(dir.path(), &["needle", "src"]);

    assert!(
        out.stdout.contains("[needle]") || out.stdout.contains("needle]"),
        "expected a symbol annotation: {:?}",
        out.stdout
    );

    let bare = grep(dir.path(), &["--no-symbol", "needle", "src"]);
    assert!(
        !bare.stdout.contains("[needle]"),
        "--no-symbol drops it: {:?}",
        bare.stdout
    );
}
