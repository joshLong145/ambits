//! Process-level tests for `ambits find`.
//!
//! These run the built binary rather than the library, because the things worth
//! pinning here only exist at that boundary: exit codes, which stream output
//! lands on, how positional arguments are split, and whether a journal file
//! appears on disk. `tests/e2e.rs` covers the library-level pipeline.

use std::path::Path;
use std::process::Command;

/// A small project: two Rust files, a Markdown file, and a gitignored one.
fn fixture() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();

    // `ignore` honours `.gitignore` only inside a git repository, so the
    // fixture needs one for the walk to behave the way it does in the field.
    std::fs::create_dir_all(root.join(".git")).unwrap();
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::create_dir_all(root.join("docs")).unwrap();
    std::fs::create_dir_all(root.join("target")).unwrap();

    std::fs::write(root.join(".gitignore"), "/target\n").unwrap();
    std::fs::write(
        root.join("src/lib.rs"),
        "pub fn needle() -> u32 {\n    42\n}\n\npub fn haystack() -> u32 {\n    needle()\n}\n",
    )
    .unwrap();
    std::fs::write(
        root.join("src/util.rs"),
        "pub fn unrelated() -> u32 {\n    7\n}\n",
    )
    .unwrap();
    std::fs::write(root.join("docs/notes.md"), "the needle is documented here\n").unwrap();
    std::fs::write(root.join("target/generated.rs"), "fn needle() {}\n").unwrap();

    dir
}

struct Output {
    code: i32,
    stdout: String,
    stderr: String,
}

fn run(root: &Path, args: &[&str]) -> Output {
    let out = Command::new(env!("CARGO_BIN_EXE_ambits"))
        // PATH arguments resolve against the working directory, as grep's do,
        // so the tests run from inside the project the way a caller would.
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

/// `find` with a session, so journal writes have somewhere to go.
fn find(root: &Path, args: &[&str]) -> Output {
    let mut all = vec!["-s", "sess", "find"];
    all.extend_from_slice(args);
    run(root, &all)
}

fn journal(root: &Path) -> String {
    // `find` writes its own shard, distinct from the primary file a running
    // TUI would write — see `journal::Journal::open_shard`.
    std::fs::read_to_string(root.join(".ambit/coverage/sess.find.ndjson")).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Exit codes
// ---------------------------------------------------------------------------

/// grep's convention, and the reason it matters: agents chain with `&&`, so
/// "no match" and "the command failed" must not be the same answer.
#[test]
fn exit_codes_follow_grep() {
    let dir = fixture();

    assert_eq!(find(dir.path(), &["needle"]).code, 0, "matched");
    assert_eq!(find(dir.path(), &["nosuchtext"]).code, 1, "no match");

    let broken = find(dir.path(), &["fn ("]);
    assert_eq!(broken.code, 2, "a pattern that cannot compile is an error");
    assert!(
        broken.stderr.contains("invalid pattern"),
        "the error names the pattern: {:?}",
        broken.stderr
    );
    assert!(broken.stdout.is_empty(), "errors never go to stdout");
}

#[test]
fn a_path_outside_the_project_root_is_an_error() {
    let dir = fixture();
    let outside = find(dir.path(), &["needle", "/etc"]);

    assert_eq!(outside.code, 2);
    assert!(
        outside.stderr.contains("outside the project root"),
        "got {:?}",
        outside.stderr
    );
}

#[test]
fn a_path_that_does_not_exist_is_an_error() {
    let dir = fixture();
    let missing = find(dir.path(), &["needle", "src/nope"]);
    assert_eq!(missing.code, 2);
    assert!(missing.stderr.contains("src/nope"), "{:?}", missing.stderr);
}

// ---------------------------------------------------------------------------
// Narrowing the walk
// ---------------------------------------------------------------------------

#[test]
fn path_arguments_narrow_the_search() {
    let dir = fixture();

    let everywhere = find(dir.path(), &["needle", "-l"]);
    assert!(everywhere.stdout.contains("src/lib.rs"));
    assert!(
        everywhere.stdout.contains("docs/notes.md"),
        "every text file is searched, not only parseable ones"
    );

    let scoped = find(dir.path(), &["needle", "-l", "src"]);
    assert!(scoped.stdout.contains("src/lib.rs"));
    assert!(!scoped.stdout.contains("docs/notes.md"), "PATH narrows it");
}

#[test]
fn glob_and_type_filters_narrow_the_walk() {
    let dir = fixture();

    let typed = find(dir.path(), &["needle", "-l", "-t", "rust"]);
    assert!(typed.stdout.contains("src/lib.rs"));
    assert!(!typed.stdout.contains("notes.md"), "-t rust excludes md");

    let globbed = find(dir.path(), &["needle", "-l", "-g", "*.md"]);
    assert!(globbed.stdout.contains("notes.md"));
    assert!(!globbed.stdout.contains("lib.rs"), "-g selects");

    let negated = find(dir.path(), &["needle", "-l", "-g", "!*.md"]);
    assert!(!negated.stdout.contains("notes.md"), "!glob excludes");
}

/// The walk is the scanner's: gitignored files stay out unless asked for.
#[test]
fn gitignored_files_are_skipped_unless_no_ignore() {
    let dir = fixture();

    let default = find(dir.path(), &["needle", "-l"]);
    assert!(!default.stdout.contains("target/"), "/target is ignored");

    let forced = find(dir.path(), &["needle", "-l", "--no-ignore"]);
    assert!(forced.stdout.contains("target/generated.rs"));
}

// ---------------------------------------------------------------------------
// Output contract
// ---------------------------------------------------------------------------

/// Piped output is what a grep consumer parses, so nothing else may appear on
/// stdout — the withheld notice included.
#[test]
fn the_withheld_trailer_goes_to_stderr_not_stdout() {
    let dir = fixture();
    let capped = find(dir.path(), &["needle", "--head-limit", "1"]);

    assert_eq!(capped.stdout.lines().count(), 1, "one match printed");
    assert!(!capped.stdout.contains("withheld"));
    assert!(
        capped.stderr.contains("withheld"),
        "the notice belongs on stderr: {:?}",
        capped.stderr
    );
}

#[test]
fn piped_output_is_one_flat_line_per_match() {
    let dir = fixture();
    let out = find(dir.path(), &["needle", "-t", "rust"]);

    for line in out.stdout.lines() {
        let mut fields = line.splitn(4, ':');
        let path = fields.next().unwrap();
        assert!(path.ends_with(".rs"), "field 1 is the path: {line:?}");
        assert!(
            fields.next().unwrap().parse::<u32>().is_ok(),
            "field 2 is the line number: {line:?}"
        );
        assert!(
            fields.next().unwrap().parse::<u32>().is_ok(),
            "field 3 is the column: {line:?}"
        );
    }
}

#[test]
fn json_is_one_event_per_line() {
    let dir = fixture();
    let out = find(dir.path(), &["needle", "-t", "rust", "--json"]);

    let events: Vec<serde_json::Value> = out
        .stdout
        .lines()
        .map(|l| serde_json::from_str(l).expect("every line parses as one event"))
        .collect();

    assert_eq!(events.first().unwrap()["type"], "begin");
    assert_eq!(events.last().unwrap()["type"], "summary");
    let matched = events.iter().find(|e| e["type"] == "match").unwrap();
    assert_eq!(
        matched["data"]["symbol"]["id"], "src/lib.rs::needle",
        "the symbol id rides along with the match"
    );
}

/// The composition the whole pair exists for: search, then fetch.
#[test]
fn find_then_show_composes_on_the_emitted_id() {
    let dir = fixture();
    let out = find(dir.path(), &["fn needle", "-t", "rust", "--json"]);
    let id = out
        .stdout
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .find(|e| e["type"] == "match")
        .and_then(|e| e["data"]["symbol"]["id"].as_str().map(str::to_owned))
        .expect("a match carries an id");

    let shown = run(dir.path(), &["show", &id]);
    assert_eq!(shown.code, 0);
    let parsed: serde_json::Value = serde_json::from_str(shown.stdout.trim()).unwrap();
    assert_eq!(parsed["results"][0]["matches"][0]["id"], id.as_str());
    assert!(
        parsed["results"][0]["matches"][0]["definition"]
            .as_str()
            .unwrap()
            .contains("fn needle"),
        "the id round-trips to the definition"
    );
}

// ---------------------------------------------------------------------------
// Journaling
// ---------------------------------------------------------------------------

/// `find` no longer writes its own journal entries — the TUI is the sole
/// writer now (see `journal.rs`'s module doc: relying on it exclusively means
/// a session with no TUI attached earns no coverage credit from searches, an
/// accepted trade for never having two processes able to write the same
/// session's journal). A search that matched and printed source still leaves
/// the journal untouched, in every mode, and regardless of `--no-journal` or
/// the `[cache]` stanza — there is nothing left for either to suppress.
#[test]
fn a_search_never_journals_its_own_reads() {
    let dir = fixture();
    assert!(journal(dir.path()).is_empty(), "no journal to start with");

    find(dir.path(), &["fn needle", "-t", "rust"]);
    assert!(
        journal(dir.path()).is_empty(),
        "a search that matched and printed source still writes nothing"
    );

    for silent in [&["needle", "-l"][..], &["needle", "-c"], &["needle", "-q"]] {
        find(dir.path(), silent);
        assert!(journal(dir.path()).is_empty());
    }
}

/// Without its own journal write, a second search has nothing recorded to
/// read back — depth stays unknown, not "read" and not "unread", across
/// repeated identical searches.
#[test]
fn a_second_identical_search_still_reports_unknown_depth() {
    let dir = fixture();

    let first = find(dir.path(), &["fn needle", "-t", "rust"]);
    assert!(first.stdout.contains("[needle]"), "{:?}", first.stdout);

    let second = find(dir.path(), &["fn needle", "-t", "rust"]);
    assert!(
        second.stdout.contains("[needle]") && !second.stdout.contains("[full needle]"),
        "no journal write means no read history to recover: {:?}",
        second.stdout
    );
}

// ---------------------------------------------------------------------------
// --files-without-match
// ---------------------------------------------------------------------------

/// The complement of `-l`, not of `-v`: files with zero matching lines.
#[test]
fn files_without_match_lists_the_complement_of_files_with_matches() {
    let dir = fixture();

    let out = find(dir.path(), &["needle", "-t", "rust", "--files-without-match"]);
    assert!(out.stdout.contains("src/util.rs"), "{:?}", out.stdout);
    assert!(!out.stdout.contains("src/lib.rs"), "lib.rs matched, so it is excluded: {:?}", out.stdout);
}

#[test]
fn files_without_match_conflicts_with_files_with_matches() {
    let dir = fixture();
    let out = find(dir.path(), &["needle", "-l", "--files-without-match"]);
    assert_eq!(out.code, 2, "clap rejects the combination: {:?}", out.stderr);
}

// ---------------------------------------------------------------------------
// --type-list
// ---------------------------------------------------------------------------

/// Takes no PATTERN — reuses the same TypesBuilder `-t` itself builds.
#[test]
fn type_list_needs_no_pattern_and_lists_known_types() {
    let dir = fixture();
    let out = find(dir.path(), &["--type-list"]);
    assert_eq!(out.code, 0, "{:?}", out.stderr);
    assert!(out.stdout.lines().any(|l| l.starts_with("rust:") && l.contains(".rs")), "{:?}", out.stdout);
}

// ---------------------------------------------------------------------------
// -f / --file
// ---------------------------------------------------------------------------

/// One pattern per non-empty line, combined into the same alternation -e is.
#[test]
fn pattern_file_patterns_join_the_alternation() {
    let dir = fixture();
    let patterns = dir.path().join("patterns.txt");
    std::fs::write(&patterns, "needle\n\nunrelated\n").unwrap();

    let out = find(dir.path(), &["-t", "rust", "-l", "-f", patterns.to_str().unwrap()]);
    assert!(out.stdout.contains("src/lib.rs"), "needle: {:?}", out.stdout);
    assert!(out.stdout.contains("src/util.rs"), "unrelated: {:?}", out.stdout);
}

/// A missing pattern file is an expected, actionable failure — exit 2 with
/// a clear message, not a panic and not a silent empty result.
#[test]
fn a_missing_pattern_file_is_an_error() {
    let dir = fixture();
    let out = find(dir.path(), &["-f", "does-not-exist.txt"]);
    assert_eq!(out.code, 2);
    assert!(out.stderr.contains("does-not-exist.txt"), "{:?}", out.stderr);
}

/// An empty pattern file, with no -e/PATTERN either, is "no pattern given" —
/// the same error as running find with nothing at all.
#[test]
fn an_empty_pattern_file_alone_is_no_pattern_given() {
    let dir = fixture();
    let patterns = dir.path().join("empty.txt");
    std::fs::write(&patterns, "\n\n").unwrap();

    let out = find(dir.path(), &["-f", patterns.to_str().unwrap()]);
    assert_eq!(out.code, 2);
    assert!(out.stderr.contains("no pattern given"), "{:?}", out.stderr);
}
