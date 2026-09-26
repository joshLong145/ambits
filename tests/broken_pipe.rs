//! Process-level tests for writing into a closed pipe — `ambits … | head`.
//!
//! The reader going away is the consumer ending the conversation, so every
//! command must stop quietly: no panic (a `println!` into a closed pipe
//! panics, exit 101), no abort (an `eprintln!` into a closed stderr panics,
//! and the panic hook's own report then fails the same way), and no error
//! report (a propagated write error would surface as grep's exit 2). The exit
//! code must be the one the command gives when its output *is* read.
//!
//! Every stdout case writes more than [`MIN_OUTPUT`], and the read end is
//! dropped before the child is waited on, so at least one write is guaranteed
//! to fail with `BrokenPipe` — the tests do not depend on timing. The size is
//! asserted, not assumed: a case whose output shrank below a pipe buffer would
//! otherwise pass without ever meeting a closed pipe.

use std::path::Path;
use std::process::{Command, Stdio};

/// Linux's default pipe capacity, and macOS's largest.
const PIPE_BUFFER: usize = 64 * 1024;
/// Every stdout case must write at least this much. Twice a buffer, so the
/// guarantee survives a platform with a somewhat larger one.
const MIN_OUTPUT: usize = 2 * PIPE_BUFFER;

// Sized to clear MIN_OUTPUT with margin (smallest case ~138 KiB) while keeping
// a debug-build run fast: parse and attribution cost scales with CALLS.
const CALLS: usize = 4_000;
const HUGE_LINES: usize = 8_000;
const FILLER_FILES: usize = 3_000;

/// `src/big.rs`: `CALLS` callers of `callee` and a `HUGE_LINES`-line function,
/// so the search, `callers`, `show` and `--dump` outputs are large; plus
/// `FILLER_FILES` files matching nothing, so `--files-without-match` is too.
fn fixture() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("src")).unwrap();

    let mut src = String::from("pub fn callee() {}\n");
    for i in 0..CALLS {
        src.push_str(&format!("pub fn f{i}() {{ callee(); }}\n"));
    }
    src.push_str("pub fn huge() {\n");
    for i in 0..HUGE_LINES {
        src.push_str(&format!("    let _v{i} = {i};\n"));
    }
    src.push_str("}\n");
    std::fs::write(dir.path().join("src/big.rs"), src).unwrap();

    let filler = dir.path().join("filler");
    std::fs::create_dir_all(&filler).unwrap();
    for i in 0..FILLER_FILES {
        std::fs::write(filler.join(format!("padding_file_with_a_long_name_{i:05}.txt")), "x\n")
            .unwrap();
    }
    dir
}

/// The binary, isolated from the developer's machine: `HOME` is the fixture,
/// so no user-global `tools.toml` warns on stderr and no Claude Code session
/// logs are picked up.
fn ambits(root: &Path, args: &[&str]) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_ambits"));
    cmd.current_dir(root)
        .env("HOME", root)
        .env("USERPROFILE", root)
        .arg("-p")
        .arg(root)
        .args(args);
    cmd
}

fn code(status: std::process::ExitStatus) -> i32 {
    // `None` means killed by a signal — an abort is exactly what these tests
    // exist to catch, so report it rather than unwrap.
    status.code().unwrap_or_else(|| panic!("terminated by a signal: {status:?}"))
}

/// Run every case at once. Each is its own process, so a test costs its
/// slowest case rather than the sum of them.
fn in_parallel<C: Sync>(cases: &[C], check: impl Fn(&C) + Sync) {
    std::thread::scope(|s| {
        let handles: Vec<_> = cases.iter().map(|c| s.spawn(|| check(c))).collect();
        for h in handles {
            if let Err(panic) = h.join() {
                std::panic::resume_unwind(panic);
            }
        }
    });
}

/// Read to the end; returns (exit code, bytes written to stdout).
fn run_normally(root: &Path, args: &[&str]) -> (i32, usize) {
    let out = ambits(root, args).stderr(Stdio::null()).output().unwrap();
    (code(out.status), out.stdout.len())
}

/// stdout into a pipe nobody reads; returns (exit code, stderr).
fn run_stdout_closed(root: &Path, args: &[&str]) -> (i32, String) {
    let mut child = ambits(root, args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    drop(child.stdout.take());
    let out = child.wait_with_output().unwrap();
    (code(out.status), String::from_utf8_lossy(&out.stderr).into_owned())
}

/// stderr into a pipe nobody reads, stdout discarded. Dropped straight after
/// spawn, before the child has scanned anything, so in practice the child's
/// first stderr write fails; if it ever won that race the write would simply
/// succeed, so this can under-test but never falsely fail.
fn run_stderr_closed(root: &Path, args: &[&str]) -> i32 {
    let mut child = ambits(root, args)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    drop(child.stderr.take());
    code(child.wait().unwrap())
}

const LIMIT: &str = "--head-limit=0"; // the default cap of 200 would fit in a pipe buffer

/// Every printing command, with the exit code it gives when read to the end.
/// `--files-without-match` is the case that pins exit codes to "was there a
/// match" rather than "was output being written": its output is the files
/// that did *not* match, so a pattern matching nothing prints a long listing
/// and must still exit 1.
const CASES: &[(&[&str], i32)] = &[
    (&["rg", LIMIT, "callee"], 0),
    (&["rg", LIMIT, "--json", "callee"], 0),
    (&["rg", "-l", LIMIT, "."], 0),
    (&["rg", "--files-without-match", "absent_from_every_file"], 1),
    (&["grep", LIMIT, "callee"], 0),
    (&["grep", "-L", "absent_from_every_file"], 1),
    (&["callers", "callee"], 0),
    (&["callers", "--format", "json", "callee"], 0),
    (&["show", "src/big.rs::huge"], 0),
    (&["--dump", "--full"], 0),
];

#[test]
fn a_closed_stdout_ends_quietly_with_the_same_exit_code() {
    let dir = fixture();
    let root = dir.path();
    in_parallel(CASES, |&(args, expected)| {
        let (code, written) = run_normally(root, args);
        assert_eq!(code, expected, "ambits {args:?} read to the end");
        assert!(
            written > MIN_OUTPUT,
            "ambits {args:?} wrote {written} bytes; under {MIN_OUTPUT} a closed pipe is not guaranteed"
        );

        let (code, stderr) = run_stdout_closed(root, args);
        assert_eq!(code, expected, "ambits {args:?} into a closed pipe, stderr:\n{stderr}");
        assert!(stderr.is_empty(), "ambits {args:?}: expected nothing on stderr, got:\n{stderr}");
    });
}

#[test]
fn a_closed_stderr_never_aborts() {
    let dir = fixture();
    let root = dir.path();
    let cases: &[(&[&str], i32)] = &[
        // More matches than the default cap: the "… N more matches withheld"
        // notice goes to stderr after the results.
        (&["rg", "callee"], 0),
        // A bad pattern: the error report itself goes to stderr.
        (&["rg", "("], 2),
    ];
    in_parallel(cases, |&(args, expected)| {
        assert_eq!(run_stderr_closed(root, args), expected, "ambits {args:?}");
    });
}
