//! Shared by the integration tests.

use std::path::Path;
use std::process::Command;

/// Hardened `git <args>` in `dir`, as production runs it; see
/// `ambits::git::test_git`.
#[allow(dead_code)]
pub fn git(dir: &Path, args: &[&str]) -> String {
    ambits::git::test_git(dir, args, &[])
}

/// `git <args>` with extra environment (a commit date, say).
#[allow(dead_code)]
pub fn git_env(dir: &Path, args: &[&str], env: &[(&str, &str)]) -> String {
    ambits::git::test_git(dir, args, env)
}

/// The ambits binary, run in `root` with `HOME` there too, so no user or
/// global config leaks in.
#[allow(dead_code)]
pub fn ambits(root: &Path) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_ambits"));
    cmd.current_dir(root).env("HOME", root);
    cmd
}

/// Run ambits with `args`, asserting success; stdout.
#[allow(dead_code)]
pub fn run_ambits(root: &Path, args: &[&str]) -> String {
    let out = ambits(root).args(args).output().expect("ambits runs");
    assert!(out.status.success(), "ambits {args:?}: {}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8(out.stdout).expect("utf-8 output")
}
