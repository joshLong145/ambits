//! Invoking git (spec §9.2): always hardened, never with a remote's input.
//!
//! Every call disables config-driven helpers that could run code
//! (`core.fsmonitor`, `diff.external`), the system config and any pager or
//! prompt. Callers put `--end-of-options` / `--` before anything that is
//! not ours, and validate commit ids with [`is_commit_id`].

use std::path::{Path, PathBuf};
use std::process::Command;

/// Variables that would point git at another repository, index or object
/// store than the one at `-C` — set, for instance, when running inside a git
/// hook. Always cleared.
const REPOSITORY_VARS: &[&str] = &[
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_NAMESPACE",
    "GIT_PREFIX",
];

/// The hardened `git -C <dir>` every call starts from.
fn command(dir: &Path) -> Command {
    let mut command = Command::new("git");
    for var in REPOSITORY_VARS {
        command.env_remove(var);
    }
    command
        .arg("-C")
        .arg(dir)
        .args(["-c", "core.fsmonitor=false", "-c", "diff.external=", "--no-pager"])
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_OPTIONAL_LOCKS", "0");
    command
}

/// `git config <scope> --get <key>` — `scope` is `--global` or `--system` —
/// with the system config visible, unlike every other call here. `Ok(None)`
/// when the key is unset; an error when git could not answer (too old for
/// the scope flag, or missing), so callers never mistake "unknown" for
/// "unset".
pub fn config_get(dir: &Path, scope: &str, key: &str) -> color_eyre::Result<Option<String>> {
    let mut cmd = command(dir);
    cmd.env_remove("GIT_CONFIG_NOSYSTEM");
    let out = cmd.args(["config", scope, "--get", key]).output()?;
    match out.status.code() {
        Some(0) => Ok(Some(String::from_utf8_lossy(&out.stdout).trim().to_string())),
        Some(1) => Ok(None),
        _ => Err(color_eyre::eyre::eyre!("git config {scope} --get {key} failed: {}", String::from_utf8_lossy(&out.stderr).trim())),
    }
}

/// `git <args>` in `dir` for tests: the same hardened command as production
/// (so a test run inside a git hook still targets `dir`), a fixed identity,
/// any extra `env`, and a panic with git's stderr on failure. Returns
/// trimmed stdout.
#[doc(hidden)]
pub fn test_git(dir: &Path, args: &[&str], env: &[(&str, &str)]) -> String {
    let mut cmd = command(dir);
    for (k, v) in [("GIT_AUTHOR_NAME", "t"), ("GIT_AUTHOR_EMAIL", "t@t"), ("GIT_COMMITTER_NAME", "t"), ("GIT_COMMITTER_EMAIL", "t@t")]
        .into_iter()
        .chain(env.iter().copied())
    {
        cmd.env(k, v);
    }
    let out = cmd.args(args).output().expect("git runs");
    assert!(out.status.success(), "git {args:?}: {}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

/// Run `git <args>` in `dir`; `None` if git is missing or the command fails.
pub fn git(dir: &Path, args: &[&str]) -> Option<Vec<u8>> {
    let output = command(dir).args(args).output().ok()?;
    output.status.success().then_some(output.stdout)
}

fn git_line(dir: &Path, args: &[&str]) -> Option<String> {
    let out = String::from_utf8(git(dir, args)?).ok()?;
    let line = out.trim_end_matches(['\n', '\r']);
    (!line.is_empty()).then(|| line.to_string())
}

/// A full commit id, as git prints one: 40 (SHA-1) or 64 (SHA-256) hex digits.
pub fn is_commit_id(s: &str) -> bool {
    matches!(s.len(), 40 | 64) && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// The repository `dir` belongs to, as far as ambits needs it.
#[derive(Debug, Clone)]
pub struct Repo {
    /// `dir` relative to the work tree's top, `/`-terminated or empty.
    pub prefix: String,
    /// The work tree's top directory.
    pub top: PathBuf,
    /// `HEAD`, or `None` before the first commit.
    pub head: Option<String>,
    dir: PathBuf,
}

impl Repo {
    /// `None` when `dir` is not inside a git work tree, or git is missing.
    pub fn discover(dir: &Path) -> Option<Self> {
        if git_line(dir, &["rev-parse", "--is-inside-work-tree"])? != "true" {
            return None;
        }
        let prefix = git_line(dir, &["rev-parse", "--show-prefix"]).unwrap_or_default();
        let top = PathBuf::from(git_line(dir, &["rev-parse", "--show-toplevel"])?);
        let head = git_line(dir, &["rev-parse", "--verify", "--quiet", "--end-of-options", "HEAD^{commit}"])
            .filter(|h| is_commit_id(h));
        Some(Self { prefix, top, head, dir: dir.to_path_buf() })
    }

    /// The directory the repository was discovered from.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Changed paths under `dir`, relative to it: tracked files modified,
    /// added or deleted, and untracked files not ignored — from
    /// `git status --porcelain=v1 -z` (§7). Renames are reported as a
    /// deletion plus an addition.
    pub fn status(&self) -> Option<Vec<StatusEntry>> {
        // `-- .` keeps git from walking the rest of a larger repository.
        let out = git(&self.dir, &["status", "--porcelain=v1", "-z", "--untracked-files=all", "--no-renames", "--", "."])?;
        let mut entries = Vec::new();
        for record in out.split(|&b| b == 0).filter(|r| r.len() > 3) {
            let code = &record[..2];
            let path = String::from_utf8_lossy(&record[3..]).into_owned();
            let Some(rel) = path.strip_prefix(&self.prefix) else { continue };
            entries.push(StatusEntry { path: rel.to_string(), untracked: code == b"??" });
        }
        Some(entries)
    }

    /// A path git resolves inside `.git`, such as `info/exclude`.
    pub fn git_path(&self, name: &str) -> Option<PathBuf> {
        let p = PathBuf::from(git_line(&self.dir, &["rev-parse", "--git-path", name])?);
        Some(if p.is_absolute() { p } else { self.dir.join(p) })
    }
}

/// One line of `git status`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatusEntry {
    /// Relative to the directory the [`Repo`] was discovered from.
    pub path: String,
    pub untracked: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sh(dir: &Path, args: &[&str]) {
        test_git(dir, args, &[]);
    }

    #[test]
    fn commit_ids_are_validated() {
        assert!(is_commit_id(&"a".repeat(40)));
        assert!(is_commit_id(&"0".repeat(64)));
        assert!(!is_commit_id(&"A".repeat(40)));
        assert!(!is_commit_id("HEAD"));
    }

    #[test]
    fn status_reports_changes_under_the_project_only() {
        let repo = tempfile::tempdir().unwrap();
        let root = repo.path();
        sh(root, &["init", "-q"]);
        std::fs::create_dir_all(root.join("proj/src")).unwrap();
        std::fs::write(root.join("proj/src/a.rs"), "fn a() {}\n").unwrap();
        std::fs::write(root.join("outside.rs"), "fn o() {}\n").unwrap();
        sh(root, &["add", "."]);
        sh(root, &["commit", "-qm", "init"]);
        assert!(Repo::discover(&root.join("proj")).unwrap().status().unwrap().is_empty());

        std::fs::write(root.join("proj/src/a.rs"), "fn a() { 1; }\n").unwrap();
        std::fs::write(root.join("proj/src/new.rs"), "fn n() {}\n").unwrap();
        std::fs::write(root.join("outside.rs"), "fn o() { 2; }\n").unwrap();
        let repo = Repo::discover(&root.join("proj")).unwrap();
        assert_eq!(repo.prefix, "proj/");
        assert!(repo.head.as_deref().is_some_and(is_commit_id));
        let mut status = repo.status().unwrap();
        status.sort_by(|a, b| a.path.cmp(&b.path));
        assert_eq!(
            status,
            vec![
                StatusEntry { path: "src/a.rs".into(), untracked: false },
                StatusEntry { path: "src/new.rs".into(), untracked: true },
            ]
        );
    }

    /// Variables that would point git at another repository (set inside a
    /// git hook, for one) are cleared on every call.
    #[test]
    fn inherited_repository_variables_are_cleared() {
        let cmd = command(Path::new("."));
        let cleared: Vec<_> = cmd.get_envs().filter(|(_, v)| v.is_none()).map(|(k, _)| k.to_owned()).collect();
        for var in ["GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"] {
            assert!(cleared.iter().any(|k| k == var), "{var} not cleared");
        }
    }

    #[test]
    fn outside_a_repository_there_is_no_repo() {
        let dir = tempfile::tempdir().unwrap();
        assert!(Repo::discover(dir.path()).is_none());
    }
}
