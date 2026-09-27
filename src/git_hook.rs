//! The optional `post-commit` hook (spec §9.3): after each commit, resolve
//! where recent agent writes landed, so `touched` answers from the links
//! index instead of walking history.
//!
//! - Installed where git looks for hooks (`git rev-parse --git-path hooks`,
//!   which honours a repository's `core.hooksPath`), never into a global or
//!   system hooks directory.
//! - Chains to a hook already there, which keeps running first.
//! - Runs ambits in the background and ignores its outcome: a post-commit
//!   hook cannot fail a commit, and this one never delays it either.
//! - Acts only when the repository's top level has `.ambits/`.

use std::path::{Path, PathBuf};

use color_eyre::eyre::{bail, eyre, Result, WrapErr};

use crate::git::git;

const MARKER: &str = "# Installed by `ambits hook install --git`";
const CHAINED: &str = "post-commit.ambits-chained";

fn hooks_dir(repo_dir: &Path) -> Result<PathBuf> {
    if crate::git::Repo::discover(repo_dir).is_none() {
        bail!("{} is not inside a git repository", repo_dir.display());
    }
    // A hooksPath set in global or system config would make the hook apply
    // to every repository (§9.3: never global).
    if let Some(scope) = git(repo_dir, &["config", "--show-scope", "--get", "core.hooksPath"]) {
        let scope = String::from_utf8_lossy(&scope);
        if !scope.starts_with("local") && !scope.starts_with("worktree") {
            bail!("core.hooksPath is set outside this repository ({}); not installing a hook every repository would run", scope.trim());
        }
    }
    let out = git(repo_dir, &["rev-parse", "--git-path", "hooks"]).ok_or_else(|| eyre!("git could not locate the hooks directory"))?;
    let path = PathBuf::from(String::from_utf8_lossy(&out).trim());
    Ok(if path.is_absolute() { path } else { repo_dir.join(path) })
}

/// Quote `s` for a POSIX shell.
fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

fn script(ambits: &Path) -> String {
    format!(
        r#"#!/bin/sh
{MARKER}; remove with `ambits hook uninstall --git`.
# Records which commit recent agent writes landed in. Runs in the
# background and never affects the commit.
status=0
chained="$(dirname "$0")/{CHAINED}"
if [ -x "$chained" ]; then
  "$chained" "$@"
  status=$?
fi
top=$(git rev-parse --show-toplevel 2>/dev/null) || exit "$status"
if [ -d "$top/.ambits" ]; then
  ( {ambits} -p "$top" links refresh >/dev/null 2>&1 || true ) &
fi
exit "$status"
"#,
        ambits = shell_quote(&ambits.to_string_lossy())
    )
}

/// Install the hook for the repository containing `repo_dir`, running
/// `ambits` (an absolute path to this binary). Returns the hook's path.
pub fn install(repo_dir: &Path, ambits: &Path) -> Result<PathBuf> {
    let dir = hooks_dir(repo_dir)?;
    std::fs::create_dir_all(&dir)?;
    let hook = dir.join("post-commit");
    if let Ok(existing) = std::fs::read_to_string(&hook) {
        if !existing.contains(MARKER) {
            let chained = dir.join(CHAINED);
            if chained.exists() {
                bail!("{} already exists; resolve it by hand before installing", chained.display());
            }
            std::fs::rename(&hook, &chained).wrap_err("keeping the existing post-commit hook")?;
        }
    }
    std::fs::write(&hook, script(ambits))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&hook, std::fs::Permissions::from_mode(0o755))?;
    }
    Ok(hook)
}

/// Remove the hook, putting back any hook it chained to. Returns whether
/// there was one to remove.
pub fn uninstall(repo_dir: &Path) -> Result<bool> {
    let dir = hooks_dir(repo_dir)?;
    let hook = dir.join("post-commit");
    let ours = std::fs::read_to_string(&hook).is_ok_and(|s| s.contains(MARKER));
    if !ours {
        return Ok(false);
    }
    std::fs::remove_file(&hook)?;
    let chained = dir.join(CHAINED);
    if chained.exists() {
        std::fs::rename(&chained, &hook)?;
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_are_shell_quoted() {
        assert_eq!(shell_quote("/a b/it's"), r"'/a b/it'\''s'");
    }
}
