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

/// Where git runs this repository's hooks, and whether that is inside the
/// work tree (a tracked `.husky/`, say).
fn hooks_dir(repo_dir: &Path) -> Result<(PathBuf, bool)> {
    let Some(repo) = crate::git::Repo::discover(repo_dir) else {
        bail!("{} is not inside a git repository", repo_dir.display());
    };
    // A hooksPath from global or system config would make the hook apply to
    // every repository (§9.3: never global). Asked per scope, with system
    // config visible, and a git too old to answer is an error, not a pass.
    for scope in ["--global", "--system"] {
        if let Some(path) = crate::git::config_get(repo_dir, scope, "core.hooksPath")? {
            bail!("core.hooksPath is set in {} config ({path}); not installing a hook every repository would run", &scope[2..]);
        }
    }
    let path = repo.git_path("hooks").ok_or_else(|| eyre!("git could not locate the hooks directory"))?;
    let git_dir = git(repo_dir, &["rev-parse", "--absolute-git-dir"]).map(|o| PathBuf::from(String::from_utf8_lossy(&o).trim()));
    let canonical = |p: &Path| p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
    let (hooks, top) = (canonical(&path), canonical(&repo.top));
    let in_work_tree = hooks.starts_with(&top) && !git_dir.as_deref().is_some_and(|g| hooks.starts_with(canonical(g)));
    Ok((path, in_work_tree))
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
  ( {ambits} -p "$top" links refresh || true ) </dev/null >/dev/null 2>&1 &
fi
exit "$status"
"#,
        ambits = shell_quote(&ambits.to_string_lossy())
    )
}

/// What [`install`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Installed {
    pub path: PathBuf,
    /// The hooks directory is inside the work tree, where it may be tracked
    /// and shared.
    pub in_work_tree: bool,
}

/// Install the hook for the repository containing `repo_dir`, running
/// `ambits` (an absolute path to this binary).
pub fn install(repo_dir: &Path, ambits: &Path) -> Result<Installed> {
    let Some(ambits) = ambits.to_str() else {
        bail!("the ambits path {} is not UTF-8, so a hook cannot name it reliably", ambits.display());
    };
    let (dir, in_work_tree) = hooks_dir(repo_dir)?;
    std::fs::create_dir_all(&dir)?;
    let hook = dir.join("post-commit");
    let chained = dir.join(CHAINED);
    let mut moved = false;
    if let Ok(existing) = std::fs::read_to_string(&hook) {
        if !existing.contains(MARKER) {
            if chained.exists() {
                bail!("{} already exists; resolve it by hand before installing", chained.display());
            }
            std::fs::rename(&hook, &chained).wrap_err("keeping the existing post-commit hook")?;
            moved = true;
        }
    }
    let written = std::fs::write(&hook, script(Path::new(ambits))).and_then(|()| {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&hook, std::fs::Permissions::from_mode(0o755))?;
        }
        Ok(())
    });
    if let Err(e) = written {
        // Never leave the repository without the hook it had.
        if moved {
            let _ = std::fs::rename(&chained, &hook);
        }
        return Err(e).wrap_err("writing the post-commit hook");
    }
    Ok(Installed { path: hook, in_work_tree })
}

/// Remove the hook, putting back any hook it chained to. Returns whether
/// there was one to remove.
pub fn uninstall(repo_dir: &Path) -> Result<bool> {
    let (dir, _) = hooks_dir(repo_dir)?;
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
