//! `[sync] ignore` (spec §4): what a snapshot leaves out.
//!
//! Applied to every record type at snapshot time — tree objects, reads,
//! writes and the dirty list — so an ignored path appears nowhere in a
//! snapshot. Only a digest of the patterns is recorded: the patterns
//! themselves may name confidential directories.

use color_eyre::eyre::{Result, WrapErr};
use ignore::gitignore::{Gitignore, GitignoreBuilder};
use serde_json::json;

use crate::ingest::tool_config::SyncConfig;

/// The effective `[sync] ignore` of one project.
pub struct SyncIgnore {
    project: Gitignore,
    /// Matched separately, so no project pattern — `!` included — can
    /// re-include what the user excluded everywhere.
    global: Gitignore,
    digest: [u8; 32],
}

impl SyncIgnore {
    pub fn new(config: &SyncConfig) -> Result<Self> {
        let project = config.ignore.clone().unwrap_or_default();
        let global = config.global_ignore.clone();
        let digest = *blake3::hash(&super::canonical::to_bytes(&json!({"global": global, "project": project}))?).as_bytes();
        Ok(Self {
            project: matcher(&project).wrap_err("[sync] ignore")?,
            global: matcher(&global).wrap_err("user-global [sync] ignore")?,
            digest,
        })
    }

    /// Nothing ignored.
    pub fn none() -> Self {
        Self::new(&SyncConfig::default()).expect("empty patterns always compile")
    }

    /// Whether the project-relative file `path` is excluded — by a pattern
    /// naming it or any directory above it.
    pub fn is_ignored(&self, path: &str) -> bool {
        let hit = |m: &Gitignore| m.matched_path_or_any_parents(path, false).is_ignore();
        hit(&self.global) || hit(&self.project)
    }

    /// Whether the symbol `id` (`<path>::<name-path>`) lives in an ignored file.
    pub fn ignores_symbol(&self, id: &str) -> bool {
        self.is_ignored(crate::symbols::split_id(id).0)
    }

    /// Digest of the effective patterns, a snapshot input (§6.1).
    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
}

fn matcher(patterns: &[String]) -> Result<Gitignore> {
    // Rooted at "" so patterns match project-relative paths.
    let mut builder = GitignoreBuilder::new("");
    for p in patterns {
        builder.add_line(None, p)?;
    }
    Ok(builder.build()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sync(project: &[&str], global: &[&str]) -> SyncIgnore {
        SyncIgnore::new(&SyncConfig {
            ignore: Some(project.iter().map(|s| s.to_string()).collect()),
            global_ignore: global.iter().map(|s| s.to_string()).collect(),
        })
        .unwrap()
    }

    #[test]
    fn patterns_use_gitignore_syntax() {
        let s = sync(&["secrets/**", "vendor/", "*.gen.rs"], &[]);
        assert!(s.is_ignored("secrets/key.rs"));
        assert!(s.is_ignored("vendor/lib/x.rs"), "a directory pattern covers what is under it");
        assert!(s.is_ignored("src/a.gen.rs"));
        assert!(!s.is_ignored("src/a.rs"));
        assert!(s.ignores_symbol("secrets/key.rs::Key/new"));
    }

    /// The project cannot negate a user-global exclusion (§4).
    #[test]
    fn a_global_pattern_cannot_be_negated_by_the_project() {
        let s = sync(&["!private/**"], &["private/**"]);
        assert!(s.is_ignored("private/a.rs"));
    }

    #[test]
    fn the_digest_follows_the_patterns_not_their_source() {
        assert_eq!(sync(&["a"], &[]).digest(), sync(&["a"], &[]).digest());
        assert_ne!(sync(&["a"], &[]).digest(), sync(&[], &["a"]).digest());
        assert_ne!(sync(&[], &[]).digest(), sync(&["a"], &[]).digest());
    }
}
