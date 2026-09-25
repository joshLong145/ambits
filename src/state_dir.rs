//! The one per-project directory ambits owns: `.ambits/`.
//!
//! Everything ambits keeps beside a project — coverage journals under
//! `.ambits/coverage/`, the project's `tools.toml` — lives here. Earlier
//! releases used `.ambit/` for the same things while debug logs were
//! conventionally sent to `.ambits/`, which left two sibling directories for
//! one tool. [`LEGACY_STATE_DIR`] is still *read*, never written:
//!
//! - journals are moved across once by [`migrate_legacy_journals`], because a
//!   journal is the only record of what a past session read;
//! - a legacy `tools.toml` is read in place with a warning, because it may be
//!   committed, and moving a tracked file is the user's decision.

use std::path::Path;

/// Directory, relative to the project root, holding all ambits state.
pub const STATE_DIR: &str = ".ambits";

/// Where earlier releases kept the same state.
pub const LEGACY_STATE_DIR: &str = ".ambit";

/// Move `.ambit/coverage/` to `.ambits/coverage/` if only the former exists.
///
/// Best-effort and idempotent: a no-op once migrated, when there is nothing to
/// migrate, or when both exist (the new directory wins and the old one is left
/// for the user to inspect). Two processes racing here is harmless — the loser's
/// `rename` fails and the winner's result is what both then read. An emptied
/// `.ambit/` is removed; one still holding a `tools.toml` is kept.
pub fn migrate_legacy_journals(project_root: &Path) {
    let legacy = project_root.join(LEGACY_STATE_DIR).join(COVERAGE);
    let current = project_root.join(STATE_DIR).join(COVERAGE);
    if current.exists() || !legacy.is_dir() {
        return;
    }
    if std::fs::create_dir_all(project_root.join(STATE_DIR)).is_err() {
        return;
    }
    if std::fs::rename(&legacy, &current).is_ok() {
        // Fails, harmlessly, unless the directory is now empty.
        let _ = std::fs::remove_dir(project_root.join(LEGACY_STATE_DIR));
    }
}

const COVERAGE: &str = "coverage";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn journal_subdir_lives_under_the_state_dir() {
        assert_eq!(
            Path::new(crate::journal::JOURNAL_SUBDIR),
            Path::new(STATE_DIR).join(COVERAGE)
        );
    }

    #[test]
    fn legacy_journals_move_and_the_emptied_legacy_dir_goes() {
        let dir = tempfile::tempdir().unwrap();
        let legacy = dir.path().join(".ambit/coverage");
        std::fs::create_dir_all(&legacy).unwrap();
        std::fs::write(legacy.join("s.ndjson"), "{}\n").unwrap();

        migrate_legacy_journals(dir.path());

        assert!(dir.path().join(".ambits/coverage/s.ndjson").is_file());
        assert!(!dir.path().join(".ambit").exists());
    }

    #[test]
    fn a_legacy_dir_still_holding_config_is_kept() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join(".ambit/coverage")).unwrap();
        std::fs::write(dir.path().join(".ambit/tools.toml"), "version = 1\n").unwrap();

        migrate_legacy_journals(dir.path());

        assert!(dir.path().join(".ambits/coverage").is_dir());
        assert!(dir.path().join(".ambit/tools.toml").is_file());
    }

    #[test]
    fn an_existing_current_dir_wins_and_nothing_moves() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join(".ambit/coverage")).unwrap();
        std::fs::write(dir.path().join(".ambit/coverage/old.ndjson"), "").unwrap();
        std::fs::create_dir_all(dir.path().join(".ambits/coverage")).unwrap();

        migrate_legacy_journals(dir.path());

        assert!(dir.path().join(".ambit/coverage/old.ndjson").is_file());
        assert!(!dir.path().join(".ambits/coverage/old.ndjson").exists());
    }

    #[test]
    fn nothing_to_migrate_creates_nothing() {
        let dir = tempfile::tempdir().unwrap();
        migrate_legacy_journals(dir.path());
        assert!(!dir.path().join(STATE_DIR).exists());
    }
}
