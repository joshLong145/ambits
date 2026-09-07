//! Inspect and remove the coverage journals under `.ambit/coverage`.
//!
//! ## Why there is no automatic pruning
//!
//! The obvious housekeeping feature — trim to the N newest sessions — was
//! considered and deliberately left out.
//!
//! Journals do not grow the way caches usually do. A journal records one entry
//! per `(symbol, agent)` read, so its size is bounded by what an agent actually
//! read in one session, which is in turn bounded by context and wall-clock
//! time — not by the size of the repository. Nor does the *number* of journals
//! cost anything at runtime: startup loads only the current session's file and
//! restore resolves a path directly from a session id, so nothing scans this
//! directory in the hot path. A thousand stale journals are inert bytes.
//!
//! Against that, deleting one is genuinely lossy. A journal is the only record
//! of what a session read, recorded as it happened. Claude Code keeps session
//! transcripts and can resume them; a resumed session whose journal was pruned
//! falls back to reconstructing that history from logs, and the user is told
//! nothing. Reclaiming megabytes is not worth quietly degrading a restore.
//!
//! So removal stays explicit: [`clear`] requires naming a session or passing
//! `--all`, and [`status`] exists so anyone who does hit a size problem can see
//! it. If automatic cleanup is ever wanted, the honest trigger is the
//! disappearance of the session transcript — at that point the session cannot
//! be resumed and the journal has nothing left to serve.

use std::path::{Path, PathBuf};

use color_eyre::eyre::{Result, WrapErr};

use crate::journal::{read_journal, JOURNAL_SUBDIR};

/// One journal file on disk.
#[derive(Debug, Clone)]
pub struct JournalStat {
    pub session_id: String,
    pub path: PathBuf,
    pub bytes: u64,
    /// Total records, including the header.
    pub records: usize,
    /// Distinct symbols recoverable from the file.
    pub symbols: usize,
    /// Whole days since last modified, or `None` if unavailable.
    pub age_days: Option<u64>,
    pub schema_version: Option<u32>,
}

/// Directory holding this project's journals.
pub fn journal_dir(project_root: &Path) -> PathBuf {
    project_root.join(JOURNAL_SUBDIR)
}

/// Gather stats for every journal, newest first.
///
/// A file that cannot be read is skipped rather than failing the listing —
/// the point of `status` is to show what is there, and one unreadable file
/// should not hide the rest.
pub fn collect(project_root: &Path) -> Vec<JournalStat> {
    let dir = journal_dir(project_root);
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };

    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("ndjson") {
            continue;
        }
        let Some(session_id) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let Ok(meta) = entry.metadata() else { continue };

        let age_days = meta
            .modified()
            .ok()
            .and_then(|m| m.elapsed().ok())
            .map(|d| d.as_secs() / 86_400);

        let contents = read_journal(&path);
        let records = std::fs::read_to_string(&path)
            .map(|s| s.lines().filter(|l| !l.trim().is_empty()).count())
            .unwrap_or(0);

        out.push(JournalStat {
            session_id: session_id.to_string(),
            path,
            bytes: meta.len(),
            records,
            symbols: contents.reads.len(),
            age_days,
            schema_version: contents.header.map(|(v, _)| v),
        });
    }

    // Newest first: the current session is what anyone running this is most
    // likely looking for.
    out.sort_by(|a, b| a.age_days.cmp(&b.age_days).then_with(|| a.session_id.cmp(&b.session_id)));
    out
}

/// Render `ambits cache status`.
pub fn status(project_root: &Path) -> Result<()> {
    let stats = collect(project_root);
    let dir = journal_dir(project_root);

    if stats.is_empty() {
        println!("No coverage journals in {}", dir.display());
        println!();
        println!("Journals are written by the TUI. Without one, `restore-context`");
        println!("reconstructs the read history from session logs instead.");
        return Ok(());
    }

    println!("{}", dir.display());
    println!();
    println!(
        "{:<40}  {:>8}  {:>8}  {:>9}  {:>4}  AGE",
        "SESSION", "SYMBOLS", "RECORDS", "SIZE", "VER"
    );
    for s in &stats {
        println!(
            "{:<40}  {:>8}  {:>8}  {:>9}  {:>4}  {}",
            s.session_id,
            s.symbols,
            s.records,
            crate::fmt::bytes(s.bytes),
            s.schema_version
                .map(|v| v.to_string())
                .unwrap_or_else(|| "?".into()),
            s.age_days
                .map(|d| if d == 0 { "today".to_string() } else { format!("{d}d") })
                .unwrap_or_else(|| "-".into()),
        );
    }

    let total_bytes: u64 = stats.iter().map(|s| s.bytes).sum();
    println!();
    println!(
        "{} journal{}, {} on disk",
        stats.len(),
        if stats.len() == 1 { "" } else { "s" },
        crate::fmt::bytes(total_bytes)
    );

    Ok(())
}

/// Delete journals. Exactly one of `session` / `all` must be given.
pub fn clear(project_root: &Path, session: Option<&str>, all: bool) -> Result<()> {
    let dir = journal_dir(project_root);

    let targets: Vec<PathBuf> = match (session, all) {
        (Some(id), _) => {
            let path = dir.join(format!("{id}.ndjson"));
            if !path.exists() {
                println!("No journal for session {id} in {}", dir.display());
                return Ok(());
            }
            vec![path]
        }
        (None, true) => collect(project_root).into_iter().map(|s| s.path).collect(),
        // Refuse to guess. Deleting every journal is not a reasonable default
        // for a bare `cache clear`.
        (None, false) => {
            return Err(color_eyre::eyre::eyre!(
                "specify what to delete: --session <id>, or --all for every journal"
            ))
        }
    };

    if targets.is_empty() {
        println!("No coverage journals in {}", dir.display());
        return Ok(());
    }

    let mut removed = 0;
    for path in &targets {
        std::fs::remove_file(path)
            .wrap_err_with(|| format!("removing {}", path.display()))?;
        removed += 1;
    }

    println!(
        "Removed {removed} journal{}.",
        if removed == 1 { "" } else { "s" }
    );
    println!("Restores for those sessions now reconstruct from session logs instead.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::{encode_hash, DepthDto, HeaderRecord, Record, SUPPORTED_SCHEMA_VERSION};

    fn write_journal(root: &Path, session: &str, reads: usize) {
        let dir = journal_dir(root);
        std::fs::create_dir_all(&dir).unwrap();
        let mut out = serde_json::to_string(&Record::Header(Box::new(HeaderRecord {
            schema_version: SUPPORTED_SCHEMA_VERSION,
            created_at: "0".into(),
            session_id: session.into(),
            environment: crate::journal::EnvironmentManifest {
                project_root: "/p".into(),
                tree_fingerprint: encode_hash(&[0u8; 32]),
                ambit_version: "test".into(),
                backend: "tree-sitter".into(),
                parsers: vec![],
                tool_config_version: Some(1),
                filter: None,
                os: "testos".into(),
                arch: "testarch".into(),
                host: "testhost".into(),
            },
        })))
        .unwrap();
        out.push('\n');
        for i in 0..reads {
            out.push_str(
                &serde_json::to_string(&Record::Read {
                    symbol_id: format!("a.rs::s{i}"),
                    hash: encode_hash(&[i as u8; 32]),
                    depth: DepthDto::FullBody,
                    agent: Some("ag".into()),
                })
                .unwrap(),
            );
            out.push('\n');
        }
        std::fs::write(dir.join(format!("{session}.ndjson")), out).unwrap();
    }

    #[test]
    fn collect_reports_symbols_and_records_separately() {
        let dir = tempfile::tempdir().unwrap();
        write_journal(dir.path(), "sess-a", 3);

        let stats = collect(dir.path());
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].session_id, "sess-a");
        assert_eq!(stats[0].symbols, 3);
        assert_eq!(stats[0].records, 4, "three reads plus the header");
        assert_eq!(stats[0].schema_version, Some(SUPPORTED_SCHEMA_VERSION));
    }

    #[test]
    fn collect_is_empty_when_no_journals_exist() {
        let dir = tempfile::tempdir().unwrap();
        assert!(collect(dir.path()).is_empty());
    }

    #[test]
    fn collect_ignores_non_journal_files() {
        let dir = tempfile::tempdir().unwrap();
        write_journal(dir.path(), "sess-a", 1);
        std::fs::write(journal_dir(dir.path()).join("notes.txt"), "hi").unwrap();

        assert_eq!(collect(dir.path()).len(), 1);
    }

    #[test]
    fn clear_removes_only_the_named_session() {
        let dir = tempfile::tempdir().unwrap();
        write_journal(dir.path(), "keep", 1);
        write_journal(dir.path(), "drop", 1);

        clear(dir.path(), Some("drop"), false).unwrap();

        let remaining: Vec<String> = collect(dir.path())
            .into_iter()
            .map(|s| s.session_id)
            .collect();
        assert_eq!(remaining, vec!["keep".to_string()]);
    }

    #[test]
    fn clear_all_removes_everything() {
        let dir = tempfile::tempdir().unwrap();
        write_journal(dir.path(), "a", 1);
        write_journal(dir.path(), "b", 1);

        clear(dir.path(), None, true).unwrap();
        assert!(collect(dir.path()).is_empty());
    }

    /// A bare `cache clear` must not be interpreted as "delete everything".
    #[test]
    fn clear_without_a_target_refuses() {
        let dir = tempfile::tempdir().unwrap();
        write_journal(dir.path(), "a", 1);

        assert!(clear(dir.path(), None, false).is_err());
        assert_eq!(collect(dir.path()).len(), 1, "nothing was deleted");
    }

    #[test]
    fn clear_on_a_missing_session_is_not_an_error() {
        let dir = tempfile::tempdir().unwrap();
        write_journal(dir.path(), "a", 1);

        clear(dir.path(), Some("nope"), false).unwrap();
        assert_eq!(collect(dir.path()).len(), 1);
    }
}
