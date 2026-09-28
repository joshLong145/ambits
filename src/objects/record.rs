//! A session's journal as a snapshot sees it: one digest (spec §6.2, D13)
//! and the `coverage` and `writes` objects folded from the same bytes
//! (§5.1, D14).

use std::path::{Path, PathBuf};

use color_eyre::eyre::Result;
use serde_json::{json, Value};

use super::store::Store;
use super::sync_ignore::SyncIgnore;
use super::{b3, canonical, Kind, ObjectId};
use crate::journal::{self, DepthDto, JournalContents};

/// A session's journal, read once.
pub struct JournalPrefix {
    /// `journal_digest` (§6.2).
    pub digest: [u8; 32],
    /// The fold of exactly the digested bytes.
    pub contents: JournalContents,
}

/// Read every shard of `session`'s journal, each up to and including its
/// last newline, and derive both the digest and the fold from those bytes.
///
/// The one rule covers a torn last line (not yet part of the journal), an
/// append racing this read (whatever landed before the read is in, nothing
/// after), several shards, and older schema versions (§6.2).
pub fn read_prefix(project_root: &Path, session: &str) -> Result<JournalPrefix> {
    let dir = journal::journal_dir(project_root);
    // Each shard read once, keeping only complete lines.
    let mut shards: Vec<(PathBuf, String, Vec<u8>)> = Vec::new();
    for path in journal::session_shard_paths(&dir, session) {
        let mut bytes = std::fs::read(&path)?;
        bytes.truncate(bytes.iter().rposition(|&b| b == b'\n').map_or(0, |i| i + 1));
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        shards.push((path, name, bytes));
    }

    // Fold in the order `read_journal_session` uses (primary shard first),
    // so a snapshot's coverage is what the TUI and `restore` see: the fold
    // is last-hash-wins, so order matters.
    let mut contents = JournalContents::default();
    for (path, _, bytes) in &shards {
        journal::merge_shard(&mut contents, journal::fold_journal_text(path, &String::from_utf8_lossy(bytes)));
    }

    // Digest in name order, the order §6.2 specifies.
    shards.sort_by(|a, b| a.1.cmp(&b.1));
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"ambits-journal v1\0");
    for (_, name, bytes) in &shards {
        for part in [name.as_bytes(), bytes.as_slice()] {
            hasher.update(part.len().to_string().as_bytes());
            hasher.update(b"\0");
            hasher.update(part);
        }
    }
    Ok(JournalPrefix { digest: *hasher.finalize().as_bytes(), contents })
}

/// What [`write_records`] stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordStats {
    pub coverage: ObjectId,
    pub writes: ObjectId,
    pub reads: usize,
    pub write_count: usize,
}

/// Store the `coverage` and `writes` objects for `contents`: the fold's
/// result, sorted, with anything in an ignored file left out (§4). Neither
/// carries `history` records or an origin (§5.1).
pub fn write_records(store: &Store, session: &str, contents: &JournalContents, ignore: &SyncIgnore) -> Result<RecordStats> {
    let mut reads: Vec<Value> = Vec::new();
    // v1 journals carried no attribution; the session is the reader, as in
    // `App::rehydrate_from_journal`.
    let attributed: Vec<(&str, &str, &[u8; 32], _)> = if contents.agent_reads.is_empty() {
        contents.reads.iter().map(|(id, (h, d))| (id.as_str(), session, h, *d)).collect()
    } else {
        contents.agent_reads.iter().map(|((id, agent), (h, d))| (id.as_str(), agent.as_str(), h, *d)).collect()
    };
    for (id, agent, hash, depth) in attributed {
        // Normalized before matching, or `secret\k.rs` escapes `secret/`.
        let id = super::normalize_path(id);
        if !depth.is_seen() || ignore.ignores_symbol(&id) {
            continue;
        }
        reads.push(json!([id, b3(hash), DepthDto::from(depth), agent]));
    }
    sort_canonically(&mut reads)?;

    let mut writes: Vec<Value> = Vec::new();
    for record in contents.writes.values() {
        if !ignore.is_ignored(&super::normalize_path(&record.file)) {
            // Where a record came from is this machine's business.
            let record = crate::writes::WriteRecord { origin: None, ..record.clone() };
            writes.push(serde_json::to_value(&record)?);
        }
    }
    sort_canonically(&mut writes)?;

    let (read_count, write_count) = (reads.len(), writes.len());
    Ok(RecordStats {
        coverage: store.put(Kind::Coverage, &json!({"reads": reads}))?,
        writes: store.put(Kind::Writes, &json!({"writes": writes}))?,
        reads: read_count,
        write_count,
    })
}

/// Sort by canonical bytes, so equal sets give equal objects whatever order
/// the fold produced them in.
fn sort_canonically(items: &mut Vec<Value>) -> Result<()> {
    let mut keyed = items.drain(..).map(|v| Ok((canonical::to_bytes(&v)?, v))).collect::<Result<Vec<_>>>()?;
    keyed.sort_by(|a, b| a.0.cmp(&b.0));
    items.extend(keyed.into_iter().map(|(_, v)| v));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    const READ_A: &str = r#"{"kind":"read","sym":"src/a.rs::f","h":"b3:0000000000000000000000000000000000000000000000000000000000000000","d":"full_body","a":"agent-1"}"#;
    const READ_B: &str = r#"{"kind":"read","sym":"secret/k.rs::K","h":"b3:0000000000000000000000000000000000000000000000000000000000000001","d":"overview","a":"agent-1"}"#;

    fn journal(root: &Path, shard: &str, lines: &str) {
        let dir = journal::journal_dir(root);
        std::fs::create_dir_all(&dir).unwrap();
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(dir.join(shard)).unwrap();
        f.write_all(lines.as_bytes()).unwrap();
    }

    /// A torn last line is not yet part of the journal: neither the digest
    /// nor the fold sees it until its newline lands.
    #[test]
    fn a_torn_last_line_is_ignored_until_complete() {
        let dir = tempfile::tempdir().unwrap();
        journal(dir.path(), "s.ndjson", &format!("{READ_A}\n"));
        let before = read_prefix(dir.path(), "s").unwrap();
        journal(dir.path(), "s.ndjson", &READ_B[..40]);
        let torn = read_prefix(dir.path(), "s").unwrap();
        assert_eq!(before.digest, torn.digest);
        assert_eq!(torn.contents.reads.len(), 1);

        journal(dir.path(), "s.ndjson", &format!("{}\n", &READ_B[40..]));
        let complete = read_prefix(dir.path(), "s").unwrap();
        assert_ne!(complete.digest, before.digest);
        assert_eq!(complete.contents.reads.len(), 2);
    }

    #[test]
    fn every_shard_counts_and_another_session_does_not() {
        let dir = tempfile::tempdir().unwrap();
        journal(dir.path(), "s.ndjson", &format!("{READ_A}\n"));
        let one = read_prefix(dir.path(), "s").unwrap().digest;
        journal(dir.path(), "s.find.ndjson", &format!("{READ_B}\n"));
        journal(dir.path(), "other.ndjson", &format!("{READ_B}\n"));
        let two = read_prefix(dir.path(), "s").unwrap();
        assert_ne!(one, two.digest);
        assert_eq!(two.contents.reads.len(), 2);
    }

    /// A read recorded with Windows separators is still matched by a `/`
    /// pattern.
    #[test]
    fn ignore_matching_sees_normalized_paths() {
        let dir = tempfile::tempdir().unwrap();
        journal(dir.path(), "s.ndjson", &format!("{}\n", READ_B.replace("secret/k.rs", "secret\\\\k.rs")));
        let prefix = read_prefix(dir.path(), "s").unwrap();
        let ignore = SyncIgnore::new(&crate::ingest::tool_config::SyncConfig {
            ignore: Some(vec!["secret/".into()]),
            global_ignore: vec![],
        })
        .unwrap();
        let stats = write_records(&Store::at(dir.path()), "s", &prefix.contents, &ignore).unwrap();
        assert_eq!(stats.reads, 0);
    }

    /// Coverage folds shards in the TUI's order (primary first), which can
    /// differ from name order: the fold is last-hash-wins.
    #[test]
    fn shards_fold_primary_first_like_the_tui() {
        let dir = tempfile::tempdir().unwrap();
        let newer = READ_A.replace("\"h\":\"b3:0000000000000000000000000000000000000000000000000000000000000000\"", "\"h\":\"b3:1111111111111111111111111111111111111111111111111111111111111111\"");
        journal(dir.path(), "s.ndjson", &format!("{READ_A}\n"));
        journal(dir.path(), "s.find.ndjson", &format!("{newer}\n"));
        let prefix = read_prefix(dir.path(), "s").unwrap();
        let tui = journal::read_journal_session(&journal::journal_dir(dir.path()), "s");
        assert_eq!(prefix.contents.reads, tui.reads);
    }

    /// Ignored paths leave no read and no write behind (§4).
    #[test]
    fn records_in_ignored_files_are_left_out() {
        let dir = tempfile::tempdir().unwrap();
        journal(dir.path(), "s.ndjson", &format!("{READ_A}\n{READ_B}\n"));
        let prefix = read_prefix(dir.path(), "s").unwrap();
        let store = Store::at(dir.path());
        let ignore = SyncIgnore::new(&crate::ingest::tool_config::SyncConfig {
            ignore: Some(vec!["secret/".into()]),
            global_ignore: vec![],
        })
        .unwrap();
        let stats = write_records(&store, "s", &prefix.contents, &ignore).unwrap();
        assert_eq!(stats.reads, 1);
        let coverage = store.get(&stats.coverage, Kind::Coverage).unwrap().to_string();
        assert!(!coverage.contains("secret"), "{coverage}");
    }
}
