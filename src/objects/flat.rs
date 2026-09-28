//! One flat, append-only NDJSON file per store — the links index, the
//! never-landed cache, the reflog, notes — instead of a file per entry.
//!
//! - **Writes lock.** Every append and every rewrite holds `<file>.lock`
//!   (an `fs4` advisory lock, so a crashed holder releases it). A rewrite
//!   therefore cannot swallow an append made meanwhile.
//! - **Reads do not.** A record is one line written with one `write`; a
//!   line a crash cut short is skipped, and the next append starts a fresh
//!   line after it.
//! - **Last wins.** Stores fold their lines by key; superseded lines are
//!   dead weight until a rewrite ([`FlatLog::rewrite`]) drops them.

use std::fs;
use std::io::{Read as _, Seek as _, SeekFrom, Write as _};
use std::path::{Path, PathBuf};

use color_eyre::eyre::{Result, WrapErr};
use serde::de::DeserializeOwned;
use serde::Serialize;

use super::store::{create_private_dir, private_options, write_atomic_with, Durability};

/// An append-only NDJSON file.
#[derive(Debug, Clone)]
pub struct FlatLog {
    path: PathBuf,
}

/// Held while writing; released on drop.
pub struct FlatLock(#[allow(dead_code)] fs::File);

impl FlatLog {
    pub fn at(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Every record that parses as `T`, in file order. A missing file is
    /// empty; unparseable or cut-short lines are skipped.
    pub fn read<T: DeserializeOwned>(&self) -> Vec<T> {
        let Ok(text) = fs::read_to_string(&self.path) else { return Vec::new() };
        text.lines().filter_map(|l| serde_json::from_str(l).ok()).collect()
    }

    /// How many lines the file holds, parseable or not.
    pub fn line_count(&self) -> usize {
        fs::read_to_string(&self.path).map_or(0, |t| t.lines().count())
    }

    /// Take the write lock, waiting for another writer to finish.
    pub fn lock(&self) -> Result<FlatLock> {
        let dir = self.path.parent().unwrap_or(Path::new("."));
        create_private_dir(dir)?;
        let mut name = self.path.file_name().unwrap_or_default().to_os_string();
        name.push(".lock");
        let lock = dir.join(name);
        let file = private_options()
            .create(true)
            .truncate(false)
            .open(&lock)
            .wrap_err_with(|| format!("opening {}", lock.display()))?;
        fs4::FileExt::lock(&file).wrap_err_with(|| format!("waiting for {}", lock.display()))?;
        Ok(FlatLock(file))
    }

    /// Append `records`, one line each, in one write. The caller holds
    /// the lock.
    pub fn append<T: Serialize>(&self, _lock: &FlatLock, records: &[T], durability: Durability) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        create_private_dir(self.path.parent().unwrap_or(Path::new(".")))?;
        let mut file = private_options()
            .create(true)
            .read(true)
            .append(true)
            .open(&self.path)
            .wrap_err_with(|| format!("opening {}", self.path.display()))?;
        // A line a crash cut short must not swallow the first new one.
        let mut body = String::new();
        if file.seek(SeekFrom::End(0))? > 0 {
            file.seek(SeekFrom::End(-1))?;
            let mut last = [0u8];
            file.read_exact(&mut last)?;
            if last[0] != b'\n' {
                body.push('\n');
            }
        }
        for r in records {
            body.push_str(&serde_json::to_string(r)?);
            body.push('\n');
        }
        file.write_all(body.as_bytes()).wrap_err_with(|| format!("appending to {}", self.path.display()))?;
        if durability == Durability::Fsync {
            file.sync_all()?;
        }
        Ok(())
    }

    /// Replace the file with `records`, atomically. The caller holds the
    /// lock, and has read the file under it.
    pub fn rewrite<T: Serialize>(&self, _lock: &FlatLock, records: &[T], durability: Durability) -> Result<()> {
        let mut body = String::new();
        for r in records {
            body.push_str(&serde_json::to_string(r)?);
            body.push('\n');
        }
        write_atomic_with(&self.path, body.as_bytes(), durability)
    }
}

/// Whether a file of `lines` lines holding `live` records is worth
/// rewriting: once most of it is dead, and not for a handful of lines.
pub fn worth_compacting(lines: usize, live: usize) -> bool {
    lines > 64 && lines > 2 * live
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct R {
        k: u32,
    }

    #[test]
    fn appends_read_back_in_order_and_a_rewrite_replaces() {
        let dir = tempfile::tempdir().unwrap();
        let log = FlatLog::at(dir.path().join("sub/x.ndjson"));
        assert!(log.read::<R>().is_empty());
        let lock = log.lock().unwrap();
        log.append(&lock, &[R { k: 1 }, R { k: 2 }], Durability::NoSync).unwrap();
        log.append(&lock, &[R { k: 3 }], Durability::Fsync).unwrap();
        assert_eq!(log.read::<R>(), vec![R { k: 1 }, R { k: 2 }, R { k: 3 }]);
        log.rewrite(&lock, &[R { k: 9 }], Durability::NoSync).unwrap();
        assert_eq!(log.read::<R>(), vec![R { k: 9 }]);
        assert_eq!(log.line_count(), 1);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            for f in ["sub/x.ndjson", "sub/x.ndjson.lock"] {
                assert_eq!(std::fs::metadata(dir.path().join(f)).unwrap().permissions().mode() & 0o777, 0o600, "{f}");
            }
        }
    }

    /// A crash mid-append leaves a cut line: it is skipped, and what is
    /// appended next is whole.
    #[test]
    fn a_cut_short_line_is_skipped_and_does_not_eat_the_next() {
        let dir = tempfile::tempdir().unwrap();
        let log = FlatLog::at(dir.path().join("x.ndjson"));
        std::fs::write(log.path(), "{\"k\":1}\n{\"k\":").unwrap();
        let lock = log.lock().unwrap();
        log.append(&lock, &[R { k: 2 }], Durability::NoSync).unwrap();
        assert_eq!(log.read::<R>(), vec![R { k: 1 }, R { k: 2 }]);
    }

    /// Writers wait for each other: many threads appending lose nothing.
    #[test]
    fn concurrent_writers_lose_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.ndjson");
        let threads: Vec<_> = (0..8)
            .map(|t| {
                let log = FlatLog::at(path.clone());
                std::thread::spawn(move || {
                    for i in 0..25 {
                        let lock = log.lock().unwrap();
                        log.append(&lock, &[R { k: t * 100 + i }], Durability::NoSync).unwrap();
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(FlatLog::at(path).read::<R>().len(), 200);
    }

    #[test]
    fn compaction_waits_for_mostly_dead_files() {
        assert!(!worth_compacting(10, 1), "too small to bother");
        assert!(!worth_compacting(100, 60));
        assert!(worth_compacting(100, 40));
    }
}
