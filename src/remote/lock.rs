//! The remote's ref lock (spec §12.2).
//!
//! An OS advisory lock means nothing to another machine sharing the remote,
//! so this one is a file created exclusively, `<remote>/refs.lock`, naming
//! who holds it: this store's random id (never a host, D16), a pid, a start
//! time and a token. A crash leaves it behind; `--break-lock` removes it
//! only when it is old enough **and** provably ours and dead, or the user
//! says so — a pid means nothing on another machine.

use std::io::Write as _;
use std::path::PathBuf;

use color_eyre::eyre::{bail, Result, WrapErr};
use serde::{Deserialize, Serialize};

use crate::objects::store::{create_private_dir, private_options, random_token, Store};
use crate::time::now_secs;

/// How old a lock must be before `--break-lock` considers it abandoned: a
/// push copies objects before it locks, so a held lock is only the ref
/// move — long past this, it is a crash, not a slow push.
pub const BREAK_AFTER_SECS: u64 = 10 * 60;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Holder {
    pub store: String,
    pub pid: u32,
    pub start: u64,
    pub token: String,
}

/// Held while moving a remote ref; removed on drop.
#[derive(Debug)]
pub struct RemoteLock {
    path: PathBuf,
    token: String,
}

fn lock_path(remote: &Store) -> PathBuf {
    remote.root().join("refs.lock")
}

impl RemoteLock {
    /// Take the lock, retrying briefly: a push holds it only to move a ref.
    pub fn acquire(remote: &Store, store_id: &str) -> Result<Self> {
        Self::acquire_within(remote, store_id, std::time::Duration::from_secs(3))
    }

    pub fn acquire_within(remote: &Store, store_id: &str, patience: std::time::Duration) -> Result<Self> {
        create_private_dir(remote.root())?;
        let path = lock_path(remote);
        let holder = Holder { store: store_id.to_string(), pid: std::process::id(), start: now_secs(), token: random_token() };
        let deadline = std::time::Instant::now() + patience;
        let mut file = loop {
            match private_options().create_new(true).open(&path) {
                Ok(f) => break f,
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists && std::time::Instant::now() < deadline => {
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    let age = read_holder(remote).map(|h| format!("taken {}s ago", now_secs().saturating_sub(h.start))).unwrap_or_default();
                    bail!("the remote is locked by another push ({age}); if none is running, retry with --break-lock");
                }
                Err(e) => return Err(e).wrap_err_with(|| format!("locking {}", path.display())),
            }
        };
        file.write_all(serde_json::to_string(&holder)?.as_bytes())?;
        file.sync_all()?;
        Ok(Self { path, token: holder.token })
    }
}

impl Drop for RemoteLock {
    fn drop(&mut self) {
        // Only our own: a lock broken and retaken meanwhile is someone else's.
        let ours = std::fs::read_to_string(&self.path)
            .ok()
            .and_then(|s| serde_json::from_str::<Holder>(&s).ok())
            .is_some_and(|h| h.token == self.token);
        if ours {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

pub fn read_holder(remote: &Store) -> Option<Holder> {
    read_holder_at(&lock_path(remote))
}

fn read_holder_at(path: &std::path::Path) -> Option<Holder> {
    let bytes = crate::objects::store::read_capped(path, 4096).ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// Whether process `pid` is alive on this machine; `None` when that cannot
/// be told.
fn pid_alive(pid: u32) -> Option<bool> {
    #[cfg(unix)]
    {
        let pid = i32::try_from(pid).ok()?;
        // SAFETY: signal 0 checks for existence; nothing is sent.
        if unsafe { libc::kill(pid, 0) } == 0 {
            return Some(true);
        }
        // Another user's process is alive, just not ours to signal.
        match std::io::Error::last_os_error().raw_os_error() {
            Some(libc::EPERM) => Some(true),
            Some(libc::ESRCH) => Some(false),
            _ => None,
        }
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        None
    }
}

/// What [`break_lock`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Broken {
    NoLock,
    Broken(Holder),
}

/// Remove an abandoned lock (spec §12.2): only once it is older than
/// [`BREAK_AFTER_SECS`], and then only if this store took it and its process
/// is gone, or `confirm` — asked with a description — says yes.
pub fn break_lock(remote: &Store, store_id: &str, confirm: impl FnOnce(&str) -> bool) -> Result<Broken> {
    let Some(holder) = read_holder(remote) else {
        if lock_path(remote).exists() {
            bail!("{} is unreadable; inspect it and delete it by hand", lock_path(remote).display());
        }
        return Ok(Broken::NoLock);
    };
    let age = now_secs().saturating_sub(holder.start);
    if age < BREAK_AFTER_SECS {
        bail!("the remote lock was taken {age}s ago; a push may still be running — wait {}s", BREAK_AFTER_SECS - age);
    }
    let ours_and_dead = holder.store == store_id && pid_alive(holder.pid) == Some(false);
    if !ours_and_dead {
        let who = if holder.store == store_id { "this store" } else { "another store" };
        let prompt = format!(
            "the remote lock is held by {who} (pid {}, {age}s ago). Breaking a live push's lock can lose that push. Break it?",
            holder.pid
        );
        if !confirm(&prompt) {
            bail!("the remote lock was left in place");
        }
    }
    // Move it aside and look again before deleting: if the lock was released
    // and retaken meanwhile, what was moved is someone else's live lock.
    let path = lock_path(remote);
    let aside = remote.root().join(format!("refs.lock.breaking-{}", random_token()));
    std::fs::rename(&path, &aside)?;
    if read_holder_at(&aside).is_some_and(|h| h.token == holder.token) {
        std::fs::remove_file(&aside)?;
        return Ok(Broken::Broken(holder));
    }
    if !path.exists() {
        std::fs::rename(&aside, &path)?;
    }
    Err(color_eyre::eyre::eyre!("the remote lock changed hands while it was being broken; nothing was removed"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn aged(remote: &Store, store: &str, pid: u32, age: u64) {
        let holder = Holder { store: store.into(), pid, start: now_secs() - age, token: "t".into() };
        std::fs::create_dir_all(remote.root()).unwrap();
        std::fs::write(lock_path(remote), serde_json::to_string(&holder).unwrap()).unwrap();
    }

    #[test]
    fn a_held_lock_refuses_a_second_pusher_and_goes_on_drop() {
        let dir = tempfile::tempdir().unwrap();
        let remote = Store::at_root(dir.path());
        let held = RemoteLock::acquire(&remote, "me").unwrap();
        let err = RemoteLock::acquire_within(&remote, "you", std::time::Duration::ZERO).unwrap_err();
        assert!(err.to_string().contains("--break-lock"));
        drop(held);
        RemoteLock::acquire(&remote, "you").unwrap();
    }

    #[test]
    fn a_young_lock_is_never_broken() {
        let dir = tempfile::tempdir().unwrap();
        let remote = Store::at_root(dir.path());
        aged(&remote, "me", u32::MAX, 5);
        assert!(break_lock(&remote, "me", |_| true).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn our_own_dead_lock_breaks_without_asking() {
        let dir = tempfile::tempdir().unwrap();
        let remote = Store::at_root(dir.path());
        // A pid far past any real one.
        aged(&remote, "me", 4_000_000, BREAK_AFTER_SECS + 1);
        assert!(matches!(break_lock(&remote, "me", |_| panic!("not asked")).unwrap(), Broken::Broken(_)));
        assert_eq!(break_lock(&remote, "me", |_| panic!("not asked")).unwrap(), Broken::NoLock);
    }

    /// Another machine's pid proves nothing: the user decides.
    #[test]
    fn another_stores_lock_needs_confirmation() {
        let dir = tempfile::tempdir().unwrap();
        let remote = Store::at_root(dir.path());
        aged(&remote, "them", 4_000_000, BREAK_AFTER_SECS + 1);
        assert!(break_lock(&remote, "me", |_| false).is_err());
        assert!(lock_path(&remote).exists());
        assert!(matches!(break_lock(&remote, "me", |p| p.contains("another store")).unwrap(), Broken::Broken(_)));
    }
}
