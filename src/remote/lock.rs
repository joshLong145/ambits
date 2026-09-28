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
    pub fn acquire(remote: &Store, store_id: &str) -> Result<Self> {
        create_private_dir(remote.root())?;
        let path = lock_path(remote);
        let holder = Holder { store: store_id.to_string(), pid: std::process::id(), start: now_secs(), token: random_token() };
        let mut file = match private_options().create_new(true).open(&path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                let age = read_holder(remote).map(|h| format!("taken {}s ago", now_secs().saturating_sub(h.start))).unwrap_or_default();
                bail!(
                    "the remote is locked by another push ({age}); if none is running, retry with --break-lock"
                );
            }
            Err(e) => return Err(e).wrap_err_with(|| format!("locking {}", path.display())),
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
    serde_json::from_str(&std::fs::read_to_string(lock_path(remote)).ok()?).ok()
}

/// Whether process `pid` is alive on this machine; `None` when that cannot
/// be told.
fn pid_alive(pid: u32) -> Option<bool> {
    #[cfg(unix)]
    {
        std::process::Command::new("kill")
            .args(["-0", &pid.to_string()])
            .stderr(std::process::Stdio::null())
            .status()
            .ok()
            .map(|s| s.success())
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
    std::fs::remove_file(lock_path(remote))?;
    Ok(Broken::Broken(holder))
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
        assert!(RemoteLock::acquire(&remote, "you").unwrap_err().to_string().contains("--break-lock"));
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
