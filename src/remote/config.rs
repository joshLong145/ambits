//! `.ambits/config`: this store's id and its remotes (spec §11, §12.2).
//!
//! ```toml
//! store_id = "3f9c…"          # random; names this store in remote locks, never a host (D16)
//!
//! [remotes.origin]
//! path = "/mnt/team/ambits"
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use color_eyre::eyre::{bail, eyre, Result, WrapErr};
use serde::{Deserialize, Serialize};

use crate::objects::refs::valid_remote_name;
use crate::objects::store::{random_token, write_atomic, Store};

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Config {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub store_id: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub remotes: BTreeMap<String, RemoteConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RemoteConfig {
    /// An absolute path: the remote is a directory laid out as `.ambits/` is.
    pub path: PathBuf,
}

fn config_path(project_root: &Path) -> PathBuf {
    project_root.join(crate::state_dir::STATE_DIR).join("config")
}

impl Config {
    /// The project's config; empty when there is none yet.
    pub fn load(project_root: &Path) -> Result<Self> {
        let path = config_path(project_root);
        match std::fs::read_to_string(&path) {
            Ok(text) => toml::from_str(&text).wrap_err_with(|| format!("reading {}", path.display())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e).wrap_err_with(|| format!("reading {}", path.display())),
        }
    }

    pub fn save(&self, project_root: &Path) -> Result<()> {
        write_atomic(&config_path(project_root), toml::to_string(self)?.as_bytes())
    }

    /// This store's id, minted and saved the first time it is asked for.
    pub fn store_id(project_root: &Path) -> Result<String> {
        let mut config = Self::load(project_root)?;
        if let Some(id) = &config.store_id {
            return Ok(id.clone());
        }
        let id = format!("{}{}", random_token(), random_token());
        config.store_id = Some(id.clone());
        config.save(project_root)?;
        Ok(id)
    }
}

/// Add remote `name` at `path`. Returns warnings: a world-writable remote
/// lets anyone rewrite the history everyone pulls.
pub fn add(project_root: &Path, name: &str, path: &Path) -> Result<Vec<String>> {
    if !valid_remote_name(name) {
        bail!("not a remote name: {name:?} (letters, digits, `.`, `_`, `-`; not starting with `.`)");
    }
    let mut config = Config::load(project_root)?;
    if config.remotes.contains_key(name) {
        bail!("remote {name} already exists; remove it first");
    }
    let path = if path.is_absolute() { path.to_path_buf() } else { std::env::current_dir()?.join(path) };
    let path = path.canonicalize().unwrap_or(path);
    if path.starts_with(project_root.join(crate::state_dir::STATE_DIR)) {
        bail!("a remote cannot live inside this project's own store");
    }
    let mut warnings = Vec::new();
    #[cfg(unix)]
    if let Ok(meta) = std::fs::metadata(&path) {
        use std::os::unix::fs::PermissionsExt;
        if meta.permissions().mode() & 0o002 != 0 {
            warnings.push(format!("{} is world-writable: anyone on this machine can rewrite what you pull", path.display()));
        }
    }
    config.remotes.insert(name.to_string(), RemoteConfig { path });
    config.save(project_root)?;
    Ok(warnings)
}

pub fn remove(project_root: &Path, name: &str) -> Result<()> {
    let mut config = Config::load(project_root)?;
    if config.remotes.remove(name).is_none() {
        bail!("no remote named {name}");
    }
    config.save(project_root)
}

/// The remote to use: `name`, or the only one there is, or `origin`.
pub fn resolve(project_root: &Path, name: Option<&str>) -> Result<(String, Store)> {
    let config = Config::load(project_root)?;
    let name = match name {
        Some(n) => n.to_string(),
        None if config.remotes.len() == 1 => config.remotes.keys().next().cloned().unwrap_or_default(),
        None if config.remotes.contains_key("origin") => "origin".into(),
        None if config.remotes.is_empty() => return Err(eyre!("no remotes; add one with `ambits remote add <name> <path>`")),
        None => return Err(eyre!("several remotes and none is `origin`; name one")),
    };
    let remote = config.remotes.get(&name).ok_or_else(|| eyre!("no remote named {name}"))?;
    Ok((name, Store::at_root(&remote.path)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remotes_are_added_resolved_and_removed() {
        let dir = tempfile::tempdir().unwrap();
        let (root, far) = (dir.path().join("p"), dir.path().join("far"));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(&far).unwrap();
        assert!(resolve(&root, None).is_err(), "none yet");
        add(&root, "team", &far).unwrap();
        let (name, store) = resolve(&root, None).unwrap();
        assert_eq!((name.as_str(), store.root()), ("team", far.canonicalize().unwrap().as_path()));
        assert!(add(&root, "team", &far).is_err(), "no duplicates");
        for bad in ["", ".hidden", "a/b", "../x", "a b"] {
            assert!(add(&root, bad, &far).is_err(), "{bad:?}");
        }
        assert!(add(&root, "self", &root.join(".ambits/objects")).is_err());
        remove(&root, "team").unwrap();
        assert!(resolve(&root, Some("team")).is_err());
    }

    #[test]
    fn the_store_id_is_minted_once() {
        let dir = tempfile::tempdir().unwrap();
        let a = Config::store_id(dir.path()).unwrap();
        assert_eq!(a.len(), 32);
        assert_eq!(Config::store_id(dir.path()).unwrap(), a);
    }

    #[cfg(unix)]
    #[test]
    fn a_world_writable_remote_is_warned_about() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let far = dir.path().join("far");
        std::fs::create_dir_all(&far).unwrap();
        std::fs::set_permissions(&far, std::fs::Permissions::from_mode(0o777)).unwrap();
        let warnings = add(dir.path(), "open", &far).unwrap();
        assert!(warnings[0].contains("world-writable"), "{warnings:?}");
    }
}
