//! The snapshot object model (spec §5–§8): content-addressed objects in a
//! local store under `.ambits/`, and the snapshots that tie them together.
//!
//! - [`canonical`] — the one byte encoding every object has.
//! - [`store`] — loose objects on disk, written atomically.
//! - [`record`] — `coverage` and `writes` objects from a journal prefix.
//! - [`inputs`] — what a snapshot's id is derived from (D17).
//! - [`refs`] — session refs, their lock, and the reflog.
//! - [`snapshot`] — `ambits snapshot` and `ambits log`.
//! - [`restore`] — `ambits restore`: a snapshot back into a session.
//! - [`gc`] — reclaiming unreachable objects.

pub mod canonical;
pub mod flat;
pub mod gc;
pub mod inputs;
pub mod record;
pub mod refs;
pub mod restore;
pub mod snapshot;
pub mod store;
pub mod sync_ignore;

use std::fmt;

use color_eyre::eyre::{bail, Result};
use unicode_normalization::UnicodeNormalization;

/// An object or snapshot id: 32 bytes of BLAKE3, written as 64 lowercase hex
/// digits.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ObjectId(pub [u8; 32]);

impl ObjectId {
    /// Parse a full id, validating it as untrusted input (§9.1).
    pub fn parse(s: &str) -> Result<Self> {
        if s.len() != 64 || !s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
            bail!("not an object id: {s:?}");
        }
        let mut out = [0u8; 32];
        for (i, byte) in out.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&s[2 * i..2 * i + 2], 16)?;
        }
        Ok(Self(out))
    }

    pub fn hex(&self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// The first 12 hex digits, for display.
    pub fn short(&self) -> String {
        self.hex()[..12].to_string()
    }
}

impl fmt::Display for ObjectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.hex())
    }
}

impl fmt::Debug for ObjectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ObjectId({})", self.short())
    }
}

/// Object types (§5.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    Coverage,
    Writes,
    Snapshot,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Coverage => "coverage",
            Kind::Writes => "writes",
            Kind::Snapshot => "snapshot",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "coverage" => Kind::Coverage,
            "writes" => Kind::Writes,
            "snapshot" => Kind::Snapshot,
            _ => return None,
        })
    }
}

/// Id of a content-addressed object:
/// `BLAKE3("ambits-obj v1\0" ‖ type ‖ "\0" ‖ len ‖ "\0" ‖ payload)` (§5.2).
pub fn content_id(kind: Kind, payload: &[u8]) -> ObjectId {
    let mut h = blake3::Hasher::new();
    h.update(b"ambits-obj v1\0");
    h.update(kind.name().as_bytes());
    h.update(b"\0");
    h.update(payload.len().to_string().as_bytes());
    h.update(b"\0");
    h.update(payload);
    ObjectId(*h.finalize().as_bytes())
}

/// `b3:<hex>`, the form hashes take inside object payloads — the same form
/// the journal uses.
pub fn b3(hash: &[u8; 32]) -> String {
    crate::journal::encode_hash(hash)
}

/// A project-relative path in the one form objects carry: `/`-separated and
/// NFC-normalized (§5.2), so the same file named from Windows or with a
/// decomposed accent is the same path.
pub fn normalize_path(path: &str) -> String {
    path.replace('\\', "/").nfc().collect()
}

/// A path inside the project in the one form every record carries:
/// relative, `/`-separated, NFC (see [`normalize_path`]). `None` for the
/// project root itself or anything that climbs out of it (`..`, a root or
/// drive prefix). Relative to the project already — strip the root first.
pub fn project_rel(path: &std::path::Path) -> Option<String> {
    use std::path::Component;
    let mut parts = Vec::new();
    for c in path.components() {
        match c {
            Component::Normal(p) => parts.push(p.to_string_lossy()),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => return None,
        }
    }
    (!parts.is_empty()).then(|| normalize_path(&parts.join("/")))
}

/// BLAKE3 of `parts` under a `domain` tag, each part length-prefixed so no
/// split of one sequence can collide with another's. For ambits' local keys
/// (link file names, the Serena fingerprint); the spec fixes its own
/// framings for object ids and digests, which stay as they are.
pub fn hash_framed<'a>(domain: &str, parts: impl IntoIterator<Item = &'a [u8]>) -> blake3::Hash {
    let mut h = blake3::Hasher::new();
    h.update(domain.as_bytes());
    h.update(b"\0");
    for part in parts {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part);
    }
    h.finalize()
}

/// `b3:` hash of a file's raw bytes: the `fh` of a write, a dirty file's
/// fingerprint, and what a committed blob is compared by.
pub fn file_hash(bytes: &[u8]) -> String {
    b3(blake3::hash(bytes).as_bytes())
}

/// A regular file's bytes; `None` when it is gone or is anything else — a
/// symlink is never followed (§9.1).
pub fn read_regular(path: &std::path::Path) -> Option<Vec<u8>> {
    std::fs::symlink_metadata(path).ok().filter(|m| m.is_file())?;
    std::fs::read(path).ok()
}

/// A `dir` entry name must be one plain path component (§9.1).
pub fn valid_entry_name(name: &str) -> bool {
    !name.is_empty()
        && name != "."
        && name != ".."
        && !name.chars().any(|c| matches!(c, '/' | '\\' | '\0' | ':') || c.is_control())
}

/// A project-relative record path: relative, normalized, no `..` (§9.1).
pub fn valid_record_path(path: &str) -> bool {
    !path.is_empty()
        && !path.starts_with('/')
        && path.split('/').all(valid_entry_name)
        && unicode_normalization::is_nfc(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_round_trip_and_reject_junk() {
        let id = content_id(Kind::Coverage, b"[]");
        assert_eq!(ObjectId::parse(&id.hex()).unwrap(), id);
        for bad in ["", "abc", &"A".repeat(64), &"0".repeat(63), &format!("{}g", "0".repeat(63))] {
            assert!(ObjectId::parse(bad).is_err(), "{bad:?}");
        }
    }

    /// The type is part of the id: equal payloads of two types are two objects.
    #[test]
    fn the_type_is_part_of_the_id() {
        assert_ne!(content_id(Kind::Coverage, b"[]"), content_id(Kind::Writes, b"[]"));
    }

    #[test]
    fn paths_normalize_separators_and_unicode() {
        // "é" precomposed vs "e" + combining acute.
        assert_eq!(normalize_path("src\\caf\u{e9}.rs"), normalize_path("src/cafe\u{301}.rs"));
    }

    #[test]
    fn project_rel_normalizes_and_refuses_escapes() {
        use std::path::Path;
        assert_eq!(project_rel(Path::new("./src/cafe\u{301}.rs")).as_deref(), Some("src/caf\u{e9}.rs"));
        for bad in ["", ".", "../x", "/etc/passwd", "src/../../x"] {
            assert_eq!(project_rel(Path::new(bad)), None, "{bad}");
        }
    }

    #[test]
    fn entry_names_are_one_plain_component() {
        for ok in ["a.rs", "src", ".github", "ünï"] {
            assert!(valid_entry_name(ok), "{ok}");
        }
        for bad in ["", ".", "..", "a/b", "a\\b", "c:", "a\0", "a\nb"] {
            assert!(!valid_entry_name(bad), "{bad:?}");
        }
        assert!(valid_record_path("src/a.rs"));
        assert!(!valid_record_path("/etc/passwd"));
        assert!(!valid_record_path("src/../x"));
    }
}
