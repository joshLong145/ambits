//! Durable record of which symbols were read, and what they looked like at
//! the time.
//!
//! ## Why this exists
//!
//! Claude Code's session logs already record *that* a file was read. What they
//! cannot record is what that file's symbols **looked like at the moment of the
//! read** — and without that, "is what the agent knows still accurate?" is
//! unanswerable. Replaying a historical read against a freshly scanned tree
//! stamps the *current* hash, so the comparison is vacuous (see the `KNOWN GAP`
//! note in `coverage::run_report`). This journal is precisely that missing
//! datum, and everything else it stores is in service of interpreting it.
//!
//! ## Shape
//!
//! Append-only NDJSON, one file per session. Line 1 is a header carrying the
//! environment the reads happened in; every later line records one symbol whose
//! read-state changed:
//!
//! ```text
//! {"kind":"header","schema_version":2,...}
//! {"kind":"read","sym":"src/app.rs::App/record","h":"b3:1a2b…","d":"full_body","a":"agent-3f9c"}
//! ```
//!
//! ## Written by diffing the ledger
//!
//! Rather than intercepting every read, [`Journal::sync`] walks
//! [`ContextLedger`] and appends only `(symbol, agent)` pairs whose
//! `(hash, depth)` differs from what has already been journaled. That yields
//! exactly one record per read-set change, with no plumbing through the
//! recursive symbol-marking hot path. It is sound only because
//! `ContextLedger::record` refreshes `content_hash_at_read` on *every* read —
//! see its docs; it did not always.
//!
//! ## Agent attribution and portability
//!
//! Everything that decides *whether a read is still valid* is
//! machine-independent by construction: symbol ids are project-relative and
//! hashes are content-derived, so a journal means the same thing on any host
//! with the same source. Agent ids are the exception — Claude Code mints them
//! per session — so they are recorded for fidelity (restoring agent-filtered
//! coverage and alignment) while the header carries the `host` they came from.
//! A consumer on another machine can therefore either use them as opaque
//! labels or remap them, and dropping them entirely still leaves a usable
//! symbol-level read set.
//!
//! Stale entries are skipped rather than rewritten, so a symbol that drifts
//! simply keeps its last-known-good hash on disk and fails the comparison at
//! restore time. That is the desired outcome: a stale entry is worse than no
//! entry, because it tells an agent it knows something it no longer knows.
//!
//! ## Durability
//!
//! Records are serialized whole and written with a single `write_all` under
//! `O_APPEND` — never through a `BufWriter`, which could split one logical
//! record across two syscalls and interleave it with another process's append.
//! Nothing is buffered in memory, so there is no flush to lose and no `Drop`
//! guard to write: a crash costs at most the reads since the last `sync`, which
//! only ever *under*-reports coverage. Under-reporting is the safe direction —
//! the agent re-reads something it already knew.
//!
//! Only the TUI writes. Readers (`restore-context`) never open the file for
//! writing, which removes concurrent-writer concerns rather than managing them.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::symbols::ProjectTree;
use crate::tracking::{ContextLedger, ReadDepth};

/// Journal format version. Bumped on any breaking change to the on-disk
/// shape; readers refuse newer files rather than misinterpret them. Mirrors
/// `ToolMappingConfig::SUPPORTED_VERSION`'s role for tool configs.
///
/// - **v1** — one record per symbol, no agent attribution.
/// - **v2** — one record per `(symbol, agent)`, plus `host` in the manifest.
///
/// Older versions stay readable: `a` is optional and the manifest fields are
/// `#[serde(default)]`, so a v1 file folds into the same symbol-level view and
/// only loses per-agent detail. See [`MIN_READABLE_SCHEMA_VERSION`].
pub const SUPPORTED_SCHEMA_VERSION: u32 = 2;

/// Oldest on-disk version this build can still fold. Reading an older journal
/// is always preferable to discarding it — the symbol-level read set, which is
/// what `restore-context` needs, is present in every version.
pub const MIN_READABLE_SCHEMA_VERSION: u32 = 1;

/// Directory, relative to the project root, holding per-session journals.
pub const JOURNAL_SUBDIR: &str = ".ambit/coverage";

/// Default interval between ledger diffs.
pub const DEFAULT_FLUSH_INTERVAL_MS: u64 = 5_000;

// ---------------------------------------------------------------------------
// Wire types
//
// Deliberately separate from `symbols::*` / `tracking::*`, which stay
// serde-free (see the `ReadDepthDe` mirror in ingest::tool_config). Keeping
// the format decoupled also means we never have to serialize `SymbolNode`,
// whose `&'static str` label and `Arc` fields do not round-trip.
// ---------------------------------------------------------------------------

/// String mirror of [`ReadDepth`] so the core enum stays free of serde.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DepthDto {
    Unseen,
    NameOnly,
    Overview,
    Signature,
    FullBody,
}

impl From<ReadDepth> for DepthDto {
    fn from(d: ReadDepth) -> Self {
        match d {
            ReadDepth::Unseen => DepthDto::Unseen,
            ReadDepth::NameOnly => DepthDto::NameOnly,
            ReadDepth::Overview => DepthDto::Overview,
            ReadDepth::Signature => DepthDto::Signature,
            ReadDepth::FullBody => DepthDto::FullBody,
        }
    }
}

impl From<DepthDto> for ReadDepth {
    fn from(d: DepthDto) -> Self {
        match d {
            DepthDto::Unseen => ReadDepth::Unseen,
            DepthDto::NameOnly => ReadDepth::NameOnly,
            DepthDto::Overview => ReadDepth::Overview,
            DepthDto::Signature => ReadDepth::Signature,
            DepthDto::FullBody => ReadDepth::FullBody,
        }
    }
}

/// The environment a session's reads happened in.
///
/// Only `tree_fingerprint` is load-bearing for validation — it already folds in
/// everything the parsers affect, because a grammar change alters the symbol
/// hashes it is built from. The rest is diagnostic: it answers "why did my
/// cache stop matching?" rather than deciding whether it matches.
///
/// `project_root` is recorded as provenance only. It is the one host-specific
/// value here, and a restore on another machine overwrites it rather than
/// trusting it — everything else (symbol ids, relative paths, BLAKE3 hashes) is
/// already machine-independent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentManifest {
    pub project_root: String,
    pub tree_fingerprint: String,
    pub ambit_version: String,
    pub backend: String,
    #[serde(default)]
    pub parsers: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_config_version: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<String>,
    pub os: String,
    pub arch: String,
    /// Machine the reads happened on.
    ///
    /// Nothing in restore consults this — the read set itself is entirely
    /// host-independent (project-relative symbol ids, content hashes). It is
    /// here so that agent ids, which *are* machine-scoped, can be traced back
    /// to where they were minted when a journal is carried to another host.
    /// Absent in v1 journals, hence `default`.
    #[serde(default)]
    pub host: String,
}

impl EnvironmentManifest {
    /// Capture the current environment for `tree`.
    pub fn capture(tree: &ProjectTree, backend: &str, filter: Option<String>) -> Self {
        Self {
            project_root: tree.root.to_string_lossy().into_owned(),
            tree_fingerprint: encode_hash(&tree_fingerprint(tree)),
            ambit_version: env!("CARGO_PKG_VERSION").to_string(),
            backend: backend.to_string(),
            // Diagnostic only; keep in step with Cargo.toml when grammars move.
            parsers: vec![
                "tree-sitter=0.24".into(),
                "rust=0.23".into(),
                "python=0.23".into(),
                "typescript=0.23".into(),
            ],
            tool_config_version: Some(crate::ingest::tool_config::ToolMappingConfig::SUPPORTED_VERSION),
            filter,
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            host: hostname(),
        }
    }
}

/// Best-effort machine name, without taking on a dependency for it.
///
/// Diagnostic only (see [`EnvironmentManifest::host`]), so every source is
/// allowed to fail and `"unknown"` is an acceptable answer. Runs once per
/// session, at journal open.
fn hostname() -> String {
    let from_env = ["HOSTNAME", "HOST", "COMPUTERNAME"]
        .iter()
        .find_map(|k| std::env::var(k).ok())
        .filter(|h| !h.is_empty());
    if let Some(h) = from_env {
        return h;
    }

    if let Some(h) = std::fs::read_to_string("/etc/hostname")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
    {
        return h;
    }

    std::process::Command::new("hostname")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

/// The journal's first line: what this session's reads mean.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeaderRecord {
    pub schema_version: u32,
    pub created_at: String,
    pub session_id: String,
    #[serde(flatten)]
    pub environment: EnvironmentManifest,
}

/// One line of the journal.
///
/// `Header` is boxed because it dwarfs `Read`, and `Read` is the variant we
/// construct once per symbol.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Record {
    Header(Box<HeaderRecord>),
    /// One agent's read-state for one symbol changed: it was read for the
    /// first time, re-read deeper, or re-read after its content changed.
    Read {
        #[serde(rename = "sym")]
        symbol_id: String,
        #[serde(rename = "h")]
        hash: String,
        #[serde(rename = "d")]
        depth: DepthDto,
        /// Which agent read it.
        ///
        /// Machine-scoped (Claude Code mints these per session), which is why
        /// the manifest records a `host` to trace them back. Absent in v1
        /// journals and in any record we cannot attribute, in which case the
        /// read still counts — it just cannot be filtered by agent.
        #[serde(rename = "a", default, skip_serializing_if = "Option::is_none")]
        agent: Option<String>,
    },
}

// ---------------------------------------------------------------------------
// Hashing / hex
// ---------------------------------------------------------------------------

/// Fold the project tree into a single fingerprint.
///
/// Built from the per-symbol merkle hashes, so it changes whenever the parsed
/// shape of the project changes — including when a grammar upgrade shifts node
/// boundaries. `ProjectTree.files` is already sorted by path, so this is
/// deterministic. Paths are project-relative, so it is machine-independent.
pub fn tree_fingerprint(tree: &ProjectTree) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    for file in &tree.files {
        hasher.update(file.file_path.to_string_lossy().as_bytes());
        hasher.update(b"\0");
        for sym in &file.symbols {
            hasher.update(&sym.merkle_hash);
        }
        hasher.update(b"\n");
    }
    hasher.finalize().into()
}

/// Render a hash as `b3:<hex>`. The prefix names the algorithm so a future
/// change is detectable rather than silently misread.
pub fn encode_hash(hash: &[u8; 32]) -> String {
    let mut s = String::with_capacity(35 + 64);
    s.push_str("b3:");
    for byte in hash {
        s.push_str(&format!("{byte:02x}"));
    }
    s
}

/// Parse a `b3:<hex>` hash. Returns `None` on any malformed input.
pub fn decode_hash(s: &str) -> Option<[u8; 32]> {
    let hex = s.strip_prefix("b3:")?;
    if hex.len() != 64 {
        return None;
    }
    let bytes = hex.as_bytes();
    let mut out = [0u8; 32];
    for (i, slot) in out.iter_mut().enumerate() {
        let hi = (bytes[i * 2] as char).to_digit(16)?;
        let lo = (bytes[i * 2 + 1] as char).to_digit(16)?;
        *slot = (hi * 16 + lo) as u8;
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

/// A journaled read, keyed by the agent that performed it.
pub type AgentReadKey = (String, String);

/// Everything recovered from a journal file, plus any non-fatal complaints.
#[derive(Debug, Default)]
pub struct JournalContents {
    pub header: Option<(u32, EnvironmentManifest)>,
    /// Symbol-level view: what was read, and what it looked like.
    ///
    /// Folded to mirror `ContextLedger`'s own semantics rather than by naive
    /// last-write-wins. A record carrying a *new* hash supersedes what came
    /// before (the content moved on, so earlier depths describe a version that
    /// no longer exists); a record carrying the *same* hash contributes its
    /// depth to the maximum, which is how the ledger aggregates across agents.
    pub reads: HashMap<String, ([u8; 32], ReadDepth)>,
    /// Per-`(symbol, agent)` view, for restoring agent-filtered coverage.
    /// Empty for v1 journals, which carried no attribution.
    pub agent_reads: HashMap<AgentReadKey, ([u8; 32], ReadDepth)>,
    pub warnings: Vec<String>,
}

/// Read and fold a journal file.
///
/// Corruption is never fatal: a partial *final* line is dropped silently (the
/// writer was interrupted mid-append, which is expected), while a bad
/// *interior* line warns and is skipped. Losing one record costs one symbol's
/// coverage; refusing to read the file costs all of it.
pub fn read_journal(path: &Path) -> JournalContents {
    let mut out = JournalContents::default();

    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return out,
        Err(e) => {
            out.warnings.push(format!("{}: {e}", path.display()));
            return out;
        }
    };

    let lines: Vec<&str> = content.lines().collect();
    let last_index = lines.len().saturating_sub(1);
    // A file not ending in a newline means the final line may be a torn write.
    let final_line_may_be_torn = !content.ends_with('\n');

    for (i, line) in lines.iter().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let record: Record = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(e) => {
                if !(i == last_index && final_line_may_be_torn) {
                    out.warnings
                        .push(format!("{}:{}: {e}", path.display(), i + 1));
                }
                continue;
            }
        };

        match record {
            Record::Header(header) => {
                if header.schema_version > SUPPORTED_SCHEMA_VERSION
                    || header.schema_version < MIN_READABLE_SCHEMA_VERSION
                {
                    out.warnings.push(format!(
                        "{}: schema version {} is outside the readable range {}..={}; ignoring journal",
                        path.display(),
                        header.schema_version,
                        MIN_READABLE_SCHEMA_VERSION,
                        SUPPORTED_SCHEMA_VERSION
                    ));
                    return JournalContents {
                        warnings: out.warnings,
                        ..Default::default()
                    };
                }
                out.header = Some((header.schema_version, header.environment));
            }
            Record::Read {
                symbol_id,
                hash,
                depth,
                agent,
            } => match decode_hash(&hash) {
                Some(h) => {
                    let depth: ReadDepth = depth.into();
                    out.reads
                        .entry(symbol_id.clone())
                        .and_modify(|slot| {
                            if slot.0 == h {
                                slot.1 = slot.1.max(depth);
                            } else {
                                *slot = (h, depth);
                            }
                        })
                        .or_insert((h, depth));
                    if let Some(agent) = agent {
                        out.agent_reads
                            .entry((symbol_id, agent))
                            .and_modify(|slot| {
                                if slot.0 == h {
                                    slot.1 = slot.1.max(depth);
                                } else {
                                    *slot = (h, depth);
                                }
                            })
                            .or_insert((h, depth));
                    }
                }
                None => out.warnings.push(format!(
                    "{}:{}: malformed hash {hash:?}",
                    path.display(),
                    i + 1
                )),
            },
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/// Append-only writer that keeps a session's journal in step with the ledger.
#[derive(Debug)]
pub struct Journal {
    path: PathBuf,
    file: Option<File>,
    /// What is already on disk, so a diff can skip unchanged entries.
    ///
    /// Keyed by `(symbol, agent)` since v2: two agents reading the same symbol
    /// are two facts, and collapsing them would let the first read suppress
    /// the second.
    journaled: HashMap<AgentReadKey, ([u8; 32], ReadDepth)>,
    /// Distinct symbols on disk, for [`Journal::len`]. Tracked separately so
    /// the count stays a symbol count rather than a record count.
    symbols: std::collections::HashSet<String>,
    interval: Duration,
    last_sync: Instant,
    /// Set when writing fails. Journaling then stops for the rest of the run
    /// rather than erroring on every tick — a broken cache must never take the
    /// TUI down with it.
    error: Option<String>,
    warnings: Vec<String>,
}

impl Journal {
    /// Open (or create) the journal for `session_id` under `project_root`.
    ///
    /// An existing file is read first so its contents seed the dedup map. That
    /// makes relaunching ambit idempotent: startup replays the whole session
    /// log through the ledger, and without this every launch would re-append
    /// the entire history.
    pub fn open(
        project_root: &Path,
        session_id: &str,
        manifest: EnvironmentManifest,
        interval: Duration,
    ) -> Self {
        let dir = project_root.join(JOURNAL_SUBDIR);
        let path = dir.join(format!("{session_id}.ndjson"));

        let mut journal = Self {
            path: path.clone(),
            file: None,
            journaled: HashMap::new(),
            symbols: std::collections::HashSet::new(),
            interval,
            last_sync: Instant::now(),
            error: None,
            warnings: Vec::new(),
        };

        if let Err(e) = std::fs::create_dir_all(&dir) {
            journal.error = Some(format!("{}: {e}", dir.display()));
            return journal;
        }

        let existing = read_journal(&path);
        journal.warnings.extend(existing.warnings);
        let had_header = existing.header.is_some();
        journal.symbols = existing.reads.keys().cloned().collect();
        // Seeded from the attributed view only. A v1 journal has none, so its
        // reads are re-appended once, in v2 form, on the first sync — a
        // one-time cost that upgrades the file in place rather than stranding
        // its history behind a version gate.
        journal.journaled = existing.agent_reads;

        let file = match OpenOptions::new().create(true).append(true).open(&path) {
            Ok(f) => f,
            Err(e) => {
                journal.error = Some(format!("{}: {e}", path.display()));
                return journal;
            }
        };
        journal.file = Some(file);

        // Append a fresh header when the file has none, or when its header
        // declares an older format than we are about to write. The file is
        // append-only so the old line stays, but readers take the last header
        // they see — which keeps the declared version honest once `sync` has
        // upgraded the records below it. Without this, an upgraded file would
        // still claim v1 and be discarded the day v1 leaves the readable
        // range, despite containing perfectly current records.
        let stale_header = existing
            .header
            .as_ref()
            .is_some_and(|(v, _)| *v < SUPPORTED_SCHEMA_VERSION);
        if !had_header || stale_header {
            let header = Record::Header(Box::new(HeaderRecord {
                schema_version: SUPPORTED_SCHEMA_VERSION,
                created_at: timestamp(),
                session_id: session_id.to_string(),
                environment: manifest,
            }));
            journal.write(&header);
        }

        journal
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Non-fatal complaints accumulated so far (corrupt lines, write failures).
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// Number of distinct symbols currently recorded on disk.
    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }

    /// Diff against the ledger if the flush interval has elapsed.
    pub fn maybe_sync(&mut self, ledger: &ContextLedger) -> usize {
        if self.last_sync.elapsed() < self.interval {
            return 0;
        }
        self.sync(ledger)
    }

    /// Append a record for every ledger entry whose read-state has changed
    /// since it was last journaled. Returns how many were written.
    ///
    /// Skips unseen and stale entries: we only ever journal reads we currently
    /// believe describe the file as it is on disk.
    ///
    /// One record per `(symbol, agent)` whose depth or hash moved. The content
    /// hash is a property of the symbol, not of the reader, so every agent's
    /// record for a symbol carries the same hash — what differs is how deeply
    /// each one read it.
    pub fn sync(&mut self, ledger: &ContextLedger) -> usize {
        self.last_sync = Instant::now();
        if self.error.is_some() {
            return 0;
        }

        let mut pending: Vec<(String, String, [u8; 32], ReadDepth)> = Vec::new();
        for (id, entry) in &ledger.entries {
            if !entry.depth.is_seen() || entry.stale {
                continue;
            }
            for (agent, depth) in &entry.agent_depths {
                if !depth.is_seen() {
                    continue;
                }
                let current = (entry.content_hash_at_read, *depth);
                let key = (id.clone(), agent.clone());
                if self.journaled.get(&key) != Some(&current) {
                    pending.push((id.clone(), agent.clone(), current.0, current.1));
                }
            }
        }

        // Deterministic order keeps golden-file tests stable; HashMap
        // iteration order is not.
        pending.sort_by(|a, b| (&a.0, &a.1).cmp(&(&b.0, &b.1)));

        let mut written = 0;
        for (id, agent, hash, depth) in pending {
            let record = Record::Read {
                symbol_id: id.clone(),
                hash: encode_hash(&hash),
                depth: depth.into(),
                agent: Some(agent.clone()),
            };
            if !self.write(&record) {
                break;
            }
            self.journaled.insert((id.clone(), agent), (hash, depth));
            self.symbols.insert(id);
            written += 1;
        }
        written
    }

    /// Serialize and append one record. Returns `false` once journaling has
    /// been disabled by an error.
    fn write(&mut self, record: &Record) -> bool {
        let Some(file) = self.file.as_mut() else {
            return false;
        };
        let mut buf = match serde_json::to_vec(record) {
            Ok(b) => b,
            Err(e) => {
                self.error = Some(e.to_string());
                return false;
            }
        };
        buf.push(b'\n');
        // One write_all per record: a BufWriter could split a record across
        // syscalls and interleave it with another process's append.
        if let Err(e) = file.write_all(&buf) {
            let msg = format!("{}: {e}", self.path.display());
            self.warnings.push(msg.clone());
            self.error = Some(msg);
            self.file = None;
            return false;
        }
        true
    }
}

/// Best-effort ISO-8601-ish UTC timestamp.
///
/// `ContextEntry.timestamp` is a `std::time::Instant`, which is
/// monotonic-clock-relative and cannot be serialized or compared across
/// processes, so journal timestamps come from the wall clock instead. They are
/// informational only — nothing in restore depends on them.
fn timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{secs}")
}

#[cfg(test)]
#[path = "../tests/helpers/mod.rs"]
#[allow(dead_code)]
mod helpers;

#[cfg(test)]
mod tests {
    use super::helpers::*;
    use super::*;
    use crate::symbols::merkle::content_hash;

    fn manifest() -> EnvironmentManifest {
        EnvironmentManifest {
            project_root: "/p".into(),
            tree_fingerprint: encode_hash(&[7u8; 32]),
            ambit_version: "test".into(),
            backend: "tree-sitter".into(),
            parsers: vec![],
            tool_config_version: Some(1),
            filter: None,
            os: "testos".into(),
            arch: "testarch".into(),
            host: "testhost".into(),
        }
    }

    fn journal_in(dir: &Path) -> Journal {
        Journal::open(dir, "sess", manifest(), Duration::from_millis(0))
    }

    #[test]
    fn hash_round_trips_through_hex() {
        let h = content_hash("fn main() {}");
        assert_eq!(decode_hash(&encode_hash(&h)), Some(h));
    }

    #[test]
    fn decode_rejects_malformed_hashes() {
        assert_eq!(decode_hash("deadbeef"), None, "missing algorithm prefix");
        assert_eq!(decode_hash("b3:xyz"), None, "wrong length");
        assert_eq!(decode_hash(&format!("b3:{}", "z".repeat(64))), None, "non-hex");
    }

    #[test]
    fn tree_fingerprint_changes_with_content() {
        let a = project(vec![file("a.rs", vec![sym("a.rs::x", "x")])]);
        let mut b = project(vec![file("a.rs", vec![sym("a.rs::x", "x")])]);
        b.files[0].symbols[0].merkle_hash = [9u8; 32];
        assert_ne!(tree_fingerprint(&a), tree_fingerprint(&b));
    }

    #[test]
    fn writes_header_once_and_appends_reads() {
        let dir = tempfile::tempdir().unwrap();
        let mut ledger = ContextLedger::new();
        let h = content_hash("body");
        ledger.record("a.rs::x".into(), ReadDepth::FullBody, h, "ag".into(), 5);

        let mut j = journal_in(dir.path());
        assert_eq!(j.sync(&ledger), 1);

        let raw = std::fs::read_to_string(j.path()).unwrap();
        let lines: Vec<&str> = raw.lines().collect();
        assert_eq!(lines.len(), 2, "header + one read");
        assert!(lines[0].contains("\"kind\":\"header\""));
        assert!(lines[1].contains("a.rs::x"));
        assert!(lines[1].contains("full_body"));
    }

    /// Two agents reading the same symbol are two facts. Collapsing them
    /// would let whichever synced first suppress the other, and agent-filtered
    /// coverage would come back wrong.
    #[test]
    fn each_agent_gets_its_own_record() {
        let dir = tempfile::tempdir().unwrap();
        let mut ledger = ContextLedger::new();
        let h = content_hash("body");
        ledger.record("a.rs::x".into(), ReadDepth::FullBody, h, "parent".into(), 5);
        ledger.record("a.rs::x".into(), ReadDepth::Signature, h, "child".into(), 5);

        let mut j = journal_in(dir.path());
        assert_eq!(j.sync(&ledger), 2, "one record per agent");
        assert_eq!(j.len(), 1, "but still one distinct symbol");

        let contents = read_journal(j.path());
        assert_eq!(
            contents.agent_reads[&("a.rs::x".to_string(), "parent".to_string())].1,
            ReadDepth::FullBody
        );
        assert_eq!(
            contents.agent_reads[&("a.rs::x".to_string(), "child".to_string())].1,
            ReadDepth::Signature
        );
        assert_eq!(
            contents.reads["a.rs::x"].1,
            ReadDepth::FullBody,
            "the symbol-level fold takes the deepest read, as the ledger does"
        );
    }

    /// A v1 journal predates attribution. It must still restore — losing the
    /// agent breakdown is a far smaller loss than discarding the read set.
    #[test]
    fn v1_journals_without_agents_still_fold() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("j.ndjson");
        let header = r#"{"kind":"header","schema_version":1,"created_at":"0","session_id":"s","project_root":"/p","tree_fingerprint":"b3:00","ambit_version":"0.17.0","backend":"tree-sitter","os":"linux","arch":"x86_64"}"#;
        let read = format!(
            r#"{{"kind":"read","sym":"a.rs::x","h":"{}","d":"full_body"}}"#,
            encode_hash(&[1u8; 32])
        );
        std::fs::write(&path, format!("{header}\n{read}\n")).unwrap();

        let contents = read_journal(&path);
        assert_eq!(contents.header.unwrap().0, 1);
        assert_eq!(contents.reads["a.rs::x"].1, ReadDepth::FullBody);
        assert!(
            contents.agent_reads.is_empty(),
            "no attribution existed to recover"
        );
        assert!(contents.warnings.is_empty(), "an old version is not a fault");
    }

    /// A record carrying a new hash supersedes earlier depths rather than
    /// maxing with them: those depths describe a version of the symbol that no
    /// longer exists.
    #[test]
    fn a_new_hash_supersedes_rather_than_accumulates() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("j.ndjson");
        let deep = format!(
            r#"{{"kind":"read","sym":"a.rs::x","h":"{}","d":"full_body","a":"ag"}}"#,
            encode_hash(&[1u8; 32])
        );
        let shallow_but_newer = format!(
            r#"{{"kind":"read","sym":"a.rs::x","h":"{}","d":"signature","a":"ag"}}"#,
            encode_hash(&[2u8; 32])
        );
        std::fs::write(&path, format!("{deep}\n{shallow_but_newer}\n")).unwrap();

        let contents = read_journal(&path);
        assert_eq!(contents.reads["a.rs::x"], ([2u8; 32], ReadDepth::Signature));
    }

    /// Opening a v1 journal with a v2 writer: the existing history stays
    /// readable and countable, no second header is written, and the reads are
    /// re-appended once in attributed form. That one-time growth is the cost
    /// of upgrading the file in place instead of stranding it.
    #[test]
    fn a_v1_journal_is_upgraded_in_place_on_first_sync() {
        let dir = tempfile::tempdir().unwrap();
        let coverage = dir.path().join(JOURNAL_SUBDIR);
        std::fs::create_dir_all(&coverage).unwrap();
        let path = coverage.join("sess.ndjson");

        let h = content_hash("body");
        let header = r#"{"kind":"header","schema_version":1,"created_at":"0","session_id":"sess","project_root":"/p","tree_fingerprint":"b3:00","ambit_version":"0.16.0","backend":"tree-sitter","os":"linux","arch":"x86_64"}"#;
        let read = format!(
            r#"{{"kind":"read","sym":"a.rs::x","h":"{}","d":"full_body"}}"#,
            encode_hash(&h)
        );
        std::fs::write(&path, format!("{header}\n{read}\n")).unwrap();

        let mut ledger = ContextLedger::new();
        ledger.record("a.rs::x".into(), ReadDepth::FullBody, h, "ag".into(), 5);

        let mut j = journal_in(dir.path());
        assert_eq!(j.len(), 1, "the v1 history is visible immediately");
        assert_eq!(j.sync(&ledger), 1, "re-appended once, now attributed");
        assert_eq!(j.sync(&ledger), 0, "and not again");

        let contents = read_journal(&path);
        assert_eq!(
            contents.header.unwrap().0,
            SUPPORTED_SCHEMA_VERSION,
            "the upgraded file declares the version it now actually holds"
        );
        assert_eq!(
            contents.agent_reads[&("a.rs::x".to_string(), "ag".to_string())].1,
            ReadDepth::FullBody,
            "attribution is now recoverable"
        );
    }

    #[test]
    fn unchanged_entries_are_not_rewritten() {
        let dir = tempfile::tempdir().unwrap();
        let mut ledger = ContextLedger::new();
        let h = content_hash("body");
        ledger.record("a.rs::x".into(), ReadDepth::FullBody, h, "ag".into(), 5);

        let mut j = journal_in(dir.path());
        assert_eq!(j.sync(&ledger), 1);
        assert_eq!(j.sync(&ledger), 0, "second sync writes nothing");

        // Re-reading at the same depth and hash is still not a change.
        ledger.record("a.rs::x".into(), ReadDepth::FullBody, h, "ag".into(), 5);
        assert_eq!(j.sync(&ledger), 0);
    }

    #[test]
    fn re_read_after_drift_appends_the_new_hash() {
        let dir = tempfile::tempdir().unwrap();
        let mut ledger = ContextLedger::new();
        let h1 = content_hash("v1");
        let h2 = content_hash("v2");
        ledger.record("a.rs::x".into(), ReadDepth::FullBody, h1, "ag".into(), 5);

        let mut j = journal_in(dir.path());
        assert_eq!(j.sync(&ledger), 1);

        ledger.record("a.rs::x".into(), ReadDepth::FullBody, h2, "ag".into(), 5);
        assert_eq!(j.sync(&ledger), 1, "changed hash is a read-set change");

        // Last write wins on read-back.
        let contents = read_journal(j.path());
        assert_eq!(contents.reads["a.rs::x"], (h2, ReadDepth::FullBody));
    }

    #[test]
    fn stale_and_unseen_entries_are_not_journaled() {
        let dir = tempfile::tempdir().unwrap();
        let mut ledger = ContextLedger::new();
        let h1 = content_hash("v1");
        ledger.record("a.rs::x".into(), ReadDepth::FullBody, h1, "ag".into(), 5);
        ledger.mark_stale_if_changed("a.rs::x", content_hash("v2"));

        let mut j = journal_in(dir.path());
        assert_eq!(j.sync(&ledger), 0, "stale reads are withheld");
        assert!(read_journal(j.path()).reads.is_empty());
    }

    /// Re-opening must not duplicate what is already on disk — startup replays
    /// the whole session log, so without dedup every launch would double the
    /// file.
    #[test]
    fn reopening_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let mut ledger = ContextLedger::new();
        ledger.record("a.rs::x".into(), ReadDepth::FullBody, content_hash("b"), "ag".into(), 5);

        let mut j = journal_in(dir.path());
        j.sync(&ledger);
        let after_first = std::fs::read_to_string(j.path()).unwrap();
        drop(j);

        let mut j2 = journal_in(dir.path());
        assert_eq!(j2.len(), 1, "existing records seed the dedup map");
        assert_eq!(j2.sync(&ledger), 0);
        assert_eq!(std::fs::read_to_string(j2.path()).unwrap(), after_first);
    }

    #[test]
    fn torn_final_line_is_dropped_silently() {
        let dir = tempfile::tempdir().unwrap();
        let mut ledger = ContextLedger::new();
        ledger.record("a.rs::x".into(), ReadDepth::FullBody, content_hash("b"), "ag".into(), 5);
        let mut j = journal_in(dir.path());
        j.sync(&ledger);
        let path = j.path().to_path_buf();
        drop(j);

        // Simulate a write interrupted mid-record.
        let mut raw = std::fs::read_to_string(&path).unwrap();
        raw.push_str("{\"kind\":\"read\",\"sym\":\"a.rs::y\",\"h\":\"b3:00");
        std::fs::write(&path, raw).unwrap();

        let contents = read_journal(&path);
        assert_eq!(contents.reads.len(), 1);
        assert!(contents.warnings.is_empty(), "a torn tail is expected, not an error");
    }

    #[test]
    fn corrupt_interior_line_warns_but_keeps_the_rest() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("j.ndjson");
        let good = serde_json::to_string(&Record::Read {
            symbol_id: "a.rs::x".into(),
            hash: encode_hash(&[1u8; 32]),
            depth: DepthDto::FullBody,
            agent: Some("agent-1".into()),
        })
        .unwrap();
        std::fs::write(&path, format!("not json\n{good}\n")).unwrap();

        let contents = read_journal(&path);
        assert_eq!(contents.reads.len(), 1, "the good record survives");
        assert_eq!(contents.warnings.len(), 1, "the bad one is reported");
    }

    #[test]
    fn newer_schema_version_is_refused_rather_than_misread() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("j.ndjson");
        let header = serde_json::to_string(&Record::Header(Box::new(HeaderRecord {
            schema_version: SUPPORTED_SCHEMA_VERSION + 1,
            created_at: "0".into(),
            session_id: "s".into(),
            environment: manifest(),
        })))
        .unwrap();
        let read = serde_json::to_string(&Record::Read {
            symbol_id: "a.rs::x".into(),
            hash: encode_hash(&[1u8; 32]),
            depth: DepthDto::FullBody,
            agent: Some("agent-1".into()),
        })
        .unwrap();
        std::fs::write(&path, format!("{header}\n{read}\n")).unwrap();

        let contents = read_journal(&path);
        assert!(contents.reads.is_empty());
        assert!(contents.header.is_none());
        assert_eq!(contents.warnings.len(), 1);
    }

    #[test]
    fn missing_file_reads_as_empty() {
        let contents = read_journal(Path::new("/nonexistent/nope.ndjson"));
        assert!(contents.reads.is_empty());
        assert!(contents.warnings.is_empty());
    }

    #[test]
    fn interval_gates_syncing() {
        let dir = tempfile::tempdir().unwrap();
        let mut ledger = ContextLedger::new();
        ledger.record("a.rs::x".into(), ReadDepth::FullBody, content_hash("b"), "ag".into(), 5);

        let mut j = Journal::open(dir.path(), "sess", manifest(), Duration::from_secs(3600));
        assert_eq!(j.maybe_sync(&ledger), 0, "interval has not elapsed");
        assert_eq!(j.sync(&ledger), 1, "explicit sync ignores the interval");
    }
}
