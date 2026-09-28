//! `ambits touched`: when did an agent last write this file or symbol, and is
//! that version still what is on disk? (spec §3.3)
//!
//! Reads the write records of every session's journal — not just the current
//! one, since "last touched" spans sessions — and parses only the one file in
//! question to answer "still current". Which git commit a write landed in
//! arrives with git linkage (spec §3.2, phase 4).

use std::io::Write as _;
use std::path::Path;

use color_eyre::eyre::Result;
use serde::Serialize;

use crate::cache::{journal_dir, session_ids};
use crate::journal::read_session_writes;
use crate::parser::ParserRegistry;
use crate::symbols::{nested_in, split_id};
pub use crate::writes::Status;
use crate::writes::{FileContents, Level, WriteRecord};

/// Bumped on any breaking change to the JSON shape.
pub const SCHEMA_VERSION: u32 = 1;

/// What `touched` was asked about.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Target {
    /// A project-relative, `/`-separated path.
    File(String),
    /// A symbol id, `<path>::<name-path>`.
    Symbol(String),
}

impl Target {
    /// A symbol id contains `::`; anything else is a path, taken as
    /// project-relative after trimming `./` and normalizing separators.
    pub fn parse(arg: &str) -> Self {
        let arg = crate::objects::normalize_path(arg.trim_start_matches("./"));
        if arg.contains("::") {
            Target::Symbol(arg)
        } else {
            Target::File(arg)
        }
    }

    /// The file this target lives in.
    fn file(&self) -> &str {
        match self {
            Target::File(f) => f,
            Target::Symbol(id) => split_id(id).0,
        }
    }

    /// Whether `write` changed this target; see [`WriteRecord::touches_symbol`].
    fn is_touched_by(&self, write: &WriteRecord) -> bool {
        match self {
            Target::File(f) => write.file == *f,
            Target::Symbol(id) => write.touches_symbol(id),
        }
    }
}

/// The latest write touching `target`, across every session: `(session, write)`.
///
/// Latest by [`WriteRecord::recency`].
pub fn latest(project_root: &Path, target: &Target) -> Option<(String, WriteRecord)> {
    let dir = journal_dir(project_root);
    session_ids(&dir)
        .into_iter()
        .flat_map(|session| {
            read_session_writes(&dir, &session)
                .into_values()
                .filter(|w| target.is_touched_by(w))
                .map(move |w| (session.clone(), w))
                .collect::<Vec<_>>()
        })
        .max_by(|(_, a), (_, b)| a.recency().cmp(&b.recency()))
}

/// Whether `write`'s version of `target` is still on disk.
pub fn status(project_root: &Path, registry: &ParserRegistry, target: &Target, write: &WriteRecord) -> Status {
    // Raw bytes, never through a symlink (§9.1): a file that is not UTF-8
    // is still there, and hashes the way the write's `fh` did.
    let Some(bytes) = crate::objects::read_regular(&project_root.join(target.file())) else {
        return Status::Removed;
    };
    let now = FileContents::read(target.file(), &bytes, registry);
    match target {
        Target::Symbol(id) => now.symbol_status(id, write),
        Target::File(_) => now.file_status(write),
    }
}

#[derive(Serialize)]
struct TouchedDto<'a> {
    schema_version: u32,
    target: &'a str,
    last_write: Option<WriteDto<'a>>,
}

#[derive(Serialize)]
struct WriteDto<'a> {
    session: &'a str,
    op: &'a str,
    agent: &'a str,
    time: &'a str,
    tool: &'a str,
    level: Level,
    status: Status,
    /// Which commit the write landed in (§3.2).
    landed: crate::linkage::Landed,
}

/// Print the answer for `arg`, as text or JSON.
pub fn run(project_root: &Path, arg: &str, json: bool) -> Result<()> {
    let target = Target::parse(arg);
    let found = latest(project_root, &target);
    let registry = ParserRegistry::new();
    let status = found.as_ref().map(|(_, w)| status(project_root, &registry, &target, w));
    // Only the parts of the write that are about the target: a symbol and
    // what is nested in it, or the whole file.
    let landed = match &found {
        Some((_, w)) => {
            let keep = |u: &crate::linkage::Unit| match &target {
                Target::Symbol(id) => nested_in(&u.target, id),
                Target::File(_) => true,
            };
            Some(crate::linkage::landed(crate::linkage::Resolver::new(project_root).as_mut(), w, &keep)?)
        }
        None => None,
    };
    let mut out = std::io::stdout().lock();

    if json {
        let dto = TouchedDto {
            schema_version: SCHEMA_VERSION,
            target: arg,
            last_write: found.as_ref().zip(status).zip(landed).map(|(((session, w), status), landed)| WriteDto {
                session,
                op: &w.op,
                agent: &w.a,
                time: &w.t,
                tool: &w.tool,
                level: w.level,
                status,
                landed,
            }),
        };
        writeln!(out, "{}", serde_json::to_string(&dto)?)?;
        return Ok(());
    }

    let Some(((session, w), status)) = found.as_ref().zip(status) else {
        writeln!(out, "{arg} — no agent writes recorded")?;
        return Ok(());
    };
    writeln!(out, "{arg} — last written {} by {} ({})", w.t, w.a, w.tool)?;
    writeln!(out, "  session {session}, write {}", w.op)?;
    let status_line = match status {
        Status::Current => "unchanged since the agent wrote it",
        Status::Changed => "changed since the agent wrote it",
        Status::Removed => "no longer exists",
        Status::Unknown => "unknown: a file-level write carries no hash to compare",
    };
    writeln!(out, "  {status_line}")?;
    if let Some(landed) = &landed {
        writeln!(out, "  {}", landed_line(landed))?;
    }
    Ok(())
}

fn landed_line(landed: &crate::linkage::Landed) -> String {
    use crate::linkage::Landed;
    let short = |commits: &[String]| commits.iter().map(|c| c.get(..7).unwrap_or(c)).collect::<Vec<_>>().join(", ");
    match landed {
        Landed::Verified { commits } => format!("landed in {} (verified)", short(commits)),
        Landed::Unverified { commits } => {
            format!("landed in {} (unverified: the first commit to touch the file after the write)", short(commits))
        }
        Landed::Partial { commits, unverified } => format!(
            // "In no commit" rather than "uncommitted": the rest may have
            // been changed again before committing, and so never land as
            // the agent wrote it.
            "partly landed in {}{}; the rest is in no commit",
            short(commits),
            if *unverified { " (unverified)" } else { "" }
        ),
        Landed::Uncommitted => "uncommitted".to_string(),
        Landed::NoRepository => "not in a git repository with commits".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::journal::encode_hash;

    const SRC: &str = "fn alpha() {}\nfn beta() {}\n";

    fn hash_of(src: &str, name: &str) -> String {
        FileContents::read("src/lib.rs", src.as_bytes(), &ParserRegistry::new()).hashes(name)[0].clone()
    }

    fn write(op: &str, t: &str, level: Level, syms: Vec<(&str, String)>, removed: Vec<&str>, fh: Option<String>) -> WriteRecord {
        WriteRecord {
            op: op.into(),
            av: crate::writes::ATTRIBUTION_VERSION,
            a: "agent-1".into(),
            t: t.into(),
            tool: "Edit".into(),
            file: "src/lib.rs".into(),
            level,
            syms: syms.into_iter().map(|(s, h)| (s.to_string(), h)).collect(),
            removed: removed.into_iter().map(String::from).collect(),
            fh,
            ..Default::default()
        }
    }

    /// A project with `src/lib.rs` = `SRC` and the given writes journaled
    /// under the given sessions.
    fn project(writes: &[(&str, WriteRecord)]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/lib.rs"), SRC).unwrap();
        let jdir = journal_dir(dir.path());
        std::fs::create_dir_all(&jdir).unwrap();
        for (session, w) in writes {
            let line = serde_json::to_string(&crate::journal::Record::Write(Box::new(w.clone()))).unwrap();
            let path = jdir.join(format!("{session}.ndjson"));
            let mut existing = std::fs::read_to_string(&path).unwrap_or_default();
            existing.push_str(&line);
            existing.push('\n');
            std::fs::write(path, existing).unwrap();
        }
        dir
    }

    fn check(dir: &tempfile::TempDir, arg: &str) -> Option<(String, String, Status)> {
        let target = Target::parse(arg);
        let (session, w) = latest(dir.path(), &target)?;
        let status = status(dir.path(), &ParserRegistry::new(), &target, &w);
        Some((session, w.op, status))
    }

    #[test]
    fn targets_parse_as_symbols_or_paths() {
        assert_eq!(Target::parse("src/a.rs::A/b"), Target::Symbol("src/a.rs::A/b".into()));
        assert_eq!(Target::parse("./src\\a.rs"), Target::File("src/a.rs".into()));
    }

    #[test]
    fn nothing_recorded_is_none() {
        let dir = project(&[]);
        assert_eq!(check(&dir, "src/lib.rs"), None);
    }

    /// "Last touched" spans sessions; the newest write wins.
    #[test]
    fn the_latest_write_across_sessions_wins() {
        let a = hash_of(SRC, "alpha");
        let dir = project(&[
            ("s-old", write("toolu_1", "2026-09-26T09:00:00Z", Level::Symbol, vec![("src/lib.rs::alpha", a.clone())], vec![], None)),
            ("s-new", write("toolu_2", "2026-09-26T11:00:00Z", Level::Symbol, vec![("src/lib.rs::alpha", a)], vec![], None)),
        ]);
        let (session, op, _) = check(&dir, "src/lib.rs::alpha").unwrap();
        assert_eq!((session.as_str(), op.as_str()), ("s-new", "toolu_2"));
    }

    #[test]
    fn a_symbol_is_current_until_it_changes() {
        let dir = project(&[("s1", write("toolu_1", "t", Level::Symbol, vec![("src/lib.rs::alpha", hash_of(SRC, "alpha"))], vec![], None))]);
        assert_eq!(check(&dir, "src/lib.rs::alpha").unwrap().2, Status::Current);

        std::fs::write(dir.path().join("src/lib.rs"), "fn alpha() { 1; }\nfn beta() {}\n").unwrap();
        assert_eq!(check(&dir, "src/lib.rs::alpha").unwrap().2, Status::Changed);

        std::fs::write(dir.path().join("src/lib.rs"), "fn beta() {}\n").unwrap();
        assert_eq!(check(&dir, "src/lib.rs::alpha").unwrap().2, Status::Removed);
    }

    #[test]
    fn a_symbol_the_write_removed_is_removed() {
        let dir = project(&[("s1", write("toolu_1", "t", Level::Symbol, vec![], vec!["src/lib.rs::gone"], None))]);
        assert_eq!(check(&dir, "src/lib.rs::gone").unwrap().2, Status::Removed);
    }

    /// `removed` means still absent. A symbol that came back after the agent
    /// deleted it has changed since.
    #[test]
    fn a_removed_symbol_that_came_back_has_changed() {
        let dir = project(&[("s1", write("toolu_1", "t", Level::Symbol, vec![], vec!["src/lib.rs::beta"], None))]);
        assert_eq!(check(&dir, "src/lib.rs::beta").unwrap().2, Status::Changed);
    }

    /// Writes name innermost symbols; asking about the enclosing one must
    /// still find them.
    #[test]
    fn a_parent_is_touched_through_its_children() {
        let src = "impl S {\n    fn a() {}\n    fn b() {}\n}\n";
        let a = {
            let file = ParserRegistry::new().parser_for(Path::new("src/lib.rs")).unwrap()
                .parse_file(Path::new("src/lib.rs"), src).unwrap();
            let child = &file.symbols[0].children[0];
            (child.id.clone(), encode_hash(&child.content_hash))
        };
        let dir = project(&[("s1", write("toolu_1", "t", Level::Symbol, vec![(a.0.as_str(), a.1.clone())], vec![], None))]);
        std::fs::write(dir.path().join("src/lib.rs"), src).unwrap();
        let parent = a.0.rsplit_once('/').unwrap().0.to_string();

        assert_eq!(check(&dir, &parent).unwrap().2, Status::Current);
        std::fs::write(dir.path().join("src/lib.rs"), src.replace("fn a() {}", "fn a() { 1; }")).unwrap();
        assert_eq!(check(&dir, &parent).unwrap().2, Status::Changed);
        assert_eq!(check(&dir, &format!("{parent}X")), None, "a longer name is not a child");
    }

    /// A symbol-level write names every symbol it changed, so one that
    /// omits the target did not touch it; a file-level write proves nothing.
    #[test]
    fn only_a_symbol_level_write_naming_it_touches_a_symbol() {
        let dir = project(&[
            ("s1", write("toolu_1", "2026-09-26T09:00:00Z", Level::Symbol, vec![("src/lib.rs::alpha", hash_of(SRC, "alpha"))], vec![], None)),
            ("s1", write("toolu_2", "2026-09-26T10:00:00Z", Level::Symbol, vec![("src/lib.rs::beta", hash_of(SRC, "beta"))], vec![], None)),
            ("s1", write("toolu_3", "2026-09-26T11:00:00Z", Level::File, vec![], vec![], None)),
        ]);
        assert_eq!(check(&dir, "src/lib.rs::alpha").unwrap().1, "toolu_1");
    }

    /// Equal timestamps resolve by `op`, the same way every time.
    #[test]
    fn a_timestamp_tie_resolves_by_op() {
        let a = hash_of(SRC, "alpha");
        let dir = project(&[
            ("s2", write("toolu_b", "2026-09-26T09:00:00Z", Level::Symbol, vec![("src/lib.rs::alpha", a.clone())], vec![], None)),
            ("s1", write("toolu_a", "2026-09-26T09:00:00Z", Level::Symbol, vec![("src/lib.rs::alpha", a)], vec![], None)),
        ]);
        assert_eq!(check(&dir, "src/lib.rs::alpha").unwrap().1, "toolu_b");
    }

    /// Presence is read from raw bytes: a file that is not UTF-8 is still
    /// there (it used to read as removed), and a symlink is never followed.
    #[test]
    fn a_non_utf8_file_is_present_and_a_symlink_is_not_followed() {
        let bytes = [0xffu8, 0xfe, b'\n'];
        let fh = crate::objects::file_hash(&bytes);
        let dir = project(&[("s1", write("toolu_1", "t", Level::File, vec![], vec![], Some(fh)))]);
        std::fs::write(dir.path().join("src/lib.rs"), bytes).unwrap();
        assert_eq!(check(&dir, "src/lib.rs").unwrap().2, Status::Current);

        #[cfg(unix)]
        {
            let real = dir.path().join("elsewhere.rs");
            std::fs::rename(dir.path().join("src/lib.rs"), &real).unwrap();
            std::os::unix::fs::symlink(&real, dir.path().join("src/lib.rs")).unwrap();
            assert_eq!(check(&dir, "src/lib.rs").unwrap().2, Status::Removed);
        }
    }

    /// A file-level `Write` is judged by its content hash; a file-level
    /// `Edit` has nothing to compare.
    #[test]
    fn file_status_uses_the_content_hash_or_is_unknown() {
        let fh = encode_hash(blake3::hash(SRC.as_bytes()).as_bytes());
        let dir = project(&[("s1", write("toolu_1", "t", Level::File, vec![], vec![], Some(fh)))]);
        assert_eq!(check(&dir, "src/lib.rs").unwrap().2, Status::Current);
        std::fs::write(dir.path().join("src/lib.rs"), "fn other() {}\n").unwrap();
        assert_eq!(check(&dir, "src/lib.rs").unwrap().2, Status::Changed);

        let dir = project(&[("s1", write("toolu_1", "t", Level::File, vec![], vec![], None))]);
        assert_eq!(check(&dir, "src/lib.rs").unwrap().2, Status::Unknown);
    }

    /// A symbol-level write answers for the file through its symbols.
    #[test]
    fn a_file_with_symbol_writes_is_current_while_they_all_are() {
        let dir = project(&[("s1", write("toolu_1", "t", Level::Symbol, vec![("src/lib.rs::beta", hash_of(SRC, "beta"))], vec![], None))]);
        assert_eq!(check(&dir, "src/lib.rs").unwrap().2, Status::Current);
        std::fs::write(dir.path().join("src/lib.rs"), "fn alpha() {}\nfn beta() { 2; }\n").unwrap();
        assert_eq!(check(&dir, "src/lib.rs").unwrap().2, Status::Changed);
    }
}
