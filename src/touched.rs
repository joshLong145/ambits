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
use crate::journal::{encode_hash, read_journal_session};
use crate::parser::ParserRegistry;
use crate::symbols::SymbolNode;
use crate::writes::{Level, WriteRecord};

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
        if arg.contains("::") {
            Target::Symbol(arg.to_string())
        } else {
            Target::File(arg.trim_start_matches("./").replace('\\', "/"))
        }
    }

    /// The file this target lives in.
    fn file(&self) -> &str {
        match self {
            Target::File(f) => f,
            Target::Symbol(id) => id.split("::").next().unwrap_or(id),
        }
    }

    /// Whether `write` changed this target. A symbol is touched through
    /// itself or any descendant: writes record innermost symbols only (D11),
    /// so an edit inside `App/handle_key` is an edit to `App`. Only
    /// symbol-level writes count for a symbol — a file-level write cannot say
    /// which symbols it changed, and a symbol-level one names every symbol it
    /// did (spec §2.3).
    fn is_touched_by(&self, write: &WriteRecord) -> bool {
        match self {
            Target::File(f) => write.file == *f,
            Target::Symbol(id) => {
                write.syms.iter().any(|(s, _)| within(s, id)) || write.removed.iter().any(|s| within(s, id))
            }
        }
    }
}

/// `symbol` is `id` or nested inside it.
fn within(symbol: &str, id: &str) -> bool {
    symbol.strip_prefix(id).is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
}

/// Whether the agent's version is still what is on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    /// Unchanged since the agent wrote it.
    Current,
    /// Changed since.
    Changed,
    /// No longer exists.
    Removed,
    /// A file-level write with no hash to compare (spec §3.1).
    Unknown,
}

/// The latest write touching `target`, across every session: `(session, write)`.
///
/// Ordered by timestamp (RFC 3339 UTC from the log, so lexicographic order is
/// chronological; a record with no timestamp sorts oldest), then by `op` so
/// ties resolve the same way every time.
pub fn latest(project_root: &Path, target: &Target) -> Option<(String, WriteRecord)> {
    let dir = journal_dir(project_root);
    session_ids(&dir)
        .into_iter()
        .flat_map(|session| {
            read_journal_session(&dir, &session)
                .writes
                .into_values()
                .filter(|w| target.is_touched_by(w))
                .map(move |w| (session.clone(), w))
                .collect::<Vec<_>>()
        })
        .max_by(|(_, a), (_, b)| (&a.t, &a.op).cmp(&(&b.t, &b.op)))
}

/// Whether `write`'s version of `target` is still on disk.
pub fn status(project_root: &Path, registry: &ParserRegistry, target: &Target, write: &WriteRecord) -> Status {
    let path = project_root.join(target.file());
    let Ok(source) = std::fs::read_to_string(&path) else {
        return Status::Removed;
    };

    if let Target::File(_) = target {
        if let Some(fh) = &write.fh {
            let now = encode_hash(blake3::hash(source.as_bytes()).as_bytes());
            return if &now == fh { Status::Current } else { Status::Changed };
        }
        if write.level == Level::File || write.syms.is_empty() {
            return Status::Unknown;
        }
    }

    let Some(symbols) = registry
        .parser_for(Path::new(target.file()))
        .and_then(|p| p.parse_file(Path::new(target.file()), &source).ok())
    else {
        return Status::Unknown;
    };
    // Current hashes of every symbol whose id passes `matches`.
    let hashes = |matches: &dyn Fn(&str) -> bool| -> Vec<String> {
        let mut out = Vec::new();
        let mut stack: Vec<&SymbolNode> = symbols.symbols.iter().collect();
        while let Some(s) = stack.pop() {
            if matches(&s.id) {
                out.push(encode_hash(&s.content_hash));
            }
            stack.extend(s.children.iter());
        }
        out
    };
    // Ids are not unique: every symbol with this one.
    let current = |id: &str| hashes(&|s| s == id);
    let unchanged = |id: &str, hash: &str| current(id).iter().any(|h| h == hash);

    match target {
        Target::Symbol(id) => {
            // Present if anything by that name path is: an inherent impl is
            // `impl App`, but its methods are `App/…`, so `App` can have
            // children on disk with no symbol of its own.
            if hashes(&|s| within(s, id)).is_empty() {
                return Status::Removed;
            }
            // Everything the write left under `id` must be as it left it:
            // written symbols at their hash, deleted ones still gone. That
            // includes `id` itself, which is back if the write deleted it.
            let written_intact = write.syms.iter().filter(|(s, _)| within(s, id)).all(|(s, h)| unchanged(s, h));
            let deleted_gone = write.removed.iter().filter(|s| within(s, id)).all(|s| current(s).is_empty());
            if written_intact && deleted_gone { Status::Current } else { Status::Changed }
        }
        // A file whose written symbols changed or vanished has changed since.
        Target::File(_) => {
            if write.syms.iter().all(|(s, h)| unchanged(s, h)) { Status::Current } else { Status::Changed }
        }
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
}

/// Print the answer for `arg`, as text or JSON.
pub fn run(project_root: &Path, arg: &str, json: bool) -> Result<()> {
    let target = Target::parse(arg);
    let found = latest(project_root, &target);
    let registry = ParserRegistry::new();
    let status = found.as_ref().map(|(_, w)| status(project_root, &registry, &target, w));
    let mut out = std::io::stdout().lock();

    if json {
        let dto = TouchedDto {
            schema_version: SCHEMA_VERSION,
            target: arg,
            last_write: found.as_ref().zip(status).map(|((session, w), status)| WriteDto {
                session,
                op: &w.op,
                agent: &w.a,
                time: &w.t,
                tool: &w.tool,
                level: w.level,
                status,
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SRC: &str = "fn alpha() {}\nfn beta() {}\n";

    fn hash_of(src: &str, name: &str) -> String {
        let file = ParserRegistry::new()
            .parser_for(Path::new("src/lib.rs"))
            .unwrap()
            .parse_file(Path::new("src/lib.rs"), src)
            .unwrap();
        let sym = file.symbols.iter().find(|s| &*s.name == name).unwrap();
        encode_hash(&sym.content_hash)
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
            outside_symbols: false,
            syms: syms.into_iter().map(|(s, h)| (s.to_string(), h)).collect(),
            removed: removed.into_iter().map(String::from).collect(),
            fh,
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
