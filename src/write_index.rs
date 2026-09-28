//! The current session's writes, as the TUI marks them on the tree.
//!
//! Status is judged against the in-memory tree, which the file watcher keeps
//! current, with the rule `touched` uses ([`FileContents::symbol_status`]),
//! so the render thread does no I/O and the two never disagree.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use crate::symbols::FileSymbols;
use crate::writes::{FileContents, Status, WriteRecord};

/// Every write of one session, one record per op (the journal's fold rule).
#[derive(Debug, Default)]
pub struct WriteIndex {
    records: BTreeMap<String, WriteRecord>,
}

impl WriteIndex {
    /// A session's journaled writes, from every shard.
    pub fn load(journal_dir: &Path, session: &str) -> Self {
        Self { records: crate::journal::read_session_writes(journal_dir, session) }
    }

    pub fn insert(&mut self, record: WriteRecord) {
        crate::journal::fold_write(&mut self.records, record);
    }

    pub fn into_records(self) -> impl Iterator<Item = WriteRecord> {
        self.records.into_values()
    }

    pub fn get(&self, op: &str) -> Option<&WriteRecord> {
        self.records.get(op)
    }

    pub fn clear(&mut self) {
        self.records.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// The writes by `agent` (all when `None`), grouped by file, latest first.
    pub fn by_file(&self, agent: Option<&str>) -> HashMap<&str, Vec<&WriteRecord>> {
        let mut out: HashMap<&str, Vec<&WriteRecord>> = HashMap::new();
        for w in self.records.values().filter(|w| agent.is_none_or(|a| w.a == a)) {
            out.entry(w.file.as_str()).or_default().push(w);
        }
        for writes in out.values_mut() {
            writes.sort_by(|a, b| b.recency().cmp(&a.recency()));
        }
        out
    }
}

/// How a row was written: the status of its latest write, which that is,
/// and how many writes touched it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteMark {
    pub status: Status,
    /// The latest write's op, for [`WriteIndex::get`].
    pub latest: String,
    pub count: usize,
}

/// One file's writes, against the file as the tree holds it now.
pub struct FileWrites<'a> {
    /// Latest first.
    writes: Vec<&'a WriteRecord>,
    now: FileContents,
}

impl<'a> FileWrites<'a> {
    pub fn new(writes: Vec<&'a WriteRecord>, file: &FileSymbols) -> Self {
        Self { writes, now: FileContents::from_symbols(file) }
    }

    /// Every write to the file, symbol- or file-level.
    pub fn file_mark(&self) -> Option<WriteMark> {
        let latest = self.writes.first()?;
        Some(WriteMark { status: self.now.file_status(latest), latest: latest.op.clone(), count: self.writes.len() })
    }

    /// The symbol-level writes that touched `id` or anything nested in it.
    pub fn symbol_mark(&self, id: &str) -> Option<WriteMark> {
        let mut touching = self.writes.iter().filter(|w| w.touches_symbol(id));
        let latest = touching.next()?;
        Some(WriteMark { status: self.now.symbol_status(id, latest), latest: latest.op.clone(), count: 1 + touching.count() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::ParserRegistry;
    use crate::writes::Level;

    const SRC: &str = "impl S {\n    fn a() {}\n    fn b() {}\n}\nfn c() {}\n";

    fn parse(src: &str) -> FileSymbols {
        let registry = ParserRegistry::new();
        let path = Path::new("src/lib.rs");
        registry.parser_for(path).unwrap().parse_file(path, src).unwrap()
    }

    fn hash(file: &FileSymbols, name_path: &str) -> String {
        let sym = file.walk().into_iter().find(|s| s.name_path() == name_path).unwrap();
        crate::journal::encode_hash(&sym.content_hash)
    }

    fn write(op: &str, t: &str, agent: &str, syms: Vec<(&str, String)>) -> WriteRecord {
        WriteRecord {
            op: op.into(),
            av: crate::writes::ATTRIBUTION_VERSION,
            a: agent.into(),
            t: t.into(),
            tool: "Edit".into(),
            file: "src/lib.rs".into(),
            level: if syms.is_empty() { Level::File } else { Level::Symbol },
            syms: syms.into_iter().map(|(s, h)| (format!("src/lib.rs::{s}"), h)).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn a_written_symbol_is_current_until_the_tree_changes() {
        let before = parse(SRC);
        let mut index = WriteIndex::default();
        index.insert(write("t1", "2026-09-27T10:00:00Z", "main", vec![("S/a", hash(&before, "S/a"))]));
        let by_file = index.by_file(None);
        let marks = FileWrites::new(by_file["src/lib.rs"].clone(), &before);
        assert_eq!(marks.symbol_mark("src/lib.rs::S/a").map(|m| m.status), Some(Status::Current));
        assert_eq!(marks.symbol_mark("src/lib.rs::S/b"), None);
        assert_eq!(marks.symbol_mark("src/lib.rs::c"), None);

        let after = parse(&SRC.replace("fn a() {}", "fn a() { 1; }"));
        let marks = FileWrites::new(by_file["src/lib.rs"].clone(), &after);
        assert_eq!(marks.symbol_mark("src/lib.rs::S/a").map(|m| m.status), Some(Status::Changed));
    }

    /// A parent is written through its children, and takes their latest write.
    #[test]
    fn a_parent_rolls_up_its_children() {
        let file = parse(SRC);
        let mut index = WriteIndex::default();
        index.insert(write("t1", "2026-09-27T10:00:00Z", "main", vec![("S/a", "b3:stale".into())]));
        index.insert(write("t2", "2026-09-27T10:00:01Z", "main", vec![("S/b", hash(&file, "S/b"))]));
        let by_file = index.by_file(None);
        let marks = FileWrites::new(by_file["src/lib.rs"].clone(), &file);
        assert_eq!(marks.symbol_mark("src/lib.rs::S"), Some(WriteMark { status: Status::Current, latest: "t2".into(), count: 2 }));
        assert_eq!(marks.symbol_mark("src/lib.rs::S/a").map(|m| m.status), Some(Status::Changed));
        assert_eq!(marks.file_mark().map(|m| m.count), Some(2));
    }

    /// A file-level write marks the file, and no symbol; with no hash to
    /// compare in memory its status is unknown.
    #[test]
    fn a_file_level_write_marks_only_the_file() {
        let file = parse(SRC);
        let mut index = WriteIndex::default();
        index.insert(write("t1", "2026-09-27T10:00:00Z", "main", vec![]));
        let by_file = index.by_file(None);
        let marks = FileWrites::new(by_file["src/lib.rs"].clone(), &file);
        assert_eq!(marks.file_mark(), Some(WriteMark { status: Status::Unknown, latest: "t1".into(), count: 1 }));
        assert_eq!(marks.symbol_mark("src/lib.rs::S"), None);
    }

    #[test]
    fn the_agent_filter_keeps_that_agents_writes() {
        let mut index = WriteIndex::default();
        index.insert(write("t1", "2026-09-27T10:00:00Z", "main", vec![]));
        index.insert(write("t2", "2026-09-27T10:00:01Z", "ax1", vec![]));
        assert_eq!(index.by_file(Some("ax1"))["src/lib.rs"].iter().map(|w| w.op.as_str()).collect::<Vec<_>>(), ["t2"]);
        assert_eq!(index.by_file(None)["src/lib.rs"].iter().map(|w| w.op.as_str()).collect::<Vec<_>>(), ["t2", "t1"]);
        assert!(index.by_file(Some("other")).is_empty());
    }

    /// The same op recorded twice is one write; a newer attribution wins.
    #[test]
    fn an_op_is_indexed_once() {
        let mut index = WriteIndex::default();
        let mut old = write("t1", "2026-09-27T10:00:00Z", "main", vec![]);
        old.av = 1;
        index.insert(old);
        index.insert(write("t1", "2026-09-27T10:00:00Z", "main", vec![("c", "b3:x".into())]));
        let by_file = index.by_file(None);
        assert_eq!(by_file["src/lib.rs"].len(), 1);
        assert_eq!(by_file["src/lib.rs"][0].level, Level::Symbol);
    }
}
