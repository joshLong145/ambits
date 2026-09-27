//! `symbol`, `file` and `dir` objects (spec §5.1) from a scanned project.
//!
//! Every object is written before anything that references it — symbols
//! before their file, files before their directory, the root last — so a
//! crash never leaves an object pointing at one that is missing (§8).

use std::collections::BTreeMap;

use color_eyre::eyre::{bail, eyre, Result};
use serde_json::{json, Value};

use super::store::Store;
use super::sync_ignore::SyncIgnore;
use super::{b3, normalize_path, valid_entry_name, Kind, ObjectId};
use crate::symbols::{FileSymbols, ProjectTree, SymbolNode};

/// What [`write_tree`] stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeStats {
    pub root: ObjectId,
    pub files: usize,
    pub symbols: usize,
    /// Files left out because their path cannot be stored (§9.1): a name
    /// that is not one plain component, or one that collides with another
    /// file's after case folding. One odd file costs itself, not the
    /// snapshot.
    pub skipped: Vec<String>,
}

/// Store `tree` and return its root `dir`. `parser_of` names the parser
/// that produced a file (its [`crate::parser::LanguageParser::identity`]);
/// files `ignore` excludes are left out entirely.
pub fn write_tree(
    store: &Store,
    tree: &ProjectTree,
    parser_of: &dyn Fn(&FileSymbols) -> String,
    ignore: &SyncIgnore,
) -> Result<TreeStats> {
    let mut candidates: Vec<(String, &FileSymbols)> = tree
        .files
        .iter()
        .map(|f| (normalize_path(&f.file_path.to_string_lossy()), f))
        .filter(|(path, _)| !ignore.is_ignored(path))
        .collect();
    candidates.sort_by(|a, b| a.0.cmp(&b.0));

    // Refuse, in path order so the choice is deterministic: invalid names,
    // and any path — file or directory — equal to an earlier one after case
    // folding but spelled differently (one file on a case-insensitive
    // system; readers refuse such trees).
    let mut skipped = Vec::new();
    let mut spelling: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut included: Vec<(String, &FileSymbols, String)> = Vec::with_capacity(candidates.len());
    'files: for (path, f) in candidates {
        if !path.split('/').all(valid_entry_name) {
            skipped.push(path);
            continue;
        }
        let mut prefix = String::new();
        for component in path.split('/') {
            if !prefix.is_empty() {
                prefix.push('/');
            }
            prefix.push_str(component);
            match spelling.get(&prefix.to_lowercase()) {
                Some(seen) if *seen != prefix => {
                    skipped.push(path);
                    continue 'files;
                }
                _ => {}
            }
        }
        let mut prefix = String::new();
        for component in path.split('/') {
            if !prefix.is_empty() {
                prefix.push('/');
            }
            prefix.push_str(component);
            spelling.entry(prefix.to_lowercase()).or_insert_with(|| prefix.clone());
        }
        let parser = parser_of(f);
        included.push((path, f, parser));
    }

    // Files are independent, and each object write ends in an fsync, so a
    // first snapshot of a large project is dominated by waiting on the disk:
    // spread the files across threads. Directories follow once every file
    // is stored.
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).clamp(1, included.len().max(1));
    let chunk = included.len().div_ceil(threads).max(1);
    let written: Vec<Result<Vec<(String, ObjectId, usize)>>> = std::thread::scope(|scope| {
        let handles: Vec<_> = included
            .chunks(chunk)
            .map(|batch| {
                scope.spawn(move || {
                    batch
                        .iter()
                        .map(|(path, file, parser)| {
                            let mut count = 0;
                            let id = write_file(store, file, parser, &mut count)?;
                            Ok((path.clone(), id, count))
                        })
                        .collect()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(eyre!("a writer thread panicked")))).collect()
    });

    let mut root = Dir::default();
    let (mut files, mut symbols) = (0, 0);
    for batch in written {
        for (path, id, count) in batch? {
            // A file that is also a directory's name: keep the first.
            if root.insert(&path, id).is_err() {
                skipped.push(path);
                continue;
            }
            files += 1;
            symbols += count;
        }
    }
    skipped.sort();
    Ok(TreeStats { root: root.write(store)?, files, symbols, skipped })
}

fn write_file(store: &Store, file: &FileSymbols, parser: &str, count: &mut usize) -> Result<ObjectId> {
    // Ids are built from the path as the parser saw it — separators and
    // Unicode form untouched — so that is the prefix to strip; `name_path`
    // is normalized afterwards.
    let prefix = format!("{}::", file.file_path.to_string_lossy());
    let mut ids = Vec::with_capacity(file.symbols.len());
    for sym in &file.symbols {
        ids.push(write_symbol(store, sym, &prefix, count)?);
    }
    let payload = json!({
        "lines": file.total_lines,
        "parser": parser,
        "symbols": ids.iter().map(ObjectId::hex).collect::<Vec<_>>(),
    });
    store.put(Kind::File, &payload)
}

/// One symbol and, first, its children. The payload carries `name_path` —
/// the id without its file — because a child's id cannot be rebuilt from
/// its parent's (an inherent impl is `impl App`, its methods `App/…`), and
/// leaving the file out keeps the object shared when the file is renamed.
/// `estimated_tokens` is left out: it is recomputed on restore (§5.1).
fn write_symbol(store: &Store, sym: &SymbolNode, prefix: &str, count: &mut usize) -> Result<ObjectId> {
    let mut children = Vec::with_capacity(sym.children.len());
    for child in &sym.children {
        children.push(write_symbol(store, child, prefix, count)?.hex());
    }
    let name_path = sym.id.strip_prefix(prefix).unwrap_or(&sym.id);
    let payload = json!({
        "bytes": [sym.byte_range.start, sym.byte_range.end],
        "category": sym.category.to_string(),
        "children": children,
        "hash": b3(&sym.content_hash),
        "label": sym.label,
        "lines": [sym.line_range.start, sym.line_range.end],
        "name": super::nfc(&sym.name),
        "name_path": super::nfc(name_path),
    });
    *count += 1;
    store.put(Kind::Symbol, &payload)
}

/// A directory being assembled: entries by name.
#[derive(Default)]
struct Dir(BTreeMap<String, Entry>);

enum Entry {
    File(ObjectId),
    Dir(Dir),
}

impl Dir {
    fn insert(&mut self, path: &str, id: ObjectId) -> Result<()> {
        let (first, rest) = match path.split_once('/') {
            Some((first, rest)) => (first, Some(rest)),
            None => (path, None),
        };
        if !valid_entry_name(first) {
            bail!("cannot snapshot path {path:?}: {first:?} is not a valid name");
        }
        match (rest, self.0.entry(first.to_string())) {
            (None, std::collections::btree_map::Entry::Vacant(v)) => {
                v.insert(Entry::File(id));
            }
            (Some(rest), slot) => match slot.or_insert_with(|| Entry::Dir(Dir::default())) {
                Entry::Dir(dir) => dir.insert(rest, id)?,
                Entry::File(_) => return Err(eyre!("cannot snapshot {path:?}: {first:?} is both a file and a directory")),
            },
            (None, _) => return Err(eyre!("cannot snapshot {path:?}: listed twice")),
        }
        Ok(())
    }

    /// Write subdirectories, then this one.
    fn write(self, store: &Store) -> Result<ObjectId> {
        // Two names equal after case folding would be one file on a
        // case-insensitive system, and readers refuse them (§9.1).
        let mut folded = std::collections::HashSet::new();
        let mut entries = Vec::with_capacity(self.0.len());
        for (name, entry) in self.0 {
            if !folded.insert(name.to_lowercase()) {
                bail!("cannot snapshot: two entries named {name:?} differ only in case");
            }
            let (kind, id) = match entry {
                Entry::File(id) => ("file", id),
                Entry::Dir(dir) => ("dir", dir.write(store)?),
            };
            entries.push(json!({"id": id.hex(), "kind": kind, "name": name}));
        }
        store.put(Kind::Dir, &json!({"entries": entries}))
    }
}

/// Walk the tree under `root`, calling `visit` with every object id it
/// references, and the kind of each. Iterative, with a visited set, so a
/// hostile or corrupt store cannot recurse without bound (§9.5).
pub fn walk(store: &Store, root: ObjectId, visit: &mut dyn FnMut(ObjectId, Kind)) -> Result<()> {
    let mut stack = vec![(root, Kind::Dir, 0usize)];
    let mut seen = std::collections::HashSet::new();
    while let Some((id, kind, depth)) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        if depth > MAX_DEPTH {
            bail!("tree under {} is nested over {MAX_DEPTH} levels deep", root.short());
        }
        visit(id, kind);
        let payload = store.get(&id, kind)?;
        for (child, child_kind) in references(kind, &payload)? {
            stack.push((child, child_kind, depth + 1));
        }
    }
    Ok(())
}

/// Deepest nesting of directories and symbols a walk accepts (§9.5).
pub const MAX_DEPTH: usize = 4096;

/// The objects a tree object references directly.
pub fn references(kind: Kind, payload: &Value) -> Result<Vec<(ObjectId, Kind)>> {
    let ids = |key: &str, kind: Kind| -> Result<Vec<(ObjectId, Kind)>> {
        payload
            .get(key)
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(|v| Ok((ObjectId::parse(v.as_str().unwrap_or_default())?, kind)))
            .collect()
    };
    match kind {
        Kind::Symbol => ids("children", Kind::Symbol),
        Kind::File => ids("symbols", Kind::Symbol),
        Kind::Dir => payload
            .get("entries")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(|e| {
                let kind = match e.get("kind").and_then(Value::as_str) {
                    Some("file") => Kind::File,
                    Some("dir") => Kind::Dir,
                    other => return Err(eyre!("dir entry of unknown kind {other:?}")),
                };
                Ok((ObjectId::parse(e.get("id").and_then(Value::as_str).unwrap_or_default())?, kind))
            })
            .collect(),
        _ => Ok(Vec::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::*;

    fn parser(_: &FileSymbols) -> String {
        "rust:test@0:schema=1".into()
    }

    fn tree(files: Vec<FileSymbols>) -> ProjectTree {
        project(files)
    }

    #[test]
    fn equal_trees_give_equal_roots_and_every_object_is_reachable() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::at(dir.path());
        let t = tree(vec![
            file("src/a.rs", vec![sym_with_children("src/a.rs::A", "A", vec![sym("src/a.rs::A/f", "f")])]),
            file("src/ui/b.rs", vec![sym("src/ui/b.rs::g", "g")]),
        ]);
        let first = write_tree(&store, &t, &parser, &SyncIgnore::none()).unwrap();
        let second = write_tree(&store, &t, &parser, &SyncIgnore::none()).unwrap();
        assert_eq!(first, second);
        assert_eq!((first.files, first.symbols), (2, 3));

        let mut reached = 0;
        walk(&store, first.root, &mut |_, _| reached += 1).unwrap();
        // root, src, src/ui; two files; three symbols.
        assert_eq!(reached, 8);
        assert_eq!(store.list().len(), 8, "nothing stored that the root does not reach");
    }

    /// `\\` and `/`, and composed vs decomposed accents, name the same file
    /// — and its symbols, whose ids carry the path in whichever form the
    /// parser was given it.
    #[test]
    fn separators_and_unicode_forms_do_not_change_the_tree() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::at(dir.path());
        let (win, nfd) = ("src\\caf\u{e9}.rs", "src/cafe\u{301}.rs");
        let a = tree(vec![file(win, vec![sym(&format!("{win}::A"), "A")])]);
        let b = tree(vec![file(nfd, vec![sym(&format!("{nfd}::A"), "A")])]);
        assert_eq!(
            write_tree(&store, &a, &parser, &SyncIgnore::none()).unwrap().root,
            write_tree(&store, &b, &parser, &SyncIgnore::none()).unwrap().root
        );
    }

    #[test]
    fn ignored_files_are_left_out() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::at(dir.path());
        let ignore = SyncIgnore::new(&crate::ingest::tool_config::SyncConfig {
            ignore: Some(vec!["secrets/".into()]),
            global_ignore: vec![],
        })
        .unwrap();
        let with = tree(vec![file("src/a.rs", vec![]), file("secrets/k.rs", vec![sym("secrets/k.rs::KEY", "KEY")])]);
        let without = tree(vec![file("src/a.rs", vec![])]);
        let stats = write_tree(&store, &with, &parser, &ignore).unwrap();
        assert_eq!(stats.files, 1);
        assert_eq!(stats.root, write_tree(&store, &without, &parser, &SyncIgnore::none()).unwrap().root);
    }

    /// Case collisions and invalid names cost the file, not the snapshot,
    /// and which file goes is decided by path order, not scan order.
    #[test]
    fn unstorable_paths_are_skipped_deterministically() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::at(dir.path());
        let files = || vec![file("src/a.rs", vec![]), file("src/A.rs", vec![]), file("SRC/b.rs", vec![]), file("c:d.rs", vec![])];
        let forward = write_tree(&store, &tree(files()), &parser, &SyncIgnore::none()).unwrap();
        let mut reversed = files();
        reversed.reverse();
        let backward = write_tree(&store, &tree(reversed), &parser, &SyncIgnore::none()).unwrap();
        assert_eq!(forward, backward);
        assert_eq!(forward.files, 1);
        // Byte order puts `SRC/b.rs` first, so both `src/…` collide with it.
        assert_eq!(forward.skipped, vec!["c:d.rs", "src/A.rs", "src/a.rs"]);
    }
}
