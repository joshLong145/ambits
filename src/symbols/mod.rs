use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub mod merkle;

pub type SymbolId = String;

/// Universal symbol categories for cross-language operations.
/// These represent broad semantic categories, not language-specific constructs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SymbolCategory {
    /// Modules, packages, namespaces
    Module,
    /// Classes, structs, enums, interfaces, traits, type aliases
    Type,
    /// Functions, methods, procedures, lambdas
    Function,
    /// Variables, constants, fields, properties, statics
    Variable,
    /// Macros, decorators, annotations, attributes
    Macro,
    /// Implementation blocks (useful for grouping)
    Implementation,
    /// Fallback for unrecognized node types
    Unknown,
}

impl fmt::Display for SymbolCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SymbolCategory::Module => write!(f, "module"),
            SymbolCategory::Type => write!(f, "type"),
            SymbolCategory::Function => write!(f, "function"),
            SymbolCategory::Variable => write!(f, "variable"),
            SymbolCategory::Macro => write!(f, "macro"),
            SymbolCategory::Implementation => write!(f, "impl"),
            SymbolCategory::Unknown => write!(f, "unknown"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SymbolNode {
    pub id: SymbolId,
    /// Symbol's short name (`foo`, `Bar`, `method_0`, etc). Stored as
    /// `Arc<str>` so common names (`new`, `default`, `get`, `build`,
    /// `method_0` repeating across types in a file) share one heap
    /// allocation. Construct via [`NameInterner::intern`] from a parser, or
    /// directly with `Arc::from(name)` from a test/fixture. Display, Eq, and
    /// `as_ref()` work via deref to `&str`; explicit comparisons against
    /// `String` need `&*sym.name == &**other` or `sym.name.as_ref() == other`.
    pub name: Arc<str>,
    pub category: SymbolCategory,
    pub label: &'static str, // Language-specific label (e.g., "class", "struct", "def")
    /// Path to the source file that produced this symbol. Wrapped in `Arc` so
    /// symbols within the same file share one backing `PathBuf` instead of
    /// each carrying their own clone — saves real heap at monorepo scale where
    /// a typical file has dozens of symbols. Reads go through deref: most
    /// consumers don't notice the change (`.as_path()`, `.display()`,
    /// `.to_string_lossy()` all work). Equality checks against `Path` /
    /// `PathBuf` need `&**arc` or `arc.as_path()` to unwrap.
    pub file_path: Arc<PathBuf>,
    /// Byte range of the symbol in its source file. `u32` is plenty for any
    /// realistic single-file size (4 GB cap) and saves 8 B/symbol over `usize`
    /// on 64-bit targets. Cast to `usize` at slice sites:
    /// `&src[sym.byte_range.start as usize..sym.byte_range.end as usize]`.
    pub byte_range: Range<u32>,
    /// 1-based inclusive line range. `u32` for the same reason as `byte_range`.
    pub line_range: Range<u32>,
    pub content_hash: [u8; 32],
    pub merkle_hash: [u8; 32],
    pub children: Vec<SymbolNode>,
    pub estimated_tokens: u32,
}

/// Per-file interner for symbol names. Real code has heavy name repetition
/// within a file (think methods named `new`, `default`, `build` across every
/// type in a module), so a small hash cache here lets us share one `Arc<str>`
/// across every reuse instead of allocating one heap buffer per occurrence.
///
/// Uses interior mutability so it can be threaded as `&NameInterner` alongside
/// other immutable parser state without requiring `&mut` propagation through
/// every recursive helper.
///
/// Scope is intentionally per-file rather than global: simpler borrow story
/// (no thread safety needed), and the wins from within-file repetition
/// dominate.
#[derive(Debug, Default)]
pub struct NameInterner {
    cache: RefCell<HashMap<String, Arc<str>>>,
}

impl NameInterner {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn intern(&self, s: &str) -> Arc<str> {
        let mut cache = self.cache.borrow_mut();
        if let Some(arc) = cache.get(s) {
            return Arc::clone(arc);
        }
        let arc: Arc<str> = Arc::from(s);
        cache.insert(s.to_string(), Arc::clone(&arc));
        arc
    }
}

impl SymbolNode {
    pub fn total_symbols(&self) -> usize {
        1 + self.children.iter().map(|c| c.total_symbols()).sum::<usize>()
    }

    pub fn total_tokens(&self) -> usize {
        self.estimated_tokens as usize
            + self.children.iter().map(|c| c.total_tokens()).sum::<usize>()
    }

    /// The nesting path within the file, e.g. `App/process_compaction`.
    ///
    /// `name` is only the leaf, and the full path lives in `id` behind the
    /// `<file>::` prefix. Deriving it here keeps the id's shape from being
    /// re-implemented by every caller that needs the other half.
    pub fn name_path(&self) -> &str {
        split_id(&self.id).1
    }
}

/// A file's worth of symbols, organized hierarchically.
#[derive(Debug, Clone)]
pub struct FileSymbols {
    pub file_path: PathBuf,
    pub symbols: Vec<SymbolNode>,
    pub total_lines: usize,
}

impl FileSymbols {
    pub fn total_symbols(&self) -> usize {
        self.symbols.iter().map(|s| s.total_symbols()).sum()
    }

    /// The innermost symbol whose byte range contains `byte`.
    ///
    /// Innermost rather than outermost: a call inside a method should be
    /// attributed to the method, not to the `impl` block wrapping it.
    ///
    /// One containment search for every consumer that has an offset and wants
    /// the symbol around it — `callers` asks it of a call site, `find` asks it
    /// of a match. It lived privately in `callers` first, which is fine until
    /// a second caller has to either re-implement the descent or forget it.
    pub fn enclosing(&self, byte: u32) -> Option<&SymbolNode> {
        enclosing(&self.symbols, byte)
    }
}

/// Recursive half of [`FileSymbols::enclosing`].
fn enclosing(symbols: &[SymbolNode], byte: u32) -> Option<&SymbolNode> {
    innermost(symbols, &|sym| sym.byte_range.start <= byte && byte < sym.byte_range.end)
}

/// The deepest symbol `contains` accepts, descending only into symbols it
/// accepts: the one containment search, whatever the position is measured
/// in (a byte for `enclosing`, a line for write attribution). A symbol whose
/// children do not contain the position is itself the answer.
pub fn innermost<'a>(symbols: &'a [SymbolNode], contains: &dyn Fn(&SymbolNode) -> bool) -> Option<&'a SymbolNode> {
    let hit = symbols.iter().find(|s| contains(s))?;
    innermost(&hit.children, contains).or(Some(hit))
}

impl FileSymbols {
    /// Every symbol in the file, depth-first — the one descent through
    /// `children` for a single file (see [`ProjectTree::walk`]).
    pub fn walk(&self) -> Vec<&SymbolNode> {
        walk_symbols(&self.symbols)
    }
}

/// Every symbol in `symbols` and beneath them, depth-first.
pub fn walk_symbols(symbols: &[SymbolNode]) -> Vec<&SymbolNode> {
    fn descend<'a>(syms: &'a [SymbolNode], out: &mut Vec<&'a SymbolNode>) {
        for sym in syms {
            out.push(sym);
            descend(&sym.children, out);
        }
    }
    let mut out = Vec::new();
    descend(symbols, &mut out);
    out
}

/// Split a symbol id into its file and its name path within the file:
/// `src/app.rs::App/new` → `("src/app.rs", "App/new")`. At the *first*
/// `::`, since name paths can contain one (`impl fmt::Display for X`) and
/// file paths in practice do not. An id without one is all name path.
pub fn split_id(id: &str) -> (&str, &str) {
    match id.split_once("::") {
        Some((file, name_path)) => (file, name_path),
        None => ("", id),
    }
}

/// `name_path` is `ancestor` or nested inside it, by name path: `App`
/// covers `App/handle_key` but not `Application`. Works on whole ids too.
pub fn nested_in(name_path: &str, ancestor: &str) -> bool {
    name_path.strip_prefix(ancestor).is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
}

/// The full project symbol tree, organized by directory structure.
#[derive(Debug, Clone)]
pub struct ProjectTree {
    pub root: PathBuf,
    pub files: Vec<FileSymbols>,
}

impl ProjectTree {
    /// Every symbol in the tree, depth-first, paired with its owning file.
    ///
    /// The single descent through `SymbolNode::children`. Four had accumulated
    /// — one each for flattening, indexing by id, indexing by content hash,
    /// and marking reads — all structurally identical and each an opportunity
    /// to forget the recursive step and silently skip nested symbols.
    ///
    /// Returns a `Vec` rather than an iterator because every caller collects or
    /// folds immediately, and a borrowing recursive iterator would cost more in
    /// complexity than the allocation saves.
    pub fn walk(&self) -> Vec<(&Path, &SymbolNode)> {
        self.files
            .iter()
            .flat_map(|file| file.walk().into_iter().map(move |sym| (file.file_path.as_path(), sym)))
            .collect()
    }

    pub fn total_symbols(&self) -> usize {
        self.files.iter().map(|f| f.total_symbols()).sum()
    }

    pub fn total_files(&self) -> usize {
        self.files.len()
    }
}

#[cfg(test)]
mod tests {
    use crate::helpers::{file, sym_with_bytes, sym_with_children};
    use crate::symbols::{nested_in, split_id, FileSymbols};

    #[test]
    fn ids_split_at_the_first_separator() {
        assert_eq!(split_id("src/a.rs::App/run"), ("src/a.rs", "App/run"));
        assert_eq!(split_id("src/a.rs::impl fmt::Display for X"), ("src/a.rs", "impl fmt::Display for X"));
        assert_eq!(split_id("no-separator"), ("", "no-separator"));
        assert!(nested_in("App/run", "App") && nested_in("App", "App") && !nested_in("Application", "App"));
    }

    /// `sym_with_children` leaves the parent spanning 0..100, so the child's
    /// 40..60 is genuinely nested inside it.
    fn thing() -> FileSymbols {
        file(
            "a.rs",
            vec![sym_with_children(
                "a.rs::Thing",
                "Thing",
                vec![sym_with_bytes("a.rs::Thing/method", "method", 40, 60)],
            )],
        )
    }

    /// The reason this is innermost-first: a call inside a method belongs to
    /// the method, not to the `impl` block wrapping it.
    #[test]
    fn enclosing_picks_the_innermost_symbol() {
        assert_eq!(
            thing().enclosing(50).unwrap().id,
            "a.rs::Thing/method",
            "an offset inside the child must not be attributed to the parent"
        );
        assert_eq!(
            thing().enclosing(10).unwrap().id,
            "a.rs::Thing",
            "an offset inside the parent alone belongs to the parent"
        );
    }

    /// File-scope offsets — a `use` line, a top-level comment — have no
    /// enclosing symbol, and must report that rather than the nearest one.
    #[test]
    fn enclosing_is_none_outside_every_range() {
        assert!(thing().enclosing(200).is_none());
    }
}
