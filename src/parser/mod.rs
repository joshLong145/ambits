pub mod python;
pub mod rust;
pub mod typescript;

use std::fs;
use std::path::{Path, PathBuf};

use color_eyre::eyre::Result;

use crate::filter::PathFilter;
use crate::symbols::{FileSymbols, ProjectTree, SymbolCategory};

/// What a grammar node maps to in the symbol tree.
///
/// Identical in all three parsers before this — each declared its own copy, so
/// a change to the shape had to be made three times or silently diverge. The
/// per-language constants built from it stay with their parsers, since those
/// genuinely differ.
#[derive(Debug, Clone, Copy)]
pub struct SymbolMeta {
    pub category: SymbolCategory,
    pub label: &'static str,
}

/// Trait for language-specific parsers.
/// Implement this trait to add support for a new language.
pub trait LanguageParser: Send + Sync {
    /// File extensions this parser handles (e.g., ["rs"] for Rust).
    fn extensions(&self) -> &[&str];

    /// Parse a source file into a hierarchical symbol tree.
    fn parse_file(&self, path: &Path, source: &str) -> color_eyre::Result<FileSymbols>;

    /// The grammar, for callers that want to run their own queries against it.
    fn language(&self) -> tree_sitter::Language;

    /// The grammar's shipped tags query, which already defines
    /// `@reference.call` among other captures.
    ///
    /// These come with the grammar crates, so reference extraction needs no
    /// per-language rules of our own — but see [`Self::tags_supplement`],
    /// because shipped is not the same as complete.
    fn tags_query(&self) -> &'static str;

    /// Extra `@reference.call` patterns appended to the shipped query.
    ///
    /// Empty by default. A grammar's own tags query is written for tagging and
    /// navigation, not for exhaustive reference finding, and it shows: Rust's
    /// captures a bare `foo()` but not `some::path::foo()`, which in Rust is
    /// most calls to anything not in scope.
    fn tags_supplement(&self) -> &'static str {
        ""
    }
}

/// Registry of all available language parsers.
pub struct ParserRegistry {
    parsers: Vec<Box<dyn LanguageParser>>,
}

impl ParserRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            parsers: Vec::new(),
        };
        registry.register(Box::new(rust::RustParser::new()));
        registry.register(Box::new(python::PythonParser::new()));
        registry.register(Box::new(typescript::TypescriptParser::new()));
        registry
    }

    pub fn register(&mut self, parser: Box<dyn LanguageParser>) {
        self.parsers.push(parser);
    }

    /// Return all file extensions supported by registered parsers.
    pub fn supported_extensions(&self) -> std::collections::HashSet<String> {
        self.parsers
            .iter()
            .flat_map(|p| p.extensions().iter().map(|e| (*e).to_string()))
            .collect()
    }

    /// Find the appropriate parser for a given file path based on extension.
    pub fn parser_for(&self, path: &Path) -> Option<&dyn LanguageParser> {
        let ext = path.extension()?.to_str()?;
        self.parsers
            .iter()
            .find(|p| p.extensions().contains(&ext))
            .map(|p| p.as_ref())
    }

    /// Walk `root` (respecting .gitignore and hidden files) and parse all
    /// recognized source files into a `ProjectTree`.
    ///
    /// If `filter` is `Some`, files whose project-relative path does not
    /// satisfy the filter are skipped *before* I/O — excluded files are not
    /// read from disk or parsed.
    pub fn scan_project(
        &self,
        root: &Path,
        filter: Option<&PathFilter>,
    ) -> Result<ProjectTree> {
        use ignore::WalkBuilder;

        // Walk first, parse second. Measured on this repo, walking the tree is
        // effectively free — restricting the parse to a single file costs the
        // same as parsing none — while parsing every file is ~95ms of the
        // ~112ms a command spends before it can answer anything. Since files
        // parse independently, that is the one part worth spreading across
        // cores.
        let mut targets: Vec<(PathBuf, PathBuf)> = Vec::new();
        for result in WalkBuilder::new(root).hidden(true).git_ignore(true).build() {
            let entry = match result {
                Ok(e) => e,
                Err(_) => continue,
            };

            let path = entry.path();
            if path.is_dir() {
                continue;
            }

            let rel_path = path.strip_prefix(root).unwrap_or(path);
            if let Some(f) = filter {
                if !f.matches(rel_path) {
                    continue;
                }
            }
            if self.parser_for(path).is_some() {
                targets.push((path.to_path_buf(), rel_path.to_path_buf()));
            }
        }

        let mut files: Vec<FileSymbols> = Vec::with_capacity(targets.len());
        if !targets.is_empty() {
            let threads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
                .min(targets.len());
            let chunk = targets.len().div_ceil(threads);

            // A read failure stays fatal, as it was when this ran serially —
            // a project we cannot read is not a project with fewer symbols.
            let results: Vec<Result<Vec<FileSymbols>>> = std::thread::scope(|scope| {
                let handles: Vec<_> = targets
                    .chunks(chunk)
                    .map(|batch| {
                        scope.spawn(move || {
                            let mut out = Vec::with_capacity(batch.len());
                            for (abs, rel) in batch {
                                let Some(parser) = self.parser_for(abs) else {
                                    continue;
                                };
                                let source = fs::read_to_string(abs)?;
                                match parser.parse_file(rel, &source) {
                                    Ok(file_symbols) => out.push(file_symbols),
                                    Err(e) => {
                                        eprintln!(
                                            "Warning: failed to parse {}: {}",
                                            abs.display(),
                                            e
                                        );
                                    }
                                }
                            }
                            Ok(out)
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().unwrap_or_else(|_| Ok(Vec::new())))
                    .collect()
            });

            for batch in results {
                files.extend(batch?);
            }
        }

        // Threads finish out of order, so the sort is now load-bearing rather
        // than cosmetic: every consumer expects files in path order.
        files.sort_by(|a, b| a.file_path.cmp(&b.file_path));

        Ok(ProjectTree {
            root: root.to_path_buf(),
            files,
        })
    }
}

impl Default for ParserRegistry {
    fn default() -> Self {
        Self::new()
    }
}
