pub mod markdown;
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

/// The earliest node in an unbroken run of comments immediately above
/// `node` — no blank line, no other node between one and the next — or
/// `None` if `node` has no such comment directly above it.
///
/// A `///`/`//!` doc comment is the common case this exists for, but the
/// rule is contiguity, not comment syntax: a plain `//` note glued to what
/// follows reads as being about it too, so it is swept in exactly like a doc
/// comment would be. `comment_kinds` names the grammar's comment node kinds
/// (Rust distinguishes `line_comment`/`block_comment`; Python and
/// TypeScript each use a single `comment` kind), since that differs per
/// language and nothing else in this walk does.
///
/// A comment separated from `node` by a blank line is left alone — this
/// never even looks past it — and so is a comment with nothing recognized
/// following it at all, since callers only invoke this once a node has
/// already been classified as a symbol. That is what keeps a standalone
/// file-level comment attributed to nothing, matching `find`'s
/// `a_hit_between_symbols_has_no_symbol`: widening symbols is all this
/// does, never inventing one.
///
/// Called once per symbol at parse time, not per search, so walking
/// `prev_sibling()` a few times costs nothing worth avoiding.
pub fn leading_comment_start<'a>(
    node: tree_sitter::Node<'a>,
    comment_kinds: &[&str],
) -> Option<tree_sitter::Node<'a>> {
    let mut result = None;
    let mut boundary_row = node.start_position().row;
    let mut sibling = node.prev_sibling();

    while let Some(sib) = sibling {
        if !comment_kinds.contains(&sib.kind()) {
            break;
        }
        // More than one row of gap between this comment and whatever sits
        // just below it means a blank line separates them.
        if boundary_row.saturating_sub(sib.end_position().row) > 1 {
            break;
        }
        boundary_row = sib.start_position().row;
        result = Some(sib);
        sibling = sib.prev_sibling();
    }

    result
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

/// Where a walk may go, beyond the defaults every command shares.
///
/// The defaults reproduce the scanner's historical behaviour exactly — hidden
/// files skipped, ignore files honoured, no glob or type narrowing — so a
/// caller that wants that says `WalkOptions::default()` and nothing else.
/// `find` populates the rest from its ripgrep-compatible flags.
#[derive(Debug, Default)]
pub struct WalkOptions<'a> {
    /// Project-relative path filter: `--filter` / `--filter-regex`, and
    /// `find`'s positional PATH arguments.
    pub filter: Option<&'a PathFilter>,
    /// Glob overrides (`-g`), from `ignore::overrides::OverrideBuilder`.
    pub overrides: Option<ignore::overrides::Override>,
    /// File-type narrowing (`-t`), from `ignore::types::TypesBuilder`.
    pub types: Option<ignore::types::Types>,
    /// Include hidden files (`--hidden`). Inverted relative to
    /// `WalkBuilder::hidden`, which takes "skip hidden".
    pub hidden: bool,
    /// Ignore `.gitignore`, `.ignore`, and their global and parent variants
    /// (`--no-ignore`).
    pub no_ignore: bool,
    /// Subtrees to walk instead of the whole project — grep's `PATH...`.
    ///
    /// Absolute, and inside the project root (the CLI checks that, so a bad
    /// path fails with a message rather than silently yielding nothing).
    /// Walking only what was asked for beats walking everything and discarding
    /// most of it, which is why these are roots rather than another filter.
    pub roots: Vec<PathBuf>,
}

/// Every file under `root` the options admit, as `(absolute, project-relative)`.
///
/// Split out of [`ParserRegistry::scan_project`] because `find` needs the same
/// traversal without the parse that used to follow it, and because glob and
/// type narrowing belong to the walk rather than to any one command. Walk
/// errors are skipped rather than propagated: one unreadable directory should
/// narrow the answer, not fail the command.
pub fn walk_files(root: &Path, opts: &WalkOptions<'_>) -> Vec<(PathBuf, PathBuf)> {
    use ignore::WalkBuilder;

    let mut builder = WalkBuilder::new(opts.roots.first().map_or(root, |p| p.as_path()));
    for extra in opts.roots.iter().skip(1) {
        builder.add(extra);
    }
    builder
        .hidden(!opts.hidden)
        .ignore(!opts.no_ignore)
        .git_ignore(!opts.no_ignore)
        .git_global(!opts.no_ignore)
        .git_exclude(!opts.no_ignore);
    if let Some(o) = &opts.overrides {
        builder.overrides(o.clone());
    }
    if let Some(t) = &opts.types {
        builder.types(t.clone());
    }

    let mut out = Vec::new();
    for entry in builder.build().flatten() {
        let path = entry.path();
        if path.is_dir() {
            continue;
        }
        let rel_path = path.strip_prefix(root).unwrap_or(path);
        if let Some(f) = opts.filter {
            if !f.matches(rel_path) {
                continue;
            }
        }
        out.push((path.to_path_buf(), rel_path.to_path_buf()));
    }
    out
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
        registry.register(Box::new(markdown::MarkdownParser::new()));
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
        // Walk first, parse second. Measured on this repo, walking the tree is
        // effectively free — restricting the parse to a single file costs the
        // same as parsing none — while parsing every file is ~95ms of the
        // ~112ms a command spends before it can answer anything. Since files
        // parse independently, that is the one part worth spreading across
        // cores.
        //
        // The extension test stays here rather than in the walk: a scan wants
        // only files it can parse, but a content search wants every text file
        // in scope and attributes symbols to the subset that parses.
        let targets: Vec<(PathBuf, PathBuf)> = walk_files(
            root,
            &WalkOptions {
                filter,
                ..Default::default()
            },
        )
        .into_iter()
        .filter(|(abs, _)| self.parser_for(abs).is_some())
        .collect();

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
