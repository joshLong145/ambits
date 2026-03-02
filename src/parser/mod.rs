pub mod python;
pub mod rust;
pub mod typescript;

use std::fs;
use std::path::Path;

use color_eyre::eyre::Result;

use crate::symbols::{FileSymbols, ProjectTree};

/// Trait for language-specific parsers.
/// Implement this trait to add support for a new language.
pub trait LanguageParser {
    /// File extensions this parser handles (e.g., ["rs"] for Rust).
    fn extensions(&self) -> &[&str];

    /// Parse a source file into a hierarchical symbol tree.
    fn parse_file(&self, path: &Path, source: &str) -> color_eyre::Result<FileSymbols>;
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
    pub fn scan_project(&self, root: &Path) -> Result<ProjectTree> {
        use ignore::WalkBuilder;

        let mut files = Vec::new();

        for result in WalkBuilder::new(root).hidden(true).git_ignore(true).build() {
            let entry = match result {
                Ok(e) => e,
                Err(_) => continue,
            };

            let path = entry.path();
            if path.is_dir() {
                continue;
            }

            if let Some(parser) = self.parser_for(path) {
                let source = fs::read_to_string(path)?;
                let rel_path = path.strip_prefix(root).unwrap_or(path);
                match parser.parse_file(rel_path, &source) {
                    Ok(file_symbols) => files.push(file_symbols),
                    Err(e) => {
                        eprintln!("Warning: failed to parse {}: {}", path.display(), e);
                    }
                }
            }
        }

        files.sort_by(|a, b| a.file_path.cmp(&b.file_path));

        Ok(ProjectTree {
            root: root.to_path_buf(),
            files,
        })
    }
}
