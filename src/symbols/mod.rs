use std::fmt;
use std::ops::Range;
use std::path::PathBuf;

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
    pub name: String,
    pub category: SymbolCategory,
    pub label: String, // Language-specific label (e.g., "class", "struct", "def")
    pub file_path: PathBuf,
    pub byte_range: Range<usize>,
    pub line_range: Range<usize>,
    pub content_hash: [u8; 32],
    pub merkle_hash: [u8; 32],
    pub children: Vec<SymbolNode>,
    pub estimated_tokens: usize,
}

impl SymbolNode {
    pub fn total_symbols(&self) -> usize {
        1 + self.children.iter().map(|c| c.total_symbols()).sum::<usize>()
    }

    pub fn total_tokens(&self) -> usize {
        self.estimated_tokens + self.children.iter().map(|c| c.total_tokens()).sum::<usize>()
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
}

/// The full project symbol tree, organized by directory structure.
#[derive(Debug, Clone)]
pub struct ProjectTree {
    pub root: PathBuf,
    pub files: Vec<FileSymbols>,
}

impl ProjectTree {
    pub fn total_symbols(&self) -> usize {
        self.files.iter().map(|f| f.total_symbols()).sum()
    }

    pub fn total_files(&self) -> usize {
        self.files.len()
    }
}
