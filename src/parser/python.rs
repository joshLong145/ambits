//! Python symbol extractor using tree-sitter.
//!
//! This module parses `.py` files into a hierarchical [`SymbolNode`] tree that the
//! ambits coverage system uses to track which parts of a codebase have been reviewed.
//!
//! ## How it works
//!
//! 1. The source is fed to tree-sitter-python which produces a concrete syntax
//!    tree (CST).
//! 2. [`extract_symbols`] walks the top-level children of each node, dispatching
//!    recognized node kinds to helper functions that build [`SymbolNode`]s.
//! 3. `decorated_definition` nodes are unwrapped so that `@decorator def foo()`
//!    is treated the same as `def foo()`, with the byte range widened to include
//!    the decorator(s).
//! 4. Compound statements (`if`, `for`, `try`, etc.) are recursed into so that
//!    definitions nested inside control flow are still discovered.
//! 5. After the full tree is built, Merkle hashes are computed bottom-up so that
//!    content changes propagate to parent symbols.
//!
//! ## Supported Python constructs
//!
//! | Construct                                  | Category | Label    |
//! |--------------------------------------------|----------|----------|
//! | `def`, `async def`                         | Function | `"def"`  |
//! | `class`                                    | Type     | `"class"`|
//! | `@decorator` on `def` / `class`            | (same)   | (same)   |
//! | `type Alias = ...` (Python 3.12+)          | Type     | `"type"` |
//! | `x: int = ...` (annotated assignment)      | Variable | `"var"`  |
//! | `MAX_SIZE = ...` (UPPER_SNAKE_CASE)        | Variable | `"var"`  |
//! | class methods                              | Function | `"def"`  |

use std::path::Path;

use color_eyre::eyre::eyre;
use tree_sitter::{Node, Parser};

use crate::symbols::merkle::{compute_merkle_hash, content_hash, estimate_tokens};
use crate::symbols::{FileSymbols, SymbolCategory, SymbolNode};

use super::LanguageParser;

/// Parser for Python (`.py`) source files.
///
/// Uses the tree-sitter-python grammar to produce a CST, then extracts
/// a simplified symbol tree that ambits uses for coverage tracking.
pub struct PythonParser {
    _private: (),
}

impl PythonParser {
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl LanguageParser for PythonParser {
    fn extensions(&self) -> &[&str] {
        &["py"]
    }

    fn parse_file(&self, path: &Path, source: &str) -> color_eyre::Result<FileSymbols> {
        let mut parser = Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser
            .set_language(&language.into())
            .map_err(|e| eyre!("Failed to set language: {}", e))?;

        let tree = parser
            .parse(source, None)
            .ok_or_else(|| eyre!("Failed to parse {}", path.display()))?;

        let root = tree.root_node();
        let path_prefix = path.to_string_lossy();
        let src = source.as_bytes();
        let mut symbols = Vec::new();

        extract_symbols(root, src, path, &path_prefix, "", &mut symbols);

        for sym in symbols.iter_mut() {
            compute_merkle_hash(sym);
        }

        let total_lines = source.lines().count();

        Ok(FileSymbols {
            file_path: path.to_path_buf(),
            symbols,
            total_lines,
        })
    }
}

// ---------------------------------------------------------------------------
// Symbol metadata constants
// ---------------------------------------------------------------------------
//
// Each constant pairs a `SymbolCategory` (the semantic bucket - Function, Type,
// Variable) with a human-readable `label` that appears in the UI.
// These are referenced by the extraction functions below to avoid repeating the
// mapping logic at every call site.

/// Pairs a [`SymbolCategory`] with a display label for use in [`SymbolNode`].
struct SymbolMeta {
    category: SymbolCategory,
    label: &'static str,
}

// -- Definitions ------------------------------------------------------------
const CLASS: SymbolMeta = SymbolMeta { category: SymbolCategory::Type, label: "class" };
const DEF: SymbolMeta = SymbolMeta { category: SymbolCategory::Function, label: "def" };

// -- Variables & type aliases -----------------------------------------------
const VAR: SymbolMeta = SymbolMeta { category: SymbolCategory::Variable, label: "var" };
const TYPE_ALIAS: SymbolMeta = SymbolMeta { category: SymbolCategory::Type, label: "type" };

// ---------------------------------------------------------------------------
// Core symbol extraction
// ---------------------------------------------------------------------------

/// Recursively walk the children of `node` and extract recognized Python symbols.
///
/// This is the main dispatch loop. For each child it matches on the tree-sitter
/// node kind and either:
/// - Returns a `(name, SymbolMeta)` pair for leaf symbols (`def`, `class`,
///   assignments, type aliases),
/// - Delegates to a helper that pushes symbols directly into `out`
///   (`decorated_definition`, compound statements), or
/// - Skips the node entirely (imports, comments, plain expressions).
///
/// For class definitions, the function recurses into the class `body` to
/// extract methods as child symbols.
///
/// The function is called at the top level (with `root_node`) and recursively
/// by [`extract_from_compound_bodies`] to discover definitions inside control
/// flow blocks, and by itself to find methods inside class bodies.
fn extract_symbols(
    node: Node,
    src: &[u8],
    file_path: &Path,
    path_prefix: &str,
    parent_name_path: &str,
    out: &mut Vec<SymbolNode>,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        let symbol_info = match child.kind() {
            "function_definition" => {
                child_name(&child, src).map(|n| (n, DEF))
            }
            "class_definition" => child_name(&child, src).map(|n| (n, CLASS)),
            // Decorated definitions: unwrap the decorator to find the inner def/class.
            "decorated_definition" => {
                extract_decorated(&child, src, file_path, path_prefix, parent_name_path, out);
                None
            }
            // Module-level assignments: annotated or UPPER_SNAKE_CASE only.
            "expression_statement" => extract_assignment(&child, src),
            // Python 3.12+ type alias: `type Alias = int`
            "type_alias_statement" => extract_type_alias(&child, src),
            // Recurse into compound statement bodies for nested definitions.
            "if_statement" | "for_statement" | "while_statement"
            | "try_statement" | "with_statement" | "match_statement" => {
                extract_from_compound_bodies(
                    &child, src, file_path, path_prefix, parent_name_path, out,
                );
                None
            }
            _ => None,
        };

        if let Some((name, meta)) = symbol_info {
            let name_path = if parent_name_path.is_empty() {
                name.clone()
            } else {
                format!("{parent_name_path}/{name}")
            };

            let id = format!("{path_prefix}::{name_path}");
            let byte_range = child.byte_range();
            let start_line = child.start_position().row + 1;
            let end_line = child.end_position().row + 1;
            let text = std::str::from_utf8(&src[byte_range.clone()]).unwrap_or("");

            let mut sym = SymbolNode {
                id,
                name: name.clone(),
                category: meta.category,
                label: meta.label.to_string(),
                file_path: file_path.to_path_buf(),
                byte_range,
                line_range: start_line..end_line,
                content_hash: content_hash(text),
                merkle_hash: [0u8; 32],
                children: Vec::new(),
                estimated_tokens: estimate_tokens(text),
            };

            // For classes, recurse into the body block to find methods.
            if meta.category == SymbolCategory::Type {
                if let Some(body) = child.child_by_field_name("body") {
                    extract_symbols(body, src, file_path, path_prefix, &name_path, &mut sym.children);
                }
            }

            out.push(sym);
        }
    }
}

// ---------------------------------------------------------------------------
// Decorator unwrapping
// ---------------------------------------------------------------------------

/// Unwrap a `decorated_definition` node to extract the inner `def` or `class`.
///
/// Python decorators are represented in tree-sitter as a wrapper node
/// (`decorated_definition`) whose children are one or more `decorator` nodes
/// followed by a `function_definition` or `class_definition`. We walk the
/// children to find the inner declaration, then build a [`SymbolNode`] whose
/// byte range spans the *entire* decorated block (including the `@` lines)
/// so that coverage tracking includes the decorators.
///
/// For decorated classes, we recurse into the class body to extract methods.
fn extract_decorated(
    node: &Node,
    src: &[u8],
    file_path: &Path,
    path_prefix: &str,
    parent_name_path: &str,
    out: &mut Vec<SymbolNode>,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "function_definition" | "class_definition" => {
                // Re-use the parent extraction logic but with the decorator node's range.
                let name = match child_name(&child, src) {
                    Some(n) => n,
                    None => return,
                };
                let meta = match child.kind() {
                    "class_definition" => CLASS,
                    _ => DEF,
                };

                let name_path = if parent_name_path.is_empty() {
                    name.clone()
                } else {
                    format!("{parent_name_path}/{name}")
                };

                let id = format!("{path_prefix}::{name_path}");
                // Use the outer decorated_definition range to include decorators.
                let byte_range = node.byte_range();
                let start_line = node.start_position().row + 1;
                let end_line = node.end_position().row + 1;
                let text = std::str::from_utf8(&src[byte_range.clone()]).unwrap_or("");

                let mut sym = SymbolNode {
                    id,
                    name: name.clone(),
                    category: meta.category,
                    label: meta.label.to_string(),
                    file_path: file_path.to_path_buf(),
                    byte_range,
                    line_range: start_line..end_line,
                    content_hash: content_hash(text),
                    merkle_hash: [0u8; 32],
                    children: Vec::new(),
                    estimated_tokens: estimate_tokens(text),
                };

                if meta.category == SymbolCategory::Type {
                    if let Some(body) = child.child_by_field_name("body") {
                        extract_symbols(body, src, file_path, path_prefix, &name_path, &mut sym.children);
                    }
                }

                out.push(sym);
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Tree-sitter node helpers
// ---------------------------------------------------------------------------

/// Extract the identifier name from a node via its `"name"` field.
///
/// Most Python declaration nodes (`function_definition`, `class_definition`)
/// expose their identifier through a field called `"name"`. Returns `None`
/// if the field is missing or the text cannot be decoded as UTF-8.
fn child_name(node: &Node, src: &[u8]) -> Option<String> {
    node.child_by_field_name("name")?
        .utf8_text(src)
        .ok()
        .map(|s| s.to_string())
}

// ---------------------------------------------------------------------------
// Variable & type-alias extraction
// ---------------------------------------------------------------------------

/// Check whether `name` follows `UPPER_SNAKE_CASE` convention.
///
/// Returns `true` when the name is non-empty, contains at least one uppercase
/// ASCII letter, and every character is uppercase ASCII, a digit, or `_`.
/// This intentionally rejects names like `_` or `__` (no uppercase letter)
/// and mixed-case names like `Max_Size`.
fn is_upper_snake_case(name: &str) -> bool {
    !name.is_empty()
        && name.chars().any(|c| c.is_ascii_uppercase())
        && name.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}

/// Try to extract a variable symbol from an `expression_statement`.
///
/// In tree-sitter-python, assignments live inside `expression_statement` nodes.
/// This function iterates the statement's children looking for an `assignment`
/// node, then checks whether it qualifies:
///
/// 1. **Annotated assignment** - the `assignment` node has a `type` field
///    (e.g. `x: int = 42` or the bare annotation `x: int`).
/// 2. **UPPER_SNAKE_CASE constant** - the left-hand side identifier matches
///    the `UPPER_SNAKE_CASE` convention (e.g. `MAX_SIZE = 100`).
///
/// Plain lowercase assignments (`x = 42`), tuple unpacking (`a, b = ...`),
/// attribute assignments (`self.x = ...`), and augmented assignments (`x += 1`)
/// are all intentionally ignored.
fn extract_assignment(node: &Node, src: &[u8]) -> Option<(String, SymbolMeta)> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() != "assignment" {
            continue;
        }

        let left = child.child_by_field_name("left")?;
        if left.kind() != "identifier" {
            return None;
        }

        let name = left.utf8_text(src).ok()?.to_string();
        let has_annotation = child.child_by_field_name("type").is_some();

        if has_annotation || is_upper_snake_case(&name) {
            return Some((name, VAR));
        }

        return None;
    }
    None
}

/// Extract a type alias from a `type_alias_statement` node (Python 3.12+).
///
/// Handles the `type Alias = int` syntax introduced in PEP 695. The grammar
/// exposes the alias name via the `"left"` field and the target type via
/// `"right"`. We only need the name to build a [`SymbolNode`].
fn extract_type_alias(node: &Node, src: &[u8]) -> Option<(String, SymbolMeta)> {
    let left = node.child_by_field_name("left")?;
    let name = left.utf8_text(src).ok()?.to_string();
    Some((name, TYPE_ALIAS))
}

// ---------------------------------------------------------------------------
// Compound statement traversal
// ---------------------------------------------------------------------------

/// Recurse into a compound statement's body/bodies to find nested definitions.
///
/// Python allows `def` and `class` inside any block:
/// ```python
/// if __name__ == "__main__":
///     def main(): ...
/// ```
///
/// This function walks the various clause types of each compound statement and
/// calls [`extract_symbols`] on each body block so that nested definitions are
/// discovered. The compound statement itself is *not* emitted as a symbol.
///
/// Because `extract_symbols` is called (not a reduced variant), nested compound
/// statements are handled transitively (e.g. `try` > `if` > `def`).
///
/// ## Grammar field names
///
/// | Statement        | Body field                                          |
/// |------------------|-----------------------------------------------------|
/// | `if_statement`   | `consequence`; children: `elif_clause`/`else_clause`|
/// | `for_statement`  | `body`                                              |
/// | `while_statement`| `body`                                              |
/// | `with_statement` | `body`                                              |
/// | `try_statement`  | `body`; children: `except_clause`/`finally_clause`  |
/// | `match_statement`| `body` (block of `case_clause`s via `consequence`)  |
fn extract_from_compound_bodies(
    node: &Node,
    src: &[u8],
    file_path: &Path,
    path_prefix: &str,
    parent_name_path: &str,
    out: &mut Vec<SymbolNode>,
) {
    match node.kind() {
        "for_statement" | "while_statement" | "with_statement" => {
            if let Some(body) = node.child_by_field_name("body") {
                extract_symbols(body, src, file_path, path_prefix, parent_name_path, out);
            }
        }
        "if_statement" => {
            if let Some(body) = node.child_by_field_name("consequence") {
                extract_symbols(body, src, file_path, path_prefix, parent_name_path, out);
            }
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                match child.kind() {
                    "elif_clause" => {
                        if let Some(body) = child.child_by_field_name("consequence") {
                            extract_symbols(
                                body, src, file_path, path_prefix, parent_name_path, out,
                            );
                        }
                    }
                    "else_clause" => {
                        if let Some(body) = child.child_by_field_name("body") {
                            extract_symbols(
                                body, src, file_path, path_prefix, parent_name_path, out,
                            );
                        }
                    }
                    _ => {}
                }
            }
        }
        "try_statement" => {
            if let Some(body) = node.child_by_field_name("body") {
                extract_symbols(body, src, file_path, path_prefix, parent_name_path, out);
            }
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                match child.kind() {
                    "except_clause" | "except_group_clause" | "finally_clause" => {
                        let mut inner_cursor = child.walk();
                        for inner in child.children(&mut inner_cursor) {
                            if inner.kind() == "block" {
                                extract_symbols(
                                    inner, src, file_path, path_prefix, parent_name_path, out,
                                );
                            }
                        }
                    }
                    "else_clause" => {
                        if let Some(body) = child.child_by_field_name("body") {
                            extract_symbols(
                                body, src, file_path, path_prefix, parent_name_path, out,
                            );
                        }
                    }
                    _ => {}
                }
            }
        }
        "match_statement" => {
            if let Some(body) = node.child_by_field_name("body") {
                let mut cursor = body.walk();
                for child in body.children(&mut cursor) {
                    if child.kind() == "case_clause" {
                        if let Some(consequence) = child.child_by_field_name("consequence") {
                            extract_symbols(
                                consequence, src, file_path, path_prefix, parent_name_path, out,
                            );
                        }
                    }
                }
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::LanguageParser;
    use std::path::Path;

    fn parse(src: &str) -> Vec<SymbolNode> {
        let parser = PythonParser::new();
        let file = parser.parse_file(Path::new("test.py"), src).unwrap();
        file.symbols
    }

    #[test]
    fn extensions() {
        let parser = PythonParser::new();
        assert_eq!(parser.extensions(), &["py"]);
    }

    #[test]
    fn parse_empty_file() {
        let syms = parse("");
        assert!(syms.is_empty());
    }

    #[test]
    fn parse_simple_function() {
        let syms = parse("def foo():\n    pass\n");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "foo");
        assert_eq!(syms[0].category, SymbolCategory::Function);
        assert_eq!(syms[0].label, "def");
        assert_eq!(syms[0].id, "test.py::foo");
        assert!(syms[0].children.is_empty());
    }

    #[test]
    fn parse_multiple_functions() {
        let syms = parse("def foo():\n    pass\n\ndef bar():\n    pass\n");
        assert_eq!(syms.len(), 2);
        assert_eq!(syms[0].name, "foo");
        assert_eq!(syms[1].name, "bar");
    }

    #[test]
    fn parse_simple_class() {
        let syms = parse("class Foo:\n    pass\n");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "Foo");
        assert_eq!(syms[0].category, SymbolCategory::Type);
        assert_eq!(syms[0].label, "class");
        assert_eq!(syms[0].id, "test.py::Foo");
    }

    #[test]
    fn parse_class_with_methods() {
        let syms = parse(
            "class Foo:\n    def __init__(self):\n        pass\n    def bar(self):\n        pass\n",
        );
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "Foo");
        assert_eq!(syms[0].category, SymbolCategory::Type);
        assert_eq!(syms[0].children.len(), 2);
        assert_eq!(syms[0].children[0].name, "__init__");
        assert_eq!(syms[0].children[0].category, SymbolCategory::Function);
        assert_eq!(syms[0].children[0].id, "test.py::Foo/__init__");
        assert_eq!(syms[0].children[1].name, "bar");
        assert_eq!(syms[0].children[1].id, "test.py::Foo/bar");
    }

    #[test]
    fn parse_decorated_function() {
        let syms = parse("@staticmethod\ndef foo():\n    pass\n");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "foo");
        assert_eq!(syms[0].category, SymbolCategory::Function);
        assert_eq!(syms[0].label, "def");
        // Decorated definitions include the decorator in the byte range
        assert_eq!(syms[0].line_range.start, 1);
    }

    #[test]
    fn parse_decorated_class() {
        let syms = parse("@dataclass\nclass Point:\n    x: int\n    y: int\n");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "Point");
        assert_eq!(syms[0].category, SymbolCategory::Type);
        assert_eq!(syms[0].label, "class");
        // Decorated class range includes the decorator
        assert_eq!(syms[0].line_range.start, 1);
    }

    #[test]
    fn parse_decorated_class_with_methods() {
        let syms = parse(
            "@dataclass\nclass Point:\n    def __init__(self):\n        pass\n    def distance(self):\n        pass\n",
        );
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "Point");
        assert_eq!(syms[0].category, SymbolCategory::Type);
        assert_eq!(syms[0].children.len(), 2);
        assert_eq!(syms[0].children[0].name, "__init__");
        assert_eq!(syms[0].children[1].name, "distance");
    }

    #[test]
    fn parse_function_and_class_mixed() {
        let syms = parse(
            "def helper():\n    pass\n\nclass Foo:\n    def method(self):\n        pass\n\ndef another():\n    pass\n",
        );
        assert_eq!(syms.len(), 3);
        assert_eq!(syms[0].name, "helper");
        assert_eq!(syms[0].category, SymbolCategory::Function);
        assert_eq!(syms[1].name, "Foo");
        assert_eq!(syms[1].category, SymbolCategory::Type);
        assert_eq!(syms[1].children.len(), 1);
        assert_eq!(syms[2].name, "another");
        assert_eq!(syms[2].category, SymbolCategory::Function);
    }

    #[test]
    fn parse_nested_class() {
        let syms = parse(
            "class Outer:\n    class Inner:\n        def method(self):\n            pass\n",
        );
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "Outer");
        assert_eq!(syms[0].children.len(), 1);
        assert_eq!(syms[0].children[0].name, "Inner");
        assert_eq!(syms[0].children[0].category, SymbolCategory::Type);
        assert_eq!(syms[0].children[0].id, "test.py::Outer/Inner");
        // Inner class should have its own children
        assert_eq!(syms[0].children[0].children.len(), 1);
        assert_eq!(syms[0].children[0].children[0].name, "method");
        assert_eq!(syms[0].children[0].children[0].id, "test.py::Outer/Inner/method");
    }

    #[test]
    fn line_ranges_are_correct() {
        let syms = parse("def foo():\n    pass\n\ndef bar():\n    pass\n");
        assert_eq!(syms[0].line_range, 1..2);
        assert_eq!(syms[1].line_range, 4..5);
    }

    #[test]
    fn merkle_hashes_are_computed() {
        let syms = parse("def foo():\n    pass\n");
        // After parsing, merkle hash should be non-zero
        assert_ne!(syms[0].merkle_hash, [0u8; 32]);
    }

    #[test]
    fn content_hashes_are_set() {
        let syms = parse("def foo():\n    pass\n");
        assert_ne!(syms[0].content_hash, [0u8; 32]);
    }

    #[test]
    fn estimated_tokens_are_positive() {
        let syms = parse("def foo():\n    pass\n");
        assert!(syms[0].estimated_tokens > 0);
    }

    #[test]
    fn total_lines_counted() {
        let parser = PythonParser::new();
        let file = parser
            .parse_file(Path::new("test.py"), "def foo():\n    pass\n\ndef bar():\n    pass\n")
            .unwrap();
        assert_eq!(file.total_lines, 5);
    }

    #[test]
    fn file_path_is_set() {
        let parser = PythonParser::new();
        let file = parser
            .parse_file(Path::new("src/main.py"), "def foo():\n    pass\n")
            .unwrap();
        assert_eq!(file.file_path, Path::new("src/main.py"));
        assert_eq!(file.symbols[0].file_path, Path::new("src/main.py"));
        assert_eq!(file.symbols[0].id, "src/main.py::foo");
    }

    #[test]
    fn parse_only_comments_and_imports() {
        // Statements that aren't function/class definitions should be ignored
        let syms = parse("import os\nfrom sys import path\n# a comment\nx = 42\n");
        assert!(syms.is_empty());
    }

    #[test]
    fn parse_decorated_method_inside_class() {
        let syms = parse(
            "class Foo:\n    @staticmethod\n    def bar():\n        pass\n",
        );
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "Foo");
        assert_eq!(syms[0].children.len(), 1);
        assert_eq!(syms[0].children[0].name, "bar");
        assert_eq!(syms[0].children[0].category, SymbolCategory::Function);
    }

    #[test]
    fn parse_multiple_decorators() {
        let syms = parse("@decorator1\n@decorator2\ndef foo():\n    pass\n");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "foo");
        assert_eq!(syms[0].line_range.start, 1);
    }

    #[test]
    fn parse_async_function() {
        // tree-sitter-python wraps async def in function_definition,
        // but the parser only matches "function_definition" kind
        let syms = parse("async def foo():\n    pass\n");
        // async def is not a direct `function_definition` – it is an expression_statement
        // or may be wrapped. Verify actual behavior either way.
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "foo");
        assert_eq!(syms[0].category, SymbolCategory::Function);
    }

    #[test]
    fn byte_ranges_are_valid() {
        let src = "def foo():\n    pass\n";
        let syms = parse(src);
        let range = &syms[0].byte_range;
        assert!(range.start < range.end);
        assert!(range.end <= src.len());
    }

    // --- Variable extraction tests ---

    #[test]
    fn parse_annotated_variable() {
        let syms = parse("x: int = 42\n");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "x");
        assert_eq!(syms[0].category, SymbolCategory::Variable);
        assert_eq!(syms[0].label, "var");
        assert_eq!(syms[0].id, "test.py::x");
        assert!(syms[0].children.is_empty());
    }

    #[test]
    fn parse_annotated_variable_no_value() {
        let syms = parse("x: int\n");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "x");
        assert_eq!(syms[0].category, SymbolCategory::Variable);
    }

    #[test]
    fn parse_upper_case_constant() {
        let syms = parse("MAX_SIZE = 100\n");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "MAX_SIZE");
        assert_eq!(syms[0].category, SymbolCategory::Variable);
        assert_eq!(syms[0].label, "var");
    }

    #[test]
    fn parse_upper_case_no_underscore() {
        let syms = parse("DEBUG = True\n");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "DEBUG");
        assert_eq!(syms[0].category, SymbolCategory::Variable);
    }

    #[test]
    fn parse_lowercase_assignment_ignored() {
        let syms = parse("x = 42\n");
        assert!(syms.is_empty());
    }

    #[test]
    fn parse_mixed_case_assignment_ignored() {
        let syms = parse("myVar = 10\n");
        assert!(syms.is_empty());
    }

    #[test]
    fn parse_tuple_assignment_ignored() {
        let syms = parse("A, B = 1, 2\n");
        assert!(syms.is_empty());
    }

    #[test]
    fn parse_attribute_assignment_ignored() {
        let syms = parse("obj.MAX = 100\n");
        assert!(syms.is_empty());
    }

    #[test]
    fn parse_variables_mixed_with_functions() {
        let syms = parse("MAX: int = 100\n\ndef foo():\n    pass\n\nDEBUG = True\n");
        assert_eq!(syms.len(), 3);
        assert_eq!(syms[0].name, "MAX");
        assert_eq!(syms[0].category, SymbolCategory::Variable);
        assert_eq!(syms[1].name, "foo");
        assert_eq!(syms[1].category, SymbolCategory::Function);
        assert_eq!(syms[2].name, "DEBUG");
        assert_eq!(syms[2].category, SymbolCategory::Variable);
    }

    // --- Type alias tests ---

    #[test]
    fn parse_type_alias() {
        let syms = parse("type Vector = list[float]\n");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "Vector");
        assert_eq!(syms[0].category, SymbolCategory::Type);
        assert_eq!(syms[0].label, "type");
        assert_eq!(syms[0].id, "test.py::Vector");
    }

    #[test]
    fn parse_type_alias_with_functions() {
        let syms = parse("type ID = int\n\ndef process(x: ID) -> None:\n    pass\n");
        assert_eq!(syms.len(), 2);
        assert_eq!(syms[0].name, "ID");
        assert_eq!(syms[0].category, SymbolCategory::Type);
        assert_eq!(syms[1].name, "process");
        assert_eq!(syms[1].category, SymbolCategory::Function);
    }

    // --- Control flow recursion tests ---

    #[test]
    fn parse_function_inside_if() {
        let syms = parse("if True:\n    def foo():\n        pass\n");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "foo");
        assert_eq!(syms[0].category, SymbolCategory::Function);
    }

    #[test]
    fn parse_function_inside_if_else() {
        let syms = parse(
            "if sys.platform == 'win32':\n    def init():\n        pass\nelse:\n    def init():\n        pass\n",
        );
        assert_eq!(syms.len(), 2);
        assert_eq!(syms[0].name, "init");
        assert_eq!(syms[1].name, "init");
    }

    #[test]
    fn parse_function_inside_elif() {
        let syms = parse(
            "if False:\n    pass\nelif True:\n    def handler():\n        pass\n",
        );
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "handler");
    }

    #[test]
    fn parse_class_inside_try() {
        let syms = parse(
            "try:\n    class Foo:\n        pass\nexcept Exception:\n    class FallbackFoo:\n        pass\n",
        );
        assert_eq!(syms.len(), 2);
        assert_eq!(syms[0].name, "Foo");
        assert_eq!(syms[0].category, SymbolCategory::Type);
        assert_eq!(syms[1].name, "FallbackFoo");
        assert_eq!(syms[1].category, SymbolCategory::Type);
    }

    #[test]
    fn parse_function_inside_try_finally() {
        let syms = parse(
            "try:\n    def setup():\n        pass\nfinally:\n    def cleanup():\n        pass\n",
        );
        assert_eq!(syms.len(), 2);
        assert_eq!(syms[0].name, "setup");
        assert_eq!(syms[1].name, "cleanup");
    }

    #[test]
    fn parse_function_inside_for() {
        let syms = parse("for i in range(1):\n    def worker():\n        pass\n");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "worker");
    }

    #[test]
    fn parse_function_inside_while() {
        let syms = parse("while True:\n    def loop_body():\n        pass\n    break\n");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "loop_body");
    }

    #[test]
    fn parse_function_inside_with() {
        let syms = parse("with open('f') as f:\n    def process():\n        pass\n");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "process");
    }

    #[test]
    fn parse_decorated_function_inside_if() {
        let syms = parse("if True:\n    @decorator\n    def foo():\n        pass\n");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "foo");
        assert_eq!(syms[0].category, SymbolCategory::Function);
    }

    #[test]
    fn parse_nested_control_flow() {
        let syms = parse(
            "try:\n    if True:\n        def deeply_nested():\n            pass\nexcept:\n    pass\n",
        );
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "deeply_nested");
    }

    #[test]
    fn parse_class_with_methods_inside_if() {
        let syms = parse(
            "if True:\n    class Foo:\n        def method(self):\n            pass\n",
        );
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "Foo");
        assert_eq!(syms[0].category, SymbolCategory::Type);
        assert_eq!(syms[0].children.len(), 1);
        assert_eq!(syms[0].children[0].name, "method");
    }

    // --- is_upper_snake_case unit tests ---

    #[test]
    fn upper_snake_case_detection() {
        assert!(is_upper_snake_case("MAX_SIZE"));
        assert!(is_upper_snake_case("DEBUG"));
        assert!(is_upper_snake_case("HTTP2_TIMEOUT"));
        assert!(is_upper_snake_case("_PRIVATE"));
        assert!(is_upper_snake_case("A"));
        assert!(!is_upper_snake_case(""));
        assert!(!is_upper_snake_case("_"));
        assert!(!is_upper_snake_case("__"));
        assert!(!is_upper_snake_case("myVar"));
        assert!(!is_upper_snake_case("Max_Size"));
        assert!(!is_upper_snake_case("123"));
    }
}
