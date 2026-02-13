use std::path::Path;

use color_eyre::eyre::eyre;
use tree_sitter::{Node, Parser};

use crate::symbols::merkle::{compute_merkle_hash, content_hash, estimate_tokens};
use crate::symbols::{FileSymbols, SymbolCategory, SymbolNode};

use super::LanguageParser;

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

/// Symbol metadata: category and display label
struct SymbolMeta {
    category: SymbolCategory,
    label: &'static str,
}

const CLASS: SymbolMeta = SymbolMeta { category: SymbolCategory::Type, label: "class" };
const DEF: SymbolMeta = SymbolMeta { category: SymbolCategory::Function, label: "def" };

/// Walk top-level children of a Python module node and extract symbols.
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

/// Handle decorated definitions (@decorator followed by def/class).
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

/// Extract the name from a function_definition or class_definition node.
fn child_name(node: &Node, src: &[u8]) -> Option<String> {
    node.child_by_field_name("name")?
        .utf8_text(src)
        .ok()
        .map(|s| s.to_string())
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
}
