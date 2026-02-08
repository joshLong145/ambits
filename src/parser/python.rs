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
