use std::path::Path;

use color_eyre::eyre::eyre;
use tree_sitter::{Node, Parser};

use crate::symbols::merkle::{compute_merkle_hash, content_hash, estimate_tokens};
use crate::symbols::{FileSymbols, SymbolCategory, SymbolNode};

use super::LanguageParser;

pub struct RustParser {
    _private: (),
}

impl RustParser {
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl LanguageParser for RustParser {
    fn extensions(&self) -> &[&str] {
        &["rs"]
    }

    fn parse_file(&self, path: &Path, source: &str) -> color_eyre::Result<FileSymbols> {
        let mut parser = Parser::new();
        let language = tree_sitter_rust::LANGUAGE;
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

const MOD: SymbolMeta = SymbolMeta { category: SymbolCategory::Module, label: "mod" };
const STRUCT: SymbolMeta = SymbolMeta { category: SymbolCategory::Type, label: "struct" };
const ENUM: SymbolMeta = SymbolMeta { category: SymbolCategory::Type, label: "enum" };
const TRAIT: SymbolMeta = SymbolMeta { category: SymbolCategory::Type, label: "trait" };
const IMPL: SymbolMeta = SymbolMeta { category: SymbolCategory::Implementation, label: "impl" };
const FN: SymbolMeta = SymbolMeta { category: SymbolCategory::Function, label: "fn" };
const CONST: SymbolMeta = SymbolMeta { category: SymbolCategory::Variable, label: "const" };
const STATIC: SymbolMeta = SymbolMeta { category: SymbolCategory::Variable, label: "static" };
const TYPE_ALIAS: SymbolMeta = SymbolMeta { category: SymbolCategory::Type, label: "type" };
const MACRO: SymbolMeta = SymbolMeta { category: SymbolCategory::Macro, label: "macro" };

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
            "function_item" => named_symbol(&child, src, &FN),
            "struct_item" => named_symbol(&child, src, &STRUCT),
            "enum_item" => named_symbol(&child, src, &ENUM),
            "trait_item" => named_symbol(&child, src, &TRAIT),
            "impl_item" => impl_symbol(&child, src),
            "const_item" => named_symbol(&child, src, &CONST),
            "static_item" => named_symbol(&child, src, &STATIC),
            "type_item" => named_symbol(&child, src, &TYPE_ALIAS),
            "macro_definition" => named_symbol(&child, src, &MACRO),
            "mod_item" => named_symbol(&child, src, &MOD),
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

            // Recurse into container types for their children.
            if matches!(meta.category, SymbolCategory::Implementation | SymbolCategory::Module)
                || meta.label == "trait"
            {
                if let Some(body) = child_by_kind(&child, "declaration_list") {
                    extract_body_children(body, src, file_path, path_prefix, &name_path, &mut sym.children);
                }
            }

            out.push(sym);
        }
    }
}

fn extract_body_children(
    body: Node,
    src: &[u8],
    file_path: &Path,
    path_prefix: &str,
    parent_name_path: &str,
    out: &mut Vec<SymbolNode>,
) {
    let mut cursor = body.walk();
    for child in body.children(&mut cursor) {
        let symbol_info = match child.kind() {
            "function_item" => named_symbol(&child, src, &FN),
            "const_item" => named_symbol(&child, src, &CONST),
            "type_item" => named_symbol(&child, src, &TYPE_ALIAS),
            "macro_definition" => named_symbol(&child, src, &MACRO),
            _ => None,
        };

        if let Some((name, meta)) = symbol_info {
            let name_path = format!("{parent_name_path}/{name}");
            let id = format!("{path_prefix}::{name_path}");
            let byte_range = child.byte_range();
            let start_line = child.start_position().row + 1;
            let end_line = child.end_position().row + 1;
            let text = std::str::from_utf8(&src[byte_range.clone()]).unwrap_or("");

            out.push(SymbolNode {
                id,
                name,
                category: meta.category,
                label: meta.label.to_string(),
                file_path: file_path.to_path_buf(),
                byte_range,
                line_range: start_line..end_line,
                content_hash: content_hash(text),
                merkle_hash: [0u8; 32],
                children: Vec::new(),
                estimated_tokens: estimate_tokens(text),
            });
        }
    }
}

/// Extract name from a node that has an `identifier` or `type_identifier` child.
fn named_symbol(node: &Node, src: &[u8], meta: &SymbolMeta) -> Option<(String, SymbolMeta)> {
    let name = find_name(node, src)?;
    Some((name, SymbolMeta { category: meta.category, label: meta.label }))
}

/// Build a descriptive name for `impl` blocks: "Foo" or "Trait for Foo".
/// The label "impl" is provided separately, so we don't include it in the name.
fn impl_symbol(node: &Node, src: &[u8]) -> Option<(String, SymbolMeta)> {
    let mut parts = Vec::new();
    let mut cursor = node.walk();

    for child in node.children(&mut cursor) {
        match child.kind() {
            "type_identifier" | "scoped_type_identifier" | "generic_type" => {
                if let Ok(text) = child.utf8_text(src) {
                    parts.push(text.to_string());
                }
            }
            "for" => {
                parts.push("for".to_string());
            }
            // Stop once we hit the body.
            "declaration_list" => break,
            _ => {}
        }
    }

    if parts.is_empty() {
        return None;
    }

    Some((parts.join(" "), IMPL))
}

/// Find the first `identifier` or `type_identifier` child and return its text.
fn find_name(node: &Node, src: &[u8]) -> Option<String> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "identifier" || child.kind() == "type_identifier" {
            return child.utf8_text(src).ok().map(|s| s.to_string());
        }
    }
    None
}

fn child_by_kind<'a>(node: &'a Node<'a>, kind: &str) -> Option<Node<'a>> {
    let mut cursor = node.walk();
    let result = node.children(&mut cursor).find(|c| c.kind() == kind);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::LanguageParser;
    use std::path::Path;

    fn parse(src: &str) -> Vec<SymbolNode> {
        let parser = RustParser::new();
        let file = parser.parse_file(Path::new("test.rs"), src).unwrap();
        file.symbols
    }

    #[test]
    fn parse_function() {
        let syms = parse("fn foo() {}");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "foo");
        assert_eq!(syms[0].category, SymbolCategory::Function);
    }

    #[test]
    fn parse_struct_with_impl() {
        let syms = parse(
            "struct Point { x: i32 }\nimpl Point {\n    fn new() -> Self { Self { x: 0 } }\n}",
        );
        assert_eq!(syms.len(), 2);
        assert_eq!(syms[0].name, "Point");
        assert_eq!(syms[0].category, SymbolCategory::Type);
        assert_eq!(syms[1].name, "Point");
        assert_eq!(syms[1].category, SymbolCategory::Implementation);
        assert_eq!(syms[1].children.len(), 1);
        assert_eq!(syms[1].children[0].name, "new");
    }

    #[test]
    fn parse_nested_module() {
        let syms = parse("mod inner {\n    fn bar() {}\n}");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "inner");
        assert_eq!(syms[0].category, SymbolCategory::Module);
        assert_eq!(syms[0].children.len(), 1);
        assert_eq!(syms[0].children[0].name, "bar");
    }

    #[test]
    fn parse_empty_file() {
        let syms = parse("");
        assert!(syms.is_empty());
    }
}
