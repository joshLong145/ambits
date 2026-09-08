use std::path::{Path, PathBuf};
use std::sync::Arc;

use color_eyre::eyre::eyre;
use tree_sitter::{Node, Parser};

use super::SymbolMeta;
use crate::symbols::merkle::{compute_merkle_hash, content_hash, estimate_tokens};
use crate::symbols::{FileSymbols, NameInterner, SymbolCategory, SymbolNode};

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

    fn language(&self) -> tree_sitter::Language {
        tree_sitter_rust::LANGUAGE.into()
    }

    fn tags_query(&self) -> &'static str {
        tree_sitter_rust::TAGS_QUERY
    }

    /// Rust calls things through paths constantly — `crate::journal::foo()`,
    /// `super::helper()`, `Type::new()` — and the shipped query captures none
    /// of them, because it only matches a bare `(identifier)` in function
    /// position. Testing against this repo, that missed both call sites of
    /// `centered_rect` and one of two for `hash_hex`.
    fn tags_supplement(&self) -> &'static str {
        r#"
        (call_expression
            function: (scoped_identifier
                name: (identifier) @name)) @reference.call

        (call_expression
            function: (generic_function
                function: (identifier) @name)) @reference.call

        (call_expression
            function: (generic_function
                function: (scoped_identifier
                    name: (identifier) @name))) @reference.call
        "#
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
        let file_path_arc = Arc::new(path.to_path_buf());
        let names = NameInterner::new();

        extract_symbols(root, src, &file_path_arc, &names, &path_prefix, "", &mut symbols);

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
    file_path: &Arc<PathBuf>,
    names: &NameInterner,
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
                name: names.intern(&name),
                category: meta.category,
                label: meta.label,
                file_path: Arc::clone(file_path),
                byte_range: byte_range.start as u32..byte_range.end as u32,
                line_range: start_line as u32..end_line as u32,
                content_hash: content_hash(text),
                merkle_hash: [0u8; 32],
                children: Vec::new(),
                estimated_tokens: estimate_tokens(text) as u32,
            };

            // Recurse into container types for their children.
            if matches!(meta.category, SymbolCategory::Implementation | SymbolCategory::Module)
                || meta.label == "trait"
            {
                if let Some(body) = child_by_kind(&child, "declaration_list") {
                    // Members hang off the type, not the impl block, so the
                    // `impl ` added above is dropped for their prefix. Trait
                    // impls keep their full `Trait for Type` qualification —
                    // two traits can give one type the same method name, and
                    // only the qualification keeps those apart.
                    let child_prefix = name_path
                        .strip_prefix("impl ")
                        .unwrap_or(&name_path);
                    extract_body_children(body, src, file_path, names, path_prefix, child_prefix, &mut sym.children);
                }
            }

            out.push(sym);
        }
    }
}

fn extract_body_children(
    body: Node,
    src: &[u8],
    file_path: &Arc<PathBuf>,
    names: &NameInterner,
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
                name: names.intern(&name),
                category: meta.category,
                label: meta.label,
                file_path: Arc::clone(file_path),
                byte_range: byte_range.start as u32..byte_range.end as u32,
                line_range: start_line as u32..end_line as u32,
                content_hash: content_hash(text),
                merkle_hash: [0u8; 32],
                children: Vec::new(),
                estimated_tokens: estimate_tokens(text) as u32,
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

    // A trait impl is already distinguishable — `Display for Foo` can collide
    // with nothing. An inherent impl is not: bare `Foo` is the same name the
    // type declaration produces, so `struct Foo` and `impl Foo` end up with
    // one id, one ledger entry, and one coverage number between them. On this
    // repo that conflated 28 pairs, including a 3-line struct with a 259-line
    // impl.
    //
    // Naming it `impl Foo` separates them. Their methods deliberately keep the
    // `Foo/method` path — see the child prefix in `extract_symbols` — because a
    // method belongs to the type, which is how Rust itself writes it:
    // `App::new`, never `impl App::new`.
    let name = parts.join(" ");
    let name = if name.contains(" for ") {
        name
    } else {
        format!("impl {name}")
    };
    Some((name, IMPL))
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
        assert_eq!(syms[0].name.as_ref(), "foo");
        assert_eq!(syms[0].category, SymbolCategory::Function);
    }

    #[test]
    fn parse_struct_with_impl() {
        let syms = parse(
            "struct Point { x: i32 }\nimpl Point {\n    fn new() -> Self { Self { x: 0 } }\n}",
        );
        assert_eq!(syms.len(), 2);
        assert_eq!(syms[0].name.as_ref(), "Point");
        assert_eq!(syms[0].category, SymbolCategory::Type);

        // The impl is named apart from the type it implements, or the two
        // share an id and therefore a single coverage entry.
        assert_eq!(syms[1].name.as_ref(), "impl Point");
        assert_eq!(syms[1].category, SymbolCategory::Implementation);
        assert_ne!(syms[0].id, syms[1].id, "a type and its impl are distinct");

        // Its members still hang off the type, the way Rust names them.
        assert_eq!(syms[1].children.len(), 1);
        assert_eq!(syms[1].children[0].name.as_ref(), "new");
        assert!(
            syms[1].children[0].id.ends_with("::Point/new"),
            "got {}",
            syms[1].children[0].id
        );
    }

    /// A trait impl was never ambiguous, and its members must keep the full
    /// qualification: two traits can give one type the same method name, and
    /// only `Trait for Type/method` keeps those apart.
    #[test]
    fn a_trait_impl_keeps_its_qualified_name() {
        let syms = parse("struct P;
impl Display for P {
    fn fmt(&self) {}
}");
        let imp = syms.iter().find(|s| s.category == SymbolCategory::Implementation).unwrap();
        assert_eq!(imp.name.as_ref(), "Display for P");
        assert!(imp.children[0].id.ends_with("::Display for P/fmt"));
    }

    #[test]
    fn parse_nested_module() {
        let syms = parse("mod inner {\n    fn bar() {}\n}");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name.as_ref(), "inner");
        assert_eq!(syms[0].category, SymbolCategory::Module);
        assert_eq!(syms[0].children.len(), 1);
        assert_eq!(syms[0].children[0].name.as_ref(), "bar");
    }

    #[test]
    fn parse_empty_file() {
        let syms = parse("");
        assert!(syms.is_empty());
    }
}

impl Default for RustParser {
    fn default() -> Self {
        Self::new()
    }
}
