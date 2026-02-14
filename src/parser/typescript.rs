//! TypeScript symbol extractor using tree-sitter.
//!
//! This module parses `.ts` files into a hierarchical [`SymbolNode`] tree that the
//! ambits coverage system uses to track which parts of a codebase have been reviewed.
//!
//! ## How it works
//!
//! 1. The source is fed to tree-sitter-typescript which produces a concrete syntax
//!    tree (CST).
//! 2. [`extract_symbols`] walks the top-level children of each node, unwrapping
//!    `export_statement` wrappers so that `export function foo()` is treated the same
//!    as `function foo()`.
//! 3. Each recognized node kind is dispatched to an emitter function that builds a
//!    [`SymbolNode`] (and, for container types like classes, recurses into members).
//! 4. After the full tree is built, Merkle hashes are computed bottom-up so that
//!    content changes propagate to parent symbols.
//!
//! ## Supported TypeScript constructs
//!
//! | Construct                                  | Category | Label              |
//! |--------------------------------------------|----------|--------------------|
//! | `function`, `async function`, `function*`  | Function | `"function"`       |
//! | `const f = () => {}` / `function(){}`      | Function | `"function"`       |
//! | `class`                                    | Type     | `"class"`          |
//! | `abstract class`                           | Type     | `"abstract class"` |
//! | `interface`                                | Type     | `"interface"`      |
//! | `type Alias = ...`                         | Type     | `"type"`           |
//! | `enum` / `const enum`                      | Type     | `"enum"`           |
//! | `namespace` / `module`                     | Module   | `"namespace"`      |
//! | class methods / interface methods          | Function | `"method"`         |
//! | class getters / setters                    | Function | `"get"` / `"set"`  |
//! | class properties / interface props         | Variable | `"property"`       |
//! | `declare ...`                              | Variable | `"declare"`        |

use std::path::Path;

use color_eyre::eyre::eyre;
use tree_sitter::{Node, Parser};

use crate::symbols::merkle::{compute_merkle_hash, content_hash, estimate_tokens};
use crate::symbols::{FileSymbols, SymbolCategory, SymbolNode};

use super::LanguageParser;

/// Parser for TypeScript (`.ts`) source files.
///
/// Uses the tree-sitter-typescript grammar to produce a CST, then extracts
/// a simplified symbol tree that ambits uses for coverage tracking.
pub struct TypescriptParser {
    _private: (),
}

impl TypescriptParser {
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl LanguageParser for TypescriptParser {
    fn extensions(&self) -> &[&str] {
        &["ts"]
    }

    fn parse_file(&self, path: &Path, source: &str) -> color_eyre::Result<FileSymbols> {
        let mut parser = Parser::new();
        let language = tree_sitter_typescript::LANGUAGE_TYPESCRIPT;
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
// Variable, Module) with a human-readable `label` that appears in the UI.
// These are referenced by the emitter functions below to avoid repeating the
// mapping logic at every call site.

/// Pairs a [`SymbolCategory`] with a display label for use in [`SymbolNode`].
struct SymbolMeta {
    category: SymbolCategory,
    label: &'static str,
}

// -- Top-level declarations -------------------------------------------------
const FN: SymbolMeta = SymbolMeta { category: SymbolCategory::Function, label: "function" };
const CLASS: SymbolMeta = SymbolMeta { category: SymbolCategory::Type, label: "class" };
const ABSTRACT_CLASS: SymbolMeta = SymbolMeta { category: SymbolCategory::Type, label: "abstract class" };
const IFACE: SymbolMeta = SymbolMeta { category: SymbolCategory::Type, label: "interface" };
const TYPE: SymbolMeta = SymbolMeta { category: SymbolCategory::Type, label: "type" };
const ENUM: SymbolMeta = SymbolMeta { category: SymbolCategory::Type, label: "enum" };
const NS: SymbolMeta = SymbolMeta { category: SymbolCategory::Module, label: "namespace" };

// -- Class / interface members ----------------------------------------------
const METHOD: SymbolMeta = SymbolMeta { category: SymbolCategory::Function, label: "method" };
const PROP: SymbolMeta = SymbolMeta { category: SymbolCategory::Variable, label: "property" };
const GET: SymbolMeta = SymbolMeta { category: SymbolCategory::Function, label: "get" };
const SET: SymbolMeta = SymbolMeta { category: SymbolCategory::Function, label: "set" };

// -- Ambient (declare) ------------------------------------------------------
const DECLARE: SymbolMeta = SymbolMeta { category: SymbolCategory::Variable, label: "declare" };

// ---------------------------------------------------------------------------
// Core symbol extraction
// ---------------------------------------------------------------------------

/// Recursively walk the children of `node` and extract recognized TypeScript symbols.
///
/// This is the main dispatch loop. For each child it:
/// 1. Unwraps `export_statement` - peels off the `export` wrapper to reach
///    the inner declaration (e.g. `export class Foo {}` -> `class_declaration`).
///    When decorators are attached to the export, the byte range is widened to
///    include them.
/// 2. Unwraps `expression_statement` - tree-sitter-typescript sometimes wraps
///    bare `namespace X {}` in an expression statement; we reach inside to find it.
/// 3. Dispatches by node kind to the appropriate emitter or builder.
///
/// The function is called at the top level (with `root_node`) and recursively by
/// `emit_namespace` to handle nested declarations inside `namespace` blocks.
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
        // Unwrap export_statement: extract the inner declaration and adjust the
        // byte range when decorators are present (decorators attach to the
        // export_statement node, not to the inner declaration).
        let (target, range_override) = if child.kind() == "export_statement" {
            match child.child_by_field_name("declaration") {
                Some(inner) => {
                    let range = if has_child_kind(&child, "decorator") {
                        Some(child.byte_range())
                    } else {
                        None
                    };
                    (inner, range)
                }
                None => continue, // re-export like `export { foo }` - skip
            }
        } else if child.kind() == "expression_statement" {
            // tree-sitter-typescript wraps bare `namespace X {}` in expression_statement.
            let mut inner_cursor = child.walk();
            for inner in child.children(&mut inner_cursor) {
                if inner.kind() == "internal_module" {
                    emit_namespace(&inner, src, file_path, path_prefix, parent_name_path, out);
                }
            }
            continue;
        } else {
            (child, None)
        };

        let byte_range = range_override.unwrap_or_else(|| target.byte_range());

        match target.kind() {
            // `function foo()` or `function* gen()` - leaf symbol, no children.
            "function_declaration" | "generator_function_declaration" => {
                if let Some(sym) = build_named_symbol(&target, src, file_path, path_prefix, parent_name_path, &FN, byte_range) {
                    out.push(sym);
                }
            }
            // `class Foo { ... }` - container, recurse into class_body for members.
            "class_declaration" => {
                emit_class(&target, src, file_path, path_prefix, parent_name_path, &CLASS, byte_range, out);
            }
            // `abstract class Base { ... }` - same as class but different label.
            "abstract_class_declaration" => {
                emit_class(&target, src, file_path, path_prefix, parent_name_path, &ABSTRACT_CLASS, byte_range, out);
            }
            // `interface Config { ... }` - container, recurse into interface_body.
            "interface_declaration" => {
                emit_interface(&target, src, file_path, path_prefix, parent_name_path, byte_range, out);
            }
            // `type Alias = ...` - leaf symbol.
            "type_alias_declaration" => {
                if let Some(sym) = build_named_symbol(&target, src, file_path, path_prefix, parent_name_path, &TYPE, byte_range) {
                    out.push(sym);
                }
            }
            // `enum Status { ... }` - leaf (we don't extract enum members).
            "enum_declaration" => {
                if let Some(sym) = build_named_symbol(&target, src, file_path, path_prefix, parent_name_path, &ENUM, byte_range) {
                    out.push(sym);
                }
            }
            // `namespace N { ... }` / `module M { ... }` - container, recurse.
            "internal_module" | "module" => {
                emit_namespace(&target, src, file_path, path_prefix, parent_name_path, out);
            }
            // `const foo = () => {}` or `let bar = function() {}` - detect arrow/fn expressions.
            "lexical_declaration" | "variable_declaration" => {
                extract_arrow_fns(&target, src, file_path, path_prefix, parent_name_path, out);
            }
            // `declare function ...`, `declare class ...`, `declare const ...`, etc.
            "ambient_declaration" => {
                extract_ambient(&target, src, file_path, path_prefix, parent_name_path, out);
            }
            // Imports, comments, expression statements, etc. - ignored.
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Leaf / special-case extractors
// ---------------------------------------------------------------------------

/// Extract arrow functions and function expressions from variable declarations.
///
/// Recognizes patterns like:
/// - `const foo = () => { ... }`
/// - `let bar = function() { ... }`
///
/// Plain value bindings (`const x = 42`) are intentionally ignored - they are
/// not considered "symbols" for coverage purposes.
///
/// The byte range used is the full declaration (including `const`/`let`), not
/// just the arrow function body, so the coverage span is accurate.
fn extract_arrow_fns(
    node: &Node,
    src: &[u8],
    file_path: &Path,
    path_prefix: &str,
    parent_name_path: &str,
    out: &mut Vec<SymbolNode>,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() != "variable_declarator" {
            continue;
        }

        // Check if the initializer is an arrow function or function expression.
        let value = match child.child_by_field_name("value") {
            Some(v) => v,
            None => continue,
        };

        if value.kind() != "arrow_function" && value.kind() != "function_expression" {
            continue;
        }

        let name = match child.child_by_field_name("name") {
            Some(n) => match n.utf8_text(src) {
                Ok(s) => s.to_string(),
                Err(_) => continue,
            },
            None => continue,
        };

        // Use the full declaration range (includes const/let keyword).
        let byte_range = node.byte_range();
        out.push(make_symbol(name, &FN, node, byte_range, src, file_path, path_prefix, parent_name_path, Vec::new()));
    }
}

/// Extract the inner declaration from a `declare ...` (ambient) statement.
///
/// Ambient declarations tell the compiler about shapes that exist at runtime
/// but aren't defined in this file (e.g. `declare function require(...)`).
/// We emit a single symbol per `ambient_declaration` node, using the
/// [`DECLARE`] metadata so the UI can distinguish them.
///
/// For `declare const/let/var`, we delegate to [`extract_ambient_vars`] since
/// a single declaration can bind multiple names.
fn extract_ambient(
    node: &Node,
    src: &[u8],
    file_path: &Path,
    path_prefix: &str,
    parent_name_path: &str,
    out: &mut Vec<SymbolNode>,
) {
    let ambient_range = node.byte_range();

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        let name = match child.kind() {
            "function_signature" | "class_declaration" | "abstract_class_declaration"
            | "interface_declaration" | "enum_declaration" | "type_alias_declaration"
            | "internal_module" | "module" => child_name(&child, src),
            "lexical_declaration" | "variable_declaration" => {
                // Extract variable names from declare const/let/var.
                extract_ambient_vars(&child, src, file_path, path_prefix, parent_name_path, &ambient_range, out);
                None
            }
            _ => None,
        };

        if let Some(name) = name {
            out.push(make_symbol(name, &DECLARE, node, ambient_range, src, file_path, path_prefix, parent_name_path, Vec::new()));
            return; // One symbol per ambient_declaration.
        }
    }
}

/// Extract variable names from `declare const x: T` / `declare let x: T`.
fn extract_ambient_vars(
    node: &Node,
    src: &[u8],
    file_path: &Path,
    path_prefix: &str,
    parent_name_path: &str,
    ambient_range: &std::ops::Range<usize>,
    out: &mut Vec<SymbolNode>,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() != "variable_declarator" {
            continue;
        }

        let name = match child.child_by_field_name("name") {
            Some(n) => match n.utf8_text(src) {
                Ok(s) => s.to_string(),
                Err(_) => continue,
            },
            None => continue,
        };

        out.push(make_symbol(name, &DECLARE, &child, ambient_range.clone(), src, file_path, path_prefix, parent_name_path, Vec::new()));
    }
}

// ---------------------------------------------------------------------------
// Container emitters (class, interface, namespace)
// ---------------------------------------------------------------------------

/// Emit a class (or abstract class) symbol and recurse into `class_body` for members.
///
/// The resulting [`SymbolNode`] will have `children` populated with methods,
/// properties, getters, and setters found inside the class body.
fn emit_class(
    node: &Node,
    src: &[u8],
    file_path: &Path,
    path_prefix: &str,
    parent_name_path: &str,
    meta: &SymbolMeta,
    byte_range: std::ops::Range<usize>,
    out: &mut Vec<SymbolNode>,
) {
    let name = match child_name(node, src) {
        Some(n) => n,
        None => return,
    };

    let name_path = if parent_name_path.is_empty() {
        name.clone()
    } else {
        format!("{parent_name_path}/{name}")
    };

    let mut children = Vec::new();
    if let Some(body) = child_by_kind(node, "class_body") {
        extract_members(body, src, file_path, path_prefix, &name_path, &mut children);
    }

    out.push(make_symbol(name, meta, node, byte_range, src, file_path, path_prefix, parent_name_path, children));
}

/// Extract members from a `class_body` or `interface_body` node.
///
/// Recognizes:
/// - `method_definition` - regular methods, plus getters (`get`) and setters (`set`)
///   which are distinguished by checking for a `get`/`set` keyword child node.
/// - `public_field_definition` / `property_signature` - class fields and
///   interface properties (e.g. `host: string`).
/// - `abstract_method_signature` / `method_signature` - abstract and
///   interface method declarations without bodies.
fn extract_members(
    body: Node,
    src: &[u8],
    file_path: &Path,
    path_prefix: &str,
    parent_name_path: &str,
    out: &mut Vec<SymbolNode>,
) {
    let mut cursor = body.walk();
    for child in body.children(&mut cursor) {
        let (name, meta) = match child.kind() {
            "method_definition" => {
                let name = match child_name(&child, src) {
                    Some(n) => n,
                    None => continue,
                };
                let meta = if has_child_kind(&child, "get") {
                    &GET
                } else if has_child_kind(&child, "set") {
                    &SET
                } else {
                    &METHOD
                };
                (name, meta)
            }
            "public_field_definition" | "property_signature" => {
                match child_name(&child, src) {
                    Some(n) => (n, &PROP),
                    None => continue,
                }
            }
            "abstract_method_signature" | "method_signature" => {
                match child_name(&child, src) {
                    Some(n) => (n, &METHOD),
                    None => continue,
                }
            }
            _ => continue,
        };

        let byte_range = child.byte_range();
        out.push(make_symbol(name, meta, &child, byte_range, src, file_path, path_prefix, parent_name_path, Vec::new()));
    }
}

/// Emit an interface symbol and recurse into `interface_body` for members.
///
/// Works identically to [`emit_class`] but looks for `interface_body` instead
/// of `class_body`, and always uses the [`IFACE`] metadata.
fn emit_interface(
    node: &Node,
    src: &[u8],
    file_path: &Path,
    path_prefix: &str,
    parent_name_path: &str,
    byte_range: std::ops::Range<usize>,
    out: &mut Vec<SymbolNode>,
) {
    let name = match child_name(node, src) {
        Some(n) => n,
        None => return,
    };

    let name_path = if parent_name_path.is_empty() {
        name.clone()
    } else {
        format!("{parent_name_path}/{name}")
    };

    let mut children = Vec::new();
    if let Some(body) = child_by_kind(node, "interface_body") {
        extract_members(body, src, file_path, path_prefix, &name_path, &mut children);
    }

    out.push(make_symbol(name, &IFACE, node, byte_range, src, file_path, path_prefix, parent_name_path, children));
}

/// Emit a `namespace`/`module` symbol and recurse into `statement_block` for nested declarations.
///
/// Unlike classes and interfaces, namespaces can contain arbitrary top-level
/// declarations (functions, classes, other namespaces, etc.), so we call back
/// into [`extract_symbols`] rather than [`extract_members`].
fn emit_namespace(
    node: &Node,
    src: &[u8],
    file_path: &Path,
    path_prefix: &str,
    parent_name_path: &str,
    out: &mut Vec<SymbolNode>,
) {
    let name = match child_name(node, src) {
        Some(n) => n,
        None => return,
    };

    let name_path = if parent_name_path.is_empty() {
        name.clone()
    } else {
        format!("{parent_name_path}/{name}")
    };

    let mut children = Vec::new();
    if let Some(body) = child_by_kind(node, "statement_block") {
        extract_symbols(body, src, file_path, path_prefix, &name_path, &mut children);
    }

    let byte_range = node.byte_range();
    out.push(make_symbol(name, &NS, node, byte_range, src, file_path, path_prefix, parent_name_path, children));
}

// ---------------------------------------------------------------------------
// Symbol construction helpers
// ---------------------------------------------------------------------------

/// Construct a [`SymbolNode`] with all derived fields populated.
///
/// Derived fields:
/// - `name_path` - hierarchical path like `"ClassName/methodName"`, built by
///   joining `parent_name_path` with `name`.
/// - `id` - globally unique id: `"<file_path>::<name_path>"`.
/// - `line_range` - 1-based inclusive line range from the tree-sitter node position.
/// - `content_hash` - SHA-256 of the raw source text covered by `byte_range`.
/// - `merkle_hash` - initialized to zeroes here; filled in by [`compute_merkle_hash`]
///   after the full tree is assembled.
/// - `estimated_tokens` - rough LLM token count for the source text.
fn make_symbol(
    name: String,
    meta: &SymbolMeta,
    line_node: &Node,
    byte_range: std::ops::Range<usize>,
    src: &[u8],
    file_path: &Path,
    path_prefix: &str,
    parent_name_path: &str,
    children: Vec<SymbolNode>,
) -> SymbolNode {
    let name_path = if parent_name_path.is_empty() {
        name.clone()
    } else {
        format!("{parent_name_path}/{name}")
    };
    let id = format!("{path_prefix}::{name_path}");
    let start_line = line_node.start_position().row + 1;
    let end_line = line_node.end_position().row + 1;
    let text = std::str::from_utf8(&src[byte_range.clone()]).unwrap_or("");

    SymbolNode {
        id,
        name,
        category: meta.category,
        label: meta.label.to_string(),
        file_path: file_path.to_path_buf(),
        byte_range,
        line_range: start_line..end_line,
        content_hash: content_hash(text),
        merkle_hash: [0u8; 32],
        children,
        estimated_tokens: estimate_tokens(text),
    }
}

/// Convenience wrapper: build a leaf symbol (no children) from a named node.
///
/// Returns `None` if the node has no extractable name (see [`child_name`]).
fn build_named_symbol(
    node: &Node,
    src: &[u8],
    file_path: &Path,
    path_prefix: &str,
    parent_name_path: &str,
    meta: &SymbolMeta,
    byte_range: std::ops::Range<usize>,
) -> Option<SymbolNode> {
    let name = child_name(node, src)?;
    Some(make_symbol(name, meta, node, byte_range, src, file_path, path_prefix, parent_name_path, Vec::new()))
}

// ---------------------------------------------------------------------------
// Tree-sitter node traversal helpers
// ---------------------------------------------------------------------------

/// Extract the name from a tree-sitter node.
///
/// Strategy:
/// 1. Try the `"name"` field first - most TypeScript declaration nodes
///    (class, function, interface, etc.) expose their identifier this way.
/// 2. Fall back to scanning direct children for the first `identifier` or
///    `type_identifier` node. This covers edge cases where the grammar
///    doesn't use a named field.
fn child_name(node: &Node, src: &[u8]) -> Option<String> {
    if let Some(name_node) = node.child_by_field_name("name") {
        return name_node.utf8_text(src).ok().map(|s| s.to_string());
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "identifier" || child.kind() == "type_identifier" {
            return child.utf8_text(src).ok().map(|s| s.to_string());
        }
    }
    None
}

/// Find the first direct child of `node` whose `kind()` matches `kind`.
///
/// NOTE: The `let result = ...; result` pattern is intentional - it ensures the
/// temporary iterator is dropped before `cursor`, satisfying the borrow checker.
fn child_by_kind<'a>(node: &'a Node<'a>, kind: &str) -> Option<Node<'a>> {
    let mut cursor = node.walk();
    let result = node.children(&mut cursor).find(|c| c.kind() == kind);
    result
}

/// Check whether `node` has any direct child whose `kind()` matches `kind`.
///
/// Used to detect `get`/`set` keyword children inside `method_definition`,
/// and `decorator` children inside `export_statement`.
fn has_child_kind(node: &Node, kind: &str) -> bool {
    let mut cursor = node.walk();
    let result = node.children(&mut cursor).any(|c| c.kind() == kind);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: parse TypeScript source and return symbols.
    fn parse(source: &str) -> Vec<SymbolNode> {
        let parser = TypescriptParser::new();
        let result = parser
            .parse_file(Path::new("test.ts"), source)
            .expect("parse failed");
        result.symbols
    }

    /// Helper: find a symbol by name in a flat list.
    fn find<'a>(symbols: &'a [SymbolNode], name: &str) -> &'a SymbolNode {
        symbols
            .iter()
            .find(|s| s.name == name)
            .unwrap_or_else(|| panic!("symbol '{}' not found", name))
    }

    // ---------------------------------------------------------------
    // Function declarations
    // ---------------------------------------------------------------

    #[test]
    fn function_declaration() {
        let syms = parse("function greet(name: string): string { return name; }");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "greet");
        assert_eq!(syms[0].label, "function");
        assert_eq!(syms[0].category, SymbolCategory::Function);
    }

    #[test]
    fn generator_function() {
        let syms = parse("function* gen() { yield 1; }");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "gen");
        assert_eq!(syms[0].label, "function");
        assert_eq!(syms[0].category, SymbolCategory::Function);
    }

    #[test]
    fn async_function() {
        let syms = parse("async function fetchData(): Promise<void> {}");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "fetchData");
        assert_eq!(syms[0].label, "function");
    }

    // ---------------------------------------------------------------
    // Arrow function detection
    // ---------------------------------------------------------------

    #[test]
    fn arrow_function_const() {
        let syms = parse("const handler = () => {};");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "handler");
        assert_eq!(syms[0].label, "function");
        assert_eq!(syms[0].category, SymbolCategory::Function);
    }

    #[test]
    fn arrow_function_with_body() {
        let syms = parse("const process = (data: string) => {\n  return data;\n};");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "process");
        assert_eq!(syms[0].label, "function");
    }

    #[test]
    fn function_expression() {
        let syms = parse("const handler = function() {};");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "handler");
        assert_eq!(syms[0].label, "function");
        assert_eq!(syms[0].category, SymbolCategory::Function);
    }

    #[test]
    fn arrow_function_let() {
        let syms = parse("let handler = () => {};");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "handler");
        assert_eq!(syms[0].label, "function");
    }

    #[test]
    fn non_function_const_ignored() {
        let syms = parse("const x = 42;");
        assert_eq!(syms.len(), 0, "non-function const should not be detected");
    }

    #[test]
    fn non_function_object_ignored() {
        let syms = parse("const config = { key: 'value' };");
        assert_eq!(syms.len(), 0, "object literal const should not be detected");
    }

    #[test]
    fn non_function_array_ignored() {
        let syms = parse("const items = [1, 2, 3];");
        assert_eq!(syms.len(), 0, "array const should not be detected");
    }

    // ---------------------------------------------------------------
    // Class declarations
    // ---------------------------------------------------------------

    #[test]
    fn class_declaration() {
        let syms = parse("class Foo {}");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "Foo");
        assert_eq!(syms[0].label, "class");
        assert_eq!(syms[0].category, SymbolCategory::Type);
    }

    #[test]
    fn class_with_methods() {
        let syms = parse(
            "class Service {
                doWork(): void {}
                process(data: string): string { return data; }
            }",
        );
        assert_eq!(syms.len(), 1);
        let cls = &syms[0];
        assert_eq!(cls.name, "Service");
        assert_eq!(cls.children.len(), 2);
        assert_eq!(cls.children[0].name, "doWork");
        assert_eq!(cls.children[0].label, "method");
        assert_eq!(cls.children[1].name, "process");
        assert_eq!(cls.children[1].label, "method");
    }

    #[test]
    fn class_with_properties() {
        let syms = parse(
            "class Config {
                host: string;
                port: number;
            }",
        );
        assert_eq!(syms.len(), 1);
        let cls = &syms[0];
        assert_eq!(cls.children.len(), 2);
        assert_eq!(cls.children[0].name, "host");
        assert_eq!(cls.children[0].label, "property");
        assert_eq!(cls.children[0].category, SymbolCategory::Variable);
        assert_eq!(cls.children[1].name, "port");
        assert_eq!(cls.children[1].label, "property");
    }

    #[test]
    fn class_getter_setter() {
        let syms = parse(
            "class Box {
                get value(): number { return this._v; }
                set value(v: number) { this._v = v; }
            }",
        );
        assert_eq!(syms.len(), 1);
        let cls = &syms[0];
        assert_eq!(cls.children.len(), 2);
        assert_eq!(cls.children[0].label, "get");
        assert_eq!(cls.children[0].category, SymbolCategory::Function);
        assert_eq!(cls.children[1].label, "set");
        assert_eq!(cls.children[1].category, SymbolCategory::Function);
    }

    #[test]
    fn class_constructor() {
        let syms = parse(
            "class App {
                constructor(private db: Database) {}
            }",
        );
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].children.len(), 1);
        assert_eq!(syms[0].children[0].name, "constructor");
        assert_eq!(syms[0].children[0].label, "method");
    }

    // ---------------------------------------------------------------
    // Abstract classes
    // ---------------------------------------------------------------

    #[test]
    fn abstract_class() {
        let syms = parse(
            "abstract class Base {
                abstract doWork(): void;
                concrete(): void {}
            }",
        );
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "Base");
        assert_eq!(syms[0].label, "abstract class");
        assert_eq!(syms[0].category, SymbolCategory::Type);
        assert_eq!(syms[0].children.len(), 2);
        assert_eq!(syms[0].children[0].name, "doWork");
        assert_eq!(syms[0].children[0].label, "method");
        assert_eq!(syms[0].children[1].name, "concrete");
        assert_eq!(syms[0].children[1].label, "method");
    }

    // ---------------------------------------------------------------
    // Interfaces
    // ---------------------------------------------------------------

    #[test]
    fn interface_declaration() {
        let syms = parse("interface Foo {}");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "Foo");
        assert_eq!(syms[0].label, "interface");
        assert_eq!(syms[0].category, SymbolCategory::Type);
    }

    #[test]
    fn interface_with_members() {
        let syms = parse(
            "interface Config {
                host: string;
                port: number;
                connect(): void;
                disconnect(force?: boolean): Promise<void>;
            }",
        );
        assert_eq!(syms.len(), 1);
        let iface = &syms[0];
        assert_eq!(iface.children.len(), 4);
        assert_eq!(iface.children[0].name, "host");
        assert_eq!(iface.children[0].label, "property");
        assert_eq!(iface.children[0].category, SymbolCategory::Variable);
        assert_eq!(iface.children[1].name, "port");
        assert_eq!(iface.children[1].label, "property");
        assert_eq!(iface.children[2].name, "connect");
        assert_eq!(iface.children[2].label, "method");
        assert_eq!(iface.children[2].category, SymbolCategory::Function);
        assert_eq!(iface.children[3].name, "disconnect");
        assert_eq!(iface.children[3].label, "method");
    }

    // ---------------------------------------------------------------
    // Type aliases
    // ---------------------------------------------------------------

    #[test]
    fn type_alias() {
        let syms = parse("type UserId = string;");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "UserId");
        assert_eq!(syms[0].label, "type");
        assert_eq!(syms[0].category, SymbolCategory::Type);
    }

    #[test]
    fn generic_type_alias() {
        let syms = parse("type Result<T> = { ok: true; value: T } | { ok: false; error: Error };");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "Result");
        assert_eq!(syms[0].label, "type");
    }

    // ---------------------------------------------------------------
    // Enums
    // ---------------------------------------------------------------

    #[test]
    fn enum_declaration() {
        let syms = parse("enum Status { Active, Inactive, Pending }");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "Status");
        assert_eq!(syms[0].label, "enum");
        assert_eq!(syms[0].category, SymbolCategory::Type);
        assert_eq!(syms[0].children.len(), 0, "enum should not recurse into members");
    }

    #[test]
    fn const_enum() {
        let syms = parse("const enum Direction { Up, Down, Left, Right }");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "Direction");
        assert_eq!(syms[0].label, "enum");
    }

    // ---------------------------------------------------------------
    // Namespaces
    // ---------------------------------------------------------------

    #[test]
    fn namespace_declaration() {
        let syms = parse(
            "namespace Validation {
                export function isValid(s: string): boolean { return s.length > 0; }
            }",
        );
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "Validation");
        assert_eq!(syms[0].label, "namespace");
        assert_eq!(syms[0].category, SymbolCategory::Module);
        // Namespace should recurse into children.
        assert_eq!(syms[0].children.len(), 1);
        assert_eq!(syms[0].children[0].name, "isValid");
        assert_eq!(syms[0].children[0].label, "function");
    }

    #[test]
    fn module_declaration() {
        let syms = parse(
            "module MyModule {
                export class Inner {}
            }",
        );
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "MyModule");
        assert_eq!(syms[0].label, "namespace");
        assert_eq!(syms[0].children.len(), 1);
        assert_eq!(syms[0].children[0].name, "Inner");
        assert_eq!(syms[0].children[0].label, "class");
    }

    // ---------------------------------------------------------------
    // Ambient declarations
    // ---------------------------------------------------------------

    #[test]
    fn declare_function() {
        let syms = parse("declare function require(id: string): any;");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "require");
        assert_eq!(syms[0].label, "declare");
        assert_eq!(syms[0].category, SymbolCategory::Variable);
    }

    #[test]
    fn declare_const() {
        let syms = parse("declare const __dirname: string;");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "__dirname");
        assert_eq!(syms[0].label, "declare");
    }

    #[test]
    fn declare_module() {
        let syms = parse("declare module 'express' { export interface Request {} }");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].label, "declare");
    }

    #[test]
    fn declare_class() {
        let syms = parse("declare class Buffer { constructor(str: string); }");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "Buffer");
        assert_eq!(syms[0].label, "declare");
    }

    // ---------------------------------------------------------------
    // Export wrapping
    // ---------------------------------------------------------------

    #[test]
    fn export_function() {
        let syms = parse("export function greet(): void {}");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "greet");
        assert_eq!(syms[0].label, "function");
    }

    #[test]
    fn export_class() {
        let syms = parse("export class Service {}");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "Service");
        assert_eq!(syms[0].label, "class");
    }

    #[test]
    fn export_interface() {
        let syms = parse("export interface Config { host: string; }");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "Config");
        assert_eq!(syms[0].label, "interface");
        assert_eq!(syms[0].children.len(), 1);
    }

    #[test]
    fn export_type_alias() {
        let syms = parse("export type Id = string;");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "Id");
        assert_eq!(syms[0].label, "type");
    }

    #[test]
    fn export_enum() {
        let syms = parse("export enum Color { Red, Green, Blue }");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "Color");
        assert_eq!(syms[0].label, "enum");
    }

    #[test]
    fn export_arrow_function() {
        let syms = parse("export const handler = () => {};");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "handler");
        assert_eq!(syms[0].label, "function");
        assert_eq!(syms[0].category, SymbolCategory::Function);
    }

    #[test]
    fn export_default_function() {
        let syms = parse("export default function main() {}");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "main");
        assert_eq!(syms[0].label, "function");
    }

    #[test]
    fn export_default_class() {
        let syms = parse("export default class App {}");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name, "App");
        assert_eq!(syms[0].label, "class");
    }

    #[test]
    fn reexport_ignored() {
        let syms = parse("export { foo } from './foo';");
        assert_eq!(syms.len(), 0, "re-exports should not produce symbols");
    }

    #[test]
    fn export_star_ignored() {
        let syms = parse("export * from './utils';");
        assert_eq!(syms.len(), 0, "wildcard re-exports should not produce symbols");
    }

    // ---------------------------------------------------------------
    // Symbol IDs and nesting paths
    // ---------------------------------------------------------------

    #[test]
    fn symbol_ids_use_path_prefix() {
        let syms = parse("function foo() {}");
        assert_eq!(syms[0].id, "test.ts::foo");
    }

    #[test]
    fn nested_symbol_ids() {
        let syms = parse(
            "class Svc {
                run(): void {}
            }",
        );
        assert_eq!(syms[0].id, "test.ts::Svc");
        assert_eq!(syms[0].children[0].id, "test.ts::Svc/run");
    }

    #[test]
    fn namespace_nested_ids() {
        let syms = parse(
            "namespace A {
                export function b() {}
            }",
        );
        assert_eq!(syms[0].id, "test.ts::A");
        assert_eq!(syms[0].children[0].id, "test.ts::A/b");
    }

    // ---------------------------------------------------------------
    // Line ranges
    // ---------------------------------------------------------------

    #[test]
    fn line_ranges_are_correct() {
        let syms = parse(
            "function a() {}\nfunction b() {}\nfunction c() {}",
        );
        assert_eq!(syms.len(), 3);
        assert_eq!(syms[0].line_range, 1..1);
        assert_eq!(syms[1].line_range, 2..2);
        assert_eq!(syms[2].line_range, 3..3);
    }

    #[test]
    fn multiline_class_range() {
        let syms = parse(
            "class Foo {\n  bar(): void {}\n  baz(): void {}\n}",
        );
        assert_eq!(syms[0].line_range, 1..4);
    }

    // ---------------------------------------------------------------
    // Combined / integration
    // ---------------------------------------------------------------

    #[test]
    fn mixed_top_level_symbols() {
        let source = "\
function greet() {}
const handler = () => {};
class Service {}
interface Config {}
type Id = string;
enum Status { A }
namespace Utils { export function help() {} }
declare function require(id: string): any;
";
        let syms = parse(source);
        assert_eq!(syms.len(), 8);
        assert_eq!(find(&syms, "greet").label, "function");
        assert_eq!(find(&syms, "handler").label, "function");
        assert_eq!(find(&syms, "Service").label, "class");
        assert_eq!(find(&syms, "Config").label, "interface");
        assert_eq!(find(&syms, "Id").label, "type");
        assert_eq!(find(&syms, "Status").label, "enum");
        assert_eq!(find(&syms, "Utils").label, "namespace");
        assert_eq!(find(&syms, "require").label, "declare");
    }

    #[test]
    fn extensions_returns_ts() {
        let parser = TypescriptParser::new();
        assert_eq!(parser.extensions(), &["ts"]);
    }

    #[test]
    fn content_hashes_are_nonzero() {
        let syms = parse("function foo() {}");
        assert_ne!(syms[0].content_hash, [0u8; 32]);
    }

    #[test]
    fn merkle_hashes_are_nonzero() {
        let syms = parse("function foo() {}");
        assert_ne!(syms[0].merkle_hash, [0u8; 32]);
    }

    #[test]
    fn estimated_tokens_nonzero() {
        let syms = parse("function foo() { return 42; }");
        assert!(syms[0].estimated_tokens > 0);
    }
}
