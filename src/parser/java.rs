//! Java symbol extractor using tree-sitter.
//!
//! This module parses `.java` files into a hierarchical [`SymbolNode`] tree that the
//! ambits coverage system uses to track which parts of a codebase have been reviewed.
//!
//! ## How it works
//!
//! 1. The source is fed to tree-sitter-java which produces a concrete syntax
//!    tree (CST).
//! 2. [`extract_symbols`] walks the children of the compilation unit and of
//!    every type body, dispatching recognized declarations to builders.
//! 3. Annotations are part of a declaration's `modifiers` child, so a member's
//!    range already includes `@Override` and the like; a Javadoc or other
//!    comment glued above it widens the range as in the other parsers.
//! 4. After the full tree is built, Merkle hashes are computed bottom-up so that
//!    content changes propagate to parent symbols.
//!
//! ## Supported Java constructs
//!
//! | Construct                                        | Category | Label              |
//! |--------------------------------------------------|----------|--------------------|
//! | `class`                                          | Type     | `"class"`          |
//! | `abstract class`                                 | Type     | `"abstract class"` |
//! | `interface`                                      | Type     | `"interface"`      |
//! | `enum`                                           | Type     | `"enum"`           |
//! | `record`                                         | Type     | `"record"`         |
//! | `@interface` (annotation type)                   | Type     | `"@interface"`     |
//! | methods, `default` methods, annotation elements  | Function | `"method"`         |
//! | constructors, compact record constructors        | Function | `"constructor"`    |
//! | fields and interface constants, one per variable | Variable | `"field"`          |
//!
//! Types nest to any depth (`Outer/Inner/method`). Overloads share a name and
//! so an id, which ids already allow; their content hashes tell them apart.
//! Enum constants are not symbols, as Rust's and TypeScript's enum members are
//! not, and nothing inside a method or constructor body is either: a local or
//! anonymous class belongs to the member that declares it.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use color_eyre::eyre::eyre;
use tree_sitter::{Node, Parser};

use super::SymbolMeta;
use crate::symbols::merkle::{compute_merkle_hash, content_hash, estimate_tokens};
use crate::symbols::{FileSymbols, NameInterner, SymbolCategory, SymbolNode};

use super::LanguageParser;

/// Parser for Java (`.java`) source files.
///
/// Uses the tree-sitter-java grammar to produce a CST, then extracts
/// a simplified symbol tree that ambits uses for coverage tracking.
pub struct JavaParser {
    _private: (),
}

impl JavaParser {
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl LanguageParser for JavaParser {
    fn extensions(&self) -> &[&str] {
        &["java"]
    }

    fn language(&self) -> tree_sitter::Language {
        tree_sitter_java::LANGUAGE.into()
    }

    fn tags_query(&self) -> &'static str {
        tree_sitter_java::TAGS_QUERY
    }

    /// The shipped query tags `new Foo()` as `@reference.class`, not a call,
    /// so `callers Foo` would miss every constructor call. A qualified or
    /// generic type is matched down to its last identifier, the class itself.
    fn tags_supplement(&self) -> &'static str {
        r#"
        (object_creation_expression
            type: (type_identifier) @name) @reference.call

        (object_creation_expression
            type: (scoped_type_identifier
                (type_identifier) @name .)) @reference.call

        (object_creation_expression
            type: (generic_type
                (type_identifier) @name)) @reference.call

        (object_creation_expression
            type: (generic_type
                (scoped_type_identifier
                    (type_identifier) @name .))) @reference.call
        "#
    }

    fn parse_file(&self, path: &Path, source: &str) -> color_eyre::Result<FileSymbols> {
        let mut parser = Parser::new();
        let language = tree_sitter_java::LANGUAGE;
        parser
            .set_language(&language.into())
            .map_err(|e| eyre!("Failed to set language: {}", e))?;

        let tree = parser
            .parse(source, None)
            .ok_or_else(|| eyre!("Failed to parse {}", path.display()))?;

        let root = tree.root_node();
        let path_prefix = path.to_string_lossy();
        let mut symbols = Vec::new();
        let file_path_arc = Arc::new(path.to_path_buf());
        let names = NameInterner::new();

        let ctx = FileCtx {
            src: source.as_bytes(),
            file_path: &file_path_arc,
            names: &names,
            path_prefix: &path_prefix,
        };
        extract_symbols(root, &ctx, "", &mut symbols);

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

// -- Type declarations ------------------------------------------------------
const CLASS: SymbolMeta = SymbolMeta {
    category: SymbolCategory::Type,
    label: "class",
};
const ABSTRACT_CLASS: SymbolMeta = SymbolMeta {
    category: SymbolCategory::Type,
    label: "abstract class",
};
const IFACE: SymbolMeta = SymbolMeta {
    category: SymbolCategory::Type,
    label: "interface",
};
const ENUM: SymbolMeta = SymbolMeta {
    category: SymbolCategory::Type,
    label: "enum",
};
const RECORD: SymbolMeta = SymbolMeta {
    category: SymbolCategory::Type,
    label: "record",
};
const ANNOTATION: SymbolMeta = SymbolMeta {
    category: SymbolCategory::Type,
    label: "@interface",
};

// -- Members ----------------------------------------------------------------
const METHOD: SymbolMeta = SymbolMeta {
    category: SymbolCategory::Function,
    label: "method",
};
const CONSTRUCTOR: SymbolMeta = SymbolMeta {
    category: SymbolCategory::Function,
    label: "constructor",
};
const FIELD: SymbolMeta = SymbolMeta {
    category: SymbolCategory::Variable,
    label: "field",
};

/// tree-sitter-java's two comment kinds: `//` is `line_comment`; `/* */` and
/// Javadoc `/** */` are `block_comment`.
const COMMENT_KINDS: &[&str] = &["line_comment", "block_comment"];

/// What stays fixed while walking one file — see the TypeScript parser's
/// `FileCtx`, which this mirrors.
struct FileCtx<'a> {
    src: &'a [u8],
    file_path: &'a Arc<PathBuf>,
    names: &'a NameInterner,
    path_prefix: &'a str,
}

// ---------------------------------------------------------------------------
// Core symbol extraction
// ---------------------------------------------------------------------------

/// Walk the children of `node` — the compilation unit or a type body — and
/// extract the declarations among them.
///
/// Type declarations recurse into their `body` field, whatever its kind
/// (`class_body`, `interface_body`, `enum_body`, `annotation_type_body`).
/// An enum's members sit one level further down, in `enum_body_declarations`
/// after the constants, and are lifted to the enum itself. Package and import
/// declarations, enum constants and comments produce nothing.
fn extract_symbols(node: Node, ctx: &FileCtx, parent_name_path: &str, out: &mut Vec<SymbolNode>) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        let meta = match child.kind() {
            "class_declaration" if is_abstract(&child, ctx.src) => ABSTRACT_CLASS,
            "class_declaration" => CLASS,
            "interface_declaration" => IFACE,
            "enum_declaration" => ENUM,
            "record_declaration" => RECORD,
            "annotation_type_declaration" => ANNOTATION,
            "method_declaration" | "annotation_type_element_declaration" => METHOD,
            "constructor_declaration" | "compact_constructor_declaration" => CONSTRUCTOR,
            "field_declaration" | "constant_declaration" => {
                extract_fields(child, ctx, parent_name_path, out);
                continue;
            }
            "enum_body_declarations" => {
                extract_symbols(child, ctx, parent_name_path, out);
                continue;
            }
            _ => continue,
        };
        let Some(name) = child_name(&child, ctx.src) else {
            continue;
        };

        let mut children = Vec::new();
        if meta.category == SymbolCategory::Type {
            if let Some(body) = child.child_by_field_name("body") {
                extract_symbols(body, ctx, &join(parent_name_path, &name), &mut children);
            }
        }
        out.push(make_symbol(
            &name,
            &meta,
            child,
            ctx,
            parent_name_path,
            children,
        ));
    }
}

/// One symbol per variable a field declaration declares, each spanning the
/// whole declaration — `int width, height;` is a single statement, and
/// splitting it would leave its type and modifiers in neither half. This is
/// the TypeScript parser's treatment of a multi-binding `const`.
fn extract_fields(node: Node, ctx: &FileCtx, parent_name_path: &str, out: &mut Vec<SymbolNode>) {
    let mut cursor = node.walk();
    for declarator in node.children_by_field_name("declarator", &mut cursor) {
        if let Some(name) = child_name(&declarator, ctx.src) {
            out.push(make_symbol(
                &name,
                &FIELD,
                node,
                ctx,
                parent_name_path,
                Vec::new(),
            ));
        }
    }
}

// ---------------------------------------------------------------------------
// Symbol construction helpers
// ---------------------------------------------------------------------------

/// Construct a [`SymbolNode`] spanning `node`, widened to include an unbroken
/// run of comments immediately above it (see `parser::leading_comment_start`).
fn make_symbol(
    name: &str,
    meta: &SymbolMeta,
    node: Node,
    ctx: &FileCtx,
    parent_name_path: &str,
    children: Vec<SymbolNode>,
) -> SymbolNode {
    let id = format!("{}::{}", ctx.path_prefix, join(parent_name_path, name));
    let (start_byte, start_line) = match super::leading_comment_start(node, COMMENT_KINDS) {
        Some(leading) => (leading.start_byte(), leading.start_position().row + 1),
        None => (node.start_byte(), node.start_position().row + 1),
    };
    let end_byte = node.end_byte();
    let end_line = node.end_position().row + 1;
    let text = std::str::from_utf8(&ctx.src[start_byte..end_byte]).unwrap_or("");

    SymbolNode {
        id,
        name: ctx.names.intern(name),
        category: meta.category,
        label: meta.label,
        file_path: Arc::clone(ctx.file_path),
        byte_range: start_byte as u32..end_byte as u32,
        line_range: start_line as u32..end_line as u32,
        head_line: super::head_line(node),
        content_hash: content_hash(text),
        merkle_hash: [0u8; 32],
        children,
        estimated_tokens: estimate_tokens(text) as u32,
    }
}

fn join(parent_name_path: &str, name: &str) -> String {
    if parent_name_path.is_empty() {
        name.to_string()
    } else {
        format!("{parent_name_path}/{name}")
    }
}

// ---------------------------------------------------------------------------
// Tree-sitter node helpers
// ---------------------------------------------------------------------------

/// The identifier in `node`'s `"name"` field. Every declaration this parser
/// recognizes, and every `variable_declarator`, has one; the name excludes
/// type parameters, so `class Box<T>` is `Box`.
fn child_name(node: &Node, src: &[u8]) -> Option<String> {
    node.child_by_field_name("name")?
        .utf8_text(src)
        .ok()
        .map(|s| s.to_string())
}

/// Whether a class declaration's `modifiers` include `abstract`.
fn is_abstract(node: &Node, src: &[u8]) -> bool {
    let mut cursor = node.walk();
    let modifiers = node.children(&mut cursor).find(|c| c.kind() == "modifiers");
    modifiers.is_some_and(|m| {
        let mut inner = m.walk();
        let found = m
            .children(&mut inner)
            .any(|c| c.utf8_text(src) == Ok("abstract"));
        found
    })
}

impl Default for JavaParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(source: &str) -> Vec<SymbolNode> {
        JavaParser::new()
            .parse_file(Path::new("Test.java"), source)
            .expect("parse failed")
            .symbols
    }

    /// `(name, label)` of each child of `sym`, in order.
    fn members(sym: &SymbolNode) -> Vec<(&str, &str)> {
        sym.children
            .iter()
            .map(|c| (c.name.as_ref(), c.label))
            .collect()
    }

    #[test]
    fn extensions() {
        assert_eq!(JavaParser::new().extensions(), &["java"]);
    }

    #[test]
    fn the_registry_picks_this_parser_for_java_files() {
        let registry = crate::parser::ParserRegistry::new();
        let parser = registry
            .parser_for(Path::new("src/Main.java"))
            .expect("a parser for .java");
        assert_eq!(parser.extensions(), &["java"]);
    }

    #[test]
    fn parse_empty_file() {
        assert!(parse("").is_empty());
    }

    #[test]
    fn package_and_imports_are_not_symbols() {
        let syms = parse(
            "package com.example;\n\nimport java.util.List;\nimport static java.lang.Math.max;\n",
        );
        assert!(syms.is_empty());
    }

    // ---------------------------------------------------------------
    // Type declarations
    // ---------------------------------------------------------------

    #[test]
    fn class_declaration() {
        let syms = parse("public class Foo {}");
        assert_eq!(syms.len(), 1);
        assert_eq!(syms[0].name.as_ref(), "Foo");
        assert_eq!(syms[0].label, "class");
        assert_eq!(syms[0].category, SymbolCategory::Type);
        assert_eq!(syms[0].id, "Test.java::Foo");
    }

    #[test]
    fn abstract_class() {
        let syms = parse("public abstract class Base {\n    abstract void run();\n}");
        assert_eq!(syms[0].label, "abstract class");
        assert_eq!(members(&syms[0]), vec![("run", "method")]);
    }

    #[test]
    fn every_kind_of_type_declaration() {
        let syms = parse(
            "class A {}\ninterface B {}\nenum C { X }\nrecord D(int x) {}\n@interface E {}\n",
        );
        let kinds: Vec<(&str, &str)> = syms.iter().map(|s| (s.name.as_ref(), s.label)).collect();
        assert_eq!(
            kinds,
            vec![
                ("A", "class"),
                ("B", "interface"),
                ("C", "enum"),
                ("D", "record"),
                ("E", "@interface")
            ]
        );
        assert!(syms.iter().all(|s| s.category == SymbolCategory::Type));
    }

    #[test]
    fn generic_type_names_exclude_type_parameters() {
        let syms = parse("class Box<T extends Comparable<T>> implements Supplier<T> {\n    <R> R map(Function<T, R> f) { return null; }\n}");
        assert_eq!(syms[0].name.as_ref(), "Box");
        assert_eq!(syms[0].children[0].id, "Test.java::Box/map");
    }

    // ---------------------------------------------------------------
    // Members
    // ---------------------------------------------------------------

    #[test]
    fn class_members() {
        let syms = parse(
            "class Svc {
                private final Db db;
                Svc(Db db) { this.db = db; }
                void run() {}
                static int count() { return 0; }
            }",
        );
        assert_eq!(
            members(&syms[0]),
            vec![
                ("db", "field"),
                ("Svc", "constructor"),
                ("run", "method"),
                ("count", "method")
            ]
        );
        assert_eq!(syms[0].children[0].category, SymbolCategory::Variable);
        assert_eq!(syms[0].children[1].category, SymbolCategory::Function);
        assert_eq!(syms[0].children[1].id, "Test.java::Svc/Svc");
        assert_eq!(syms[0].children[2].id, "Test.java::Svc/run");
    }

    /// Each variable of a multi-variable field is its own symbol, and each
    /// spans the whole declaration.
    #[test]
    fn one_symbol_per_declared_field_variable() {
        let src = "class P {\n    private int x, y = 3;\n}";
        let syms = parse(src);
        assert_eq!(members(&syms[0]), vec![("x", "field"), ("y", "field")]);
        let decl = src.find("private").unwrap() as u32..(src.find(";").unwrap() + 1) as u32;
        assert_eq!(syms[0].children[0].byte_range, decl);
        assert_eq!(syms[0].children[1].byte_range, decl);
    }

    /// Overloads share a name, and so an id — ids are not unique, as with
    /// Rust's repeated inherent impls. Their ranges and content hashes are
    /// their own, which is what tells them apart downstream.
    #[test]
    fn overloads_share_an_id_but_not_a_hash() {
        let syms = parse("class M {\n    void draw() {}\n    void draw(int scale) {}\n}");
        let [a, b] = &syms[0].children[..] else {
            panic!("two overloads: {:?}", members(&syms[0]))
        };
        assert_eq!(
            (a.id.as_str(), b.id.as_str()),
            ("Test.java::M/draw", "Test.java::M/draw")
        );
        assert_eq!((a.line_range.start, b.line_range.start), (2, 3));
        assert_ne!(a.content_hash, b.content_hash);
    }

    #[test]
    fn interface_members_including_default_and_static_methods_and_constants() {
        let syms = parse(
            "interface Shape {
                double PI = 3.14;
                double area();
                default String describe() { return \"shape\"; }
                static Shape unit() { return null; }
            }",
        );
        assert_eq!(syms[0].label, "interface");
        assert_eq!(
            members(&syms[0]),
            vec![
                ("PI", "field"),
                ("area", "method"),
                ("describe", "method"),
                ("unit", "method")
            ]
        );
    }

    /// An enum's constants are not symbols, as Rust's and TypeScript's enum
    /// members are not; its fields, constructors and methods are, and hang
    /// off the enum directly rather than off the grammar's intermediate
    /// `enum_body_declarations` node.
    #[test]
    fn enum_members_but_not_constants() {
        let syms = parse(
            "enum Color {
                RED(1), GREEN(2) { int value() { return 0; } };
                private final int v;
                Color(int v) { this.v = v; }
                int value() { return v; }
            }",
        );
        assert_eq!(
            members(&syms[0]),
            vec![
                ("v", "field"),
                ("Color", "constructor"),
                ("value", "method")
            ]
        );
        assert_eq!(syms[0].children[2].id, "Test.java::Color/value");
    }

    #[test]
    fn record_with_compact_constructor_and_methods() {
        let syms = parse(
            "record Point(int x, int y) {
                Point { if (x < 0) throw new IllegalArgumentException(); }
                static Point origin() { return new Point(0, 0); }
            }",
        );
        assert_eq!(syms[0].label, "record");
        assert_eq!(
            members(&syms[0]),
            vec![("Point", "constructor"), ("origin", "method")]
        );
    }

    #[test]
    fn annotation_type_elements_and_constants() {
        let syms = parse("@interface Retry {\n    int times() default 3;\n    int MAX = 10;\n}");
        assert_eq!(syms[0].label, "@interface");
        assert_eq!(
            members(&syms[0]),
            vec![("times", "method"), ("MAX", "field")]
        );
    }

    // ---------------------------------------------------------------
    // Nesting
    // ---------------------------------------------------------------

    #[test]
    fn nested_types_nest_to_any_depth() {
        let syms = parse(
            "class Outer {
                static class Nested {
                    interface Deep { void go(); }
                }
                class Inner { void m() {} }
                enum Mode { A }
            }",
        );
        assert_eq!(
            members(&syms[0]),
            vec![("Nested", "class"), ("Inner", "class"), ("Mode", "enum")]
        );
        let deep = &syms[0].children[0].children[0];
        assert_eq!(deep.id, "Test.java::Outer/Nested/Deep");
        assert_eq!(deep.children[0].id, "Test.java::Outer/Nested/Deep/go");
        assert_eq!(
            syms[0].children[1].children[0].id,
            "Test.java::Outer/Inner/m"
        );
    }

    /// Nothing inside a method body is a symbol — no other parser here looks
    /// into function bodies — so local and anonymous classes belong to the
    /// method that declares them.
    #[test]
    fn local_and_anonymous_classes_belong_to_their_method() {
        let syms = parse(
            "class A {
                void m() {
                    class Local { void x() {} }
                    Runnable r = new Runnable() { public void run() {} };
                }
            }",
        );
        assert_eq!(members(&syms[0]), vec![("m", "method")]);
        assert!(syms[0].children[0].children.is_empty());
    }

    #[test]
    fn several_top_level_types() {
        let syms = parse("public class Main {}\nclass Helper {}\n");
        let names: Vec<&str> = syms.iter().map(|s| s.name.as_ref()).collect();
        assert_eq!(names, vec!["Main", "Helper"]);
    }

    // ---------------------------------------------------------------
    // Ranges, annotations and comments
    // ---------------------------------------------------------------

    #[test]
    fn line_ranges_are_correct() {
        let syms = parse("class A {\n    void a() {}\n\n    void b() {\n    }\n}\n");
        assert_eq!(syms[0].line_range, 1..6);
        assert_eq!(syms[0].children[0].line_range, 2..2);
        assert_eq!(syms[0].children[1].line_range, 4..5);
    }

    /// Annotations are part of the declaration's own `modifiers`, so they are
    /// inside its range without any widening.
    #[test]
    fn annotations_are_part_of_the_member() {
        let src = "class A {\n    @Override\n    @SuppressWarnings(\"x\")\n    public String toString() { return \"\"; }\n}";
        let syms = parse(src);
        let m = &syms[0].children[0];
        assert_eq!(m.name.as_ref(), "toString");
        assert_eq!(m.byte_range.start as usize, src.find("@Override").unwrap());
        assert_eq!(m.line_range.start, 2);
        assert_eq!(
            m.head_line, 4,
            "its name is on the line after its annotations"
        );
    }

    /// A Javadoc glued to a declaration widens it, as `///` does in Rust,
    /// and the annotation between them does not break the run.
    #[test]
    fn a_javadoc_widens_the_symbol() {
        let src = "/** A service. */\n@Deprecated\npublic class Service {\n    /**\n     * Runs.\n     */\n    @Override\n    public void run() {}\n}";
        let syms = parse(src);
        assert_eq!(syms[0].byte_range.start, 0);
        let run = &syms[0].children[0];
        assert_eq!(
            run.byte_range.start as usize,
            src.find("    /**").unwrap() + 4
        );
        assert_eq!(run.line_range.start, 4);
    }

    /// A comment above the first member of a body is still that member's
    /// sibling in this grammar — not the gap Python's grammar has.
    #[test]
    fn a_line_comment_above_the_first_member_widens_it() {
        let src = "class A {\n    // Makes one.\n    A() {}\n}";
        let syms = parse(src);
        let ctor = &syms[0].children[0];
        assert_eq!(
            ctor.byte_range.start as usize,
            src.find("// Makes one.").unwrap()
        );
        assert_eq!(
            syms[0].byte_range.start, 0,
            "the class is not widened by its member's comment"
        );
    }

    #[test]
    fn a_comment_separated_by_a_blank_line_does_not_widen() {
        let src = "// unrelated\n\nclass A {}";
        let syms = parse(src);
        assert_eq!(
            syms[0].byte_range.start as usize,
            src.find("class").unwrap()
        );
    }

    #[test]
    fn a_trailing_comment_belongs_to_the_code_before_it() {
        let src = "class A {\n    int x; // about x\n    void f() {}\n}";
        let syms = parse(src);
        let f = &syms[0].children[1];
        assert_eq!(f.byte_range.start as usize, src.find("void").unwrap());
    }

    #[test]
    fn hashes_and_tokens_are_set() {
        let syms = parse("class A { void f() { int x = 1; } }");
        assert_ne!(syms[0].content_hash, [0u8; 32]);
        assert_ne!(syms[0].merkle_hash, [0u8; 32]);
        assert!(syms[0].children[0].estimated_tokens > 0);
    }

    #[test]
    fn file_path_and_total_lines_are_set() {
        let file = JavaParser::new()
            .parse_file(Path::new("src/main/java/App.java"), "class App {\n}\n")
            .unwrap();
        assert_eq!(file.file_path, Path::new("src/main/java/App.java"));
        assert_eq!(file.total_lines, 2);
        assert_eq!(file.symbols[0].id, "src/main/java/App.java::App");
    }

    // ---------------------------------------------------------------
    // Call sites
    // ---------------------------------------------------------------

    /// Method calls come from the grammar's own tags query; constructor calls
    /// from the supplement, down to the class name however it is qualified.
    #[test]
    fn method_and_constructor_calls_are_call_sites() {
        let src = "class A {\n    void f() {\n        helper(1);\n        this.list.add(2);\n        new Foo();\n        new java.util.ArrayList<String>();\n        new Outer.Inner();\n        new Box<>();\n    }\n}";
        let registry = crate::parser::ParserRegistry::new();
        let sites = crate::callers::call_sites(&registry, Path::new("A.java"), src);
        let names: Vec<(&str, u32)> = sites
            .iter()
            .map(|(n, line, _)| (n.as_str(), *line))
            .collect();
        assert_eq!(
            names,
            vec![
                ("helper", 3),
                ("add", 4),
                ("Foo", 5),
                ("ArrayList", 6),
                ("Inner", 7),
                ("Box", 8)
            ]
        );
    }
}
