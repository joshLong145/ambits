//! Who calls this symbol, for `ambits callers`.
//!
//! ## What this can and cannot know
//!
//! Reference sites come from the grammar's own `tags.scm`, which every
//! tree-sitter crate we depend on ships. Those queries already define
//! `@reference.call` over call expressions, method calls, and macro
//! invocations, so nothing here authors per-language rules.
//!
//! What the grammar gives is *syntax*, not resolution. A call to `walk()` is
//! visibly a call to something named `walk`; which `walk` it binds to needs
//! type inference that tree-sitter does not do. On this repo that mostly does
//! not bite — 838 of 916 function names are unique — but it bites hard on the
//! ones you would most want to trace: twelve definitions are named `new`, six
//! `render`. So this reports **call sites whose callee is named X**, and says
//! so, rather than claiming to have resolved anything.
//!
//! The consolation is that being syntax-aware is already most of the value.
//! Unlike a text search, a reference cannot be a mention in a doc comment or a
//! line inside a string literal — measured on this repo, that was three of
//! five hits for one query, two of them Rust source embedded in a benchmark
//! fixture's raw string.
//!
//! ## Attribution
//!
//! A call site's line number is not the useful answer; the enclosing function
//! is. Every symbol carries a `byte_range`, so the innermost symbol containing
//! the reference is a containment search over the file's already-scanned
//! symbols. That turns `app.rs:1847` into
//! `src/app.rs::App/process_agent_event`, which is an id `show` will accept.
//!
//! ## Why references are not stored
//!
//! They are extracted on demand rather than kept on `ProjectTree`. Call sites
//! vastly outnumber definitions, and every other command — `find`, `show`, the
//! TUI — would carry that weight without using it. Re-reading the files costs
//! this one command a fraction of a second and costs the rest nothing.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use color_eyre::eyre::{Result, WrapErr};
use serde::Serialize;
use streaming_iterator::StreamingIterator;
use tree_sitter::{Parser, Query, QueryCursor};

use crate::parser::ParserRegistry;
use crate::symbols::{ProjectTree, SymbolNode};

/// Bumped on any breaking change to the emitted shape.
pub const SCHEMA_VERSION: u32 = 1;

/// How deep to follow macros inside macros before giving up.
const MAX_MACRO_DEPTH: usize = 8;

/// One call site.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallSite {
    /// Name of the thing being called, as written.
    pub callee: String,
    /// Project-relative file the call appears in.
    pub file: PathBuf,
    /// 1-based line.
    pub line: u32,
    /// Id of the innermost symbol containing the call, when there is one.
    /// Absent for calls at file scope.
    pub caller: Option<String>,
}

/// Extract every `@reference.call` capture from one already-read source file.
///
/// Returns nothing rather than failing when the grammar has no usable tags
/// query: a language we cannot introspect should narrow the answer, not abort
/// the command.
pub fn call_sites(registry: &ParserRegistry, path: &Path, source: &str) -> Vec<(String, u32, u32)> {
    let Some(parser_impl) = registry.parser_for(path) else {
        return Vec::new();
    };
    let language = parser_impl.language();

    // Supplemented rather than replaced: the shipped query is a good base and
    // stays maintained upstream, but it is written for tagging rather than for
    // exhaustive reference finding.
    let source_query = format!(
        "{}\n{}",
        parser_impl.tags_query(),
        parser_impl.tags_supplement()
    );
    let Ok(query) = Query::new(&language, &source_query) else {
        return Vec::new();
    };

    let mut parser = Parser::new();
    if parser.set_language(&language).is_err() {
        return Vec::new();
    }
    let Some(tree) = parser.parse(source, None) else {
        return Vec::new();
    };

    let src = source.as_bytes();
    let mut out = Vec::new();
    collect(&query, tree.root_node(), src, 0, 0, &mut out);

    // Descend into macro bodies. tree-sitter does not parse them — a macro's
    // arguments arrive as an unparsed `token_tree` — so a call written inside
    // `println!` or `format!` is invisible to the query above. The grammar's
    // own injections.scm prescribes the remedy: re-parse that token tree as
    // Rust. The fragment is rarely valid on its own (`"{}", x` is not a
    // program), but the parser is error-tolerant and still recognizes the call
    // expressions inside it.
    //
    // Skipped without this, `crate::fmt::tokens` in digest.rs and both
    // `crate::fmt::bytes` calls in cache.rs went unreported.
    // A worklist rather than one pass, because macros nest: `ui/stats.rs`
    // wraps a `format!` inside a `vec![`, so the inner call only becomes
    // visible after the outer token tree has itself been re-parsed. Offsets
    // compound as it descends. Bounded so a pathological nesting cannot spin.
    let mut queue = Vec::new();
    collect_macro_bodies(tree.root_node(), src, 0, 0, &mut queue);
    let mut depth = 0;
    while let Some((start_byte, start_row, text)) = queue.pop() {
        if depth > MAX_MACRO_DEPTH * 64 {
            break;
        }
        depth += 1;
        let Some(sub) = parser.parse(&text, None) else {
            continue;
        };
        collect(
            &query,
            sub.root_node(),
            text.as_bytes(),
            start_byte,
            start_row,
            &mut out,
        );
        collect_macro_bodies(sub.root_node(), text.as_bytes(), start_byte, start_row, &mut queue);
    }

    out.sort_by_key(|(_, line, byte)| (*line, *byte));
    out.dedup();
    out
}

/// Byte offset, starting row, and text of every macro argument list.
///
/// Nested macros are reached by recursing through the whole tree rather than
/// only its top level, so a call inside `assert!(matches!(...))` is still
/// found.
fn collect_macro_bodies(
    node: tree_sitter::Node,
    src: &[u8],
    byte_offset: u32,
    row_offset: u32,
    out: &mut Vec<(u32, u32, String)>,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "macro_invocation" {
            let mut inner = child.walk();
            for part in child.children(&mut inner) {
                if part.kind() == "token_tree" {
                    if let Ok(text) = part.utf8_text(src) {
                        out.push((
                            part.start_byte() as u32 + byte_offset,
                            part.start_position().row as u32 + row_offset,
                            text.to_string(),
                        ));
                    }
                }
            }
        }
        collect_macro_bodies(child, src, byte_offset, row_offset, out);
    }
}

/// Run the call query over one tree, offsetting positions into the original
/// file so a re-parsed fragment still reports where it really lives.
fn collect(
    query: &Query,
    root: tree_sitter::Node,
    src: &[u8],
    byte_offset: u32,
    row_offset: u32,
    out: &mut Vec<(String, u32, u32)>,
) {
    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(query, root, src);
    while let Some(m) = matches.next() {
        // A pattern is a call reference when its own capture set includes the
        // `reference.call` marker.
        let is_call = m.captures.iter().any(|c| {
            query
                .capture_names()
                .get(c.index as usize)
                .is_some_and(|n| *n == "reference.call")
        });
        if !is_call {
            continue;
        }
        for c in m.captures {
            // `tags.scm` captures far more than calls. Only the `@name` inside
            // a call reference is wanted, identified by capture name so this
            // stays correct if a grammar reorders its patterns.
            if query.capture_names().get(c.index as usize) != Some(&"name") {
                continue;
            }
            if let Ok(text) = c.node.utf8_text(src) {
                out.push((
                    text.to_string(),
                    c.node.start_position().row as u32 + 1 + row_offset,
                    c.node.start_byte() as u32 + byte_offset,
                ));
            }
        }
    }
}

/// The innermost symbol whose byte range contains `byte`.
///
/// Innermost rather than outermost: a call inside a method should be
/// attributed to the method, not to the `impl` block wrapping it.
fn enclosing(symbols: &[SymbolNode], byte: u32) -> Option<&SymbolNode> {
    for sym in symbols {
        if sym.byte_range.start <= byte && byte < sym.byte_range.end {
            return enclosing(&sym.children, byte).or(Some(sym));
        }
    }
    None
}

/// Find every call site whose callee matches one of `names`, case-sensitively.
///
/// Matching is exact on the callee as written, not a substring: `new` should
/// not report every call to `new_tailer`.
pub fn find_callers(
    project_root: &Path,
    tree: &ProjectTree,
    registry: &ParserRegistry,
    names: &[String],
) -> Vec<CallSite> {
    let wanted: HashMap<&str, ()> = names.iter().map(|n| (n.as_str(), ())).collect();
    let mut out = Vec::new();

    for file in &tree.files {
        let full = project_root.join(&file.file_path);
        let Ok(source) = std::fs::read_to_string(&full) else {
            continue;
        };
        for (callee, line, byte) in call_sites(registry, &file.file_path, &source) {
            if !wanted.contains_key(callee.as_str()) {
                continue;
            }
            out.push(CallSite {
                callee,
                file: file.file_path.clone(),
                line,
                caller: enclosing(&file.symbols, byte).map(|s| s.id.clone()),
            });
        }
    }

    out.sort_by(|a, b| {
        a.callee
            .cmp(&b.callee)
            .then_with(|| a.file.cmp(&b.file))
            .then_with(|| a.line.cmp(&b.line))
    });
    out
}

#[derive(Serialize)]
struct SiteDto<'a> {
    file: String,
    line: u32,
    /// Innermost enclosing symbol id, usable directly as a `show` selector.
    #[serde(skip_serializing_if = "Option::is_none")]
    caller: Option<&'a str>,
}

#[derive(Serialize)]
struct ResultDto<'a> {
    callee: &'a str,
    call_sites: usize,
    /// Distinct enclosing symbols, which is usually the number a reader wants.
    callers: usize,
    sites: Vec<SiteDto<'a>>,
}

#[derive(Serialize)]
struct CallersDto<'a> {
    schema_version: u32,
    /// Always true: matching is by callee name, without resolving which
    /// definition of that name is actually bound. Emitted so a consumer cannot
    /// mistake this for a resolved call graph.
    name_matched_only: bool,
    results: Vec<ResultDto<'a>>,
}

/// Resolve and print, as JSON or as a grouped listing.
pub fn run(
    project_root: &Path,
    tree: &ProjectTree,
    registry: &ParserRegistry,
    names: &[String],
    json: bool,
) -> Result<()> {
    let sites = find_callers(project_root, tree, registry, names);

    let mut by_callee: HashMap<&str, Vec<&CallSite>> = HashMap::new();
    for s in &sites {
        by_callee.entry(s.callee.as_str()).or_default().push(s);
    }

    if json {
        let results: Vec<ResultDto> = names
            .iter()
            .map(|n| {
                let group = by_callee.get(n.as_str()).cloned().unwrap_or_default();
                let mut distinct: Vec<&str> =
                    group.iter().filter_map(|s| s.caller.as_deref()).collect();
                distinct.sort_unstable();
                distinct.dedup();
                ResultDto {
                    callee: n,
                    call_sites: group.len(),
                    callers: distinct.len(),
                    sites: group
                        .iter()
                        .map(|s| SiteDto {
                            file: s.file.display().to_string(),
                            line: s.line,
                            caller: s.caller.as_deref(),
                        })
                        .collect(),
                }
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string(&CallersDto {
                schema_version: SCHEMA_VERSION,
                name_matched_only: true,
                results,
            })
            .wrap_err("serializing caller results")?
        );
        return Ok(());
    }

    for (i, name) in names.iter().enumerate() {
        if i > 0 {
            println!();
        }
        let group = by_callee.get(name.as_str()).cloned().unwrap_or_default();
        if group.is_empty() {
            println!("{name} — no call sites");
            continue;
        }
        let mut distinct: Vec<&str> = group.iter().filter_map(|s| s.caller.as_deref()).collect();
        distinct.sort_unstable();
        distinct.dedup();
        println!(
            "{name} — {} call site{} in {} caller{}",
            group.len(),
            if group.len() == 1 { "" } else { "s" },
            distinct.len(),
            if distinct.len() == 1 { "" } else { "s" },
        );
        for s in &group {
            match &s.caller {
                Some(c) => println!("  {c}  ({}:{})", s.file.display(), s.line),
                None => println!("  <file scope>  ({}:{})", s.file.display(), s.line),
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::*;

    fn registry() -> ParserRegistry {
        ParserRegistry::new()
    }

    const SRC: &str = r#"
fn helper() -> u32 { 1 }

fn outer() -> u32 {
    helper() + helper()
}

struct Thing;

impl Thing {
    fn method(&self) -> u32 {
        helper()
    }
}

/// A doc comment mentioning helper() which is not a call.
const SAMPLE: &str = "helper() inside a string literal";
"#;

    fn sites(src: &str) -> Vec<(String, u32, u32)> {
        call_sites(&registry(), Path::new("a.rs"), src)
    }

    #[test]
    fn call_sites_are_found_by_the_grammars_own_tags_query() {
        let names: Vec<String> = sites(SRC).into_iter().map(|(n, _, _)| n).collect();
        assert_eq!(
            names.iter().filter(|n| *n == "helper").count(),
            3,
            "two in outer, one in method"
        );
    }

    /// The structural advantage over text search: a mention in prose or inside
    /// a string is not a call node, so it cannot be reported as one.
    #[test]
    fn comments_and_string_literals_are_not_call_sites() {
        let found = sites(SRC).into_iter().filter(|(n, _, _)| n == "helper").count();
        assert_eq!(found, 3, "the doc comment and the string literal are excluded");
        assert!(
            SRC.matches("helper()").count() > found,
            "the text really does contain more occurrences than there are calls"
        );
    }

    /// The gap that shipped-query-only extraction had: Rust reaches most
    /// things through a path, and `tags.scm` matches only a bare identifier in
    /// function position. Against the real repo this missed both call sites of
    /// `centered_rect` and half of `hash_hex`'s.
    #[test]
    fn path_qualified_calls_are_found() {
        let src = r#"
fn target() {}
fn a() { super::target(); }
fn b() { crate::x::target(); }
fn c() { Thing::target(); }
fn d() { target::<u8>(); }
fn e() { target(); }
"#;
        let found = sites(src)
            .into_iter()
            .filter(|(n, _, _)| n == "target")
            .count();
        assert_eq!(found, 5, "four path or generic forms, plus the bare call");
    }

    /// A call is attributed to the function containing it, not to the `impl`
    /// block wrapping that function.
    #[test]
    fn attribution_picks_the_innermost_enclosing_symbol() {
        let outer = sym_with_range("a.rs::Thing", "Thing", 0, 100);
        let inner = sym_with_range("a.rs::Thing/method", "method", 40, 60);
        let tree = vec![sym_with_children_ranged(outer, vec![inner])];

        assert_eq!(enclosing(&tree, 50).unwrap().id, "a.rs::Thing/method");
        assert_eq!(enclosing(&tree, 10).unwrap().id, "a.rs::Thing");
        assert!(enclosing(&tree, 200).is_none(), "outside every symbol");
    }

    /// Rust puts an enormous amount of real code inside macros — `println!`,
    /// `format!`, `assert_eq!` — and tree-sitter does not parse macro bodies:
    /// the arguments arrive as an unparsed `token_tree`. Without descending
    /// into those, calls made from inside a macro are simply invisible.
    #[test]
    fn calls_inside_macro_invocations_are_found() {
        let src = r#"
fn target(x: u32) -> u32 { x }
fn caller() {
    println!("{}", target(1));
    let s = format!("{}", crate::m::target(2));
}
"#;
        let found = sites(src).into_iter().filter(|(n, _, _)| n == "target").count();
        assert_eq!(found, 2, "one bare and one path-qualified, both inside macros");
    }

    /// Macros nest. `ui/stats.rs` wraps a `format!` inside a `vec![`, and the
    /// inner call is only reachable after the outer token tree has itself been
    /// re-parsed — so one pass of injection is not enough.
    #[test]
    fn calls_inside_nested_macros_are_found() {
        let src = r#"
fn target(x: u32) -> u32 { x }
fn caller() {
    let v = vec![format!("{}", target(1))];
}
"#;
        let found = sites(src).into_iter().filter(|(n, _, _)| n == "target").count();
        assert_eq!(found, 1, "format! inside vec! must still be reached");
    }

    /// Positions from a re-parsed fragment must map back to the real file, or
    /// every macro-embedded call reports the wrong line.
    #[test]
    fn injected_positions_map_back_to_the_original_file() {
        let src = "fn a() {}\nfn b() {}\nfn c() { println!(\"{}\", a()); }\n";
        let hit = sites(src).into_iter().find(|(n, _, _)| n == "a").unwrap();
        assert_eq!(hit.1, 3, "the call is on line 3 of the original file");
    }

    #[test]
    fn an_unparseable_language_yields_no_sites_rather_than_failing() {
        assert!(call_sites(&registry(), Path::new("a.txt"), "helper()").is_empty());
    }

    fn sym_with_range(id: &str, name: &str, start: u32, end: u32) -> SymbolNode {
        let mut s = sym(id, name);
        s.byte_range = start..end;
        s
    }

    fn sym_with_children_ranged(mut parent: SymbolNode, children: Vec<SymbolNode>) -> SymbolNode {
        parent.children = children;
        parent
    }
}
