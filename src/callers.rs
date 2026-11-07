//! Who calls this symbol, for `ambits callers`.
//!
//! ## What this can and cannot know
//!
//! Reference sites come from the grammar's own `tags.scm`, which every
//! tree-sitter crate we depend on ships. Most of those queries already define
//! `@reference.call` over call expressions, method calls, and macro
//! invocations, so nothing here authors per-language rules. Each parser adds
//! what its query leaves out through [`LanguageParser::tags_supplement`];
//! for TypeScript, that is every call.
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
//!
//! Nor does it need every file parsed to start with. A file that never mentions
//! a name cannot call it, so the scan behind this command skips those
//! ([`crate::parser::ScanScope::Mentioning`]) as [`find_callers`] does.

use std::collections::HashMap;
use std::ffi::OsStr;
use std::io::Write;
use std::path::{Path, PathBuf};

use color_eyre::eyre::{Result, WrapErr};
use serde::Serialize;
use streaming_iterator::StreamingIterator;
use tree_sitter::{Language, Parser, Query, QueryCursor};

use crate::parallel::par_map;
use crate::parser::{LanguageParser, ParserRegistry};
use crate::symbols::ProjectTree;

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

/// A language's call query. Compiled once per language: it is hundreds of
/// patterns, and compiling it costs more than running it over a typical file.
struct CallQuery {
    language: Language,
    query: Query,
}

impl CallQuery {
    /// `None` when the grammar has no usable tags query: a language we cannot
    /// introspect should narrow the answer, not abort the command.
    fn of(parser: &dyn LanguageParser) -> Option<Self> {
        let language = parser.language();
        // Supplemented rather than replaced: the shipped query is a good base
        // and stays maintained upstream, but it is written for tagging rather
        // than for exhaustive reference finding.
        let source = format!("{}\n{}", parser.tags_query(), parser.tags_supplement());
        let query = Query::new(&language, &source).ok()?;
        Some(Self { language, query })
    }
}

/// Extract every `@reference.call` capture from one already-read source file.
///
/// Returns nothing rather than failing for a language with no call query.
pub fn call_sites(registry: &ParserRegistry, path: &Path, source: &str) -> Vec<(String, u32, u32)> {
    registry
        .parser_for(path)
        .and_then(CallQuery::of)
        .map_or_else(Vec::new, |query| sites_in(&query, source, None))
}

/// As [`call_sites`], but skipping macro bodies that cannot contain any of
/// `wanted`.
///
/// Re-parsing macro token trees is the bulk of this module's cost — Rust code
/// is dense with `println!`, `format!` and `assert_eq!` — and a body whose
/// text lacks the name outright cannot hold a call to it.
fn sites_in(
    call_query: &CallQuery,
    source: &str,
    wanted: Option<&[String]>,
) -> Vec<(String, u32, u32)> {
    let query = &call_query.query;
    let mut parser = Parser::new();
    if parser.set_language(&call_query.language).is_err() {
        return Vec::new();
    }
    let Some(tree) = parser.parse(source, None) else {
        return Vec::new();
    };

    let src = source.as_bytes();
    let mut out = Vec::new();
    collect(query, tree.root_node(), src, 0, 0, &mut out);

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
    let keep =
        |text: &str| wanted.is_none_or(|names| names.iter().any(|n| text.contains(n.as_str())));
    queue.retain(|(_, _, text)| keep(text));
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
            query,
            sub.root_node(),
            text.as_bytes(),
            start_byte,
            start_row,
            &mut out,
        );
        let before = queue.len();
        collect_macro_bodies(
            sub.root_node(),
            text.as_bytes(),
            start_byte,
            start_row,
            &mut queue,
        );
        // Nested bodies get the same treatment as the outer ones.
        let mut added: Vec<_> = queue.split_off(before);
        added.retain(|(_, _, t)| keep(t));
        queue.extend(added);
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

    let mut queries: HashMap<&OsStr, Option<CallQuery>> = HashMap::new();
    for file in &tree.files {
        if let Some(extension) = file.file_path.extension() {
            queries
                .entry(extension)
                .or_insert_with(|| registry.parser_for(&file.file_path).and_then(CallQuery::of));
        }
    }

    let per_file = par_map(&tree.files, |file| {
        let query = file
            .file_path
            .extension()
            .and_then(|extension| queries.get(extension)?.as_ref())?;
        let full = project_root.join(&file.file_path);
        let source = std::fs::read_to_string(&full).ok()?;
        // A name that appears nowhere in the file's text cannot be called from
        // it, so the parse and the macro re-parses below can be skipped
        // outright. Sound because every call site contains the callee as a
        // substring, and worth it because most files mention most names never.
        if !names.iter().any(|n| source.contains(n.as_str())) {
            return None;
        }
        let sites: Vec<CallSite> = sites_in(query, &source, Some(names))
            .into_iter()
            .filter(|(callee, _, _)| wanted.contains_key(callee.as_str()))
            .map(|(callee, line, byte)| CallSite {
                callee,
                file: file.file_path.clone(),
                line,
                caller: file.enclosing(byte).map(|s| s.id.clone()),
            })
            .collect();
        Some(sites)
    });
    let mut out: Vec<CallSite> = per_file.into_iter().flatten().flatten().collect();

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
    let mut out = std::io::stdout().lock();

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
        writeln!(
            out,
            "{}",
            serde_json::to_string(&CallersDto {
                schema_version: SCHEMA_VERSION,
                name_matched_only: true,
                results,
            })
            .wrap_err("serializing caller results")?
        )?;
        return Ok(());
    }

    for (i, name) in names.iter().enumerate() {
        if i > 0 {
            writeln!(out)?;
        }
        let group = by_callee.get(name.as_str()).cloned().unwrap_or_default();
        if group.is_empty() {
            writeln!(out, "{name} — no call sites")?;
            continue;
        }
        let mut distinct: Vec<&str> = group.iter().filter_map(|s| s.caller.as_deref()).collect();
        distinct.sort_unstable();
        distinct.dedup();
        writeln!(
            out,
            "{name} — {} call site{} in {} caller{}",
            group.len(),
            if group.len() == 1 { "" } else { "s" },
            distinct.len(),
            if distinct.len() == 1 { "" } else { "s" },
        )?;
        for s in &group {
            let caller = s.caller.as_deref().unwrap_or("<file scope>");
            writeln!(out, "  {caller}  ({}:{})", s.file.display(), s.line)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let found = sites(SRC)
            .into_iter()
            .filter(|(n, _, _)| n == "helper")
            .count();
        assert_eq!(
            found, 3,
            "the doc comment and the string literal are excluded"
        );
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
        let found = sites(src)
            .into_iter()
            .filter(|(n, _, _)| n == "target")
            .count();
        assert_eq!(
            found, 2,
            "one bare and one path-qualified, both inside macros"
        );
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
        let found = sites(src)
            .into_iter()
            .filter(|(n, _, _)| n == "target")
            .count();
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

    /// Callee names in TypeScript `src`, in source order.
    fn ts_calls(src: &str) -> Vec<String> {
        call_sites(&registry(), Path::new("a.ts"), src)
            .into_iter()
            .map(|(name, _, _)| name)
            .collect()
    }

    /// The shipped TypeScript query has no call patterns, so without the
    /// supplement every one of these came back empty.
    #[test]
    fn typescript_bare_and_member_calls_are_found() {
        let src = r#"
bare();
a.member();
this.own();
a?.optional();
a.maybe?.();
super.inherited();
a.b.deep();
"#;
        assert_eq!(
            ts_calls(src),
            [
                "bare",
                "member",
                "own",
                "optional",
                "maybe",
                "inherited",
                "deep"
            ]
        );
    }

    /// A chain is one call per link, each reported at its own name, so a
    /// fluent chain over several lines lands on the line that makes each call.
    #[test]
    fn each_link_of_a_typescript_call_chain_is_a_call() {
        let found = call_sites(
            &registry(),
            Path::new("a.ts"),
            "a\n  .first()\n  .second();\n",
        );
        let links: Vec<(&str, u32)> = found.iter().map(|(n, l, _)| (n.as_str(), *l)).collect();
        assert_eq!(links, [("first", 2), ("second", 3)]);
    }

    /// `target<T>()` and `target!()` wrap the callee differently in the
    /// grammar, but a reader sees the same call to the same name.
    #[test]
    fn type_syntax_around_a_typescript_callee_does_not_hide_the_call() {
        let src = r#"
generic<T>();
a.genericMember<T>();
asserted!();
a.assertedMember!();
a!.onAsserted();
(a as any).onCast();
"#;
        assert_eq!(
            ts_calls(src),
            [
                "generic",
                "genericMember",
                "asserted",
                "assertedMember",
                "onAsserted",
                "onCast"
            ]
        );
    }

    /// The name is reported as written, `#` included, because that is also
    /// the name of the symbol it calls.
    #[test]
    fn a_typescript_private_method_call_keeps_its_hash() {
        assert_eq!(ts_calls("class K { m() { this.#p(); } }"), ["#p"]);
    }

    /// Calls are expressions, so they turn up well outside function bodies.
    /// `@decorated()` is a call but a bare `@decorated` is not, as in Python.
    #[test]
    fn typescript_calls_are_found_wherever_an_expression_can_go() {
        let src = r#"
const arrow = () => inArrow();
class K {
    field = inField();
    @decorated() method(arg = inDefault()) {}
    @notCalled other() {}
}
@ns.decoratedClass() class D {}
const text = `before ${inTemplate()} after`;
"#;
        assert_eq!(
            ts_calls(src),
            [
                "inArrow",
                "inField",
                "decorated",
                "inDefault",
                "decoratedClass",
                "inTemplate"
            ]
        );
    }

    /// A tagged template is a call without parentheses: `tag` runs, and the
    /// grammar says so with a `call_expression`.
    #[test]
    fn a_typescript_tagged_template_is_a_call() {
        assert_eq!(ts_calls("tag`text ${inside()}`;"), ["tag", "inside"]);
    }

    /// A constructor call is a call to the class, as it is in Java and Python.
    /// The shipped query's `@reference.class` pattern matches the same node,
    /// and must not make it a second site.
    #[test]
    fn typescript_new_is_a_call_to_the_class() {
        assert_eq!(
            ts_calls("new Foo(); new ns.Bar(); new Generic<T>(); new Bare;"),
            ["Foo", "Bar", "Generic", "Bare"]
        );
        assert_eq!(
            ts_calls("new Foo(arg()).method();"),
            ["Foo", "arg", "method"]
        );
    }

    /// No name in the syntax, so nothing to report. A string key could be read
    /// as one, but Python and Rust leave `a['b']()` out too, and `a[key]()`
    /// could not be reported at all. The object is still ordinary code.
    #[test]
    fn a_typescript_callee_named_only_at_runtime_is_not_reported() {
        assert_eq!(
            ts_calls("a['computed'](); a[key](); (a.wrapped)(); (0, indirect)();"),
            Vec::<String>::new()
        );
        assert_eq!(ts_calls("object()['computed']();"), ["object"]);
    }

    /// `super` and `import` are keywords in callee position, not names.
    #[test]
    fn typescript_super_and_import_calls_are_not_reported() {
        let src = "class K extends B { constructor() { super(); import('x'); } }";
        assert_eq!(ts_calls(src), Vec::<String>::new());
    }

    /// What a text search for `dispose(` drowns in: every one of these has the
    /// shape of a call and is not one.
    #[test]
    fn typescript_declarations_comments_and_strings_are_not_call_sites() {
        let src = r#"
declare function target(): void;
function target(): void {}
interface I { target(): void; }
abstract class A { abstract target(): void; target() {} }
type T = ReturnType<typeof target>;
// target()
/* target() */
const text = "target()" + `target()`;
"#;
        assert_eq!(ts_calls(src), Vec::<String>::new());
    }

    /// One call is one site. Each call expression has a single callee, so the
    /// alternatives in the supplement cannot both match it, and two calls on
    /// a line stay two because a site is keyed by its byte as well.
    #[test]
    fn each_typescript_call_is_reported_once() {
        let found = call_sites(
            &registry(),
            Path::new("a.ts"),
            "a.target(); a.target!(); a?.target();",
        );
        let bytes: Vec<u32> = found.iter().map(|(_, _, byte)| *byte).collect();
        assert_eq!(bytes, [2, 14, 28]);
    }

    /// The enclosing symbol is what the output leads with. Object literals and
    /// callbacks have no symbol of their own, so they fall through to whatever
    /// holds them, and a method's decorator sits beside it in the class body.
    #[test]
    fn typescript_calls_are_attributed_to_the_innermost_symbol() {
        let src = r#"
target();
function plain() { target(); }
const arrow = () => { target(); };
class K {
    field = target();
    handler = () => { target(); };
    method() { [1].map(() => target()); }
    @target() decorated() {}
}
namespace N { export function nested() { target(); } }
const table = { entry() { target(); } };
describe('suite', () => { target(); });
"#;
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.ts"), src).unwrap();
        let registry = registry();
        let tree = registry.scan_project(dir.path(), None).unwrap();

        let found = find_callers(dir.path(), &tree, &registry, &["target".to_string()]);
        let callers: Vec<Option<&str>> = found.iter().map(|s| s.caller.as_deref()).collect();
        assert_eq!(
            callers,
            [
                None,
                Some("a.ts::plain"),
                Some("a.ts::arrow"),
                Some("a.ts::K/field"),
                Some("a.ts::K/handler"),
                Some("a.ts::K/method"),
                Some("a.ts::K"),
                Some("a.ts::N/nested"),
                None,
                None,
            ]
        );
    }

    #[test]
    fn an_unparseable_language_yields_no_sites_rather_than_failing() {
        assert!(call_sites(&registry(), Path::new("a.txt"), "helper()").is_empty());
    }

    /// Files that never mention a name are left out of the scan, which must
    /// change nothing: not the sites, the callers they sit in, or their order.
    #[test]
    fn a_scan_of_the_mentioning_files_finds_the_same_callers() {
        use crate::parser::ScanScope;

        let dir = tempfile::tempdir().unwrap();
        let files = [
            ("a.rs", "fn a() { target(); println!(\"{}\", target()); }\n"),
            ("b.rs", "fn b() {}\n"),
            ("c.rs", "// target is only mentioned here\nfn c() {}\n"),
            ("d.py", "def d():\n    target()\n"),
            ("e.ts", "function e() { target(); other(); }\n"),
            ("f.md", "# target\n"),
        ];
        for (name, source) in files {
            std::fs::write(dir.path().join(name), source).unwrap();
        }

        let registry = registry();
        let names = vec!["target".to_string(), "other".to_string()];
        let full = registry.scan_project(dir.path(), None).unwrap();
        let narrowed = registry
            .scan_scoped(dir.path(), None, ScanScope::Mentioning(&names))
            .unwrap();

        assert!(narrowed.files.len() < full.files.len());
        let found = find_callers(dir.path(), &narrowed, &registry, &names);
        assert_eq!(found, find_callers(dir.path(), &full, &registry, &names));
        let in_file = |name: &str| found.iter().filter(|s| s.file == Path::new(name)).count();
        assert_eq!(
            (in_file("a.rs"), in_file("d.py"), in_file("e.ts")),
            (2, 1, 2)
        );
    }
}
