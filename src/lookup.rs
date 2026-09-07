//! Resolve a symbol to its definition, for `ambits show`.
//!
//! ## What this is for
//!
//! The digest tells an agent *that* it knows `App/process_compaction` and that
//! the symbol currently sits at `src/app.rs:340-388`. Acting on that still
//! costs a `Read` with the right offset, or a `find_symbol` round-trip. This
//! closes the gap: one call, one JSON object, the definition in hand.
//!
//! ## Two selectors, because only one of them is always available
//!
//! Content hash is the *exact* key — content-addressed, immune to line drift,
//! and the same value that decided the symbol was restorable in the first
//! place. But nothing puts a hash in front of an agent by default; the
//! markdown digest cannot afford 67 characters per symbol. So ids are accepted
//! too, since an agent reading the digest already has one: the `###` heading
//! is the path and the entry is the name path, and `<path>::<name-path>` is
//! exactly how ids are formed.
//!
//! Hash prefixes are accepted from 8 hex characters, which is enough to be
//! unique in any real project while staying short enough to pass around.
//!
//! ## Ambiguity is reported, never resolved by guessing
//!
//! Neither key is unique. Symbol ids collide by construction — `struct Foo`
//! and `impl Foo` both yield `<path>::Foo`. Content hashes collide when bodies
//! are byte-identical, which for this repo is about 1% of symbols (`mod
//! helpers;`, small duplicated utilities). Rather than pick one, every match
//! is returned and the caller decides. A `matches` array of length two is
//! information; silently returning the first would be a lie.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use color_eyre::eyre::{Result, WrapErr};
use serde::Serialize;

use crate::journal::{encode_hash, hash_hex};
use crate::symbols::{ProjectTree, SymbolNode};

/// Bumped on any breaking change to the emitted shape.
pub const SCHEMA_VERSION: u32 = 1;

/// Shortest hash prefix accepted. Below this, a "match" says more about the
/// prefix being short than about the symbol.
pub const MIN_HASH_PREFIX: usize = 8;

/// How a query names a symbol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Selector {
    /// Lowercase hex, with any `b3:` prefix stripped. May be a prefix of the
    /// full 64-character digest.
    Hash(String),
    /// `<project-relative-path>::<name-path>`.
    Id(String),
    /// Neither form — reported back rather than guessed at.
    Unrecognized(String),
}

/// Classify a query string.
///
/// An id is recognized by `::`, which cannot appear in a hex digest, so the
/// two forms are unambiguous and no flag is needed to disambiguate them.
pub fn parse_selector(query: &str) -> Selector {
    let q = query.trim();
    if q.contains("::") {
        return Selector::Id(q.to_string());
    }
    let hex = q.strip_prefix("b3:").unwrap_or(q);
    let looks_hex = hex.len() >= MIN_HASH_PREFIX
        && hex.len() <= 64
        && hex.chars().all(|c| c.is_ascii_hexdigit());
    if looks_hex {
        Selector::Hash(hex.to_ascii_lowercase())
    } else {
        Selector::Unrecognized(q.to_string())
    }
}

/// One matched symbol. Shared with `ambits find` so both commands describe a
/// symbol identically and their output composes — `find` locates, `show`
/// reads, and a caller can hand the `id` straight from one to the other.
#[derive(Serialize)]
pub(crate) struct MatchDto<'a> {
    id: &'a str,
    name: &'a str,
    file: String,
    /// 1-based inclusive, matching the digest's `name:first-last`.
    lines: [u32; 2],
    /// Byte offsets into the file, for callers that want to slice it
    /// themselves.
    bytes: [u32; 2],
    content_hash: String,
    /// Syntactic kind, e.g. `fn`, `struct`, `impl`.
    label: &'a str,
    estimated_tokens: u32,
    /// Immediate children, so a caller can walk into a container without a
    /// second scan.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    children: Vec<&'a str>,
    /// The definition source. Omitted under `--no-body`.
    #[serde(skip_serializing_if = "Option::is_none")]
    definition: Option<String>,
    /// True when `--max-bytes` cut the definition short. A truncated
    /// definition is not valid source, so this is never left implicit.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    truncated: bool,
}

#[derive(Serialize)]
struct ResultDto<'a> {
    query: &'a str,
    /// `hash`, `id`, or `unrecognized` — so a caller can tell "no such symbol"
    /// from "that was not a valid selector".
    selector: &'static str,
    /// Every symbol matching the query, sorted by file then line. Empty means
    /// no match.
    matches: Vec<MatchDto<'a>>,
}

#[derive(Serialize)]
struct ShowDto<'a> {
    schema_version: u32,
    results: Vec<ResultDto<'a>>,
}

/// Read the definition's source text out of the file.
///
/// Slices by byte range rather than by line so the result is exactly the span
/// the parser identified. Files are cached because a single invocation
/// commonly asks for several symbols from one file.
/// Describe a symbol without its source. The body-bearing fields stay `None`,
/// which is exactly what `--no-body` and `find` both want.
pub(crate) fn describe<'a>(file: &'a Path, node: &'a SymbolNode) -> MatchDto<'a> {
    MatchDto {
        id: &node.id,
        name: &node.name,
        file: file.display().to_string(),
        lines: [node.line_range.start, node.line_range.end],
        bytes: [node.byte_range.start, node.byte_range.end],
        content_hash: encode_hash(&node.content_hash),
        label: node.label,
        estimated_tokens: node.estimated_tokens,
        children: node.children.iter().map(|c| c.id.as_str()).collect(),
        definition: None,
        truncated: false,
    }
}

fn definition_of(
    root: &Path,
    file: &Path,
    node: &SymbolNode,
    cache: &mut HashMap<PathBuf, String>,
    max_bytes: Option<usize>,
) -> (Option<String>, bool) {
    let full = root.join(file);
    let src = match cache.get(&full) {
        Some(s) => s,
        None => {
            let Ok(text) = std::fs::read_to_string(&full) else {
                return (None, false);
            };
            cache.entry(full.clone()).or_insert(text)
        }
    };

    let start = node.byte_range.start as usize;
    let end = (node.byte_range.end as usize).min(src.len());
    if start >= end {
        return (None, false);
    }
    // Byte ranges come from the same parse as the tree, so they land on
    // character boundaries; `get` rather than slicing keeps a mismatch from
    // panicking if that ever stops holding.
    let Some(text) = src.get(start..end) else {
        return (None, false);
    };

    match max_bytes {
        Some(limit) if text.len() > limit => {
            // Back off to a character boundary so the output stays valid UTF-8.
            let mut cut = limit;
            while cut > 0 && !text.is_char_boundary(cut) {
                cut -= 1;
            }
            (Some(text[..cut].to_string()), true)
        }
        _ => (Some(text.to_string()), false),
    }
}

/// Resolve `queries` against `tree` and print one JSON object to stdout.
///
/// Exits successfully even when nothing matches: "no symbol has that hash" is
/// an answer, not a failure, and a caller parsing JSON should not have to
/// interpret an exit code to learn it. Only a genuinely broken invocation —
/// an unreadable project — returns `Err`.
pub fn run(
    project_root: &Path,
    tree: &ProjectTree,
    queries: &[String],
    include_body: bool,
    max_bytes: Option<usize>,
) -> Result<()> {
    let out = resolve(project_root, tree, queries, include_body, max_bytes);
    // Single line, like the other JSON surfaces, so it survives being piped
    // through line-oriented tooling.
    println!(
        "{}",
        serde_json::to_string(&out).wrap_err("serializing lookup results")?
    );
    Ok(())
}

/// The whole of `run` except printing, so tests exercise the real resolution
/// path instead of a copy of it.
fn resolve<'a>(
    project_root: &Path,
    tree: &'a ProjectTree,
    queries: &'a [String],
    include_body: bool,
    max_bytes: Option<usize>,
) -> ShowDto<'a> {
    let all = tree.walk();
    // Hex once per symbol rather than once per (symbol, query).
    let hashes: Vec<String> = all
        .iter()
        .map(|(_, s)| hash_hex(&s.content_hash))
        .collect();

    let mut cache: HashMap<PathBuf, String> = HashMap::new();
    let mut results = Vec::with_capacity(queries.len());

    for query in queries {
        let selector = parse_selector(query);
        let mut hits: Vec<(&Path, &SymbolNode)> = match &selector {
            Selector::Hash(prefix) => all
                .iter()
                .zip(&hashes)
                .filter(|(_, h)| h.starts_with(prefix.as_str()))
                .map(|((f, s), _)| (*f, *s))
                .collect(),
            Selector::Id(id) => all
                .iter()
                .filter(|(_, s)| s.id == *id)
                .map(|(f, s)| (*f, *s))
                .collect(),
            Selector::Unrecognized(_) => Vec::new(),
        };

        // Stable output: the tree walk is deterministic, but sorting makes the
        // contract explicit rather than incidental.
        hits.sort_by(|a, b| {
            a.0.cmp(b.0)
                .then_with(|| a.1.line_range.start.cmp(&b.1.line_range.start))
                .then_with(|| a.1.id.cmp(&b.1.id))
        });

        let matches = hits
            .into_iter()
            .map(|(file, node)| {
                let (definition, truncated) = if include_body {
                    definition_of(project_root, file, node, &mut cache, max_bytes)
                } else {
                    (None, false)
                };
                MatchDto {
                    definition,
                    truncated,
                    ..describe(file, node)
                }
            })
            .collect();

        results.push(ResultDto {
            query,
            selector: match selector {
                Selector::Hash(_) => "hash",
                Selector::Id(_) => "id",
                Selector::Unrecognized(_) => "unrecognized",
            },
            matches,
        });
    }

    ShowDto {
        schema_version: SCHEMA_VERSION,
        results,
    }
}

#[cfg(test)]
mod tests {
    use crate::helpers::*;
    use super::*;

    #[test]
    fn ids_are_recognized_by_their_separator() {
        assert_eq!(
            parse_selector("src/app.rs::App/new"),
            Selector::Id("src/app.rs::App/new".into())
        );
    }

    #[test]
    fn hashes_are_accepted_with_or_without_the_algorithm_prefix() {
        let bare = "deadbeef".repeat(8);
        assert_eq!(parse_selector(&bare), Selector::Hash(bare.clone()));
        assert_eq!(
            parse_selector(&format!("b3:{bare}")),
            Selector::Hash(bare.clone())
        );
        assert_eq!(
            parse_selector(&bare.to_uppercase()),
            Selector::Hash(bare),
            "hex case is not meaningful"
        );
    }

    #[test]
    fn short_prefixes_are_accepted_but_stubs_are_not() {
        assert_eq!(parse_selector("deadbeef"), Selector::Hash("deadbeef".into()));
        assert_eq!(
            parse_selector("deadbee"),
            Selector::Unrecognized("deadbee".into()),
            "one character below the floor"
        );
    }

    /// A caller must be able to tell "not a selector" from "no such symbol",
    /// so a bare name is rejected rather than silently matching nothing.
    #[test]
    fn a_bare_name_is_unrecognized_rather_than_an_empty_match() {
        assert_eq!(
            parse_selector("process_compaction"),
            Selector::Unrecognized("process_compaction".into())
        );
    }

    #[test]
    fn walk_descends_into_children() {
        let tree = project(vec![file(
            "a.rs",
            vec![sym_with_children(
                "a.rs::Outer",
                "Outer",
                vec![sym("a.rs::Outer/inner", "inner")],
            )],
        )]);
        let flat = tree.walk();
        assert_eq!(flat.len(), 2);
        assert!(flat.iter().any(|(_, s)| s.id == "a.rs::Outer/inner"));
    }

    fn tmp_project(files: &[(&str, &str)]) -> (tempfile::TempDir, ProjectTree) {
        let dir = tempfile::tempdir().unwrap();
        for (name, body) in files {
            std::fs::write(dir.path().join(name), body).unwrap();
        }
        let tree = crate::parser::ParserRegistry::new()
            .scan_project(dir.path(), None)
            .unwrap();
        (dir, tree)
    }

    fn show(root: &Path, tree: &ProjectTree, q: &[&str], body: bool) -> serde_json::Value {
        let queries: Vec<String> = q.iter().map(|s| s.to_string()).collect();
        serde_json::to_value(resolve(root, tree, &queries, body, None)).unwrap()
    }

    /// The definition must be the exact source span, not an approximation
    /// reconstructed from line numbers.
    #[test]
    fn the_definition_is_the_exact_source_span() {
        let (dir, tree) = tmp_project(&[("a.rs", "fn before() {}\n\nfn target(x: u8) -> u8 {\n    x + 1\n}\n")]);
        let v = show(dir.path(), &tree, &["a.rs::target"], true);
        let m = &v["results"][0]["matches"][0];
        assert_eq!(
            m["definition"].as_str().unwrap(),
            "fn target(x: u8) -> u8 {\n    x + 1\n}"
        );
    }

    /// A hash found in one place must resolve there, and the hash it reports
    /// back must be the one that was asked for.
    #[test]
    fn a_hash_prefix_resolves_to_the_symbol_that_owns_it() {
        let (dir, tree) = tmp_project(&[("a.rs", "fn only() {\n    let x = 41;\n}\n")]);
        let full = show(dir.path(), &tree, &["a.rs::only"], false)["results"][0]["matches"][0]
            ["content_hash"]
            .as_str()
            .unwrap()
            .to_string();

        let prefix = &full[3..3 + MIN_HASH_PREFIX];
        let v = show(dir.path(), &tree, &[prefix], false);
        assert_eq!(v["results"][0]["selector"], "hash");
        assert_eq!(v["results"][0]["matches"][0]["content_hash"], full);
    }

    /// Colliding ids are reported in full. Returning the first would be a
    /// silent lie about which symbol the caller got.
    #[test]
    fn every_match_for_an_ambiguous_id_is_returned() {
        let (dir, tree) = tmp_project(&[(
            "a.rs",
            "struct Foo { a: u8 }\n\nimpl Foo {\n    fn go(&self) {}\n}\n",
        )]);
        let v = show(dir.path(), &tree, &["a.rs::Foo"], false);
        let matches = v["results"][0]["matches"].as_array().unwrap();
        assert_eq!(matches.len(), 2, "struct Foo and impl Foo share an id");
        assert!(matches[0]["lines"][0].as_u64() < matches[1]["lines"][0].as_u64());
    }

    /// A miss and a malformed selector are different answers, and a caller
    /// must be able to tell them apart without guessing.
    #[test]
    fn a_miss_and_a_bad_selector_are_distinguishable() {
        let (dir, tree) = tmp_project(&[("a.rs", "fn only() {}\n")]);
        let v = show(dir.path(), &tree, &["a.rs::nope", "wat", &"0".repeat(64)], false);
        assert_eq!(v["results"][0]["selector"], "id");
        assert_eq!(v["results"][0]["matches"].as_array().unwrap().len(), 0);
        assert_eq!(v["results"][1]["selector"], "unrecognized");
        assert_eq!(v["results"][2]["selector"], "hash");
        assert_eq!(v["results"][2]["matches"].as_array().unwrap().len(), 0);
    }

    /// Results come back in query order, so a batched caller can zip them
    /// against its input without matching on the query string.
    #[test]
    fn results_follow_query_order() {
        let (dir, tree) = tmp_project(&[("a.rs", "fn one() {}\nfn two() {}\n")]);
        let v = show(dir.path(), &tree, &["a.rs::two", "a.rs::one"], false);
        assert_eq!(v["results"][0]["query"], "a.rs::two");
        assert_eq!(v["results"][1]["query"], "a.rs::one");
    }
}
