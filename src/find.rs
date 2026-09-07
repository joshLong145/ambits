//! Pattern search over the symbol index, for `ambits find`.
//!
//! ## Why this is not `show`
//!
//! [`crate::lookup`] is addressable-only: you must already know an exact id or
//! content hash. That covers acting on a digest, and nothing else. Every other
//! question — *what is in this file*, *where is this name defined*, *what
//! methods hang off this type* — is a query, and there was no way to ask one.
//!
//! Kept a separate command deliberately. `show` reports
//! `"selector": "unrecognized"` for anything that is neither an id nor a hash,
//! and that signal is load-bearing: it is how a caller distinguishes a
//! malformed query from a symbol that does not exist. Making bare names mean
//! "search" would quietly destroy it.
//!
//! ## The pattern grammar
//!
//! `[path]::[name]`, or a bare `name` when there is no `::`. Both halves are
//! case-insensitive, and an empty half matches everything — which is what
//! makes `src/app.rs::` an enumeration of that file.
//!
//! **The path half matches whole components, not raw substrings.** A substring
//! rule looks right until you try it: `ui::` then matches `src/tui.rs`,
//! because "ui" sits inside "tui". Requiring consecutive path components to
//! prefix-match drops exactly that and keeps everything else, including
//! `app` → `app.rs` and multi-segment `ui/stats`.
//!
//! **The name half matches the leaf, unless the pattern contains `/`.** Leaf
//! matching is what keeps results honest: against this repo, `test` matches 41
//! symbols by leaf but 486 by full path, because every `tests/foo` matches
//! through its parent. A `/` in the pattern means the caller is addressing
//! nesting on purpose, so the whole name path is matched instead — which is
//! what makes `::App/` return every member of `App` rather than nothing.
//!
//! ## What it does not do
//!
//! This searches *definitions*. It has no notion of usages: a method call like
//! `is_none_or` returns nothing, because no symbol in the tree is named that.
//! For call sites, grep remains the right tool.

use std::path::Path;

use color_eyre::eyre::{Result, WrapErr};
use serde::Serialize;

use crate::symbols::{ProjectTree, SymbolNode};

/// Bumped on any breaking change to the emitted shape.
pub const SCHEMA_VERSION: u32 = 2;

/// Results reported per query before truncating.
///
/// Chosen against real volume rather than taste: on this repo a single letter
/// matches over a thousand symbols, so an uncapped search is a wall of text.
/// Every realistic query lands far below this.
pub const DEFAULT_LIMIT: usize = 100;

/// A parsed `[path]::[name]` query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pattern {
    /// `/`-separated path segments, lowercased. Empty matches every file.
    pub path: Vec<String>,
    /// Lowercased name fragment. Empty matches every symbol.
    pub name: String,
    /// Set when `name` contained a `/`, which switches matching from the leaf
    /// to the full name path.
    pub nested: bool,
}

impl Pattern {
    pub fn parse(query: &str) -> Self {
        let q = query.trim();
        let (path, name) = match q.split_once("::") {
            Some((p, n)) => (p, n),
            None => ("", q),
        };
        Pattern {
            path: path
                .to_ascii_lowercase()
                .split('/')
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect(),
            name: name.to_ascii_lowercase(),
            nested: name.contains('/'),
        }
    }

    /// Whether any run of consecutive path components prefix-matches the
    /// pattern's segments.
    pub fn matches_path(&self, path: &Path) -> bool {
        if self.path.is_empty() {
            return true;
        }
        let comps: Vec<String> = path
            .components()
            .map(|c| c.as_os_str().to_string_lossy().to_ascii_lowercase())
            .collect();
        if comps.len() < self.path.len() {
            return false;
        }
        (0..=comps.len() - self.path.len()).any(|i| {
            self.path
                .iter()
                .enumerate()
                .all(|(j, seg)| comps[i + j].starts_with(seg.as_str()))
        })
    }

    /// Whether the symbol's name matches, against the leaf or the whole name
    /// path depending on how the pattern was written.
    pub fn matches_name(&self, name_path: &str) -> bool {
        if self.name.is_empty() {
            return true;
        }
        let hay = if self.nested {
            name_path
        } else {
            name_path.rsplit('/').next().unwrap_or(name_path)
        };
        hay.to_ascii_lowercase().contains(&self.name)
    }
}

/// Everything matching `pattern`, and how many there were in total.
///
/// The total is reported separately from the returned slice so a truncated
/// result can say how much it withheld rather than implying it found only
/// what it shows.
pub fn search<'a>(
    tree: &'a ProjectTree,
    pattern: &Pattern,
    limit: usize,
) -> (Vec<(&'a Path, &'a SymbolNode)>, usize) {
    let mut hits: Vec<(&Path, &SymbolNode)> = tree
        .walk()
        .into_iter()
        .filter(|(file, sym)| pattern.matches_path(file) && pattern.matches_name(sym.name_path()))
        .collect();

    // Deterministic: the walk is ordered, but stating it makes the contract
    // explicit rather than incidental.
    hits.sort_by(|a, b| {
        a.0.cmp(b.0)
            .then_with(|| a.1.line_range.start.cmp(&b.1.line_range.start))
            .then_with(|| a.1.id.cmp(&b.1.id))
    });

    let total = hits.len();
    hits.truncate(limit);
    (hits, total)
}

#[derive(Serialize)]
struct ResultDto<'a> {
    query: &'a str,
    /// Total matches, which exceeds `matches.len()` when truncated.
    matched: usize,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    truncated: bool,
    matches: Vec<crate::lookup::MatchDto<'a>>,
    /// How many of the reported matches have a recorded read. Absent without a
    /// coverage journal, which is not the same as zero.
    #[serde(skip_serializing_if = "Option::is_none")]
    read: Option<usize>,
}

#[derive(Serialize)]
struct FindDto<'a> {
    schema_version: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    coverage: Option<crate::lookup::CoverageDto<'a>>,
    results: Vec<ResultDto<'a>>,
}

/// Resolve `queries` and print them, as JSON or as an aligned listing.
///
/// Exits successfully when nothing matches, for the same reason `show` does:
/// "no symbol is named that" is an answer, not a failure.
pub fn run(
    tree: &ProjectTree,
    queries: &[String],
    limit: usize,
    json: bool,
    coverage: Option<&crate::restore::CoverageIndex>,
) -> Result<()> {
    if json {
        let results: Vec<ResultDto> = queries
            .iter()
            .map(|q| {
                let (hits, total) = search(tree, &Pattern::parse(q), limit);
                let read = coverage.map(|c| {
                    hits.iter()
                        .filter(|(_, n)| c.depth_of(&n.id).is_some())
                        .count()
                });
                ResultDto {
                    query: q,
                    matched: total,
                    truncated: total > hits.len(),
                    matches: hits
                        .into_iter()
                        .map(|(file, node)| {
                            crate::lookup::describe_summary(file, node, coverage)
                        })
                        .collect(),
                    read,
                }
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string(&FindDto {
                schema_version: SCHEMA_VERSION,
                coverage: crate::lookup::CoverageDto::of(coverage),
                results,
            })
            .wrap_err("serializing find results")?
        );
        return Ok(());
    }

    for (i, query) in queries.iter().enumerate() {
        if i > 0 {
            println!();
        }
        let (hits, total) = search(tree, &Pattern::parse(query), limit);
        if total == 0 {
            println!("{query} — no matches");
            continue;
        }
        let read = coverage.map(|c| {
            hits.iter()
                .filter(|(_, n)| c.depth_of(&n.id).is_some())
                .count()
        });
        match read {
            Some(r) => println!(
                "{query} — {total} match{} ({r} read)",
                if total == 1 { "" } else { "es" }
            ),
            None => println!("{query} — {total} match{}", if total == 1 { "" } else { "es" }),
        }

        let width = hits
            .iter()
            .map(|(_, s)| s.label.len())
            .max()
            .unwrap_or(0);
        for (file, sym) in &hits {
            // The depth column is omitted entirely without a journal: an empty
            // column would read as "unread" when the truth is "unknown".
            let depth = match coverage {
                Some(c) => match c.depth_of(&sym.id) {
                    Some(d) => format!("  {d}"),
                    None => "  —".to_string(),
                },
                None => String::new(),
            };
            println!(
                "  [{:width$}] {}::{}  L{}-{}{}",
                sym.label,
                file.display(),
                sym.name_path(),
                sym.line_range.start,
                sym.line_range.end,
                depth,
            );
        }
        if total > hits.len() {
            println!("  … {} more (use --limit)", total - hits.len());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::*;
    use crate::tracking::ReadDepth;

    fn tree() -> ProjectTree {
        project(vec![
            file(
                "src/ui/stats.rs",
                vec![sym_with_children(
                    "src/ui/stats.rs::App",
                    "App",
                    vec![sym("src/ui/stats.rs::App/render", "render")],
                )],
            ),
            file("src/tui.rs", vec![sym("src/tui.rs::render", "render")]),
            file("src/app.rs", vec![sym("src/app.rs::render", "render")]),
        ])
    }

    fn ids(tree: &ProjectTree, q: &str) -> Vec<String> {
        search(tree, &Pattern::parse(q), DEFAULT_LIMIT)
            .0
            .into_iter()
            .map(|(_, s)| s.id.clone())
            .collect()
    }

    /// The false positive that motivated component matching: a raw substring
    /// rule makes `ui` match `tui.rs`, because "ui" sits inside "tui".
    #[test]
    fn the_path_half_matches_components_not_substrings() {
        let t = tree();
        let hits = ids(&t, "ui::render");
        assert_eq!(hits, vec!["src/ui/stats.rs::App/render"]);
        assert!(
            !hits.iter().any(|i| i.contains("tui.rs")),
            "`ui` must not match `tui.rs`"
        );
    }

    #[test]
    fn a_path_segment_may_match_a_file_stem() {
        assert_eq!(ids(&tree(), "app::render"), vec!["src/app.rs::render"]);
    }

    #[test]
    fn multi_segment_paths_match_consecutively() {
        assert_eq!(
            ids(&tree(), "ui/stats::render"),
            vec!["src/ui/stats.rs::App/render"]
        );
    }

    /// A bare name matches the leaf everywhere, which is the "where is this
    /// defined" question.
    #[test]
    fn a_bare_name_matches_every_file() {
        assert_eq!(ids(&tree(), "render").len(), 3);
    }

    /// An empty name half enumerates a file — the case that replaces
    /// `--dump --filter` plus hand-built selectors.
    #[test]
    fn an_empty_name_half_enumerates_the_file() {
        assert_eq!(
            ids(&tree(), "src/ui/stats.rs::"),
            vec![
                "src/ui/stats.rs::App",
                "src/ui/stats.rs::App/render"
            ]
        );
    }

    /// Leaf matching is what keeps a common word from dragging in every
    /// symbol nested under a container of that name.
    #[test]
    fn the_name_half_matches_the_leaf_by_default() {
        let t = tree();
        assert!(
            ids(&t, "App").iter().all(|i| i.ends_with("::App")),
            "matching the leaf must not pull in App's children"
        );
    }

    /// …but a `/` says the caller means the nesting, so the whole path is
    /// matched and container members come back.
    #[test]
    fn a_slash_switches_to_matching_the_whole_name_path() {
        assert_eq!(
            ids(&tree(), "::App/"),
            vec!["src/ui/stats.rs::App/render"]
        );
    }

    #[test]
    fn matching_ignores_case_on_both_halves() {
        assert_eq!(ids(&tree(), "UI::RENDER").len(), 1);
    }

    #[test]
    fn a_query_matching_nothing_is_empty_rather_than_an_error() {
        assert!(ids(&tree(), "nosuchsymbol").is_empty());
    }

    /// A truncated result must say how much it withheld, or a caller reads it
    /// as the complete answer.
    #[test]
    fn truncation_reports_the_full_total() {
        let t = tree();
        let (hits, total) = search(&t, &Pattern::parse("render"), 2);
        assert_eq!(hits.len(), 2);
        assert_eq!(total, 3, "the total counts what was withheld");
    }

    /// A search returning many symbols must not spend most of its bytes on
    /// child ids — and for a search over test modules those ids are largely
    /// results in their own right, listed twice.
    #[test]
    fn children_are_summarized_rather_than_listed() {
        let t = tree();
        let (hits, _) = search(&t, &Pattern::parse("App"), DEFAULT_LIMIT);
        let (file, node) = hits[0];
        let dto = crate::lookup::describe_summary(file, node, None);
        let v = serde_json::to_value(&dto).unwrap();

        assert_eq!(v["children_count"], 1, "the count survives");
        assert!(
            v.get("children").is_none(),
            "the list itself does not, or find output balloons"
        );
        assert_eq!(
            v["id"], "src/ui/stats.rs::App",
            "the id is unchanged, so it still composes into `show`"
        );
    }

    fn coverage_for(ids: &[(&str, ReadDepth)]) -> crate::restore::CoverageIndex {
        let reads = ids
            .iter()
            .map(|(id, d)| (id.to_string(), ([0u8; 32], *d)))
            .collect();
        crate::restore::CoverageIndex::from_read_set(reads, "test-session")
    }

    fn dto_of(cov: Option<&crate::restore::CoverageIndex>) -> serde_json::Value {
        let t = tree();
        let (hits, _) = search(&t, &Pattern::parse("render"), DEFAULT_LIMIT);
        let (file, node) = hits
            .iter()
            .find(|(_, n)| n.id == "src/app.rs::render")
            .copied()
            .unwrap();
        serde_json::to_value(crate::lookup::describe_summary(file, node, cov)).unwrap()
    }

    /// The point of the annotation: a caller can tell, from a search alone,
    /// whether it needs to read the result.
    #[test]
    fn a_read_symbol_reports_the_depth_it_was_read_at() {
        let cov = coverage_for(&[("src/app.rs::render", ReadDepth::FullBody)]);
        assert_eq!(dto_of(Some(&cov))["read_depth"], "full");
    }

    #[test]
    fn an_unread_symbol_carries_no_depth() {
        let cov = coverage_for(&[("src/other.rs::thing", ReadDepth::FullBody)]);
        assert!(dto_of(Some(&cov)).get("read_depth").is_none());
    }

    /// Without a journal there is no coverage context at all, and that must
    /// not be reported the same way as "read nothing" — an agent told a symbol
    /// is unread will go read it; one told nothing is known should not.
    #[test]
    fn no_journal_is_distinguishable_from_nothing_read() {
        let unread = coverage_for(&[("src/other.rs::thing", ReadDepth::FullBody)]);

        // Both omit `read_depth` on the match itself…
        assert!(dto_of(Some(&unread)).get("read_depth").is_none());
        assert!(dto_of(None).get("read_depth").is_none());

        // …so the envelope is what distinguishes them.
        assert!(crate::lookup::CoverageDto::of(Some(&unread)).is_some());
        assert!(crate::lookup::CoverageDto::of(None).is_none());
    }

    #[test]
    fn parse_splits_on_the_first_separator() {
        let p = Pattern::parse("src/ui::App/render");
        assert_eq!(p.path, vec!["src", "ui"]);
        assert_eq!(p.name, "app/render");
        assert!(p.nested);
    }
}
