//! Render a [`RestoreReport`] as something worth putting in an agent's
//! context after a compaction.
//!
//! ## What this is competing with
//!
//! After a compaction the agent is handed a prose summary written by a model
//! that was itself losing the context it was describing. This is the
//! alternative: a deterministic list of symbols that were demonstrably read
//! and demonstrably have not changed since. Its whole claim to being worth
//! tokens is that every line is verified rather than remembered.
//!
//! That is also why the omissions are printed. Saying "these files changed,
//! re-read them if you need them" is a fact the summary cannot supply, and it
//! is the part that stops an agent trusting knowledge it no longer has.
//!
//! ## Budgeting
//!
//! The budget is measured against **rendered output**, not
//! `SymbolNode.estimated_tokens`. The latter measures how big a symbol's
//! *source* is; the digest emits its *name*. Budgeting on source size would
//! under-fill by more than an order of magnitude.
//!
//! Files are emitted in descending order of the source tokens they represent,
//! so when the budget runs out it has been spent on the files that save the
//! most re-reading. Depth deliberately plays no part: any depth of read is
//! worth restoring, so ranking by it would be inventing a preference the data
//! does not support.

use std::path::Path;

use serde::Serialize;

use crate::restore::{RestoreReport, RestoredSymbol};

/// Rough chars-per-token, matching the approximation used elsewhere in the
/// codebase (`symbols::merkle::estimate_tokens`). Good enough for a budget;
/// nothing downstream depends on it being exact.
const CHARS_PER_TOKEN: usize = 4;

/// Default budget for injected context. Large enough to be useful, small
/// enough that it never competes with the context it is trying to restore.
///
/// Raised from 2,000 when line ranges were added. Ranges cost roughly 16% more
/// per symbol, which at the old budget would have bought precision by dropping
/// symbols — the wrong trade, since a symbol that goes unnamed is one the agent
/// does not know it has. The extra 1,000 tokens buys both.
pub const DEFAULT_MAX_TOKENS: usize = 3_000;

/// How to render a digest. Mirrors [`crate::coverage::CoverageFormatter`].
pub trait DigestFormatter {
    fn format(&self, report: &RestoreReport, max_tokens: usize) -> String;
}

/// One file's worth of restored symbols, with its aggregate weight.
struct FileGroup<'a> {
    path: &'a Path,
    symbols: Vec<&'a RestoredSymbol>,
    /// Summed source tokens — how much re-reading this file would cost.
    source_tokens: u64,
}

/// Group restored symbols by file, heaviest first.
fn grouped(report: &RestoreReport) -> Vec<FileGroup<'_>> {
    let mut groups: Vec<FileGroup> = report
        .outcome
        .restored_by_file()
        .into_iter()
        .map(|(path, symbols)| {
            let source_tokens = symbols.iter().map(|s| s.estimated_tokens as u64).sum();
            FileGroup {
                path,
                symbols,
                source_tokens,
            }
        })
        .collect();
    groups.sort_by(|a, b| {
        b.source_tokens
            .cmp(&a.source_tokens)
            .then_with(|| a.path.cmp(b.path))
    });
    groups
}

/// Render one symbol as `name:start-end`.
///
/// The line range comes from the tree node whose hash matched, so it describes
/// the code as it is *now*, not as it was when read. That distinction is what
/// makes it safe to print: the journal deliberately stores no coordinates,
/// because a symbol keeps its content hash when code above it shifts, and any
/// stored line number would silently rot in exactly the files being edited
/// most.
///
/// It earns its ~16% size cost downstream: an agent that knows a symbol sits
/// at `src/app.rs:227-280` can read 53 lines instead of pulling a 40k-token
/// file or spending a `find_symbol` round-trip to locate it.
///
/// A symbol that moved additionally carries where it moved *from*, because the
/// agent is holding that stale address. Told only the new location it cannot
/// connect it to what it remembers; told only the old one it reads the wrong
/// file. Which half of the old id is shown depends on what actually changed —
/// naming the old file is useless when the symbol never left it.
fn symbol_label(sym: &RestoredSymbol) -> String {
    let base = format!(
        "{}:{}-{}",
        sym.name_path, sym.line_range.start, sym.line_range.end
    );
    let Some(ref old_id) = sym.moved_from else {
        return base;
    };
    match crate::restore::split_symbol_id(old_id) {
        Some((old_file, _)) if Path::new(old_file) != sym.file_path => {
            format!("{base} (was {old_file})")
        }
        Some((_, old_name)) => format!("{base} (was {old_name})"),
        None => format!("{base} (was {old_id})"),
    }
}

/// Fit as many comma-separated names as `budget` chars allow, returning the
/// rendered list and how many were left out.
///
/// Always emits at least one name — a file heading with no symbols under it
/// tells the reader nothing they can act on.
fn fit_names(names: &[String], budget: usize) -> (String, usize) {
    let mut out = String::new();
    let mut used = 0usize;
    for (i, name) in names.iter().enumerate() {
        let cost = name.len() + if i == 0 { 0 } else { 2 }; // ", "
        if i > 0 && used + cost > budget {
            return (out, names.len() - i);
        }
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(name);
        used += cost;
    }
    (out, 0)
}


fn format_tokens(n: u64) -> String {
    if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

/// Markdown digest, intended to be pasted or piped straight into a session.
#[derive(Debug, Clone, Default)]
pub struct MarkdownFormatter;

impl DigestFormatter for MarkdownFormatter {
    fn format(&self, report: &RestoreReport, max_tokens: usize) -> String {
        let mut out = String::new();
        let budget_chars = max_tokens.saturating_mul(CHARS_PER_TOKEN);

        // The title carries the verification claim, so it has to track the
        // source — a heading that says "Verified" above an UNVERIFIED body
        // is worse than no heading at all.
        out.push_str(if report.source.verifies_drift() {
            "## Verified prior reads (ambit)"
        } else {
            "## Prior reads (ambit) — UNVERIFIED"
        });
        if let Some(ref sid) = report.session_id {
            out.push_str(&format!(" — session {sid}"));
        }
        out.push('\n');

        if report.source.verifies_drift() {
            out.push_str(
                "These symbols were read earlier in this session and are unchanged since.\n\
                 Treat them as known. Anything not listed here is not covered.\n\
                 Each entry is `name:first-last`, current line numbers — read that range\n\
                 directly rather than re-reading the whole file.\n\n",
            );
        } else {
            out.push_str(
                "Recovered from session logs, which do not record what a file looked like\n\
                 when it was read. These symbols were read at some point, but whether they\n\
                 have changed since is unknown. Re-read before relying on them.\n\
                 Each entry is `name:first-last`, current line numbers.\n\n",
            );
        }

        if report.outcome.restored.is_empty() {
            out.push_str("_No prior reads recovered._\n");
            return out;
        }

        let groups = grouped(report);
        let mut emitted = 0usize;

        for group in &groups {
            let header = format!(
                "### {} — {} symbol{} (~{} tok)\n",
                group.path.display(),
                group.symbols.len(),
                if group.symbols.len() == 1 { "" } else { "s" },
                format_tokens(group.source_tokens),
            );

            // Always emit at least one file: a digest that fits perfectly but
            // says nothing is worse than one that overruns slightly.
            if emitted > 0 && out.len() + header.len() > budget_chars {
                break;
            }

            let names: Vec<String> = group.symbols.iter().map(|s| symbol_label(s)).collect();
            let room = budget_chars.saturating_sub(out.len() + header.len());
            let (listed, hidden) = fit_names(&names, room);

            out.push_str(&header);
            out.push_str(&listed);
            if hidden > 0 {
                out.push_str(&format!(" … (+{hidden} more)"));
            }
            out.push_str("\n\n");
            emitted += 1;

            // A file whose symbol list had to be trimmed has already consumed
            // the remaining budget; anything after it would be a lie about
            // fitting.
            if hidden > 0 {
                break;
            }
        }

        if emitted < groups.len() {
            out.push_str(&format!(
                "_{} more file{} omitted to stay within the {max_tokens}-token budget._\n\n",
                groups.len() - emitted,
                if groups.len() - emitted == 1 { "" } else { "s" },
            ));
        }

        append_omissions(&mut out, report);
        out
    }
}

/// Report what is deliberately *not* covered. Only meaningful when drift was
/// actually checked — the session-log fallback cannot detect any of it, so
/// claiming "nothing changed" there would be a lie by omission.
fn append_omissions(out: &mut String, report: &RestoreReport) {
    if !report.source.verifies_drift() {
        return;
    }
    let outcome = &report.outcome;

    if !outcome.drifted.is_empty() {
        let mut files: Vec<&Path> = outcome.drifted.iter().map(|s| s.file_path.as_path()).collect();
        files.sort_unstable();
        files.dedup();
        let list: Vec<String> = files.iter().map(|p| p.display().to_string()).collect();
        out.push_str(&format!(
            "**Changed since reading** ({} symbol{} in {}): re-read if you need them.\n",
            outcome.drifted.len(),
            if outcome.drifted.len() == 1 { "" } else { "s" },
            list.join(", "),
        ));
    }

    if !outcome.removed.is_empty() {
        out.push_str(&format!(
            "**No longer present** ({} symbol{}): {}\n",
            outcome.removed.len(),
            if outcome.removed.len() == 1 { "" } else { "s" },
            outcome
                .removed
                .iter()
                .map(|s| s.symbol_id.as_str())
                .take(10)
                .collect::<Vec<_>>()
                .join(", "),
        ));
    }
}

/// Claude Code `SessionStart` hook envelope.
///
/// Wraps the markdown digest in the exact shape Claude Code parses from a
/// hook's stdout, so `settings.json` can invoke `ambits` directly — no wrapper
/// script and no `jq` dependency to escape the payload.
///
/// Emits **nothing** when there is nothing to restore. Empty stdout injects
/// nothing, which makes the hook self-regulating: silent on a fresh session,
/// useful after a compaction.
#[derive(Debug, Clone, Default)]
pub struct HookFormatter;

#[derive(Serialize)]
struct HookSpecificOutput<'a> {
    #[serde(rename = "hookEventName")]
    hook_event_name: &'static str,
    #[serde(rename = "additionalContext")]
    additional_context: &'a str,
}

#[derive(Serialize)]
struct HookEnvelope<'a> {
    #[serde(rename = "hookSpecificOutput")]
    hook_specific_output: HookSpecificOutput<'a>,
}

impl DigestFormatter for HookFormatter {
    fn format(&self, report: &RestoreReport, max_tokens: usize) -> String {
        if report.outcome.restored.is_empty() {
            return String::new();
        }
        let body = MarkdownFormatter.format(report, max_tokens);
        let envelope = HookEnvelope {
            hook_specific_output: HookSpecificOutput {
                hook_event_name: "SessionStart",
                additional_context: &body,
            },
        };
        // serde handles the JSON string escaping the payload needs.
        serde_json::to_string(&envelope).unwrap_or_default()
    }
}

/// Machine-readable digest. Schema-versioned like the coverage report's JSON.
#[derive(Debug, Clone, Default)]
pub struct JsonFormatter;

#[derive(Serialize)]
struct SymbolDto<'a> {
    id: &'a str,
    name: &'a str,
    lines: [u32; 2],
    depth: String,
    tokens: u32,
    /// Present only when the symbol was located by content hash under a
    /// different id — see `RestoredSymbol::moved_from`.
    #[serde(skip_serializing_if = "Option::is_none")]
    moved_from: Option<&'a str>,
}

#[derive(Serialize)]
struct FileDto<'a> {
    path: String,
    symbols: Vec<SymbolDto<'a>>,
    source_tokens: u64,
}

#[derive(Serialize)]
struct OmittedDto<'a> {
    id: &'a str,
    path: String,
}

#[derive(Serialize)]
struct ReportDto<'a> {
    schema_version: u32,
    source: &'static str,
    /// False for the session-log fallback: reads are listed but unverified.
    drift_verified: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    session_id: Option<&'a str>,
    restored_symbols: usize,
    files: Vec<FileDto<'a>>,
    drifted: Vec<OmittedDto<'a>>,
    removed: Vec<OmittedDto<'a>>,
    #[serde(skip_serializing_if = "<[String]>::is_empty")]
    warnings: &'a [String],
}

impl DigestFormatter for JsonFormatter {
    fn format(&self, report: &RestoreReport, max_tokens: usize) -> String {
        let budget_chars = max_tokens.saturating_mul(CHARS_PER_TOKEN);
        let groups = grouped(report);

        let mut files = Vec::new();
        let mut used = 0usize;
        for group in &groups {
            let symbols: Vec<SymbolDto> = group
                .symbols
                .iter()
                .map(|s| SymbolDto {
                    id: &s.symbol_id,
                    name: &s.name_path,
                    lines: [s.line_range.start, s.line_range.end],
                    depth: s.depth.to_string(),
                    tokens: s.estimated_tokens,
                    moved_from: s.moved_from.as_deref(),
                })
                .collect();
            // Approximate the rendered cost the same way the markdown path
            // does, so `--max-tokens` means the same thing in both formats.
            let cost: usize = group.symbols.iter().map(|s| s.name_path.len() + 2).sum();
            if !files.is_empty() && used + cost > budget_chars {
                break;
            }
            used += cost;
            files.push(FileDto {
                path: group.path.display().to_string(),
                symbols,
                source_tokens: group.source_tokens,
            });
        }

        let dto = ReportDto {
            schema_version: 2,
            source: match report.source {
                crate::restore::RestoreSource::Journal => "journal",
                crate::restore::RestoreSource::SessionLogs => "session_logs",
            },
            drift_verified: report.source.verifies_drift(),
            session_id: report.session_id.as_deref(),
            restored_symbols: report.outcome.restored.len(),
            files,
            drifted: report
                .outcome
                .drifted
                .iter()
                .map(|s| OmittedDto {
                    id: &s.symbol_id,
                    path: s.file_path.display().to_string(),
                })
                .collect(),
            removed: report
                .outcome
                .removed
                .iter()
                .map(|s| OmittedDto {
                    id: &s.symbol_id,
                    path: s.file_path.display().to_string(),
                })
                .collect(),
            warnings: &report.warnings,
        };
        serde_json::to_string(&dto).unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::restore::{OmissionReason, OmittedSymbol, RestoreOutcome, RestoreSource};
    use crate::tracking::ReadDepth;
    use std::path::PathBuf;

    fn names_of<const N: usize>(names: [&str; N]) -> Vec<String> {
        names.iter().map(|n| n.to_string()).collect()
    }

    fn restored(file: &str, name: &str, tokens: u32) -> RestoredSymbol {
        RestoredSymbol {
            symbol_id: format!("{file}::{name}"),
            file_path: PathBuf::from(file),
            name_path: name.to_string(),
            depth: ReadDepth::FullBody,
            line_range: 1..10,
            estimated_tokens: tokens,
            moved_from: None,
        }
    }

    fn report(restored_syms: Vec<RestoredSymbol>, source: RestoreSource) -> RestoreReport {
        RestoreReport {
            outcome: RestoreOutcome {
                restored: restored_syms,
                drifted: vec![],
                removed: vec![],
            },
            source,
            session_id: Some("sess".into()),
            warnings: vec![],
        }
    }

    #[test]
    fn markdown_lists_symbols_grouped_by_file() {
        let r = report(
            vec![
                restored("a.rs", "one", 10),
                restored("a.rs", "two", 10),
                restored("b.rs", "three", 5),
            ],
            RestoreSource::Journal,
        );
        let out = MarkdownFormatter.format(&r, DEFAULT_MAX_TOKENS);
        assert!(out.contains("### a.rs — 2 symbols"));
        assert!(out.contains("one:1-10, two:1-10"));
        assert!(out.contains("### b.rs — 1 symbol"));
        assert!(out.contains("unchanged since"));
    }

    /// A moved symbol lists under where it lives now, but must name where it
    /// came from — that stale address is what the agent is still holding.
    #[test]
    fn a_moved_symbol_names_the_file_it_came_from() {
        let mut sym = restored("src/digest.rs", "format_tokens", 40);
        sym.line_range = 126..132;
        sym.moved_from = Some("src/ui/stats.rs::format_tokens".into());
        let r = report(vec![sym], RestoreSource::Journal);

        let out = MarkdownFormatter.format(&r, DEFAULT_MAX_TOKENS);
        assert!(out.contains("### src/digest.rs"), "grouped by where it is now");
        assert!(out.contains("format_tokens:126-132 (was src/ui/stats.rs)"));
    }

    /// When it never left the file, naming the file says nothing — the old
    /// name path is the part that moved.
    #[test]
    fn a_symbol_reparented_within_a_file_names_the_old_path() {
        let mut sym = restored("src/app.rs", "Helper/run", 40);
        sym.line_range = 10..20;
        sym.moved_from = Some("src/app.rs::App/run".into());
        let r = report(vec![sym], RestoreSource::Journal);

        let out = MarkdownFormatter.format(&r, DEFAULT_MAX_TOKENS);
        assert!(out.contains("Helper/run:10-20 (was App/run)"));
    }

    #[test]
    fn json_carries_moved_from_only_when_set() {
        let mut moved = restored("b.rs", "x", 10);
        moved.moved_from = Some("a.rs::x".into());
        let r = report(vec![moved, restored("b.rs", "y", 10)], RestoreSource::Journal);

        let out = JsonFormatter.format(&r, DEFAULT_MAX_TOKENS);
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        let syms = &v["files"][0]["symbols"];
        assert_eq!(syms[0]["moved_from"], "a.rs::x");
        assert!(
            syms[1].get("moved_from").is_none(),
            "omitted for the ordinary case rather than emitted as null"
        );
    }

    /// Line ranges are what let an agent read a slice instead of a whole file,
    /// so they have to survive into the rendered output and be explained.
    #[test]
    fn symbols_carry_their_current_line_range() {
        let mut sym = restored("src/app.rs", "App/process_compaction", 400);
        sym.line_range = 227..280;
        let r = report(vec![sym], RestoreSource::Journal);

        let out = MarkdownFormatter.format(&r, DEFAULT_MAX_TOKENS);
        assert!(out.contains("App/process_compaction:227-280"));
        assert!(
            out.contains("name:first-last"),
            "the notation is explained, or the numbers are ambiguous"
        );
    }

    /// Heaviest files first, so a truncated digest keeps the most valuable
    /// part.
    #[test]
    fn files_are_ordered_by_source_weight() {
        let r = report(
            vec![restored("light.rs", "a", 5), restored("heavy.rs", "b", 500)],
            RestoreSource::Journal,
        );
        let out = MarkdownFormatter.format(&r, DEFAULT_MAX_TOKENS);
        let heavy = out.find("heavy.rs").unwrap();
        let light = out.find("light.rs").unwrap();
        assert!(heavy < light, "heavier file should come first");
    }

    #[test]
    fn budget_truncates_and_says_so() {
        let syms: Vec<RestoredSymbol> = (0..40)
            .map(|i| restored(&format!("f{i}.rs"), &format!("symbol_number_{i}"), 100))
            .collect();
        let r = report(syms, RestoreSource::Journal);

        let big = MarkdownFormatter.format(&r, 4_000);
        let small = MarkdownFormatter.format(&r, 60);
        assert!(small.len() < big.len(), "a smaller budget yields less output");
        assert!(small.contains("omitted to stay within"));
        assert!(!big.contains("omitted to stay within"), "a large budget fits everything");
    }

    /// A single oversized file is still emitted — an empty digest helps nobody.
    #[test]
    fn always_emits_at_least_one_file() {
        let r = report(vec![restored("a.rs", &"x".repeat(500), 10)], RestoreSource::Journal);
        let out = MarkdownFormatter.format(&r, 1);
        assert!(out.contains("a.rs"));
    }

    /// One file with very many symbols must not blow the budget on its own —
    /// the symbol list is trimmed within the file rather than emitted whole.
    #[test]
    fn a_single_huge_file_is_trimmed_not_dumped() {
        let syms: Vec<RestoredSymbol> = (0..400)
            .map(|i| restored("big.rs", &format!("symbol_number_{i}"), 50))
            .collect();
        let r = report(syms, RestoreSource::Journal);
        let out = MarkdownFormatter.format(&r, 100);

        assert!(out.contains("(+"), "should say how many were withheld");
        assert!(out.contains("more)"));
        // Allow the header/footer prose some slack, but nothing like 400
        // symbols' worth.
        assert!(
            out.len() < 100 * CHARS_PER_TOKEN * 3,
            "budget overshot badly: {} chars",
            out.len()
        );
        // The count in the heading still reports the true total.
        assert!(out.contains("400 symbols"));
    }

    #[test]
    fn fit_names_keeps_one_name_minimum_and_counts_the_rest() {
        let (listed, hidden) = fit_names(&names_of(["alpha", "beta", "gamma"]), 0);
        assert_eq!(listed, "alpha", "at least one name always survives");
        assert_eq!(hidden, 2);

        let (listed, hidden) = fit_names(&names_of(["alpha", "beta", "gamma"]), 1_000);
        assert_eq!(listed, "alpha, beta, gamma");
        assert_eq!(hidden, 0);
    }

    #[test]
    fn session_log_source_is_labelled_unverified() {
        let r = report(vec![restored("a.rs", "one", 10)], RestoreSource::SessionLogs);
        let out = MarkdownFormatter.format(&r, DEFAULT_MAX_TOKENS);
        assert!(out.contains("UNVERIFIED"));
        assert!(!out.contains("unchanged since"), "must not claim verification it lacks");
        // The heading must not contradict the body.
        assert!(!out.contains("## Verified"), "heading claims verification it lacks");
    }

    /// Drift can't be detected from session logs, so we must not imply that
    /// an empty drift list means nothing changed.
    #[test]
    fn omissions_are_suppressed_for_the_unverified_source() {
        let mut r = report(vec![restored("a.rs", "one", 10)], RestoreSource::SessionLogs);
        r.outcome.drifted.push(OmittedSymbol {
            symbol_id: "b.rs::x".into(),
            file_path: PathBuf::from("b.rs"),
            name_path: "x".into(),
            reason: OmissionReason::Drifted,
        });
        let out = MarkdownFormatter.format(&r, DEFAULT_MAX_TOKENS);
        assert!(!out.contains("Changed since reading"));
    }

    #[test]
    fn omissions_are_reported_for_the_journal_source() {
        let mut r = report(vec![restored("a.rs", "one", 10)], RestoreSource::Journal);
        r.outcome.drifted.push(OmittedSymbol {
            symbol_id: "b.rs::x".into(),
            file_path: PathBuf::from("b.rs"),
            name_path: "x".into(),
            reason: OmissionReason::Drifted,
        });
        r.outcome.removed.push(OmittedSymbol {
            symbol_id: "c.rs::gone".into(),
            file_path: PathBuf::from("c.rs"),
            name_path: "gone".into(),
            reason: OmissionReason::Removed,
        });
        let out = MarkdownFormatter.format(&r, DEFAULT_MAX_TOKENS);
        assert!(out.contains("Changed since reading"));
        assert!(out.contains("b.rs"));
        assert!(out.contains("No longer present"));
        assert!(out.contains("c.rs::gone"));
    }

    #[test]
    fn empty_restore_says_so_rather_than_emitting_nothing() {
        let r = report(vec![], RestoreSource::Journal);
        let out = MarkdownFormatter.format(&r, DEFAULT_MAX_TOKENS);
        assert!(out.contains("No prior reads recovered"));
    }

    #[test]
    fn hook_envelope_matches_the_sessionstart_contract() {
        let r = report(vec![restored("a.rs", "one", 10)], RestoreSource::Journal);
        let out = HookFormatter.format(&r, DEFAULT_MAX_TOKENS);
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["hookSpecificOutput"]["hookEventName"], "SessionStart");
        let ctx = v["hookSpecificOutput"]["additionalContext"].as_str().unwrap();
        assert!(ctx.contains("### a.rs"), "markdown body is carried through");
        // Must be one line: Claude Code parses stdout as a single JSON value.
        assert!(!out.contains('\n'));
    }

    /// Empty stdout injects nothing, so the hook stays quiet on a fresh
    /// session instead of announcing that it has nothing to say.
    #[test]
    fn hook_emits_nothing_when_there_is_nothing_to_restore() {
        let r = report(vec![], RestoreSource::Journal);
        assert_eq!(HookFormatter.format(&r, DEFAULT_MAX_TOKENS), "");
    }

    #[test]
    fn hook_respects_the_budget() {
        let syms: Vec<RestoredSymbol> = (0..200)
            .map(|i| restored(&format!("f{i}.rs"), &format!("symbol_number_{i}"), 100))
            .collect();
        let r = report(syms, RestoreSource::Journal);
        let small = HookFormatter.format(&r, 50);
        let big = HookFormatter.format(&r, 4_000);
        assert!(small.len() < big.len());
    }

    #[test]
    fn json_is_schema_versioned_and_flags_verification() {
        let r = report(vec![restored("a.rs", "one", 10)], RestoreSource::Journal);
        let out = JsonFormatter.format(&r, DEFAULT_MAX_TOKENS);
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["schema_version"], 2);
        assert_eq!(v["source"], "journal");
        assert_eq!(v["drift_verified"], true);
        assert_eq!(v["restored_symbols"], 1);
        assert_eq!(v["files"][0]["path"], "a.rs");
        assert_eq!(v["files"][0]["symbols"][0]["name"], "one");
    }

    #[test]
    fn json_flags_the_unverified_source() {
        let r = report(vec![restored("a.rs", "one", 10)], RestoreSource::SessionLogs);
        let out = JsonFormatter.format(&r, DEFAULT_MAX_TOKENS);
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["source"], "session_logs");
        assert_eq!(v["drift_verified"], false);
    }

    #[test]
    fn json_respects_the_budget() {
        let syms: Vec<RestoredSymbol> = (0..40)
            .map(|i| restored(&format!("f{i}.rs"), &format!("symbol_number_{i}"), 100))
            .collect();
        let r = report(syms, RestoreSource::Journal);
        let small: serde_json::Value =
            serde_json::from_str(&JsonFormatter.format(&r, 5)).unwrap();
        let big: serde_json::Value =
            serde_json::from_str(&JsonFormatter.format(&r, 4_000)).unwrap();
        assert!(
            small["files"].as_array().unwrap().len() < big["files"].as_array().unwrap().len()
        );
        // The headline count always reflects everything restored, not just
        // what fit — otherwise a truncated digest would misreport coverage.
        assert_eq!(small["restored_symbols"], 40);
    }
}
