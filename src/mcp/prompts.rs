//! MCP prompt templates for ambit.
//!
//! Exposes a single prompt — `coverage_analysis` — that bakes the coverage
//! interpretation guide into a ready-to-use message structure.  Clients can
//! retrieve this prompt via `prompts/get` to prime Claude with the correct
//! framing before asking it to interpret a coverage report.

use rmcp::model::{GetPromptResult, PromptMessage, PromptMessageContent, PromptMessageRole};

/// Name of the single prompt this server exposes.
pub const COVERAGE_ANALYSIS: &str = "coverage_analysis";

/// Interpretation guidance — compiled into the binary from the skill file so
/// the MCP server doesn't need the skill directory at runtime.
const COVERAGE_GUIDE: &str = include_str!("../../skills/ambit/coverage-guide.md");

/// Build the `GetPromptResult` for the `coverage_analysis` prompt.
///
/// `session_id` is an optional argument passed by the client; it is woven
/// into the message text to give Claude context about which session is being
/// discussed.
pub fn get_coverage_analysis(session_id: Option<&str>) -> GetPromptResult {
    let session_note = session_id
        .map(|id| format!("for session `{id}`"))
        .unwrap_or_else(|| "for the most recent session".into());

    let text = format!(
        "Please analyse the ambit coverage report {session_note}.\n\n\
         Use the following interpretation guide to explain the results, \
         identify knowledge gaps, and suggest areas that may need more attention.\n\n\
         ---\n\n\
         {COVERAGE_GUIDE}"
    );

    GetPromptResult {
        description: Some(
            "Interpret an ambit coverage report to identify knowledge gaps \
             and well-understood areas in the codebase."
                .into(),
        ),
        messages: vec![PromptMessage {
            role: PromptMessageRole::User,
            content: PromptMessageContent::text(text),
        }],
    }
}
