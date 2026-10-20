// Test fixtures shared by every module's unit tests. Declared once here:
// five modules each declared their own `#[path]` copy, which built the same
// file five times over and is what clippy's "loaded as a module multiple
// times" warning was pointing at.
#[cfg(test)]
#[path = "../tests/helpers/mod.rs"]
#[allow(dead_code)]
pub mod helpers;

pub mod app;
pub mod cache;
pub mod callers;
pub mod coverage;
pub mod digest;
pub mod editor;
pub mod expansion;
pub mod filter;
pub mod fmt;
pub mod git;
pub mod git_hook;
pub mod ingest;
pub mod journal;
pub mod linkage;
pub mod logging;
pub mod lookup;
pub mod objects;
pub mod output;
pub mod parser;
pub mod remote;
pub mod report;
pub mod restore;
pub mod search;
pub mod state_dir;
pub mod symbols;
pub mod text;
pub mod time;
pub mod touched;
pub mod trace;
pub mod tracking;
pub mod write_index;
pub mod writes;

/// This build's version: the crate's, and the commit it was built from when
/// the build knew it (`0.20.0 (8c47137)`).
pub fn version() -> &'static str {
    static VERSION: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    VERSION.get_or_init(|| match env!("AMBITS_COMMIT") {
        "" => env!("CARGO_PKG_VERSION").to_string(),
        commit => format!("{} ({commit})", env!("CARGO_PKG_VERSION")),
    })
}
