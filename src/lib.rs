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
pub mod ingest;
pub mod journal;
pub mod logging;
pub mod lookup;
pub mod parser;
pub mod restore;
pub mod search;
pub mod state_dir;
pub mod symbols;
pub mod tracking;
