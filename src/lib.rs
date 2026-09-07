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
pub mod coverage;
pub mod digest;
pub mod filter;
pub mod find;
pub mod fmt;
pub mod ingest;
pub mod journal;
pub mod lookup;
pub mod parser;
pub mod restore;
pub mod symbols;
pub mod tracking;
