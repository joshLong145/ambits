//! Project-scale benches: simulate operating on a synthetic monorepo's
//! symbol tree (up to ~500k symbols). These do NOT invoke tree-sitter —
//! they target the code paths downstream of parsing: merkle hashing,
//! coverage report construction, and ledger population.
//!
//! Tier sizing (`SYMBOLS_PER_FILE = 100`):
//!   100   files →  10,000 symbols
//!   1,000 files → 100,000 symbols
//!   5,000 files → 500,000 symbols   (top tier for monorepo target)

#[path = "support/fixtures.rs"]
mod fixtures;

use ambits::coverage::CoverageReport;
use ambits::symbols::ProjectTree;
use ambits::symbols::merkle::compute_merkle_hash;
use ambits::tracking::{ContextLedger, ReadDepth};
use fixtures::make_monorepo_tree;

fn main() {
    divan::main();
}

const SYMBOLS_PER_FILE: usize = 100;

/// Time `compute_merkle_hash` over every top-level symbol in a project-scale tree.
/// Captures merkle pass cost on monorepo loads — this is the operation parsers
/// perform once per file after symbol extraction.
#[divan::bench(args = [100, 1_000, 5_000], sample_count = 20)]
fn compute_merkle_hash_project(bencher: divan::Bencher, n_files: &usize) {
    bencher
        .with_inputs(|| make_monorepo_tree(*n_files, SYMBOLS_PER_FILE))
        .bench_local_values(|mut tree| {
            for file in tree.files.iter_mut() {
                for sym in file.symbols.iter_mut() {
                    compute_merkle_hash(divan::black_box(sym));
                }
            }
            divan::black_box(tree)
        });
}

/// Time `CoverageReport::from_project` on a fully-populated ledger. Models
/// the end-of-session coverage report at monorepo scale.
#[divan::bench(args = [100, 1_000], sample_count = 20)]
fn coverage_full_project(bencher: divan::Bencher, n_files: &usize) {
    let project = make_monorepo_tree(*n_files, SYMBOLS_PER_FILE);
    let ledger = populate_full(&project);
    bencher.bench(|| {
        divan::black_box(CoverageReport::from_project(
            divan::black_box(&project),
            divan::black_box(&ledger),
            None,
        ))
    });
}

/// Time recording every symbol in the project tree into a fresh ledger. Models
/// the cost of ingesting a session that read the entire repo once.
#[divan::bench(args = [100, 1_000], sample_count = 20)]
fn ledger_populate_full_project(bencher: divan::Bencher, n_files: &usize) {
    let project = make_monorepo_tree(*n_files, SYMBOLS_PER_FILE);
    bencher
        .with_inputs(ContextLedger::new)
        .bench_local_values(|mut ledger| {
            for file in &project.files {
                for sym in &file.symbols {
                    ledger.record(
                        sym.id.clone(),
                        ReadDepth::Overview,
                        sym.content_hash,
                        "agent-0".to_string(),
                        sym.estimated_tokens as usize,
                    );
                }
            }
            divan::black_box(ledger)
        });
}

fn populate_full(project: &ProjectTree) -> ContextLedger {
    let mut ledger = ContextLedger::new();
    for file in &project.files {
        for sym in &file.symbols {
            ledger.record(
                sym.id.clone(),
                ReadDepth::FullBody,
                sym.content_hash,
                "agent-0".to_string(),
                sym.estimated_tokens as usize,
            );
        }
    }
    ledger
}
