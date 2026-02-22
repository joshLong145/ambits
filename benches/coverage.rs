#[path = "support/fixtures.rs"]
mod fixtures;

use ambits::coverage::{count_symbols, CoverageReport};
use fixtures::{make_flat_symbols, make_populated_ledger, make_project, make_symbol_tree};

fn main() {
    divan::main();
}

/// Benchmark `CoverageReport::from_project` over projects of varying total symbol counts.
/// The arg is total symbols; split evenly across 10 files.
#[divan::bench(args = [10, 100, 1000, 10000])]
fn from_project_n_symbols(bencher: divan::Bencher, n: &usize) {
    let symbols_per_file = (*n / 10).max(1);
    let project = make_project(10, symbols_per_file);
    let ledger = make_populated_ledger(&project);
    bencher.bench(|| {
        divan::black_box(CoverageReport::from_project(
            divan::black_box(&project),
            divan::black_box(&ledger),
            None,
        ))
    });
}

/// Benchmark `count_symbols` over flat symbol arrays of varying size.
#[divan::bench(args = [10, 100, 1000])]
fn count_symbols_flat(bencher: divan::Bencher, n: &usize) {
    let n = *n;
    let symbols = make_flat_symbols(n);
    let project = make_project(1, n);
    let ledger = make_populated_ledger(&project);
    bencher.bench(|| {
        divan::black_box(count_symbols(
            divan::black_box(&symbols),
            divan::black_box(&ledger),
            None,
        ))
    });
}

/// Benchmark `count_symbols` with nested symbol trees of varying depth.
/// Each tree has depth d and breadth 3 (so total nodes ~= 3^d).
#[divan::bench(args = [2, 3, 5])]
fn count_symbols_nested_depth(bencher: divan::Bencher, depth: &usize) {
    let tree = make_symbol_tree(*depth, 3);
    let project = make_project(1, 1);
    let ledger = make_populated_ledger(&project);
    bencher.bench(|| {
        divan::black_box(count_symbols(
            divan::black_box(std::slice::from_ref(&tree)),
            divan::black_box(&ledger),
            None,
        ))
    });
}
