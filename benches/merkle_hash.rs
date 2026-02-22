#[path = "support/fixtures.rs"]
mod fixtures;

use ambits::symbols::merkle::{compute_merkle_hash, content_hash, estimate_tokens};
use fixtures::{make_symbol_tree, rust_source};

fn main() {
    divan::main();
}

/// Benchmark `content_hash` (which internally calls `normalize_source`).
/// Covers the O(n) normalize + SHA256 pipeline across source sizes.
#[divan::bench(args = ["tiny", "small", "medium", "large"])]
fn bench_content_hash(bencher: divan::Bencher, size: &&str) {
    let source = rust_source(size);
    bencher.bench(|| {
        divan::black_box(content_hash(divan::black_box(source)))
    });
}

/// Benchmark `estimate_tokens` (O(1) division, but parameterized for throughput reporting).
#[divan::bench(args = ["tiny", "small", "medium", "large"])]
fn bench_estimate_tokens(bencher: divan::Bencher, size: &&str) {
    let source = rust_source(size);
    bencher.bench(|| {
        divan::black_box(estimate_tokens(divan::black_box(source)))
    });
}

/// Benchmark `compute_merkle_hash` over trees of varying node counts.
/// The arg is the number of leaf nodes; tree depth is fixed at 3, breadth scales.
#[divan::bench(args = [1, 10, 100, 1000])]
fn bench_compute_merkle_hash(bencher: divan::Bencher, n_nodes: &usize) {
    // Build the tree outside the timed region.
    let breadth = (*n_nodes as f64).cbrt().ceil() as usize;
    let breadth = breadth.max(1);
    bencher.with_inputs(|| make_symbol_tree(3, breadth)).bench_local_values(|mut tree| {
        compute_merkle_hash(divan::black_box(&mut tree));
    });
}
