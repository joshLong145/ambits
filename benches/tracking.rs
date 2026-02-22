#[path = "support/fixtures.rs"]
mod fixtures;

use ambits::tracking::{ContextLedger, ReadDepth};

fn main() {
    divan::main();
}

/// Benchmark inserting N symbols into a fresh ledger.
/// Each symbol is a new entry — measures HashMap insertion + per-agent depth setup.
#[divan::bench(args = [10, 100, 1000, 10000])]
fn record_n_symbols(bencher: divan::Bencher, n: &usize) {
    let n = *n;
    let symbols: Vec<String> = (0..n).map(|i| format!("src/bench.rs::sym_{i}")).collect();
    let hash = [42u8; 32];

    bencher.with_inputs(ContextLedger::new).bench_local_values(|mut ledger| {
        for id in &symbols {
            ledger.record(id.clone(), ReadDepth::Overview, hash, "agent-0".to_string(), 30);
        }
        divan::black_box(ledger)
    });
}

/// Benchmark upgrading depth for the same N symbols across 3 agents.
/// Exercises the "never downgrade" logic and aggregate max recomputation.
#[divan::bench(args = [10, 100, 1000])]
fn record_upgrade_n_symbols(bencher: divan::Bencher, n: &usize) {
    let n = *n;
    let symbols: Vec<String> = (0..n).map(|i| format!("src/bench.rs::sym_{i}")).collect();
    let hash = [1u8; 32];
    let agents = ["agent-0", "agent-1", "agent-2"];
    let depths = [ReadDepth::NameOnly, ReadDepth::Overview, ReadDepth::FullBody];

    bencher.with_inputs(ContextLedger::new).bench_local_values(|mut ledger| {
        for (agent, depth) in agents.iter().zip(depths.iter()) {
            for id in &symbols {
                ledger.record(id.clone(), *depth, hash, agent.to_string(), 30);
            }
        }
        divan::black_box(ledger)
    });
}

/// Benchmark `depth_of` lookups over a pre-populated ledger of N symbols.
/// Measures HashMap lookup performance (O(1) average).
#[divan::bench(args = [10, 100, 1000, 10000])]
fn depth_of_n_symbols(bencher: divan::Bencher, n: &usize) {
    let n = *n;
    let symbols: Vec<String> = (0..n).map(|i| format!("src/bench.rs::sym_{i}")).collect();
    let hash = [0u8; 32];

    let mut ledger = ContextLedger::new();
    for id in &symbols {
        ledger.record(id.clone(), ReadDepth::FullBody, hash, "agent-0".to_string(), 30);
    }

    bencher.bench(|| {
        let mut sum = 0usize;
        for id in &symbols {
            let depth = ledger.depth_of(divan::black_box(id));
            sum += depth.is_seen() as usize;
        }
        divan::black_box(sum)
    });
}

/// Benchmark `mark_stale_if_changed` for N symbols where hash has changed.
/// Exercises the stale-marking path including per-agent depth updates.
#[divan::bench(args = [10, 100, 1000])]
fn mark_stale_n_symbols(bencher: divan::Bencher, n: &usize) {
    let n = *n;
    let symbols: Vec<String> = (0..n).map(|i| format!("src/bench.rs::sym_{i}")).collect();
    let old_hash = [1u8; 32];
    let new_hash = [2u8; 32];

    bencher
        .with_inputs(|| {
            let mut ledger = ContextLedger::new();
            for id in &symbols {
                ledger.record(id.clone(), ReadDepth::FullBody, old_hash, "agent-0".to_string(), 30);
            }
            ledger
        })
        .bench_local_values(|mut ledger| {
            for id in &symbols {
                ledger.mark_stale_if_changed(divan::black_box(id), divan::black_box(new_hash));
            }
            divan::black_box(ledger)
        });
}
