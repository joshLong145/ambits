#[path = "support/fixtures.rs"]
mod fixtures;

use std::path::Path;

use ambits::coverage::CoverageReport;
use ambits::parser::LanguageParser;
use ambits::parser::python::PythonParser;
use ambits::parser::rust::RustParser;
use ambits::parser::typescript::TypescriptParser;
use ambits::tracking::{ContextLedger, ReadDepth};
use ambits::symbols::ProjectTree;
use fixtures::{py_source, ts_source, rust_source};

fn main() {
    divan::main();
}

/// Benchmark the Rust tree-sitter parser across source sizes.
#[divan::bench(args = ["tiny", "small", "medium", "large"])]
fn rust_parse_file(bencher: divan::Bencher, size: &&str) {
    let source = rust_source(size);
    let path = Path::new("src/bench.rs");
    let parser = RustParser::new();
    bencher.bench(|| {
        divan::black_box(parser.parse_file(divan::black_box(path), divan::black_box(source)).unwrap())
    });
}

/// Benchmark the TypeScript parser with TS-specific constructs (classes, interfaces, enums).
/// Also includes a `js_module` variant — pure JS is valid TS, verifying JS compat.
#[divan::bench(args = ["tiny", "small", "medium", "large", "js_module"])]
fn typescript_parse_file(bencher: divan::Bencher, size: &&str) {
    let source = ts_source(size);
    let path = Path::new("src/bench.ts");
    let parser = TypescriptParser::new();
    bencher.bench(|| {
        divan::black_box(parser.parse_file(divan::black_box(path), divan::black_box(source)).unwrap())
    });
}

/// Benchmark the Python parser across source sizes.
/// Exercises classes, decorators, async defs, dataclasses, and control-flow-embedded functions.
#[divan::bench(args = ["tiny", "small", "medium", "large"])]
fn python_parse_file(bencher: divan::Bencher, size: &&str) {
    let source = py_source(size);
    let path = Path::new("src/bench.py");
    let parser = PythonParser::new();
    bencher.bench(|| {
        divan::black_box(parser.parse_file(divan::black_box(path), divan::black_box(source)).unwrap())
    });
}

/// End-to-end pipeline: TypeScript parse → populate ledger → generate coverage report.
/// Measures the full ambit workflow for a single file.
#[divan::bench(args = ["tiny", "small", "medium", "large"])]
fn typescript_parse_and_coverage(bencher: divan::Bencher, size: &&str) {
    let source = ts_source(size);
    let path = Path::new("src/bench.ts");
    let parser = TypescriptParser::new();

    bencher.bench_local(|| {
        // 1. Parse
        let file_syms = parser.parse_file(path, source).unwrap();

        // 2. Build ledger: mark all symbols FullBody
        let mut ledger = ContextLedger::new();
        let hash = [0u8; 32];
        for sym in &file_syms.symbols {
            ledger.record(sym.id.clone(), ReadDepth::FullBody, hash, "agent".to_string(), sym.estimated_tokens);
        }

        // 3. Build project tree from the single file
        use std::path::PathBuf;
        let project = ProjectTree {
            root: PathBuf::from("/bench"),
            files: vec![file_syms],
        };

        // 4. Generate coverage report
        divan::black_box(CoverageReport::from_project(&project, &ledger, None))
    });
}
