#[path = "support/fixtures.rs"]
mod fixtures;

use ambits::ingest::claude::{parse_jsonl_line, parse_log_file};
use ambits::ingest::tool_config::ToolMappingConfig;
use fixtures::{jsonl_malformed_line, jsonl_multi_tool_line, jsonl_read_line, make_jsonl_lines};

fn main() {
    divan::main();
}

/// Benchmark parsing a single-tool assistant JSONL line (the common case).
/// Exercises JSON deserialization → tool type dispatch → ReadDepth classification.
#[divan::bench]
fn parse_jsonl_line_read(bencher: divan::Bencher) {
    let line = jsonl_read_line();
    let cfg = ToolMappingConfig::builtin().unwrap();
    bencher.bench(|| {
        divan::black_box(parse_jsonl_line(divan::black_box(line), "default-agent", &cfg))
    });
}

/// Benchmark parsing a message with 5 tool_use blocks.
/// Tests the inner loop that iterates content array entries and dispatches each tool.
#[divan::bench]
fn parse_jsonl_line_multi_tool(bencher: divan::Bencher) {
    let line = jsonl_multi_tool_line();
    let cfg = ToolMappingConfig::builtin().unwrap();
    bencher.bench(|| {
        divan::black_box(parse_jsonl_line(divan::black_box(line.as_str()), "default-agent", &cfg))
    });
}

/// Benchmark the fast-fail path for malformed JSON.
/// serde_json early-exits on parse error; this measures that overhead.
#[divan::bench]
fn parse_jsonl_line_malformed(bencher: divan::Bencher) {
    let line = jsonl_malformed_line();
    let cfg = ToolMappingConfig::builtin().unwrap();
    bencher.bench(|| {
        divan::black_box(parse_jsonl_line(divan::black_box(line), "default-agent", &cfg))
    });
}

/// Benchmark `parse_log_file` over files of varying line counts.
/// Uses `tempfile` for realistic I/O. The file write happens in `with_inputs` (setup phase).
#[divan::bench(args = [10, 100, 1000, 10000])]
fn parse_log_file_n_lines(bencher: divan::Bencher, n: &usize) {
    let n = *n;
    let cfg = ToolMappingConfig::builtin().unwrap();
    bencher
        .with_inputs(|| {
            let content = make_jsonl_lines(n);
            let mut tmp = tempfile::NamedTempFile::new().unwrap();
            use std::io::Write;
            tmp.write_all(content.as_bytes()).unwrap();
            tmp
        })
        .bench_local_values(|tmp| {
            divan::black_box(parse_log_file(divan::black_box(tmp.path()), &cfg))
        });
}
