//! Project-scale heap profile harness.
//!
//! Build:   cargo build --release --features dhat-heap --bin heap_probe
//! Run:     ./target/release/heap_probe <n_files>
//!
//! Writes `dhat-heap.json` to CWD. View with `dh_view.html` or `dhat` from
//! https://valgrind.org/docs/manual/dh-manual.html — drop the JSON onto the
//! viewer to get an interactive breakdown of allocation sites and bytes.
//!
//! Default tier (omit arg) is 1000 files ≈ 100k symbols.

use std::path::PathBuf;
use std::sync::Arc;

use ambits::symbols::merkle::{compute_merkle_hash, content_hash};
use ambits::symbols::{FileSymbols, ProjectTree, SymbolCategory, SymbolNode};

#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

const SYMBOLS_PER_FILE: usize = 100;

fn main() {
    let _profiler = dhat::Profiler::new_heap();

    let n_files: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000);

    eprintln!(
        "heap_probe: building tree with {n_files} files × {SYMBOLS_PER_FILE} symbols/file",
    );

    let project = make_monorepo_tree(n_files, SYMBOLS_PER_FILE);

    // Drive a few representative operations against the tree so dhat sees
    // the steady-state allocation pattern of real consumers, not just the
    // initial construction.
    let mut total_symbols = 0usize;
    for file in &project.files {
        for sym in &file.symbols {
            total_symbols += count_subtree(sym);
        }
    }
    eprintln!("heap_probe: total symbols (including nested children) = {total_symbols}");

    // Keep the tree live until the profiler captures the heap.
    std::hint::black_box(&project);
}

fn count_subtree(node: &SymbolNode) -> usize {
    1 + node.children.iter().map(count_subtree).sum::<usize>()
}

// ─── Inlined monorepo fixture (mirrors benches/support/fixtures.rs) ──────────
// Duplicated here so the binary is self-contained; this lives outside any
// bench harness and is short enough that duplication beats the path-include
// gymnastics needed to share with the bench fixture file.

fn make_monorepo_tree(n_files: usize, symbols_per_file: usize) -> ProjectTree {
    const EXTENSIONS: [&str; 3] = ["rs", "ts", "py"];
    let files: Vec<FileSymbols> = (0..n_files)
        .map(|file_idx| {
            let ext = EXTENSIONS[file_idx % EXTENSIONS.len()];
            let pkg_idx = file_idx / 20;
            let path_str = format!("src/pkg_{pkg_idx:04}/module_{file_idx:06}.{ext}");
            let symbols = build_file_symbols(&path_str, symbols_per_file);
            FileSymbols {
                file_path: PathBuf::from(&path_str),
                symbols,
                total_lines: symbols_per_file * 5,
            }
        })
        .collect();
    ProjectTree {
        root: PathBuf::from("/bench/monorepo"),
        files,
    }
}

fn build_file_symbols(file_path: &str, n: usize) -> Vec<SymbolNode> {
    let groups = n / 5;
    let remainder = n - groups * 5;
    let mut out = Vec::with_capacity(groups + remainder);
    let file_path_arc = Arc::new(PathBuf::from(file_path));
    for g in 0..groups {
        let parent_id = format!("{file_path}::Type_{g:04}");
        let mut parent = make_sym(&parent_id, &format!("Type_{g:04}"));
        parent.category = SymbolCategory::Type;
        parent.label = "struct";
        parent.file_path = Arc::clone(&file_path_arc);
        parent.children = (0..4)
            .map(|i| {
                let child_id = format!("{parent_id}::method_{i}");
                let mut sym = make_sym(&child_id, &format!("method_{i}"));
                sym.file_path = Arc::clone(&file_path_arc);
                sym
            })
            .collect();
        compute_merkle_hash(&mut parent);
        out.push(parent);
    }
    for i in 0..remainder {
        let leaf_id = format!("{file_path}::fn_{i:04}");
        let mut sym = make_sym(&leaf_id, &format!("fn_{i:04}"));
        sym.file_path = Arc::clone(&file_path_arc);
        out.push(sym);
    }
    out
}

fn make_sym(id: &str, name: &str) -> SymbolNode {
    let hash = content_hash(name);
    SymbolNode {
        id: id.to_string(),
        name: name.to_string(),
        category: SymbolCategory::Function,
        label: "fn",
        file_path: Arc::new(PathBuf::from("src/bench.rs")),
        byte_range: 0..100,
        line_range: 1..10,
        content_hash: hash,
        merkle_hash: hash,
        children: Vec::new(),
        estimated_tokens: 30,
    }
}
