//! Benchmarks for `ambits find`.
//!
//! The pair that matters is `search_without_matches` against
//! `search_with_matches` over the same corpus. A search parses only the files
//! that matched, so the first should stay far below the second — if the
//! prefilter is ever removed or defeated, they converge, and that convergence is
//! what these thresholds guard.

#[path = "support/fixtures.rs"]
mod fixtures;

use std::path::PathBuf;
use std::sync::OnceLock;

use ambits::find::{search, Matcher, Options};
use ambits::parser::ParserRegistry;
use fixtures::rust_source;

fn main() {
    divan::main();
}

/// A temporary tree and the `(absolute, project-relative)` pairs a walk would
/// have produced for it.
type Corpus = (tempfile::TempDir, Vec<(PathBuf, PathBuf)>);

/// A corpus of `n` Rust files, built once and reused by every iteration.
///
/// Building it per iteration would measure `tempfile` and the filesystem rather
/// than the search, and would dwarf both. The files are the `large` fixture
/// because the whole point of the comparison is the parse a search avoids, and
/// a toy file has almost no parse to avoid.
fn corpus(n: usize) -> &'static Corpus {
    static CORPORA: OnceLock<std::sync::Mutex<Vec<(usize, &'static Corpus)>>> = OnceLock::new();

    let cache = CORPORA.get_or_init(|| std::sync::Mutex::new(Vec::new()));
    let mut cache = cache.lock().unwrap();
    if let Some((_, built)) = cache.iter().find(|(size, _)| *size == n) {
        return built;
    }

    let dir = tempfile::tempdir().expect("tempdir");
    let mut targets = Vec::with_capacity(n);
    for i in 0..n {
        let rel = PathBuf::from(format!("src/file_{i}.rs"));
        let abs = dir.path().join(&rel);
        std::fs::create_dir_all(abs.parent().unwrap()).unwrap();
        std::fs::write(&abs, rust_source("large")).unwrap();
        targets.push((abs, rel));
    }

    // Leaked deliberately: the corpus must outlive every benchmark iteration,
    // and the process is about to exit anyway.
    let built: &'static _ = Box::leak(Box::new((dir, targets)));
    cache.push((n, built));
    built
}

fn options(pattern: &str) -> Options {
    Options::new(vec![pattern.to_string()])
}

/// The prefilter path: every file is read, none can match, so none is parsed.
#[divan::bench(args = [8, 64])]
fn search_without_matches(bencher: divan::Bencher, files: &usize) {
    let (_dir, targets) = corpus(*files);
    let registry = ParserRegistry::new();
    let opts = options("zzz_no_such_identifier_zzz");
    let matcher = Matcher::new(&opts).unwrap();

    bencher.bench(|| {
        divan::black_box(search(
            divan::black_box(&matcher),
            &registry,
            targets,
            &opts,
            None,
        ))
    });
}

/// The full path: every file matches, so every file is parsed and every hit
/// attributed to its enclosing symbol.
#[divan::bench(args = [8, 64])]
fn search_with_matches(bencher: divan::Bencher, files: &usize) {
    let (_dir, targets) = corpus(*files);
    let registry = ParserRegistry::new();
    // `access_history` is in the `large` fixture; a pattern that is not would
    // silently turn this into a second copy of the benchmark above.
    let opts = options("access_history");
    let matcher = Matcher::new(&opts).unwrap();

    bencher.bench(|| {
        divan::black_box(search(
            divan::black_box(&matcher),
            &registry,
            targets,
            &opts,
            None,
        ))
    });
}

/// Pattern compilation, which every invocation pays once.
#[divan::bench(args = ["literal", "alternation", "classes"])]
fn compile_pattern(bencher: divan::Bencher, shape: &&str) {
    let opts = match *shape {
        "literal" => options("depth_of"),
        "alternation" => Options::new(vec!["depth_of".into(), "content_hash".into(), "fn \\w+".into()]),
        _ => options(r"(?:pub\s+)?fn\s+[a-z_]+\([^)]*\)\s*->\s*[A-Za-z<>:]+"),
    };
    bencher.bench(|| divan::black_box(Matcher::new(divan::black_box(&opts)).unwrap()));
}
