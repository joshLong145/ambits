//! Records the tree-sitter grammar versions this binary was built with, read
//! from `Cargo.lock`, so snapshot inputs (spec §6.1) name the grammars that
//! actually produced a tree rather than a hand-maintained list that drifts.

use std::fmt::Write as _;

fn main() {
    println!("cargo:rerun-if-changed=Cargo.lock");
    let lock = std::fs::read_to_string("Cargo.lock").unwrap_or_default();

    // `[[package]]` stanzas: `name = "…"` followed by `version = "…"`.
    let mut grammars = Vec::new();
    let mut name: Option<&str> = None;
    for line in lock.lines() {
        if let Some(n) = line.strip_prefix("name = \"").and_then(|s| s.strip_suffix('"')) {
            name = Some(n);
        } else if let Some(v) = line.strip_prefix("version = \"").and_then(|s| s.strip_suffix('"')) {
            if let Some(n) = name.take().filter(|n| n.starts_with("tree-sitter")) {
                grammars.push((n, v));
            }
        }
    }
    grammars.sort();

    let mut out = String::from("/// `(crate, version)` for every tree-sitter crate in `Cargo.lock`.\n");
    out.push_str("pub const GRAMMARS: &[(&str, &str)] = &[\n");
    for (n, v) in grammars {
        let _ = writeln!(out, "    ({n:?}, {v:?}),");
    }
    out.push_str("];\n");
    let dest = std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join("grammars.rs");
    std::fs::write(dest, out).unwrap();
}
