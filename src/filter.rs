//! Filters that restrict which files in a `--project` tree are scanned and
//! tracked.
//!
//! Two flavors:
//! - [`PathFilter::Literal`] — a project-relative subpath. Matches by path
//!   *component* prefix (`src/parser` matches `src/parser/rust.rs` but NOT
//!   `src/parser_extra.rs`).
//! - [`PathFilter::Regex`] — a compiled regex matched against the
//!   slash-normalized project-relative path. Unanchored by default; callers
//!   anchor with `^...$` as needed.
//!
//! Construction is split from validation: parsing a filter from CLI input
//! is infallible for literals (only normalization happens) and fails only for
//! invalid regex syntax. Whether a *literal* path actually exists under the
//! project root is checked by [`PathFilter::validate`] so the CLI can produce
//! a clear "not accessible from root" error before any scan work begins.

use std::path::{Component, Path, PathBuf};

use color_eyre::eyre::{eyre, Result};

/// Decides whether a path is part of the project at all, independently of any
/// user-supplied [`PathFilter`].
///
/// This exists because the scanner and the file watcher disagreed. The scanner
/// walks with `ignore`'s gitignore handling, so `target/` never enters the
/// tree; the watcher filtered only on file extension, so a build tool writing
/// `target/**/out/*.rs` — rust-analyzer running `cargo check`, say — pushed
/// generated files into the tree one event at a time, and nothing ever removed
/// them.
///
/// Deliberately *narrower* in fidelity than the scanner's walk: it reads the
/// root `.gitignore` and `.git/info/exclude`, not `.gitignore` files nested in
/// subdirectories, which `ignore::WalkBuilder` discovers as it descends. That
/// asymmetry is the safe direction. Being too permissive means a file the
/// scanner would have skipped can still reach the tree — the old behaviour,
/// for an exotic layout. Being too strict would silently stop live updates for
/// files the scanner *did* include, which is a worse failure and much harder
/// to notice. `scope_agrees_with_the_scanner` pins the invariant against this
/// repo.
#[derive(Debug)]
pub struct ProjectScope {
    root: PathBuf,
    ignore: ignore::gitignore::Gitignore,
}

impl ProjectScope {
    /// Build a scope for `root`, reading its `.gitignore` and
    /// `.git/info/exclude`. Unreadable or absent files simply contribute no
    /// patterns — a missing `.gitignore` is normal, not an error.
    pub fn new(root: &Path) -> Self {
        let mut builder = ignore::gitignore::GitignoreBuilder::new(root);
        // `add` returns Some(err) on failure and None on success; a missing
        // file is reported the same way, so both are ignored on purpose.
        builder.add(root.join(".gitignore"));
        builder.add(root.join(".git/info/exclude"));

        let ignore = builder.build().unwrap_or_else(|_| {
            ignore::gitignore::Gitignore::empty()
        });

        Self {
            root: root.to_path_buf(),
            ignore,
        }
    }

    /// Is `path` inside the project and not excluded?
    ///
    /// Rejects anything outside `root`, anything under a dot-prefixed
    /// component (matching the scanner's `hidden(true)`), and anything the
    /// gitignore patterns exclude. Parent directories are consulted, so a file
    /// under an ignored directory is excluded even when only the directory is
    /// named in `.gitignore` — which is how `/target` excludes
    /// `target/debug/build/serde/out/private.rs`.
    pub fn contains(&self, path: &Path) -> bool {
        let Ok(rel) = path.strip_prefix(&self.root) else {
            return false;
        };

        let hidden = rel
            .components()
            .any(|c| matches!(c, Component::Normal(s) if s.to_string_lossy().starts_with('.')));
        if hidden {
            return false;
        }

        !self
            .ignore
            .matched_path_or_any_parents(rel, false)
            .is_ignore()
    }
}

/// Restricts which files are included in a project scan.
#[derive(Debug, Clone)]
pub enum PathFilter {
    /// Project-relative path components. `src/parser` and `/src/parser` both
    /// normalize to `["src", "parser"]`.
    Literal(Vec<String>),
    /// Validated regex matched (unanchored) against the slash-normalized
    /// project-relative path.
    Regex(regex::Regex),
}

impl PathFilter {
    /// Parse a literal subpath. Strips a leading `/` and collapses any empty
    /// components produced by consecutive separators. Never fails — semantic
    /// validation (does the path exist?) is [`PathFilter::validate`].
    pub fn literal(raw: &str) -> Self {
        let components: Vec<String> = raw
            .trim_start_matches('/')
            .split('/')
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect();
        Self::Literal(components)
    }

    /// Compile a regex pattern. Errors propagate the underlying compile error
    /// with the offending pattern included.
    pub fn regex(pattern: &str) -> Result<Self> {
        let re = regex::Regex::new(pattern)
            .map_err(|e| eyre!("invalid filter regex {pattern:?}: {e}"))?;
        Ok(Self::Regex(re))
    }

    /// For a literal filter, confirm that joining the components onto
    /// `project_root` yields an existing, accessible path. Regex filters
    /// always succeed here — they're checked against discovered files at scan
    /// time. An empty literal (e.g. the user passed `""` or `"/"`) is treated
    /// as no filter at all and validates trivially.
    pub fn validate(&self, project_root: &Path) -> Result<()> {
        let Self::Literal(components) = self else {
            return Ok(());
        };
        if components.is_empty() {
            return Ok(());
        }
        let mut joined = project_root.to_path_buf();
        for c in components {
            joined.push(c);
        }
        if !joined.exists() {
            return Err(eyre!(
                "filter path {:?} is not accessible from project root {}",
                components.join("/"),
                project_root.display()
            ));
        }
        Ok(())
    }

    /// True if the given **project-relative** path satisfies the filter.
    ///
    /// Literal: filter components must be a prefix of the path's *normal*
    /// components. Path separators are compared component-wise, so
    /// `src/parser` cannot accidentally match `src/parser_extra.rs`.
    ///
    /// Regex: the path is converted to forward slashes (so the pattern is
    /// portable across platforms) and tested for an unanchored match.
    pub fn matches(&self, rel_path: &Path) -> bool {
        match self {
            Self::Literal(filter_components) => {
                if filter_components.is_empty() {
                    return true;
                }
                let path_components: Vec<&str> = rel_path
                    .components()
                    .filter_map(|c| match c {
                        Component::Normal(os) => os.to_str(),
                        _ => None,
                    })
                    .collect();
                if path_components.len() < filter_components.len() {
                    return false;
                }
                filter_components
                    .iter()
                    .zip(path_components.iter())
                    .all(|(fc, pc)| fc == pc)
            }
            Self::Regex(re) => {
                let s = rel_path.to_string_lossy().replace('\\', "/");
                re.is_match(&s)
            }
        }
    }

    /// Human-readable form for headers in the TUI and coverage reports.
    pub fn display(&self) -> String {
        match self {
            Self::Literal(components) => components.join("/"),
            Self::Regex(re) => format!("re:{}", re.as_str()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn lit(s: &str) -> PathFilter {
        PathFilter::literal(s)
    }

    fn re(s: &str) -> PathFilter {
        PathFilter::regex(s).expect("regex must compile in test")
    }

    // ─── literal parsing / normalization ─────────────────────────────────────

    #[test]
    fn literal_normalizes_leading_slash() {
        let a = lit("/src/parser");
        let b = lit("src/parser");
        match (&a, &b) {
            (PathFilter::Literal(ac), PathFilter::Literal(bc)) => assert_eq!(ac, bc),
            _ => panic!("expected literal"),
        }
    }

    #[test]
    fn literal_collapses_consecutive_separators() {
        match lit("src//parser") {
            PathFilter::Literal(c) => assert_eq!(c, vec!["src", "parser"]),
            _ => panic!("expected literal"),
        }
    }

    #[test]
    fn literal_empty_input_yields_empty_components() {
        match lit("") {
            PathFilter::Literal(c) => assert!(c.is_empty()),
            _ => panic!("expected literal"),
        }
        match lit("/") {
            PathFilter::Literal(c) => assert!(c.is_empty()),
            _ => panic!("expected literal"),
        }
    }

    // ─── literal matching semantics ──────────────────────────────────────────

    #[test]
    fn literal_matches_path_components() {
        assert!(lit("src/parser").matches(Path::new("src/parser/rust.rs")));
        assert!(lit("src/parser").matches(Path::new("src/parser/typescript.rs")));
    }

    #[test]
    fn literal_does_not_match_prefix_only() {
        // The bug we explicitly designed around: string prefix would match,
        // component-aware matching must not.
        assert!(!lit("src/parser").matches(Path::new("src/parser_extra.rs")));
        assert!(!lit("src").matches(Path::new("src_alt/foo.rs")));
    }

    #[test]
    fn literal_matches_exact_single_file() {
        assert!(lit("src/main.rs").matches(Path::new("src/main.rs")));
    }

    #[test]
    fn literal_does_not_match_shallower_path() {
        // Filter is more specific than the candidate path — can't match.
        assert!(!lit("src/parser/rust.rs").matches(Path::new("src/parser")));
        assert!(!lit("src/parser").matches(Path::new("src")));
    }

    #[test]
    fn literal_top_level_filter_matches_all_deeper() {
        assert!(lit("src").matches(Path::new("src/parser/rust.rs")));
        assert!(lit("src").matches(Path::new("src/main.rs")));
        assert!(!lit("src").matches(Path::new("tests/e2e.rs")));
    }

    #[test]
    fn literal_empty_filter_matches_everything() {
        assert!(lit("").matches(Path::new("src/anything.rs")));
        assert!(lit("/").matches(Path::new("anywhere")));
    }

    #[test]
    fn literal_with_leading_slash_matches_same_as_without() {
        assert!(lit("/src/parser").matches(Path::new("src/parser/rust.rs")));
    }

    // ─── literal validation ──────────────────────────────────────────────────

    #[test]
    fn literal_validate_existing_path_ok() {
        let dir = TempDir::new().unwrap();
        let sub = dir.path().join("src").join("parser");
        std::fs::create_dir_all(&sub).unwrap();
        lit("src/parser")
            .validate(dir.path())
            .expect("existing path must validate");
    }

    #[test]
    fn literal_validate_existing_file_ok() {
        let dir = TempDir::new().unwrap();
        let f = dir.path().join("main.rs");
        std::fs::write(&f, "fn main() {}").unwrap();
        lit("main.rs").validate(dir.path()).unwrap();
    }

    #[test]
    fn literal_validate_missing_path_errors() {
        let dir = TempDir::new().unwrap();
        let err = lit("src/does_not_exist")
            .validate(dir.path())
            .expect_err("missing path must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("not accessible from project root"),
            "error message should explain inaccessibility, got: {msg}",
        );
        assert!(
            msg.contains("src/does_not_exist"),
            "error message should include the offending filter path, got: {msg}",
        );
    }

    #[test]
    fn literal_empty_validate_is_noop() {
        // Empty filter degenerates to "no filter" — always validates.
        let nowhere = PathBuf::from("/definitely/not/a/real/path/" );
        lit("").validate(&nowhere).unwrap();
    }

    // ─── regex parsing / matching ────────────────────────────────────────────

    #[test]
    fn regex_compiles_valid_pattern() {
        let _ = PathFilter::regex("^src/.*\\.rs$").unwrap();
    }

    #[test]
    fn regex_invalid_pattern_errors() {
        let err = PathFilter::regex("[unterminated").unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("invalid filter regex"),
            "error should call out invalid regex, got: {msg}",
        );
    }

    #[test]
    fn regex_matches_relative_path() {
        let f = re(r"^src/.*\.rs$");
        assert!(f.matches(Path::new("src/main.rs")));
        assert!(f.matches(Path::new("src/parser/rust.rs")));
        assert!(!f.matches(Path::new("tests/e2e.rs")));
        assert!(!f.matches(Path::new("src/main.py")));
    }

    #[test]
    fn regex_validate_is_always_ok() {
        // Regex validation doesn't touch the filesystem.
        let f = re(".*");
        f.validate(Path::new("/nonexistent")).unwrap();
    }

    #[test]
    fn regex_unanchored_substring_match() {
        // Unanchored: pattern can match anywhere in the path string.
        let f = re("parser");
        assert!(f.matches(Path::new("src/parser/rust.rs")));
        assert!(f.matches(Path::new("crates/parser/lib.rs")));
        assert!(!f.matches(Path::new("src/ingest/claude.rs")));
    }

    // ─── display ─────────────────────────────────────────────────────────────

    #[test]
    fn display_literal_drops_leading_slash() {
        assert_eq!(lit("/src/parser").display(), "src/parser");
        assert_eq!(lit("src/parser").display(), "src/parser");
    }

    #[test]
    fn display_regex_uses_re_prefix() {
        assert_eq!(re("^foo$").display(), "re:^foo$");
    }

    // -- ProjectScope ------------------------------------------------------

    /// A project root with a `.gitignore` and one file at each of the paths
    /// in `files`.
    fn scoped_project(gitignore: &str, files: &[&str]) -> TempDir {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join(".gitignore"), gitignore).unwrap();
        for f in files {
            let path = dir.path().join(f);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, "fn x() {}\n").unwrap();
        }
        dir
    }

    /// The case that started this: build output under an ignored directory.
    #[test]
    fn scope_excludes_files_under_an_ignored_directory() {
        let dir = scoped_project(
            "/target\n",
            &["src/main.rs", "target/release/build/serde/out/private.rs"],
        );
        let scope = ProjectScope::new(dir.path());

        assert!(scope.contains(&dir.path().join("src/main.rs")));
        assert!(
            !scope.contains(&dir.path().join("target/release/build/serde/out/private.rs")),
            "`/target` must exclude everything beneath it, not just the directory entry"
        );
    }

    #[test]
    fn scope_excludes_hidden_components() {
        let dir = scoped_project("", &["src/main.rs", ".cache/gen.rs"]);
        let scope = ProjectScope::new(dir.path());

        assert!(scope.contains(&dir.path().join("src/main.rs")));
        assert!(!scope.contains(&dir.path().join(".cache/gen.rs")));
    }

    #[test]
    fn scope_excludes_a_named_file_pattern() {
        let dir = scoped_project("*.gen.rs\n", &["src/a.rs", "src/b.gen.rs"]);
        let scope = ProjectScope::new(dir.path());

        assert!(scope.contains(&dir.path().join("src/a.rs")));
        assert!(!scope.contains(&dir.path().join("src/b.gen.rs")));
    }

    #[test]
    fn scope_rejects_paths_outside_the_root() {
        let dir = scoped_project("", &["src/main.rs"]);
        let scope = ProjectScope::new(dir.path());

        assert!(!scope.contains(Path::new("/etc/passwd")));
    }

    #[test]
    fn scope_without_a_gitignore_admits_everything_visible() {
        let dir = TempDir::new().unwrap();
        let scope = ProjectScope::new(dir.path());

        assert!(scope.contains(&dir.path().join("src/main.rs")));
    }

    /// The invariant the watcher depends on: every file the scanner puts in
    /// the tree must also be in scope. If these two ever drift, live updates
    /// silently stop for whatever the scope wrongly rejects.
    #[test]
    fn scope_agrees_with_the_scanner() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let scope = ProjectScope::new(root);
        let tree = crate::parser::ParserRegistry::new()
            .scan_project(root, None)
            .expect("scanning this repo must succeed");

        assert!(!tree.files.is_empty(), "the scan found nothing to compare");
        for file in &tree.files {
            let abs = root.join(&file.file_path);
            assert!(
                scope.contains(&abs),
                "scanner included {} but the scope rejects it",
                file.file_path.display()
            );
        }

        // And the converse for the path that motivated all this.
        assert!(!scope.contains(&root.join("target/debug/build/serde/out/private.rs")));
    }
}
