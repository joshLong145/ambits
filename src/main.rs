// Shared modules live in the library crate (src/lib.rs).
use ambits::coverage;
use ambits::ingest;

// Binary-only modules.
mod events;
mod hook;
mod serena;
mod skill;
mod tui;
mod ui;

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use clap::{Parser as ClapParser, Subcommand, ValueEnum};
use color_eyre::eyre::Result;
use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

use std::sync::Arc;

use ambits::app::App;
use ambits::filter::PathFilter;
use ambits::ingest::tool_config::ToolMappingConfig;
use ambits::ingest::{SessionIngester, ToolCallMapper};
use events::AppEvent;
use ambits::parser::ParserRegistry;

#[derive(ClapParser, Debug)]
#[command(name = "ambits", about = "Visualize LLM agent context coverage")]
struct Cli {
    /// Path to the project root to analyze. Defaults to the nearest enclosing
    /// directory containing `.git` or `.ambits`, else the current directory.
    #[arg(short, long)]
    project: Option<PathBuf>,

    /// Optional session ID to track (auto-detects latest if omitted).
    #[arg(short, long)]
    session: Option<String>,

    /// Path to Claude Code log directory (auto-derived if omitted).
    #[arg(long)]
    log_dir: Option<PathBuf>,

    /// Print symbol tree to stdout instead of launching TUI.
    #[arg(long)]
    dump: bool,

    /// With `--dump`, how many levels of children to descend into. 0 (the
    /// default) prints top-level symbols only; a symbol with hidden children
    /// is marked `+N`. Ignored unless `--dump` is set.
    #[arg(long, default_value_t = 0)]
    depth: usize,

    /// With `--dump`, descend to every level regardless of `--depth` — the
    /// full tree, as `--dump` always printed before a cheaper default existed.
    #[arg(long)]
    full: bool,

    /// Print coverage report to stdout instead of launching TUI.
    #[arg(long)]
    coverage: bool,

    /// Use Serena's LSP symbol cache instead of tree-sitter parsing.
    #[arg(long)]
    serena: bool,

    /// Filter coverage to a specific agent ID (supports prefix matching).
    #[arg(short, long)]
    agent: Option<String>,

    /// Output directory for event logs. If set, writes processed events to <dir>/<session>.log.
    #[arg(long)]
    log_output: Option<PathBuf>,

    /// Path to a custom tool call mapping config (TOML).
    /// Overrides project-local (.ambits/tools.toml) and user-global configs.
    #[arg(long)]
    tools_config: Option<PathBuf>,

    /// Editor command template for "open in editor" (Enter on a symbol row
    /// in the TUI). Supports `{file}`/`{line}` placeholders, e.g.
    /// "code -g {file}:{line}". Overrides tools.toml `[editor]` and
    /// $VISUAL/$EDITOR.
    #[arg(long)]
    editor: Option<String>,

    /// Output format for --coverage. Ignored when --coverage is not set.
    #[arg(long, value_enum, default_value = "table")]
    format: CoverageFormat,

    /// Restrict analysis to a project-relative subpath (e.g. `src/parser`).
    /// Leading slash is accepted but optional. Matches by path component:
    /// `src/parser` matches `src/parser/rust.rs` but NOT `src/parser_extra.rs`.
    /// Errors if the path does not exist under --project.
    #[arg(long, conflicts_with = "filter_regex")]
    filter: Option<String>,

    /// Restrict analysis to files whose project-relative path matches a regex
    /// (e.g. `^src/.*\.rs$`). Unanchored by default — anchor with `^...$` as
    /// needed. Mutually exclusive with --filter.
    #[arg(long, conflicts_with = "filter")]
    filter_regex: Option<String>,

    /// Disable the coverage journal (`.ambits/coverage/<session>.ndjson`), which
    /// records which symbols were read and what they looked like at the time.
    /// Overrides `enabled` in the `[cache]` section of tools.toml.
    #[arg(long)]
    no_journal: bool,

    /// How often, in milliseconds, to diff the ledger into the coverage
    /// journal. Overrides `flush_interval_ms` in the `[cache]` section.
    #[arg(long)]
    flush_interval_ms: Option<u64>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum CoverageFormat {
    /// Human-readable ASCII table (default).
    Table,
    /// Compact, schema-versioned JSON on a single line.
    Json,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Manage the Claude Code skill for ambit
    Skill {
        #[command(subcommand)]
        command: SkillCommands,
    },

    /// Manage the Claude Code SessionStart hook that injects restored context
    /// automatically after a compaction
    Hook {
        #[command(subcommand)]
        command: HookCommands,
    },

    /// Print the symbols read earlier in this session that are still
    /// unchanged, for re-injection after a compaction.
    ///
    /// Reads the coverage journal when one exists. Otherwise falls back to
    /// replaying the session logs, which cannot detect drift — that output is
    /// labelled UNVERIFIED.
    RestoreContext {
        /// Approximate token budget for the output.
        #[arg(long, default_value_t = ambits::digest::DEFAULT_MAX_TOKENS)]
        max_tokens: usize,

        /// Output format.
        #[arg(long, value_enum, default_value = "markdown")]
        format: DigestFormat,
    },

    /// Print a symbol's definition as JSON, looked up by content hash or id.
    ///
    /// Intended as the follow-up to `restore-context`: that names what is
    /// known and where it lives, this hands back the source. Accepts several
    /// selectors at once so a batch of lookups costs one process, and reports
    /// every match rather than guessing when a selector is ambiguous.
    Show {
        /// Content hash (`b3:<hex>`, or at least 8 hex characters) or symbol
        /// id (`<path>::<name-path>`). Repeatable.
        #[arg(required = true)]
        selector: Vec<String>,

        /// Omit the `definition` field and return only location metadata.
        #[arg(long)]
        no_body: bool,

        /// Truncate each definition to this many bytes, flagging it with
        /// `"truncated": true`. Unlimited by default — a cut definition is not
        /// valid source, so shortening one is the caller's decision to make.
        #[arg(long)]
        max_bytes: Option<usize>,
    },

    /// Search file contents with ripgrep's flag set, reporting the symbol each
    /// match lands in and how deeply it has already been read.
    ///
    /// The dialect Claude Code's own `Grep` tool speaks, so it is the one an
    /// agent most likely already knows. Four defaults deviate on purpose:
    /// results are always sorted by `(path, line, column)`; `--head-limit` and
    /// `--max-columns` are capped, because this output lands in a context
    /// window; and `--column` is on.
    ///
    /// Exits 0 when something matched, 1 when nothing did, 2 on error.
    Rg(RgArgs),

    /// The same search with GNU grep's flag set.
    ///
    /// Exists because grep and ripgrep give the same short flags opposite
    /// meanings — `-L`, `-z` and `-r` each mean one thing in one tool and
    /// something else in the other — so no single namespace can be faithful to
    /// both. Line numbers and columns are off by default here, as in grep;
    /// `-h` is `--no-filename`, not help.
    ///
    /// One deliberate infidelity: the search is recursive by default. Scoping
    /// to a project is what ambit is for, and `-r`/`-R` are accepted as no-ops
    /// so a habitual `grep -r` still works.
    ///
    /// Exits 0 when something matched, 1 when nothing did, 2 on error.
    // `-h` belongs to --no-filename here, so clap's auto-generated help flag
    // has to be removed and replaced by the long-only one on `GrepArgs`.
    // Without this, clap panics on a duplicate short — but only in debug
    // builds, where its assertions run, so a release smoke test will not
    // notice.
    #[command(disable_help_flag = true)]
    Grep(GrepArgs),

    /// List the call sites of a function, and which symbol each sits in.
    ///
    /// Call sites come from the grammar's own tags query, so a mention in a
    /// comment or inside a string literal is not reported — unlike a text
    /// search. Matching is by callee *name*: tree-sitter does not resolve
    /// which definition a call binds to, so a name shared by several
    /// definitions returns all of their call sites together.
    Callers {
        /// Function name, exactly as written at the call site. Repeatable.
        #[arg(required = true)]
        name: Vec<String>,

        /// Output format.
        #[arg(long, value_enum, default_value = "text")]
        format: FindFormat,
    },

    /// Inspect or remove the coverage journals under .ambits/coverage.
    Cache {
        #[command(subcommand)]
        command: CacheCommands,
    },
}

/// `ambits rg` — ripgrep's flag set, plus the symbol column.
#[derive(clap::Args, Debug)]
struct RgArgs {
        /// PATTERN, then optional PATHs. Every argument is a PATH when `-e` is
        /// given, exactly as in ripgrep.
        #[arg(value_name = "PATTERN|PATH")]
        args: Vec<String>,

        /// Additional pattern. Repeatable; combined as an alternation.
        #[arg(short = 'e', long = "regexp", value_name = "PATTERN")]
        regexp: Vec<String>,

        /// Treat patterns as literal strings.
        #[arg(short = 'F', long)]
        fixed_strings: bool,

        /// Case-insensitive matching.
        #[arg(short, long)]
        ignore_case: bool,

        /// Match only whole words.
        #[arg(short, long)]
        word_regexp: bool,

        /// Match only whole lines. Takes precedence over --word-regexp.
        #[arg(short = 'x', long)]
        line_regexp: bool,

        /// Allow patterns to match across line boundaries.
        #[arg(short = 'U', long)]
        multiline: bool,

        /// Report non-matching lines instead.
        #[arg(short = 'v', long)]
        invert_match: bool,

        /// Include files matching this glob; prefix with ! to exclude.
        #[arg(short, long, value_name = "GLOB")]
        glob: Vec<String>,

        /// Restrict to a file type, e.g. rust, py, ts, md.
        #[arg(short = 't', long = "type", value_name = "TYPE")]
        file_type: Vec<String>,

        /// List every type name --type accepts, with the globs it expands
        /// to, and exit. Takes no PATTERN.
        #[arg(long)]
        type_list: bool,

        /// Additional pattern(s), one per line, read from a file.
        /// Repeatable; combined with -e and PATTERN as one alternation.
        #[arg(short = 'f', long = "file", value_name = "PATTERNFILE")]
        pattern_file: Vec<PathBuf>,

        /// Print one line per match rather than grouping matches that share
        /// a line — vim/emacs quickfix format. Also splits --json's
        /// per-line events back into one per match, matching -o's existing
        /// behavior there.
        #[arg(long)]
        vimgrep: bool,

        /// Lines of context after each match.
        #[arg(short = 'A', long, default_value_t = 0, value_name = "N")]
        after_context: usize,

        /// Lines of context before each match.
        #[arg(short = 'B', long, default_value_t = 0, value_name = "N")]
        before_context: usize,

        /// Lines of context around each match.
        #[arg(short = 'C', long, value_name = "N")]
        context: Option<usize>,

        /// Show line numbers (default).
        #[arg(short = 'n', long, overrides_with = "no_line_number")]
        line_number: bool,

        /// Hide line numbers.
        #[arg(short = 'N', long)]
        no_line_number: bool,

        /// Hide the column of each match.
        #[arg(long)]
        no_column: bool,

        /// Print only the matched part of each line.
        #[arg(short, long)]
        only_matching: bool,

        /// Print only the paths of files with a match.
        #[arg(short = 'l', long, conflicts_with_all = ["count", "count_matches", "quiet"])]
        files_with_matches: bool,

        /// Print only the paths of files with *no* match — the complement of
        /// `-l`, not of `-v`.
        #[arg(long, conflicts_with_all = ["files_with_matches", "count", "count_matches", "quiet"])]
        files_without_match: bool,

        /// Print only a count of matching lines per file.
        #[arg(short, long, conflicts_with_all = ["count_matches", "quiet"])]
        count: bool,

        /// Print only a count of individual matches per file.
        #[arg(long, conflicts_with = "quiet")]
        count_matches: bool,

        /// Print nothing; the exit code is the answer.
        #[arg(short, long)]
        quiet: bool,

        /// Stop after this many matches per file.
        #[arg(short = 'm', long, value_name = "NUM")]
        max_count: Option<usize>,

        /// Truncate lines longer than this, marking the cut. 0 for unlimited.
        #[arg(short = 'M', long, default_value_t = ambits::search::DEFAULT_MAX_COLUMNS, value_name = "NUM")]
        max_columns: usize,

        /// Cap total matches across all files. 0 for unlimited. Not a ripgrep
        /// flag: unbounded output is a hazard in a context window.
        #[arg(long, default_value_t = ambits::search::DEFAULT_HEAD_LIMIT, value_name = "NUM")]
        head_limit: usize,

        /// Search hidden files and directories.
        #[arg(long)]
        hidden: bool,

        /// Ignore .gitignore and friends.
        #[arg(long)]
        no_ignore: bool,

        /// Group matches under a file heading (default on a terminal).
        #[arg(long, overrides_with = "no_heading")]
        heading: bool,

        /// Print one flat `file:line:col:` line per match.
        #[arg(long)]
        no_heading: bool,

        /// Omit the symbol column, for byte-identical ripgrep output.
        #[arg(long)]
        no_symbol: bool,

        /// When to colorize output.
        #[arg(long, value_enum, default_value = "auto", value_name = "WHEN")]
        color: ColorWhen,

        /// Emit ripgrep's JSON Lines events, with an added `symbol` field.
        #[arg(long)]
        json: bool,
}

/// `ambits grep` — GNU grep's flag set over the same engine.
///
/// Only the flags whose *meaning* differs from `ambits rg` need explaining
/// here; everything shared (`-i`, `-v`, `-w`, `-x`, `-F`, `-e`, `-f`, context,
/// `-m`, `-q`, `-o`, `-c`, `-l`) means what it means in both tools.
#[derive(clap::Args, Debug)]
struct GrepArgs {
    /// PATTERN, then optional PATHs. Every argument is a PATH when -e or -f
    /// is given.
    #[arg(value_name = "PATTERN|PATH")]
    args: Vec<String>,

    /// Additional pattern. Repeatable; combined as an alternation.
    #[arg(short = 'e', long = "regexp", value_name = "PATTERN")]
    regexp: Vec<String>,

    /// Additional pattern(s), one per line, read from a file.
    #[arg(short = 'f', long = "file", value_name = "PATTERNFILE")]
    pattern_file: Vec<PathBuf>,

    /// Treat patterns as literal strings.
    #[arg(short = 'F', long)]
    fixed_strings: bool,

    /// Accepted for compatibility. One regex engine, close enough to ERE for
    /// anything portable; no behaviour change.
    #[arg(short = 'E', long)]
    extended_regexp: bool,

    /// Accepted for compatibility. See --extended-regexp.
    #[arg(short = 'G', long)]
    basic_regexp: bool,

    /// Rejected, deliberately. The regex crate has no lookaround or
    /// backreferences, so a PCRE pattern would silently match something else
    /// rather than fail.
    #[arg(short = 'P', long)]
    perl_regexp: bool,

    /// Case-insensitive matching.
    #[arg(short, long)]
    ignore_case: bool,

    /// Match only whole words.
    #[arg(short, long)]
    word_regexp: bool,

    /// Match only whole lines.
    #[arg(short = 'x', long)]
    line_regexp: bool,

    /// Report non-matching lines instead.
    #[arg(short = 'v', long)]
    invert_match: bool,

    /// Accepted as a no-op: the search is always recursive.
    #[arg(short = 'r', long)]
    recursive: bool,

    /// Accepted as a no-op. See --recursive.
    #[arg(short = 'R', long)]
    dereference_recursive: bool,

    /// Search only files matching this glob. Repeatable.
    #[arg(long, value_name = "GLOB")]
    include: Vec<String>,

    /// Skip files matching this glob. Repeatable.
    #[arg(long, value_name = "GLOB")]
    exclude: Vec<String>,

    /// Skip directories matching this glob, and everything under them.
    #[arg(long, value_name = "DIR")]
    exclude_dir: Vec<String>,

    /// Prefix each match with its line number. Off by default, as in grep.
    #[arg(short = 'n', long)]
    line_number: bool,

    /// Also print the column of each match. Not a grep flag; ambit's own
    /// output carries one when asked.
    #[arg(long)]
    column: bool,

    /// Print the path on each line (the default).
    #[arg(short = 'H', long, overrides_with = "no_filename")]
    with_filename: bool,

    /// Omit the path from each line.
    #[arg(short = 'h', long)]
    no_filename: bool,

    /// Terminate each path with NUL instead of the usual separator, for
    /// `xargs -0`.
    #[arg(short = 'Z', long = "null")]
    null: bool,

    /// Rejected: NUL-separated *input* lines would change what a line is for
    /// matching, attribution and context alike.
    #[arg(short = 'z', long)]
    null_data: bool,

    /// Print only the matched part of each line.
    #[arg(short, long)]
    only_matching: bool,

    /// Print only the paths of files with a match.
    #[arg(short = 'l', long, conflicts_with_all = ["count", "quiet", "files_without_match"])]
    files_with_matches: bool,

    /// Print only the paths of files with no match.
    #[arg(short = 'L', long, conflicts_with_all = ["count", "quiet"])]
    files_without_match: bool,

    /// Print only a count of matching lines per file.
    #[arg(short, long, conflicts_with = "quiet")]
    count: bool,

    /// Print nothing; the exit code is the answer.
    #[arg(short, long)]
    quiet: bool,

    /// Lines of context after each match.
    #[arg(short = 'A', long, default_value_t = 0, value_name = "N")]
    after_context: usize,

    /// Lines of context before each match.
    #[arg(short = 'B', long, default_value_t = 0, value_name = "N")]
    before_context: usize,

    /// Lines of context around each match.
    #[arg(short = 'C', long, value_name = "N")]
    context: Option<usize>,

    /// Stop after this many matches per file.
    #[arg(short = 'm', long, value_name = "NUM")]
    max_count: Option<usize>,

    /// Truncate lines longer than this, marking the cut. 0 for unlimited.
    #[arg(long, default_value_t = ambits::search::DEFAULT_MAX_COLUMNS, value_name = "NUM")]
    max_columns: usize,

    /// Cap total matches across all files. 0 for unlimited.
    #[arg(long, default_value_t = ambits::search::DEFAULT_HEAD_LIMIT, value_name = "NUM")]
    head_limit: usize,

    /// Search hidden files and directories.
    #[arg(long)]
    hidden: bool,

    /// Ignore .gitignore and friends.
    #[arg(long)]
    no_ignore: bool,

    /// Omit the symbol column.
    #[arg(long)]
    no_symbol: bool,

    /// When to colorize output.
    #[arg(long, value_enum, default_value = "auto", value_name = "WHEN")]
    color: ColorWhen,

    /// Print help. Long-only: `-h` is --no-filename here, as in grep.
    #[arg(long, action = clap::ArgAction::Help)]
    help: Option<bool>,
}

#[derive(Subcommand, Debug)]
enum CacheCommands {
    /// List the coverage journals on disk with their size and age.
    Status,

    /// Delete coverage journals.
    ///
    /// Deliberately requires naming what to remove. Journals are the only
    /// record of what a past session read *and what it looked like at the
    /// time*; deleting one silently downgrades any later restore of that
    /// session to the UNVERIFIED session-log fallback.
    Clear {
        /// Delete only this session's journal.
        #[arg(long, conflicts_with = "all")]
        session: Option<String>,

        /// Delete every journal for this project.
        #[arg(long)]
        all: bool,
    },
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum ColorWhen {
    /// Colorize when stdout is a terminal.
    Auto,
    Always,
    Never,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum FindFormat {
    /// Aligned listing, one symbol per line (default).
    Text,
    /// Schema-versioned JSON. Match objects are shaped exactly like
    /// `show --no-body`, so `find` output feeds straight into `show`.
    Json,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum DigestFormat {
    /// Markdown, intended to be pasted or piped into a session (default).
    Markdown,
    /// Compact, schema-versioned JSON on a single line.
    Json,
    /// Claude Code `SessionStart` hook envelope. Prints nothing when there is
    /// nothing to restore. See `ambits hook install`.
    Hook,
}

#[derive(Subcommand, Debug)]
enum HookCommands {
    /// Register the SessionStart hook in .claude/settings.json.
    ///
    /// Merges into any existing settings; a file that cannot be parsed is left
    /// untouched and the snippet is printed instead.
    Install {
        /// Install to ~/.claude/settings.json (applies to all projects).
        #[arg(long, short)]
        global: bool,

        /// Project directory to install for (defaults to the current directory).
        #[arg(long, short)]
        project: Option<PathBuf>,
    },
}

#[derive(Subcommand, Debug)]
enum SkillCommands {
    /// Install the ambit skill for Claude Code
    Install {
        /// Install globally to ~/.claude/skills/ambit/ (available in all projects)
        #[arg(long, short)]
        global: bool,

        /// Install to a specific project directory
        #[arg(long, short)]
        project: Option<PathBuf>,
    },
}

/// What a dialect hands the shared search engine.
///
/// Both `ambits grep` and `ambits rg` reduce to this; nothing below here knows
/// which one was typed.
struct SearchRequest {
    options: ambits::search::Options,
    roots: Vec<PathBuf>,
    overrides: Option<ignore::overrides::Override>,
    types: Option<ignore::types::Types>,
    hidden: bool,
    no_ignore: bool,
}

/// Split positionals into patterns and paths, then fold in `-e` and `-f`.
///
/// The first positional is the pattern unless `-e`/`-f` supplied one, in which
/// case every positional is a path — ripgrep's rule, and grep's.
fn assemble_patterns(
    positional: &[String],
    regexp: &[String],
    pattern_files: &[PathBuf],
) -> Result<(Vec<String>, Vec<String>)> {
    use color_eyre::eyre::eyre;

    let supplied = !regexp.is_empty() || !pattern_files.is_empty();
    let (mut patterns, paths): (Vec<String>, Vec<String>) = if supplied {
        (regexp.to_vec(), positional.to_vec())
    } else {
        match positional.split_first() {
            Some((pattern, rest)) => (vec![pattern.clone()], rest.to_vec()),
            None => (Vec::new(), Vec::new()),
        }
    };

    // One pattern per non-empty line, joined into the same alternation as
    // -e/PATTERN. A missing or unreadable file is an expected, actionable
    // failure, so it propagates to the "2 on error" exit code rather than
    // panicking or being silently skipped.
    for path in pattern_files {
        let text = std::fs::read_to_string(path)
            .map_err(|e| eyre!("{}: {e}", path.display()))?;
        patterns.extend(text.lines().filter(|l| !l.is_empty()).map(str::to_string));
    }

    if patterns.is_empty() {
        return Err(eyre!("no pattern given"));
    }
    Ok((patterns, paths))
}

/// Resolve PATH arguments against the working directory, as grep's are.
///
/// Walking only what was asked for beats walking the project and discarding
/// most of it, so these become walk roots rather than another filter.
fn resolve_roots(paths: &[String], project_path: &Path) -> Result<Vec<PathBuf>> {
    use color_eyre::eyre::eyre;

    let mut roots = Vec::with_capacity(paths.len());
    for raw in paths {
        let resolved = std::fs::canonicalize(raw).map_err(|e| eyre!("{raw}: {e}"))?;
        if !resolved.starts_with(project_path) {
            return Err(eyre!(
                "{raw} is outside the project root {}",
                project_path.display()
            ));
        }
        roots.push(resolved);
    }
    Ok(roots)
}

fn build_overrides(
    globs: &[String],
    project_path: &Path,
) -> Result<Option<ignore::overrides::Override>> {
    use color_eyre::eyre::eyre;

    if globs.is_empty() {
        return Ok(None);
    }
    let mut builder = ignore::overrides::OverrideBuilder::new(project_path);
    for glob in globs {
        builder
            .add(glob)
            .map_err(|e| eyre!("invalid glob {glob:?}: {e}"))?;
    }
    Ok(Some(builder.build()?))
}

fn build_types(names: &[String]) -> Result<Option<ignore::types::Types>> {
    use color_eyre::eyre::eyre;

    if names.is_empty() {
        return Ok(None);
    }
    let mut builder = ignore::types::TypesBuilder::new();
    builder.add_defaults();
    for name in names {
        builder.select(name);
    }
    Ok(Some(
        builder.build().map_err(|e| eyre!("invalid --type: {e}"))?,
    ))
}

fn color_choice(when: ColorWhen) -> ambits::search::ColorChoice {
    use ambits::search::ColorChoice;
    match when {
        ColorWhen::Auto => ColorChoice::Auto,
        ColorWhen::Always => ColorChoice::Always,
        ColorWhen::Never => ColorChoice::Never,
    }
}

/// `ambits rg` — ripgrep's dialect.
fn request_from_rg(args: &RgArgs, project_path: &Path) -> Result<SearchRequest> {
    use ambits::search::{Options, OutputMode};

    let (patterns, paths) = assemble_patterns(&args.args, &args.regexp, &args.pattern_file)?;

    let mode = if args.quiet {
        OutputMode::Quiet
    } else if args.files_with_matches {
        OutputMode::FilesWithMatches
    } else if args.files_without_match {
        OutputMode::FilesWithoutMatch
    } else if args.count {
        OutputMode::Count
    } else if args.count_matches {
        OutputMode::CountMatches
    } else {
        OutputMode::Content
    };

    Ok(SearchRequest {
        options: Options {
            patterns,
            fixed_strings: args.fixed_strings,
            ignore_case: args.ignore_case,
            word_regexp: args.word_regexp,
            line_regexp: args.line_regexp,
            multiline: args.multiline,
            invert_match: args.invert_match,
            mode,
            json: args.json,
            heading: match (args.heading, args.no_heading) {
                (true, false) => Some(true),
                (false, true) => Some(false),
                // Neither, or both via `overrides_with`: follow the terminal.
                _ => None,
            },
            line_number: !args.no_line_number,
            column: !args.no_column,
            only_matching: args.only_matching,
            vimgrep: args.vimgrep,
            before_context: args.context.unwrap_or(args.before_context),
            after_context: args.context.unwrap_or(args.after_context),
            max_columns: args.max_columns,
            max_count: args.max_count,
            head_limit: args.head_limit,
            no_symbol: args.no_symbol,
            show_filename: true,
            null_separator: false,
            color: color_choice(args.color),
        },
        roots: resolve_roots(&paths, project_path)?,
        overrides: build_overrides(&args.glob, project_path)?,
        types: build_types(&args.file_type)?,
        hidden: args.hidden,
        no_ignore: args.no_ignore,
    })
}

/// `ambits grep` — GNU grep's dialect.
///
/// The translation is mostly one-to-one; what differs is which letters carry
/// which meaning, plus three flags that cannot be honoured and say so rather
/// than pretending.
fn request_from_grep(args: &GrepArgs, project_path: &Path) -> Result<SearchRequest> {
    use ambits::search::{Options, OutputMode};
    use color_eyre::eyre::eyre;

    if args.perl_regexp {
        return Err(eyre!(
            "-P/--perl-regexp is not supported: this searches with the regex crate, \
             which has no lookaround or backreferences, so a PCRE pattern would \
             match something other than what it says"
        ));
    }
    if args.null_data {
        return Err(eyre!(
            "-z/--null-data is not supported: NUL-separated input would change what \
             a line is for matching, symbol attribution and context alike"
        ));
    }

    let (patterns, paths) = assemble_patterns(&args.args, &args.regexp, &args.pattern_file)?;

    // grep's include/exclude map onto the same glob engine `rg -g` uses; the
    // only translation is that grep spells exclusion with a separate flag
    // where rg spells it with a leading `!`.
    let mut globs: Vec<String> = args.include.clone();
    globs.extend(args.exclude.iter().map(|g| format!("!{g}")));
    // A directory exclusion has to name what is *under* it, since the globs
    // are matched against file paths.
    globs.extend(
        args.exclude_dir
            .iter()
            .flat_map(|d| [format!("!{d}/**"), format!("!**/{d}/**")]),
    );

    let mode = if args.quiet {
        OutputMode::Quiet
    } else if args.files_with_matches {
        OutputMode::FilesWithMatches
    } else if args.files_without_match {
        OutputMode::FilesWithoutMatch
    } else if args.count {
        OutputMode::Count
    } else {
        OutputMode::Content
    };

    Ok(SearchRequest {
        options: Options {
            patterns,
            fixed_strings: args.fixed_strings,
            ignore_case: args.ignore_case,
            word_regexp: args.word_regexp,
            line_regexp: args.line_regexp,
            // grep has no multiline mode; a pattern that wants one can still
            // say `(?m)` for itself.
            multiline: false,
            invert_match: args.invert_match,
            mode,
            json: false,
            // grep has no heading mode at all.
            heading: Some(false),
            // Both off by default here, unlike the rg dialect: `grep -n` is
            // opt-in, and grep has no column at all.
            line_number: args.line_number,
            column: args.column,
            only_matching: args.only_matching,
            vimgrep: false,
            before_context: args.context.unwrap_or(args.before_context),
            after_context: args.context.unwrap_or(args.after_context),
            max_columns: args.max_columns,
            max_count: args.max_count,
            head_limit: args.head_limit,
            no_symbol: args.no_symbol,
            show_filename: !args.no_filename,
            null_separator: args.null,
            color: color_choice(args.color),
        },
        roots: resolve_roots(&paths, project_path)?,
        overrides: build_overrides(&globs, project_path)?,
        types: None,
        hidden: args.hidden,
        no_ignore: args.no_ignore,
    })
}

/// Walk, search, and report — the half both dialects share.
fn execute(
    request: SearchRequest,
    registry: &ParserRegistry,
    project_path: &Path,
    filter: Option<&PathFilter>,
    coverage: Option<&ambits::restore::CoverageIndex>,
) -> Result<ambits::search::Outcome> {
    use ambits::parser::{walk_files, WalkOptions};

    let targets = walk_files(
        project_path,
        &WalkOptions {
            filter,
            overrides: request.overrides,
            types: request.types,
            hidden: request.hidden,
            no_ignore: request.no_ignore,
            roots: request.roots,
        },
    );

    ambits::search::run(registry, &targets, &request.options, coverage)
}

/// Build the project symbol tree with whichever backend was selected.
///
/// Named because two call sites need it and they must not drift: `find`
/// scans on its own, ahead of the project-wide scan every other command shares.
fn scan_tree(
    serena_backend: bool,
    registry: &ParserRegistry,
    project_path: &Path,
    filter: Option<&PathFilter>,
) -> Result<ambits::symbols::ProjectTree> {
    if serena_backend {
        serena::scan_project_serena(project_path, filter)
    } else {
        registry.scan_project(project_path, filter)
    }
}

/// Config and journal warnings. Always stderr: every command's stdout is data
/// someone may be parsing (`--coverage --format json | jq`), and a warning
/// there would corrupt it.
fn report_warnings<W: std::fmt::Display>(warnings: impl IntoIterator<Item = W>) {
    for w in warnings {
        ambits::try_eprintln!("[ambit warning] {w}");
    }
}

fn main() -> Result<()> {
    match run() {
        // The reader went away (`ambits … | head`): the consumer ended the
        // conversation, which is not a failure. See `ambits::output`.
        Err(e) if ambits::output::is_broken_pipe(&e) => Ok(()),
        result => result,
    }
}

fn run() -> Result<()> {
    color_eyre::install()?;
    let mut cli = Cli::parse();

    // `skill` is the only subcommand that doesn't need --project, so it is
    // handled here. `restore-context` needs a scanned tree to compare against,
    // so it falls through and is dispatched once the project is resolved.
    let command = cli.command.take();
    match &command {
        Some(Commands::Skill { command }) => {
            return match command {
                SkillCommands::Install { global, project } => {
                    skill::install(*global, project.clone())
                }
            };
        }
        Some(Commands::Hook { command }) => {
            return match command {
                HookCommands::Install { global, project } => {
                    hook::install(*global, project.clone())
                }
            };
        }
        _ => {}
    }

    // Without `--project`, the project containing the working directory — so
    // `ambits rg foo` works from anywhere inside it, as `rg foo` would.
    // Resolved before the config, which is looked up relative to it.
    let project = match cli.project {
        Some(p) => p,
        None => ambits::state_dir::find_project_root(&std::env::current_dir()?),
    };
    let project_path = project.canonicalize().unwrap_or(project);

    // Resolve tool call mapping config. Warnings go to stderr via `report_warnings`.
    let (tool_config, config_warnings) =
        ToolMappingConfig::resolve(cli.tools_config.as_deref(), &project_path);

    // Capture the `[cache]`/`[editor]` stanzas before `tool_config` is
    // coerced into the mapper trait object below and its concrete type is no
    // longer reachable.
    let cache_cfg = tool_config.cache.clone();
    let editor_template = ambits::editor::resolve_editor_template(
        cli.editor.as_deref(),
        tool_config.editor.command.as_deref(),
        std::env::var("VISUAL").ok().as_deref(),
        std::env::var("EDITOR").ok().as_deref(),
    );

    // Both the TUI and `find` write to the journal, and `find` dispatches long
    // before the TUI is built, so the decision is made once here. CLI flags win
    // over the `[cache]` stanza in tools.toml.
    let journal_enabled = !cli.no_journal && cache_cfg.enabled.unwrap_or(true);

    // Build the session ingester — coerce ToolMappingConfig to Arc<dyn ToolCallMapper>.
    let mapper: Arc<dyn ToolCallMapper> = tool_config;
    let ingester: Arc<dyn SessionIngester> =
        Arc::new(ingest::claude::ClaudeIngester::new(Arc::clone(&mapper)));

    // Build the optional path filter. clap already enforces mutual exclusion
    // between --filter and --filter-regex, so at most one branch fires.
    let filter: Option<PathFilter> = match (&cli.filter, &cli.filter_regex) {
        (Some(p), _) => Some(PathFilter::literal(p)),
        (_, Some(r)) => Some(PathFilter::regex(r)?),
        _ => None,
    };
    if let Some(ref f) = filter {
        f.validate(&project_path)?;
    }

    // Dispatched before the tree scan: inspecting or deleting journal files
    // needs the project path and nothing else, and scanning first would make
    // `cache status` pay seconds for an answer it does not use.
    if let Some(Commands::Cache { command }) = &command {
        return match command {
            CacheCommands::Status => ambits::cache::status(&project_path),
            CacheCommands::Clear { session, all } => {
                ambits::cache::clear(&project_path, session.as_deref(), *all)
            }
        };
    }

    let registry = ParserRegistry::new();

    // Resolve log directory and session. Hoisted above the scan because `find`
    // dispatches before it: neither depends on the symbol tree, and both read
    // the CLI values rather than consuming them so `--coverage` below still
    // resolves its own.
    let log_dir = cli
        .log_dir
        .clone()
        .or_else(|| ingester.log_dir_for_project(&project_path));

    let session_id = cli.session.clone().or_else(|| {
        log_dir
            .as_ref()
            .and_then(|d| ingester.find_latest_session(d))
    });

    ambits::logging::init(cli.log_output.as_deref(), session_id.as_deref());

    // Coverage context for `find` and `show`. Loaded once, from the journal
    // the TUI maintains, so both can report whether a symbol has already been
    // read. `None` when there is no session or no journal — which callers must
    // not confuse with "nothing has been read".
    let coverage_index = ambits::restore::CoverageIndex::load(&project_path, session_id.as_deref());

    // Both search dialects run before the project-wide scan, the way `cache`
    // does. A content search reads the files it walks and parses only the ones
    // that match, so paying for a full parse first would be paying for work it
    // discards.
    let search = match &command {
        Some(Commands::Rg(args)) => {
            // Takes no PATTERN, so it is handled before the pattern-required
            // path — reuses the same TypesBuilder `-t` itself builds, calling
            // `.definitions()` on it before `.build()`.
            if args.type_list {
                let mut builder = ignore::types::TypesBuilder::new();
                builder.add_defaults();
                let mut defs = builder.definitions();
                defs.sort_by(|a, b| a.name().cmp(b.name()));
                let mut out = io::stdout().lock();
                for def in defs {
                    writeln!(out, "{}: {}", def.name(), def.globs().join(", "))?;
                }
                return Ok(());
            }
            Some(("rg", request_from_rg(args, &project_path)))
        }
        Some(Commands::Grep(args)) => Some(("grep", request_from_grep(args, &project_path))),
        _ => None,
    };

    if let Some((dialect, request)) = search {
        report_warnings(&config_warnings);

        // grep's exit codes, which agents chain on: 0 matched, 1 did not,
        // 2 something went wrong. `color_eyre` would exit 1 for an error,
        // which is indistinguishable from an honest "no match".
        let outcome = request.and_then(|request| {
            execute(
                request,
                &registry,
                &project_path,
                filter.as_ref(),
                coverage_index.as_ref(),
            )
        });

        match outcome {
            Ok(outcome) => {
                // Neither dialect journals its own reads — the TUI is the sole
                // journal writer (see journal.rs's module doc). A search run
                // with no TUI attached to the session earns no coverage credit;
                // that is an accepted trade for never having two processes able
                // to write the same session's journal.
                if outcome.matched {
                    return Ok(());
                }
                std::process::exit(1)
            }
            Err(e) => {
                ambits::try_eprintln!("ambits {dialect}: {e:#}");
                std::process::exit(2);
            }
        }
    }

    let project_tree = scan_tree(cli.serena, &registry, &project_path, filter.as_ref())?;

    if cli.dump {
        report_warnings(&config_warnings);
        let depth = if cli.full { None } else { Some(cli.depth) };
        coverage::dump_tree(&project_path, &project_tree, filter.as_ref(), depth)?;
        return Ok(());
    }

    if cli.coverage {
        report_warnings(&config_warnings);
        let formatter: Box<dyn coverage::CoverageFormatter> = match cli.format {
            CoverageFormat::Table => Box::new(coverage::TextFormatter::default()),
            CoverageFormat::Json => Box::new(coverage::JsonFormatter),
        };
        return coverage::run_report(
            &project_path,
            &project_tree,
            &cli.log_dir,
            &cli.session,
            &cli.agent,
            filter.as_ref(),
            &*ingester,
            &*formatter,
        );
    }

    if let Some(Commands::Show {
        selector,
        no_body,
        max_bytes,
    }) = &command
    {
        report_warnings(&config_warnings);
        return ambits::lookup::run(
            &project_path,
            &project_tree,
            selector,
            !no_body,
            *max_bytes,
            coverage_index.as_ref(),
        );
    }

    if let Some(Commands::Callers { name, format }) = &command {
        report_warnings(&config_warnings);
        return ambits::callers::run(
            &project_path,
            &project_tree,
            &registry,
            name,
            matches!(format, FindFormat::Json),
        );
    }

    if let Some(Commands::RestoreContext { max_tokens, format }) = command {
        report_warnings(&config_warnings);
        return run_restore_context(
            &project_path,
            &project_tree,
            &log_dir,
            &session_id,
            &*ingester,
            max_tokens,
            format,
        );
    }

    // Everything below happens before the terminal is touched. None of it
    // needs a terminal, and all of it can fail: a session log with a line we
    // cannot parse used to panic *after* raw mode was on, which left the
    // terminal swallowing input with the panic message painted on an alternate
    // screen nobody would ever see again.

    let mut app = App::new(project_tree, project_path.clone());
    app.set_editor_template(editor_template);
    app.filter = filter.map(Arc::new);
    app.set_session_id(session_id.clone());
    app.session_slug = log_dir.as_ref()
        .zip(session_id.as_ref())
        .and_then(|(ld, sid)| ingester.session_slug(ld, sid));

    // Pre-populate the ledger from existing session logs.
    if let (Some(ref log_dir), Some(ref session_id)) = (&log_dir, &session_id) {
        let log_files = ingester.session_log_files(log_dir, session_id);
        for log_file in &log_files {
            for event in ingester.parse_log_file_with_root(log_file, &project_path) {
                match event {
                    ingest::SessionEvent::ToolCall(tc) => app.process_agent_event(tc),
                    ingest::SessionEvent::Compacted { summary, timestamp, agent_id, metadata } => {
                        app.process_compaction(summary, timestamp, agent_id, metadata);
                    }
                    ingest::SessionEvent::SessionCleared => app.reset_session(),
                }
            }
        }
    }

    let serena_mode = cli.serena;

    // Fold in the journal before opening it for writing. The replay above
    // rebuilt *which* symbols were read but stamped each with the hash it has
    // now, so nothing looks drifted; only the journal knows what they looked
    // like at the time. Skipped entirely when journaling is off, since then
    // there is no journal to trust.
    if journal_enabled {
        if let Some(stats) = app.rehydrate_from_journal() {
            if stats.drifted > 0 || stats.inserted > 0 || stats.moved > 0 {
                ambits::try_eprintln!(
                    "[ambit] rehydrated from journal: {} corrected, {} recovered, {} moved, {} stale",
                    stats.corrected, stats.inserted, stats.moved, stats.drifted
                );
            }
        }
    }

    // Open the coverage journal *after* the startup replay above. `Journal::open`
    // seeds its dedup map from what is already on disk, so the immediate sync
    // below appends only genuinely new reads — which is what keeps relaunching
    // ambit idempotent instead of re-appending the whole session every time.
    // CLI flags win over the `[cache]` stanza in tools.toml.
    if journal_enabled {
        let interval = std::time::Duration::from_millis(
            cli.flush_interval_ms
                .or(cache_cfg.flush_interval_ms)
                .unwrap_or(ambits::journal::DEFAULT_FLUSH_INTERVAL_MS),
        );
        let backend = if serena_mode { "serena" } else { "tree-sitter" };
        report_warnings(app.enable_journal(backend, interval));
        app.sync_journal();
    }

    // Terminal setup, as late as possible. The guard restores it on every
    // path out of here — `?`, panic, or a clean return — and the panic hook
    // gets there first so the report lands on the normal screen.
    install_panic_restore();
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let _guard = TerminalGuard;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = run_tui(&mut terminal, &mut app, &project_path, &log_dir, session_id, &registry, serena_mode, &ingester);

    // Capture the tail of the session. Records are written unbuffered, so this
    // is not a buffer flush — it is a final diff of anything read since the
    // last interval tick.
    app.sync_journal();

    // Flush the log file before exiting.
    log::logger().flush();

    // `_guard` restores the rest as it drops.
    terminal.show_cursor()?;

    result
}

/// Put the terminal back the way it was found.
///
/// Deliberately ignores its errors: it runs on paths where something has
/// already gone wrong, and a failure to restore must not mask what that was.
fn restore_terminal() {
    let _ = disable_raw_mode();
    let _ = execute!(io::stdout(), LeaveAlternateScreen, DisableMouseCapture);
}

/// Restores the terminal however this scope is left — a clean return, a `?`, or
/// an unwind.
///
/// The teardown used to be three statements at the end of `main`, which is the
/// one place an early exit never reaches.
struct TerminalGuard;

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        restore_terminal();
    }
}

/// Restore the terminal *before* the panic report is printed.
///
/// The `Drop` guard alone is not enough: a hook runs before unwinding begins,
/// so `color_eyre` would paint its report onto the alternate screen and the
/// guard would then tear that screen down, taking the message with it. Chaining
/// in front of the existing hook puts the report on the normal screen, where it
/// can be read.
///
/// Not hypothetical for a long-running TUI: the log tailer parses new lines as
/// they arrive, so a session log can start failing to parse at any point during
/// a session, not only at startup.
fn install_panic_restore() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        restore_terminal();
        previous(info);
    }));
}

/// Print the still-valid prior reads for a session.
///
/// Prefers the coverage journal, which records what each symbol looked like
/// when it was read and can therefore prove a read is still accurate. Falls
/// back to replaying session logs — which cannot prove anything of the sort,
/// because Claude Code never recorded those hashes — rather than returning
/// nothing for sessions that predate journaling or never ran the TUI. The
/// formatter labels that output UNVERIFIED.
#[allow(clippy::too_many_arguments)]
fn run_restore_context(
    project_path: &std::path::Path,
    project_tree: &ambits::symbols::ProjectTree,
    log_dir: &Option<PathBuf>,
    session_id: &Option<String>,
    ingester: &dyn SessionIngester,
    max_tokens: usize,
    format: DigestFormat,
) -> Result<()> {
    use ambits::digest::{DigestFormatter, HookFormatter, JsonFormatter, MarkdownFormatter};
    use ambits::restore::{self, RestoreReport, RestoreSource};

    let journaled = session_id
        .as_ref()
        .and_then(|sid| restore::load_from_journal(project_path, sid));

    let (reads, source, warnings) = match journaled {
        Some((reads, warnings)) => (reads, RestoreSource::Journal, warnings),
        None => match (log_dir.as_ref(), session_id.as_ref()) {
            (Some(dir), Some(sid)) => {
                let ledger =
                    restore::replay_session_logs(project_path, project_tree, dir, sid, ingester);
                (
                    restore::reads_from_ledger(&ledger),
                    RestoreSource::SessionLogs,
                    Vec::new(),
                )
            }
            _ => (Default::default(), RestoreSource::Journal, Vec::new()),
        },
    };

    let report = RestoreReport {
        outcome: restore::classify(&reads, project_tree),
        source,
        session_id: session_id.clone(),
        warnings,
    };

    let formatter: Box<dyn DigestFormatter> = match format {
        DigestFormat::Markdown => Box::new(MarkdownFormatter),
        DigestFormat::Json => Box::new(JsonFormatter),
        DigestFormat::Hook => Box::new(HookFormatter),
    };
    let rendered = formatter.format(&report, max_tokens);
    // The hook format returns an empty string when there is nothing to
    // restore; printing a bare newline there would be stdout Claude Code has
    // to parse and reject.
    if !rendered.is_empty() {
        writeln!(io::stdout().lock(), "{rendered}")?;
    }
    Ok(())
}

fn run_tui(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
    project_path: &Path,
    log_dir: &Option<PathBuf>,
    session_id: Option<String>,
    registry: &ParserRegistry,
    serena_mode: bool,
    ingester: &Arc<dyn SessionIngester>,
) -> Result<()> {
    let (tx, rx) = flume::bounded::<AppEvent>(512);

    events::spawn_key_reader(tx.clone());
    events::spawn_tick_timer(tx.clone(), Duration::from_millis(250));

    let mut session = tui::TuiSession::new(
        project_path,
        log_dir,
        session_id,
        registry.supported_extensions(),
        Arc::clone(ingester),
        serena_mode,
        &tx,
    )?;

    loop {
        terminal.draw(|f| ui::render(f, app))?;

        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(AppEvent::Key(key)) => app.handle_key(key),
            Ok(AppEvent::Mouse(mouse)) => app.handle_mouse(mouse),
            Ok(AppEvent::FileChanged(path)) => {
                tui::TuiSession::handle_file_changed(path, project_path, registry, app);
            }
            Ok(AppEvent::FileRemoved(path)) => {
                tui::TuiSession::handle_file_removed(path, project_path, app);
            }
            Ok(AppEvent::Tick) => {
                session.handle_tick(log_dir, app, serena_mode, project_path);
            }
            Err(flume::RecvTimeoutError::Timeout) => {}
            Err(flume::RecvTimeoutError::Disconnected) => break,
        }

        if let Some((path, line)) = app.pending_editor_request.take() {
            let template = app.editor_template.clone();
            suspend_for_editor(
                terminal,
                &rx,
                app,
                &mut session,
                project_path,
                log_dir,
                registry,
                serena_mode,
                &path,
                line,
                template.as_deref(),
            )?;
        }

        if app.should_quit {
            break;
        }
    }

    Ok(())
}

/// Suspend the TUI, run the editor synchronously to let the user look at (and
/// possibly edit) the symbol's file, then resume.
///
/// `template = None` means nothing resolved to launch — the terminal is left
/// untouched. A spawn failure or nonzero exit is captured into
/// `app.last_editor_error` rather than propagated: many editors exit nonzero
/// for reasons that have nothing to do with whether the visit worked, so
/// treating it as fatal to the TUI would be wrong.
#[allow(clippy::too_many_arguments)]
fn suspend_for_editor(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    rx: &flume::Receiver<AppEvent>,
    app: &mut App,
    session: &mut tui::TuiSession,
    project_path: &Path,
    log_dir: &Option<PathBuf>,
    registry: &ParserRegistry,
    serena_mode: bool,
    path: &Path,
    line: u32,
    template: Option<&str>,
) -> Result<()> {
    let Some(template) = template else {
        app.last_editor_error =
            Some("no editor configured (set $VISUAL, $EDITOR, or --editor)".to_string());
        return Ok(());
    };

    disable_raw_mode()?;
    execute!(io::stdout(), LeaveAlternateScreen, DisableMouseCapture)?;

    let argv = ambits::editor::build_editor_argv(template, path, line);
    app.last_editor_error = match argv.split_first() {
        Some((cmd, args)) => match std::process::Command::new(cmd).args(args).status() {
            Ok(status) if status.success() => None,
            Ok(status) => Some(format!("'{cmd}' exited with {status}")),
            Err(e) => Some(format!("failed to launch '{cmd}': {e}")),
        },
        None => Some("no editor command resolved".to_string()),
    };

    enable_raw_mode()?;
    execute!(io::stdout(), EnterAlternateScreen, EnableMouseCapture)?;
    terminal.clear()?;

    // Drain input queued while the editor had the terminal — those keys and
    // clicks were meant for the editor, not the TUI. Other event kinds
    // (file-watch, tick) are still real state changes and get applied
    // normally so nothing goes stale across the suspension.
    while let Ok(ev) = rx.try_recv() {
        match ev {
            AppEvent::Key(_) | AppEvent::Mouse(_) => {}
            AppEvent::FileChanged(p) => {
                tui::TuiSession::handle_file_changed(p, project_path, registry, app);
            }
            AppEvent::FileRemoved(p) => {
                tui::TuiSession::handle_file_removed(p, project_path, app);
            }
            AppEvent::Tick => {
                session.handle_tick(log_dir, app, serena_mode, project_path);
            }
        }
    }

    Ok(())
}
