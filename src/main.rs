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

use std::fs;
use std::io;
use std::io::Write;
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
    /// Path to the project root to analyze.
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
    /// Overrides project-local (.ambit/tools.toml) and user-global configs.
    #[arg(long)]
    tools_config: Option<PathBuf>,

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

    /// Disable the coverage journal (`.ambit/coverage/<session>.ndjson`), which
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

fn main() -> Result<()> {
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

    // Resolve tool call mapping config. Warnings are displayed to stdout before TUI launch.
    let (tool_config, config_warnings) =
        ToolMappingConfig::resolve(cli.tools_config.as_deref());

    // Capture the `[cache]` stanza before `tool_config` is coerced into the
    // mapper trait object below and its concrete type is no longer reachable.
    let cache_cfg = tool_config.cache.clone();

    // Build the session ingester — coerce ToolMappingConfig to Arc<dyn ToolCallMapper>.
    let mapper: Arc<dyn ToolCallMapper> = tool_config;
    let ingester: Arc<dyn SessionIngester> =
        Arc::new(ingest::claude::ClaudeIngester::new(Arc::clone(&mapper)));

    // Original behavior — require --project for all other modes.
    let project = cli.project.ok_or_else(|| {
        color_eyre::eyre::eyre!("--project is required (use `ambits --project <path>`)")
    })?;
    let project_path = project.canonicalize().unwrap_or(project);

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

    let registry = ParserRegistry::new();
    let project_tree = if cli.serena {
        serena::scan_project_serena(&project_path, filter.as_ref())?
    } else {
        registry.scan_project(&project_path, filter.as_ref())?
    };

    if cli.dump {
        for w in &config_warnings {
            println!("[ambit warning] {w}");
        }
        coverage::dump_tree(&project_path, &project_tree, filter.as_ref());
        return Ok(());
    }

    if cli.coverage {
        for w in &config_warnings {
            println!("[ambit warning] {w}");
        }
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

    // Resolve log directory and session.
    let log_dir = cli
        .log_dir
        .or_else(|| ingester.log_dir_for_project(&project_path));

    let session_id = cli.session.or_else(|| {
        log_dir
            .as_ref()
            .and_then(|d| ingester.find_latest_session(d))
    });

    if let Some(Commands::RestoreContext { max_tokens, format }) = command {
        for w in &config_warnings {
            eprintln!("[ambit warning] {w}");
        }
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

    // Launch TUI.
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Set up event log writer if --log-output is specified.
    let event_log = if let Some(ref log_output_dir) = cli.log_output {
        fs::create_dir_all(log_output_dir)?;
        let log_name = session_id
            .as_deref()
            .unwrap_or("unknown-session");
        let log_path = log_output_dir.join(format!("{log_name}.log"));
        let file = fs::File::create(&log_path)?;
        Some(io::BufWriter::new(file))
    } else {
        None
    };

    let mut app = App::new(project_tree, project_path.clone(), event_log);
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

    // Open the coverage journal *after* the startup replay above. `Journal::open`
    // seeds its dedup map from what is already on disk, so the immediate sync
    // below appends only genuinely new reads — which is what keeps relaunching
    // ambit idempotent instead of re-appending the whole session every time.
    // CLI flags win over the `[cache]` stanza in tools.toml.
    let journal_enabled = !cli.no_journal && cache_cfg.enabled.unwrap_or(true);
    if journal_enabled {
        let interval = std::time::Duration::from_millis(
            cli.flush_interval_ms
                .or(cache_cfg.flush_interval_ms)
                .unwrap_or(ambits::journal::DEFAULT_FLUSH_INTERVAL_MS),
        );
        let backend = if serena_mode { "serena" } else { "tree-sitter" };
        for warning in app.enable_journal(backend, interval) {
            eprintln!("[ambit warning] {warning}");
        }
        app.sync_journal();
    }

    let result = run_tui(&mut terminal, &mut app, &project_path, &log_dir, session_id, &registry, serena_mode, &ingester);

    // Capture the tail of the session. Records are written unbuffered, so this
    // is not a buffer flush — it is a final diff of anything read since the
    // last interval tick.
    app.sync_journal();

    // Flush event log before exiting.
    if let Some(ref mut writer) = app.event_log {
        let _ = writer.flush();
    }

    // Restore terminal.
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
    terminal.show_cursor()?;

    result
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
        println!("{rendered}");
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
            Ok(AppEvent::AgentEvent(event)) => app.process_agent_event(event),
            Ok(AppEvent::SessionCleared) => app.reset_session(),
            Ok(AppEvent::Compacted(ev)) => {
                app.process_compaction(ev.summary, ev.timestamp, ev.agent_id, ev.metadata);
            }
            Ok(AppEvent::Tick) => {
                session.handle_tick(log_dir, app, serena_mode, project_path);
            }
            Err(flume::RecvTimeoutError::Timeout) => {}
            Err(flume::RecvTimeoutError::Disconnected) => break,
        }

        if app.should_quit {
            break;
        }
    }

    Ok(())
}
