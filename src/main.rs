// Shared modules live in the library crate (src/lib.rs).
use ambits::coverage;
use ambits::ingest;

// Binary-only modules.
mod events;
mod serena;
mod skill;
mod tui;
mod ui;

use std::fs;
use std::io;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use clap::{Parser as ClapParser, Subcommand};
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

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Manage the Claude Code skill for ambit
    Skill {
        #[command(subcommand)]
        command: SkillCommands,
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
    let cli = Cli::parse();

    // Handle subcommands first (don't require --project).
    if let Some(command) = cli.command {
        return match command {
            Commands::Skill { command } => match command {
                SkillCommands::Install { global, project } => skill::install(global, project),
            },
        };
    }

    // Resolve tool call mapping config. Warnings are displayed to stdout before TUI launch.
    let (tool_config, config_warnings) =
        ToolMappingConfig::resolve(cli.tools_config.as_deref());

    // Build the session ingester — coerce ToolMappingConfig to Arc<dyn ToolCallMapper>.
    let mapper: Arc<dyn ToolCallMapper> = tool_config;
    let ingester: Arc<dyn SessionIngester> =
        Arc::new(ingest::claude::ClaudeIngester::new(Arc::clone(&mapper)));

    // Original behavior — require --project for all other modes.
    let project = cli.project.ok_or_else(|| {
        color_eyre::eyre::eyre!("--project is required (use `ambits --project <path>`)")
    })?;
    let project_path = project.canonicalize().unwrap_or(project);
    let registry = ParserRegistry::new();
    let project_tree = if cli.serena {
        serena::scan_project_serena(&project_path)?
    } else {
        registry.scan_project(&project_path)?
    };

    if cli.dump {
        for w in &config_warnings {
            println!("[ambit warning] {w}");
        }
        coverage::dump_tree(&project_path, &project_tree);
        return Ok(());
    }

    if cli.coverage {
        for w in &config_warnings {
            println!("[ambit warning] {w}");
        }
        return coverage::run_report(&project_path, &project_tree, &cli.log_dir, &cli.session, &cli.agent, &*ingester);
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
    app.session_id = session_id.clone();

    // Pre-populate the ledger from existing session logs.
    if let (Some(ref log_dir), Some(ref session_id)) = (&log_dir, &session_id) {
        let log_files = ingester.session_log_files(log_dir, session_id);
        for log_file in &log_files {
            for event in ingester.parse_log_file(log_file) {
                app.process_agent_event(event);
            }
        }
    }

    let serena_mode = cli.serena;
    let result = run_tui(&mut terminal, &mut app, &project_path, &log_dir, session_id, &registry, serena_mode, &ingester);

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
