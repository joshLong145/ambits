// Shared modules live in the library crate (src/lib.rs).
use ambits::app;
use ambits::coverage;
use ambits::ingest;
use ambits::symbols;
use ambits::tracking;

// Binary-only modules.
mod events;
mod serena;
mod skill;
mod ui;

use std::fs;
use std::io;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Duration;

use clap::{Parser as ClapParser, Subcommand};
use color_eyre::eyre::Result;
use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use notify::{Event as NotifyEvent, EventKind, RecursiveMode, Watcher};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

use ambits::app::App;
use events::AppEvent;
use ambits::parser::ParserRegistry;
use ambits::symbols::ProjectTree;

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

    /// Output directory for event logs. If set, writes processed events to <dir>/<session>.log.
    #[arg(long)]
    log_output: Option<PathBuf>,

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

    // Original behavior — require --project for all other modes.
    let project = cli.project.ok_or_else(|| {
        color_eyre::eyre::eyre!("--project is required (use `ambits --project <path>`)")
    })?;
    let project_path = project.canonicalize().unwrap_or(project);
    let registry = ParserRegistry::new();
    let project_tree = if cli.serena {
        serena::scan_project_serena(&project_path)?
    } else {
        scan_project(&project_path, &registry)?
    };

    if cli.dump {
        dump_tree(&project_path, &project_tree);
        return Ok(());
    }

    if cli.coverage {
        return run_coverage_report(&project_path, &project_tree, &cli.log_dir, &cli.session);
    }

    // Resolve log directory and session.
    let log_dir = cli
        .log_dir
        .or_else(|| ingest::claude::log_dir_for_project(&project_path));
    
    let _ = std::fs::write("/tmp/marker-debug.txt", format!("log_dir: {:?}\n", log_dir));

    let session_id = cli.session.or_else(|| {
        log_dir
            .as_ref()
            .and_then(|d| ingest::claude::find_latest_session(d))
    });
    
    let _ = std::fs::OpenOptions::new().append(true).open("/tmp/marker-debug.txt")
        .and_then(|mut f| {
            use std::io::Write;
            writeln!(f, "session_id: {:?}", session_id)
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
        let _ = std::fs::write("/tmp/marker-log-debug.txt", format!("Creating log file at {:?}\n", log_path));
        let file = fs::File::create(&log_path)?;
        let _ = std::fs::OpenOptions::new().append(true).open("/tmp/marker-log-debug.txt")
            .and_then(|mut f| { use std::io::Write; writeln!(f, "Log file created successfully") });
        Some(io::BufWriter::new(file))
    } else {
        let _ = std::fs::write("/tmp/marker-log-debug.txt", "log_output is None\n");
        None
    };

    let mut app = App::new(project_tree, project_path.clone(), event_log);
    app.session_id = session_id.clone();

    // Pre-populate the ledger from existing session logs.
    if let (Some(ref log_dir), Some(ref session_id)) = (&log_dir, &session_id) {
        use std::io::Write;
        let mut debug_file = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/marker-debug.txt").unwrap();
        writeln!(debug_file, "=== ENTERING PRE-POPULATION ===").unwrap();
        let log_files = ingest::claude::session_log_files(log_dir, session_id);
        let _ = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/marker-debug.txt")
            .and_then(|mut f| {
                use std::io::Write;
                writeln!(f, "Found {} log files for session {}", log_files.len(), session_id)
            });
        for log_file in &log_files {
            let events = ingest::claude::parse_log_file(log_file);
            let _ = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/marker-debug.txt")
                .and_then(|mut f| {
                    use std::io::Write;
                    writeln!(f, "Parsed {} events from {:?}", events.len(), log_file.file_name())
                });
            for event in events {
                app.process_agent_event(event);
            }
        }
        let _ = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/marker-debug.txt")
            .and_then(|mut f| {
                use std::io::Write;
                writeln!(f, "Finished pre-population. Total events in activity: {}", app.activity.len())
            });
    }

    let serena_mode = cli.serena;
    let result = run_tui(&mut terminal, &mut app, &project_path, &log_dir, &session_id, &registry, serena_mode);

    // Flush event log before exiting.
    let _ = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/marker-log-debug.txt")
        .and_then(|mut f| { 
            use std::io::Write; 
            writeln!(f, "About to flush. event_log is_some: {}", app.event_log.is_some())
        });
    if let Some(ref mut writer) = app.event_log {
        let _ = writer.flush();
        let _ = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/marker-log-debug.txt")
            .and_then(|mut f| { use std::io::Write; writeln!(f, "Flushed event log") });
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
    session_id: &Option<String>,
    registry: &ParserRegistry,
    serena_mode: bool,
) -> Result<()> {
    let (tx, rx) = mpsc::channel::<AppEvent>();

    // Spawn key reader thread.
    events::spawn_key_reader(tx.clone());

    // Spawn tick timer (250ms).
    events::spawn_tick_timer(tx.clone(), Duration::from_millis(250));

    // Set up file watcher for project source changes.
    let tx_file = tx.clone();
    let watched_extensions = registry.supported_extensions();
    let mut _project_watcher = notify::recommended_watcher(move |res: Result<NotifyEvent, notify::Error>| {
        if let Ok(event) = res {
            if matches!(event.kind, EventKind::Modify(_) | EventKind::Create(_)) {
                for path in event.paths {
                    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                        if watched_extensions.contains(ext) {
                            let _ = tx_file.send(AppEvent::FileChanged(path));
                        }
                    }
                }
            }
        }
    })?;
    _project_watcher.watch(project_path, RecursiveMode::Recursive)?;

    // Set up log file tailer.
    let mut log_tailer = if let (Some(ref ld), Some(ref sid)) = (log_dir, session_id) {
        let files = ingest::claude::session_log_files(ld, sid);
        Some(ingest::claude::LogTailer::new(files))
    } else {
        None
    };

    // Set up file watcher for log directory (to detect new agent files).
    let tx_log = tx.clone();
    let mut _log_watcher = if let Some(ref ld) = log_dir {
        let ld_clone = ld.clone();
        let mut watcher = notify::recommended_watcher(move |res: Result<NotifyEvent, notify::Error>| {
            if let Ok(event) = res {
                if matches!(event.kind, EventKind::Modify(_) | EventKind::Create(_)) {
                    for path in event.paths {
                        if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
                            // Signal that log files changed — we'll poll in the tick handler.
                            let _ = tx_log.send(AppEvent::Tick);
                        }
                    }
                }
            }
        })?;
        watcher.watch(&ld_clone, RecursiveMode::NonRecursive)?;
        Some(watcher)
    } else {
        None
    };

    // Track Serena .pkl file modification times for live cache rebuilds.
    let mut pkl_mtimes: Vec<(PathBuf, std::time::SystemTime)> = if serena_mode {
        serena::find_serena_caches(project_path)
            .into_iter()
            .filter_map(|p| fs::metadata(&p).ok()?.modified().ok().map(|t| (p, t)))
            .collect()
    } else {
        Vec::new()
    };

    loop {
        terminal.draw(|f| ui::render(f, app))?;

        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(AppEvent::Key(key)) => app.handle_key(key),
            Ok(AppEvent::Mouse(mouse)) => app.handle_mouse(mouse),
            Ok(AppEvent::FileChanged(path)) => {
                // Re-parse the changed file and update the project tree.
                if let Ok(rel) = path.strip_prefix(project_path) {
                    if let Some(parser) = registry.parser_for(&path) {
                        if let Ok(source) = fs::read_to_string(&path) {
                            if let Ok(new_file) = parser.parse_file(rel, &source) {
                                // Replace the file in the project tree.
                                let rel_str = rel.to_string_lossy().to_string();
                                if let Some(existing) = app.project_tree.files.iter_mut().find(|f| {
                                    f.file_path.to_string_lossy() == rel_str
                                }) {
                                    // Mark symbols as stale if their hashes changed.
                                    mark_stale_symbols(&existing.symbols, &new_file.symbols, &mut app.ledger);
                                    *existing = new_file;
                                } else {
                                    app.project_tree.files.push(new_file);
                                    app.project_tree.files.sort_by(|a, b| a.file_path.cmp(&b.file_path));
                                }
                                app.rebuild_tree_rows();
                            }
                        }
                    }
                }
            }
            Ok(AppEvent::AgentEvent(event)) => {
                app.process_agent_event(event);
            }
            Ok(AppEvent::Tick) => {
                // Poll log tailer for new events.
                if let Some(ref mut tailer) = log_tailer {
                    // Check for new agent files in the log directory.
                    if let (Some(ref ld), Some(ref sid)) = (log_dir, session_id) {
                        let current_files = ingest::claude::session_log_files(ld, sid);
                        for f in current_files {
                            tailer.add_file(f);
                        }
                    }

                    let new_events = tailer.read_new_events();
                    for event in new_events {
                        app.process_agent_event(event);
                    }
                }

                // Check if Serena cache files changed.
                if serena_mode {
                    let mut changed = false;
                    for (path, mtime) in pkl_mtimes.iter_mut() {
                        if let Ok(new_mtime) = fs::metadata(&*path).and_then(|m| m.modified()) {
                            if new_mtime != *mtime {
                                *mtime = new_mtime;
                                changed = true;
                            }
                        }
                    }
                    if changed {
                        if let Ok(new_tree) = serena::scan_project_serena(project_path) {
                            // Collect old hashes, then check staleness against new tree.
                            let mut old_map = std::collections::HashMap::new();
                            for file in &app.project_tree.files {
                                collect_symbol_hashes(&file.symbols, &mut old_map);
                            }
                            app.project_tree = new_tree;
                            for file in &app.project_tree.files {
                                check_staleness(&file.symbols, &old_map, &mut app.ledger);
                            }
                            app.rebuild_tree_rows();
                        }
                    }
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }

        if app.should_quit {
            break;
        }
    }

    Ok(())
}

/// Compare old and new symbols and mark changed ones as stale in the ledger.
fn mark_stale_symbols(
    old_symbols: &[symbols::SymbolNode],
    new_symbols: &[symbols::SymbolNode],
    ledger: &mut tracking::ContextLedger,
) {
    // Build a map of old symbol IDs to their hashes.
    let mut old_map = std::collections::HashMap::new();
    collect_symbol_hashes(old_symbols, &mut old_map);

    // Check new symbols against old hashes.
    check_staleness(new_symbols, &old_map, ledger);
}

fn collect_symbol_hashes(
    symbols: &[symbols::SymbolNode],
    map: &mut std::collections::HashMap<String, [u8; 32]>,
) {
    for sym in symbols {
        map.insert(sym.id.clone(), sym.content_hash);
        collect_symbol_hashes(&sym.children, map);
    }
}

fn check_staleness(
    symbols: &[symbols::SymbolNode],
    old_map: &std::collections::HashMap<String, [u8; 32]>,
    ledger: &mut tracking::ContextLedger,
) {
    for sym in symbols {
        if let Some(old_hash) = old_map.get(&sym.id) {
            if *old_hash != sym.content_hash {
                ledger.mark_stale_if_changed(&sym.id, sym.content_hash);
            }
        }
        check_staleness(&sym.children, old_map, ledger);
    }
}

fn run_coverage_report(
    project_path: &Path,
    project_tree: &ProjectTree,
    log_dir_opt: &Option<PathBuf>,
    session_opt: &Option<String>,
) -> Result<()> {
    use coverage::{CoverageFormatter, CoverageReport, TextFormatter};
    use tracking::ContextLedger;

    // 1. Resolve log directory
    let log_dir = log_dir_opt
        .clone()
        .or_else(|| ingest::claude::log_dir_for_project(project_path));

    // 2. Find session (auto-detect if not provided)
    let session_id = session_opt.clone().or_else(|| {
        log_dir
            .as_ref()
            .and_then(|d| ingest::claude::find_latest_session(d))
    });

    // 3. Build ledger from session logs
    let mut ledger = ContextLedger::new();
    if let (Some(ref log_dir), Some(ref sid)) = (&log_dir, &session_id) {
        let log_files = ingest::claude::session_log_files(log_dir, sid);
        for log_file in &log_files {
            let events = ingest::claude::parse_log_file(log_file);
            for event in events {
                if let Some(ref file_path) = event.file_path {
                    // Normalize the tool call path
                    let tool_rel = app::normalize_tool_path(file_path, project_path);

                    for file in &project_tree.files {
                        if file.file_path == tool_rel {
                            if event.target_symbol.is_some() || event.target_lines.is_some() {
                                app::mark_targeted_symbols(&file.symbols, &event, &mut ledger);
                            } else {
                                app::mark_file_symbols(&file.symbols, &event, &mut ledger);
                            }
                        }
                    }
                }
            }
        }
    }

    // 4. Generate report
    let mut report = CoverageReport::from_project(project_tree, &ledger);
    report.session_id = session_id;

    // 5. Format and print
    let formatter = TextFormatter::default();
    print!("{}", formatter.format(&report));

    Ok(())
}

fn dump_tree(root: &Path, project_tree: &ProjectTree) {
    println!(
        "Project: {} ({} files, {} symbols)",
        root.display(),
        project_tree.total_files(),
        project_tree.total_symbols(),
    );
    println!();

    for file in &project_tree.files {
        println!("  {} ({} lines)", file.file_path.display(), file.total_lines);
        for sym in &file.symbols {
            print_symbol(sym, 4);
        }
    }
}

fn print_symbol(sym: &symbols::SymbolNode, indent: usize) {
    let pad = " ".repeat(indent);
    println!(
        "{}{} {} [L{}-{}] (~{} tokens)",
        pad,
        sym.label,
        sym.name,
        sym.line_range.start,
        sym.line_range.end,
        sym.estimated_tokens,
    );
    for child in &sym.children {
        print_symbol(child, indent + 2);
    }
}

fn scan_project(root: &Path, registry: &ParserRegistry) -> Result<ProjectTree> {
    use ignore::WalkBuilder;

    let mut files = Vec::new();

    // Ignore dot files and files in .gitignore
    for result in WalkBuilder::new(root)
        .hidden(true)
        .git_ignore(true)
        .build()
    {
        let entry = match result {
            Ok(e) => e,
            Err(_) => continue,
        };

        let path = entry.path();
        if path.is_dir() {
            continue;
        }

        if let Some(parser) = registry.parser_for(path) {
            let source = fs::read_to_string(path)?;
            let rel_path = path.strip_prefix(root).unwrap_or(path);
            match parser.parse_file(rel_path, &source) {
                Ok(file_symbols) => files.push(file_symbols),
                Err(e) => {
                    eprintln!("Warning: failed to parse {}: {}", path.display(), e);
                }
            }
        }
    }

    files.sort_by(|a, b| a.file_path.cmp(&b.file_path));

    Ok(ProjectTree {
        root: root.to_path_buf(),
        files,
    })
}
