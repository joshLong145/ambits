use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use color_eyre::eyre::Result;
use notify::{Event as NotifyEvent, EventKind, RecursiveMode, Watcher};

use ambits::app::App;
use ambits::ingest::{EventTailer, SessionIngester};

use crate::events::AppEvent;

/// All mutable state owned by the TUI event loop.
///
/// Extracted from `run_tui` so that `handle_tick` and `handle_file_changed`
/// can be called and tested independently from the terminal/event-loop machinery.
pub struct TuiSession {
    /// The session ID the tailer is currently following.
    pub current_session_id: Option<String>,
    /// The moment the TUI was launched — used to ignore pre-existing session files.
    started_at: SystemTime,
    /// Tails new log lines from the active session.
    log_tailer: Option<Box<dyn EventTailer>>,
    /// Session format + tool mapper strategy.
    ingester: Arc<dyn SessionIngester>,
    /// (path, mtime) pairs for Serena .pkl cache files — used to detect live rebuilds.
    pkl_mtimes: Vec<(PathBuf, SystemTime)>,
    /// Holds the project-source watcher alive.
    _project_watcher: notify::RecommendedWatcher,
    /// Holds the log-directory watcher alive (None if no log dir configured).
    _log_watcher: Option<notify::RecommendedWatcher>,
}

impl TuiSession {
    /// Create a new session, setting up both filesystem watchers and the log tailer.
    pub fn new(
        project_path: &Path,
        log_dir: &Option<PathBuf>,
        session_id: Option<String>,
        watched_extensions: std::collections::HashSet<String>,
        ingester: Arc<dyn SessionIngester>,
        serena_mode: bool,
        tx: &flume::Sender<AppEvent>,
    ) -> Result<Self> {
        // Project source watcher — fires FileChanged for recognized extensions.
        let tx_file = tx.clone();
        let mut project_watcher =
            notify::recommended_watcher(move |res: Result<NotifyEvent, notify::Error>| {
                if let Ok(event) = res {
                    if matches!(event.kind, EventKind::Modify(_) | EventKind::Create(_)) {
                        for path in event.paths {
                            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                                if watched_extensions.contains(ext) {
                                    let _ = tx_file.try_send(AppEvent::FileChanged(path));
                                }
                            }
                        }
                    }
                }
            })?;
        project_watcher.watch(project_path, RecursiveMode::Recursive)?;

        // Log tailer — follows the current session's .jsonl files.
        let log_tailer: Option<Box<dyn EventTailer>> =
            if let (Some(ref ld), Some(ref sid)) = (log_dir, &session_id) {
                let files = ingester.session_log_files(ld, sid);
                Some(ingester.new_tailer(files))
            } else {
                None
            };

        // Log directory watcher — fires Tick when .jsonl files appear or change.
        let log_watcher = if let Some(ref ld) = log_dir {
            let ld_clone = ld.clone();
            let tx_log = tx.clone();
            let mut watcher =
                notify::recommended_watcher(move |res: Result<NotifyEvent, notify::Error>| {
                    if let Ok(event) = res {
                        if matches!(event.kind, EventKind::Modify(_) | EventKind::Create(_)) {
                            for path in event.paths {
                                if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
                                    let _ = tx_log.try_send(AppEvent::Tick);
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

        // Collect Serena .pkl cache modification times for live-rebuild detection.
        let pkl_mtimes: Vec<(PathBuf, SystemTime)> = if serena_mode {
            crate::serena::find_serena_caches(project_path)
                .into_iter()
                .filter_map(|p| {
                    fs::metadata(&p).ok()?.modified().ok().map(|t| (p, t))
                })
                .collect()
        } else {
            Vec::new()
        };

        Ok(Self {
            current_session_id: session_id,
            started_at: SystemTime::now(),
            log_tailer,
            ingester,
            pkl_mtimes,
            _project_watcher: project_watcher,
            _log_watcher: log_watcher,
        })
    }

    /// Handle a `Tick` event: detect new sessions, poll the tailer, and check Serena caches.
    pub fn handle_tick(
        &mut self,
        log_dir: &Option<PathBuf>,
        app: &mut App,
        serena_mode: bool,
        project_path: &Path,
    ) {
        // Check if Claude Code has started a new session (e.g. after /clear).
        if let Some(ref ld) = log_dir {
            if let Some(latest) = self.ingester.find_latest_session(ld) {
                let is_new_session =
                    self.current_session_id.as_deref() != Some(latest.as_str());
                let created_after_start = ld
                    .join(format!("{latest}.jsonl"))
                    .metadata()
                    .and_then(|m| m.modified())
                    .map(|mtime| mtime > self.started_at)
                    .unwrap_or(false);

                if is_new_session && created_after_start {
                    // New session detected — reset app state and switch tailer.
                    app.reset_session();
                    self.current_session_id = Some(latest.clone());
                    app.session_id = self.current_session_id.clone();

                    // Pre-populate from lines already written before this tick.
                    let new_files = self.ingester.session_log_files(ld, &latest);
                    for log_file in &new_files {
                        for event in self.ingester.parse_log_file(log_file) {
                            app.process_agent_event(event);
                        }
                    }

                    // Replace the tailer.
                    self.log_tailer = Some(self.ingester.new_tailer(new_files));
                }
            }
        }

        // Poll log tailer for new events.
        if let Some(ref mut tailer) = self.log_tailer {
            // Check for new agent files in the log directory.
            if let (Some(ref ld), Some(ref sid)) = (log_dir, &self.current_session_id) {
                let current_files = self.ingester.session_log_files(ld, sid);
                for f in current_files {
                    tailer.add_file(f);
                }
            }

            let output = tailer.read_new_events();
            for event in output.events {
                app.process_agent_event(event);
            }
        }

        // Check if Serena cache files changed.
        if serena_mode {
            let mut changed = false;
            for (path, mtime) in self.pkl_mtimes.iter_mut() {
                if let Ok(new_mtime) = fs::metadata(&*path).and_then(|m| m.modified()) {
                    if new_mtime != *mtime {
                        *mtime = new_mtime;
                        changed = true;
                    }
                }
            }
            if changed {
                if let Ok(new_tree) = crate::serena::scan_project_serena(project_path) {
                    let mut old_map = std::collections::HashMap::new();
                    for file in &app.project_tree.files {
                        ambits::tracking::collect_symbol_hashes(&file.symbols, &mut old_map);
                    }
                    app.project_tree = new_tree;
                    for file in &app.project_tree.files {
                        ambits::tracking::check_staleness(&file.symbols, &old_map, &mut app.ledger);
                    }
                    app.rebuild_tree_rows();
                }
            }
        }
    }

    /// Re-parse a changed source file and update the project tree in `app`.
    pub fn handle_file_changed(
        path: PathBuf,
        project_path: &Path,
        registry: &ambits::parser::ParserRegistry,
        app: &mut App,
    ) {
        if let Ok(rel) = path.strip_prefix(project_path) {
            if let Some(parser) = registry.parser_for(&path) {
                if let Ok(source) = fs::read_to_string(&path) {
                    if let Ok(new_file) = parser.parse_file(rel, &source) {
                        let rel_str = rel.to_string_lossy().to_string();
                        if let Some(existing) = app
                            .project_tree
                            .files
                            .iter_mut()
                            .find(|f| f.file_path.to_string_lossy() == rel_str)
                        {
                            ambits::tracking::mark_stale_symbols(
                                &existing.symbols,
                                &new_file.symbols,
                                &mut app.ledger,
                            );
                            *existing = new_file;
                        } else {
                            app.project_tree.files.push(new_file);
                            app.project_tree
                                .files
                                .sort_by(|a, b| a.file_path.cmp(&b.file_path));
                        }
                        app.rebuild_tree_rows();
                    }
                }
            }
        }
    }
}
