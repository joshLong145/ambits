use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use color_eyre::eyre::Result;
use notify::{Event as NotifyEvent, EventKind, RecursiveMode, Watcher};

use ambits::app::App;
use ambits::filter::ProjectScope;
use ambits::ingest::{EventTailer, SessionIngester};

use crate::events::AppEvent;

/// All mutable state owned by the TUI event loop.
///
/// Extracted from `run_tui` so that `handle_tick` and `handle_file_changed`
/// can be called and tested independently from the terminal/event-loop machinery.
pub struct TuiSession {
    /// The session ID the tailer is currently following.
    pub current_session_id: Option<String>,
    /// Session ids whose logs already existed when the TUI launched.
    ///
    /// Switching is driven by *appearance*, not recency: only an id absent
    /// from this set is a session that started after us. See [`Self::handle_tick`].
    known_sessions: HashSet<String>,
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
    /// Hands write events, tagged with their session, to the attribution
    /// worker (spec §1).
    write_tx: flume::Sender<(Arc<str>, ambits::ingest::WriteEvent)>,
}

/// Attribute writes on a worker thread, never the render thread (spec §1):
/// each costs up to two parses, and a replay can queue hundreds. Results come
/// back to the TUI loop as `AppEvent::WriteRecorded`. The worker owns its own
/// `ParserRegistry`, so nothing is shared across threads.
fn spawn_write_attributor(
    project_root: PathBuf,
    symbol_level: bool,
    tx: flume::Sender<AppEvent>,
) -> flume::Sender<(Arc<str>, ambits::ingest::WriteEvent)> {
    let (write_tx, write_rx) = flume::unbounded::<(Arc<str>, ambits::ingest::WriteEvent)>();
    std::thread::spawn(move || {
        let registry = ambits::parser::ParserRegistry::new();
        for (session, event) in write_rx.iter() {
            let Some(record) =
                ambits::writes::build_record(&event, &project_root, &registry, symbol_level)
            else {
                continue;
            };
            if tx.send(AppEvent::WriteRecorded { session, record }).is_err() {
                break;
            }
        }
    });
    write_tx
}

/// Replay a session's logs into `app` — every tool call, compaction, clear
/// and write already on disk — and return where its tailer continues.
///
/// The one replay loop, shared by startup and session switches.
pub fn replay_session(
    app: &mut App,
    ingester: &dyn SessionIngester,
    files: Vec<PathBuf>,
    project_root: &Path,
) -> ambits::ingest::Handoff {
    use ambits::ingest::SessionEvent;
    let mut handoff = ambits::ingest::Handoff { project_root: Some(project_root.to_path_buf()), ..Default::default() };
    for file in files {
        let replay = ingester.replay_log_file(&file, project_root);
        for event in replay.events {
            match event {
                SessionEvent::ToolCall(tc) => app.process_agent_event(tc),
                SessionEvent::Compacted { summary, timestamp, agent_id, metadata } => {
                    app.process_compaction(summary, timestamp, agent_id, metadata);
                }
                SessionEvent::SessionCleared => app.reset_session(),
                SessionEvent::Write(w) => app.queue_write(w),
                SessionEvent::ToolFinished(f) => app.process_tool_finished(&f),
                SessionEvent::Prompt(p) => app.process_prompt(&p),
            }
        }
        handoff.files.push((file, replay.offset));
        handoff.awaiting.extend(replay.awaiting);
    }
    handoff
}

/// The session the TUI starts on, as replayed before it opened.
pub struct StartingSession {
    pub id: Option<String>,
    /// Where the startup replay stopped; the tailer continues from here.
    pub handoff: ambits::ingest::Handoff,
}

impl TuiSession {
    /// Create a new session, setting up both filesystem watchers and the log tailer.
    pub fn new(
        project_path: &Path,
        log_dir: &Option<PathBuf>,
        starting: StartingSession,
        watched_extensions: std::collections::HashSet<String>,
        ingester: Arc<dyn SessionIngester>,
        serena_mode: bool,
        tx: &flume::Sender<AppEvent>,
    ) -> Result<Self> {
        // Project source watcher — fires FileChanged for recognized extensions
        // that are actually part of the project. Without the scope check the
        // watcher sees everything under the root, including the build output
        // the scanner deliberately skips.
        let tx_file = tx.clone();
        let scope = ProjectScope::new(project_path);
        let mut project_watcher =
            notify::recommended_watcher(move |res: Result<NotifyEvent, notify::Error>| {
                if let Ok(event) = res {
                    let removed = matches!(event.kind, EventKind::Remove(_));
                    if removed || matches!(event.kind, EventKind::Modify(_) | EventKind::Create(_))
                    {
                        for path in event.paths {
                            let watched = path
                                .extension()
                                .and_then(|e| e.to_str())
                                .is_some_and(|ext| watched_extensions.contains(ext));
                            if !watched || !scope.contains(&path) {
                                continue;
                            }
                            let _ = tx_file.try_send(if removed {
                                AppEvent::FileRemoved(path)
                            } else {
                                AppEvent::FileChanged(path)
                            });
                        }
                    }
                }
            })?;
        project_watcher.watch(project_path, RecursiveMode::Recursive)?;

        // Log tailer — follows the current session's .jsonl files from where
        // the startup replay stopped.
        let StartingSession { id: session_id, handoff } = starting;
        let log_tailer: Option<Box<dyn EventTailer>> = match (log_dir, &session_id) {
            (Some(_), Some(_)) => Some(ingester.resume_tailer(handoff)),
            _ => None,
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

        // Snapshot before the first tick: everything already on disk is, by
        // definition, not a session that started after us.
        let known_sessions = log_dir
            .as_ref()
            .map(|ld| existing_session_ids(ld))
            .unwrap_or_default();

        // Serena mode's tree ids need not match a tree-sitter parse, so its
        // writes are file-level (spec §2.5).
        let write_tx = spawn_write_attributor(project_path.to_path_buf(), !serena_mode, tx.clone());

        Ok(Self {
            current_session_id: session_id,
            known_sessions,
            log_tailer,
            ingester,
            pkl_mtimes,
            _project_watcher: project_watcher,
            _log_watcher: log_watcher,
            write_tx,
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
        // Poll the tailer first. On a session switch below, this is the old
        // session's last chance: anything it logged since the previous tick
        // (reads, writes, their results) would otherwise be dropped with the
        // tailer, and the outgoing journal sync would have nothing to flush.
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
            for write in output.writes {
                app.queue_write(write);
            }
            for finished in &output.finished {
                app.process_tool_finished(finished);
            }
            for prompt in &output.prompts {
                app.process_prompt(prompt);
            }
            for compaction in output.compactions {
                app.process_compaction(compaction.summary, compaction.timestamp, compaction.agent_id, compaction.metadata);
            }
        }

        // Check if Claude Code has started a new session (e.g. after /clear).
        if let Some(ref ld) = log_dir {
            if let Some(latest) = self.ingester.find_latest_session(ld) {
                let is_new_session =
                    self.current_session_id.as_deref() != Some(latest.as_str());
                // Appearance, not recency. `find_latest_session` ranks purely
                // by mtime, and a log file's mtime moves for reasons unrelated
                // to session activity — a session whose final line is
                // `continued-in` (i.e. already dead) still gets touched. The
                // guard this replaced compared that mtime against the TUI's
                // start time, so any such touch marked a corpse "new": every
                // time the live session paused, `latest` flipped to the dead
                // one and back, each flip resetting the app and re-parsing
                // megabytes of JSONL on the render thread.
                let is_unseen = !self.known_sessions.contains(&latest);

                if is_new_session && is_unseen {
                    // New session detected — reset app state and switch tailer.
                    self.known_sessions.insert(latest.clone());
                    self.current_session_id = Some(latest.clone());
                    // Adopt the id *before* resetting: the reset re-seeds the
                    // agent tree's root from it. See `App::switch_session`.
                    app.switch_session(self.current_session_id.clone());
                    app.session_slug = log_dir.as_ref()
                        .zip(self.current_session_id.as_ref())
                        .and_then(|(ld2, sid)| self.ingester.session_slug(ld2, sid));

                    // Pre-populate from lines already written before this
                    // tick, then tail from exactly where that stopped.
                    let new_files = self.ingester.session_log_files(ld, &latest);
                    let handoff = replay_session(app, &*self.ingester, new_files, project_path);
                    self.log_tailer = Some(self.ingester.resume_tailer(handoff));

                    // Same order as startup: journal after the replay.
                    let attach = app.attach_journal();
                    for warning in attach.warnings {
                        log::warn!(target: "ambits::journal", "{warning}");
                    }
                    if let Some(stats) = attach.rehydrated {
                        log::info!(
                            target: "ambits::journal",
                            corrected = stats.corrected, recovered = stats.inserted,
                            moved = stats.moved, stale = stats.drifted;
                            "rehydrated from journal"
                        );
                    }
                }
            }
        }

        // Hand queued writes — from the tailer, a session switch, or startup
        // replay — to the attribution worker. A send fails only if the worker
        // died; the writes are then simply not journaled.
        for write in app.take_pending_writes() {
            let _ = self.write_tx.send(write);
        }

        // Bring the coverage journal up to date. Interval-gated internally, so
        // this is a cheap no-op on most ticks.
        app.maybe_sync_journal();

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
                let filter = app.filter.as_deref();
                if let Ok(new_tree) = crate::serena::scan_project_serena(project_path, filter) {
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
    ///
    /// If `app.filter` is set and the changed file's project-relative path
    /// does not satisfy the filter, the function returns early — keeping the
    /// excluded file out of the tree even after a `notify` event fires on it.
    pub fn handle_file_changed(
        path: PathBuf,
        project_path: &Path,
        registry: &ambits::parser::ParserRegistry,
        app: &mut App,
    ) {
        if let Ok(rel) = path.strip_prefix(project_path) {
            if let Some(filter) = app.filter.as_deref() {
                if !filter.matches(rel) {
                    return;
                }
            }
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

    /// Drop a deleted file's row from the project tree.
    ///
    /// The ledger keeps its entries for the removed symbols on purpose: they
    /// record what an agent read, which stays true after the file is gone, and
    /// a rename fires Remove + Create so discarding them would lose coverage
    /// across an ordinary refactor.
    pub fn handle_file_removed(path: PathBuf, project_path: &Path, app: &mut App) {
        let Ok(rel) = path.strip_prefix(project_path) else {
            return;
        };
        let rel_str = rel.to_string_lossy().to_string();
        let before = app.project_tree.files.len();
        app.project_tree
            .files
            .retain(|f| f.file_path.to_string_lossy() != rel_str);

        if app.project_tree.files.len() != before {
            app.rebuild_tree_rows();
        }
    }
}

/// Session ids whose log files already exist in `log_dir`.
///
/// Deliberately does not validate the stem as a session id. The set is only
/// ever consulted to *suppress* a switch, and anything that is not a real
/// session id can never match what `find_latest_session` returns — so an
/// over-inclusive snapshot costs nothing, while a parser that disagreed with
/// the ingester's idea of a session id would silently re-open this bug.
fn existing_session_ids(log_dir: &Path) -> HashSet<String> {
    let Ok(entries) = fs::read_dir(log_dir) else {
        return HashSet::new();
    };
    entries
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                return None;
            }
            path.file_stem().and_then(|s| s.to_str()).map(String::from)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ambits::expansion::RowKind;
    use ambits::ingest::{SessionEvent, TailerOutput};
    use ambits::symbols::ProjectTree;
    use std::sync::Mutex;

    /// Reports whatever session id the test most recently set, standing in for
    /// `find_latest_session`'s mtime ranking without touching the clock.
    struct MockIngester {
        latest: Mutex<Option<String>>,
        /// Writes the next tailer yields on its first poll.
        tailed_writes: Mutex<Vec<ambits::ingest::WriteEvent>>,
    }

    impl MockIngester {
        fn new(latest: &str) -> Arc<Self> {
            Arc::new(Self {
                latest: Mutex::new(Some(latest.to_string())),
                tailed_writes: Mutex::new(Vec::new()),
            })
        }

        fn set_latest(&self, id: &str) {
            *self.latest.lock().unwrap() = Some(id.to_string());
        }
    }

    impl SessionIngester for MockIngester {
        fn log_dir_for_project(&self, _project_path: &Path) -> Option<PathBuf> {
            None
        }
        fn find_latest_session(&self, _log_dir: &Path) -> Option<String> {
            self.latest.lock().unwrap().clone()
        }
        fn session_log_files(&self, _log_dir: &Path, _session_id: &str) -> Vec<PathBuf> {
            Vec::new()
        }
        fn parse_log_file(&self, _path: &Path) -> Vec<SessionEvent> {
            Vec::new()
        }
        fn new_tailer(&self, _files: Vec<PathBuf>) -> Box<dyn EventTailer> {
            Box::new(MockTailer { writes: std::mem::take(&mut *self.tailed_writes.lock().unwrap()) })
        }
    }

    struct MockTailer {
        writes: Vec<ambits::ingest::WriteEvent>,
    }

    impl EventTailer for MockTailer {
        fn add_file(&mut self, _path: PathBuf) {}
        fn read_new_events(&mut self) -> TailerOutput {
            TailerOutput {
                events: Vec::new(),
                compactions: Vec::new(),
                session_cleared: false,
                writes: std::mem::take(&mut self.writes),
                finished: Vec::new(),
                prompts: Vec::new(),
            }
        }
    }

    fn touch_session(log_dir: &Path, id: &str) {
        fs::write(log_dir.join(format!("{id}.jsonl")), "{}\n").unwrap();
    }

    fn empty_app(root: &Path, session_id: &str) -> App {
        let tree = ProjectTree {
            root: root.to_path_buf(),
            files: Vec::new(),
        };
        let mut app = App::new(tree, root.to_path_buf());
        app.set_session_id(Some(session_id.to_string()));
        app
    }

    /// Builds a session following `current`, with every id in `existing`
    /// already on disk before construction.
    fn session_with(
        project: &Path,
        log_dir: &Path,
        existing: &[&str],
        current: &str,
        ingester: Arc<MockIngester>,
    ) -> (TuiSession, flume::Receiver<AppEvent>) {
        for id in existing {
            touch_session(log_dir, id);
        }
        let (tx, rx) = flume::bounded(64);
        let session = TuiSession::new(
            project,
            &Some(log_dir.to_path_buf()),
            StartingSession { id: Some(current.to_string()), handoff: Default::default() },
            HashSet::new(),
            ingester,
            false,
            &tx,
        )
        .unwrap();
        (session, rx)
    }

    /// F1: after a switch the tree's root must be the *live* session.
    ///
    /// The bug: `reset_session` rebuilt the agent tree and re-seeded its root
    /// from the still-stale `session_id`, and `AgentTree::add_agent` latches
    /// `root_id` on the first parentless node — so the root stayed pinned to
    /// the dead session and the new one was appended as a second parentless
    /// node, rendering two rows both labelled `main`.
    #[test]
    fn switching_sessions_moves_the_agent_tree_root_to_the_new_session() {
        let project = tempfile::tempdir().unwrap();
        let logs = tempfile::tempdir().unwrap();
        let ingester = MockIngester::new("old-session");

        let (mut session, _rx) = session_with(
            project.path(),
            logs.path(),
            &["old-session"],
            "old-session",
            Arc::clone(&ingester),
        );
        let mut app = empty_app(project.path(), "old-session");
        assert_eq!(app.agent_tree.root_id.as_deref(), Some("old-session"));

        // A session that did not exist at startup appears and becomes latest.
        touch_session(logs.path(), "new-session");
        ingester.set_latest("new-session");
        session.handle_tick(&Some(logs.path().to_path_buf()), &mut app, false, project.path());

        assert_eq!(session.current_session_id.as_deref(), Some("new-session"));
        assert_eq!(
            app.agent_tree.root_id.as_deref(),
            Some("new-session"),
            "root must follow the live session, not stay latched to the dead one"
        );
        assert_eq!(
            app.flattened_agents().len(),
            1,
            "the dead session must not survive as a second parentless node"
        );
    }

    /// F2: a session that already existed at startup must never trigger a
    /// switch, however recently its file was written.
    ///
    /// The bug: the guard compared the candidate's *mtime* against the TUI's
    /// start time. Log files get touched without gaining events — a dead
    /// session's mtime moved ~16h past its last line in the wild — so any
    /// pre-existing session could win `find_latest_session`'s mtime ranking
    /// and force a full reset plus a re-parse of its entire log.
    #[test]
    fn a_pre_existing_session_never_triggers_a_switch() {
        let project = tempfile::tempdir().unwrap();
        let logs = tempfile::tempdir().unwrap();
        let ingester = MockIngester::new("live-session");

        let (mut session, _rx) = session_with(
            project.path(),
            logs.path(),
            &["live-session", "dead-session"],
            "live-session",
            Arc::clone(&ingester),
        );
        let mut app = empty_app(project.path(), "live-session");

        // The dead session's file is written again — exactly what bumped its
        // mtime past `started_at` under the old guard.
        touch_session(logs.path(), "dead-session");
        ingester.set_latest("dead-session");
        session.handle_tick(&Some(logs.path().to_path_buf()), &mut app, false, project.path());

        assert_eq!(
            session.current_session_id.as_deref(),
            Some("live-session"),
            "a session present at startup is not a new session"
        );
        assert_eq!(app.agent_tree.root_id.as_deref(), Some("live-session"));
    }

    /// Switching twice must leave exactly one root, not accumulate corpses.
    #[test]
    fn repeated_switches_do_not_accumulate_root_nodes() {
        let project = tempfile::tempdir().unwrap();
        let logs = tempfile::tempdir().unwrap();
        let ingester = MockIngester::new("s1");

        let (mut session, _rx) = session_with(
            project.path(),
            logs.path(),
            &["s1"],
            "s1",
            Arc::clone(&ingester),
        );
        let mut app = empty_app(project.path(), "s1");

        for id in ["s2", "s3"] {
            touch_session(logs.path(), id);
            ingester.set_latest(id);
            session.handle_tick(&Some(logs.path().to_path_buf()), &mut app, false, project.path());
        }

        assert_eq!(session.current_session_id.as_deref(), Some("s3"));
        assert_eq!(app.agent_tree.root_id.as_deref(), Some("s3"));
        assert_eq!(app.flattened_agents().len(), 1);
    }

    /// Once switched to, a session must not be re-detected as new on the next
    /// tick — otherwise the reset loop simply moves to the new id.
    #[test]
    fn the_adopted_session_is_not_switched_to_again() {
        let project = tempfile::tempdir().unwrap();
        let logs = tempfile::tempdir().unwrap();
        let ingester = MockIngester::new("s1");

        let (mut session, _rx) = session_with(
            project.path(),
            logs.path(),
            &["s1"],
            "s1",
            Arc::clone(&ingester),
        );
        let mut app = empty_app(project.path(), "s1");

        touch_session(logs.path(), "s2");
        ingester.set_latest("s2");
        session.handle_tick(&Some(logs.path().to_path_buf()), &mut app, false, project.path());

        // Plant a sentinel that only a reset would clear.
        app.session_slug = Some("sentinel".to_string());
        session.handle_tick(&Some(logs.path().to_path_buf()), &mut app, false, project.path());

        assert_eq!(session.current_session_id.as_deref(), Some("s2"));
        assert_eq!(
            app.session_slug.as_deref(),
            Some("sentinel"),
            "a second reset would have cleared this"
        );
    }

    /// Build a real two-file project and scan it, so the tree under test is
    /// shaped exactly as the TUI's is.
    fn scanned_project() -> (tempfile::TempDir, App) {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("alpha.rs"), "pub fn alpha() {}\n").unwrap();
        fs::write(dir.path().join("beta.rs"), "pub fn beta() {}\n").unwrap();

        let tree = ambits::parser::ParserRegistry::new()
            .scan_project(dir.path(), None)
            .unwrap();
        assert_eq!(tree.files.len(), 2);

        let app = App::new(tree, dir.path().to_path_buf());
        (dir, app)
    }

    /// Before this, nothing handled `EventKind::Remove` — a deleted file kept
    /// its row until the TUI was restarted, so the tree only ever grew.
    #[test]
    fn removing_a_file_drops_it_from_the_tree() {
        let (dir, mut app) = scanned_project();

        TuiSession::handle_file_removed(dir.path().join("beta.rs"), dir.path(), &mut app);

        let remaining: Vec<String> = app
            .project_tree
            .files
            .iter()
            .map(|f| f.file_path.to_string_lossy().to_string())
            .collect();
        assert_eq!(remaining, vec!["alpha.rs".to_string()]);
    }

    /// The ledger is a record of what was read, which a deletion does not
    /// falsify — and a rename arrives as Remove + Create, so dropping entries
    /// here would lose coverage across an ordinary refactor.
    #[test]
    fn removing_a_file_leaves_the_ledger_alone() {
        let (dir, mut app) = scanned_project();
        let symbol = app.project_tree.files[1].symbols[0].clone();
        app.ledger.record(
            symbol.id.clone(),
            ambits::tracking::ReadDepth::FullBody,
            symbol.content_hash,
            "agent-1".to_string(),
            10,
        );
        let before = app.ledger.entries.len();
        assert!(before > 0);

        TuiSession::handle_file_removed(dir.path().join("beta.rs"), dir.path(), &mut app);

        assert_eq!(app.ledger.entries.len(), before);
    }

    #[test]
    fn removing_an_unknown_file_is_a_no_op() {
        let (dir, mut app) = scanned_project();

        TuiSession::handle_file_removed(dir.path().join("never_existed.rs"), dir.path(), &mut app);

        assert_eq!(app.project_tree.files.len(), 2);
    }

    #[test]
    fn existing_session_ids_collects_only_jsonl_stems() {
        let dir = tempfile::tempdir().unwrap();
        touch_session(dir.path(), "alpha");
        touch_session(dir.path(), "beta");
        fs::write(dir.path().join("notes.txt"), "hi").unwrap();

        let ids = existing_session_ids(dir.path());
        assert_eq!(ids.len(), 2);
        assert!(ids.contains("alpha"));
        assert!(ids.contains("beta"));
    }

    #[test]
    fn existing_session_ids_is_empty_for_a_missing_directory() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("nope");
        assert!(existing_session_ids(&missing).is_empty());
    }

    // --- expansion of files arriving after startup ---

    /// Write `name` and deliver it to the TUI as the file watcher would.
    fn save(dir: &Path, app: &mut App, name: &str, src: &str) {
        let path = dir.join(name);
        fs::write(&path, src).unwrap();
        TuiSession::handle_file_changed(path, dir, &ambits::parser::ParserRegistry::new(), app);
    }

    fn row<'a>(app: &'a App, id: &str) -> Option<&'a ambits::app::TreeRow> {
        app.tree_rows.iter().find(|r| r.symbol_id == id)
    }

    #[test]
    fn files_present_at_startup_start_collapsed() {
        let (_dir, app) = scanned_project();
        assert!(app.tree_rows.iter().all(|r| r.is_file() && !r.is_expanded));
    }

    /// A file created while the TUI runs used to arrive expanded: the
    /// collapsed set was seeded once at startup, so nothing ever collapsed a
    /// later arrival.
    #[test]
    fn a_file_created_after_startup_starts_collapsed() {
        let (dir, mut app) = scanned_project();

        save(dir.path(), &mut app, "gamma.rs", "pub fn gamma() {}\n");

        assert!(!row(&app, "gamma.rs").expect("gamma.rs row").is_expanded);
        assert!(
            row(&app, "gamma.rs::gamma").is_none(),
            "a collapsed file's symbols must not be flattened into rows"
        );
    }

    #[test]
    fn a_reparsed_file_keeps_its_expansion() {
        let (dir, mut app) = scanned_project();
        app.set_expanded("alpha.rs", RowKind::File, true);

        save(dir.path(), &mut app, "alpha.rs", "pub fn alpha() {}\npub fn alpha2() {}\n");

        assert!(row(&app, "alpha.rs").unwrap().is_expanded);
        assert!(row(&app, "alpha.rs::alpha2").is_some(), "the new symbol is visible");
    }

    /// An editor's atomic save arrives as Remove + Create; the file must not
    /// snap shut on every save.
    #[test]
    fn an_expanded_file_deleted_and_recreated_stays_expanded() {
        let (dir, mut app) = scanned_project();
        app.set_expanded("beta.rs", RowKind::File, true);

        TuiSession::handle_file_removed(dir.path().join("beta.rs"), dir.path(), &mut app);
        assert!(row(&app, "beta.rs").is_none());
        save(dir.path(), &mut app, "beta.rs", "pub fn beta() {}\n");

        assert!(row(&app, "beta.rs").unwrap().is_expanded);
    }

    /// Writes queued from any source are attributed off the render thread
    /// and come back as `AppEvent::WriteRecorded` (spec §1).
    #[test]
    fn a_queued_write_is_attributed_by_the_worker_and_returned() {
        let project = tempfile::tempdir().unwrap();
        let log_dir = tempfile::tempdir().unwrap();
        let ingester = MockIngester::new("s1");
        let (mut session, rx) = session_with(project.path(), log_dir.path(), &["s1"], "s1", ingester);
        let mut app = empty_app(project.path(), "s1");

        app.queue_write(new_file_write(project.path()));
        session.handle_tick(&Some(log_dir.path().to_path_buf()), &mut app, false, project.path());
        assert!(app.take_pending_writes().is_empty(), "drained to the worker");

        let (session, record) = loop {
            match rx.recv_timeout(std::time::Duration::from_secs(10)) {
                Ok(AppEvent::WriteRecorded { session, record }) => break (session, record),
                Ok(_) => continue,
                Err(e) => panic!("no WriteRecorded: {e:?}"),
            }
        };
        assert_eq!(&*session, "s1", "tagged with the session it happened in");
        assert_eq!(record.op, "toolu_1");
        assert_eq!(record.file, "src/new.rs");
        assert_eq!(record.level, ambits::writes::Level::Symbol);
        assert_eq!(record.syms[0].0, "src/new.rs::a");
    }

    /// An agent creating `src/new.rs` with one function, `a`.
    fn new_file_write(project: &Path) -> ambits::ingest::WriteEvent {
        ambits::ingest::WriteEvent {
            op: Arc::from("toolu_1"),
            agent_id: Arc::from("agent-1"),
            tool_name: Arc::from("Write"),
            path: project.join("src/new.rs"),
            timestamp: "2026-09-26T10:00:01Z".into(),
            source: ambits::ingest::WriteSource::Write {
                original: None,
                content: "fn a() {}\n".into(),
                create: true,
                hunks: None,
                user_modified: false,
            },
        }
    }

    /// The old session's tailer is drained before a switch in the same tick,
    /// so its last write is still attributed to it rather than lost with the
    /// tailer.
    #[test]
    fn a_write_tailed_in_the_tick_a_session_ends_stays_with_that_session() {
        let project = tempfile::tempdir().unwrap();
        let log_dir = tempfile::tempdir().unwrap();
        let ingester = MockIngester::new("s1");
        ingester.tailed_writes.lock().unwrap().push(new_file_write(project.path()));
        let (mut session, rx) =
            session_with(project.path(), log_dir.path(), &["s1"], "s1", Arc::clone(&ingester));
        let mut app = empty_app(project.path(), "s1");

        ingester.set_latest("s2");
        touch_session(log_dir.path(), "s2");
        session.handle_tick(&Some(log_dir.path().to_path_buf()), &mut app, false, project.path());
        assert_eq!(app.session_id.as_deref(), Some("s2"), "switched in this tick");

        let session_of_write = loop {
            match rx.recv_timeout(std::time::Duration::from_secs(10)) {
                Ok(AppEvent::WriteRecorded { session, .. }) => break session,
                Ok(_) => continue,
                Err(e) => panic!("no WriteRecorded: {e:?}"),
            }
        };
        assert_eq!(&*session_of_write, "s1");
    }
}
