use std::collections::HashSet;
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

        // Snapshot before the first tick: everything already on disk is, by
        // definition, not a session that started after us.
        let known_sessions = log_dir
            .as_ref()
            .map(|ld| existing_session_ids(ld))
            .unwrap_or_default();

        Ok(Self {
            current_session_id: session_id,
            known_sessions,
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

                    // Pre-populate from lines already written before this tick.
                    let new_files = self.ingester.session_log_files(ld, &latest);
                    for log_file in &new_files {
                        for event in self.ingester.parse_log_file_with_root(log_file, project_path) {
                            match event {
                                ambits::ingest::SessionEvent::ToolCall(tc) => app.process_agent_event(tc),
                                ambits::ingest::SessionEvent::Compacted { summary, timestamp, agent_id, metadata } => {
                                    app.process_compaction(summary, timestamp, agent_id, metadata);
                                }
                                ambits::ingest::SessionEvent::SessionCleared => app.reset_session(),
                            }
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
            for compaction in output.compactions {
                app.process_compaction(compaction.summary, compaction.timestamp, compaction.agent_id, compaction.metadata);
            }
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
    use ambits::ingest::{SessionEvent, TailerOutput};
    use ambits::symbols::ProjectTree;
    use std::sync::Mutex;

    /// Reports whatever session id the test most recently set, standing in for
    /// `find_latest_session`'s mtime ranking without touching the clock.
    struct MockIngester {
        latest: Mutex<Option<String>>,
    }

    impl MockIngester {
        fn new(latest: &str) -> Arc<Self> {
            Arc::new(Self {
                latest: Mutex::new(Some(latest.to_string())),
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
            Box::new(NoopTailer)
        }
    }

    struct NoopTailer;

    impl EventTailer for NoopTailer {
        fn add_file(&mut self, _path: PathBuf) {}
        fn read_new_events(&mut self) -> TailerOutput {
            TailerOutput {
                events: Vec::new(),
                compactions: Vec::new(),
                session_cleared: false,
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
        let mut app = App::new(tree, root.to_path_buf(), None);
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
            Some(current.to_string()),
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
}
