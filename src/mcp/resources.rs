//! MCP resource support for ambit.
//!
//! Exposes one subscribable resource per project:
//!
//!   `ambit://coverage?path=<percent-encoded-canonical-project-path>`
//!
//! Clients subscribe to this URI and receive a `ResourceUpdated` notification
//! whenever the underlying Claude Code session log changes.  A single
//! `notify::RecommendedWatcher` per project path is shared across all
//! subscribers; `tokio::sync::broadcast` fans each file-change event out to
//! every concurrent MCP connection.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use notify::{Event as NotifyEvent, EventKind, RecursiveMode, Watcher};
use rmcp::model::ResourceUpdatedNotificationParam;
use rmcp::service::{Peer, RoleServer};
use tokio::sync::broadcast;

// ── URI helpers ───────────────────────────────────────────────────────────────

/// Canonical resource URI for a project's live coverage report.
pub fn resource_uri(project_path: &Path) -> String {
    let lossy = project_path.to_string_lossy();
    let encoded = urlencoding::encode(lossy.as_ref());
    format!("ambit://coverage?path={encoded}")
}

/// Parse a coverage resource URI back to a canonical project path.
/// Returns `None` for unrecognised URI schemes.
pub fn parse_resource_path(uri: &str) -> Option<PathBuf> {
    let encoded = uri.strip_prefix("ambit://coverage?path=")?;
    let decoded = urlencoding::decode(encoded).ok()?;
    Some(PathBuf::from(decoded.as_ref()))
}

// ── watcher registry ──────────────────────────────────────────────────────────

struct ProjectWatcher {
    /// Kept alive to maintain the OS-level watch.
    _watcher: notify::RecommendedWatcher,
    /// Broadcast sender; one receiver is created per subscriber.
    tx: broadcast::Sender<()>,
}

/// Shared registry — one instance per `ambits mcp` process, stored inside
/// `AmbitServer` behind an `Arc`.  All `AmbitServer` clones (one per MCP
/// connection) share the same `Arc<WatcherRegistry>`.
pub struct WatcherRegistry {
    inner: Mutex<HashMap<PathBuf, ProjectWatcher>>,
}

impl WatcherRegistry {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(HashMap::new()),
        })
    }

    /// Register a subscription for `project_path`.
    ///
    /// A background tokio task is spawned that forwards broadcast ticks to
    /// `peer.notify_resource_updated(uri)`.  The OS-level watcher for
    /// `log_dir` is created lazily on the first subscriber.
    pub fn subscribe(
        self: &Arc<Self>,
        log_dir: PathBuf,
        project_path: PathBuf,
        peer: Peer<RoleServer>,
    ) {
        let uri = resource_uri(&project_path);

        let rx = {
            let mut guard = self.inner.lock().unwrap();
            let entry = guard.entry(project_path).or_insert_with(|| {
                let (tx, _) = broadcast::channel::<()>(32);
                let tx_notify = tx.clone();

                let mut watcher = notify::recommended_watcher(
                    move |res: Result<NotifyEvent, notify::Error>| {
                        if let Ok(event) = res {
                            // Fire only on .jsonl create/modify events.
                            let relevant = matches!(
                                event.kind,
                                EventKind::Create(_) | EventKind::Modify(_)
                            ) && event.paths.iter().any(|p| {
                                p.extension().map_or(false, |e| e == "jsonl")
                            });
                            if relevant {
                                let _ = tx_notify.send(());
                            }
                        }
                    },
                )
                .expect("failed to create file watcher");

                // Start watching the log dir if it already exists.
                if log_dir.is_dir() {
                    let _ = watcher.watch(&log_dir, RecursiveMode::NonRecursive);
                }

                ProjectWatcher { _watcher: watcher, tx }
            });

            entry.tx.subscribe()
        }; // lock released

        // Spawn a task that forwards broadcast ticks to the subscriber's peer.
        tokio::spawn(async move {
            let mut rx = rx;
            loop {
                match rx.recv().await {
                    Ok(()) => {
                        let _ = peer
                            .notify_resource_updated(ResourceUpdatedNotificationParam {
                                uri: uri.clone(),
                            })
                            .await;
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => {
                        // Missed some ticks; send one notification to stay in sync.
                        let _ = peer
                            .notify_resource_updated(ResourceUpdatedNotificationParam {
                                uri: uri.clone(),
                            })
                            .await;
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        });
    }

    /// Remove the watcher for `project_path` once all receivers have gone.
    pub fn unsubscribe(&self, project_path: &Path) {
        let mut guard = self.inner.lock().unwrap();
        if let Some(entry) = guard.get(project_path) {
            if entry.tx.receiver_count() == 0 {
                guard.remove(project_path);
            }
        }
    }
}
