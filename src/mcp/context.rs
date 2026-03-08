use std::path::PathBuf;

use color_eyre::eyre::{eyre, Result};
use rmcp::service::{Peer, RoleServer};

use ambits::parser::ParserRegistry;
use ambits::symbols::ProjectTree;

use crate::mcp::server::AmbitServer;

/// Resolved once per tool invocation from MCP Roots + server state.
/// All four tool handlers call `ProjectContext::resolve(…)` as their first step —
/// no tool re-implements root discovery, tree parsing, or session resolution.
pub struct ProjectContext {
    pub project_path: PathBuf,
    pub project_tree: ProjectTree,
    pub session_id:   Option<String>,
    pub log_dir:      Option<PathBuf>,
}

impl ProjectContext {
    /// Resolve context from MCP Roots + server state.
    ///
    /// Resolution order for project path:
    ///   1. `override_path` (tool parameter `project`)
    ///   2. First `file://` root returned by `peer.list_roots()`
    ///   3. Error with a clear message
    pub async fn resolve(
        server:        &AmbitServer,
        peer:          &Peer<RoleServer>,
        override_path: Option<PathBuf>,
    ) -> Result<Self> {
        // 1. Determine project path.
        let project_path = match override_path {
            Some(p) => p.canonicalize()?,
            None => {
                let result = peer.list_roots().await?;
                result
                    .roots
                    .into_iter()
                    .find_map(|r| {
                        // MCP root URIs are `file:///absolute/path`
                        r.uri
                            .strip_prefix("file://")
                            .map(PathBuf::from)
                    })
                    .ok_or_else(|| {
                        eyre!(
                            "no MCP roots configured; pass a `project` parameter explicitly"
                        )
                    })?
            }
        };

        // 2. Parse the symbol tree.
        // ParserRegistry is created here (not stored in AmbitServer) because
        // dyn LanguageParser is not Send+Sync.
        let registry = ParserRegistry::new();
        let project_tree = registry.scan_project(&project_path)?;

        // 3. Resolve log dir and latest session.
        let (log_dir, session_id) = server.latest_session_for(&project_path);

        Ok(Self { project_path, project_tree, session_id, log_dir })
    }
}
