use std::path::{Path, PathBuf};
use std::sync::Arc;

use rmcp::ServerHandler;
use rmcp::handler::server::tool::ToolCallContext;
use rmcp::model::{
    AnnotateAble, CallToolRequestParam, CallToolResult, Content,
    GetPromptRequestParam, GetPromptResult, ListPromptsResult, ListResourcesResult, ListToolsResult,
    PaginatedRequestParam, Prompt, PromptArgument, RawResource, ReadResourceRequestParam,
    ReadResourceResult, ResourceContents, ServerCapabilities, ServerInfo, SubscribeRequestParam,
    UnsubscribeRequestParam,
};
use rmcp::service::{Peer, RequestContext, RoleServer};

use ambits::coverage;
use ambits::ingest::SessionIngester;

use crate::mcp::context::ProjectContext;
use crate::mcp::prompts::{get_coverage_analysis, COVERAGE_ANALYSIS};
use crate::mcp::resources::{parse_resource_path, resource_uri, WatcherRegistry};
use crate::mcp::tools::{
    coverage_response, file_coverage_to_json, is_uuid, mcp_err, project_tree_to_json,
};

/// Shared server state — one instance per `ambits mcp` process.
/// Must be `Clone + Send + Sync + 'static` to satisfy `ServerHandler`.
///
/// Note: `ParserRegistry` is NOT stored here because `dyn LanguageParser` lacks
/// `Send + Sync`. A new registry is created per call in `ProjectContext::resolve`
/// (construction is cheap — just vtable registration, no file I/O).
#[derive(Clone)]
pub struct AmbitServer {
    pub ingester:          Arc<dyn SessionIngester>,
    /// Optional log-dir override supplied via `--log-dir` CLI flag.
    pub log_dir_override:  Option<PathBuf>,
    /// Shared file-change watchers (one per project path, fanned out via broadcast).
    pub watcher_registry:  Arc<WatcherRegistry>,
    /// Populated by rmcp after the handshake via `set_peer`.
    peer: Option<Peer<RoleServer>>,
}

impl AmbitServer {
    pub fn new(
        ingester:         Arc<dyn SessionIngester>,
        log_dir_override: Option<PathBuf>,
    ) -> Self {
        Self {
            ingester,
            log_dir_override,
            watcher_registry: WatcherRegistry::new(),
            peer: None,
        }
    }

    /// Extract the active MCP peer, returning an error if not yet connected.
    fn peer(&self) -> Result<Peer<RoleServer>, rmcp::Error> {
        self.peer
            .clone()
            .ok_or_else(|| rmcp::Error::internal_error("no MCP peer connected", None))
    }

    /// Resolve the log directory for `project_path`, preferring the override flag.
    pub(crate) fn log_dir_for(&self, project_path: &Path) -> Option<PathBuf> {
        self.log_dir_override
            .clone()
            .or_else(|| self.ingester.log_dir_for_project(project_path))
    }

    /// Resolve the log directory and latest session ID for `project_path`.
    pub(crate) fn latest_session_for(
        &self,
        project_path: &Path,
    ) -> (Option<PathBuf>, Option<String>) {
        let log_dir = self.log_dir_for(project_path);
        let session_id = log_dir
            .as_ref()
            .and_then(|d| self.ingester.find_latest_session(d));
        (log_dir, session_id)
    }
}

// ── tool implementations ──────────────────────────────────────────────────────
//
// `#[rmcp::tool(tool_box)]` generates a private `AmbitServer::tool_box()`
// function that is used by `impl ServerHandler` below (same module → accessible).

#[rmcp::tool(tool_box)]
impl AmbitServer {
    /// Return the project symbol tree as JSON.
    #[rmcp::tool(description = "\
        Return the project symbol tree as JSON — all files, classes, functions, and other \
        symbols parsed from source code. Useful for understanding the project's architecture \
        at a glance.")]
    pub async fn symbol_tree(
        &self,
        #[tool(param)]
        #[schemars(description = "Absolute path to the project root \
            (omit to use the MCP client root)")]
        project: Option<String>,
    ) -> Result<CallToolResult, rmcp::Error> {
        let peer = self.peer()?;
        let ctx = ProjectContext::resolve(self, &peer, project.map(PathBuf::from))
            .await
            .map_err(mcp_err)?;

        let tree_json = project_tree_to_json(&ctx.project_tree);
        let out = serde_json::to_string_pretty(&tree_json).map_err(mcp_err)?;
        Ok(CallToolResult::success(vec![Content::text(out)]))
    }

    /// List all Claude Code sessions recorded for a project.
    #[rmcp::tool(description = "\
        List all Claude Code sessions recorded for a project, ordered by most-recent first. \
        Each entry includes the session UUID and last-modified Unix timestamp.")]
    pub async fn list_sessions(
        &self,
        #[tool(param)]
        #[schemars(description = "Absolute path to the project root \
            (omit to use the MCP client root)")]
        project: Option<String>,
    ) -> Result<CallToolResult, rmcp::Error> {
        let peer = self.peer()?;
        let ctx = ProjectContext::resolve(self, &peer, project.map(PathBuf::from))
            .await
            .map_err(mcp_err)?;

        let mut sessions: Vec<serde_json::Value> = match &ctx.log_dir {
            None => vec![],
            Some(log_dir) => std::fs::read_dir(log_dir)
                .map_err(mcp_err)?
                .flatten()
                .filter_map(|entry| {
                    let path = entry.path();
                    let name = path.file_name()?.to_str()?;
                    if !name.ends_with(".jsonl") {
                        return None;
                    }
                    let stem = name.strip_suffix(".jsonl")?;
                    if !is_uuid(stem) {
                        return None;
                    }
                    let meta = std::fs::metadata(&path).ok()?;
                    if meta.len() == 0 {
                        return None;
                    }
                    let modified_unix = meta
                        .modified()
                        .ok()
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    Some(serde_json::json!({
                        "session_id":    stem,
                        "modified_unix": modified_unix,
                    }))
                })
                .collect(),
        };
        sessions.sort_by(|a, b| {
            b["modified_unix"]
                .as_u64()
                .unwrap_or(0)
                .cmp(&a["modified_unix"].as_u64().unwrap_or(0))
        });

        let out = serde_json::to_string_pretty(&sessions).map_err(mcp_err)?;
        Ok(CallToolResult::success(vec![Content::text(out)]))
    }

    /// Generate a project-wide symbol-coverage report.
    #[rmcp::tool(description = "\
        Generate a symbol-level coverage report showing which parts of the project were \
        read by Claude during a session. Returns raw JSON metrics; if the MCP client \
        supports sampling, also adds an AI-generated interpretation.")]
    pub async fn coverage(
        &self,
        #[tool(param)]
        #[schemars(description = "Absolute path to the project root \
            (omit to use the MCP client root)")]
        project: Option<String>,
        #[tool(param)]
        #[schemars(description = "Session UUID to report on (omit for the latest session)")]
        session: Option<String>,
        #[tool(param)]
        #[schemars(description = "Agent ID prefix for filtering (omit to include all agents)")]
        agent: Option<String>,
    ) -> Result<CallToolResult, rmcp::Error> {
        let peer = self.peer()?;
        let mut ctx = ProjectContext::resolve(self, &peer, project.map(PathBuf::from))
            .await
            .map_err(mcp_err)?;

        if let Some(s) = session {
            ctx.session_id = Some(s);
        }

        let report = coverage::build_report(
            &ctx.project_path,
            &ctx.project_tree,
            &ctx.log_dir,
            &ctx.session_id,
            agent.as_deref(),
            &*self.ingester,
        );

        let raw_json = serde_json::json!({
            "session_id": report.session_id,
            "agent_id":   report.agent_id,
            "files": report.files.iter().map(file_coverage_to_json).collect::<Vec<_>>(),
        });

        coverage_response(&peer, raw_json).await
    }

    /// Get symbol-level coverage for a single source file.
    #[rmcp::tool(description = "\
        Get symbol-level coverage for a single source file — which symbols were seen, \
        how deeply, and by which agent. Returns raw JSON metrics plus an optional \
        AI-generated interpretation via sampling.")]
    pub async fn coverage_file(
        &self,
        #[tool(param)]
        #[schemars(description = "Relative path to the source file within the project")]
        file: String,
        #[tool(param)]
        #[schemars(description = "Absolute path to the project root \
            (omit to use the MCP client root)")]
        project: Option<String>,
        #[tool(param)]
        #[schemars(description = "Session UUID to report on (omit for the latest session)")]
        session: Option<String>,
        #[tool(param)]
        #[schemars(description = "Agent ID prefix for filtering (omit to include all agents)")]
        agent: Option<String>,
    ) -> Result<CallToolResult, rmcp::Error> {
        let peer = self.peer()?;
        let mut ctx = ProjectContext::resolve(self, &peer, project.map(PathBuf::from))
            .await
            .map_err(mcp_err)?;

        if let Some(s) = session {
            ctx.session_id = Some(s);
        }

        let report = coverage::build_report(
            &ctx.project_path,
            &ctx.project_tree,
            &ctx.log_dir,
            &ctx.session_id,
            agent.as_deref(),
            &*self.ingester,
        );

        let Some(fc) = report
            .files
            .iter()
            .find(|f| f.path == file || f.path.ends_with(&file))
        else {
            return Ok(CallToolResult::error(vec![Content::text(format!(
                "File not found in project tree: {file}"
            ))]));
        };

        coverage_response(&peer, file_coverage_to_json(fc)).await
    }
}

// ── ServerHandler ─────────────────────────────────────────────────────────────

/// `ServerHandler` implementation.
///
/// `call_tool` and `list_tools` delegate to `Self::tool_box()`, which is
/// generated by `#[rmcp::tool(tool_box)] impl AmbitServer { … }` in `tools.rs`.
/// We implement these two methods manually rather than using the
/// `#[rmcp::tool(tool_box)]` attribute on this block, because the macro's
/// parser rejects `fn set_peer(&mut self, …)`.
impl ServerHandler for AmbitServer {
    fn get_peer(&self) -> Option<Peer<RoleServer>> {
        self.peer.clone()
    }

    fn set_peer(&mut self, peer: Peer<RoleServer>) {
        self.peer = Some(peer);
    }

    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            server_info: rmcp::model::Implementation {
                name: "ambit".into(),
                version: env!("CARGO_PKG_VERSION").into(),
            },
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .enable_resources()
                .enable_resources_subscribe()
                .enable_prompts()
                .build(),
            instructions: Some(
                "Ambit analyses Claude Code session logs to report symbol-level coverage \
                 and expose the project symbol tree. Use `coverage` to see which parts of \
                 a codebase an AI agent has read, `symbol_tree` for architecture overview, \
                 and `list_sessions` to enumerate recorded sessions."
                    .into(),
            ),
            ..Default::default()
        }
    }

    async fn list_tools(
        &self,
        _request: PaginatedRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, rmcp::Error> {
        Ok(ListToolsResult {
            next_cursor: None,
            tools: Self::tool_box().list(),
        })
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParam,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, rmcp::Error> {
        let tcc = ToolCallContext::new(self, request, context);
        Self::tool_box().call(tcc).await
    }

    // ── resource handlers ─────────────────────────────────────────────────────

    async fn list_resources(
        &self,
        _request: PaginatedRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, rmcp::Error> {
        // Dynamically discover the project path from MCP roots; if unavailable,
        // return an empty list — the client can still read via explicit URI.
        let resources = if let Some(peer) = self.peer.clone() {
            match peer.list_roots().await {
                Ok(result) => result
                    .roots
                    .into_iter()
                    .filter_map(|r| {
                        let path = r.uri.strip_prefix("file://").map(std::path::PathBuf::from)?;
                        let uri = resource_uri(&path);
                        let name = path
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("project")
                            .to_string();
                        Some(
                            RawResource {
                                uri,
                                name: format!("{name} coverage"),
                                description: Some(
                                    "Live symbol-coverage report for this project".into(),
                                ),
                                mime_type: Some("application/json".into()),
                                size: None,
                            }
                            .no_annotation(),
                        )
                    })
                    .collect(),
                Err(_) => vec![],
            }
        } else {
            vec![]
        };
        Ok(ListResourcesResult { next_cursor: None, resources })
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResult, rmcp::Error> {
        let project_path = parse_resource_path(&request.uri)
            .ok_or_else(|| rmcp::Error::invalid_params("unrecognised resource URI", None))?;

        // Build the coverage report without sampling (read_resource is synchronous-style).
        let registry = ambits::parser::ParserRegistry::new();
        let project_tree = registry
            .scan_project(&project_path)
            .map_err(mcp_err)?;

        let (log_dir, session_id) = self.latest_session_for(&project_path);

        let report = coverage::build_report(
            &project_path,
            &project_tree,
            &log_dir,
            &session_id,
            None,
            &*self.ingester,
        );

        let json = serde_json::json!({
            "session_id": report.session_id,
            "files": report.files.iter().map(file_coverage_to_json).collect::<Vec<_>>(),
        });

        let text = serde_json::to_string_pretty(&json).map_err(mcp_err)?;
        Ok(ReadResourceResult {
            contents: vec![ResourceContents::text(text, request.uri)],
        })
    }

    async fn subscribe(
        &self,
        request: SubscribeRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<(), rmcp::Error> {
        let project_path = parse_resource_path(&request.uri)
            .ok_or_else(|| rmcp::Error::invalid_params("unrecognised resource URI", None))?;

        let log_dir = self
            .log_dir_for(&project_path)
            .unwrap_or_else(|| project_path.clone());

        let peer = self.peer()?;

        self.watcher_registry
            .subscribe(log_dir, project_path, peer);
        Ok(())
    }

    async fn unsubscribe(
        &self,
        request: UnsubscribeRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<(), rmcp::Error> {
        if let Some(project_path) = parse_resource_path(&request.uri) {
            self.watcher_registry.unsubscribe(&project_path);
        }
        Ok(())
    }

    // ── prompt handlers ───────────────────────────────────────────────────────

    async fn list_prompts(
        &self,
        _request: PaginatedRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListPromptsResult, rmcp::Error> {
        Ok(ListPromptsResult {
            next_cursor: None,
            prompts: vec![Prompt {
                name: COVERAGE_ANALYSIS.into(),
                description: Some(
                    "Interpret an ambit coverage report to identify knowledge gaps \
                     and well-understood areas in the codebase."
                        .into(),
                ),
                arguments: Some(vec![PromptArgument {
                    name: "session_id".into(),
                    description: Some(
                        "Session UUID to analyse (omit for the most recent session)".into(),
                    ),
                    required: Some(false),
                }]),
            }],
        })
    }

    async fn get_prompt(
        &self,
        request: GetPromptRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> Result<GetPromptResult, rmcp::Error> {
        if request.name != COVERAGE_ANALYSIS {
            return Err(rmcp::Error::invalid_params(
                format!("unknown prompt: {}", request.name),
                None,
            ));
        }
        let session_id = request
            .arguments
            .as_ref()
            .and_then(|m| m.get("session_id"))
            .and_then(|v| v.as_str());
        Ok(get_coverage_analysis(session_id))
    }
}
