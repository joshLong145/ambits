pub mod context;
pub mod prompts;
pub mod resources;
pub mod server;
pub mod tools;

pub use server::AmbitServer;

use std::path::PathBuf;
use std::sync::Arc;

use color_eyre::eyre::{Result, WrapErr};
use rmcp::ServiceExt;

use ambits::ingest::{SessionIngester, ToolCallMapper};
use ambits::ingest::claude::ClaudeIngester;
use ambits::ingest::tool_config::ToolMappingConfig;

/// Start the ambit MCP server on stdio and block until the client disconnects.
///
/// This is the entry point for `ambits mcp [--log-dir <path>] [--tools-config <path>]`.
pub fn serve(
    log_dir:      Option<PathBuf>,
    tools_config: Option<PathBuf>,
) -> Result<()> {
    // Load tool-call mapping config (falls back to built-in if not supplied).
    let (config, _warnings) = ToolMappingConfig::resolve(tools_config.as_deref());

    // Coerce to trait objects and build the ingester.
    let mapper:   Arc<dyn ToolCallMapper>   = config;
    let ingester: Arc<dyn SessionIngester>  =
        Arc::new(ClaudeIngester::new(Arc::clone(&mapper)));

    let server = AmbitServer::new(ingester, log_dir);

    // Build a multi-thread tokio runtime, run the MCP server on stdio.
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .wrap_err("failed to build tokio runtime")?
        .block_on(async {
            let transport = rmcp::transport::stdio();
            let running = server
                .serve(transport)
                .await
                .wrap_err("MCP server failed to start")?;
            running
                .waiting()
                .await
                .wrap_err("MCP server task panicked")?;
            Ok(())
        })
}
