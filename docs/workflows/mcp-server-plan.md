# ambit MCP Server — Implementation Brief

## Goal

Replace the current Claude Code skill (4 markdown files + `Bash(ambits *)` calls) with a
first-class MCP server binary that exposes coverage intelligence directly to the LLM context.
The server uses **rmcp** (official Rust MCP SDK), **Sampling** to ask Claude to interpret
raw coverage data, **Roots** to discover the project path automatically, and **Resources** for
live subscribable coverage updates.

---

## New CLI Subcommand

```
ambits mcp [--log-dir <path>] [--tools-config <path>]
```

Starts the MCP server on stdio. Added to the existing `Commands` enum in `src/main.rs`
(same pattern as `Commands::Skill`). No `--project` needed — path comes from MCP Roots.

---

## Cargo Changes

Add behind an `mcp` feature flag (keeps CLI binary lean for users who don't need MCP):

```toml
[features]
mcp = ["dep:rmcp", "dep:tokio", "dep:schemars"]

[dependencies]
rmcp    = { version = "0.17", features = ["server", "transport-io", "macros"], optional = true }
tokio   = { version = "1",    features = ["rt-multi-thread", "macros"],        optional = true }
schemars = { version = "0.8",                                                  optional = true }
```

Build with `cargo build --features mcp` (or default the feature on in `Cargo.toml`).

---

## New Module: `src/mcp/`

```
src/mcp/
  mod.rs        — re-exports, starts stdio server
  server.rs     — AmbitServer struct + ServerHandler impl
  context.rs    — ProjectContext: shared per-call resolver (NEW)
  tools.rs      — #[tool] implementations (4 tools)
  resources.rs  — ambit://coverage/{project} live resource
  prompts.rs    — #[prompt] coverage analysis template
```

`AmbitServer` holds:
- `ingester: Arc<dyn SessionIngester>`
- `parser_registry: Arc<ParserRegistry>`

---

## Shared Per-Call Resolver: `ProjectContext` (`src/mcp/context.rs`)

All four tools share an identical bootstrap sequence: resolve the project path from MCP
Roots, parse the symbol tree, find the latest session. Rather than duplicate these steps in
each tool handler, a single `ProjectContext` struct resolves them once per call.

```rust
/// Resolved once per tool invocation; passed into every tool handler.
pub struct ProjectContext {
    pub project_path: PathBuf,
    pub project_tree: ProjectTree,
    pub session_id:   Option<String>,
    pub log_dir:      Option<PathBuf>,
}

impl ProjectContext {
    /// Resolve from MCP Roots + server state.
    /// Accepts an optional `project` override (for clients that pass it explicitly).
    pub async fn resolve(
        server:   &AmbitServer,
        peer:     &dyn PeerProxy,
        override_path: Option<PathBuf>,
    ) -> Result<Self> {
        // 1. Determine project path: explicit override > first MCP root > error.
        let project_path = match override_path {
            Some(p) => p.canonicalize()?,
            None => {
                let roots = peer.list_roots().await?;
                roots
                    .into_iter()
                    .find_map(|r| r.uri.to_file_path().ok())
                    .ok_or_else(|| anyhow!("no MCP roots configured and --project not supplied"))?
            }
        };

        // 2. Parse symbol tree.
        let project_tree = server.parser_registry.scan_project(&project_path)?;

        // 3. Resolve log dir + latest session.
        let log_dir   = server.ingester.log_dir_for_project(&project_path);
        let session_id = log_dir.as_ref()
            .and_then(|d| server.ingester.find_latest_session(d));

        Ok(Self { project_path, project_tree, session_id, log_dir })
    }
}
```

**Every tool handler signature becomes:**
```rust
async fn coverage(
    &self,
    ctx: RequestContext<RoleServer>,
    params: CoverageParams,          // includes optional `project: Option<String>`
) -> Result<CallToolResult> {
    let pctx = ProjectContext::resolve(self, &ctx.peer, params.project.map(PathBuf::from)).await?;
    // tool-specific logic only below this line
    …
}
```

No tool re-implements root discovery, tree parsing, or session resolution.

---

## MCP Tools (4)

### `coverage`
1. `ProjectContext::resolve(…)` — project path, tree, session.
2. Build `CoverageReport` (reuse `coverage::build_report`).
3. Serialize report to JSON.
4. Send **Sampling** request (with `tokio::time::timeout(30s, …)`):
   ```rust
   context.peer.create_message(CreateMessageRequest {
       messages: vec![SamplingMessage {
           role: Role::User,
           content: Content::text(format!("Coverage data:\n{report_json}")),
       }],
       system_prompt: Some(include_str!("../../coverage-guide.md").into()),
       include_context: ContextInclusion::ThisServer,
       max_tokens: 2048,
       ..Default::default()
   }).await
   ```
5. **Two-path return**: if sampling succeeds → `{ interpretation, raw_report }`; if sampling
   unavailable or timed out → `{ raw_report }` only. Never fails due to sampling absence.

### `list_sessions`
1. `ProjectContext::resolve(…)`.
2. Scan `log_dir` for all session files; return structured list:
   `[{ session_id, created_at, event_count }]`.
No sampling.

### `symbol_tree`
1. `ProjectContext::resolve(…)`.
2. Serialize `project_tree` as JSON.
Returns raw symbol hierarchy. No sampling.

### `coverage_file`
Input: `file_path: String`, `interpret: Option<bool>`.
1. `ProjectContext::resolve(…)`.
2. Build per-symbol coverage for the requested file.
3. If `interpret` is `true` (default when client supports sampling): attempt sampling with
   30 s timeout; degrade to raw data on failure.
Returns per-symbol coverage breakdown.

---

## MCP Resources

### URI Scheme

Resources use the canonical project path, percent-encoded, to avoid collision between
same-named projects in different directories:

```
ambit://coverage?path=%2FUsers%2Fjoshlong%2Fcode%2Fmyproject
```

`resources.rs` exposes a `resource_uri(project_path: &Path) -> String` helper that
canonicalizes the path and URL-encodes it. All subscribe/unsubscribe/notify calls go through
this helper — no ad-hoc URI construction anywhere else.

### Shared Watcher Model

A single `notify::RecommendedWatcher` per project path is shared across all subscribers.
`resources.rs` owns a `HashMap<PathBuf, ProjectWatcher>` behind a `Mutex`:

```rust
struct ProjectWatcher {
    _watcher: notify::RecommendedWatcher,   // kept alive
    tx:       broadcast::Sender<()>,         // fan-out to all subscribers
}
```

When a client subscribes:
1. If an entry already exists for `project_path` → clone `tx` and subscribe to the broadcast.
2. If no entry exists → spawn a new `notify::RecommendedWatcher` watching `log_dir` (Non-Recursive),
   create a `broadcast::channel`, insert `ProjectWatcher`, then subscribe.

When the watcher fires a `.jsonl` Create/Modify event:
- `tx.send(())` — all active subscribers receive the tick.
- Each subscriber calls `notify_resource_updated(uri)` on its peer.

When a client unsubscribes or disconnects:
- Drop the subscriber's `broadcast::Receiver`.
- If `tx.receiver_count() == 0`, remove the entry from the map (drops the watcher).

---

## MCP Prompts

One prompt template (`coverage_analysis`) that bakes the `coverage-guide.md` + `examples.md`
content into a reusable message structure. Replaces the current skill's static markdown files.

---

## Skill Update

`skills/ambit/SKILL.md` — add a note pointing users to `ambits mcp` as the preferred
integration for Claude Code (richer context, sampling, live updates), with the existing
Bash-based commands as a fallback for environments without MCP support.

---

## Files Created / Modified

| File | Change |
|---|---|
| `src/mcp/mod.rs` | New — server entry point |
| `src/mcp/server.rs` | New — `AmbitServer` + `ServerHandler` |
| `src/mcp/context.rs` | New — `ProjectContext` shared resolver |
| `src/mcp/tools.rs` | New — 4 `#[tool]` implementations |
| `src/mcp/resources.rs` | New — coverage resource + subscription |
| `src/mcp/prompts.rs` | New — `#[prompt]` coverage analysis template |
| `src/main.rs` | Add `Commands::Mcp` subcommand + feature gate |
| `Cargo.toml` | Add `mcp` feature + optional deps |
| `skills/ambit/SKILL.md` | Update to reference MCP server |

---

## Verification

```bash
cargo check --features mcp
cargo test  --features mcp
# In Claude Code: /mcp add stdio -- ambits mcp
# Then: /coverage (invokes coverage tool via MCP)
```
