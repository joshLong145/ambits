# ambit MCP Server — Implementation Workflow

**Source plan**: `docs/workflows/mcp-server-plan.md`
**Strategy**: Systematic, bottom-up (foundations before features)
**Checkpoints**: `cargo check --features mcp` after every phase

---

## Dependency Map

```
Phase 1: Foundation (Cargo + module scaffold)
    │
    ▼
Phase 2: Core Server (AmbitServer + ProjectContext)
    │
    ▼
Phase 3: Tools (4 × #[tool], simplest first)
    │
    ├──▶ Phase 4a: Resources (shared watcher + subscriptions)  ─┐
    │                                                            ├──▶ Phase 5: Integration (wire main.rs + verify)
    └──▶ Phase 4b: Prompts + Skill update (independent)        ─┘
```

---

## Phase 1 — Foundation

**Goal**: Get the `mcp` feature compiling with an empty server stub before any logic is
written. All subsequent phases build on a green `cargo check`.

### Task 1.1 — Cargo.toml: add `mcp` feature + optional deps

**File**: `Cargo.toml`

Add the feature gate and three optional dependencies:

```toml
[features]
mcp = ["dep:rmcp", "dep:tokio", "dep:schemars"]

[dependencies]
rmcp     = { version = "0.17", features = ["server", "transport-io", "macros"], optional = true }
tokio    = { version = "1",    features = ["rt-multi-thread", "macros"],        optional = true }
schemars = { version = "0.8",                                                   optional = true }
```

Check `notify` (already a dependency for `tui.rs`) is available — `resources.rs` will reuse it.

**Checkpoint**: `cargo check` (no features) still passes — no regressions.

---

### Task 1.2 — Create `src/mcp/` module skeleton

Create six empty files with the minimal content needed to compile:

| File | Minimal content |
|---|---|
| `src/mcp/mod.rs` | `pub mod server; pub mod context; pub mod tools; pub mod resources; pub mod prompts;` |
| `src/mcp/server.rs` | Empty `pub struct AmbitServer;` stub |
| `src/mcp/context.rs` | Empty `pub struct ProjectContext;` stub |
| `src/mcp/tools.rs` | `// tools placeholder` |
| `src/mcp/resources.rs` | `// resources placeholder` |
| `src/mcp/prompts.rs` | `// prompts placeholder` |

Gate the module in `src/lib.rs` (or `src/main.rs`) behind the feature:

```rust
#[cfg(feature = "mcp")]
pub mod mcp;
```

**Checkpoint**: `cargo check --features mcp` — all stubs compile.

---

### Task 1.3 — `src/main.rs`: add `Commands::Mcp` stub

Extend the existing `Commands` enum:

```rust
#[cfg(feature = "mcp")]
Commands::Mcp { log_dir, tools_config } => {
    mcp::serve(log_dir, tools_config)
}
```

Add the CLI variant to `Commands`:

```rust
#[cfg(feature = "mcp")]
/// Start the ambit MCP server on stdio
Mcp {
    #[arg(long)] log_dir:      Option<PathBuf>,
    #[arg(long)] tools_config: Option<PathBuf>,
},
```

`mcp::serve` is a placeholder `fn serve(...) -> Result<()> { todo!() }` until Phase 5.

**Checkpoint**: `cargo check --features mcp`.

---

## Phase 2 — Core Server

**Goal**: Implement `AmbitServer` (holds shared state) and `ProjectContext` (per-call
resolver). These are the two load-bearing types that all tools depend on.

### Task 2.1 — `src/mcp/server.rs`: `AmbitServer`

```rust
#[cfg(feature = "mcp")]
pub struct AmbitServer {
    pub ingester:        Arc<dyn SessionIngester>,
    pub parser_registry: Arc<ParserRegistry>,
}

impl AmbitServer {
    pub fn new(
        ingester:        Arc<dyn SessionIngester>,
        parser_registry: Arc<ParserRegistry>,
    ) -> Self {
        Self { ingester, parser_registry }
    }
}
```

Derive / implement `rmcp::ServerHandler` with empty stubs for now — the `#[tool_box]` macro
will fill these in Phase 3.

**Dependencies**: `SessionIngester` (from `src/ingest/mod.rs`), `ParserRegistry` (from
`src/parser/mod.rs`). Both are already `Arc`-safe.

---

### Task 2.2 — `src/mcp/context.rs`: `ProjectContext`

Implement the full resolver as specified in the plan:

```rust
pub struct ProjectContext {
    pub project_path: PathBuf,
    pub project_tree: ProjectTree,
    pub session_id:   Option<String>,
    pub log_dir:      Option<PathBuf>,
}

impl ProjectContext {
    pub async fn resolve(
        server:        &AmbitServer,
        peer:          &dyn PeerProxy,
        override_path: Option<PathBuf>,
    ) -> Result<Self> {
        // 1. project path: override > first MCP root > error
        // 2. parse symbol tree via server.parser_registry
        // 3. resolve log_dir + session_id via server.ingester
    }
}
```

**Error cases to handle explicitly**:
- `list_roots()` returns empty → return `Err` with message: _"no MCP roots configured; pass
  `project` parameter explicitly"_
- `canonicalize()` fails (path doesn't exist) → propagate with context

**Dependencies**: Task 2.1 (`AmbitServer`), rmcp `PeerProxy` trait.

**Checkpoint**: `cargo check --features mcp`.

---

## Phase 3 — Tools

**Goal**: Implement all four `#[tool]` handlers in `src/mcp/tools.rs`. Build simplest → most
complex so each tool can be verified before the next is started.

### Task 3.1 — `symbol_tree` tool (simplest)

**Input**: `project: Option<String>`
**Logic**:
1. `ProjectContext::resolve(…)`
2. Serialize `pctx.project_tree` to JSON (`serde_json::to_string_pretty`)
3. Return as `CallToolResult::text(json)`

**No sampling.** This tool validates that `ProjectContext` resolves correctly end-to-end.

---

### Task 3.2 — `list_sessions` tool

**Input**: `project: Option<String>`
**Output schema**: `[{ session_id: String, created_at: String, event_count: u32 }]`

**Logic**:
1. `ProjectContext::resolve(…)`
2. If `pctx.log_dir` is `None` → return empty list (not an error)
3. Scan log dir for `*.jsonl` files; for each:
   - Parse session ID from filename
   - Read file mtime as `created_at`
   - Count newlines as proxy for `event_count`
4. Return JSON array sorted by `created_at` descending

**No sampling.**

---

### Task 3.3 — `coverage` tool (primary tool, two-path return)

**Input**: `project: Option<String>`
**Output**:
- With sampling: `{ interpretation: String, raw_report: CoverageReport }`
- Without sampling: `{ raw_report: CoverageReport }`

**Logic**:
1. `ProjectContext::resolve(…)`
2. Build `CoverageReport` via `coverage::build_report(&pctx.project_path, &pctx.project_tree, &pctx.log_dir, &pctx.session_id, &*server.ingester)`
3. Serialize report to JSON string `report_json`
4. Attempt sampling with 30 s timeout:
   ```rust
   let interpretation = tokio::time::timeout(
       Duration::from_secs(30),
       ctx.peer.create_message(CreateMessageRequest {
           messages: vec![SamplingMessage {
               role: Role::User,
               content: Content::text(format!("Coverage data:\n{report_json}")),
           }],
           system_prompt: Some(include_str!("../../coverage-guide.md").into()),
           include_context: ContextInclusion::ThisServer,
           max_tokens: 2048,
           ..Default::default()
       }),
   ).await.ok().and_then(|r| r.ok());
   ```
5. Return `{ interpretation, raw_report }` — `interpretation` is `null` if sampling failed.

**Note**: `coverage::build_report` may need a small refactor to be callable without the
full CLI argument set. Audit `src/coverage.rs` before implementing; extract a
`build_report(…) -> CoverageReport` free function if it doesn't already exist as one.

---

### Task 3.4 — `coverage_file` tool

**Input**: `file_path: String`, `project: Option<String>`, `interpret: Option<bool>`
**Output**: per-symbol coverage breakdown for the requested file, optionally with sampling.

**Logic**:
1. `ProjectContext::resolve(…)`
2. Find the `ParsedFile` in `pctx.project_tree` matching `file_path`
3. Build per-symbol coverage for that file only (reuse ledger lookup logic from
   `coverage::run_report`)
4. If `interpret` defaults to `true` AND sampling available: same pattern as Task 3.3 but
   scoped to the single file
5. Degrade to raw data on timeout or unavailability

**Checkpoint after all 4 tools**: `cargo check --features mcp` + manual smoke test via
`echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' | ambits mcp`

---

## Phase 4a — Resources

**Goal**: Implement the subscribable `ambit://coverage` resource with a shared watcher.

### Task 4a.1 — `resource_uri` helper

```rust
pub fn resource_uri(project_path: &Path) -> String {
    let encoded = percent_encoding::utf8_percent_encode(
        project_path.to_string_lossy().as_ref(),
        percent_encoding::NON_ALPHANUMERIC,
    );
    format!("ambit://coverage?path={encoded}")
}
```

Add `percent-encoding` to `[dependencies]` (already widely used in the Rust ecosystem;
gate it under `mcp` feature if preferred).

---

### Task 4a.2 — `ProjectWatcher` + shared watcher registry

```rust
use tokio::sync::broadcast;
use std::collections::HashMap;
use std::sync::Mutex;

struct ProjectWatcher {
    _watcher: notify::RecommendedWatcher,
    tx:       broadcast::Sender<()>,
}

// Owned by AmbitServer or a module-level static
pub struct WatcherRegistry {
    inner: Mutex<HashMap<PathBuf, ProjectWatcher>>,
}
```

Implement `subscribe(project_path, log_dir)` → `broadcast::Receiver<()>`:
1. Lock registry
2. If entry exists → return `tx.subscribe()`
3. Else → create watcher on `log_dir` (NonRecursive, `.jsonl` filter), create broadcast
   channel, insert, return `tx.subscribe()`

Implement `maybe_drop(project_path)`:
1. Lock registry
2. If `tx.receiver_count() == 0` → remove entry (drops watcher)

---

### Task 4a.3 — Wire subscription into `ServerHandler`

In `server.rs` (or `resources.rs`), implement the `subscribe_resource` / `unsubscribe_resource`
hooks provided by `rmcp::ServerHandler`. On subscription:

```rust
let mut rx = registry.subscribe(&pctx.project_path, &pctx.log_dir)?;
tokio::spawn(async move {
    while rx.recv().await.is_ok() {
        peer.notify_resource_updated(resource_uri(&pctx.project_path)).await;
    }
    registry.maybe_drop(&pctx.project_path);
});
```

**Checkpoint**: `cargo check --features mcp`.

---

## Phase 4b — Prompts + Skill Update

*Parallel with Phase 4a — no shared dependencies.*

### Task 4b.1 — `src/mcp/prompts.rs`: `coverage_analysis` prompt

```rust
#[prompt]
/// Analyze ambit coverage data and identify gaps
fn coverage_analysis() -> String {
    // Combines coverage-guide.md + examples.md as a structured prompt template
    format!(
        "{}\n\n## Examples\n{}",
        include_str!("../../coverage-guide.md"),
        include_str!("../../skills/ambit/examples.md"),
    )
}
```

Register the prompt on `AmbitServer` via the `#[tool_box]` / prompt registration mechanism
in rmcp.

---

### Task 4b.2 — `skills/ambit/SKILL.md` update

Add a section at the top of the skill file:

```markdown
## Preferred Integration: MCP Server

For Claude Code 1.x+, run the ambit MCP server instead of the Bash skill:

    ambits mcp

Then add to your Claude Code config:
    { "mcpServers": { "ambit": { "command": "ambits", "args": ["mcp"] } } }

The MCP server provides: live coverage updates, structured JSON output, and
Claude-interpreted coverage summaries via Sampling.

The Bash commands below remain available as a fallback.
```

---

## Phase 5 — Integration & Verification

**Goal**: Wire `mcp::serve` in `main.rs`, run end-to-end tests, confirm the full feature
compiles and behaves correctly.

### Task 5.1 — Implement `mcp::serve` in `src/mcp/mod.rs`

Replace the `todo!()` stub:

```rust
pub fn serve(
    log_dir:      Option<PathBuf>,
    tools_config: Option<PathBuf>,
) -> color_eyre::eyre::Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let (tool_config, _warnings) = ToolMappingConfig::resolve(tools_config.as_deref());
        let mapper:   Arc<dyn ToolCallMapper>  = tool_config;
        let ingester: Arc<dyn SessionIngester> =
            Arc::new(ingest::claude::ClaudeIngester::new(Arc::clone(&mapper)));
        let registry = Arc::new(ParserRegistry::new());
        let server   = AmbitServer::new(ingester, registry);

        rmcp::ServiceExt::serve(server, rmcp::transport::stdio()).await?;
        Ok(())
    })
}
```

---

### Task 5.2 — Verification checklist

```bash
# 1. Clean compile
cargo check --features mcp

# 2. All existing tests still pass (no regressions)
cargo test

# 3. MCP feature tests pass
cargo test --features mcp

# 4. Bench still compiles (imports free functions, not mcp module)
cargo bench --no-run

# 5. Manual smoke test: list tools
echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' | ambits mcp

# 6. Manual smoke test: call symbol_tree
echo '{"jsonrpc":"2.0","method":"tools/call","id":2,"params":{"name":"symbol_tree","arguments":{}}}' \
  | ambits mcp

# 7. Register with Claude Code
# Add to ~/.claude/claude.json mcpServers block:
# { "ambit": { "command": "ambits", "args": ["mcp"] } }
# Then verify /coverage invokes the coverage tool
```

---

## Execution Order Summary

```
1.1  Cargo.toml — mcp feature + deps
1.2  src/mcp/ skeleton (6 empty files)
1.3  src/main.rs — Commands::Mcp stub
     ✓ cargo check --features mcp

2.1  src/mcp/server.rs — AmbitServer
2.2  src/mcp/context.rs — ProjectContext resolver
     ✓ cargo check --features mcp

3.1  tools.rs — symbol_tree
3.2  tools.rs — list_sessions
3.3  tools.rs — coverage (two-path + sampling)
3.4  tools.rs — coverage_file
     ✓ cargo check --features mcp

4a.1  resources.rs — resource_uri helper
4a.2  resources.rs — WatcherRegistry
4a.3  server.rs    — wire subscription hooks

4b.1  prompts.rs   — coverage_analysis prompt   ← parallel with 4a
4b.2  SKILL.md     — MCP server section          ← parallel with 4a

     ✓ cargo check --features mcp

5.1  mod.rs — mcp::serve (replaces todo!())
5.2  Full verification checklist
     ✓ cargo test --features mcp
     ✓ Manual smoke tests
```

---

## Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| `coverage::build_report` not extractable as free fn | Medium | Audit `src/coverage.rs` at Task 3.3 start; refactor if needed |
| rmcp `PeerProxy` trait surface differs from plan pseudocode | Low | Consult rmcp 0.17 docs at Task 2.2; adjust method names |
| `notify` watcher behaviour differs macOS vs Linux | Low | Use `RecursiveMode::NonRecursive` + `.jsonl` filter consistently |
| Sampling unavailable in test environment | Expected | Two-path return design handles this; tests assert on `raw_report` only |
| `percent-encoding` crate version conflict | Low | Prefer `urlencoding` (zero-dep) if conflict arises |

---

## Deferred (Out of Scope for This Workflow)

- `--format claude|cursor|windsurf` flag for ingester selection (follow-on)
- Parse-result caching in `ProjectContext` (follow-on, seam is ready in `context.rs`)
- `coverage-guide.md` single-source consolidation (follow-on)
