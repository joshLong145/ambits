# Spike: Compaction Awareness

**Status**: Ready for implementation  
**Branch target**: feature/compaction-awareness  
**References**: https://platform.claude.com/docs/en/build-with-claude/compaction

---

## Background

Claude Code uses server-side context compaction (`compact-2026-01-12` beta) to summarize older conversation history when approaching token limits. When compaction fires, Claude produces a `compaction` block in its assistant response — a prose summary of the context it is dropping. This event appears in the JSONL session logs ambit already reads.

Currently ambit ignores compaction completely (`parse_compact_command_returns_ignored` test). This spike adds full awareness: detect compaction boundaries, snapshot ledger state at each boundary, surface the data in the TUI and coverage reports.

---

## JSONL Log Signal

A compaction event appears as an `assistant` record whose `message.content` array contains a block with `"type": "compaction"`:

```json
{
  "type": "assistant",
  "timestamp": "2026-05-11T14:23:00Z",
  "sessionId": "abc123",
  "message": {
    "content": [
      {
        "type": "compaction",
        "content": "Summary of the conversation: The user is building a CLI parser..."
      },
      {
        "type": "text",
        "text": "Based on our conversation so far..."
      }
    ]
  }
}
```

A `/compact` user command (manual trigger) already appears in logs as a user message and is currently ignored — leave it ignored, only the resulting `compaction` block in the assistant response matters.

The `agent-acompact-*` JSONL files are already excluded from `session_log_files` — do not change that.

---

## Architecture Overview

### Core principle

ambit's ledger does not forget when Claude does. Compaction affects Claude's context window, not our recorded tool calls. The meaningful data is:

- **Pre-compaction slice**: what files and symbols the agent had accessed up to the boundary
- **Summary text**: what Claude distilled that history into
- **Post-compaction slice**: what has been accessed since (derived live from current ledger state)

This naturally extends to N compactions by keeping an ordered `Vec<CompactionEvent>`.

---

## Implementation Plan

### Step 1 — Extend `ParsedLine` (`src/ingest/claude.rs`)

Add a `Compacted` arm to the existing enum:

```rust
pub enum ParsedLine {
    Events(Vec<AgentToolCall>),
    Compacted { summary: String, timestamp: String },
    SessionCleared,
    Ignored,
}
```

In `parse_jsonl_line`, after confirming `msg_type == "assistant"`, scan the content array for a block where `block["type"] == "compaction"` and extract `block["content"].as_str()` as the summary. Return `ParsedLine::Compacted` immediately when found (before the tool-use loop, since compaction and tool-use blocks are mutually exclusive in practice).

```rust
// Inside parse_jsonl_line, after the assistant check:
for block in content {
    if block.get("type").and_then(|v| v.as_str()) == Some("compaction") {
        let summary = block
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        return ParsedLine::Compacted { summary, timestamp: timestamp_str };
    }
}
// existing tool_use loop follows...
```

Add a unit test `parse_compaction_block_returns_compacted` alongside the existing `parse_compact_command_returns_ignored`.

---

### Step 2 — Introduce `SessionEvent` (`src/ingest/mod.rs`)

Replace `Vec<AgentToolCall>` return types with a richer ordered enum:

```rust
pub enum SessionEvent {
    ToolCall(AgentToolCall),
    Compacted { summary: String, timestamp: String, agent_id: Arc<str> },
    SessionCleared,
}
```

Update `SessionIngester` trait:

```rust
pub trait SessionIngester: Send + Sync {
    // existing methods unchanged...
    fn parse_log_file(&self, path: &Path) -> Vec<SessionEvent>;
    fn parse_log_file_with_root(&self, path: &Path, project_root: &Path) -> Vec<SessionEvent>;
}
```

Update `parse_log_file_with_mapper` in `claude.rs` to emit `SessionEvent::Compacted` when `ParsedLine::Compacted` is returned, preserving log order.

Update all callers (`main.rs`, `tui.rs`, `coverage.rs`) to match on `SessionEvent` instead of iterating `AgentToolCall` directly.

---

### Step 3 — New data types (`src/ingest/mod.rs` or `src/compaction.rs`)

```rust
/// Point-in-time ledger snapshot captured at a compaction boundary.
pub struct LedgerSnapshot {
    pub tool_call_count: usize,
    pub files_accessed: BTreeSet<PathBuf>,  // relative paths, sorted
    pub symbols_seen: usize,
    pub seen_percent: f64,
}

/// A detected compaction event with summary and pre-compaction state.
pub struct CompactionEvent {
    pub sequence: u32,           // 1-based: 1st, 2nd, 3rd compaction
    pub timestamp: String,
    pub agent_id: Arc<str>,
    pub summary: String,
    pub ledger_before: LedgerSnapshot,
}
```

---

### Step 4 — `AppEvent` extension (`src/events.rs`)

```rust
pub enum AppEvent {
    Key(KeyEvent),
    Mouse(MouseEvent),
    FileChanged(PathBuf),
    AgentEvent(AgentToolCall),
    SessionCleared,
    Compacted(CompactionEvent),  // ← new
    Tick,
}
```

---

### Step 5 — `App` state and processing (`src/app.rs`)

New fields on `App`:

```rust
pub struct App {
    // existing fields unchanged...
    pub compaction_history: Vec<CompactionEvent>,
    pub compaction_call_count: usize,   // running tool call counter
}
```

New method. Derives the file list from the project tree + ledger at snapshot time — no ledger schema changes needed:

```rust
pub fn process_compaction(
    &mut self,
    summary: String,
    timestamp: String,
    agent_id: Arc<str>,
) {
    let files_before: BTreeSet<PathBuf> = self.project_tree.files.iter()
        .filter(|f| {
            f.symbols.iter().any(|sym| self.ledger.depth_of(&sym.id).is_seen())
        })
        .map(|f| f.file_path.clone())
        .collect();

    let total = self.project_tree.total_symbols();
    let seen = self.ledger.total_seen();

    let snapshot = LedgerSnapshot {
        tool_call_count: self.compaction_call_count,
        files_accessed: files_before,
        symbols_seen: seen,
        seen_percent: if total > 0 {
            seen as f64 / total as f64 * 100.0
        } else { 0.0 },
    };

    self.compaction_history.push(CompactionEvent {
        sequence: self.compaction_history.len() as u32 + 1,
        timestamp,
        agent_id,
        summary,
        ledger_before: snapshot,
    });
}
```

Increment `compaction_call_count` in `process_agent_event`:

```rust
pub fn process_agent_event(&mut self, event: AgentToolCall) {
    self.compaction_call_count += 1;
    // existing logic...
}
```

Update `reset_session` to clear both new fields.

---

### Step 6 — Event dispatch (`main.rs`, `tui.rs`)

Update the pre-population loop in `main.rs`:

```rust
for event in ingester.parse_log_file_with_root(log_file, &project_path) {
    match event {
        SessionEvent::ToolCall(tc)               => app.process_agent_event(tc),
        SessionEvent::Compacted { summary, timestamp, agent_id } => {
            app.process_compaction(summary, timestamp, agent_id);
        }
        SessionEvent::SessionCleared             => app.reset_session(),
    }
}
```

Same pattern in `tui.rs` for live-session pre-population on session switch.

Wire `AppEvent::Compacted(ev)` in the TUI event loop to call `app.process_compaction(...)`.

---

### Step 7 — TUI: activity feed marker (`src/ui/activity.rs`)

Render a distinct separator line when a compaction is present in the activity stream. Since the activity feed is currently event-list-based, the cleanest approach is to record compaction events as a synthetic entry in the activity list with a special render style:

```
─── compaction #1 · 31.2% seen · 12 files ───────────────────
```

Style: `Color::Yellow` or `Color::Magenta`, full-width rule with summary info inline.

---

### Step 8 — TUI: compaction diff overlay (`src/ui/compaction.rs`)

A new toggle panel activated by pressing `C` (capital). Renders the last compaction by default; `[` / `]` step through `compaction_history`.

Layout:

```
┌─ Compaction #1 — 2026-05-11 14:23 ──────────────────────────┐
│ Summary                                                       │
│   "The user is building a CLI parser. Files modified:        │
│    src/parser/mod.rs, src/cli.rs. Next step: flag validation"│
├───────────────────────────────────────────────────────────────┤
│ Before (47 calls · 12 files · 31.2% seen)                    │
│   src/cli.rs                    FullBody                      │
│   src/parser/mod.rs             Signature                     │
│   src/parser/rust.rs            Overview                      │
│   src/symbols/mod.rs            NameOnly                      │
│   ... (8 more)                                                │
├───────────────────────────────────────────────────────────────┤
│ After  (current · 8 calls since compaction)                   │
│   src/main.rs                   NameOnly                      │
└───────────────────────────────────────────────────────────────┘
```

"After" side is derived live from the current ledger minus the pre-compaction snapshot — files that appear in the current ledger but were not in `ledger_before.files_accessed`.

Add `FocusPanel::CompactionOverlay` variant or a boolean `show_compaction_overlay: bool` on `App` — prefer the boolean to avoid changing focus cycle logic.

---

### Step 9 — Stats panel (`src/ui/stats.rs`)

Append to the stats panel when `!app.compaction_history.is_empty()`:

```
  Compactions: 2
  Last: 2026-05-11 15:41  58.0% seen before
```

---

## Coverage Report Extension (`src/coverage.rs`)

### New type

```rust
#[derive(Debug, Clone, Serialize)]
pub struct CompactionSummary {
    pub sequence: u32,
    pub timestamp: String,
    pub summary: String,
    pub tool_calls_before: usize,
    pub files_before: Vec<String>,       // relative paths, sorted
    pub symbols_seen_before: usize,
    pub seen_percent_before: f64,
}
```

### `CoverageReport` extension

```rust
#[derive(Debug, Clone, Serialize)]
pub struct CoverageReport {
    pub session_id: Option<String>,
    pub agent_id: Option<String>,
    pub files: Vec<FileCoverage>,
    pub compactions: Vec<CompactionSummary>,  // ← new, empty when none
}
```

### `run_report` loop

Maintain a `BTreeSet<PathBuf>` alongside the ledger. Snapshot it on each `SessionEvent::Compacted`:

```rust
let mut files_accessed: BTreeSet<PathBuf> = BTreeSet::new();
let mut tool_call_count: usize = 0;
let mut compactions: Vec<CompactionSummary> = Vec::new();

for event in ingester.parse_log_file_with_root(log_file, project_path) {
    match event {
        SessionEvent::ToolCall(tc) => {
            tool_call_count += 1;
            if let Some(ref fp) = tc.file_path {
                let rel = normalize_tool_path(fp, project_path);
                files_accessed.insert(rel.clone());
            }
            // existing ledger-update / mark_file_symbols logic...
        }
        SessionEvent::Compacted { summary, timestamp, .. } => {
            let seen = ledger.total_seen();
            let total = project_tree.total_symbols();
            compactions.push(CompactionSummary {
                sequence: compactions.len() as u32 + 1,
                timestamp,
                summary,
                tool_calls_before: tool_call_count,
                files_before: files_accessed.iter()
                    .map(|p| p.to_string_lossy().into_owned())
                    .collect(),
                symbols_seen_before: seen,
                seen_percent_before: if total > 0 {
                    seen as f64 / total as f64 * 100.0
                } else { 0.0 },
            });
        }
        SessionEvent::SessionCleared => {
            ledger = ContextLedger::new();
            files_accessed.clear();
        }
    }
}

let mut report = CoverageReport::from_project(project_tree, &ledger, resolved_agent.as_deref());
report.session_id = session_id;
report.compactions = compactions;
```

### Text formatter output

After the TOTAL row, when `!report.compactions.is_empty()`:

```
Context Compactions (2)
───────────────────────────────────────────────────────────────
#1  2026-05-11T14:23:00Z  47 calls · 31.2% seen
    Files (12):
      src/cli.rs
      src/parser/mod.rs
      src/parser/rust.rs
      src/symbols/mod.rs
      ... (8 more)
    Summary: "The user is building a CLI parser. Files modified:
      src/parser/mod.rs, src/cli.rs. Next step: add flag validation..."

#2  2026-05-11T15:41:00Z  89 calls · 58.0% seen
    Files (28):
      src/coverage.rs
      src/ingest/claude.rs
      ...
    Summary: "Work continued on the CLI parser. TypeScript support added..."
```

Show the first 10 file paths then `... (N more)`. Wrap summary text at 80 chars with a hanging indent.

### JSON formatter output

Bump `schema_version` from `1` → `2`. Add `compactions` array (always present, empty when none):

```json
{
  "schema_version": 2,
  "session_id": "abc123",
  "agent_id": null,
  "totals": { "symbols": 142, "seen": 87, "full": 23, "seen_percent": 61.3, "full_percent": 16.2 },
  "files": [ ... ],
  "compactions": [
    {
      "sequence": 1,
      "timestamp": "2026-05-11T14:23:00Z",
      "summary": "The user is building a CLI parser...",
      "state_before": {
        "tool_calls": 47,
        "files_accessed": [
          "src/cli.rs",
          "src/parser/mod.rs",
          "src/parser/rust.rs",
          "src/symbols/mod.rs"
        ],
        "symbols_seen": 31,
        "seen_percent": 31.2
      }
    }
  ]
}
```

Use a `state_before` nested object so future fields (e.g. `token_count_before`) slot in cleanly.

---

## Implementation Order

Work in this sequence — each step compiles and passes tests before starting the next:

1. `src/ingest/claude.rs` — detect `type:"compaction"` blocks; add `ParsedLine::Compacted`; add unit test
2. `src/ingest/mod.rs` — add `SessionEvent`, `LedgerSnapshot`, `CompactionEvent`; update trait signatures
3. `src/ingest/claude.rs` — update `parse_log_file_with_mapper` to emit `SessionEvent::Compacted` in order
4. `src/events.rs` — add `AppEvent::Compacted`
5. `src/app.rs` — add `compaction_history`, `compaction_call_count`, `process_compaction()`; update `process_agent_event` and `reset_session`; update event dispatch callers
6. `src/main.rs` + `src/tui.rs` — update to match on `SessionEvent`
7. `src/ui/activity.rs` — render compaction marker in feed
8. `src/ui/compaction.rs` — new diff overlay panel; wire `C` key in `handle_key`
9. `src/ui/stats.rs` — add compaction count + last-compaction line
10. `src/coverage.rs` — add `CompactionSummary`, extend `CoverageReport`, update `run_report` loop, update both formatters; bump JSON schema to v2; update tests

---

## What Does NOT Change

- `ContextLedger` internal structure — no new fields or methods
- `SymbolId` format
- `agent-acompact-*` file exclusion — already correct
- The `--coverage` CLI flag signature
- Existing JSON consumers checking `schema_version == 1` should treat `2` as a superset

---

## Test Checklist

- [ ] `parse_compaction_block_returns_compacted` — unit test in `claude.rs`
- [ ] `parse_compact_command_returns_ignored` — existing test must still pass
- [ ] `process_compaction_snapshots_files` — unit test in `app.rs`
- [ ] `compaction_history_clears_on_reset_session` — unit test in `app.rs`
- [ ] `coverage_report_includes_compaction_summary` — unit test in `coverage.rs`
- [ ] `json_formatter_schema_version_2` — update existing schema_version test
- [ ] `json_formatter_compaction_array_empty_when_none` — new test
- [ ] `json_formatter_compaction_array_with_entries` — new test
- [ ] `text_formatter_compaction_section_omitted_when_empty` — new test
- [ ] `text_formatter_compaction_section_with_entries` — new test
