# Extended Examples

## Find, Then Fetch

Search for what you need, then read just the symbol or section it is in:

```bash
# Where is a write's status worked out?
ambits -p . rg 'fn symbol_status' -t rust
# src/writes.rs:198:9:[— FileContents/symbol_status]     pub fn symbol_status(&self, …
ambits -p . show 'src/writes.rs::FileContents/symbol_status'

# Who calls it? (call nodes only — no comments, no strings)
ambits -p . callers symbol_status

# What do the docs say about --break-lock?
ambits -p . rg -t md 'break-lock' -l                     # which documents
ambits -p . rg '^#{1,3} ' docs/wiki/Sharing.md           # that document's outline
ambits -p . show 'docs/wiki/Sharing.md::Sharing/Push'    # the one section
```

A match marked `[full …]` is already read: skip the `show`. For a large
symbol, `show --no-body` first — its `estimated_tokens` says what the full
fetch costs.

## Pre-Implementation Check

Before implementing features, verify you've seen the relevant code:
```bash
ambits -p . --coverage
# Look for low Seen% in files you'll be modifying
# If coverage is low, ask Claude to read those files first
```

## Post-Review Audit

After a code review session, verify coverage:
```bash
ambits -p . --coverage
# Ensure critical files have high Full% coverage
# Flag any security-sensitive files with low Full%
```

## Historical Analysis

Review what was examined in previous sessions:
```bash
# Find sessions from a previous review
ls -lt ~/.claude/projects/<slug>/*.jsonl

# Check coverage from an earlier session
ambits -p . --coverage --session <old-session-id>

# Compare with current session
ambits -p . --coverage
```

## Comparing Sessions

```bash
# Run coverage for two different sessions and compare
ambits -p . --coverage --session <session-1> > /tmp/session1.txt
ambits -p . --coverage --session <session-2> > /tmp/session2.txt
diff /tmp/session1.txt /tmp/session2.txt
```

## Interactive TUI Mode

```bash
# Launch the interactive tree view
ambits -p .

# With a specific session
ambits -p . --session <session-id>
```

In the TUI:
- `j`/`k` to navigate up/down
- `h`/`l` to collapse/expand
- `/` to search symbols
- `a` to cycle agent filter
- `Tab` to switch panel focus
- `q` to quit

## Using with Serena MCP

If your project uses Serena MCP for LSP-based symbol analysis:
```bash
# Uses Serena's symbol cache for more languages and finer detail
ambits -p . --serena --coverage
```

This supports any language Serena's LSP can parse, not just Rust and Python.
