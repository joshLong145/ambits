# Restoring Context

Every read is tracked per symbol and per agent, at the depth the tool implies —
a `Read` gives full body, a grep match gives overview, a glob gives name only
(the full mapping is data; see [Configuration](Configuration#tool-mappings)).
`restore-context` reports that history:

```bash
ambits -p . restore-context
```

```
### src/app.rs — 100 symbols (~49.2k tok)
App/switch_session:342-345, App/process_compaction:364-412,
App/rebuild_tree_rows:415-482, App/handle_key:484-557, …

### src/digest.rs — 47 symbols (~13.7k tok)
grouped:64-84, symbol_label:104-119, fit_names:126-141, …
```

Entries are `name:first-last`. Line numbers come from a fresh scan at print
time rather than from storage, so they stay correct in files edited since the
read — the agent can read that range directly instead of pulling the file.

Only symbols that are **still unchanged** are reported: each journaled read
carries the content hash it was read at, and a symbol whose source has changed
since is dropped, because the agent's memory of it is stale.

A symbol that moved between files is annotated `(was <old path>)`. ambits
identifies symbols by content as well as by path, so hoisting a helper into a
shared module does not lose it.

## Options

| Flag | Effect |
|---|---|
| `--max-tokens N` | Fit an approximate token budget (default 3000) |
| `--format markdown` | The default, for pasting or piping into a session |
| `--format json` | The same data, schema-versioned, including each symbol's `content_hash` for exact `show` lookups |
| `--format hook` | Claude Code's `SessionStart` envelope; prints nothing when there is nothing to restore |

## Where the history comes from

From the coverage journal when one exists — see
[Configuration → The read journal](Configuration#the-read-journal). Without
one, `restore-context` falls back to replaying the session logs. That cannot
detect drift, so its output is labelled **UNVERIFIED**.

## Automatic hand-back after compaction

```bash
ambits hook install --project .      # or --global for ~/.claude/settings.json
```

Registers a `SessionStart` hook with `matcher: "compact"` in
`.claude/settings.json`, so Claude Code runs `restore-context --format hook`
and injects the result the moment a compaction completes. It merges into
existing settings and is safe to re-run; a settings file that cannot be parsed
is left untouched and the snippet is printed for you to add by hand.

Symbols carried over from before a compaction render dimmed in the
[TUI](TUI) — the read happened, but it is no longer in the agent's live
context.

## Lookups count as reads

ambits parses `show` invocations out of the session log and credits the
symbols they name, so reading efficiently costs nothing in coverage versus a
plain `Read`. `--no-body` credits name-level only — the agent learned where a
symbol is, not what it says. `rg`/`grep` credit the symbols whose matching
lines they printed; see [Searching Code](Searching-Code#searching-is-reading).

Credit is best-effort: it is reconstructed from the logged command text, so a
selector passed through a shell variable or command substitution is not
visible. It fails toward under-reporting, never over.
