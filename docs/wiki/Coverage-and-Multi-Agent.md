# Coverage and Multi-Agent

## Coverage reports

```bash
ambits -p . --coverage
```

```
Coverage Report (session: 30172621-…)
─────────────────────────────────────────────────────────────────────────────
File                                      Symbols    Seen    Full   Seen%   Full%
─────────────────────────────────────────────────────────────────────────────
src/events.rs                                   3       3       3    100%    100%
src/parser/mod.rs                              15       2       2     13%     13%
src/app.rs                                    100     100     100    100%    100%
…
─────────────────────────────────────────────────────────────────────────────
TOTAL                                        1307     309     309     24%     24%
```

- **Seen%** — symbols the agent has any awareness of
- **Full%** — symbols read completely

```bash
ambits -p . --coverage --format json | jq '.totals.full_percent'
```

Combine with `--filter` / `--filter-regex` to report on part of the project
(see [Configuration](Configuration#restricting-scope)).

> **Caveat:** `--coverage` scores the session's reads against the current
> source, and a one-shot run has no way to tell that a symbol read earlier has
> since changed. It can therefore overstate `Full%` for code edited after it
> was read. `restore-context`, which consults the journal's content hashes, does
> not have this problem.

## Multi-agent sessions

When a session spawns sub-agents with the Task tool, ambits tracks each
independently:

```
Agents: 5
  ▶ [All]              Seen: 95%
  ├─ 7842313b          35%
  │  ├─ a38e68c        20%
  │  ├─ a9fe23c        41%
  │  └─ a845182        15%
  └─ compact-0aff      10%
```

In the [TUI](TUI): `Tab` to the Stats panel, `j`/`k` to move, `Enter` to
filter — tree, activity feed and depth breakdown all follow. `a` cycles agents
from any panel.

`d` opens the **alignment view**, which scores each pair of agents file by
file — useful for spotting sub-agents that duplicated each other's exploration.

Outside the TUI:

```bash
ambits -p . --coverage --agent a9fe23c
ambits -p . --coverage --agent a9fe        # prefix match
```

A prefix matching no agent, or several, warns rather than guessing.
