# TUI

```bash
ambits -p .
```

![screenshot](https://raw.githubusercontent.com/joshLong145/ambits/main/images/screenshot.png)

Tails the session log and updates live. A header line gives the session,
the agent filter and its totals; below it, the file tree on the left and the
**inspector** on the right, with `Tab` moving between them. `i` swaps the
inspector for the session pane (coverage by depth, agents, compactions), `f`
shows the activity feed, where edits appear marked `(write)` — they are
journaled as [writes](Agent-Writes), not reads. A legend line explains every
glyph.

Files start collapsed, including ones created while the TUI is running;
expanding a file shows its full outline. A file you expanded stays expanded
across edits and editor saves.

It watches your source files too, re-parsing one when it changes so the tree
and the coverage numbers follow your edits without a restart. The watcher
honours the project's `.gitignore`, so generated code stays out of the tree —
without that, a build tool writing into `target/` (rust-analyzer running
`cargo check`, say) pushes rows in for files you never wrote. Deleted files
leave the tree rather than lingering until you quit.

- **A state gutter** — every symbol's read depth, freshness and writes as glyphs in fixed columns (see [the legend](#reading-the-tree))
- **File bars** — each file's symbols as a bar of read in full, partly read and unseen, with `seen/total`, `!` and `✎` counts, so coverage shows without expanding
- **Inspector** — the selected row's states in words, and the prompts whose calls read or wrote it; `Tab` to it, `Enter` opens that trace at the call
- **Sortable tree** — alphabetical, or grouped by coverage to surface half-read files first
- **Search** — `/` to jump to a symbol by name
- **Compaction history** — `C` for this session's compaction boundaries
- **Trace panel** — in the trace view, the right-hand panel sums up the selected trace or call, and its rows lead to files, calls and commits (see [below](#the-trace-panel))
- **Trace view** — `t` lists the session's prompts; `Enter` puts one prompt's tool calls on a time axis, as an OpenTelemetry waterfall or Perfetto-style agent tracks (see [below](#trace-view))
- **Sub-agent alignment** — `d` compares two agents file by file: where they read the same code, and where only one looked (see [Coverage and Multi-Agent](Coverage-and-Multi-Agent))

While it runs, the TUI is also the sole writer of the
[read journal](Configuration#the-read-journal).

## Keybindings

Keys go to the focused panel — the left one (the tree, or the trace view in
its place), the right one, or the activity feed — whose border is cyan and
whose keys the status bar lists. These work everywhere:

| Key | Action |
|---|---|
| `Tab` / `Shift+Tab` | Move focus: left panel, right panel, feed (when shown) |
| `[` / `]` | Previous / next agent filter (the compaction overlay uses them to page) |
| `Esc` | From the right panel or feed, focus back to the left; on the left, up a level |
| `i` | Inspector / session pane on the right |
| `f` | Show or hide the activity feed |
| `q`, `Ctrl+C` | Quit |

In the tree:

| Key | Action |
|---|---|
| `j` / `k`, `↓` / `↑` | Move down / up |
| `h` / `l`, `←` / `→` | Collapse / expand tree nodes |
| `Enter` | Expand a node with children; on a leaf, [open it in your editor](#opening-a-symbol-in-your-editor) |
| `/` | Search symbols |
| `s` | Toggle sort (alphabetical / coverage) |
| `a` / `A` | Next / previous agent filter, as `]` / `[` |
| `d` | Sub-agent alignment view |
| `t` | [Trace view](#trace-view) in place of the tree |
| `C` | Compaction history (`[` / `]` to page) |
| `g` / `G` | Jump to first / last |
| `PgUp` / `PgDn` | Scroll by page |
| `Esc` | Close the alignment view, or cancel a search |

In the right panel: `j` / `k` choose a row, `Enter` opens it — in the
inspector, a trace that touched the row; in the session pane, an agent to
filter by; in the trace view, [what the row points at](#the-trace-panel).

## Reading the tree

Each symbol row starts with three columns, one state each:

```
    ●!✎ fn render        L12-80 ~310 tok
    ◑   fn move_selection
    ◔◌  fn old
▶ src/app.rs    ███▓▓░░░░░  42/120  !4  ✎3
```

| Column | Glyph | Meaning |
|---|---|---|
| Read depth | `●` | Full body read |
| | `◕` | Signature seen |
| | `◑` | Overview (grep match, symbol listing) |
| | `◔` | Name only (a glob, a listing, a `callers` or `show --no-body` result) |
| | `·` | Unseen |
| Freshness | `!` | Changed since it was read: what the agent knows is out of date |
| | `◌` | Read before a compaction: no longer in the agent's context |
| Write | `✎` green | Written this session, and the agent's version is still there |
| | `✎` amber / red | Changed since, or gone |
| | `✎` cyan | A file-level write: nothing in memory to compare it with |

Names are white once read, grey before. A folded symbol shows `seen/total`
for what it hides, as a file does. A written symbol is marked through
itself or anything nested in it — the agent filter's writes, when one is
set — by the same rule as [`ambits touched`](Agent-Writes#ambits-touched),
judged against the tree as it is now, so an edit of yours turns it amber at
once. Writes made before this run come from the
[read journal](Configuration#the-read-journal).

A file row's bar is its symbols in proportion: `█` read in full (and
unchanged), `▓` read less deeply, `░` unseen. Then `seen/total`, `!N` read
symbols changed since, and `✎N` writes, coloured by the latest.

## The inspector

The selected row in words:

```
 read     ● full body · main, a03cd45 (name only)
 context  live
 changed  no
 written  ✎ 2026-09-27 10:05Z by main (Edit) · 2 writes
          unchanged since the agent wrote it
 size     L577-661 · ~900 tokens

 traces   3 prompts · Enter opens
 › 09-27 14:02 read, wrote  "lets fix the blockers…"
   09-27 15:20 read         "lets now move on to ui-7"
```

A file shows its coverage, how many read symbols changed since, its latest
write and its size. **traces** lists the prompts whose calls read or wrote
the row — a read of the whole file, or of the symbol, something inside it
or something it is inside; a write attributed to it — oldest first. `Tab`
focuses the inspector, `j` / `k` choose, `Enter` opens that
[trace](#trace-view) with the call selected.

## Trace view

`t` replaces the tree with the session's **traces, one per prompt**: when
you asked, what, how long answering took, how many tool calls it made, how
many failed, how many subagents it started, and how many git commits were
made while it ran. The right-hand panel sums up the selected one. `j` / `k`
choose, `Enter` opens one, `Esc` goes back to the tree.

An open trace is that prompt's tool calls on a time axis: the prompt is the
root span, every call made answering it a sub-span, each subagent's calls
under the delegation that started it. It is the model
[`ambits trace`](Traces) exports, so what you see here is what Jaeger or
Perfetto would show. `Esc` returns to the list.

Two layouts, `v` to switch:

- **Waterfall** (as Jaeger or Tempo show an OpenTelemetry trace): one row per
  call, nested by delegation, with its duration. A delegation lasts until
  its agent last stopped; `h` / `l` or `Space` fold its calls away.
- **Tracks** (as Perfetto shows a system trace): one track per agent, in
  delegation order, overlapping calls stacked into lanes, names drawn inside
  bars wide enough to hold them. A folded track (`Space`) is one row of
  density.

Bars take the tree's colours: reads by depth, writes by whether their
version is still there, failures red, delegations grey. `▼` marks a
compaction, and `│` a git commit (hash and subject), among the calls it
followed. Commits are found on local branches and `HEAD` by committer
time, off the render thread, every few seconds while the view is open. The
line under the bars names the selected call — agent, duration, and why it
failed — and the right-hand panel has the rest.

The timeline follows its trace live until you zoom; `0` goes back to that.
Its keys are modal, as in Perfetto:

| Key | Action |
|---|---|
| `w` / `s` | Zoom in / out around the selection |
| `a` / `d` | Pan earlier / later |
| `0` | Fit the whole session, and follow it |
| `j` / `k`, `g` / `G` | Next / previous row; first / last |
| `h` / `l` | Waterfall: fold / unfold. Tracks: previous / next call on the lane |
| `Space` | Fold the selected delegation (waterfall) or track (tracks) |
| `Enter` | On a delegation, into its agent's calls; on a read or write, to its symbol or file in the tree |
| `/` | Waterfall: list only calls whose name contains the text (`Esc` clears) |
| `e` | Next failed call |
| `v` | Waterfall / tracks |
| `Esc` | Back to the list of traces |
| `t` | Back to the tree |

The mouse wheel zooms at the pointer, a drag pans, a click selects.

### The trace panel

In the trace view the right-hand panel follows the selection.

**A trace** (on the list, or its prompt selected):

```
 lets now move on to ui-7
 09-28 00:54 · 17m21s · 29 calls · main + 1 agent(s)
 Read 12 · Bash 9 · Edit 6 · Agent 2
 1 failed · 2 commit(s)

 files                          read wrote
 › src/trace/view.rs               4     3  ✎ still there
     read  ● full body  (whole file)
     read  ◐ signature  TraceView/zoom
     wrote ✎ still there TraceView/fit
     wrote ✎ changed     Viewport
     … 3 more
   src/app.rs                      6     2  ✎ changed
 agents
   Expert review of phase 3  7m57s · 17 calls
 failed
   ✗ Edit src/ui/mod.rs — String to replace not found in file.
 commits
   commit 7fc8b45 Trace view: the layout core
```

The prompt in full, when and how long, calls by tool, the files it read and
wrote — and whether its writes are still there, changed or gone — its
agents, its failed calls with the reason, and the commits made meanwhile.

Under each file, what the trace read of it — each symbol once, at the
deepest it was read, `(whole file)` for a read of all of it — then what it
wrote, each symbol as its latest write in the trace left it. Up to six
lines a file; the panel scrolls to keep the selected row in view.

**A call**: who made it, when and for how long, in which trace; why it
failed (the error's own line: a traceback's last, a compiler's first
`error`, "rejected by the user"); what it read and the symbol's depth now;
what it wrote, symbol by symbol, with whether each still stands; the
command it ran, in full; what an agent it started did; and the other calls
on its file, before and after it.

`Tab` to the panel, `j` / `k` to a row, `Enter` to open it: a file shows in
the tree; an agent, a failure, a related call or a commit is selected in
the timeline (the trace opened, and unfolded down to it). Focus stays on
the panel, so you can keep following the trail. `Esc` returns to the
timeline.

## Opening a symbol in your editor

`Enter` on a leaf row — a symbol with no children, or a file with none — opens
that file in an external editor, jumped to the symbol's declaration line. A row
with children still expands and collapses.

The editor is resolved in this order:

1. `--editor <TEMPLATE>` on the command line
2. `[editor]` in `tools.toml` (whichever one is in use — see [Configuration](Configuration#toolstoml))
3. `$VISUAL`
4. `$EDITOR`
5. none — `Enter` shows a message in the status bar instead of guessing

A resolved value is a command template. Plain `EDITOR=vim` or `--editor code`
still jumps to the right line: a short built-in table supplies the line-jump
syntax for common editors, keyed off the command's basename.

| Editor | Template |
|---|---|
| `vim`, `nvim`, `vi` | `{editor} +{line} {file}` |
| `emacs`, `emacsclient` | `{editor} +{line} {file}` |
| `nano` | `{editor} +{line} {file}` |
| `code`, `code-insiders` | `{editor} --goto {file}:{line}` |
| `subl`, `sublime_text` | `{editor} {file}:{line}` |
| `zed` | `{editor} {file}:{line}` |
| anything else | `{editor} {file}` (opens the file, no line jump) |

If your editor isn't in the table, or its binary goes by a different name
(invoked by full path, say), write the template explicitly. A template
containing `{file}`/`{line}` is used verbatim, with no basename guessing:

```toml
[editor]
command = "/path/to/some-editor --line {line} {file}"
```

A spawn failure or nonzero exit doesn't crash the TUI — it shows the error in
the status bar and leaves everything else running.
