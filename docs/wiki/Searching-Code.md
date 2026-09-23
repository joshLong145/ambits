# Searching Code

A grep whose every hit knows which symbol it landed in.

```bash
ambits -p . rg 'is_binary'                 # every use and definition
ambits -p . rg 'fn enclosing' -t rust      # one file type (--type-list for all)
ambits -p . rg 'TODO' -g '!tests/**'       # globs; ! excludes
ambits -p . rg 'Journal::open' -A 3        # with trailing context
ambits -p . rg 'unwrap\(\)' -c             # matching lines per file
```

```
src/search.rs:73:7:[full BINARY_SNIFF_BYTES] const BINARY_SNIFF_BYTES: usize = 8 * 1024;
src/search.rs:390:4:[full is_binary] fn is_binary(buf: &[u8]) -> bool {
src/search.rs:391:21:[full is_binary]     buf.iter().take(BINARY_SNIFF_BYTES).any(|&b| b == 0)
src/search.rs:495:8:[— search_file]     if is_binary(&buf) || !matcher.worth_searching(&buf) {
```

`file:line:column:` — the prefix every grep consumer already parses — then the
symbol the match sits in and how deeply this session has read it, then the
line. `--no-symbol` drops that field for output byte-identical to ripgrep's.

## What the symbol column means

| Form | Meaning |
|---|---|
| `[full name]`, `[signature name]`, … | This session has read the symbol, at that depth |
| `[— name]` | It has not |
| `[name]` | No coverage journal: *unknown*, which is not the same as unread |
| `[-]` | The match is not inside any symbol — a `use` line, or a file no parser handles |

The third row is the one that matters. An empty column would read as "unread"
when the honest answer is "nobody was watching", and only one of those means
go read it. In `--json`, a `coverage` object on the summary event is what
distinguishes them.

Every text file is searched, not only the parseable ones — a hit in a TOML file
is a real hit, it simply has no symbol. Nested symbols are shown by their
name-path: a hit under a Markdown heading reads `[For the agent/Searching code]`.

## Why two commands

`grep(1)` and ripgrep assign **opposite meanings to the same short flags**, so
no single command can be faithful to both:

| Flag | GNU grep | ripgrep |
|---|---|---|
| `-L` | `--files-without-match` | `--follow` (symlinks) |
| `-z` | `--null-data` | `--search-zip` |
| `-r` | `--recursive` | `--replace` |
| `-h` | `--no-filename` | help |

`ambits rg` and `ambits grep` are two front ends over one engine — the same
matcher, the same symbol attribution, the same output contract — each faithful
to the tool it is named after.

**`rg` is the one to reach for.** Claude Code's own `Grep` tool is
ripgrep-backed, so it is the dialect agents already speak, and it uses the same
`regex` crate — patterns behave identically, including the shared absence of
backreferences and lookaround.

### `ambits grep`

Keeps grep's defaults rather than ours: line numbers are opt-in (`-n`), there
is no column unless `--column` is given, and `-h` is `--no-filename` (help is
`--help` only). It takes grep's own filters — `--include`, `--exclude`,
`--exclude-dir` — and `-Z` for NUL-terminated paths.

Flags it cannot honour say so rather than pretending:

- `-P` is **refused**: there is no lookaround in this engine, so a PCRE pattern
  would match something other than what it says.
- `-z` is **refused**: NUL-separated input would change what a line is.
- `-r` / `-R` are **accepted no-ops**: the search is always recursive, since
  scoping to a project is what ambits is for.
- `-E` / `-G` are accepted with no behaviour change: one engine, close enough
  to ERE for anything portable.

## Deviations from ripgrep, all deliberate

- **Output is always sorted** by path, line and column — determinism is worth
  more to an agent than the microseconds.
- **`--head-limit` caps at 200 matches** and **`-M` clips lines at 300
  columns**, because this output lands in a context window rather than a
  terminal. `0` lifts either.
- **`--column` is on by default** — it is what disambiguates two matches on
  one line. `--no-column` turns it off.
- **Exit codes are grep's**: `0` matched, `1` nothing matched, `2` error.

`--json` emits ripgrep's JSON Lines events with an added `symbol` field; see
the [CLI Reference](CLI-Reference#ambits-rg) for the full flag list.

## Searching is reading

A search prints source into an agent's context, so it records what it showed:
every symbol whose matching line was printed is journaled as read, at the hash
it was searched at. Modes that print no source — `-q`, `-l`, `-c` — record
nothing, and neither do matches cut past `--head-limit`. The journal is a
record of what was *seen*, not of what the process computed.

Credit is only recorded while the TUI is running for the session — see
[Configuration → The read journal](Configuration#the-read-journal).

## The pipeline runs backwards

Every other command scans the project first: walk, parse every file, then
answer. A search inverts that — walk, read, reject on the raw bytes, and parse
only the files that matched. A pattern that matches nothing parses nothing; on
this repository a typical search completes in well under 10 ms, where a
symbol-index search would pay for a full parse every time.
