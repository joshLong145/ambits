# Finding Callers

```bash
ambits -p . callers centered_rect
```

```
centered_rect — 2 call sites in 2 callers
  src/ui/alignment.rs::render  (src/ui/alignment.rs:16)
  src/ui/compaction.rs::render  (src/ui/compaction.rs:15)
```

Call sites come from the grammar's own tags query, so a mention in a comment or
inside a string literal is never reported — the answer is a call node or it is
not there. Each site is attributed to the innermost symbol containing it, and
that id goes straight into [`show`](Reading-by-Symbol).

Several names can be passed at once. `--format json` returns match objects
shaped exactly like `show --no-body`.

## Matching is by name, not by resolution

tree-sitter parses; it does not do type inference, so a call to `new()` cannot
be tied to one of the many definitions named `new`. Most function names in a
codebase are unique, so most answers are exact — but `callers new` returns
every call to anything named `new` (618 sites in 390 callers on this
repository). `--format json` sets `name_matched_only: true` so a consumer
cannot mistake this for a resolved call graph.

## Cost

References are extracted on demand rather than stored, so `rg`, `show` and the
TUI pay nothing for this. A `callers` query re-reads and re-parses — skipping
any file whose text does not contain the name at all — so it costs more than a
search, but unlike one it reports call nodes only: the definition and the doc
comments that mention it do not come back with them.

Macro arguments are handled too: tree-sitter leaves them as unparsed token
trees, so they are re-parsed as source. Without that, anything called from
inside `println!` or `assert_eq!` would be invisible. See
[Architecture](Architecture#how-a-query-is-served).
