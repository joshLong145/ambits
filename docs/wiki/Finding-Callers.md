# Finding Callers

```bash
ambits -p . callers centered_rect
```

```
centered_rect — 2 call sites in 2 callers
  src/ui/alignment.rs::render  (src/ui/alignment.rs:16)
  src/ui/compaction.rs::render  (src/ui/compaction.rs:15)
```

Call sites come from the grammar's own tags query, plus patterns of ours where
that falls short, so a mention in a comment or inside a string literal is never
reported — the answer is a call node or it is not there. Each site is attributed
to the innermost symbol containing it, and that id goes straight into
[`show`](Reading-by-Symbol).

Several names can be passed at once. `--format json` gives, per name, the
counts and a `sites` list of `{file, line, caller}`, where `caller` is the
enclosing symbol's id — absent for a call at file scope:

```json
{"schema_version":1,"name_matched_only":true,"results":[{"callee":"centered_rect",
"call_sites":2,"callers":2,"sites":[{"file":"src/ui/alignment.rs","line":16,
"caller":"src/ui/alignment.rs::render"}, …]}]}
```

## Matching is by name, not by resolution

tree-sitter parses; it does not do type inference, so a call to `new()` cannot
be tied to one of the many definitions named `new`. Most function names in a
codebase are unique, so most answers are exact — but `callers new` returns
every call to anything named `new` (618 sites in 390 callers on this
repository). `--format json` sets `name_matched_only: true` so a consumer
cannot mistake this for a resolved call graph.

## What counts as a call

A call is reported when its callee has a name in the syntax. In TypeScript that
covers `foo()`, `a.b()`, `this.b()`, `a?.b()`, each link of `a.b().c()`,
`foo<T>()`, `a.b!()`, `this.#b()` (as `#b`), a decorator written `@foo()`, a
tagged template, and `new Foo()` as a call to `Foo`, as in Java and Python,
wherever the call appears.

These are not reported, in TypeScript or in the languages with an equivalent: a
computed `a['b']()`, a parenthesized `(a.b)()`, and a bare decorator `@foo`. Nor
are TypeScript's `super()` and `import()`.

Of the JavaScript family only `.ts` files are parsed. `.tsx`, `.mts`, `.js` and
`.jsx` are not, so calls in them are not reported.

A callback or object literal has no symbol of its own, so a call inside one
belongs to the symbol around it, or to `<file scope>` if there is none. Test
suites written as `suite('…', () => { … })` report mostly `<file scope>`.

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
