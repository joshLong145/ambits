# Reading by Symbol

```bash
ambits -p . show 'src/search.rs::is_binary'
```

```json
{"schema_version":2,"results":[{"query":"src/search.rs::is_binary","selector":"id",
"matches":[{"id":"src/search.rs::is_binary","name":"is_binary","lines":[386,392],
"bytes":[15143,15444],"content_hash":"b3:53a28842…","label":"fn","estimated_tokens":115,
"definition":"/// Whether `buf` looks like something a reader would want printed.\n…\nfn is_binary(buf: &[u8]) -> bool {\n    buf.iter().take(BINARY_SNIFF_BYTES).any(|&b| b == 0)\n}"}]}]}
```

## Selectors

A selector is either:

- a **symbol id** — `<path>::<name-path>`, exactly what `restore-context`,
  `rg` and `callers` print. Nested symbols join with `/`:
  `src/app.rs::App/handle_key`, `README.md::Ambits/Quick start`.
- a **content hash** — `b3:<hex>`, full or at least 8 hex characters.

Several resolve per invocation, so a batch of lookups costs one process:

```bash
ambits -p . show b3:53a28842 'src/digest.rs::grouped' 'src/app.rs::App/handle_key'
```

## The output

| Field | Meaning |
|---|---|
| `lines` | 1-based, inclusive — the same range `restore-context` prints |
| `bytes` | Byte offsets, for callers that want to slice the file themselves |
| `content_hash` | BLAKE3 over whitespace-normalized source |
| `label` | Syntactic kind: `fn`, `struct`, `impl`, `h2`, … |
| `children` | Immediate child names, so a container can be walked without a second scan |
| `definition` | The exact source span, sliced by byte offset rather than reconstructed from lines |
| `truncated` | Present and `true` only when `--max-bytes` cut the definition |
| `read_depth` | How deeply this session has read the symbol |

The top-level `coverage` object (`session_id`, `symbols_read`) is present only
when a coverage journal was loaded. That is what tells a consumer whether a
missing `read_depth` means *unread* or *unknown*.

`--no-body` returns location metadata only. `--max-bytes N` caps each
definition and flags it `"truncated": true`; it is unlimited by default,
because a cut definition is no longer valid source and shortening one is the
caller's decision.

## Ambiguity is reported, not resolved

`matches` is an array because ids are not guaranteed unique — Rust allows a
type several inherent `impl` blocks in one file, and nothing in the name
distinguishes them. A content hash always names exactly one symbol.

- Empty `matches` — no such symbol.
- `"selector": "unrecognized"` — the query was neither an id nor a hash.

The command exits `0` either way: "nothing matches" is an answer, not a
failure.

## Why it matters

Reading a symbol instead of its file is a large saving. `src/filter.rs` is
several hundred lines — thousands of tokens to read whole — while the handful
of `PathFilter` methods a caller actually needs come to a few hundred.
`estimated_tokens` on every match makes that cost visible before the body is
fetched: ask with `--no-body` first, then fetch only what is worth it.

`show` lookups are credited as reads, like a plain `Read` — see
[Restoring Context → Lookups count as reads](Restoring-Context#lookups-count-as-reads).
