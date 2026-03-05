# Parser Optimization Workflow
**Project**: ambit (`src/parser/`, `src/symbols/`, `src/app.rs`, `src/ui/`)
**Strategy**: Systematic
**Generated**: 2026-03-01

---

## Overview

Two independent optimizations with no inter-dependency. Task 1 (label) is a mechanical
find-and-replace. Task 2 (iterative DFS) is a non-trivial behavioral refactor. Execute
Task 1 first — it is low-risk, verifiable in one test run, and its changes are fully
orthogonal to the DFS work.

---

## Phase 1 — `label: &'static str` (mechanical, zero-risk)

**Goal**: Eliminate one `String` heap allocation per `SymbolNode` construction by storing
the language-specific label as a `&'static str` instead of a `String`.

**Why safe**: All label values are `&'static str` constants defined at module scope in each
parser. No runtime string is ever stored in this field. All test assertions
(`assert_eq!(sym.label, "fn")`) remain valid since `PartialEq<str>` is implemented for
`&'static str`.

### Step 1.1 — Core struct change
**File**: `src/symbols/mod.rs:48`
```rust
// BEFORE
pub label: String,
// AFTER
pub label: &'static str,
```
**Checkpoint**: `cargo check` — expect cascade of type errors to guide remaining steps.

### Step 1.2 — TreeRow struct change
**File**: `src/app.rs:34`
```rust
// BEFORE
pub label: String,
// AFTER
pub label: &'static str,
```

### Step 1.3 — Fix TreeRow construction sites in app.rs
**File**: `src/app.rs:192` (file header row)
```rust
// BEFORE
label: String::new(),
// AFTER
label: "",
```
**File**: `src/app.rs:587` (symbol row — `&'static str` is `Copy`, no `.clone()` needed)
```rust
// BEFORE
label: sym.label.clone(),
// AFTER
label: sym.label,
```
**Do NOT change**: `src/app.rs:512` — that line is `AgentNode.label: String` from
`Arc<str>`, completely unrelated.

### Step 1.4 — Fix parser construction sites
All three parsers have `label: meta.label.to_string()` which converts a `&'static str`
constant to `String` unnecessarily.

**File**: `src/parser/rust.rs` (~lines 116, 170)
```rust
// BEFORE
label: meta.label.to_string(),
// AFTER
label: meta.label,
```

**File**: `src/parser/python.rs` (~lines 188, 263)
```rust
// BEFORE
label: meta.label.to_string(),
// AFTER
label: meta.label,
```

**File**: `src/parser/typescript.rs` — inside `make_symbol` (~line 552)
```rust
// BEFORE
label: meta.label.to_string(),
// AFTER
label: meta.label,
```

### Step 1.5 — Fix UI test-fixture construction sites
**File**: `src/ui/tree_view.rs:123`
```rust
// BEFORE
label: "fn".into(),
// AFTER
label: "fn",
```
**File**: `src/ui/stats.rs:245`
```rust
// BEFORE
label: "fn".into(),
// AFTER
label: "fn",
```

### Step 1.6 — Fix bench and test fixture construction sites
**File**: `benches/support/fixtures.rs:964`
```rust
// BEFORE
label: "fn".to_string(),
// AFTER
label: "fn",
```
**File**: `tests/helpers/mod.rs:15`
```rust
// BEFORE
label: "fn".to_string(),
// AFTER
label: "fn",
```
**File**: `tests/e2e.rs:32`
```rust
// BEFORE
label: "fn".to_string(),
// AFTER
label: "fn",
```

### Step 1.7 — Validation gate
```
cargo test
```
All 271 tests must pass. No behavioral change — this is a type-only refactor.

---

## Phase 2 — Iterative DFS (behavioral refactor, parser-by-parser)

**Goal**: Replace recursive tree-walks with explicit `Vec`-based stack DFS to:
- Eliminate stack overflow risk on deeply nested source code
- Remove function call overhead for deep call chains
- Consolidate mutual recursion (`extract_symbols` ↔ `extract_from_compound_bodies`)
  into a single loop

**Why non-trivial**: `SymbolNode` requires children to be populated *before* the parent
is pushed to output. This means a two-phase stack frame approach is needed:
- **Discover frame**: visit a node, build the `SymbolNode` stub with empty `children`,
  push a *populate frame* for its children, then push the stub to output.
- **Populate frame**: iterate over children of a container node and push discover frames
  for each recognized child.

**Stack frame enum** (illustrative — adapt per parser):
```rust
enum Frame<'a> {
    /// Discover and emit symbols from children of `node`.
    Discover {
        node: Node<'a>,
        parent_name_path: String,
        /// Index into `out` where children should be pushed, or None for top-level.
        parent_idx: Option<usize>,
    },
}
```
Since `SymbolNode.children: Vec<SymbolNode>` is owned, the simplest approach is:
1. Push symbol to `out` with `children: Vec::new()`.
2. Record `out.len() - 1` as `parent_idx`.
3. Push a new `Discover` frame that targets `out[parent_idx].children`.

> **Note on TypeScript**: `emit_namespace`, `emit_class`, `emit_interface` currently
> collect children into a local `Vec` then move them into the parent. This pattern maps
> cleanly to the iterative approach — the stack frame carries a `Vec<SymbolNode>` for
> in-progress children.

### Step 2.1 — Rust parser (`src/parser/rust.rs`) — DEFERRED ✗

**Decision: no recursion exists.** `extract_symbols` calls `extract_body_children` which is a flat subroutine — no recursive calls anywhere. "Converting to iterative DFS" would only mean inlining `extract_body_children`, a purely cosmetic change that reduces readability (merges two well-named, focused functions). Deferred.

---

### ~~Step 2.1~~ — Original Rust scope

**Scope**: Replace `extract_symbols` + `extract_body_children` with a single iterative
`extract_symbols` function.

**Recursion map**:
```
extract_symbols(root_node)
  └─ for each recognized child:
       build SymbolNode
       if container (impl / trait / mod):
         extract_body_children(declaration_list_node)  ← depth-1 only, no deeper nesting
           └─ for each recognized child: build SymbolNode (no further recursion)
```
The Rust parser recurses at most **two levels** deep (top-level → impl/trait/mod body).
There is no mutual recursion. The iterative version is a straightforward stack loop.

**Iterative design**:
```rust
struct Frame<'a> {
    node: Node<'a>,
    parent_name_path: String,
    // None = push to top-level `out`; Some(i) = push to out[i].children
    out_target: OutTarget,
}
enum OutTarget { TopLevel, Child(usize) }
```

**Implementation notes**:
- Push initial `Frame { node: root, parent_name_path: "", out_target: TopLevel }`.
- Pop frame, iterate `node.children(cursor)`.
- On recognized child: create `SymbolNode` stub, push to the correct target vec,
  record index if it is a container, push a child `Frame` for its `declaration_list`.
- No change to `named_symbol`, `impl_symbol`, `find_name`, or `child_by_kind` helpers.

**Checkpoint**: Run `cargo test -- rust` (Rust parser tests only).

### Step 2.2 — Python parser (`src/parser/python.rs`) — DEFERRED ✗

**Decision: do not convert.** Direct comparison of the current recursive implementation
against the two-pass DP approach found the refactor net-negative:

| Dimension | Current | Two-pass DP |
|---|---|---|
| Allocations per file | N SymbolNodes only | + `Vec<WorkItem>` + `HashMap` + N `String` clones |
| Stack depth risk | ~60KB at 15 nesting levels (negligible) | None |
| Readability | High — grammar fields documented in table | Lower — ordering invariant non-obvious |
| Bug surface | Low — 40+ tests, locally verifiable | New invariant: parent_idx < child_idx must always hold |
| Adding a new compound type | 1 match arm | Equivalent complexity |

The mutual recursion between `extract_symbols` and `extract_from_compound_bodies` is
correct decomposition, not pathological recursion. Each function has a single
responsibility; the call cycle captures the transitive semantics of Python definitions
inside control flow. Two-pass DP adds complexity and allocations to fix a runtime risk
that does not exist for human-authored source files.

**What to do instead**: apply only the Phase 1 `label: &'static str` fix to `python.rs`.

---

### ~~Step 2.2~~ — Compound statement handling (original scope)

**Recursion map**:
```
extract_symbols(node)
  ├─ class_definition → build SymbolNode, recurse into body via extract_symbols
  ├─ decorated_definition → extract_decorated → may call extract_symbols
  └─ compound statement (if/for/while/try/with/match)
       └─ extract_from_compound_bodies → calls extract_symbols on each clause body
```

#### DP Strategy Analysis

Three strategies were evaluated (see layer2 research). The mutual recursion is not a
stack overflow risk for human-written Python — at ~2 frames per nesting level, 50 levels
deep is ~100KB, well under Rust's 8MB default stack. The risk is architectural
(readability, maintainability, testability), not runtime.

| Strategy | Complexity | Stack Safe | Allocation Cost | Verdict |
|---|---|---|---|---|
| **A: Explicit stack** (`Vec<Frame>`) | Medium | Yes | Low | Viable but requires index-based parent tracking to avoid borrow issues |
| **B: Trampolining** | High | Yes | Medium | Poor fit — trampoline assumes flat output; tree building is awkward |
| **C: Two-pass post-order DP** | Medium-High | Yes | Medium | Most principled for `children-before-parent` constraint |

**Recommended approach: Strategy C — two-pass post-order DP.**

This maps most cleanly onto the `SymbolNode.children` constraint and eliminates the
borrow-checker hazard in Strategy A (`parent_idx` invalidation on Vec reallocation).

**Pass 1 — structural collection (pre-order, iterative):**

Use an explicit `Vec` work-stack to collect `WorkItem` records without building
`SymbolNode`s. No borrow issues — just `Node<'a>` (which is `Copy`) + metadata.

```rust
struct WorkItem<'a> {
    node: Node<'a>,
    parent_name_path: String,
    parent_idx: Option<usize>,  // index into work_items, None = top-level
}

fn collect_work_items<'a>(root: Node<'a>) -> Vec<WorkItem<'a>> {
    let mut stack: Vec<(Node<'a>, String, Option<usize>)> = vec![(root, String::new(), None)];
    let mut items: Vec<WorkItem<'a>> = Vec::new();
    while let Some((node, parent_path, parent_idx)) = stack.pop() {
        let idx = items.len();
        items.push(WorkItem { node, parent_name_path: parent_path.clone(), parent_idx });
        // push recognized children (class body, compound clause bodies) onto stack
    }
    items
}
```

**Pass 2 — bottom-up SymbolNode construction (post-order):**

Process `work_items` in reverse (post-order guarantee: parent index is always < child
index in pre-order collection, so reverse = post-order). Use
`HashMap<usize, Vec<SymbolNode>>` as the DP accumulator table.

```rust
fn build_symbols(items: Vec<WorkItem>, src: &[u8], ...) -> Vec<SymbolNode> {
    let mut table: HashMap<usize, Vec<SymbolNode>> = HashMap::new();
    let mut out: Vec<SymbolNode> = Vec::new();

    for (idx, item) in items.iter().enumerate().rev() {
        let children = table.remove(&idx).unwrap_or_default();
        if let Some(sym) = build_symbol_node(&item, children, src, ...) {
            match item.parent_idx {
                None => out.push(sym),
                Some(parent) => table.entry(parent).or_default().push(sym),
            }
        }
    }
    out.reverse(); // restore declaration order
    out
}
```

**Key properties**:
- `extract_from_compound_bodies` is eliminated — its clause-unwrapping logic moves into
  Pass 1's stack loop as additional push rules.
- `extract_decorated` is converted to push `WorkItem`s instead of recursing.
- No raw pointers, no `unsafe`, no borrow-checker gymnastics.
- The DP "table" (`HashMap<usize, Vec<SymbolNode>>`) is the accumulator that respects
  the children-before-parent constraint.

**Checkpoint**: Run `cargo test -- python` (Python parser tests only).

### Step 2.3 — TypeScript parser (`src/parser/typescript.rs`) — DEFERRED ✗

**Decision: correct iterative form is net-negative.** The only recursion is `extract_symbols → emit_namespace → extract_symbols`. A safe iterative version requires `Rc<RefCell<Vec<SymbolNode>>>` as the shared target type (raw `*mut Vec` into `out[i].children` is unsound on Vec reallocation), plus a `FinalizeNS` frame type. Cost vs. current:

| | Recursive | Iterative |
|---|---|---|
| LOC for namespace handling | ~10 (emit_namespace) | ~40 (frame enum + loop) |
| Allocs per namespace | 1 Vec | 1 Vec + 2 Rc allocs |
| Stack risk | ≤ nesting depth (≤ 4 in practice) | None |
| Correctness risk | Low | FinalizeNS ordering invariant |

Deferred for the same reasons as Python and Rust.

---

### ~~Step 2.3~~ — Original TypeScript scope

**Scope**: Make `extract_symbols` iterative. Convert `emit_namespace` (which calls back
into `extract_symbols`) to push a stack frame instead of recursing.

**Recursion map**:
```
extract_symbols(root)
  ├─ class_declaration → emit_class → extract_members (no further recursion)
  ├─ abstract_class_declaration → emit_class → extract_members
  ├─ interface_declaration → emit_interface → extract_members
  ├─ internal_module / module → emit_namespace → extract_symbols  ← recursion
  ├─ expression_statement → emit_namespace → extract_symbols      ← recursion
  ├─ ambient_declaration → extract_ambient → may call extract_symbols
  └─ lexical_declaration / variable_declaration → extract_arrow_fns (no recursion)
```

**Iterative design**:
- `extract_symbols` becomes the stack loop.
- `emit_namespace`: instead of calling `extract_symbols` recursively, push a new stack
  frame for the namespace's `statement_block`, targeting the namespace symbol's children.
- `emit_class` and `emit_interface` call `extract_members` which is flat (no recursion)
  — leave these as-is or inline them; either is fine.
- `extract_ambient` may call `extract_symbols` — convert to push a stack frame.
- `extract_arrow_fns` is flat — leave as-is.

**Checkpoint**: Run `cargo test -- typescript` (TypeScript parser tests only).

### Step 2.4 — Full validation gate
```
cargo test
cargo bench --no-run   # verify bench compilation
```
All 271+ tests must pass.

---

## Dependency Graph

```
Phase 1 (label: &'static str)
  1.1 → 1.2 → 1.3 → 1.4 → 1.5 → 1.6 → [1.7 gate]
                                              │
Phase 2 (iterative DFS)                       │
  2.1 (rust)    ──────────────────────────────┤
  2.2 (python)  ──────────────────────────────┤  (each independent of the others)
  2.3 (ts)      ──────────────────────────────┤
                                              │
                                         [2.4 gate]
```

Phase 1 and Phase 2 are **fully independent** — either can be executed first, or in
parallel branches. Within Phase 2, each parser conversion is **independent** of the other
two and can be executed in any order.

---

## Risk Assessment

| Task | Risk | Mitigation |
|---|---|---|
| Phase 1 (label) | Low | Mechanical type change; type errors are exhaustive; tests catch regressions |
| Phase 2 Rust | Low | Depth-1 recursion only; straightforward stack loop |
| Phase 2 Python | **N/A — deferred** | Two-pass DP adds allocations and a non-obvious ordering invariant in exchange for eliminating a risk that does not exist. Current implementation retained. |
| Phase 2 TypeScript | Medium | `emit_namespace` re-entry + `extract_ambient` re-entry; test suite covers all node types |

### Python Risk Revision

The original "Medium" risk for Python was based on the mutual recursion structure. Layer 2
analysis found:
- **Runtime risk is negligible**: human-authored Python files nest at most 10-15 levels deep
- **Architectural risk is real**: mutual recursion complicates future modifications
- **The DP strategy directly addresses the architectural risk** without adding complexity
  — two-pass post-order is cleaner than the single-pass `Walk`/`Compound` frame enum
  originally proposed
- **Borrow safety**: Strategy C avoids the `parent_idx` invalidation hazard present
  in Strategy A (index into `Vec<SymbolNode>` is invalidated on reallocation)

---

## Execution Recommendation

1. Run Phase 1 end-to-end first (`cargo test` gate at 1.7).
2. Run Phase 2 parser-by-parser: Rust → TypeScript. Python parser is NOT converted (see Step 2.2 decision).
3. After each parser: run that parser's test subset before moving on.
4. Full `cargo test` + `cargo bench --no-run` at Phase 2.4 gate.
