# Benchmarks

- [ambits vs. Serena](https://github.com/joshLong145/ambits/blob/main/docs/benchmarks/ambits-vs-serena.md) —
  latency and output size for search, symbol lookup, references and structure
  overview, measured on this repository against Serena's rust-analyzer backend
  (2026-09-15).

Micro-benchmarks live in [`benches/`](https://github.com/joshLong145/ambits/tree/main/benches)
and run with:

```bash
cargo bench
```
