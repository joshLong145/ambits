#!/usr/bin/env python3
"""
Parse divan benchmark stdout into github-action-benchmark's customSmallerIsBetter JSON.

Divan outputs a tree-style table like:

    merkle_hash           fastest  │ slowest  │ median   │ mean
    ╰─ content_hash
       ├─ tiny    1.23 µs │ 2.3 µs │ 1.5 µs  │ 1.6 µs  │ 100 │ 1000
       ╰─ small   10.2 µs │ 15.3 µs│ 11.5 µs │ 12 µs   │ 100 │ 100

This script extracts the median time for each leaf entry, converts to nanoseconds,
and writes a JSON array suitable for github-action-benchmark's customSmallerIsBetter format:

    [{"name": "merkle_hash/content_hash/tiny", "unit": "ns", "value": 1500}, ...]

Usage:
    cargo bench 2>&1 | tee bench-output.txt
    python3 .github/scripts/parse_divan_bench.py bench-output.txt > bench-results.json

    # With static threshold checking (fails with exit code 1 if any threshold exceeded):
    python3 .github/scripts/parse_divan_bench.py bench-output.txt \\
        --thresholds .github/bench-thresholds.json > bench-results.json
"""

import argparse
import json
import re
import sys

# Matches a time value like "1.23 µs", "456 ns", "2.1 ms"
TIME_RE = re.compile(r"([\d.]+)\s*(ns|µs|us|ms|s)\b")

# Unicode box-drawing characters used by divan
BRANCH_CHARS = {"├", "╰", "│", "─", "╮", "╭"}


def to_ns(value: float, unit: str) -> float:
    """Convert a time value to nanoseconds."""
    factors = {"ns": 1, "µs": 1_000, "us": 1_000, "ms": 1_000_000, "s": 1_000_000_000}
    return value * factors.get(unit, 1)


def strip_tree_prefix(line: str) -> tuple[int, str]:
    """
    Remove divan's tree-drawing prefix characters and return (indent_level, clean_text).
    Indent level is determined by how many leading whitespace/box chars there are.
    """
    # Count leading whitespace to estimate nesting depth
    stripped = line.lstrip()
    indent = len(line) - len(stripped)
    # Remove box-drawing characters from the start of the stripped text
    clean = stripped.lstrip("".join(BRANCH_CHARS) + " ")
    return indent, clean


def extract_median(line: str) -> float | None:
    """
    Extract the median column (3rd │-delimited timing value) from a divan data row.
    Returns the value in nanoseconds, or None if the line has no timing data.
    """
    if "│" not in line:
        return None
    # Split by │ — columns are: fastest | slowest | median | mean | samples | iters
    parts = line.split("│")
    if len(parts) < 3:
        return None
    median_col = parts[2].strip()
    m = TIME_RE.search(median_col)
    if not m:
        return None
    return to_ns(float(m.group(1)), m.group(2))


def parse_bench_output(text: str) -> list[dict]:
    """
    Parse the full divan stdout and return a list of benchmark result dicts.
    """
    results = []
    # Stack tracks (indent, name) pairs for hierarchy reconstruction
    bench_file = None   # top-level name (e.g. "merkle_hash")

    # We track a simple path stack: [(indent, label), ...]
    path_stack: list[tuple[int, str]] = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line:
            continue

        # Detect a new bench binary header: a line without │ and no tree chars at column 0
        # These look like "merkle_hash           fastest  │ ..."  (has │ in header)
        # or just "merkle_hash" with the column headers on the same line
        if "fastest" in line and "slowest" in line:
            # This is the column header row — the bench name is before the first space/tab
            header_name = line.split()[0].strip().lstrip("".join(BRANCH_CHARS) + " ")
            if header_name:
                bench_file = header_name
            path_stack = []
            continue

        # Skip pure separator lines
        if all(c in "─│╰├╭╮ \t" for c in line):
            continue

        # Parse indent level and clean label
        indent, clean = strip_tree_prefix(line)

        # Does this line have timing data?
        median_ns = extract_median(clean)

        # Extract the label (the part before any │)
        label_part = clean.split("│")[0].strip() if "│" in clean else clean.strip()

        if not label_part:
            continue

        # Update path stack: pop entries at same or deeper indent
        while path_stack and path_stack[-1][0] >= indent:
            path_stack.pop()

        if median_ns is not None:
            # This is a leaf data row — build the full name
            parts = [bench_file] if bench_file else []
            parts += [name for (_, name) in path_stack]
            parts.append(label_part)
            name = "/".join(filter(None, parts))
            results.append({"name": name, "unit": "ns", "value": round(median_ns)})
        else:
            # Structural node — push onto stack
            path_stack.append((indent, label_part))

    return results


def check_thresholds(results: list[dict], thresholds: dict) -> int:
    """
    Compare benchmark results against static thresholds.

    Prints a pass/fail table to stderr. Returns the number of failures.
    Only benchmarks that appear in the thresholds dict are checked;
    extras are silently ignored.
    """
    failures = 0
    checked = [(r, thresholds[r["name"]]) for r in results if r["name"] in thresholds]

    if not checked:
        print("Warning: no benchmark names matched the threshold file", file=sys.stderr)
        return 0

    col_w = max(len(r["name"]) for r, _ in checked)
    sep = "─" * (col_w + 44)

    print(f"── Static Threshold Check {sep[26:]}", file=sys.stderr)
    for result, limit_ns in checked:
        actual = result["value"]
        status = "PASS" if actual <= limit_ns else "FAIL"
        if status == "FAIL":
            failures += 1
        print(
            f"  {status}  {result['name']:<{col_w}}  {actual:>12,} ns / {limit_ns:>12,} ns",
            file=sys.stderr,
        )
    print(sep, file=sys.stderr)

    if failures:
        print(f"{failures} benchmark(s) exceeded their threshold.", file=sys.stderr)
    else:
        print("All benchmarks within threshold.", file=sys.stderr)

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse divan bench output → customSmallerIsBetter JSON"
    )
    parser.add_argument("input_file", help="Path to captured divan stdout (bench-output.txt)")
    parser.add_argument(
        "--thresholds",
        metavar="FILE",
        default=None,
        help="JSON file mapping benchmark names to ns limits; fails with exit 1 if exceeded",
    )
    args = parser.parse_args()

    with open(args.input_file, encoding="utf-8") as f:
        text = f.read()

    results = parse_bench_output(text)

    if not results:
        print("Warning: no benchmark results found in output", file=sys.stderr)

    # Always emit JSON to stdout (piped to bench-results.json in CI)
    print(json.dumps(results, indent=2))

    # Optional static threshold check — runs after JSON is emitted
    if args.thresholds:
        with open(args.thresholds, encoding="utf-8") as f:
            thresholds = json.load(f)
        failures = check_thresholds(results, thresholds)
        if failures:
            sys.exit(1)


if __name__ == "__main__":
    main()
