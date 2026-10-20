#!/usr/bin/env bash
# Fail unless the binary at $1 is statically linked and works: it prints its
# version and finds a symbol in a throwaway project.
set -euo pipefail
bin=$1
file "$bin"
# From the ELF, not file's wording, which says "static-pie linked" for a static PIE.
if readelf -lW "$bin" | grep -q 'Requesting program interpreter'; then echo "$bin needs a program interpreter" >&2; exit 1; fi
if readelf -dW "$bin" | grep -q '(NEEDED)'; then echo "$bin links shared libraries" >&2; exit 1; fi
"$bin" --version
dir=$(mktemp -d) && git -C "$dir" init -q && printf 'fn a() {}\n' > "$dir/a.rs"
"$bin" -p "$dir" --dump | grep -q 'fn a'
