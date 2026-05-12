use sha2::{Digest, Sha256};

use super::SymbolNode;

/// Compute content hash from the raw source text of a symbol.
/// Normalizes whitespace to make hashing resilient to formatting changes.
///
/// For ASCII input — virtually all source code — we stream normalized bytes
/// directly into the SHA-256 hasher through a stack buffer, skipping the
/// intermediate `String` allocation and Unicode-aware classification used by
/// the general fallback. Both paths feed bit-for-bit identical byte sequences
/// to the hasher, so hash values are unchanged.
pub fn content_hash(source: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    if source.is_ascii() {
        hash_normalized_ascii(source.trim().as_bytes(), &mut hasher);
    } else {
        let normalized = normalize_source(source);
        hasher.update(normalized.as_bytes());
    }
    hasher.finalize().into()
}

/// Stream `bytes` into `hasher` after collapsing runs of ASCII whitespace into a
/// single space, matching `normalize_source` byte-for-byte. Caller must pre-trim.
#[inline]
fn hash_normalized_ascii(bytes: &[u8], hasher: &mut Sha256) {
    let mut buf = [0u8; 256];
    let mut len = 0;
    let mut prev_was_space = false;
    for &b in bytes {
        let push = if is_unicode_ws_ascii(b) {
            if prev_was_space {
                continue;
            }
            prev_was_space = true;
            b' '
        } else {
            prev_was_space = false;
            b
        };
        buf[len] = push;
        len += 1;
        if len == buf.len() {
            hasher.update(buf);
            len = 0;
        }
    }
    if len > 0 {
        hasher.update(&buf[..len]);
    }
}

/// Compute the Merkle hash for a symbol node.
/// Combines the node's own content hash with all children's Merkle hashes.
/// This must be called bottom-up (children first).
pub fn compute_merkle_hash(node: &mut SymbolNode) {
    // First, recursively compute children's merkle hashes.
    for child in node.children.iter_mut() {
        compute_merkle_hash(child);
    }

    let mut hasher = Sha256::new();
    hasher.update(node.content_hash);
    for child in &node.children {
        hasher.update(child.merkle_hash);
    }
    node.merkle_hash = hasher.finalize().into();
}

/// Normalize source code for hashing: collapse runs of whitespace into single spaces,
/// trim leading/trailing whitespace. This makes the hash resilient to formatting changes
/// while still detecting meaningful code changes.
fn normalize_source(source: &str) -> String {
    let trimmed = source.trim();
    let mut result = String::with_capacity(trimmed.len());
    let mut prev_was_space = false;

    for ch in trimmed.chars() {
        if ch.is_whitespace() {
            if !prev_was_space {
                result.push(' ');
                prev_was_space = true;
            }
        } else {
            result.push(ch);
            prev_was_space = false;
        }
    }

    result
}

/// Estimate the number of LLM tokens a source string would consume.
///
/// # Why word-boundary counting instead of a flat characters-per-token ratio
///
/// Most LLMs (including Claude and GPT-4) use Byte Pair Encoding (BPE) tokenisation.
/// BPE builds a vocabulary of frequently occurring byte sequences by iteratively merging
/// the most common adjacent pairs. For source code this produces two well-studied patterns:
///
/// 1. **Identifiers and keywords are split into subwords of ~4 characters on average.**
///    Short tokens like `fn`, `let`, `if`, `i32` map to a single vocabulary entry.
///    Longer names like `calculate_result` (16 chars) are split into roughly 4 subwords.
///    The `ceil(len / 4)` formula approximates this without a full vocabulary lookup.
///
/// 2. **Punctuation and operators are almost always single tokens.**
///    Characters like `{`, `}`, `(`, `)`, `;`, `->`, `<`, `>` each occupy one token
///    regardless of their role. Code is punctuation-dense, so a flat char-ratio formula
///    systematically undercounts by folding punctuation into the word budget.
///
/// # Why not a flat ratio (e.g. 3.5 chars/token)?
///
/// A flat ratio is accurate for natural-language prose where punctuation is sparse.
/// For code the ratio varies widely by language style: Rust/TypeScript signatures with
/// `->`, `::`, `<>`, and `()` produce far more punctuation tokens than the ratio
/// predicts. The word-boundary model handles both ends of the spectrum correctly.
///
/// # Limitations
///
/// This is still an approximation — the true count depends on the specific model's
/// vocabulary. It will be most accurate for English-identifier code (Rust, Python,
/// TypeScript) and less accurate for heavily symbolic code (e.g. APL, dense regex).
pub fn estimate_tokens(source: &str) -> usize {
    if source.is_ascii() {
        return estimate_tokens_ascii(source.as_bytes());
    }
    estimate_tokens_unicode(source)
}

#[inline]
fn estimate_tokens_ascii(bytes: &[u8]) -> usize {
    let mut count = 0usize;
    let mut word_len = 0usize;
    for &b in bytes {
        if b.is_ascii_alphanumeric() || b == b'_' {
            word_len += 1;
        } else {
            if word_len > 0 {
                count += word_len.div_ceil(4);
                word_len = 0;
            }
            if !is_unicode_ws_ascii(b) {
                count += 1;
            }
        }
    }
    if word_len > 0 {
        count += word_len.div_ceil(4);
    }
    count
}

fn estimate_tokens_unicode(source: &str) -> usize {
    let mut count = 0usize;
    let mut word_len = 0usize;

    for ch in source.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            word_len += 1;
        } else {
            if word_len > 0 {
                count += word_len.div_ceil(4);
                word_len = 0;
            }
            if !ch.is_whitespace() {
                count += 1; // operator, bracket, punctuation — each ~1 token
            }
        }
    }
    if word_len > 0 {
        count += word_len.div_ceil(4);
    }
    count
}

/// ASCII subset of Unicode's `White_Space` property: matches `char::is_whitespace()`
/// for any ASCII byte. `u8::is_ascii_whitespace()` excludes U+000B (vertical tab),
/// which Unicode treats as whitespace — we include it so the fast path is bit-for-bit
/// equivalent to the Unicode path for ASCII input.
#[inline]
fn is_unicode_ws_ascii(b: u8) -> bool {
    matches!(b, b'\t' | b'\n' | 0x0B | 0x0C | b'\r' | b' ')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_collapses_whitespace() {
        let input = "fn  foo(\n    x: i32,\n    y: i32\n) -> i32 {\n    x + y\n}";
        let normalized = normalize_source(input);
        assert_eq!(normalized, "fn foo( x: i32, y: i32 ) -> i32 { x + y }");
    }

    #[test]
    fn test_content_hash_deterministic() {
        let h1 = content_hash("fn foo() {}");
        let h2 = content_hash("fn foo() {}");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_content_hash_whitespace_insensitive() {
        let h1 = content_hash("fn foo() { }");
        let h2 = content_hash("fn  foo()  {  }");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_content_hash_detects_changes() {
        let h1 = content_hash("fn foo() {}");
        let h2 = content_hash("fn bar() {}");
        assert_ne!(h1, h2);
    }

    /// Reference implementation: hash the output of the existing Unicode
    /// `normalize_source` path. Used to assert the ASCII fast path is
    /// bit-for-bit equivalent.
    fn content_hash_via_unicode(source: &str) -> [u8; 32] {
        let normalized = normalize_source(source);
        let mut hasher = Sha256::new();
        hasher.update(normalized.as_bytes());
        hasher.finalize().into()
    }

    #[test]
    fn content_hash_ascii_path_matches_unicode_path() {
        let samples = [
            "",
            "fn foo() {}",
            "fn  foo()  {  }",
            "   \n\t  fn bar() {}\n\n",
            "let x = (a + b) * c;\nlet y = x * 2;",
            "a\x0Bb",  // vertical tab — Unicode whitespace, not ASCII whitespace
            include_str!("merkle.rs"),  // larger ASCII payload: this very file
        ];
        for s in samples {
            assert_eq!(
                content_hash(s),
                content_hash_via_unicode(s),
                "content_hash disagreement on {s:?}",
            );
        }
    }

    #[test]
    fn ascii_and_unicode_paths_agree_on_ascii_input() {
        // The fast path takes the ASCII branch; we manually compare against the
        // Unicode fallback for the same input.
        let samples = [
            "",
            "fn foo() {}",
            "calculate_result",
            "let x = (a + b) * c;",
            "  \n\t  ",
            "a\x0Bb",  // includes vertical tab — Unicode whitespace but not ASCII whitespace
        ];
        for s in samples {
            assert_eq!(
                estimate_tokens_ascii(s.as_bytes()),
                estimate_tokens_unicode(s),
                "estimate_tokens disagreement on {s:?}",
            );
        }
    }

    #[test]
    fn test_estimate_tokens() {
        // Short keywords are 1 token each; punctuation is 1 token each.
        // "fn foo() {}" → "fn"(1) + "foo"(1) + '('(1) + ')'(1) + '{'(1) + '}'(1) = 6
        assert_eq!(estimate_tokens("fn foo() {}"), 6);

        // Long identifier splits into subwords: "calculate_result" = 16 chars → ceil(16/4) = 4
        assert_eq!(estimate_tokens("calculate_result"), 4);

        // Short identifier stays as one token: "x" = 1 char → ceil(1/4) = 1
        assert_eq!(estimate_tokens("x"), 1);

        // Empty input → 0
        assert_eq!(estimate_tokens(""), 0);

        // Only whitespace → 0
        assert_eq!(estimate_tokens("   \n\t  "), 0);
    }
}
