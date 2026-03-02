use sha2::{Digest, Sha256};

use super::SymbolNode;

/// Compute content hash from the raw source text of a symbol.
/// Normalizes whitespace to make hashing resilient to formatting changes.
pub fn content_hash(source: &str) -> [u8; 32] {
    let normalized = normalize_source(source);
    let mut hasher = Sha256::new();
    hasher.update(normalized.as_bytes());
    hasher.finalize().into()
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
    let mut count = 0usize;
    let mut word_len = 0usize;

    for ch in source.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            word_len += 1;
        } else {
            if word_len > 0 {
                count += (word_len + 3) / 4;
                word_len = 0;
            }
            if !ch.is_whitespace() {
                count += 1; // operator, bracket, punctuation — each ~1 token
            }
        }
    }
    if word_len > 0 {
        count += (word_len + 3) / 4;
    }
    count
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
