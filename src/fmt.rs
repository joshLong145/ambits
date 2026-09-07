//! Compact human-readable magnitudes.
//!
//! These had drifted into three copies of one function. Two in `src/ui` were
//! byte-identical; the one in `digest.rs` had lost its megabyte branch, so a
//! file group worth two million tokens rendered as `2000.0k`. Divergence
//! between copies is the failure mode this module exists to prevent, not the
//! duplication itself.

/// Token counts: `847`, `12.4k`, `2.1M`.
pub fn tokens(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

/// Byte counts: `512 B`, `1.5 KB`, `2.0 MB`.
///
/// Binary multiples, matching what a file manager reports for the same file.
pub fn bytes(n: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    if n >= MB {
        format!("{:.1} MB", n as f64 / MB as f64)
    } else if n >= KB {
        format!("{:.1} KB", n as f64 / KB as f64)
    } else {
        format!("{n} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_scale_at_each_threshold() {
        assert_eq!(tokens(0), "0");
        assert_eq!(tokens(999), "999");
        assert_eq!(tokens(1_000), "1.0k");
        assert_eq!(tokens(999_999), "1000.0k");
        assert_eq!(tokens(1_000_000), "1.0M");
    }

    /// The regression the consolidation fixes: the digest's copy lacked this
    /// branch, so large groups rendered in thousands without limit.
    #[test]
    fn millions_do_not_render_as_thousands() {
        assert!(tokens(2_400_000).ends_with('M'));
    }

    #[test]
    fn bytes_scale_at_each_threshold() {
        assert_eq!(bytes(0), "0 B");
        assert_eq!(bytes(1023), "1023 B");
        assert_eq!(bytes(1024), "1.0 KB");
        assert_eq!(bytes(1024 * 1024), "1.0 MB");
    }
}
