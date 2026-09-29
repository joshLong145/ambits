//! Text for the screen: wrapped to a width, control characters made safe.

use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

/// Tabs as four spaces, other control characters as one.
pub fn clean(s: &str) -> String {
    s.chars().flat_map(|c| if c == '\t' { vec![' '; 4] } else if c.is_control() { vec![' '] } else { vec![c] }).collect()
}

/// Lines being wrapped: the first `keep` made, the rest only counted.
struct Lines {
    room: usize,
    keep: usize,
    out: Vec<String>,
    line: String,
    used: usize,
    total: usize,
}

impl Lines {
    fn newline(&mut self) {
        if self.total <= self.keep {
            self.out.push(self.line.trim_end().to_string());
        }
        self.line.clear();
        self.total += 1;
        self.used = 0;
    }

    fn push(&mut self, c: char) {
        let w = c.width().unwrap_or(0);
        if self.used + w > self.room && self.used > 0 {
            self.newline();
            // A space that ends a full line is the break itself.
            if c == ' ' {
                return;
            }
        }
        if self.total <= self.keep {
            self.line.push(c);
        }
        self.used += w;
    }
}

/// `s` in lines at most `room` columns wide, broken between words — a
/// word wider than a line is broken where it must be — with trailing
/// spaces dropped. Only the first `keep` lines are made; the second value
/// is how many there are in all.
pub fn wrap(s: &str, room: usize, keep: usize) -> (Vec<String>, usize) {
    let mut lines = Lines { room: room.max(1), keep, out: Vec::new(), line: String::new(), used: 0, total: 1 };
    for word in s.split_inclusive(' ') {
        // A word that fits on a line of its own starts one, not breaks.
        let w = word.trim_end().width();
        if lines.used + w > lines.room && lines.used > 0 && w <= lines.room {
            lines.newline();
        }
        word.chars().for_each(|c| lines.push(c));
    }
    if lines.total <= lines.keep {
        let last = lines.line.trim_end().to_string();
        lines.out.push(last);
    }
    (lines.out, lines.total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_between_words_and_breaks_a_long_one() {
        assert_eq!(wrap("you are an expert reviewer", 12, usize::MAX), (vec!["you are an".into(), "expert".into(), "reviewer".into()], 3));
        assert_eq!(wrap("abcdefghij k", 8, usize::MAX).0, vec!["abcdefgh", "ij k"]);
        assert_eq!(wrap("", 8, usize::MAX), (vec![String::new()], 1));
    }

    #[test]
    fn keeps_only_what_it_is_asked_for_and_counts_the_rest() {
        assert_eq!(wrap("a b c d e f", 2, 2), (vec!["a".into(), "b".into()], 6));
    }

    #[test]
    fn measures_by_display_width() {
        assert_eq!(wrap("日本語 日本", 6, usize::MAX).0, vec!["日本語", "日本"]);
    }
}
