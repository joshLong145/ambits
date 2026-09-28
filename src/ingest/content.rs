//! What a call read or wrote, as its agent's log recorded it: a write's
//! change as diff hunks, a read's text with its line numbers, anything
//! else's output. For display only — loaded on demand, held in memory, and
//! never written anywhere (spec §9.6).

use std::borrow::Cow;

/// Lines of text or output kept, at most: a read is capped by its tool
/// long before this, a command's output need not be.
pub const MAX_LINES: usize = 5000;

/// Which content a call has, from what the trace knows of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContentKind {
    Read,
    Write,
    Other,
}

/// A call's content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallContent {
    /// A change to a file. `exact` when the hunks are the tool's own patch,
    /// with line numbers; otherwise they are what the call asked for (its
    /// input), which is all the log has when it failed or its tool records
    /// no patch.
    Change { hunks: Vec<Hunk>, exact: bool },
    /// Text a call read, its first line numbered `start`; `cut` lines more
    /// were left out.
    Read { start: u32, lines: Vec<String>, cut: usize },
    /// What a call returned; `cut` lines more were left out.
    Output { lines: Vec<String>, cut: usize },
}

/// One hunk of a change. Line numbers are 1-based; `None` when unknown.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hunk {
    pub old_start: Option<u32>,
    pub new_start: Option<u32>,
    pub lines: Vec<DiffLine>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffLine {
    Same(String),
    Removed(String),
    Added(String),
}

/// What a row of content is, for its colour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowKind {
    /// `@@ -552,7 +552,7 @@`.
    Hunk,
    Same,
    Removed,
    Added,
    /// A line read or output.
    Text,
    /// A note: lines left out.
    Note,
}

/// One screen row of content, with its line numbers in the old and new
/// file (a read's are in `new`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContentRow<'a> {
    pub kind: RowKind,
    pub old: Option<u32>,
    pub new: Option<u32>,
    pub text: Cow<'a, str>,
}

impl<'a> ContentRow<'a> {
    fn new(kind: RowKind, old: Option<u32>, new: Option<u32>, text: impl Into<Cow<'a, str>>) -> Self {
        Self { kind, old, new, text: text.into() }
    }
}

impl Hunk {
    /// `@@ -552,7 +552,7 @@`, or `@@ @@` with no line numbers.
    pub fn header(&self) -> String {
        let count = |f: fn(&DiffLine) -> bool| self.lines.iter().filter(|l| f(l)).count();
        let old = count(|l| !matches!(l, DiffLine::Added(_)));
        let new = count(|l| !matches!(l, DiffLine::Removed(_)));
        match (self.old_start, self.new_start) {
            (Some(o), Some(n)) => format!("@@ -{o},{old} +{n},{new} @@"),
            _ => format!("@@ -{old} +{new} lines @@"),
        }
    }
}

impl CallContent {
    /// Its rows, as a view draws them: each hunk's header, then its lines
    /// numbered; a read's lines numbered; output as it is.
    pub fn rows(&self) -> Vec<ContentRow<'_>> {
        let mut out = Vec::new();
        match self {
            CallContent::Change { hunks, .. } => {
                for hunk in hunks {
                    out.push(ContentRow::new(RowKind::Hunk, None, None, hunk.header()));
                    let (mut old, mut new) = (hunk.old_start, hunk.new_start);
                    let step = |n: &mut Option<u32>| {
                        let at = *n;
                        *n = n.map(|n| n + 1);
                        at
                    };
                    for line in &hunk.lines {
                        out.push(match line {
                            DiffLine::Same(t) => ContentRow::new(RowKind::Same, step(&mut old), step(&mut new), t.as_str()),
                            DiffLine::Removed(t) => ContentRow::new(RowKind::Removed, step(&mut old), None, t.as_str()),
                            DiffLine::Added(t) => ContentRow::new(RowKind::Added, None, step(&mut new), t.as_str()),
                        });
                    }
                }
            }
            CallContent::Read { start, lines, cut } => {
                out.extend(lines.iter().zip(*start..).map(|(t, n)| ContentRow::new(RowKind::Text, None, Some(n), t.as_str())));
                out.extend(Self::cut_note(*cut));
            }
            CallContent::Output { lines, cut } => {
                out.extend(lines.iter().map(|t| ContentRow::new(RowKind::Text, None, None, t.as_str())));
                out.extend(Self::cut_note(*cut));
            }
        }
        out
    }

    fn cut_note(cut: usize) -> Option<ContentRow<'static>> {
        (cut > 0).then(|| ContentRow::new(RowKind::Note, None, None, format!("… {cut} more lines not kept")))
    }

    /// The rows each hunk starts on, for stepping between them.
    pub fn hunk_rows(&self) -> Vec<usize> {
        self.rows().iter().enumerate().filter(|(_, r)| r.kind == RowKind::Hunk).map(|(i, _)| i).collect()
    }

    /// Lines as text, capped at [`MAX_LINES`]: the kept lines and how many
    /// more there were.
    pub fn capped(text: &str) -> (Vec<String>, usize) {
        let all: Vec<&str> = text.lines().collect();
        let cut = all.len().saturating_sub(MAX_LINES);
        (all.into_iter().take(MAX_LINES).map(String::from).collect(), cut)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_hunk_numbers_its_lines_in_both_files() {
        let c = CallContent::Change {
            hunks: vec![Hunk {
                old_start: Some(10),
                new_start: Some(10),
                lines: vec![DiffLine::Same("a".into()), DiffLine::Removed("b".into()), DiffLine::Added("c".into()), DiffLine::Added("d".into()), DiffLine::Same("e".into())],
            }],
            exact: true,
        };
        let rows = c.rows();
        assert_eq!(rows[0].text, "@@ -10,3 +10,4 @@");
        let numbers: Vec<(RowKind, Option<u32>, Option<u32>)> = rows[1..].iter().map(|r| (r.kind, r.old, r.new)).collect();
        assert_eq!(
            numbers,
            vec![
                (RowKind::Same, Some(10), Some(10)),
                (RowKind::Removed, Some(11), None),
                (RowKind::Added, None, Some(11)),
                (RowKind::Added, None, Some(12)),
                (RowKind::Same, Some(12), Some(13)),
            ]
        );
        assert_eq!(c.hunk_rows(), vec![0]);
    }

    #[test]
    fn a_hunk_without_numbers_says_how_many_lines() {
        let h = Hunk { old_start: None, new_start: None, lines: vec![DiffLine::Removed("x".into()), DiffLine::Added("y".into()), DiffLine::Added("z".into())] };
        assert_eq!(h.header(), "@@ -1 +2 lines @@");
    }

    #[test]
    fn a_read_is_numbered_from_its_start_and_says_what_was_cut() {
        let c = CallContent::Read { start: 40, lines: vec!["x".into(), "y".into()], cut: 3 };
        let rows = c.rows();
        assert_eq!((rows[0].new, rows[1].new), (Some(40), Some(41)));
        assert_eq!(rows[2].kind, RowKind::Note);
    }
}
