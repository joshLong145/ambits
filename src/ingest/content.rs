//! What a call read or wrote, as its agent's log recorded it: a write's
//! change as diff hunks, a read's text with its line numbers, anything
//! else's output. For display only — loaded on demand, held in memory, and
//! never written anywhere (spec §9.6).

use std::borrow::Cow;

use serde_json::Value;

/// Lines of text or output kept, at most: a read is capped by its tool
/// long before this, a command's output need not be.
pub const MAX_LINES: usize = 5000;

/// A call as its log recorded it: the arguments it was called with, and
/// what it read, wrote or returned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallDetail {
    pub args: Vec<Arg>,
    pub content: Option<CallContent>,
}

/// One argument of a call, for display.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Arg {
    pub key: String,
    /// Its value as text: a string as it is (lines and all), a list one
    /// item a line, anything else as compact JSON.
    pub value: String,
    /// File contents the call wrote (`old_string`, `content`, …): not
    /// repeated here, since the change shows them.
    pub bulk: bool,
}

/// Arguments that hold file contents: summed up, not shown, as the change
/// already shows them.
const BULK: &[&str] = &["old_string", "new_string", "content", "body", "new_source", "edits"];

/// Arguments to show first, in this order: what the call is for, then
/// what it acts on.
const FIRST: &[&str] = &[
    "description", "subagent_type", "command", "file_path", "notebook_path", "relative_path", "path", "name_path", "name_path_pattern",
    "pattern", "glob", "url", "query", "prompt",
];

/// A call's arguments from its input, in a reading order: what it is for
/// and what it acts on first, file contents last.
pub fn args(input: &Value) -> Vec<Arg> {
    let Value::Object(map) = input else { return Vec::new() };
    let mut out: Vec<Arg> = map
        .iter()
        .filter(|(_, v)| !v.is_null())
        .map(|(key, v)| {
            let bulk = BULK.contains(&key.as_str());
            let value = match v {
                Value::String(s) if bulk => match s.lines().count() {
                    0 | 1 => "1 line · in the change below".to_string(),
                    n => format!("{n} lines · in the change below"),
                },
                Value::Array(a) if bulk => format!("{} edits · in the change below", a.len()),
                Value::String(s) => s.clone(),
                Value::Array(items) => items.iter().map(|i| i.as_str().map_or_else(|| i.to_string(), String::from)).collect::<Vec<_>>().join("\n"),
                other => other.to_string(),
            };
            Arg { key: key.clone(), value, bulk }
        })
        .collect();
    let rank = |a: &Arg| (a.bulk, FIRST.iter().position(|k| *k == a.key).unwrap_or(FIRST.len()));
    out.sort_by(|a, b| rank(a).cmp(&rank(b)).then_with(|| a.key.cmp(&b.key)));
    out
}

/// The widest an argument's key column gets; a longer key is cut.
const KEY_MAX: usize = 16;

/// `s` in lines at most `room` columns wide, broken between words; a word
/// wider than a line is broken where it must be.
fn wrap(s: &str, room: usize) -> Vec<String> {
    use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};
    let mut out = vec![String::new()];
    let mut used = 0;
    for word in s.split_inclusive(' ') {
        let w = word.width();
        if used + w > room && used > 0 {
            out.push(String::new());
            used = 0;
        }
        if w <= room {
            out.last_mut().expect("never empty").push_str(word);
            used += w;
            continue;
        }
        for c in word.chars() {
            let w = c.width().unwrap_or(0);
            if used + w > room && used > 0 {
                out.push(String::new());
                used = 0;
            }
            out.last_mut().expect("never empty").push(c);
            used += w;
        }
    }
    out
}

/// Tabs as four spaces, other control characters as one.
pub fn clean(s: &str) -> String {
    s.chars().flat_map(|c| if c == '\t' { vec![' '; 4] } else if c.is_control() { vec![' '] } else { vec![c] }).collect()
}

impl CallDetail {
    /// The width of the arguments' key column.
    pub fn key_width(&self) -> usize {
        self.args.iter().map(|a| a.key.chars().count()).max().unwrap_or(0).min(KEY_MAX)
    }

    /// Its arguments' rows for a view `width` columns wide: each key, its
    /// value beside it, wrapped under it.
    pub fn arg_rows(&self, width: usize) -> Vec<ContentRow<'_>> {
        let room = width.saturating_sub(self.key_width() + 2).max(8);
        let mut out = Vec::new();
        for arg in &self.args {
            if arg.bulk {
                out.push(ContentRow { key: Some(&arg.key), ..ContentRow::new(RowKind::ArgBulk, None, None, arg.value.as_str()) });
                continue;
            }
            let lines: Vec<String> = arg.value.lines().chain(arg.value.is_empty().then_some("")).flat_map(|l| wrap(&clean(l), room)).collect();
            for (n, line) in lines.into_iter().enumerate() {
                let kind = if n == 0 { RowKind::Arg } else { RowKind::ArgMore };
                out.push(ContentRow { key: (n == 0).then_some(arg.key.as_str()), ..ContentRow::new(kind, None, None, line) });
            }
        }
        out
    }

    /// Its rows, as the content view draws them `width` columns wide: its
    /// arguments, then its content, each under a heading.
    pub fn rows(&self, width: usize) -> Vec<ContentRow<'_>> {
        let mut out = Vec::new();
        if !self.args.is_empty() {
            out.push(ContentRow::new(RowKind::Section, None, None, "arguments"));
            out.extend(self.arg_rows(width));
        }
        if let Some(content) = &self.content {
            if !out.is_empty() {
                out.push(ContentRow::new(RowKind::Note, None, None, ""));
            }
            out.push(ContentRow::new(RowKind::Section, None, None, content.title()));
            out.extend(content.rows());
        }
        out
    }

    /// The rows each hunk starts on, for stepping between them.
    pub fn hunk_rows(&self, width: usize) -> Vec<usize> {
        self.rows(width).iter().enumerate().filter(|(_, r)| r.kind == RowKind::Hunk).map(|(i, _)| i).collect()
    }
}

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
    /// An argument's first line, with its key.
    Arg,
    /// An argument's further lines.
    ArgMore,
    /// An argument holding file contents, summed up.
    ArgBulk,
    /// A heading: `arguments`, `change`, `read`, `output`.
    Section,
}

/// One screen row of content, with its line numbers in the old and new
/// file (a read's are in `new`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContentRow<'a> {
    pub kind: RowKind,
    pub old: Option<u32>,
    pub new: Option<u32>,
    pub text: Cow<'a, str>,
    /// An argument's key, on its first row.
    pub key: Option<&'a str>,
}

impl<'a> ContentRow<'a> {
    fn new(kind: RowKind, old: Option<u32>, new: Option<u32>, text: impl Into<Cow<'a, str>>) -> Self {
        Self { kind, old, new, text: text.into(), key: None }
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
    /// What its rows are headed.
    pub fn title(&self) -> &'static str {
        match self {
            CallContent::Change { .. } => "change",
            CallContent::Read { .. } => "read",
            CallContent::Output { .. } => "output",
        }
    }

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
        let detail = CallDetail { args: args(&serde_json::json!({"file_path": "a.rs", "old_string": "x\ny"})), content: Some(c) };
        let kinds: Vec<RowKind> = detail.rows(40).iter().map(|r| r.kind).take(6).collect();
        assert_eq!(kinds, [RowKind::Section, RowKind::Arg, RowKind::ArgBulk, RowKind::Note, RowKind::Section, RowKind::Hunk]);
        assert_eq!(detail.hunk_rows(40), vec![5]);
    }

    #[test]
    fn arguments_read_what_for_then_on_what_then_the_rest() {
        let input = serde_json::json!({
            "timeout": 600000, "command": "cargo test\n  -q", "description": "Run the tests", "run_in_background": false, "extra": null,
        });
        let got: Vec<(String, String)> = args(&input).into_iter().map(|a| (a.key, a.value)).collect();
        let want = [("description", "Run the tests"), ("command", "cargo test\n  -q"), ("run_in_background", "false"), ("timeout", "600000")];
        assert_eq!(got, want.map(|(k, v)| (k.to_string(), v.to_string())));
        let edit = args(&serde_json::json!({"file_path": "a.rs", "new_string": "a\nb\nc", "edits": [1, 2]}));
        assert_eq!(edit.iter().map(|a| (a.bulk, a.value.as_str())).collect::<Vec<_>>(), vec![
            (false, "a.rs"), (true, "2 edits · in the change below"), (true, "3 lines · in the change below"),
        ]);
    }

    #[test]
    fn a_long_value_wraps_under_its_key() {
        let d = CallDetail { args: args(&serde_json::json!({"command": "abcdefghij\tk"})), content: None };
        let rows: Vec<(RowKind, Option<&str>, String)> = d.arg_rows(7 + 2 + 8).into_iter().map(|r| (r.kind, r.key, r.text.into_owned())).collect();
        assert_eq!(rows, vec![
            (RowKind::Arg, Some("command"), "abcdefgh".to_string()),
            (RowKind::ArgMore, None, "ij    k".to_string()),
        ]);
    }

    #[test]
    fn values_wrap_between_words() {
        assert_eq!(wrap("you are an expert reviewer", 12), vec!["you are an ", "expert ", "reviewer"]);
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
