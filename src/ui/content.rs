//! A call's content — a write's change as a diff, a read's text, a
//! command's output — as the trace panel previews it and `o` shows it in
//! full. From the agent's log, in memory only (spec §9.6).

use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;

use ambits::app::{App, ContentState};
use ambits::ingest::content::{clean, CallContent, CallDetail, ContentRow, RowKind};

use super::inspector::text;
use super::trace_view::span_name;
use super::fit;

/// Rows of content the trace panel previews.
const PREVIEW: usize = 12;

/// Rows of one argument the trace panel shows.
const ARG_PREVIEW: usize = 4;

/// Line-number columns: one (the new file's, else the old's) in the
/// narrow panel, both in the full view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Numbers {
    One,
    Both,
}

/// How rows line up: their line-number columns and the arguments' key
/// column.
#[derive(Debug, Clone, Copy)]
struct Columns {
    numbers: Numbers,
    digits: usize,
    key: usize,
}

impl Columns {
    fn new(numbers: Numbers, detail: &CallDetail, rows: &[ContentRow<'_>]) -> Self {
        let digits = rows.iter().flat_map(|r| [r.old, r.new]).flatten().max().map_or(1, |n| n.to_string().len());
        Self { numbers, digits, key: detail.key_width() }
    }

    /// The number columns a diff row shows: in the panel, the new file's
    /// number, else the old's; in full, both.
    fn gutter(&self, old: Option<u32>, new: Option<u32>) -> String {
        let digits = self.digits;
        let num = |n: Option<u32>| n.map_or_else(|| " ".repeat(digits), |n| format!("{n:>digits$}"));
        match (self.numbers, new) {
            (Numbers::Both, _) => format!("{} {}", num(old), num(new)),
            (Numbers::One, None) => num(old),
            (Numbers::One, new) => num(new),
        }
    }
}

fn row_line(row: &ContentRow<'_>, cols: Columns, width: usize) -> Line<'static> {
    let dim = Color::DarkGray;
    let key = |k: Option<&str>| format!("{:<w$}  ", fit(k.unwrap_or(""), cols.key), w = cols.key);
    let (lead, sign, color) = match row.kind {
        RowKind::Section => return Line::from(Span::styled(row.text.to_string(), Style::default().fg(dim).add_modifier(Modifier::BOLD))),
        RowKind::Hunk => return Line::from(text(fit(&row.text, width), Color::Cyan)),
        RowKind::Note => return Line::from(text(fit(&row.text, width), dim)),
        RowKind::Arg | RowKind::ArgMore => return Line::from(vec![text(key(row.key), Color::Cyan), text(row.text.to_string(), Color::White)]),
        RowKind::ArgBulk => return Line::from(vec![text(key(row.key), Color::Cyan), text(fit(&row.text, width.saturating_sub(cols.key + 2)), dim)]),
        RowKind::Text if row.new.is_none() => return Line::from(text(fit(&clean(&row.text), width), Color::Gray)),
        RowKind::Text => (format!("{:>w$}", row.new.map_or(String::new(), |n| n.to_string()), w = cols.digits), "│", Color::Gray),
        RowKind::Same => (cols.gutter(row.old, row.new), " ", Color::Gray),
        RowKind::Removed => (cols.gutter(row.old, None), "-", Color::Red),
        RowKind::Added => (cols.gutter(None, row.new), "+", Color::Green),
    };
    let lead = format!("{lead} ");
    let room = width.saturating_sub(super::width(&lead) + 2);
    Line::from(vec![text(lead, dim), text(format!("{sign} "), color), text(fit(&clean(&row.text), room), color)])
}

/// A note above a change the log had no patch for: what the hunks are.
fn note(app: &App, span: usize, detail: &CallDetail) -> Option<Line<'static>> {
    let Some(CallContent::Change { exact: false, .. }) = &detail.content else { return None };
    let s = &app.trace.spans()[span];
    Some(Line::from(if s.error {
        text("✗ not applied — what it tried", Color::Red)
    } else if s.end.is_none() {
        text("running — what it asked for", Color::DarkGray)
    } else {
        text("what it asked for — the log has no patch", Color::DarkGray)
    }))
}

/// Whether the trace panel has `span`'s arguments to show, in place of
/// its one-line description.
pub(super) fn has_args(app: &App, span: usize) -> bool {
    matches!(app.content_state(span), ContentState::Loaded(d) if !d.args.is_empty())
}

/// The trace panel's view of call `span` as its log recorded it: its
/// arguments, a few lines of each, then the first rows of its content and
/// how to see the rest. Nothing for a call the log has nothing on.
pub(super) fn preview(app: &App, span: usize, width: usize) -> Vec<Line<'static>> {
    let detail = match app.content_state(span) {
        ContentState::Loaded(d) => d,
        ContentState::Loading => return vec![Line::from(""), Line::from(text(" loading…", Color::DarkGray))],
        ContentState::Missing => return Vec::new(),
    };
    let width = width.saturating_sub(3);
    let heading = |name: &str| Line::from(Span::styled(format!(" {name}"), Style::default().fg(Color::DarkGray).add_modifier(Modifier::BOLD)));
    let mut out = Vec::new();
    let args = detail.arg_rows(width);
    let cols = Columns::new(Numbers::One, detail, &[]);
    if !args.is_empty() {
        out.push(Line::from(""));
        out.push(heading("arguments"));
        // Each argument's first rows, and how many more it has.
        let mut shown = 0;
        let mut hidden = 0;
        for (n, row) in args.iter().enumerate() {
            if row.kind != RowKind::ArgMore {
                shown = 0;
            }
            shown += 1;
            if shown <= ARG_PREVIEW {
                out.push(indent(row_line(row, cols, width)));
            } else {
                hidden += 1;
            }
            let last_of_arg = args.get(n + 1).is_none_or(|next| next.kind != RowKind::ArgMore);
            if last_of_arg && hidden > 0 {
                out.push(indent(Line::from(text(format!("{:w$}  … {hidden} more lines", "", w = cols.key), Color::DarkGray))));
                hidden = 0;
            }
        }
    }
    let mut more = 0;
    if let Some(content) = &detail.content {
        let rows = content.rows();
        let cols = Columns::new(Numbers::One, detail, &rows);
        out.push(Line::from(""));
        out.push(heading(content.title()));
        out.extend(note(app, span, detail).map(indent));
        out.extend(rows.iter().take(PREVIEW).map(|r| indent(row_line(r, cols, width))));
        more = rows.len().saturating_sub(PREVIEW);
    }
    if !out.is_empty() {
        let hint = if more > 0 { format!(" … {more} more · o opens in full") } else { " o opens in full".to_string() };
        out.push(Line::from(text(hint, Color::DarkGray)));
    }
    out
}

fn indent(line: Line<'static>) -> Line<'static> {
    let mut spans = vec![Span::raw("   ")];
    spans.extend(line.spans);
    Line::from(spans)
}

/// The content view (`o`): call `span` as its log recorded it — its
/// arguments, then its content — over the whole main area, scrolled.
pub(super) fn render_view(f: &mut Frame, app: &App, area: Rect) {
    let Some(view) = &app.content_view else { return };
    let s = &app.trace.spans()[view.span];
    let title = format!(" {} · {} · {} ", span_name(app, view.span), app.agent_title(&s.agent), ambits::time::clock(s.start));
    let block = Block::default()
        .title(fit(&title, area.width.saturating_sub(4) as usize))
        .title_bottom(Line::from(text(" j/k scroll · n/N hunk · g/G ends · o/Esc close ", Color::DarkGray)))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD));
    let inner = block.inner(area);
    f.render_widget(Clear, area);
    f.render_widget(block, area);
    // A column of margin either side.
    let inner = Rect { x: inner.x + 1, width: inner.width.saturating_sub(2), ..inner };

    let width = inner.width as usize;
    let detail = match app.content_state(view.span) {
        ContentState::Loaded(d) => d,
        ContentState::Loading => return f.render_widget(Paragraph::new(text(" loading…", Color::DarkGray)), inner),
        ContentState::Missing => return f.render_widget(Paragraph::new(text(" The log has nothing on this call.", Color::DarkGray)), inner),
    };
    let mut lines: Vec<Line<'static>> = note(app, view.span, detail).into_iter().collect();
    let height = (inner.height as usize).saturating_sub(lines.len());
    view.height.set(height);
    view.width.set(width);
    let rows = detail.rows(width);
    let cols = Columns::new(Numbers::Both, detail, &rows);
    let first = view.scroll.min(rows.len().saturating_sub(height));
    lines.extend(rows.iter().skip(first).take(height).map(|r| row_line(r, cols, width)));
    f.render_widget(Paragraph::new(lines), inner);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ambits::ingest::content::{DiffLine, Hunk};

    fn change() -> CallContent {
        CallContent::Change {
            hunks: vec![Hunk {
                old_start: Some(9),
                new_start: Some(9),
                lines: vec![DiffLine::Same("keep".into()), DiffLine::Removed("\told".into()), DiffLine::Added("\tnew".into())],
            }],
            exact: true,
        }
    }

    fn lines(content: &CallContent, numbers: Numbers) -> Vec<String> {
        let rows = content.rows();
        let detail = CallDetail { args: Vec::new(), content: None };
        let cols = Columns::new(numbers, &detail, &rows);
        rows.iter().map(|r| row_line(r, cols, 40).spans.iter().map(|s| s.content.to_string()).collect()).collect()
    }

    #[test]
    fn a_diff_shows_signs_and_line_numbers() {
        assert_eq!(lines(&change(), Numbers::Both), vec!["@@ -9,2 +9,2 @@", " 9  9   keep", "10    -     old", "   10 +     new"]);
        assert_eq!(lines(&change(), Numbers::One), vec!["@@ -9,2 +9,2 @@", " 9   keep", "10 -     old", "10 +     new"]);
    }

    #[test]
    fn a_read_is_numbered_and_output_is_not() {
        let read = CallContent::Read { start: 99, lines: vec!["a".into(), "b".into()], cut: 0 };
        assert_eq!(lines(&read, Numbers::One), vec![" 99 │ a", "100 │ b"]);
        let out = CallContent::Output { lines: vec!["ok".into()], cut: 0 };
        assert_eq!(lines(&out, Numbers::One), vec!["ok"]);
    }
}
