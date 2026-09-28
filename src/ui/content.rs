//! A call's content — a write's change as a diff, a read's text, a
//! command's output — as the trace panel previews it and `o` shows it in
//! full. From the agent's log, in memory only (spec §9.6).

use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;

use ambits::app::{App, ContentState};
use ambits::ingest::content::{CallContent, ContentRow, RowKind};

use super::inspector::text;
use super::trace_view::span_name;
use super::fit;

/// Rows the trace panel previews.
const PREVIEW: usize = 12;

/// Line-number columns: one (the new file's, else the old's) in the
/// narrow panel, both in the full view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Numbers {
    One,
    Both,
}

/// Digits the widest line number in `rows` takes.
fn digits(rows: &[ContentRow<'_>]) -> usize {
    rows.iter().flat_map(|r| [r.old, r.new]).flatten().max().map_or(1, |n| n.to_string().len())
}

/// Tabs as four spaces, other control characters as one.
fn clean(s: &str) -> String {
    s.chars().flat_map(|c| if c == '\t' { vec![' '; 4] } else if c.is_control() { vec![' '] } else { vec![c] }).collect()
}

fn row_line(row: &ContentRow<'_>, numbers: Numbers, digits: usize, width: usize) -> Line<'static> {
    let num = |n: Option<u32>| n.map_or_else(|| " ".repeat(digits), |n| format!("{n:>digits$}"));
    let dim = Color::DarkGray;
    let (columns, sign, color) = match row.kind {
        RowKind::Hunk => return Line::from(text(fit(&row.text, width), Color::Cyan)),
        RowKind::Note => return Line::from(text(fit(&row.text, width), dim)),
        RowKind::Text if row.new.is_none() => return Line::from(text(fit(&clean(&row.text), width), Color::Gray)),
        RowKind::Text => (num(row.new), "│", Color::Gray),
        RowKind::Same => (gutter(numbers, num(row.old), num(row.new)), " ", Color::Gray),
        RowKind::Removed => (gutter(numbers, num(row.old), num(None)), "-", Color::Red),
        RowKind::Added => (gutter(numbers, num(None), num(row.new)), "+", Color::Green),
    };
    let lead = format!("{columns} ");
    let room = width.saturating_sub(super::width(&lead) + 2);
    Line::from(vec![text(lead, dim), text(format!("{sign} "), color), text(fit(&clean(&row.text), room), color)])
}

/// The number columns a diff row shows: in the panel, the new file's
/// number, else the old's; in full, both.
fn gutter(numbers: Numbers, old: String, new: String) -> String {
    match numbers {
        Numbers::Both => format!("{old} {new}"),
        Numbers::One if new.trim().is_empty() => old,
        Numbers::One => new,
    }
}

/// A note above a change the log had no patch for: what the hunks are.
fn note(app: &App, span: usize, content: &CallContent) -> Option<Line<'static>> {
    let CallContent::Change { exact: false, .. } = content else { return None };
    let s = &app.trace.spans()[span];
    Some(Line::from(if s.error {
        text("✗ not applied — what it tried", Color::Red)
    } else if s.end.is_none() {
        text("running — what it asked for", Color::DarkGray)
    } else {
        text("what it asked for — the log has no patch", Color::DarkGray)
    }))
}

/// The trace panel's preview of call `span`'s content: its first rows and
/// how to see the rest. Nothing for a call with none.
pub(super) fn preview(app: &App, span: usize, width: usize) -> Vec<Line<'static>> {
    let content = match app.content_state(span) {
        ContentState::Loaded(c) => c,
        ContentState::Loading => return vec![Line::from(""), Line::from(text(" loading…", Color::DarkGray))],
        ContentState::Missing => return Vec::new(),
    };
    let rows = content.rows();
    if rows.is_empty() {
        return Vec::new();
    }
    let digits = digits(&rows);
    let mut out = vec![Line::from("")];
    out.extend(note(app, span, content).map(indent));
    out.extend(rows.iter().take(PREVIEW).map(|r| indent(row_line(r, Numbers::One, digits, width.saturating_sub(1)))));
    let more = rows.len().saturating_sub(PREVIEW);
    let hint = if more > 0 { format!(" … {more} more · o opens in full") } else { " o opens in full".to_string() };
    out.push(Line::from(text(hint, Color::DarkGray)));
    out
}

fn indent(line: Line<'static>) -> Line<'static> {
    let mut spans = vec![Span::raw(" ")];
    spans.extend(line.spans);
    Line::from(spans)
}

/// The content view (`o`): call `span`'s content over the whole main area,
/// scrolled.
pub(super) fn render_view(f: &mut Frame, app: &App, area: Rect) {
    let Some(view) = &app.content_view else { return };
    let s = &app.trace.spans()[view.span];
    let title = format!(" {} · {} · {} ", span_name(app, view.span), app.agent_name(&s.agent), ambits::time::clock(s.start));
    let block = Block::default()
        .title(fit(&title, area.width.saturating_sub(4) as usize))
        .title_bottom(Line::from(text(" j/k scroll · n/N hunk · g/G ends · o/Esc close ", Color::DarkGray)))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD));
    let inner = block.inner(area);
    f.render_widget(Clear, area);
    f.render_widget(block, area);

    let width = inner.width as usize;
    let content = match app.content_state(view.span) {
        ContentState::Loaded(c) => c,
        ContentState::Loading => return f.render_widget(Paragraph::new(text(" loading…", Color::DarkGray)), inner),
        ContentState::Missing => return f.render_widget(Paragraph::new(text(" The log has no content for this call.", Color::DarkGray)), inner),
    };
    let mut lines: Vec<Line<'static>> = note(app, view.span, content).into_iter().collect();
    let height = (inner.height as usize).saturating_sub(lines.len());
    view.height.set(height);
    let rows = content.rows();
    let digits = digits(&rows);
    let first = view.scroll.min(rows.len().saturating_sub(height));
    lines.extend(rows.iter().skip(first).take(height).map(|r| row_line(r, Numbers::Both, digits, width)));
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
        let d = digits(&rows);
        rows.iter().map(|r| row_line(r, numbers, d, 40).spans.iter().map(|s| s.content.to_string()).collect()).collect()
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
