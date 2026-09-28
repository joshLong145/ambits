use ratatui::layout::Rect;
pub mod colors;
pub mod tree_view;
pub mod stats;
pub mod activity;
pub mod compaction;
pub mod alignment;
pub mod trace_view;

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout};

use ambits::app::{App, SortMode};

pub fn render(f: &mut Frame, app: &App) {
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(10),       // top: tree + stats
            Constraint::Length(8),     // bottom: activity feed
            Constraint::Length(1),     // detail line
            Constraint::Length(1),     // status bar
        ])
        .split(f.area());

    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(62),  // tree
            Constraint::Percentage(38),  // stats
        ])
        .split(outer[0]);

    if app.trace_view.open {
        trace_view::render(f, app, top[0]);
    } else {
        tree_view::render(f, app, top[0]);
    }
    stats::render(f, app, top[1]);
    activity::render(f, app, outer[1]);
    render_detail_line(f, app, outer[2]);
    render_status_bar(f, app, outer[3]);

    if app.show_compaction_overlay && !app.compaction_history.is_empty() {
        compaction::render(f, app, f.area());
    }

    if app.show_alignment_overlay {
        alignment::render(f, app, f.area());
    }
}

/// The selected row's facts ([`App::detail_line`]), cut to fit.
fn render_detail_line(f: &mut Frame, app: &App, area: Rect) {
    use ratatui::style::{Color, Style};
    use ratatui::widgets::Paragraph;

    let text = app.detail_line().map(|line| fit(&format!(" {line}"), area.width as usize)).unwrap_or_default();
    f.render_widget(Paragraph::new(text).style(Style::default().fg(Color::Gray)), area);
}

/// Display columns of `text`.
pub(crate) fn width(text: &str) -> usize {
    unicode_width::UnicodeWidthStr::width(text)
}

/// `text` in at most `width` columns, ending in `…` when cut.
pub(crate) fn fit(text: &str, width: usize) -> String {
    use unicode_width::UnicodeWidthChar;
    if unicode_width::UnicodeWidthStr::width(text) <= width {
        return text.to_string();
    }
    let mut out = String::new();
    let mut used = 0;
    for c in text.chars() {
        let w = c.width().unwrap_or(0);
        if used + w + 1 > width {
            break;
        }
        out.push(c);
        used += w;
    }
    if width > 0 {
        out.push('…');
    }
    out
}

fn render_status_bar(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    use ratatui::style::{Color, Style};
    use ratatui::text::{Line, Span};
    use ratatui::widgets::Paragraph;

    let status = if let Some(ref message) = app.last_editor_error {
        Line::from(vec![Span::styled(
            format!(" {message}"),
            Style::default().fg(Color::Red),
        )])
    } else if app.trace_view.open {
        let key = |k: &'static str, what: &'static str| [Span::styled(k, Style::default().fg(Color::DarkGray)), Span::raw(what)];
        let mut spans = vec![Span::raw(" ")];
        for (k, what) in [
            ("[t]", "ree "), ("[v]", "layout "), ("[w/s]", "zoom "), ("[a/d]", "pan "), ("[0]", "fit "),
            ("[j/k]", "row "), ("[h/l]", "fold/step "), ("[enter]", "follow "), ("[space]", "fold "),
            ("[/]", "find "), ("[e]", "rror "), ("[tab]", "agent "),
        ] {
            spans.extend(key(k, what));
        }
        Line::from(spans)
    } else if app.search_mode {
        Line::from(vec![
            Span::styled(" /", Style::default().fg(Color::Yellow)),
            Span::raw(&app.search_query),
            Span::styled("_", Style::default().fg(Color::Yellow)),
        ])
    } else {
        let mut spans = vec![
            Span::styled(" [q]", Style::default().fg(Color::DarkGray)),
            Span::raw("uit "),
            Span::styled("[j/k]", Style::default().fg(Color::DarkGray)),
            Span::raw("nav "),
            Span::styled("[h/l]", Style::default().fg(Color::DarkGray)),
            Span::raw("expand "),
            Span::styled("[enter]", Style::default().fg(Color::DarkGray)),
            Span::raw("open "),
            Span::styled("[/]", Style::default().fg(Color::DarkGray)),
            Span::raw("search "),
            Span::styled("[s]", Style::default().fg(Color::DarkGray)),
            Span::raw(match app.sort_mode {
                SortMode::Alphabetical => "ort:A-Z ",
                SortMode::ByCoverage => "ort:cov ",
            }),
            Span::styled("[a/A]", Style::default().fg(Color::DarkGray)),
            Span::raw("gents "),
            Span::styled("[tab]", Style::default().fg(Color::DarkGray)),
            Span::raw("focus "),
            Span::styled("[t]", Style::default().fg(Color::DarkGray)),
            Span::raw("race "),
        ];

        if !app.compaction_history.is_empty() {
            spans.push(Span::styled("[C]", Style::default().fg(Color::Yellow)));
            spans.push(Span::raw(if app.show_compaction_overlay {
                "close "
            } else {
                "ompact "
            }));
        }

        if app.agent_filter.is_some() {
            spans.push(Span::styled("[d]", Style::default().fg(Color::DarkGray)));
            spans.push(Span::raw("iff-align "));
        }

        // Show current agent filter
        if let Some(ref agent_id) = app.agent_filter {
            spans.push(Span::styled(
                format!(" Agent: {}", stats::short_id(agent_id)),
                Style::default().fg(Color::Yellow),
            ));
        }

        Line::from(spans)
    };

    f.render_widget(
        Paragraph::new(status).style(Style::default().bg(Color::DarkGray).fg(Color::White)),
        area,
    );
}

/// Center a box of `percent_x` × `percent_y` inside `area`.
///
/// Shared by the overlays. Both had their own byte-identical copy, which is
/// the kind of duplication that stays invisible until the two quietly disagree
/// about how to round.
pub fn centered_rect(area: Rect, percent_x: u16, percent_y: u16) -> Rect {
    let w = area.width * percent_x / 100;
    let h = area.height * percent_y / 100;
    let x = area.x + (area.width.saturating_sub(w)) / 2;
    let y = area.y + (area.height.saturating_sub(h)) / 2;
    Rect { x, y, width: w, height: h }
}

#[cfg(test)]
mod tests {
    use super::fit;

    #[test]
    fn fit_cuts_to_the_width_with_an_ellipsis() {
        assert_eq!(fit("short", 10), "short");
        assert_eq!(fit("exactly10!", 10), "exactly10!");
        assert_eq!(fit("a longer line", 8), "a longe…");
        assert_eq!(fit("✎✎✎", 2), "✎…");
        assert_eq!(fit("anything", 0), "");
    }
}
