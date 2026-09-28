use ratatui::layout::Rect;
pub mod colors;
pub mod tree_view;
pub mod stats;
pub mod activity;
pub mod compaction;
pub mod alignment;
pub mod trace_view;
pub mod inspector;

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout};

use ambits::app::{App, RightPane, SortMode};

pub fn render(f: &mut Frame, app: &App) {
    let activity_h = if app.show_activity { 8 } else { 0 };
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),          // header: session and totals
            Constraint::Min(10),            // tree | inspector
            Constraint::Length(activity_h), // activity feed, on `f`
            Constraint::Length(1),          // legend
            Constraint::Length(1),          // status bar
        ])
        .split(f.area());

    let main = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(outer[1]);

    render_header(f, app, outer[0]);
    if app.trace_view.open {
        trace_view::render(f, app, main[0]);
    } else {
        tree_view::render(f, app, main[0]);
    }
    match app.right_pane {
        RightPane::Inspector => inspector::render(f, app, main[1]),
        RightPane::Session => stats::render(f, app, main[1]),
    }
    if app.show_activity {
        activity::render(f, app, outer[2]);
    }
    render_legend(f, outer[3]);
    render_status_bar(f, app, outer[4]);

    if app.show_compaction_overlay && !app.compaction_history.is_empty() {
        compaction::render(f, app, f.area());
    }

    if app.show_alignment_overlay {
        alignment::render(f, app, f.area());
    }
}

/// ` ambits · <session> · all agents    42% seen 120/284  ●90 ◕10 ◑12 ◔8  !4 ◌20  ✎12`
fn render_header(f: &mut Frame, app: &App, area: Rect) {
    use ambits::tracking::ReadDepth;
    use ratatui::style::{Color, Modifier, Style};
    use ratatui::text::{Line, Span};
    use ratatui::widgets::Paragraph;

    let filter = app.agent_filter.as_deref();
    let (counts, seen) = match filter {
        Some(a) => (app.ledger.count_by_depth_for_agent(a), app.ledger.total_seen_for_agent(a)),
        None => (app.ledger.count_by_depth(), app.ledger.total_seen()),
    };
    let total = app.project_tree.total_symbols();
    let pct = (seen * 100).checked_div(total).unwrap_or(0);
    let session = app.session_slug.clone().or_else(|| app.session_id.as_ref().map(|s| s.chars().take(8).collect())).unwrap_or_else(|| "no session".into());
    let who = filter.map_or("all agents".to_string(), |a| app.agent_name(a).to_string());
    let dim = Style::default().fg(Color::DarkGray);
    let mut spans = vec![
        Span::styled(" ambits", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
        Span::styled(format!(" · {session} · {who}"), Style::default().fg(Color::Gray)),
    ];
    if let Some(p) = &app.filter {
        spans.push(Span::styled(format!(" · {}", p.display()), dim));
    }
    spans.push(Span::styled(format!("    {pct}% seen", ), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)));
    spans.push(Span::styled(format!(" {seen}/{total} "), dim));
    for depth in [ReadDepth::FullBody, ReadDepth::Signature, ReadDepth::Overview, ReadDepth::NameOnly] {
        let n = counts.get(&depth).copied().unwrap_or(0);
        spans.push(Span::styled(format!(" {}{n}", tree_view::depth_glyph(depth)), Style::default().fg(tree_view::depth_color(depth, false))));
    }
    let writes: usize = app.writes.by_file(filter).values().map(Vec::len).sum();
    spans.push(Span::styled(format!("   !{}", app.ledger.total_stale()), Style::default().fg(colors::DEPTH_STALE)));
    spans.push(Span::styled(format!(" ◌{}", app.ledger.total_restored()), Style::default().fg(colors::ACCENT_MUTED)));
    spans.push(Span::styled(format!(" ✎{writes}"), Style::default().fg(colors::WRITE_CURRENT)));
    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// What every glyph in the tree means, always on screen.
fn render_legend(f: &mut Frame, area: Rect) {
    use ambits::tracking::ReadDepth;
    use ratatui::style::{Color, Style};
    use ratatui::text::{Line, Span};
    use ratatui::widgets::Paragraph;

    let dim = Style::default().fg(Color::DarkGray);
    let mut spans = vec![Span::raw(" ")];
    for (depth, word) in [
        (ReadDepth::FullBody, "full"),
        (ReadDepth::Signature, "signature"),
        (ReadDepth::Overview, "overview"),
        (ReadDepth::NameOnly, "name"),
        (ReadDepth::Unseen, "unseen"),
    ] {
        spans.push(Span::styled(tree_view::depth_glyph(depth).to_string(), Style::default().fg(tree_view::depth_color(depth, false))));
        spans.push(Span::styled(format!(" {word}  "), dim));
    }
    spans.extend([
        Span::styled("!", Style::default().fg(colors::DEPTH_STALE)),
        Span::styled(" changed since read  ", dim),
        Span::styled("◌", Style::default().fg(colors::ACCENT_MUTED)),
        Span::styled(" before a compaction  ", dim),
        Span::styled("✎", Style::default().fg(colors::WRITE_CURRENT)),
        Span::styled(" written: ", dim),
        Span::styled("still there", Style::default().fg(colors::WRITE_CURRENT)),
        Span::styled(" · ", dim),
        Span::styled("changed", Style::default().fg(colors::WRITE_CHANGED)),
        Span::styled(" · ", dim),
        Span::styled("gone", Style::default().fg(colors::WRITE_REMOVED)),
    ]);
    f.render_widget(Paragraph::new(Line::from(spans)), area);
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
            Span::raw("races "),
            Span::styled("[i]", Style::default().fg(Color::DarkGray)),
            Span::raw(match app.right_pane {
                RightPane::Inspector => "session ",
                RightPane::Session => "nspector ",
            }),
            Span::styled("[f]", Style::default().fg(Color::DarkGray)),
            Span::raw("eed "),
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

/// Rendering to text, for the panels' tests.
#[cfg(test)]
pub(crate) mod test_render {
    use ratatui::backend::TestBackend;
    use ratatui::{Frame, Terminal};

    /// Row `row` of `backend`'s buffer, as text.
    pub fn row(backend: &TestBackend, row: u16) -> String {
        let buf = backend.buffer();
        (0..buf.area.width).map(|x| buf[(x, row)].symbol().to_string()).collect()
    }

    /// Every row of what `draw` renders in a `width` × `height` terminal.
    pub fn lines(width: u16, height: u16, draw: impl FnOnce(&mut Frame)) -> Vec<String> {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal.draw(draw).unwrap();
        (0..height).map(|y| row(terminal.backend(), y)).collect()
    }
}
