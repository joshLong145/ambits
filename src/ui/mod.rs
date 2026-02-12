pub mod colors;
pub mod tree_view;
pub mod stats;
pub mod activity;

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout};

use ambits::app::{App, SortMode};

pub fn render(f: &mut Frame, app: &App) {
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(10),       // top: tree + stats
            Constraint::Length(8),     // bottom: activity feed
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

    tree_view::render(f, app, top[0]);
    stats::render(f, app, top[1]);
    activity::render(f, app, outer[1]);
    render_status_bar(f, app, outer[2]);
}

fn render_status_bar(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    use ratatui::style::{Color, Style};
    use ratatui::text::{Line, Span};
    use ratatui::widgets::Paragraph;

    let status = if app.search_mode {
        Line::from(vec![
            Span::styled(" /", Style::default().fg(Color::Yellow)),
            Span::raw(&app.search_query),
            Span::styled("_", Style::default().fg(Color::Yellow)),
        ])
    } else {
        Line::from(vec![
            Span::styled(" [q]", Style::default().fg(Color::DarkGray)),
            Span::raw("uit "),
            Span::styled("[j/k]", Style::default().fg(Color::DarkGray)),
            Span::raw("nav "),
            Span::styled("[h/l]", Style::default().fg(Color::DarkGray)),
            Span::raw("expand "),
            Span::styled("[/]", Style::default().fg(Color::DarkGray)),
            Span::raw("search "),
            Span::styled("[s]", Style::default().fg(Color::DarkGray)),
            Span::raw(match app.sort_mode {
                SortMode::Alphabetical => "ort:A-Z ",
                SortMode::ByCoverage => "ort:cov ",
            }),
            Span::styled("[a]", Style::default().fg(Color::DarkGray)),
            Span::raw("gents "),
            Span::styled("[tab]", Style::default().fg(Color::DarkGray)),
            Span::raw("focus "),
        ])
    };

    f.render_widget(
        Paragraph::new(status).style(Style::default().bg(Color::DarkGray).fg(Color::White)),
        area,
    );
}
