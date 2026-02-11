use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};

use ambits::app::{App, FocusPanel};

pub fn render(f: &mut Frame, app: &App, area: Rect) {
    let border_style = if app.focus == FocusPanel::Activity {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let block = Block::default()
        .title(" Activity Feed ")
        .borders(Borders::ALL)
        .border_style(border_style);

    // Show the most recent events that fit in the area.
    let max_lines = area.height.saturating_sub(2) as usize;
    let start = app.activity.len().saturating_sub(max_lines);
    let visible = &app.activity[start..];

    let lines: Vec<Line> = visible
        .iter()
        .map(|event| {
            let agent_short = if event.agent_id.len() > 8 {
                &event.agent_id[..8]
            } else {
                &event.agent_id
            };

            Line::from(vec![
                Span::styled(
                    format!(" [{}] ", agent_short),
                    Style::default().fg(Color::Rgb(120, 120, 180)),
                ),
                Span::styled(
                    &event.description,
                    Style::default().fg(Color::White),
                ),
                Span::styled(
                    format!("  ({})", event.read_depth),
                    Style::default().fg(Color::DarkGray),
                ),
            ])
        })
        .collect();

    let paragraph = if lines.is_empty() {
        Paragraph::new(Line::from(
            Span::styled(
                "  No agent activity yet",
                Style::default().fg(Color::DarkGray),
            ),
        ))
        .block(block)
    } else {
        Paragraph::new(lines).block(block)
    };

    f.render_widget(paragraph, area);
}
