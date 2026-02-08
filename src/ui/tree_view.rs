use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState};

use crate::app::{App, FocusPanel};
use crate::tracking::ReadDepth;

pub fn render(f: &mut Frame, app: &App, area: Rect) {
    let border_style = if app.focus == FocusPanel::Tree {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let block = Block::default()
        .title(" Symbol Tree ")
        .borders(Borders::ALL)
        .border_style(border_style);

    let items: Vec<ListItem> = app
        .tree_rows
        .iter()
        .map(|row| {
            let indent = "  ".repeat(row.depth);
            let icon = if row.is_file {
                if row.is_expanded { "▼ " } else { "▶ " }
            } else if row.has_children {
                if row.is_expanded { "▾ " } else { "▸ " }
            } else {
                "  "
            };

            let color = depth_color(row.read_depth);

            let mut spans = vec![
                Span::raw(indent),
                Span::styled(icon, Style::default().fg(Color::DarkGray)),
            ];

            if row.is_file {
                // Use a noticeable color if the file has coverage
                let file_color = if row.read_depth != ReadDepth::Unseen {
                    Color::Rgb(80, 220, 120) // Bright green for files with coverage
                } else {
                    Color::White // White for files without coverage
                };
                spans.push(Span::styled(
                    &row.display_name,
                    Style::default().fg(file_color).add_modifier(Modifier::BOLD),
                ));
                spans.push(Span::styled(
                    format!("  ({})", row.line_range),
                    Style::default().fg(Color::DarkGray),
                ));
            } else {
                spans.push(Span::styled(
                    format!("{} ", row.label),
                    Style::default().fg(Color::DarkGray),
                ));
                spans.push(Span::styled(&row.display_name, Style::default().fg(color)));
                spans.push(Span::styled(
                    format!("  [{}] ~{} tok", row.line_range, row.token_count),
                    Style::default().fg(Color::DarkGray),
                ));
            }

            ListItem::new(Line::from(spans))
        })
        .collect();

    let mut state = ListState::default();
    state.select(Some(app.selected_index));

    let list = List::new(items)
        .block(block)
        .highlight_style(
            Style::default()
                .bg(Color::Rgb(60, 55, 50))
                .fg(Color::Rgb(255, 220, 150))
                .add_modifier(Modifier::BOLD),
        );

    f.render_stateful_widget(list, area, &mut state);
}

fn depth_color(depth: ReadDepth) -> Color {
    match depth {
        ReadDepth::Unseen => Color::Rgb(100, 100, 100),
        ReadDepth::NameOnly => Color::Rgb(160, 160, 160),
        ReadDepth::Overview => Color::Rgb(120, 160, 220),
        ReadDepth::Signature => Color::Rgb(80, 140, 255),
        ReadDepth::FullBody => Color::Rgb(80, 220, 120),
        ReadDepth::Stale => Color::Rgb(230, 160, 60),
    }
}
