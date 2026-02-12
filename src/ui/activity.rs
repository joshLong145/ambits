use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};

use ambits::app::{App, FocusPanel};

use super::colors;

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
                    Style::default().fg(colors::ACCENT_MUTED),
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use ambits::ingest::AgentToolCall;
    use ambits::symbols::{ProjectTree, FileSymbols};
    use ambits::tracking::ReadDepth;

    fn test_app() -> App {
        let tree = ProjectTree {
            root: PathBuf::from("/test"),
            files: vec![
                FileSymbols { file_path: "mock/a.rs".into(), symbols: Vec::new(), total_lines: 10 },
            ],
        };
        App::new(tree, PathBuf::from("/test"), None)
    }

    /// Find the foreground color of the first cell in `row` that contains part of `text`.
    fn fg_color_of(backend: &TestBackend, row: u16, text: &str) -> Option<Color> {
        let buf = backend.buffer();
        let row_str: String = (0..buf.area.width)
            .map(|x| buf[(x, row)].symbol().to_string())
            .collect::<String>();
        let col = row_str.find(text)? as u16;
        Some(buf[(col, row)].fg)
    }

    #[test]
    fn render_with_activity_uses_accent_color() {
        let mut app = test_app();
        app.activity.push(AgentToolCall {
            agent_id: "agent-abc123".into(),
            tool_name: "Read".into(),
            file_path: Some(PathBuf::from("mock/a.rs")),
            read_depth: ReadDepth::FullBody,
            description: "Read a.rs".into(),
            timestamp_str: "2025-01-01T00:00:00Z".into(),
            target_symbol: None,
            target_lines: None,
        });

        let backend = TestBackend::new(60, 10);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| render(f, &app, f.area())).unwrap();

        let color = fg_color_of(terminal.backend(), 1, "agent-ab").unwrap();
        assert_eq!(color, colors::ACCENT_MUTED);
    }
}
