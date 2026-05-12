use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};

use ambits::app::{App, FocusPanel};
use ambits::ingest::{AgentToolCall, CompactionEvent};

use super::colors;

/// A single rendered row in the activity feed: either a tool call or a
/// compaction boundary marker.
enum FeedEntry<'a> {
    Tool(&'a AgentToolCall),
    Compaction(&'a CompactionEvent),
}

/// Build the merged feed, sorted by timestamp. Tool calls are filtered by
/// the active agent filter; compactions are always shown.
fn build_feed<'a>(app: &'a App) -> Vec<FeedEntry<'a>> {
    let filter = app.agent_filter.as_deref();
    let mut entries: Vec<(&'a str, FeedEntry<'a>)> = app
        .activity
        .iter()
        .filter(|e| filter.is_none_or(|id| &*e.agent_id == id))
        .map(|e| (e.timestamp_str.as_str(), FeedEntry::Tool(e)))
        .chain(
            app.compaction_history
                .iter()
                .map(|c| (c.timestamp.as_str(), FeedEntry::Compaction(c))),
        )
        .collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    entries.into_iter().map(|(_, e)| e).collect()
}

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

    let feed = build_feed(app);

    // Window the visible events using the scroll offset.
    let max_lines = area.height.saturating_sub(2) as usize;
    let total = feed.len();
    // Clamp scroll offset so we can't scroll past the top.
    let max_offset = total.saturating_sub(max_lines);
    let offset = app.activity_scroll_offset.min(max_offset);
    let end = total.saturating_sub(offset);
    let start = end.saturating_sub(max_lines);
    let visible = &feed[start..end];

    let width = area.width.saturating_sub(2) as usize;
    let lines: Vec<Line> = visible
        .iter()
        .map(|entry| match entry {
            FeedEntry::Tool(event) => {
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
            }
            FeedEntry::Compaction(c) => compaction_marker_line(c, width),
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

/// Render a single-line separator marking a compaction boundary.
/// Format: ─── compaction #1 · 31.2% seen · 12 files ───────
fn compaction_marker_line<'a>(c: &'a CompactionEvent, width: usize) -> Line<'a> {
    let info = format!(
        " compaction #{} · {:.1}% seen · {} files ",
        c.sequence,
        c.ledger_before.seen_percent,
        c.ledger_before.files_accessed.len(),
    );
    let lead = "─".repeat(3);
    let trail_count = width.saturating_sub(lead.chars().count() + info.chars().count());
    let trail = "─".repeat(trail_count);
    Line::from(vec![
        Span::styled(lead, Style::default().fg(Color::Yellow)),
        Span::styled(info, Style::default().fg(Color::Yellow)),
        Span::styled(trail, Style::default().fg(Color::Yellow)),
    ])
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
            label: "agent-abc123".into(),
        });

        let backend = TestBackend::new(60, 10);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| render(f, &app, f.area())).unwrap();

        let color = fg_color_of(terminal.backend(), 1, "agent-ab").unwrap();
        assert_eq!(color, colors::ACCENT_MUTED);
    }
}
