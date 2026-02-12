use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};

use ambits::app::{App, FocusPanel};
use ambits::tracking::ReadDepth;

use super::colors;

pub fn render(f: &mut Frame, app: &App, area: Rect) {
    let border_style = if app.focus == FocusPanel::Stats {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let block = Block::default()
        .title(" Coverage Stats ")
        .borders(Borders::ALL)
        .border_style(border_style);

    let total = app.project_tree.total_symbols();
    let counts = app.ledger.count_by_depth();
    let seen = app.ledger.total_seen();

    let pct = if total > 0 {
        (seen as f64 / total as f64 * 100.0) as u32
    } else {
        0
    };

    let count_for = |d: ReadDepth| -> usize { *counts.get(&d).unwrap_or(&0) };

    let mut lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::raw("  Coverage: "),
            Span::styled(
                format!("{}%", pct),
                Style::default()
                    .fg(coverage_color(pct))
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("  ({}/{})", seen, total),
                Style::default().fg(Color::DarkGray),
            ),
        ]),
        Line::from(""),
        stat_line("  Full Body", count_for(ReadDepth::FullBody), colors::DEPTH_FULL_BODY),
        stat_line("  Signature", count_for(ReadDepth::Signature), colors::DEPTH_SIGNATURE),
        stat_line("  Overview ", count_for(ReadDepth::Overview), colors::DEPTH_OVERVIEW),
        stat_line("  Name Only", count_for(ReadDepth::NameOnly), colors::DEPTH_NAME_ONLY),
        stat_line("  Stale    ", count_for(ReadDepth::Stale), colors::DEPTH_STALE),
        stat_line(
            "  Unseen   ",
            total.saturating_sub(seen),
            colors::DEPTH_UNSEEN,
        ),
        Line::from(""),
        Line::from(vec![
            Span::raw("  Files: "),
            Span::styled(
                format!("{}", app.project_tree.total_files()),
                Style::default().fg(Color::White),
            ),
            Span::raw("  Symbols: "),
            Span::styled(
                format!("{}", total),
                Style::default().fg(Color::White),
            ),
        ]),
    ];

    // Session info.
    if let Some(ref sid) = app.session_id {
        let short = if sid.len() > 12 { &sid[..12] } else { sid };
        lines.push(Line::from(vec![
            Span::raw("  Session: "),
            Span::styled(short, Style::default().fg(colors::ACCENT_MUTED)),
        ]));
    }

    // Agents section.
    if !app.agents_seen.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled(
                format!("  Agents: {} ", app.agents_seen.len()),
                Style::default().fg(Color::White),
            ),
            Span::styled(
                match &app.agent_filter {
                    None => "[all]".to_string(),
                    Some(id) => format!("[{}]", short_id(id)),
                },
                Style::default().fg(Color::Yellow),
            ),
        ]));

        for agent_id in &app.agents_seen {
            let is_selected = app.agent_filter.as_deref() == Some(agent_id);
            let prefix = if agent_id.starts_with("agent-") {
                "  \u{251c}\u{2500} "
            } else {
                "  \u{2502} "
            };
            let color = if is_selected {
                Color::Yellow
            } else {
                colors::ACCENT_MUTED
            };
            lines.push(Line::from(vec![
                Span::styled(prefix, Style::default().fg(Color::DarkGray)),
                Span::styled(short_id(agent_id), Style::default().fg(color)),
            ]));
        }
    }

    let paragraph = Paragraph::new(lines).block(block);
    f.render_widget(paragraph, area);
}

fn stat_line(label: &str, count: usize, color: Color) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            format!("{}: ", label),
            Style::default().fg(Color::DarkGray),
        ),
        Span::styled(format!("{:>5}", count), Style::default().fg(color)),
    ])
}

fn short_id(id: &str) -> String {
    if id.len() > 12 {
        id[..12].to_string()
    } else {
        id.to_string()
    }
}

fn coverage_color(pct: u32) -> Color {
    match pct {
        0..=20 => colors::PCT_LOW,
        21..=50 => colors::PCT_MID_LOW,
        51..=80 => colors::PCT_MID_HIGH,
        _ => colors::PCT_HIGH,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;
    use ambits::symbols::{ProjectTree, FileSymbols, SymbolCategory, SymbolNode};

    fn sym(id: &str, name: &str) -> SymbolNode {
        let hash = ambits::symbols::merkle::content_hash(name);
        SymbolNode {
            id: id.into(), name: name.into(), category: SymbolCategory::Function,
            label: "fn".into(), file_path: PathBuf::new(),
            byte_range: 0..100, line_range: 1..10, content_hash: hash,
            merkle_hash: hash, children: Vec::new(), estimated_tokens: 30,
        }
    }

    fn test_app() -> App {
        let tree = ProjectTree {
            root: PathBuf::from("/test"),
            files: vec![
                FileSymbols { file_path: "mock/a.rs".into(), symbols: vec![sym("a1", "alpha")], total_lines: 50 },
            ],
        };
        App::new(tree, PathBuf::from("/test"), None)
    }

    /// Find the foreground color of the first cell matching `text` in the entire buffer.
    fn fg_color_of(backend: &TestBackend, text: &str) -> Option<Color> {
        let buf = backend.buffer();
        for y in 0..buf.area.height {
            let row_str: String = (0..buf.area.width)
                .map(|x| buf[(x, y)].symbol().to_string())
                .collect();
            if let Some(col) = row_str.find(text) {
                return Some(buf[(col as u16, y)].fg);
            }
        }
        None
    }

    #[test]
    fn coverage_color_gradient() {
        assert_eq!(coverage_color(0), colors::PCT_LOW);
        assert_eq!(coverage_color(20), colors::PCT_LOW);
        assert_eq!(coverage_color(21), colors::PCT_MID_LOW);
        assert_eq!(coverage_color(50), colors::PCT_MID_LOW);
        assert_eq!(coverage_color(51), colors::PCT_MID_HIGH);
        assert_eq!(coverage_color(80), colors::PCT_MID_HIGH);
        assert_eq!(coverage_color(81), colors::PCT_HIGH);
        assert_eq!(coverage_color(100), colors::PCT_HIGH);
    }

    #[test]
    fn short_id_truncates() {
        assert_eq!(short_id("abcdefghijklmnop"), "abcdefghijkl");
        assert_eq!(short_id("short"), "short");
        assert_eq!(short_id("exactly12chr"), "exactly12chr");
    }

    #[test]
    fn stat_line_format() {
        let line = stat_line("  Full Body", 42, colors::DEPTH_FULL_BODY);
        let spans: Vec<_> = line.spans.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(spans[0], "  Full Body: ");
        assert_eq!(spans[1], "   42");
    }

    #[test]
    fn render_shows_zero_coverage() {
        let app = test_app();
        let backend = TestBackend::new(40, 20);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| render(f, &app, f.area())).unwrap();

        // The percentage is the only bold-styled content; find it by modifier.
        let buf = terminal.backend().buffer();
        let bold_cell = (0..buf.area.height)
            .flat_map(|y| (0..buf.area.width).map(move |x| &buf[(x, y)]))
            .find(|cell| cell.modifier.contains(Modifier::BOLD))
            .expect("bold percentage cell not found");
        assert_eq!(bold_cell.fg, colors::PCT_LOW);
    }

    #[test]
    fn render_shows_full_coverage() {
        let mut app = test_app();
        app.ledger.record("a1".into(), ReadDepth::FullBody, [0; 32], "ag".into(), 10);
        app.rebuild_tree_rows();

        let backend = TestBackend::new(40, 20);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| render(f, &app, f.area())).unwrap();

        let color = fg_color_of(terminal.backend(), "100%").unwrap();
        assert_eq!(color, colors::PCT_HIGH);
    }

    #[test]
    fn render_with_session_and_agents() {
        let mut app = test_app();
        app.session_id = Some("abcdef123456789".into());
        app.agents_seen.push("agent-abc123456789".into());

        let backend = TestBackend::new(40, 20);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| render(f, &app, f.area())).unwrap();

        let color = fg_color_of(terminal.backend(), "abcdef123456").unwrap();
        assert_eq!(color, colors::ACCENT_MUTED);
    }
}
