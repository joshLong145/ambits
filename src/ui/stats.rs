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
    let (counts, seen) = match app.agent_filter.as_deref() {
        Some(agent_id) => (
            app.ledger.count_by_depth_for_agent(agent_id),
            app.ledger.total_seen_for_agent(agent_id),
        ),
        None => (app.ledger.count_by_depth(), app.ledger.total_seen()),
    };

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
        ]));

        // "All" entry
        let stats_focused = app.focus == FocusPanel::Stats;
        let all_active = app.agent_filter.is_none();
        let all_cursor = stats_focused && app.agent_selection_index == 0;
        let all_marker = if all_active { "\u{25b6} " } else if all_cursor { "> " } else { "  " };
        let all_color = if all_active { Color::Yellow } else if all_cursor { Color::White } else { colors::ACCENT_MUTED };
        let all_pct = if total > 0 {
            (app.ledger.total_seen() as f64 / total as f64 * 100.0) as u32
        } else {
            0
        };
        lines.push(Line::from(vec![
            Span::styled(format!("  {}", all_marker), Style::default().fg(all_color)),
            Span::styled("[All]", Style::default().fg(all_color).add_modifier(Modifier::BOLD)),
            Span::styled(
                format!("  Seen: {}%", all_pct),
                Style::default().fg(Color::DarkGray),
            ),
        ]));

        // Per-agent entries in hierarchy order
        let flat = app.flattened_agents();
        for (i, (agent_id, indent)) in flat.iter().enumerate() {
            let is_active = app.agent_filter.as_deref() == Some(agent_id.as_str());
            let is_cursor = stats_focused && app.agent_selection_index == i + 1;
            let marker = if is_active { "\u{25b6} " } else if is_cursor { "> " } else { "  " };
            let color = if is_active { Color::Yellow } else if is_cursor { Color::White } else { colors::ACCENT_MUTED };

            // Determine if this is the last sibling at its indent level.
            let is_last_sibling = flat[i + 1..]
                .iter()
                .all(|(_, d)| *d > *indent || *d < *indent)
                || flat[i + 1..]
                    .iter()
                    .take_while(|(_, d)| *d >= *indent)
                    .all(|(_, d)| *d > *indent);

            // Build indent prefix: for each ancestor level, show │ if that
            // ancestor has more siblings below, or blank if it was the last.
            let mut prefix = String::new();
            for level in 0..*indent {
                // Check if there's a later entry at this level (meaning the
                // ancestor at this level still has siblings below us).
                let ancestor_continues = flat[i + 1..]
                    .iter()
                    .any(|(_, d)| *d == level);
                if ancestor_continues {
                    prefix.push_str("\u{2502} ");
                } else {
                    prefix.push_str("  ");
                }
            }

            let branch = if is_last_sibling {
                "\u{2514}\u{2500} "
            } else {
                "\u{251c}\u{2500} "
            };

            let display_name = short_id(agent_id);

            // Per-agent seen%
            let agent_seen = app.ledger.total_seen_for_agent(agent_id);
            let agent_pct = if total > 0 {
                (agent_seen as f64 / total as f64 * 100.0) as u32
            } else {
                0
            };

            lines.push(Line::from(vec![
                Span::styled(
                    format!("  {}{}{}", marker, prefix, branch),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(display_name, Style::default().fg(color)),
                Span::styled(
                    format!("  {}%", agent_pct),
                    Style::default().fg(Color::DarkGray),
                ),
            ]));
        }
    }

    // Scroll to keep the selected agent visible when the panel is focused.
    let visible_height = area.height.saturating_sub(2) as usize; // -2 for borders
    let scroll_offset = if app.focus == FocusPanel::Stats && !app.agents_seen.is_empty() {
        // The agent list starts after the fixed header lines.
        // "All" entry is at header_lines, agents start at header_lines + 1.
        let header_lines = lines.len().saturating_sub(app.flattened_agents().len() + 1); // +1 for "All"
        let cursor_line = header_lines + app.agent_selection_index;
        if cursor_line >= visible_height {
            (cursor_line - visible_height + 1) as u16
        } else {
            0
        }
    } else {
        0
    };

    let paragraph = Paragraph::new(lines)
        .block(block)
        .scroll((scroll_offset, 0));
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
