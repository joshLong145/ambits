use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};

use ambits::app::{App, FocusPanel};
use ambits::tracking::ReadDepth;

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
        stat_line("  Full Body", count_for(ReadDepth::FullBody), Color::Rgb(80, 220, 120)),
        stat_line("  Signature", count_for(ReadDepth::Signature), Color::Rgb(80, 140, 255)),
        stat_line("  Overview ", count_for(ReadDepth::Overview), Color::Rgb(120, 160, 220)),
        stat_line("  Name Only", count_for(ReadDepth::NameOnly), Color::Rgb(160, 160, 160)),
        stat_line("  Stale    ", count_for(ReadDepth::Stale), Color::Rgb(230, 160, 60)),
        stat_line(
            "  Unseen   ",
            total.saturating_sub(seen),
            Color::Rgb(100, 100, 100),
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
            Span::styled(short, Style::default().fg(Color::Rgb(120, 120, 180))),
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
                Color::Rgb(120, 120, 180)
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
        0..=20 => Color::Rgb(180, 60, 60),
        21..=50 => Color::Rgb(230, 160, 60),
        51..=80 => Color::Rgb(200, 200, 80),
        _ => Color::Rgb(80, 220, 120),
    }
}
