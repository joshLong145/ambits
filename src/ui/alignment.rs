use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};

use ambits::app::App;
use ambits::tracking::alignment::FileAlignment;

use super::colors;
use super::stats::short_id;

/// Render the sub-agent alignment popup over `area`. The caller must verify
/// `app.show_alignment_overlay` before calling.
pub fn render(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(area, 80, 70);
    f.render_widget(Clear, popup_area);

    let mut lines: Vec<Line> = Vec::new();

    if app.agent_alignment.is_empty() {
        lines.push(Line::from(Span::styled(
            "No sibling agents to compare.",
            Style::default().fg(Color::DarkGray),
        )));
    } else {
        for (i, pair) in app.agent_alignment.iter().enumerate() {
            if i > 0 {
                lines.push(Line::from(""));
            }
            lines.push(Line::from(Span::styled(
                format!("{}  \u{2194}  {}", short_id(&pair.agent_a), short_id(&pair.agent_b)),
                Style::default().add_modifier(Modifier::BOLD),
            )));
            lines.push(Line::from(vec![
                Span::raw("  score: "),
                Span::styled(
                    format!("{:.1}%", pair.score * 100.0),
                    Style::default()
                        .fg(score_color(pair.score))
                        .add_modifier(Modifier::BOLD),
                ),
            ]));
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(
                    format!("Aligned: {}", pair.aligned_count),
                    Style::default().fg(colors::FILE_FULLY_COVERED),
                ),
                Span::raw("   "),
                Span::styled(
                    format!("DepthMismatch: {}", pair.mismatch_count),
                    Style::default().fg(colors::FILE_PARTIALLY_COVERED),
                ),
                Span::raw("   "),
                Span::styled(
                    format!("Unshared: {}", pair.unshared_count),
                    Style::default().fg(colors::DEPTH_UNSEEN),
                ),
            ]));

            // Per-file breakdown, already sorted worst-to-best by
            // `pair_alignment` (DepthMismatch, worst first, then Unshared,
            // then Aligned) so the files needing attention are at the top.
            for entry in &pair.files {
                let (label, color) = match entry.status {
                    FileAlignment::DepthMismatch => ("DepthMismatch", colors::FILE_PARTIALLY_COVERED),
                    FileAlignment::Unshared => ("Unshared", colors::DEPTH_UNSEEN),
                    FileAlignment::Aligned => ("Aligned", colors::FILE_FULLY_COVERED),
                };
                let mut spans = vec![
                    Span::raw(format!("    {:<40} ", entry.path)),
                    Span::styled(label, Style::default().fg(color)),
                ];
                if let Some(fraction) = entry.matched_fraction {
                    spans.push(Span::styled(
                        format!("  {:.0}% matched", fraction * 100.0),
                        Style::default().fg(color),
                    ));
                }
                lines.push(Line::from(spans));
            }
        }
    }

    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "  Esc close",
        Style::default().fg(Color::DarkGray),
    )));

    let title = format!(
        " Sub-agent alignment — {} pair{} ",
        app.agent_alignment.len(),
        if app.agent_alignment.len() == 1 { "" } else { "s" }
    );

    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow));
    let paragraph = Paragraph::new(lines).block(block).wrap(Wrap { trim: false });
    f.render_widget(paragraph, popup_area);
}

/// Color the aggregate score along the shared low/mid/high coverage gradient.
fn score_color(score: f64) -> Color {
    if score >= 0.75 {
        colors::PCT_HIGH
    } else if score >= 0.5 {
        colors::PCT_MID_HIGH
    } else if score >= 0.25 {
        colors::PCT_MID_LOW
    } else {
        colors::PCT_LOW
    }
}

/// Compute a centered subrect taking `percent_x` × `percent_y` of `area`.
fn centered_rect(area: Rect, percent_x: u16, percent_y: u16) -> Rect {
    let w = area.width * percent_x / 100;
    let h = area.height * percent_y / 100;
    let x = area.x + (area.width.saturating_sub(w)) / 2;
    let y = area.y + (area.height.saturating_sub(h)) / 2;
    Rect { x, y, width: w, height: h }
}
