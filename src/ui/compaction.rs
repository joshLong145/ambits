use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};

use ambits::app::App;
use ambits::ingest::CompactionEvent;
use ambits::tracking::ReadDepth;

/// Render the compaction diff overlay over `area`. The caller must verify
/// `app.show_compaction_overlay` and that `compaction_history` is non-empty
/// before calling.
pub fn render(f: &mut Frame, app: &App, area: Rect) {
    let popup_area = centered_rect(area, 80, 70);
    f.render_widget(Clear, popup_area);

    let Some(ev) = app
        .compaction_history
        .get(app.compaction_overlay_index)
    else {
        return;
    };

    let mut lines: Vec<Line> = Vec::new();
    let title = format!(
        " Compaction #{} of {} — {} ",
        ev.sequence,
        app.compaction_history.len(),
        ev.timestamp,
    );

    // Summary block.
    lines.push(Line::from(Span::styled(
        "Summary",
        Style::default().add_modifier(Modifier::BOLD),
    )));
    for chunk in wrap_summary(&ev.summary, (popup_area.width as usize).saturating_sub(4)) {
        lines.push(Line::from(Span::raw(format!("  {chunk}"))));
    }
    lines.push(Line::from(""));

    // Before section.
    lines.push(Line::from(Span::styled(
        format!(
            "Before  ({} calls · {} files · {:.1}% seen)",
            ev.ledger_before.tool_call_count,
            ev.ledger_before.files_accessed.len(),
            ev.ledger_before.seen_percent,
        ),
        Style::default().add_modifier(Modifier::BOLD),
    )));
    if ev.ledger_before.files_accessed.is_empty() {
        lines.push(Line::from(Span::styled(
            "  (no files)",
            Style::default().fg(Color::DarkGray),
        )));
    } else {
        for path in ev.ledger_before.files_accessed.iter().take(20) {
            let path_str = path.to_string_lossy();
            let depth = file_depth(app, &path_str);
            lines.push(Line::from(vec![
                Span::raw(format!("  {path_str:<48} ")),
                Span::styled(
                    depth_label(depth).to_string(),
                    Style::default().fg(depth_color(depth)),
                ),
            ]));
        }
        if ev.ledger_before.files_accessed.len() > 20 {
            lines.push(Line::from(Span::styled(
                format!(
                    "  … ({} more)",
                    ev.ledger_before.files_accessed.len() - 20,
                ),
                Style::default().fg(Color::DarkGray),
            )));
        }
    }
    lines.push(Line::from(""));

    // After section: files present in the current ledger but absent from
    // the pre-compaction snapshot.
    let after_files = files_after_compaction(app, ev);
    lines.push(Line::from(Span::styled(
        format!("After  ({} files since compaction)", after_files.len()),
        Style::default().add_modifier(Modifier::BOLD),
    )));
    if after_files.is_empty() {
        lines.push(Line::from(Span::styled(
            "  (no new files since compaction)",
            Style::default().fg(Color::DarkGray),
        )));
    } else {
        for (path, depth) in after_files.iter().take(20) {
            lines.push(Line::from(vec![
                Span::raw(format!("  {path:<48} ")),
                Span::styled(
                    depth_label(*depth).to_string(),
                    Style::default().fg(depth_color(*depth)),
                ),
            ]));
        }
        if after_files.len() > 20 {
            lines.push(Line::from(Span::styled(
                format!("  … ({} more)", after_files.len() - 20),
                Style::default().fg(Color::DarkGray),
            )));
        }
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "  [ / ] step through compactions   C close",
        Style::default().fg(Color::DarkGray),
    )));

    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow));
    let paragraph = Paragraph::new(lines).block(block).wrap(Wrap { trim: false });
    f.render_widget(paragraph, popup_area);
}

/// Compute the current max read depth across all symbols in the file matching `rel_path`.
fn file_depth(app: &App, rel_path: &str) -> ReadDepth {
    let Some(file) = app
        .project_tree
        .files
        .iter()
        .find(|f| f.file_path.to_string_lossy() == rel_path)
    else {
        return ReadDepth::Unseen;
    };
    let mut best = ReadDepth::Unseen;
    walk_symbols(&file.symbols, &app.ledger, &mut best);
    best
}

fn walk_symbols(
    symbols: &[ambits::symbols::SymbolNode],
    ledger: &ambits::tracking::ContextLedger,
    best: &mut ReadDepth,
) {
    for sym in symbols {
        let d = ledger.depth_of(&sym.id);
        if d > *best {
            *best = d;
        }
        walk_symbols(&sym.children, ledger, best);
    }
}

/// Files that have been read since the compaction boundary — files present in
/// the current ledger but not in the pre-compaction snapshot.
fn files_after_compaction(app: &App, ev: &CompactionEvent) -> Vec<(String, ReadDepth)> {
    let mut out: Vec<(String, ReadDepth)> = Vec::new();
    for file in &app.project_tree.files {
        let path_str = file.file_path.to_string_lossy().into_owned();
        let mut best = ReadDepth::Unseen;
        walk_symbols(&file.symbols, &app.ledger, &mut best);
        if best.is_seen()
            && !ev
                .ledger_before
                .files_accessed
                .iter()
                .any(|p| p.to_string_lossy() == path_str)
        {
            out.push((path_str, best));
        }
    }
    out
}

fn depth_label(d: ReadDepth) -> &'static str {
    match d {
        ReadDepth::Unseen => "Unseen",
        ReadDepth::Stale => "Stale",
        ReadDepth::NameOnly => "NameOnly",
        ReadDepth::Overview => "Overview",
        ReadDepth::Signature => "Signature",
        ReadDepth::FullBody => "FullBody",
    }
}

fn depth_color(d: ReadDepth) -> Color {
    match d {
        ReadDepth::Unseen => Color::DarkGray,
        ReadDepth::Stale => Color::Red,
        ReadDepth::NameOnly => Color::Gray,
        ReadDepth::Overview => Color::Cyan,
        ReadDepth::Signature => Color::Blue,
        ReadDepth::FullBody => Color::Green,
    }
}

fn wrap_summary(text: &str, width: usize) -> Vec<String> {
    let width = width.max(20);
    let mut lines = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        if current.is_empty() {
            current.push_str(word);
        } else if current.len() + 1 + word.len() > width {
            lines.push(std::mem::take(&mut current));
            current.push_str(word);
        } else {
            current.push(' ');
            current.push_str(word);
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

/// Compute a centered subrect taking `percent_x` × `percent_y` of `area`.
fn centered_rect(area: Rect, percent_x: u16, percent_y: u16) -> Rect {
    let w = area.width * percent_x / 100;
    let h = area.height * percent_y / 100;
    let x = area.x + (area.width.saturating_sub(w)) / 2;
    let y = area.y + (area.height.saturating_sub(h)) / 2;
    Rect { x, y, width: w, height: h }
}
