//! The inspector: the selected row's states in words — how deeply it was
//! read and by whom, whether that read is still in context and still
//! current, whether an agent wrote it and whether that still stands — and
//! the traces (prompts) whose calls read or wrote it. `Enter` opens one.

use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;

use ambits::app::{App, FocusPanel};
use ambits::expansion::RowKind;
use ambits::tracking::ReadDepth;

use super::{colors, tree_view};

/// Label column: `read     `.
const LABEL: usize = 9;

pub fn render(f: &mut Frame, app: &App, area: Rect) {
    let focused = app.focus == FocusPanel::Right;
    let border = if focused { Color::Cyan } else { Color::DarkGray };
    let Some(row) = app.tree_rows.get(app.selected_index) else {
        let block = Block::default().title(" Inspector ").borders(Borders::ALL).border_style(Style::default().fg(border));
        f.render_widget(Paragraph::new(" Nothing selected.").block(block), area);
        return;
    };
    let title = match row.kind {
        RowKind::File => format!(" {} ", row.display_name),
        RowKind::Symbol => format!(" {} ", ambits::symbols::split_id(&row.symbol_id).1),
    };
    let block = Block::default().title(super::fit(&title, area.width.saturating_sub(4) as usize)).borders(Borders::ALL).border_style(Style::default().fg(border));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let mut lines = match row.kind {
        RowKind::File => file_facts(app, row),
        RowKind::Symbol => symbol_facts(app, row),
    };
    lines.push(Line::from(""));
    let used = lines.len();
    lines.extend(traces(app, focused, (inner.height as usize).saturating_sub(used), inner.width as usize));
    // Facts wrap; trace rows are cut to the width, so they never do.
    f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);
}

pub(super) fn fact(label: &str, spans: Vec<Span<'static>>) -> Line<'static> {
    let mut out = vec![Span::styled(format!(" {label:<LABEL$}"), Style::default().fg(Color::DarkGray))];
    out.extend(spans);
    Line::from(out)
}

pub(super) fn text(s: impl Into<String>, color: Color) -> Span<'static> {
    Span::styled(s.into(), Style::default().fg(color))
}

pub(super) fn depth_word(depth: ReadDepth) -> &'static str {
    match depth {
        ReadDepth::Unseen => "not read",
        ReadDepth::NameOnly => "name only",
        ReadDepth::Overview => "overview",
        ReadDepth::Signature => "signature",
        ReadDepth::FullBody => "full body",
    }
}

/// `● full body`: a read depth's glyph, in its colour, and its word.
pub(super) fn depth_spans(depth: ReadDepth) -> Vec<Span<'static>> {
    vec![text(format!("{} ", tree_view::depth_glyph(depth)), tree_view::depth_color(depth, false)), text(depth_word(depth), Color::White)]
}

fn symbol_facts(app: &App, row: &ambits::app::TreeRow) -> Vec<Line<'static>> {
    let depth = row.read_depth;
    let mut lines = Vec::new();

    // Read: the depth, and who read it how deeply.
    let mut read = depth_spans(depth);
    if let Some(entry) = app.ledger.entries.get(&row.symbol_id).filter(|_| depth.is_seen()) {
        let mut by: Vec<(String, ReadDepth)> =
            entry.agent_depths.iter().filter(|(_, d)| d.is_seen()).map(|(a, d)| (app.agent_name(a).to_string(), *d)).collect();
        by.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let who: Vec<String> = by.iter().map(|(a, d)| if *d == depth { a.clone() } else { format!("{a} ({})", depth_word(*d)) }).collect();
        if !who.is_empty() {
            read.push(text(format!(" · {}", who.join(", ")), Color::Gray));
        }
    }
    lines.push(fact("read", read));

    if depth.is_seen() {
        lines.push(fact(
            "context",
            vec![if row.restored {
                text("◌ read before a compaction: no longer in context", colors::ACCENT_MUTED)
            } else {
                text("live", Color::Gray)
            }],
        ));
        lines.push(fact(
            "changed",
            vec![if row.stale { text("! yes, since it was read", colors::DEPTH_STALE) } else { text("no", Color::Gray) }],
        ));
    }

    written(app, row, &mut lines);
    lines.push(fact("size", vec![text(format!("{} · ~{} tokens", row.line_range, row.token_count), Color::Gray)]));
    if row.coverage_total > 0 {
        lines.push(fact("inside", vec![text(format!("{}/{} seen · {} full", row.coverage_seen, row.coverage_total, row.coverage_full), Color::Gray)]));
    }
    lines
}

fn file_facts(app: &App, row: &ambits::app::TreeRow) -> Vec<Line<'static>> {
    let mut lines = vec![fact(
        "coverage",
        vec![text(format!("{}/{} seen · {} full", row.coverage_seen, row.coverage_total, row.coverage_full), Color::White)],
    )];
    if row.stale_count > 0 {
        lines.push(fact("changed", vec![text(format!("! {} read since changed", row.stale_count), colors::DEPTH_STALE)]));
    }
    written(app, row, &mut lines);
    lines.push(fact("size", vec![text(row.line_range.clone(), Color::Gray)]));
    lines
}

/// `written  ✎ 2026-09-27 10:05Z by main (Edit)` and whether it stands.
fn written(app: &App, row: &ambits::app::TreeRow, lines: &mut Vec<Line<'static>>) {
    let Some((mark, w)) = row.write.as_ref().and_then(|m| Some((m, app.writes.get(&m.latest)?))) else {
        lines.push(fact("written", vec![text("not this session", Color::Gray)]));
        return;
    };
    let when = ambits::time::short(&w.t);
    let times = if mark.count > 1 { format!(" · {} writes", mark.count) } else { String::new() };
    let color = tree_view::write_color(mark.status);
    lines.push(fact("written", vec![text("✎ ", color), text(format!("{when} by {} ({}){times}", app.agent_name(&w.a), w.tool), Color::White)]));
    lines.push(fact("", vec![text(mark.status.phrase(), color)]));
}

/// The prompts whose calls touched the row, newest last; the selected one
/// marked when the inspector has focus.
fn traces(app: &App, focused: bool, room: usize, width: usize) -> Vec<Line<'static>> {
    let touches = app.inspector_touches();
    if touches.is_empty() {
        return vec![fact("traces", vec![text("no calls on it in this session", Color::Gray)])];
    }
    let hint = if focused { " · Enter opens" } else { " · Tab to choose" };
    let n = touches.len();
    let mut lines = vec![fact("traces", vec![text(format!("{n} prompt{}{hint}", if n == 1 { "" } else { "s" }), Color::White)])];
    let selected = app.panel_index.min(touches.len() - 1);
    let rows = room.saturating_sub(1).max(1);
    let first = selected.saturating_sub(rows - 1);
    for (ix, t) in touches.iter().enumerate().skip(first).take(rows) {
        let root = &app.trace.spans()[t.root];
        let what = match (t.read, t.wrote) {
            (true, true) => "read, wrote",
            (false, true) => "wrote",
            (true, false) => "read",
            (false, false) => "touched",
        };
        let pick = focused && ix == selected;
        let style = if pick { Style::default().bg(colors::HIGHLIGHT_BG).fg(colors::HIGHLIGHT_FG).add_modifier(Modifier::BOLD) } else { Style::default().fg(Color::Gray) };
        let when = ambits::time::day_minute(root.start);
        let lead = format!(" {} {when} ", if pick { "›" } else { " " });
        let what = format!("{what:<11} ");
        let room = width.saturating_sub(super::width(&lead) + what.len());
        lines.push(Line::from(vec![
            Span::styled(lead, style),
            Span::styled(what, style.fg(if t.wrote { colors::WRITE_CURRENT } else { Color::Gray })),
            Span::styled(super::fit(&format!("\"{}\"", root.name()), room), style),
        ]));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use ambits::ingest::{Prompt, ToolFinished};
    use ambits::symbols::{FileSymbols, ProjectTree, SymbolCategory, SymbolNode};
    use std::path::PathBuf;
    use std::sync::Arc;

    /// `mock/a.rs` with `alpha`, read in full by main and written, in a
    /// session with one prompt that read it.
    fn app() -> App {
        let hash = ambits::symbols::merkle::content_hash("alpha");
        let alpha = SymbolNode {
            id: "mock/a.rs::alpha".into(), name: "alpha".into(), category: SymbolCategory::Function,
            label: "fn", file_path: Arc::new(PathBuf::from("mock/a.rs")), byte_range: 0..10, line_range: 1..3,
            content_hash: hash, merkle_hash: hash, children: Vec::new(), estimated_tokens: 12,
        };
        let tree = ProjectTree { root: PathBuf::from("/test"), files: vec![FileSymbols { file_path: "mock/a.rs".into(), symbols: vec![alpha], total_lines: 3 }] };
        let mut app = App::new(tree, PathBuf::from("/test"));
        app.set_session_id(Some("sess".into()));
        app.set_expanded("mock/a.rs", RowKind::File, true);
        app.process_prompt(&Prompt { agent_id: Arc::from("sess"), timestamp: "2026-09-27T10:00:00Z".into(), text: "look at alpha".into() });
        let mut call = crate::ui::test_render::tool_call("sess", "r1", "Read", "mock/a.rs", "2026-09-27T10:00:01Z");
        call.label = Arc::from("main");
        app.process_agent_event(call);
        app.process_tool_finished(&ToolFinished { id: Arc::from("r1"), agent_id: Arc::from("sess"), timestamp: "2026-09-27T10:00:02Z".into(), error: false, message: None, child_agent: None });
        app.record_write("sess", ambits::writes::WriteRecord {
            op: "w1".into(), a: "sess".into(), t: "2026-09-27T10:00:03Z".into(), tool: "Edit".into(), file: "mock/a.rs".into(),
            level: ambits::writes::Level::Symbol, syms: vec![("mock/a.rs::alpha".into(), ambits::journal::encode_hash(&hash))],
            ..Default::default()
        });
        app.selected_index = 1;
        app
    }

    fn screen(app: &App) -> String {
        crate::ui::test_render::lines(70, 16, |f| render(f, app, f.area())).join("\n")
    }

    #[test]
    fn the_inspector_spells_out_a_symbols_states_and_its_traces() {
        let app = app();
        let text = screen(&app);
        for want in ["read     ● full body · main", "context  live", "changed  no", "written  ✎ 2026-09-27 10:00Z by main (Edit)", "unchanged since the agent wrote it", "traces   1 prompt", "read        \"look at alpha\""] {
            assert!(text.contains(want), "{want}: {text}");
        }
    }

    /// Tab focuses it; Enter opens the trace at the call that read the row.
    #[test]
    fn enter_opens_the_trace_at_the_call() {
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
        let mut app = app();
        app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert!(screen(&app).contains("Enter opens"));
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert!(app.trace_view.open);
        assert_eq!(app.trace_view.focus, Some(0), "the prompt's trace");
        assert_eq!(app.trace_view.selected, Some(ambits::trace::view::Item::Span(1)), "at the read");
    }

    /// `i` swaps in the session pane, `f` shows the activity feed.
    #[test]
    fn i_and_f_switch_what_is_shown() {
        use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
        let mut app = app();
        app.handle_key(KeyEvent::new(KeyCode::Char('i'), KeyModifiers::NONE));
        assert_eq!(app.right_pane, ambits::app::RightPane::Session);
        app.handle_key(KeyEvent::new(KeyCode::Char('f'), KeyModifiers::NONE));
        assert!(app.show_activity);
        app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(app.focus, FocusPanel::Feed, "the feed takes focus once shown");
    }
}
