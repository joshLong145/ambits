//! The trace view's right-hand panel: what the selected trace, or call in
//! it, amounts to (`trace::summary`), in the inspector's words and colours.
//! Its rows — files, agents, failures, commits, related calls — are what
//! `Enter` opens when the panel has focus (`App::trace_panel_targets`).

use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;

use ambits::app::{App, FocusPanel, PanelSubject};
use ambits::trace::summary::{self, Row, TraceDetail};
use ambits::trace::view;
use ambits::trace::SpanKind;
use ambits::writes::Status;

use super::inspector::{depth_spans, fact, text};
use super::trace_view::{instant_glyph, span_color, span_name, TraceFrame};
use super::{colors, fit, tree_view};

pub(super) fn render(f: &mut Frame, app: &App, area: Rect, frame: &TraceFrame<'_>) {
    let focused = app.focus == FocusPanel::Right;
    let subject = app.trace_panel_subject();
    let title = match subject {
        PanelSubject::Trace(root) => format!(" \"{}\" ", app.trace.spans()[root].name()),
        PanelSubject::Call(i) => format!(" {} ", span_name(app, i)),
        PanelSubject::Instant(i) => format!(" {} ", app.trace.instants()[i].kind.label()),
        PanelSubject::Nothing => " Trace ".to_string(),
    };
    let marker = if focused { "◆" } else { "" };
    let block = Block::default()
        .title(fit(&format!("{marker}{title}"), area.width.saturating_sub(4) as usize))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(if focused { Color::Cyan } else { Color::DarkGray }));
    let inner = block.inner(area);
    f.render_widget(block, area);
    let width = inner.width as usize;
    let selected = focused.then_some(app.panel_index);
    let lines = match subject {
        PanelSubject::Trace(root) => match summary::detail(&app.trace, &frame.index, root) {
            Some(d) => {
                let mut out = trace_lines(app, &d, width);
                out.extend(rows_lines(app, frame, &d.rows(), selected, width));
                out
            }
            None => vec![Line::from(text(" nothing here", Color::DarkGray))],
        },
        PanelSubject::Call(i) => {
            let mut out = call_lines(app, i, frame, width);
            out.extend(rows_lines(app, frame, &summary::call_rows(&app.trace, &frame.index, i), selected, width));
            out
        }
        PanelSubject::Instant(i) => {
            let x = &app.trace.instants()[i];
            let (glyph, color) = instant_glyph(&x.kind);
            vec![
                Line::from(text(format!(" {glyph} {}", x.kind.label()), color)),
                fact("when", vec![text(ambits::time::clock(x.t), Color::Gray)]),
            ]
        }
        PanelSubject::Nothing => vec![Line::from(text(" No traces yet.", Color::DarkGray))],
    };
    f.render_widget(Paragraph::new(lines), inner);
}

/// A fact whose value wraps under its label, up to `max` lines.
fn wrapped(label: &str, value: &str, color: Color, width: usize, max: usize) -> Vec<Line<'static>> {
    let wrap = width.saturating_sub(11).max(8);
    let chars: Vec<char> = value.chars().collect();
    let chunks: Vec<String> = chars.chunks(wrap).map(|c| c.iter().collect()).collect();
    let cut = chunks.len() > max;
    chunks
        .into_iter()
        .take(max)
        .enumerate()
        .map(|(n, chunk)| {
            let chunk = if cut && n + 1 == max { format!("{}…", chunk.chars().take(wrap - 1).collect::<String>()) } else { chunk };
            fact(if n == 0 { label } else { "" }, vec![text(chunk, color)])
        })
        .collect()
}

/// `✎ still there` and the like, in the write colours.
fn write_word(status: Status) -> Span<'static> {
    text(format!("✎ {}", status.word()), tree_view::write_color(status))
}

/// A trace summed up: the prompt, when and how long, what it called, how
/// many calls failed and commits it saw. Its rows follow.
fn trace_lines(app: &App, d: &TraceDetail, width: usize) -> Vec<Line<'static>> {
    let mut out = Vec::new();
    // The prompt in full, up to four lines.
    let prompt: String = app.trace.spans()[d.root].description.split_whitespace().collect::<Vec<_>>().join(" ");
    let chars: Vec<char> = prompt.chars().collect();
    for chunk in chars.chunks(width.saturating_sub(2).max(1)).take(4) {
        out.push(Line::from(text(format!(" {}", chunk.iter().collect::<String>()), Color::White)));
    }
    let agents = if d.agents.is_empty() { String::new() } else { format!(" · main + {} agent(s)", d.agents.len()) };
    out.push(Line::from(text(
        format!(" {} · {} · {} calls{agents}", ambits::time::day_minute(d.start), view::duration(d.end - d.start), d.calls),
        Color::Gray,
    )));
    if !d.by_tool.is_empty() {
        let tools: Vec<String> = d.by_tool.iter().take(5).map(|(t, n)| format!("{t} {n}")).collect();
        out.push(Line::from(text(format!(" {}", tools.join(" · ")), Color::DarkGray)));
    }
    let mut tally = Vec::new();
    if !d.failed.is_empty() {
        tally.push(text(format!(" {} failed", d.failed.len()), Color::Red));
    }
    if !d.commits.is_empty() {
        tally.push(text(format!(" {} commit(s)", d.commits.len()), Color::Cyan));
    }
    if !tally.is_empty() {
        out.push(Line::from(tally));
    }
    out
}

/// One call: who, when, how long, in which trace; why it failed; what it
/// read, wrote or ran. Its rows — its file, the other calls on it — follow.
fn call_lines(app: &App, i: usize, frame: &TraceFrame<'_>, width: usize) -> Vec<Line<'static>> {
    let s = &app.trace.spans()[i];
    let (index, statuses) = (&frame.index, &frame.statuses);
    let mut out = vec![
        Line::from(Span::styled(format!(" {}", span_name(app, i)), Style::default().fg(span_color(app, statuses, i)).add_modifier(Modifier::BOLD))),
        fact("agent", vec![text(app.agent_name(&s.agent).to_string(), Color::Gray)]),
        fact("when", vec![text(frame.timing(app, i), Color::Gray)]),
    ];
    if let Some(root) = index.root_of(i).filter(|r| *r != i) {
        out.push(fact("in", vec![text(fit(&format!("\"{}\"", app.trace.spans()[root].name()), width.saturating_sub(12)), Color::Gray)]));
    }
    if s.error {
        out.extend(wrapped("failed", s.message.as_deref().unwrap_or("✗"), Color::Red, width, 3));
    }

    match s.kind {
        SpanKind::Read(depth) => {
            let mut read = depth_spans(depth);
            read.extend(s.symbol_name().map(|name| text(format!(" of {name}"), Color::Gray)));
            out.push(fact("read", read));
            if let Some(id) = s.symbol_id() {
                out.push(fact("now", depth_spans(app.ledger.depth_of(&id))));
            }
        }
        SpanKind::Write => match s.id.as_deref().and_then(|op| statuses.get(op)) {
            Some((w, status)) => {
                let level = if w.syms.is_empty() { "file-level".to_string() } else { format!("{} symbol(s)", w.syms.len()) };
                out.push(fact("wrote", vec![text(level, Color::White), text("  ", Color::Gray), write_word(*status)]));
                let now = app.project_tree.file(&w.file).map(ambits::writes::FileContents::from_symbols);
                for (id, _) in w.syms.iter().take(8) {
                    let status = now.as_ref().map_or(Status::Removed, |n| n.symbol_status(id, w));
                    let name = ambits::symbols::split_id(id).1.to_string();
                    out.push(fact("", vec![text(fit(&name, width.saturating_sub(26).max(8)), Color::White), text("  ", Color::Gray), write_word(status)]));
                }
                if w.syms.len() > 8 {
                    out.push(fact("", vec![text(format!("… {} more", w.syms.len() - 8), Color::DarkGray)]));
                }
            }
            None => out.push(fact("wrote", vec![text("not attributed (no journal entry)", Color::DarkGray)])),
        },
        SpanKind::Delegate => {
            if let Some(d) = index.root_of(i).and_then(|r| summary::detail(&app.trace, index, r)) {
                if let Some(run) = d.agents.iter().find(|a| a.delegation == i) {
                    out.push(fact("agent", vec![text(format!("{} · {} calls · {}", run.agent, run.calls, view::duration(run.duration)), Color::White)]));
                    if run.failed > 0 {
                        out.push(fact("", vec![text(format!("{} failed", run.failed), Color::Red)]));
                    }
                }
            }
            out.extend(wrapped("task", s.description.trim_start_matches("Agent: "), Color::Gray, width, 4));
        }
        SpanKind::Other | SpanKind::Prompt => {}
    }
    // A command or search — anything not about one file — in full, up to
    // six lines: a shell command can read (by naming symbols) as well.
    if s.file.is_none() && s.kind != SpanKind::Delegate {
        out.extend(wrapped("ran", &s.description, Color::Gray, width, 6));
    }
    out
}

/// The width of a files row's name column: what the counts leave.
fn file_name_width(width: usize) -> usize {
    width.saturating_sub(3 + 12 + 14).max(8)
}

/// The panel's selectable rows, under a heading per section, the selected
/// one marked: the same list, in the same order, that `Enter` opens.
fn rows_lines(app: &App, frame: &TraceFrame<'_>, rows: &[Row<'_>], selected: Option<usize>, width: usize) -> Vec<Line<'static>> {
    let spans = app.trace.spans();
    let picked = selected.map(|s| s.min(rows.len().saturating_sub(1)));
    let mut out = Vec::new();
    let mut heading = None;
    for (n, row) in rows.iter().enumerate() {
        if heading != Some(row.section()) {
            heading = Some(row.section());
            let title = match row {
                Row::File(_) => format!("files{}read wrote", " ".repeat(file_name_width(width).saturating_sub(2))),
                _ => row.section().to_string(),
            };
            out.push(Line::from(""));
            out.push(Line::from(Span::styled(format!(" {title}"), Style::default().fg(Color::DarkGray).add_modifier(Modifier::BOLD))));
        }
        let cells = match *row {
            Row::File(file) => {
                let status = file.write_spans.last().and_then(|&i| spans[i].id.as_deref()).and_then(|op| frame.statuses.get(op)).map(|(_, s)| *s);
                let name_w = file_name_width(width);
                let name = fit(&file.file, name_w);
                let pad = name_w.saturating_sub(super::width(&name));
                let mut cells = vec![
                    text(format!("{name}{}", " ".repeat(pad)), Color::White),
                    text(format!(" {:>4} {:>5}  ", file.reads, file.writes), Color::Gray),
                ];
                cells.extend(status.map(write_word));
                cells
            }
            Row::Agent(run) => {
                let failed = if run.failed > 0 { format!(" · {} ✗", run.failed) } else { String::new() };
                let what = run.description.trim_start_matches("Agent: ").to_string();
                vec![
                    text(fit(&what, width.saturating_sub(26).max(8)), Color::White),
                    text(format!("  {} · {} calls{failed}", view::duration(run.duration), run.calls), Color::Gray),
                ]
            }
            Row::Failed(i) => {
                let why = spans[i].message.as_deref().unwrap_or("failed");
                vec![text(fit(&format!("✗ {} — {why}", span_name(app, i)), width.saturating_sub(4)), Color::Red)]
            }
            Row::Commit(i) => vec![text(fit(&app.trace.instants()[i].kind.label(), width.saturating_sub(4)), Color::Cyan)],
            Row::ThisFile(file) => vec![text(fit(file, width.saturating_sub(16)), Color::White), text("  → tree", Color::DarkGray)],
            Row::Related { span: j, before } => vec![
                text(format!("{} {} ", if before { "before" } else { "after " }, ambits::time::clock(spans[j].start)), Color::DarkGray),
                text(fit(&span_name(app, j), width.saturating_sub(24).max(8)), span_color(app, &frame.statuses, j)),
            ],
        };
        let pick = picked == Some(n);
        let mut line = vec![Span::styled(if pick { " › " } else { "   " }, Style::default().fg(colors::HIGHLIGHT_FG))];
        line.extend(cells);
        let line = Line::from(line);
        out.push(if pick { line.style(Style::default().bg(colors::HIGHLIGHT_BG).add_modifier(Modifier::BOLD)) } else { line });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ambits::ingest::{Effect, Prompt, ToolFinished};
    use ambits::symbols::ProjectTree;
    use std::path::PathBuf;
    use std::sync::Arc;

    fn call(app: &mut App, id: &str, tool: &str, file: &str, at: &str, end: &str, message: Option<&str>) {
        let mut c = crate::ui::test_render::tool_call("sess", id, tool, file, at);
        c.label = Arc::from("main");
        if tool == "Edit" {
            c.effect = Effect::Write;
            c.read_depth = ambits::tracking::ReadDepth::Unseen;
        }
        let root = app.project_root.clone();
        app.trace.start(&c, &root);
        app.trace.finish(&ToolFinished {
            id: Arc::from(id),
            agent_id: Arc::from("sess"),
            timestamp: end.into(),
            error: message.is_some(),
            child_agent: None,
            message: message.map(String::from),
        });
    }

    /// A prompt (0), a read (1) and an edit (2) of a.rs, a failed edit (3)
    /// of b.rs; the trace list open.
    fn app() -> App {
        let mut app = App::new(ProjectTree { root: PathBuf::from("/test"), files: vec![] }, PathBuf::from("/test"));
        app.set_session_id(Some("sess".into()));
        app.process_prompt(&Prompt { agent_id: Arc::from("sess"), timestamp: "2026-09-27T10:00:00Z".into(), text: "make it better".into() });
        call(&mut app, "r1", "Read", "src/a.rs", "2026-09-27T10:00:01Z", "2026-09-27T10:00:02Z", None);
        call(&mut app, "e1", "Edit", "src/a.rs", "2026-09-27T10:00:03Z", "2026-09-27T10:00:04Z", None);
        call(&mut app, "e2", "Edit", "src/b.rs", "2026-09-27T10:00:05Z", "2026-09-27T10:00:06Z", Some("String to replace not found in file."));
        app.trace_view.open = true;
        app
    }

    fn screen(app: &App) -> String {
        crate::ui::test_render::lines(60, 24, |f| render(f, app, f.area(), &TraceFrame::new(app))).join("\n")
    }

    #[test]
    fn the_list_shows_the_selected_traces_summary() {
        let app = app();
        let text = screen(&app);
        for want in ["make it better", "3 calls", "Edit 2 · Read 1", "1 failed", "src/a.rs", "src/b.rs", "✗ Edit src/b.rs — String to replace not found"] {
            assert!(text.contains(want), "{want}: {text}");
        }
    }

    #[test]
    fn a_failed_call_says_why_and_lists_its_file() {
        let mut app = app();
        app.trace_view.open_trace(0);
        app.trace_view.selected = Some(ambits::trace::view::Item::Span(3));
        let text = screen(&app);
        for want in ["Edit src/b.rs", "failed   String to replace not found in file.", "in       \"make it better\"", "on this file"] {
            assert!(text.contains(want), "{want}: {text}");
        }
    }

    #[test]
    fn a_call_lists_the_other_calls_on_its_file() {
        let mut app = app();
        app.trace_view.open_trace(0);
        app.trace_view.selected = Some(ambits::trace::view::Item::Span(2));
        app.focus = FocusPanel::Right;
        app.panel_index = 1;
        let text = screen(&app);
        assert!(text.contains("◆"), "focused: {text}");
        assert!(text.contains(" › before 10:00:01.0 Read src/a.rs"), "the selected row is marked: {text}");
        assert_eq!(app.trace_panel_targets().len(), 2, "the file and the read");
    }

    fn press(app: &mut App, code: crossterm::event::KeyCode) {
        app.handle_key(crossterm::event::KeyEvent::new(code, crossterm::event::KeyModifiers::NONE));
    }

    /// From the list, Enter on a failure opens its trace at that call; from
    /// a call, Enter on a related call goes to it; j stops at the last row.
    #[test]
    fn enter_on_a_row_goes_where_it_points() {
        use crossterm::event::KeyCode;
        let mut app = app();
        press(&mut app, KeyCode::Tab);
        assert_eq!(app.focus, FocusPanel::Right);
        // Rows: src/a.rs, src/b.rs, then the failed edit.
        for _ in 0..10 {
            press(&mut app, KeyCode::Char('j'));
        }
        assert_eq!(app.panel_index, 2, "stops at the last row");
        press(&mut app, KeyCode::Enter);
        assert_eq!(app.trace_view.focus, Some(0), "its trace opened");
        assert_eq!(app.trace_view.selected, Some(ambits::trace::view::Item::Span(3)), "at the failed call");
        assert_eq!(app.focus, FocusPanel::Right, "focus stays, to go on from there");

        app.trace_view.selected = Some(ambits::trace::view::Item::Span(2));
        press(&mut app, KeyCode::Char('j'));
        press(&mut app, KeyCode::Enter);
        assert_eq!(app.trace_view.selected, Some(ambits::trace::view::Item::Span(1)), "the read before it on a.rs");
    }
}
