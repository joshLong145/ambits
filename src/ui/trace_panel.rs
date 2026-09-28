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
use ambits::trace::summary::{self, TraceDetail};
use ambits::trace::view;
use ambits::trace::SpanKind;
use ambits::writes::Status;

use super::inspector::{depth_word, fact, text};
use super::trace_view::{instant_glyph, span_color, span_name, Statuses};
use super::{colors, fit, tree_view};

pub fn render(f: &mut Frame, app: &App, area: Rect) {
    let focused = app.focus == FocusPanel::Right;
    let statuses = app.write_statuses();
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
    let rows = Rows { selected: focused.then_some(app.panel_index), next: 0, width };
    let lines = match subject {
        PanelSubject::Trace(root) => match summary::detail(&app.trace, root) {
            Some(d) => trace_lines(app, &d, &statuses, rows),
            None => vec![Line::from(text(" nothing here", Color::DarkGray))],
        },
        PanelSubject::Call(i) => call_lines(app, i, &statuses, rows),
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

/// Numbers the panel's selectable rows as they are drawn, in the order
/// `App::trace_panel_targets` lists them, and marks the selected one.
struct Rows {
    selected: Option<usize>,
    next: usize,
    width: usize,
}

impl Rows {
    fn row(&mut self, spans: Vec<Span<'static>>, count: usize) -> Line<'static> {
        let n = self.next;
        self.next += 1;
        let pick = self.selected.is_some_and(|s| s.min(count.saturating_sub(1)) == n);
        let mut out = vec![Span::styled(if pick { " › " } else { "   " }, Style::default().fg(colors::HIGHLIGHT_FG))];
        out.extend(spans);
        let line = Line::from(out);
        if pick {
            line.style(Style::default().bg(colors::HIGHLIGHT_BG).add_modifier(Modifier::BOLD))
        } else {
            line
        }
    }
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

fn section(name: &str) -> Line<'static> {
    Line::from(Span::styled(format!(" {name}"), Style::default().fg(Color::DarkGray).add_modifier(Modifier::BOLD)))
}

/// `✎ still there` and the like, in the write colours.
fn write_word(status: Status) -> Span<'static> {
    let word = match status {
        Status::Current => "still there",
        Status::Changed => "changed",
        Status::Removed => "gone",
        Status::Unknown => "file-level",
    };
    text(format!("✎ {word}"), tree_view::write_color(status))
}

/// A trace summed up: when and how long, what it called, the files it
/// touched and what became of its writes, its agents, failures and commits.
fn trace_lines(app: &App, d: &TraceDetail, statuses: &Statuses, mut rows: Rows) -> Vec<Line<'static>> {
    let spans = app.trace.spans();
    let count = d.targets().len();
    let mut out = Vec::new();

    // The prompt in full, up to four lines.
    let prompt: String = spans[d.root].description.split_whitespace().collect::<Vec<_>>().join(" ");
    let chars: Vec<char> = prompt.chars().collect();
    let wrap = rows.width.saturating_sub(2).max(1);
    for chunk in chars.chunks(wrap).take(4) {
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

    if !d.files.is_empty() {
        out.push(Line::from(""));
        let name_w = rows.width.saturating_sub(3 + 12 + 14).max(8);
        out.push(section(&format!("files{}read wrote", " ".repeat(name_w.saturating_sub(2)))));
        for file in &d.files {
            let status = file.write_spans.last().and_then(|&i| spans[i].id.as_deref()).and_then(|op| statuses.get(op)).map(|(_, s)| *s);
            let name = fit(&file.file, name_w);
            let pad = name_w.saturating_sub(super::width(&name));
            let mut line = vec![
                text(format!("{name}{}", " ".repeat(pad)), Color::White),
                text(format!(" {:>4} {:>5}  ", file.reads, file.writes), Color::Gray),
            ];
            line.extend(status.map(write_word));
            out.push(rows.row(line, count));
        }
    }
    if !d.agents.is_empty() {
        out.push(Line::from(""));
        out.push(section("agents"));
        for run in &d.agents {
            let failed = if run.failed > 0 { format!(" · {} ✗", run.failed) } else { String::new() };
            let what = run.description.trim_start_matches("Agent: ").to_string();
            let line = vec![
                text(fit(&what, rows.width.saturating_sub(26).max(8)), Color::White),
                text(format!("  {} · {} calls{failed}", view::duration(run.duration), run.calls), Color::Gray),
            ];
            out.push(rows.row(line, count));
        }
    }
    if !d.failed.is_empty() {
        out.push(Line::from(""));
        out.push(section("failed"));
        for &i in &d.failed {
            let why = spans[i].message.as_deref().unwrap_or("failed");
            let line = vec![text(fit(&format!("✗ {} — {why}", span_name(app, i)), rows.width.saturating_sub(4)), Color::Red)];
            out.push(rows.row(line, count));
        }
    }
    if !d.commits.is_empty() {
        out.push(Line::from(""));
        out.push(section("commits"));
        for &i in &d.commits {
            let line = vec![text(fit(&app.trace.instants()[i].kind.label(), rows.width.saturating_sub(4)), Color::Cyan)];
            out.push(rows.row(line, count));
        }
    }
    out
}

/// One call: who, when, how long, in which trace; why it failed; what it
/// read, wrote or ran; and the other calls on its file.
fn call_lines(app: &App, i: usize, statuses: &Statuses, mut rows: Rows) -> Vec<Line<'static>> {
    let s = &app.trace.spans()[i];
    let count = summary::call_targets(&app.trace, i).len();
    let end = view::effective_ends(&app.trace.tree()).get(&i).copied().filter(|_| s.end.is_some());
    let when = match end {
        Some(end) => format!("{} → {} ({})", ambits::time::clock(s.start), ambits::time::clock(end), view::duration(end - s.start)),
        None => format!("{} → running", ambits::time::clock(s.start)),
    };
    let mut out = vec![
        Line::from(Span::styled(format!(" {}", span_name(app, i)), Style::default().fg(span_color(app, statuses, i)).add_modifier(Modifier::BOLD))),
        fact("agent", vec![text(app.agent_name(&s.agent).to_string(), Color::Gray)]),
        fact("when", vec![text(when, Color::Gray)]),
    ];
    if let Some(root) = summary::root_of(&app.trace, i).filter(|r| *r != i) {
        out.push(fact("in", vec![text(fit(&format!("\"{}\"", app.trace.spans()[root].name()), rows.width.saturating_sub(12)), Color::Gray)]));
    }
    if s.error {
        out.extend(wrapped("failed", s.message.as_deref().unwrap_or("✗"), Color::Red, rows.width, 3));
    }

    match s.kind {
        SpanKind::Read(depth) => {
            let mut read = vec![text(format!("{} ", tree_view::depth_glyph(depth)), tree_view::depth_color(depth, false)), text(depth_word(depth), Color::White)];
            if let (Some(file), Some(sym)) = (&s.file, &s.symbol) {
                read.push(text(format!(" of {}", ambits::app::normalize_name_path(sym)), Color::Gray));
                let now = app.ledger.depth_of(&format!("{file}::{}", ambits::app::normalize_name_path(sym)));
                out.push(fact("read", read));
                out.push(fact("now", vec![text(format!("{} {}", tree_view::depth_glyph(now), depth_word(now)), tree_view::depth_color(now, false))]));
            } else {
                out.push(fact("read", read));
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
                    out.push(fact("", vec![text(fit(&name, rows.width.saturating_sub(26).max(8)), Color::White), text("  ", Color::Gray), write_word(status)]));
                }
                if w.syms.len() > 8 {
                    out.push(fact("", vec![text(format!("… {} more", w.syms.len() - 8), Color::DarkGray)]));
                }
            }
            None => out.push(fact("wrote", vec![text("not attributed (no journal entry)", Color::DarkGray)])),
        },
        SpanKind::Delegate => {
            if let Some(d) = summary::root_of(&app.trace, i).and_then(|r| summary::detail(&app.trace, r)) {
                if let Some(run) = d.agents.iter().find(|a| a.delegation == i) {
                    out.push(fact("agent", vec![text(format!("{} · {} calls · {}", run.agent, run.calls, view::duration(run.duration)), Color::White)]));
                    if run.failed > 0 {
                        out.push(fact("", vec![text(format!("{} failed", run.failed), Color::Red)]));
                    }
                }
            }
            out.extend(wrapped("task", s.description.trim_start_matches("Agent: "), Color::Gray, rows.width, 4));
        }
        SpanKind::Other | SpanKind::Prompt => {}
    }
    // A command or search — anything not about one file — in full, up to
    // six lines: a shell command can read (by naming symbols) as well.
    if s.file.is_none() && s.kind != SpanKind::Delegate {
        out.extend(wrapped("ran", &s.description, Color::Gray, rows.width, 6));
    }

    if let Some(file) = &s.file {
        out.push(Line::from(""));
        out.push(section("on this file"));
        out.push(rows.row(vec![text(fit(file, rows.width.saturating_sub(16)), Color::White), text("  → tree", Color::DarkGray)], count));
        for j in summary::related(&app.trace, i) {
            let other = &app.trace.spans()[j];
            let mark = if j < i { "before" } else { "after " };
            let line = vec![
                text(format!("{mark} {} ", ambits::time::clock(other.start)), Color::DarkGray),
                text(fit(&span_name(app, j), rows.width.saturating_sub(24).max(8)), span_color(app, statuses, j)),
            ];
            out.push(rows.row(line, count));
        }
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
        let c = ambits::ingest::AgentToolCall {
            agent_id: Arc::from("sess"),
            tool_name: Arc::from(tool),
            file_path: Some(PathBuf::from(format!("/test/{file}"))),
            read_depth: if tool == "Read" { ambits::tracking::ReadDepth::FullBody } else { ambits::tracking::ReadDepth::Unseen },
            description: format!("{tool} {file}"),
            timestamp_str: at.into(),
            target_symbol: None,
            target_lines: None,
            target_selectors: Vec::new(),
            label: Arc::from("main"),
            tool_use_id: Some(Arc::from(id)),
            effect: if tool == "Edit" { Effect::Write } else { Effect::Read },
        };
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
        crate::ui::test_render::lines(60, 24, |f| render(f, app, f.area())).join("\n")
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
}
