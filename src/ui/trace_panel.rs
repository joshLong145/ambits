//! The trace view's right-hand panel: what the selected trace, or call in
//! it, amounts to (`trace::summary`), in the inspector's words and colours.
//! Its rows — files, agents, failures, commits, related calls — are what
//! `Enter` opens when the panel has focus (`App::trace_panel_targets`).

use std::ops::Range;

use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;

use ambits::app::{App, FocusPanel, PanelSubject};
use ambits::trace::summary::{self, FileActivity, Row, SymbolChange, TraceDetail};
use ambits::trace::view;
use ambits::trace::SpanKind;
use ambits::writes::{FileContents, Status, WriteRecord};

use super::inspector::{depth_spans, fact, text};
use super::trace_view::{instant_glyph, span_color, TraceFrame};
use super::{colors, fit, tree_view};

pub(super) fn render(f: &mut Frame, app: &App, area: Rect, frame: &TraceFrame<'_>) {
    let focused = app.focus == FocusPanel::Right;
    let subject = app.trace_panel_subject();
    let title = match subject {
        PanelSubject::Trace(root) => format!(" \"{}\" ", app.trace.spans()[root].name()),
        PanelSubject::Call(i) => format!(" {} ", app.trace.spans()[i].name()),
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
    // The line the selected row is on, to keep in view.
    let mut at = None;
    let mut with_rows = |mut out: Vec<Line<'static>>, rows: &[Row<'_>], write: Option<&WriteRecord>| {
        let (lines, picked) = rows_lines(app, frame, rows, write, selected, width);
        at = picked.map(|p| p.start + out.len()..p.end + out.len());
        out.extend(lines);
        out
    };
    let lines = match subject {
        PanelSubject::Trace(root) => match summary::detail(&app.trace, &frame.index, root) {
            Some(d) => with_rows(trace_lines(app, &d, width), &d.rows(), None),
            None => vec![Line::from(text(" nothing here", Color::DarkGray))],
        },
        PanelSubject::Call(i) => {
            let write = app.write_of(i);
            with_rows(call_lines(app, i, frame, width), &summary::call_rows(&app.trace, &frame.index, i, write), write)
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
    let height = inner.height as usize;
    // Scroll only as far as the selected row needs to show what it has
    // under it and a line more — never past the row itself.
    let scroll = at.map_or(0, |at| (at.end + 1).saturating_sub(height).min(at.start)).min(lines.len().saturating_sub(height));
    f.render_widget(Paragraph::new(lines).scroll((scroll as u16, 0)), inner);
}

/// A fact whose value wraps under its label, up to `max` lines, the last
/// marked `…` when there is more.
fn wrapped(label: &str, value: &str, color: Color, width: usize, max: usize) -> Vec<Line<'static>> {
    let (lines, total) = ambits::text::wrap(&ambits::text::clean(value), width.saturating_sub(12).max(8), max);
    let cut = total > lines.len();
    let last = lines.len().saturating_sub(1);
    lines
        .into_iter()
        .enumerate()
        .map(|(n, line)| fact(if n == 0 { label } else { "" }, vec![text(if cut && n == last { format!("{line} …") } else { line }, color)]))
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
    for line in ambits::text::wrap(&prompt, width.saturating_sub(2), 4).0 {
        out.push(Line::from(text(format!(" {line}"), Color::White)));
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
    // Its arguments and content, as far as they are loaded.
    let state = app.content_state(i);
    let has_args = matches!(state, ambits::app::ContentState::Loaded(d) if !d.args.is_empty());
    let mut out = vec![
        Line::from(Span::styled(format!(" {}", s.name()), Style::default().fg(span_color(app, statuses, i)).add_modifier(Modifier::BOLD))),
        fact("agent", vec![text(fit(&app.agent_title(&s.agent), width.saturating_sub(12)), Color::Gray)]),
        fact("when", vec![text(frame.timing(app, i), Color::Gray)]),
    ];
    if let Some(root) = index.root_of(i).filter(|r| *r != i) {
        out.push(fact("in", vec![text(fit(&format!("\"{}\"", app.trace.spans()[root].name()), width.saturating_sub(12)), Color::Gray)]));
    }
    if s.error {
        out.extend(wrapped("failed", s.message.as_deref().unwrap_or("✗"), Color::Red, width, 3));
    }

    match s.kind {
        // What it was credited with reading: one symbol by name, and how
        // deeply it is known now; several counted (they are rows below).
        // Credited with none, what it targeted, if anything.
        SpanKind::Read(depth) => {
            let mut read = depth_spans(depth, 0);
            match s.read.as_slice() {
                [(id, _)] => {
                    read.push(text(format!(" of {}", ambits::symbols::split_id(id).1), Color::Gray));
                    out.push(fact("read", read));
                    out.push(fact("now", depth_spans(app.ledger.depth_of(id), 0)));
                }
                [] => {
                    read.extend(s.symbol_name().map(|name| text(format!(" of {name}"), Color::Gray)));
                    out.push(fact("read", read));
                }
                many => {
                    read.push(text(format!(" · {} symbols", many.len()), Color::Gray));
                    out.push(fact("read", read));
                }
            }
        }
        SpanKind::Write => match s.id.as_deref().and_then(|op| statuses.get(op)) {
            // What it did in a line; the symbols themselves are rows below.
            Some((w, status)) => {
                let created = w.created.len();
                let counts = [(w.syms.len() - created, "edited"), (created, "created"), (w.removed.len(), "deleted")];
                let did: Vec<String> = counts.iter().filter(|(n, _)| *n > 0).map(|(n, what)| format!("{n} {what}")).collect();
                let level = if did.is_empty() { "file-level".to_string() } else { did.join(" · ") };
                out.push(fact("wrote", vec![text(level, Color::White), text("  ", Color::Gray), write_word(*status)]));
            }
            None => out.push(fact("wrote", vec![text("not attributed (no journal entry)", Color::DarkGray)])),
        },
        SpanKind::Delegate => {
            if let Some(d) = index.root_of(i).and_then(|r| summary::detail(&app.trace, index, r)) {
                if let Some(run) = d.agents.iter().find(|a| a.delegation == i) {
                    let mut started = vec![text(format!("a subagent · {} calls · {}", run.calls, view::duration(run.duration)), Color::White)];
                    if run.failed > 0 {
                        started.push(text(format!(" · {} failed", run.failed), Color::Red));
                    }
                    out.push(fact("started", started));
                }
            }
            // Its task is among its arguments, once they are loaded.
            if !has_args {
                out.extend(wrapped("task", &s.task(), Color::Gray, width, 4));
            }
        }
        SpanKind::Other | SpanKind::Prompt => {}
    }
    // A command or search — anything not about one file — in full, up to
    // six lines: a shell command can read (by naming symbols) as well.
    // Until its arguments are loaded, what it ran — a command or search
    // not about one file — in one line.
    if s.file.is_none() && s.kind != SpanKind::Delegate && !has_args {
        out.extend(wrapped("ran", &s.description, Color::Gray, width, 6));
    }
    out.extend(super::content::preview(app, i, state, width));
    out
}

/// What symbol-level write `w` left, by name path, and whether each still
/// stands against `now`, the file as the tree holds it (`None`: gone).
fn symbols_written(now: Option<&FileContents>, w: &WriteRecord) -> Vec<(String, Status)> {
    w.syms
        .iter()
        .map(|(id, _)| (ambits::symbols::split_id(id).1.to_string(), now.map_or(Status::Removed, |n| n.symbol_status(id, w))))
        .collect()
}

/// How a read or write of a whole file is named among symbols.
const WHOLE_FILE: &str = "(whole file)";

/// Detail lines a file gets under its row, at most.
const FILE_DETAIL: usize = 6;

/// The width of a detail line's state column: `still there`.
const STATE: usize = 11;

/// What became of a trace's write to a symbol, as its latest write left it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Wrote {
    /// Journaled: whether it still stands.
    Stands(Status),
    /// The call failed: nothing was written.
    Failed,
    /// It succeeded, but no journal entry says what it wrote.
    Unattributed,
}

/// Under a trace's file row: what it read of the file, each symbol at its
/// deepest, then what it wrote, each symbol as its latest write left it.
/// A failed call is listed as failed, and never outranks one that worked.
fn file_details(app: &App, frame: &TraceFrame<'_>, file: &FileActivity, width: usize) -> Vec<Line<'static>> {
    let spans = app.trace.spans();
    let now = (!file.writes.is_empty()).then(|| app.file_contents(&file.file)).flatten();
    let mut wrote: Vec<(String, Wrote)> = Vec::new();
    let mut note = |name: String, what: Wrote| match wrote.iter_mut().find(|(n, _)| *n == name) {
        Some(slot) if what != Wrote::Failed || slot.1 == Wrote::Failed => slot.1 = what,
        Some(_) => {}
        None => wrote.push((name, what)),
    };
    for &i in &file.writes {
        let s = &spans[i];
        match s.id.as_deref().and_then(|op| frame.statuses.get(op)) {
            Some((w, status)) if w.syms.is_empty() => note(WHOLE_FILE.to_string(), Wrote::Stands(*status)),
            Some((w, _)) => symbols_written(now.as_deref(), w).into_iter().for_each(|(name, status)| note(name, Wrote::Stands(status))),
            None => {
                let name = s.symbol_name().unwrap_or_else(|| WHOLE_FILE.to_string());
                note(name, if s.error { Wrote::Failed } else { Wrote::Unattributed });
            }
        }
    }
    let name_w = width.saturating_sub(5 + 6 + 2 + STATE + 1).max(8);
    let failed = || text(format!("✗ {:<STATE$}", "failed"), Color::Red);
    let read = file.symbols_read(&app.trace).into_iter().map(|(name, depth)| {
        let mut cells = vec![text("read  ", Color::DarkGray)];
        cells.extend(depth.map_or_else(|| vec![failed()], |d| depth_spans(d, STATE)));
        cells.push(text(format!(" {}", fit(name.as_deref().unwrap_or(WHOLE_FILE), name_w)), Color::White));
        cells
    });
    let written = wrote.into_iter().map(|(name, what)| {
        let state = match what {
            Wrote::Stands(status) => text(format!("✎ {:<STATE$}", status.word()), tree_view::write_color(status)),
            Wrote::Failed => failed(),
            Wrote::Unattributed => text(format!("? {:<STATE$}", "unjournaled"), Color::DarkGray),
        };
        vec![text("wrote ", Color::DarkGray), state, text(format!(" {}", fit(&name, name_w)), Color::White)]
    });
    let all: Vec<Vec<Span<'static>>> = read.chain(written).collect();
    let more = all.len().saturating_sub(FILE_DETAIL);
    let mut out: Vec<Line<'static>> = all
        .into_iter()
        .take(FILE_DETAIL)
        .map(|cells| {
            let mut line = vec![text("     ", Color::Reset)];
            line.extend(cells);
            Line::from(line)
        })
        .collect();
    if more > 0 {
        out.push(Line::from(text(format!("     … {more} more"), Color::DarkGray)));
    }
    out
}

/// A symbol row's name path, then — dimmed, as far as `room` allows — the
/// file it is in: one call can read symbols of the same name in several.
fn symbol_cells(id: &str, room: usize) -> Vec<Span<'static>> {
    let (file, name) = ambits::symbols::split_id(id);
    let name = fit(name, room.max(8));
    let rest = room.saturating_sub(super::width(&name) + 2);
    let mut cells = vec![text(format!(" {name}"), Color::White)];
    if rest >= 6 {
        cells.push(text(format!("  {}", fit(file, rest)), Color::DarkGray));
    }
    cells
}

/// The width of a files row's name column: what the counts leave.
fn file_name_width(width: usize) -> usize {
    width.saturating_sub(3 + 12 + 14).max(8)
}

/// The panel's selectable rows, under a heading per section, the selected
/// one marked: the same list, in the same order, that `Enter` opens. With
/// the lines the selected row and what it shows under it take.
fn rows_lines(app: &App, frame: &TraceFrame<'_>, rows: &[Row<'_>], write: Option<&WriteRecord>, selected: Option<usize>, width: usize) -> (Vec<Line<'static>>, Option<Range<usize>>) {
    let spans = app.trace.spans();
    let picked = selected.map(|s| s.min(rows.len().saturating_sub(1)));
    let mut at = None;
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
            out.push(super::heading(format!(" {title}")));
        }
        let cells = match *row {
            Row::File(file) => {
                let status = file.writes.last().and_then(|&i| spans[i].id.as_deref()).and_then(|op| frame.statuses.get(op)).map(|(_, s)| *s);
                let name_w = file_name_width(width);
                let name = fit(&file.file, name_w);
                let pad = name_w.saturating_sub(super::width(&name));
                let mut cells = vec![
                    text(format!("{name}{}", " ".repeat(pad)), Color::White),
                    text(format!(" {:>4} {:>5}  ", file.reads.len(), file.writes.len()), Color::Gray),
                ];
                cells.extend(status.map(write_word));
                cells
            }
            Row::Agent(run) => {
                let failed = if run.failed > 0 { format!(" · {} ✗", run.failed) } else { String::new() };
                let what = run.task.clone();
                vec![
                    text(fit(&what, width.saturating_sub(26).max(8)), Color::White),
                    text(format!("  {} · {} calls{failed}", view::duration(run.duration), run.calls), Color::Gray),
                ]
            }
            Row::Failed(i) => {
                let why = spans[i].message.as_deref().unwrap_or("failed");
                vec![text(fit(&format!("✗ {} — {why}", app.trace.spans()[i].name()), width.saturating_sub(4)), Color::Red)]
            }
            Row::Commit(i) => vec![text(fit(&app.trace.instants()[i].kind.label(), width.saturating_sub(4)), Color::Cyan)],
            Row::ThisFile(file) => vec![text(fit(file, width.saturating_sub(16)), Color::White), text("  → tree", Color::DarkGray)],
            Row::Read { id, depth } => {
                let mut cells = depth_spans(depth, STATE);
                cells.extend(symbol_cells(id, width.saturating_sub(STATE + 6)));
                cells
            }
            Row::Wrote { id, change, file } => {
                let (mark, color) = match change {
                    SymbolChange::Created => ("+ created", tree_view::write_color(Status::Current)),
                    SymbolChange::Edited => ("~ edited", colors::DEPTH_STALE),
                    SymbolChange::Deleted => ("− deleted", tree_view::write_color(Status::Removed)),
                };
                // Whether what it wrote is still there; a deleted symbol's
                // status is its deletion.
                let now = (change != SymbolChange::Deleted).then(|| write.map(|w| (app.file_contents(file), w))).flatten();
                let stands = now.map(|(now, w)| now.map_or(Status::Removed, |n| n.symbol_status(id, w)));
                let mut cells = vec![text(format!("{mark:<STATE$}"), color)];
                cells.extend(symbol_cells(id, width.saturating_sub(STATE + 18)));
                cells.extend(stands.map(|s| text(format!("  {}", s.word()), tree_view::write_color(s))));
                cells
            }
            Row::Related { span: j, before } => vec![
                text(format!("{} {} ", if before { "before" } else { "after " }, ambits::time::clock(spans[j].start)), Color::DarkGray),
                text(fit(&app.trace.spans()[j].name(), width.saturating_sub(24).max(8)), span_color(app, &frame.statuses, j)),
            ],
        };
        let pick = picked == Some(n);
        let first = out.len();
        let mut line = vec![Span::styled(if pick { " › " } else { "   " }, Style::default().fg(colors::HIGHLIGHT_FG))];
        line.extend(cells);
        let line = Line::from(line);
        out.push(if pick { line.style(Style::default().bg(colors::HIGHLIGHT_BG).add_modifier(Modifier::BOLD)) } else { line });
        if let Row::File(file) = row {
            out.extend(file_details(app, frame, file, width));
        }
        if pick {
            at = Some(first..out.len());
        }
    }
    (out, at)
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
            shown: Vec::new(),
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

    /// Under each file: what was read of it, and what was written and
    /// whether that stands (here the tree is empty, so it is gone).
    #[test]
    fn a_files_reads_and_writes_are_listed_under_it() {
        let mut app = app();
        app.record_write("sess", ambits::writes::WriteRecord {
            op: "e1".into(), a: "sess".into(), t: "2026-09-27T10:00:04Z".into(), tool: "Edit".into(), file: "src/a.rs".into(),
            level: ambits::writes::Level::Symbol, syms: vec![("src/a.rs::App/run".into(), "b3:x".into())],
            ..Default::default()
        });
        let text = screen(&app);
        for want in ["read  ● full body   (whole file)", "wrote ✎ gone        App/run", "wrote ✗ failed      (whole file)"] {
            assert!(text.contains(want), "{want}: {text}");
        }
        let lines: Vec<&str> = text.lines().collect();
        let a = lines.iter().position(|l| l.contains("src/a.rs")).unwrap();
        assert!(lines[a + 1].contains("read") && lines[a + 2].contains("wrote"), "under src/a.rs: {text}");
    }

    /// A read that failed saw nothing: it says so, and a write no journal
    /// entry explains is not passed off as one.
    #[test]
    fn failed_and_unjournaled_calls_say_so() {
        let mut app = app();
        call(&mut app, "r2", "Read", "src/c.rs", "2026-09-27T10:00:07Z", "2026-09-27T10:00:08Z", Some("File does not exist."));
        call(&mut app, "e3", "Edit", "src/c.rs", "2026-09-27T10:00:09Z", "2026-09-27T10:00:10Z", None);
        let text = screen(&app);
        for want in ["read  ✗ failed      (whole file)", "wrote ? unjournaled (whole file)"] {
            assert!(text.contains(want), "{want}: {text}");
        }
        let lines: Vec<&str> = text.lines().collect();
        let c = lines.iter().position(|l| l.contains("src/c.rs")).unwrap();
        assert!(lines[c + 1].contains("✗ failed") && !lines[c + 1].contains("full body"), "not passed off as a read: {text}");
    }

    /// With more rows than room, the panel scrolls to the selected one.
    #[test]
    fn the_selected_row_stays_in_view() {
        let mut app = app();
        for n in 0..20 {
            call(&mut app, &format!("m{n}"), "Read", &format!("src/m{n:02}.rs"), "2026-09-27T10:00:07Z", "2026-09-27T10:00:08Z", None);
        }
        app.focus = FocusPanel::Right;
        // Rows: a.rs, b.rs, m00–m19, the failure.
        app.panel_index = 21;
        let text = screen(&app);
        let lines: Vec<&str> = text.lines().collect();
        let at = lines.iter().position(|l| l.contains(" › ")).unwrap_or_else(|| panic!("the selected row is drawn: {text}"));
        assert!(lines[at].contains("src/m19.rs"), "{text}");
        assert!(lines[at + 1].contains("read  ● full body"), "with what it shows under it: {text}");
    }

    /// A call's content: asked for once, previewed in the panel, and `o`
    /// shows it in full, scrolled by hunk.
    #[test]
    fn a_calls_content_is_previewed_and_opens_in_full() {
        use ambits::ingest::content::{args, CallContent, CallDetail, DiffLine, Hunk};
        use crossterm::event::KeyCode;
        let mut app = app();
        app.trace_view.open_trace(0);
        app.trace_view.selected = Some(ambits::trace::view::Item::Span(2));
        let (key, kind) = app.content_request().expect("the edit's content is wanted");
        assert!(app.content_request().is_none(), "asked once");
        app.trace_view.close_trace();
        app.contents = ambits::app::CallContents::default();
        assert!(app.content_request().is_none(), "no trace open, nothing wanted");
        app.trace_view.open_trace(0);
        app.trace_view.selected = Some(ambits::trace::view::Item::Span(2));
        assert_eq!(app.content_request().map(|(k, _)| k), Some(key.clone()));
        assert_eq!((&*key.id, kind), ("e1", ambits::ingest::content::ContentKind::Write));
        assert!(app.content_request().is_none(), "asked once");
        assert!(screen(&app).contains("loading…"));

        let hunk = |at: u32| Hunk { old_start: Some(at), new_start: Some(at), lines: (0..10).map(|n| DiffLine::Added(format!("line {n}"))).collect() };
        let input = serde_json::json!({"file_path": "src/a.rs", "old_string": "x", "replace_all": false, "note": "a\nb\nc\nd\ne\nf"});
        let content = CallContent::Change { hunks: vec![hunk(1), hunk(50)], exact: true, cut: 0 };
        app.set_call_content(key, Some(CallDetail { args: args(&input), content: Some(content) }));
        let text = crate::ui::test_render::lines(60, 40, |f| render(f, &app, f.area(), &TraceFrame::new(&app))).join("\n");
        for want in ["arguments", "file_path    src/a.rs", "note         a", "             d", "             … 2 more lines", "replace_all  false", "old_string   1 line · in the change below", "change", "@@ -1,0 +1,10 @@", " 1 + line 0", "… 10 more · o opens in full"] {
            assert!(text.contains(want), "{want}: {text}");
        }

        press(&mut app, KeyCode::Char('o'));
        assert!(app.content_view.is_some());
        let full = crate::ui::test_render::lines(80, 20, |f| crate::ui::render(f, &app)).join("\n");
        for want in ["Edit src/a.rs · main", "arguments", "file_path    src/a.rs", "@@ -1,0 +1,10 @@", "    1 + line 0", "o/Esc close"] {
            assert!(full.contains(want), "{want}: {full}");
        }
        app.content_view.as_ref().unwrap().height.set(5);
        // Rows: arguments (a heading, three, six of the note), a blank,
        // change (a heading, two hunks of eleven): 34.
        press(&mut app, KeyCode::Char('n'));
        assert_eq!(app.content_view.as_ref().unwrap().scroll, 12, "the first hunk");
        press(&mut app, KeyCode::Char('n'));
        assert_eq!(app.content_view.as_ref().unwrap().scroll, 23, "the second");
        press(&mut app, KeyCode::Char('j'));
        assert_eq!(app.content_view.as_ref().unwrap().scroll, 24);
        press(&mut app, KeyCode::Char('G'));
        assert_eq!(app.content_view.as_ref().unwrap().scroll, 34 - 5, "the last page");
        press(&mut app, KeyCode::Esc);
        assert!(app.content_view.is_none());
    }

    /// `src/a.rs` holding `App` with `run` in it, lines 3-5.
    fn app_with_symbols() -> App {
        use ambits::symbols::{FileSymbols, SymbolCategory, SymbolNode};
        let node = |id: &str, name: &str, lines: std::ops::Range<u32>, children: Vec<SymbolNode>| SymbolNode {
            id: id.into(), name: name.into(), category: SymbolCategory::Function, label: "fn",
            file_path: Arc::new(PathBuf::from("src/a.rs")), byte_range: 0..10, line_range: lines,
            content_hash: [1; 32], merkle_hash: [1; 32], children, estimated_tokens: 5,
        };
        let run = node("src/a.rs::App/run", "run", 3..5, vec![]);
        let tree = ProjectTree { root: PathBuf::from("/test"), files: vec![FileSymbols { file_path: "src/a.rs".into(), symbols: vec![node("src/a.rs::App", "App", 1..6, vec![run])], total_lines: 6 }] };
        let mut app = App::new(tree, PathBuf::from("/test"));
        app.set_session_id(Some("sess".into()));
        app.process_prompt(&Prompt { agent_id: Arc::from("sess"), timestamp: "2026-09-27T10:00:00Z".into(), text: "go".into() });
        app
    }

    /// A read lists the symbols it read, and Enter on one shows it in the
    /// symbol tree — its file and the symbols around it unfolded.
    #[test]
    fn a_read_lists_its_symbols_and_enter_shows_one_in_the_tree() {
        use crossterm::event::KeyCode;
        let mut app = app_with_symbols();
        let mut c = crate::ui::test_render::tool_call("sess", "r1", "Read", "src/a.rs", "2026-09-27T10:00:01Z");
        c.target_symbol = Some("App/run".into());
        app.process_agent_event(c);
        app.trace_view.open = true;
        app.trace_view.open_trace(0);
        app.trace_view.selected = Some(ambits::trace::view::Item::Span(1));
        let text = crate::ui::test_render::lines(60, 30, |f| render(f, &app, f.area(), &TraceFrame::new(&app))).join("\n");
        for want in ["symbols read", "● full body   App/run"] {
            assert!(text.contains(want), "{want}: {text}");
        }
        assert!(app.tree_rows.iter().all(|r| r.symbol_id != "src/a.rs::App/run"), "folded away to begin with");
        press(&mut app, KeyCode::Tab);
        press(&mut app, KeyCode::Enter);
        assert!(!app.trace_view.open, "back to the tree");
        assert_eq!(app.focus, FocusPanel::Left);
        assert_eq!(app.tree_rows[app.selected_index].symbol_id, "src/a.rs::App/run", "selected, its file and App unfolded");
        assert_eq!(app.pending_editor_request, None, "no editor");
    }

    /// A read targeting `run` — a name, not a path — was credited with
    /// `App/run`: the panel says so, and following it lands there.
    #[test]
    fn a_read_is_named_by_what_it_was_credited_with() {
        use crossterm::event::KeyCode;
        let mut app = app_with_symbols();
        let mut c = crate::ui::test_render::tool_call("sess", "r1", "Read", "src/a.rs", "2026-09-27T10:00:01Z");
        c.target_symbol = Some("run".into());
        app.process_agent_event(c);
        assert_eq!(app.trace.spans()[1].read, vec![("src/a.rs::App/run".into(), ambits::tracking::ReadDepth::FullBody)]);
        app.trace_view.open = true;
        app.trace_view.open_trace(0);
        app.trace_view.selected = Some(ambits::trace::view::Item::Span(1));
        let text = crate::ui::test_render::lines(60, 20, |f| render(f, &app, f.area(), &TraceFrame::new(&app))).join("\n");
        for want in ["read     ● full body of App/run", "now      ● full body"] {
            assert!(text.contains(want), "{want}: {text}");
        }
        press(&mut app, KeyCode::Enter);
        assert_eq!(app.tree_rows[app.selected_index].symbol_id, "src/a.rs::App/run", "followed to what it read");
    }

    /// A write lists what it did to each symbol: edited, created, deleted.
    #[test]
    fn a_write_lists_what_it_did_to_each_symbol() {
        let mut app = app_with_symbols();
        call(&mut app, "e1", "Edit", "src/a.rs", "2026-09-27T10:00:01Z", "2026-09-27T10:00:02Z", None);
        app.record_write("sess", ambits::writes::WriteRecord {
            op: "e1".into(), a: "sess".into(), t: "2026-09-27T10:00:02Z".into(), tool: "Edit".into(), file: "src/a.rs".into(),
            level: ambits::writes::Level::Symbol,
            syms: vec![("src/a.rs::App".into(), ambits::journal::encode_hash(&[1; 32])), ("src/a.rs::App/run".into(), ambits::journal::encode_hash(&[1; 32]))],
            created: vec!["src/a.rs::App/run".into()],
            removed: vec!["src/a.rs::App/old".into()],
            ..Default::default()
        });
        app.trace_view.open = true;
        app.trace_view.open_trace(0);
        app.trace_view.selected = Some(ambits::trace::view::Item::Span(1));
        let text = crate::ui::test_render::lines(70, 30, |f| render(f, &app, f.area(), &TraceFrame::new(&app))).join("\n");
        for want in ["1 edited · 1 created · 1 deleted", "symbols written", "~ edited    App", "+ created   App/run", "− deleted   App/old"] {
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
