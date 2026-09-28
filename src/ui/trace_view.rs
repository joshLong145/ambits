//! The trace view (`t`): first a list of traces, one per prompt; `Enter`
//! on one shows its tool calls on a time axis, as an OpenTelemetry
//! waterfall or as Perfetto-style agent tracks (`v`). Layout comes from
//! [`ambits::trace::view`]; this only draws it.

use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;

use ambits::app::{App, TraceGeometry};
use ambits::trace::view::{self, Item, Layout, Viewport};
use ambits::trace::{InstantKind, SpanKind};

use super::colors;

/// Each write's record and status by op, computed once per frame.
pub(super) type Statuses<'a> = std::collections::HashMap<&'a str, (&'a ambits::writes::WriteRecord, ambits::writes::Status)>;

/// What one frame of the trace view works out once, for the timeline and
/// the right-hand panel alike: the trace's structure, and every write's
/// status.
pub(super) struct TraceFrame<'a> {
    pub index: ambits::trace::summary::TraceIndex,
    pub statuses: Statuses<'a>,
}

impl<'a> TraceFrame<'a> {
    pub fn new(app: &'a App) -> Self {
        Self { index: ambits::trace::summary::TraceIndex::new(&app.trace), statuses: app.write_statuses() }
    }

    /// `10:00:01.2 → 10:00:03.9 (2.7s)`, or `10:00:01.2 → running`.
    pub fn timing(&self, app: &App, span: usize) -> String {
        let s = &app.trace.spans()[span];
        match self.index.end_of(&app.trace, span) {
            Some(end) => format!("{} → {} ({})", ambits::time::clock(s.start), ambits::time::clock(end), view::duration(end - s.start)),
            None => format!("{} → running", ambits::time::clock(s.start)),
        }
    }

    /// How long `span` took, or `running`.
    pub fn took(&self, app: &App, span: usize) -> String {
        self.index.end_of(&app.trace, span).map_or("running".to_string(), |e| view::duration(e - app.trace.spans()[span].start))
    }
}

/// Lines the summary strip takes at the bottom; the right-hand panel has
/// the rest (`trace_panel`).
const DETAILS: u16 = 1;
/// Columns between ticks on the ruler.
const TICK_GAP: usize = 12;

pub(super) fn render(f: &mut Frame, app: &App, area: Rect, frame: &TraceFrame<'_>) {
    let tv = &app.trace_view;
    let layout = match tv.layout {
        Layout::Waterfall => "waterfall",
        Layout::Tracks => "tracks",
    };
    let who = app.agent_filter.as_deref().map_or("all agents".to_string(), |a| app.agent_name(a).to_string());
    let title = match tv.focus {
        Some(root) => format!(" Trace — {} · {who} · {layout} ", super::fit(&app.trace.spans()[root].name(), 48)),
        None => format!(" Traces — {who} "),
    };
    let block = Block::default().title(title).borders(Borders::ALL).border_style(Style::default().fg(Color::Cyan));
    let inner = block.inner(area);
    f.render_widget(block, area);
    // Summary, ruler, at least two rows, the separator and the details.
    if inner.height < DETAILS + 5 || inner.width < 30 {
        return;
    }

    if app.trace.range().is_none() {
        f.render_widget(Paragraph::new(" No tool calls yet.").style(Style::default().fg(Color::DarkGray)), inner);
        return;
    }
    if tv.focus.is_none() {
        render_list(f, app, inner);
        return;
    }
    let Some(range) = tv.range(&app.trace) else { return };
    let vp = tv.viewport(&app.trace);
    let label_w = match tv.layout {
        Layout::Waterfall => (inner.width as usize * 2 / 5).max(24),
        Layout::Tracks => (inner.width as usize / 5).clamp(14, 28),
    };
    let dur_w = if tv.layout == Layout::Waterfall { 8 } else { 0 };
    let bars_x = inner.x + (label_w + dur_w + 1) as u16;
    let bars_w = inner.width.saturating_sub((label_w + dur_w + 1) as u16) as usize;
    let rows_y = inner.y + 2;
    let rows_h = inner.height - 3 - DETAILS;

    let mut lines = vec![summary(app, range), ruler(&vp, bars_w, range.0, label_w + dur_w + 1)];
    let (body, first_row) = match tv.layout {
        Layout::Waterfall => waterfall_lines(app, &frame.statuses, &vp, label_w, bars_w, rows_h as usize),
        Layout::Tracks => track_lines(app, frame, &vp, label_w, bars_w, rows_h as usize),
    };
    lines.extend(body);
    lines.resize(2 + rows_h as usize, Line::from(""));
    lines.push(Line::from(Span::styled("─".repeat(inner.width as usize), Style::default().fg(Color::DarkGray))));
    lines.extend(details(app, frame));
    f.render_widget(Paragraph::new(lines), inner);

    app.trace_geometry.set(Some(TraceGeometry { bars_x, bars_width: bars_w as u16, rows_y, rows: rows_h, first_row }));
}

/// The traces, one row per prompt: when, what was asked, how long it took,
/// how many calls, failures and subagents. The selected prompt in full
/// below.
fn render_list(f: &mut Frame, app: &App, inner: Rect) {
    let traces = app.trace_list();
    let dim = Style::default().fg(Color::DarkGray);
    let rows_h = inner.height - 3 - DETAILS;
    let selected = app.trace_view.list_index(&traces);
    let first = scroll(app, selected, traces.len(), rows_h as usize);
    let prompts = traces.iter().filter(|t| app.trace.spans()[t.root].kind == SpanKind::Prompt).count();

    // when · prompt · took · calls · failed · agents · commits
    let text_w = (inner.width as usize).saturating_sub(13 + 8 + 10 + 9 + 9 + 9);
    let mut lines = vec![
        Line::from(Span::styled(
            format!(" {prompts} prompts · {} traces · Enter opens one", traces.len()),
            Style::default().fg(Color::Gray),
        )),
        Line::from(Span::styled(
            format!(" {:<11} {:<text_w$} {:>7} {:>9} {:>8} {:>8} {:>8}", "when", "prompt", "took", "calls", "failed", "agents", "commits"),
            dim,
        )),
    ];
    for (ix, t) in traces.iter().enumerate().skip(first).take(rows_h as usize) {
        let root = &app.trace.spans()[t.root];
        let style = if Some(ix) == selected { selected_style() } else { Style::default() };
        let what = match root.kind {
            SpanKind::Prompt => root.name(),
            _ => format!("(before any prompt) {}", root.name()),
        };
        let what = super::fit(&what, text_w);
        let pad = text_w.saturating_sub(super::width(&what));
        let failed = if t.failed > 0 { format!("{} ✗", t.failed) } else { String::new() };
        let agents = if t.agents > 0 { t.agents.to_string() } else { String::new() };
        let commits = if t.commits > 0 { t.commits.to_string() } else { String::new() };
        lines.push(Line::from(vec![
            Span::styled(format!(" {:<11} ", ambits::time::day_minute(t.start)), style.fg(Color::DarkGray)),
            Span::styled(format!("{what}{}", " ".repeat(pad)), style.fg(Color::White)),
            Span::styled(format!(" {:>7}", view::duration(t.end - t.start)), style.fg(Color::Gray)),
            Span::styled(format!(" {:>9}", t.calls), style.fg(Color::Gray)),
            Span::styled(format!(" {failed:>8}"), style.fg(Color::Red)),
            Span::styled(format!(" {agents:>8}"), style.fg(Color::Gray)),
            Span::styled(format!(" {commits:>8}"), style.fg(Color::Cyan)),
        ]));
    }
    lines.resize(2 + rows_h as usize, Line::from(""));
    lines.push(Line::from(Span::styled("─".repeat(inner.width as usize), dim)));
    if let Some(t) = selected.map(|i| &traces[i]) {
        let line = format!(
            " {} · {} · {} calls{} · Tab for its summary",
            ambits::time::day_minute(t.start),
            view::duration(t.end - t.start),
            t.calls,
            if t.failed > 0 { format!(" · {} failed", t.failed) } else { String::new() }
        );
        lines.push(Line::from(Span::styled(super::fit(&line, inner.width as usize), Style::default().fg(Color::Gray))));
    }
    f.render_widget(Paragraph::new(lines), inner);
    app.trace_geometry.set(Some(TraceGeometry { bars_x: inner.x, bars_width: 0, rows_y: inner.y + 2, rows: rows_h, first_row: first }));
}

/// `12m04s · 1,204 calls · 3 failed · 0s–12m04s shown`
fn summary(app: &App, (start, end): (u64, u64)) -> Line<'static> {
    let (calls, failed) = match app.trace_view.focus.and_then(|r| app.trace_list().into_iter().find(|t| t.root == r)) {
        Some(t) => (t.calls, t.failed),
        None => (app.trace.spans().len(), app.trace.spans().iter().filter(|s| s.error).count()),
    };
    let vp = app.trace_view.viewport(&app.trace);
    let shown = format!("{}–{}", view::offset(vp.start.saturating_sub(start)), view::offset(vp.end.saturating_sub(start)));
    let mut out = vec![
        Span::styled(format!(" {}", view::duration(end - start)), Style::default().fg(Color::White)),
        Span::styled(format!(" · {calls} calls"), Style::default().fg(Color::Gray)),
    ];
    if failed > 0 {
        out.push(Span::styled(format!(" · {failed} failed"), Style::default().fg(Color::Red)));
    }
    let zoom = if app.trace_view.viewport.is_some() { "" } else { " (all, following)" };
    let zoom = format!("{zoom} · Esc: all traces");
    out.push(Span::styled(format!(" · {shown} shown{zoom}"), Style::default().fg(Color::DarkGray)));
    if !app.trace_view.query.is_empty() || app.trace_view.typing {
        let cursor = if app.trace_view.typing { "_" } else { "" };
        out.push(Span::styled(format!("  /{}{cursor}", app.trace_view.query), Style::default().fg(Color::Yellow)));
    }
    Line::from(out)
}

/// The time axis: `┊label` at each tick, after `indent` columns.
fn ruler(vp: &Viewport, cols: usize, origin: u64, indent: usize) -> Line<'static> {
    let mut cells = vec![' '; cols];
    for (col, label) in view::ticks(vp, cols, origin, TICK_GAP) {
        for (k, ch) in std::iter::once('┊').chain(label.chars()).enumerate() {
            if let Some(cell) = cells.get_mut(col + k) {
                *cell = ch;
            }
        }
    }
    Line::from(vec![Span::raw(" ".repeat(indent)), Span::styled(cells.into_iter().collect::<String>(), Style::default().fg(Color::DarkGray))])
}

/// The first row to draw so `selected` stays on screen, moving as little
/// as possible from the last frame's.
fn scroll(app: &App, selected: Option<usize>, total: usize, height: usize) -> usize {
    let previous = app.trace_geometry.get().map_or(0, |g| g.first_row);
    let first = match selected {
        Some(s) if s < previous => s,
        Some(s) if s >= previous + height => s + 1 - height,
        _ => previous,
    };
    first.min(total.saturating_sub(height))
}

fn waterfall_lines(app: &App, statuses: &Statuses, vp: &Viewport, label_w: usize, bars_w: usize, height: usize) -> (Vec<Line<'static>>, usize) {
    let rows = app.trace_rows();
    let selected = app.trace_view.row_index(&rows);
    let first = scroll(app, selected, rows.len(), height);
    let lines = rows
        .iter()
        .enumerate()
        .skip(first)
        .take(height)
        .map(|(ix, row)| {
            let is_selected = Some(ix) == selected;
            let indent = "  ".repeat(row.depth.min(12));
            let (label, duration, bar_line) = match row.item {
                Item::Span(i) => {
                    let s = &app.trace.spans()[i];
                    let fold = match (row.descendants, row.collapsed) {
                        (0, _) => "  ",
                        (_, true) => "▸ ",
                        (_, false) => "▾ ",
                    };
                    let mut label = format!("{indent}{fold}{}{}", if s.error { "✗ " } else { "" }, span_name(app, i));
                    if row.collapsed {
                        label.push_str(&format!(" ({} spans)", row.descendants));
                    }
                    let duration = match s.end {
                        Some(_) => view::duration(row.end - s.start),
                        None => "running".into(),
                    };
                    let color = span_color(app, statuses, i);
                    let cells = view::bar(vp.col(s.start, bars_w), vp.col(row.end.max(s.start), bars_w), bars_w);
                    (label, duration, cells_line(bars_w, cells.into_iter().map(|(c, ch)| (c, ch, Style::default().fg(color)))))
                }
                Item::Instant(i) => {
                    let x = &app.trace.instants()[i];
                    let (glyph, color) = instant_glyph(&x.kind);
                    let col = vp.col(x.t, bars_w);
                    let cells = (0.0..bars_w as f64).contains(&col).then_some((col as usize, glyph, Style::default().fg(color)));
                    (format!("{indent}  {glyph} {}", x.kind.label()), String::new(), cells_line(bars_w, cells))
                }
            };
            let style = if is_selected { selected_style() } else { Style::default() };
            let mut spans = vec![
                Span::styled(super::fit(&format!(" {label}"), label_w), style.fg(label_color(app, row.item))),
                Span::styled(" ".repeat(label_w.saturating_sub(super::width(&super::fit(&format!(" {label}"), label_w)))), style),
                Span::styled(format!("{duration:>7} "), style.fg(Color::DarkGray)),
                Span::raw(" "),
            ];
            spans.extend(bar_line);
            Line::from(spans)
        })
        .collect();
    (lines, first)
}

fn track_lines(app: &App, frame: &TraceFrame<'_>, vp: &Viewport, label_w: usize, bars_w: usize, height: usize) -> (Vec<Line<'static>>, usize) {
    let (tracks, rows) = app.trace_tracks();
    let selected_row = app.trace_view.track_row.min(rows.len().saturating_sub(1));
    let first = scroll(app, Some(selected_row), rows.len(), height);
    let open_end = app.trace_open_end();
    let selected_span = match app.trace_view.selected {
        Some(Item::Span(i)) => Some(i),
        _ => None,
    };
    let statuses = &frame.statuses;
    let lines = rows
        .iter()
        .enumerate()
        .skip(first)
        .take(height)
        .map(|(ix, row)| {
            let track = &tracks[row.track];
            let first_lane = row.lane.is_none_or(|l| l == 0);
            let label = if first_lane {
                let fold = if row.lane.is_none() { "▸ " } else { "▾ " };
                format!(" {}{fold}{}", "  ".repeat(track.depth.min(6)), track_name(app, track))
            } else {
                String::new()
            };
            let label_style = if ix == selected_row { selected_style() } else { Style::default().fg(Color::Gray) };
            let label = super::fit(&label, label_w);
            let pad = label_w.saturating_sub(super::width(&label));
            let mut spans = vec![Span::styled(label, label_style), Span::styled(" ".repeat(pad + 1), label_style)];

            let mut cells: Vec<(usize, char, Style)> = Vec::new();
            let on_row = row.spans(&tracks);
            match row.lane {
                None => {
                    let density = view::density(&app.trace, &on_row, vp, bars_w, open_end);
                    let max = density.iter().copied().max().unwrap_or(0).max(1);
                    const LEVELS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
                    for (c, n) in density.into_iter().enumerate().filter(|(_, n)| *n > 0) {
                        cells.push((c, LEVELS[((n * 8).div_ceil(max) as usize).clamp(1, 8) - 1], Style::default().fg(Color::Gray)));
                    }
                }
                Some(_) => {
                    for &i in &on_row {
                        let s = &app.trace.spans()[i];
                        let end = frame.index.effective_end(i).or(s.end).unwrap_or(open_end).max(s.start);
                        let (from, to) = (vp.col(s.start, bars_w), vp.col(end, bars_w));
                        let color = span_color(app, statuses, i);
                        let mut style = Style::default().fg(color);
                        if Some(i) == selected_span {
                            style = style.add_modifier(Modifier::REVERSED);
                        }
                        let bar = view::bar(from, to, bars_w);
                        // The name inside the bar, when there is room.
                        let name: Vec<char> = span_name(app, i).chars().collect();
                        let inside: Vec<usize> = bar.iter().filter(|(_, ch)| *ch == '█').map(|(c, _)| *c).collect();
                        let text = (inside.len() > 4).then(|| name.iter().take(inside.len() - 1));
                        cells.extend(bar.iter().map(|&(c, ch)| (c, ch, style)));
                        if let Some(text) = text {
                            let on_bar = Style::default().fg(Color::Black).bg(color);
                            let on_bar = if Some(i) == selected_span { on_bar.add_modifier(Modifier::BOLD | Modifier::UNDERLINED) } else { on_bar };
                            cells.extend(inside.iter().zip(text).map(|(&c, &ch)| (c, ch, on_bar)));
                        }
                    }
                }
            }
            if first_lane {
                for &i in &track.instants {
                    let x = &app.trace.instants()[i];
                    let col = vp.col(x.t, bars_w);
                    if (0.0..bars_w as f64).contains(&col) {
                        let (glyph, color) = instant_glyph(&x.kind);
                        cells.push((col as usize, glyph, Style::default().fg(color)));
                    }
                }
            }
            spans.extend(cells_line(bars_w, cells));
            Line::from(spans)
        })
        .collect();
    (lines, first)
}

/// `cols` cells with `cells` drawn over blanks (later ones on top), grouped
/// into runs of one style.
fn cells_line(cols: usize, cells: impl IntoIterator<Item = (usize, char, Style)>) -> Vec<Span<'static>> {
    let mut grid = vec![(' ', Style::default()); cols];
    for (c, ch, style) in cells {
        if let Some(cell) = grid.get_mut(c) {
            *cell = (ch, style);
        }
    }
    let mut out: Vec<Span<'static>> = Vec::new();
    let mut run = String::new();
    let mut run_style = Style::default();
    for (ch, style) in grid {
        if style != run_style && !run.is_empty() {
            out.push(Span::styled(std::mem::take(&mut run), run_style));
        }
        run_style = style;
        run.push(ch);
    }
    if !run.is_empty() {
        out.push(Span::styled(run, run_style));
    }
    out
}

/// The selected item in one line; the right-hand panel has the rest.
fn details(app: &App, frame: &TraceFrame<'_>) -> Vec<Line<'static>> {
    let statuses = &frame.statuses;
    let dim = Style::default().fg(Color::DarkGray);
    let line = match app.trace_view.selected {
        Some(Item::Span(i)) => {
            let Some(s) = app.trace.spans().get(i) else { return Vec::new() };
            let took = frame.took(app, i);
            let mut spans = vec![
                Span::styled(format!(" {}", span_name(app, i)), Style::default().fg(span_color(app, statuses, i)).add_modifier(Modifier::BOLD)),
                Span::styled(format!(" · {} · {took}", app.agent_name(&s.agent)), dim),
            ];
            if s.error {
                spans.push(Span::styled(format!(" · ✗ {}", s.message.as_deref().unwrap_or("failed")), Style::default().fg(Color::Red)));
            }
            Line::from(spans)
        }
        Some(Item::Instant(i)) => {
            let Some(x) = app.trace.instants().get(i) else { return Vec::new() };
            let (glyph, color) = instant_glyph(&x.kind);
            Line::from(vec![Span::styled(format!(" {glyph} {}", x.kind.label()), Style::default().fg(color)), Span::styled(format!(" · {}", ambits::time::clock(x.t)), dim)])
        }
        None => Line::from(Span::styled(" j/k to select a call · Tab for the trace's summary", dim)),
    };
    vec![line]
}

/// What a span is called on screen: a delegation by its description and
/// the agent it started.
pub(super) fn span_name(app: &App, i: usize) -> String {
    let s = &app.trace.spans()[i];
    match (&s.kind, &s.child_agent) {
        (SpanKind::Delegate, Some(child)) => format!("{} → {}", s.description, child),
        _ => s.name(),
    }
}

/// A subagent by what it was started for (its id is in the details), the
/// session's own agent as `main`.
fn track_name(app: &App, track: &view::Track) -> String {
    match track.delegation.map(|d| app.trace.spans()[d].description.trim_start_matches("Agent: ").trim()) {
        Some(what) if !what.is_empty() => what.to_string(),
        _ => app.agent_name(&track.agent).to_string(),
    }
}

/// Reads in their depth colour, writes by whether they still stand,
/// failures red, delegations neutral.
pub(super) fn span_color(app: &App, statuses: &Statuses, i: usize) -> Color {
    let s = &app.trace.spans()[i];
    if s.error {
        return Color::Red;
    }
    match s.kind {
        SpanKind::Read(depth) => super::tree_view::depth_color(depth, false),
        SpanKind::Write => match s.id.as_deref().and_then(|op| statuses.get(op)) {
            Some((_, status)) => super::tree_view::write_color(*status),
            None => colors::WRITE_UNKNOWN,
        },
        SpanKind::Delegate => Color::Gray,
        SpanKind::Prompt => Color::Cyan,
        SpanKind::Other => Color::Rgb(150, 150, 150),
    }
}

fn label_color(app: &App, item: Item) -> Color {
    match item {
        Item::Span(i) if app.trace.spans()[i].error => Color::Red,
        Item::Span(_) => Color::White,
        Item::Instant(i) => instant_glyph(&app.trace.instants()[i].kind).1,
    }
}

pub(super) fn instant_glyph(kind: &InstantKind) -> (char, Color) {
    match kind {
        InstantKind::Compaction => ('▼', Color::Yellow),
        InstantKind::Snapshot(_) => ('◆', Color::Magenta),
        InstantKind::Commit { .. } => ('│', Color::Cyan),
    }
}

fn selected_style() -> Style {
    Style::default().bg(colors::HIGHLIGHT_BG).fg(colors::HIGHLIGHT_FG).add_modifier(Modifier::BOLD)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ambits::ingest::ToolFinished;
    use ambits::symbols::ProjectTree;
    use std::path::PathBuf;
    use std::sync::Arc;

    fn call(agent: &str, id: &str, tool: &str, at: &str) -> ambits::ingest::AgentToolCall {
        crate::ui::test_render::tool_call(agent, id, tool, "src/a.rs", at)
    }

    fn app() -> App {
        let mut app = App::new(ProjectTree { root: PathBuf::from("/test"), files: vec![] }, PathBuf::from("/test"));
        app.set_session_id(Some("sess".into()));
        let finish = |id: &str, at: &str, child: Option<&str>, error: bool| ToolFinished {
            id: Arc::from(id),
            agent_id: Arc::from("sess"),
            timestamp: at.into(),
            error,
            message: None, child_agent: child.map(Arc::from),
        };
        app.trace.start(&call("sess", "r1", "Read", "2026-09-27T10:00:00.000Z"), &app.project_root.clone());
        app.trace.finish(&finish("r1", "2026-09-27T10:00:02.000Z", None, false));
        app.trace.start(&call("sess", "d1", "Agent", "2026-09-27T10:00:03.000Z"), &app.project_root.clone());
        app.trace.finish(&finish("d1", "2026-09-27T10:00:03.100Z", Some("ax1"), false));
        app.trace.start(&call("ax1", "x1", "Edit", "2026-09-27T10:00:04.000Z"), &app.project_root.clone());
        app.trace.finish(&finish("x1", "2026-09-27T10:00:09.000Z", None, true));
        app.process_prompt(&ambits::ingest::Prompt {
            agent_id: Arc::from("sess"),
            timestamp: "2026-09-27T09:59:59.000Z".into(),
            text: "review the code\nand tell me".into(),
        });
        app.trace_view.open = true;
        app.trace_view.open_trace(3);
        app
    }

    fn screen(app: &App) -> Vec<String> {
        crate::ui::test_render::lines(100, 16, |f| render(f, app, f.area(), &TraceFrame::new(app)))
    }

    #[test]
    fn the_waterfall_nests_draws_bars_and_details() {
        let mut app = app();
        app.trace_view.selected = Some(Item::Span(2));
        let lines = screen(&app);
        let text = lines.join("\n");
        assert!(lines[0].contains("· all agents · waterfall"), "{text}");
        assert!(lines[0].contains("Trace — review the code…"), "{text}");
        assert!(lines[1].contains("10s · 3 calls · 1 failed"), "{text}");
        assert!(lines.iter().any(|l| l.contains("▾ review the code…")), "the prompt is the root: {text}");
        let read = lines.iter().position(|l| l.contains("Read src/a.rs")).expect(&text);
        let edit = lines.iter().position(|l| l.contains("✗ Edit src/a.rs")).expect(&text);
        assert!(lines.iter().any(|l| l.contains("▾ Agent src/a.rs → ax1")), "{text}");
        assert!(edit > read);
        assert!(lines[read].contains(" 2s") && lines[read].contains('█'), "{}", lines[read]);
        assert!(text.contains("Edit src/a.rs · ax1 · 5s · ✗ failed"), "one line for the selection: {text}");
        assert!(app.trace_geometry.get().is_some());
    }

    #[test]
    fn the_tracks_give_each_agent_a_row_with_named_bars() {
        let mut app = app();
        app.trace_view.layout = Layout::Tracks;
        let lines = screen(&app);
        let text = lines.join("\n");
        assert!(lines.iter().any(|l| l.contains("▾ main")), "{text}");
        let sub = lines.iter().find(|l| l.contains("▾ Agent src/a.rs")).expect(&text);
        assert!(sub.contains("Edit"), "the name is drawn inside a wide enough bar: {sub}");
    }

    /// The top level lists prompts, not a time axis.
    #[test]
    fn the_list_shows_one_row_per_prompt() {
        let mut app = app();
        app.trace_view.close_trace();
        let lines = screen(&app);
        let text = lines.join("\n");
        assert!(lines[0].contains("Traces — all agents"), "{text}");
        assert!(lines[1].contains("1 prompts · 1 traces"), "{text}");
        let row = lines.iter().find(|l| l.contains("09-27 09:59")).expect(&text);
        assert!(row.contains("review the code…") && row.contains("10s") && row.contains("1 ✗"), "{row}");
        assert!(text.contains("09-27 09:59 · 10s · 3 calls · 1 failed · Tab for its summary"), "one line; the panel has the rest: {text}");
        assert!(!text.contains('█'), "no bars at the top level: {text}");
    }

    /// A commit made during a prompt is counted in the list and sits among
    /// its calls, with its subject.
    #[test]
    fn a_commit_shows_in_its_trace() {
        let mut app = app();
        app.trace.set_commits(&[ambits::git::Commit { sha: "1f32c0b".repeat(6)[..40].to_string(), t: 1_790_503_203_500, subject: "Fix the parser".into() }]);
        app.trace_view.close_trace();
        let list = screen(&app).join("\n");
        assert!(list.contains("commits"), "{list}");
        app.trace_view.open_trace(3);
        let lines = screen(&app);
        let text = lines.join("\n");
        let commit = lines.iter().position(|l| l.contains("│ commit 1f32c0b Fix the parser")).expect(&text);
        let delegation = lines.iter().position(|l| l.contains("Agent src/a.rs")).expect(&text);
        assert!(commit > delegation, "made after the delegation began: {text}");
    }

    #[test]
    fn an_empty_trace_says_so() {
        let mut app = App::new(ProjectTree { root: PathBuf::from("/test"), files: vec![] }, PathBuf::from("/test"));
        app.trace_view.open = true;
        assert!(screen(&app).join("\n").contains("No tool calls yet."));
    }
}
