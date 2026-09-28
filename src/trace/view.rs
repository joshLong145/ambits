//! The trace view's layout and state, free of any terminal so it is tested
//! as plain data: what time range is on screen, where the ticks and bars
//! go, which rows the waterfall shows, and how each agent's spans stack
//! into lanes on its track.
//!
//! The view opens on a list of traces, one per prompt ([`traces`]);
//! `Enter` on one shows its timeline. Two layouts share one [`TraceView`]:
//! - the **waterfall**, as Jaeger or Tempo show an OpenTelemetry trace: one
//!   row per span, nested by [`Trace::tree`];
//! - the **tracks**, as Perfetto shows a system trace: one track per agent,
//!   overlapping spans stacked in lanes.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::{Node, SpanKind, Trace};

/// The narrowest window zoom goes to, in milliseconds.
const MIN_WIDTH_MS: u64 = 20;

/// The time window on screen, in milliseconds since the epoch; `end > start`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Viewport {
    pub start: u64,
    pub end: u64,
}

impl Viewport {
    /// The whole of `range`, with a sliver of room after its last moment so
    /// the last span is not cut at the edge.
    pub fn fit((start, end): (u64, u64)) -> Self {
        let end = end.max(start + 1000);
        Self { start, end: end + (end - start) / 40 }
    }

    pub fn width(&self) -> u64 {
        self.end - self.start
    }

    /// Scale the window by `factor` (below 1 zooms in) keeping `at` where it is.
    pub fn zoom(&mut self, at: u64, factor: f64) {
        let at = at.clamp(self.start, self.end);
        let width = ((self.width() as f64 * factor) as u64).max(MIN_WIDTH_MS);
        let before = ((at - self.start) as f64 / self.width() as f64 * width as f64) as u64;
        self.start = at.saturating_sub(before);
        self.end = self.start + width;
    }

    /// Move the window by `fraction` of its width (negative is earlier).
    pub fn pan(&mut self, fraction: f64) {
        let shift = (self.width() as f64 * fraction.abs()) as u64;
        let width = self.width();
        self.start = if fraction < 0.0 { self.start.saturating_sub(shift) } else { self.start + shift };
        self.end = self.start + width;
    }

    /// Where `t` falls across `cols` columns; outside `0..cols` when off screen.
    pub fn col(&self, t: u64, cols: usize) -> f64 {
        (t as f64 - self.start as f64) / self.width() as f64 * cols as f64
    }

    /// The moment at the left edge of column `col`.
    pub fn time_at(&self, col: usize, cols: usize) -> u64 {
        self.start + (self.width() as f64 * col as f64 / cols.max(1) as f64) as u64
    }
}

/// Tick steps, in milliseconds: 1-2-5 below a second, then clock-friendly.
const STEPS: [u64; 27] = [
    1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 15_000, 30_000, 60_000, 120_000, 300_000,
    600_000, 900_000, 1_800_000, 3_600_000, 7_200_000, 10_800_000, 21_600_000, 43_200_000, 86_400_000,
];

/// Ticks at least `min_gap` columns apart: `(column, label)`, labelled by
/// their offset from `origin` (the session's first moment).
pub fn ticks(vp: &Viewport, cols: usize, origin: u64, min_gap: usize) -> Vec<(usize, String)> {
    if cols == 0 {
        return Vec::new();
    }
    let per_col = vp.width() as f64 / cols as f64;
    let step = STEPS.iter().copied().find(|s| *s as f64 / per_col >= min_gap as f64).unwrap_or(STEPS[STEPS.len() - 1]);
    // Ticks fall on multiples of the step from the origin, so labels are round.
    let from = vp.start.max(origin);
    let mut t = origin + (from - origin).div_ceil(step) * step;
    let mut out = Vec::new();
    while t < vp.end {
        let col = vp.col(t, cols) as usize;
        if col < cols {
            out.push((col, offset(t - origin)));
        }
        t += step;
    }
    out
}

/// An offset from the session's start: [`duration`], but the start is `0`.
pub fn offset(ms: u64) -> String {
    if ms == 0 { "0".into() } else { duration(ms) }
}

/// A duration, compactly: `250ms`, `1.5s`, `42s`, `3m05s`, `1h02m`.
pub fn duration(ms: u64) -> String {
    match ms {
        0..1_000 => format!("{ms}ms"),
        1_000..10_000 if ms % 1_000 != 0 => format!("{:.1}s", ms as f64 / 1_000.0),
        1_000..60_000 => format!("{}s", ms / 1_000),
        60_000..3_600_000 => match (ms / 60_000, ms / 1_000 % 60) {
            (m, 0) => format!("{m}m"),
            (m, s) => format!("{m}m{s:02}s"),
        },
        _ => match (ms / 3_600_000, ms / 60_000 % 60) {
            (h, 0) => format!("{h}h"),
            (h, m) => format!("{h}h{m:02}m"),
        },
    }
}

/// Left-aligned eighths, by how much of the cell is filled.
const EIGHTHS: [char; 8] = ['▏', '▎', '▍', '▌', '▋', '▊', '▉', '█'];

/// The cells of a bar from column `from` to `to` (fractional), clipped to
/// `0..cols`: full blocks inside, an eighth-block at a partial end, a right
/// half-block when it starts late in its first cell. Never invisible: a
/// bar thinner than an eighth still draws `▏`.
pub fn bar(from: f64, to: f64, cols: usize) -> Vec<(usize, char)> {
    let (from, to) = (from.max(0.0), to.min(cols as f64));
    if from >= cols as f64 || to < 0.0 {
        return Vec::new();
    }
    if to - from < 0.125 {
        return vec![((from as usize).min(cols - 1), '▏')];
    }
    let (first, last) = (from.floor() as usize, (to.ceil() as usize).saturating_sub(1).min(cols - 1));
    (first..=last)
        .map(|c| {
            let (lo, hi) = (from.max(c as f64), to.min(c as f64 + 1.0));
            let glyph = if hi < c as f64 + 1.0 {
                EIGHTHS[(((hi - lo) * 8.0).round() as usize).clamp(1, 8) - 1]
            } else if lo - c as f64 >= 0.5 {
                '▐'
            } else {
                '█'
            };
            (c, glyph)
        })
        .collect()
}

/// One trace in the list: a root of the span tree — a prompt and its
/// turn, or a call made before any prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceSummary {
    /// The root span.
    pub root: usize,
    pub start: u64,
    pub end: u64,
    /// Calls under it, at any depth.
    pub calls: usize,
    pub failed: usize,
    /// Subagents it started, at any depth.
    pub agents: usize,
    /// Commits made while it ran.
    pub commits: usize,
}

/// The traces, in time order; under `filter`, those in which that agent
/// made a call.
pub fn traces(trace: &Trace, filter: Option<&str>) -> Vec<TraceSummary> {
    let spans = trace.spans();
    trace
        .tree()
        .iter()
        .filter_map(|node| {
            let all = node.spans();
            if let Some(f) = filter {
                if !all.iter().any(|&i| spans[i].agent.starts_with(f)) {
                    return None;
                }
            }
            let under = &all[1..];
            Some(TraceSummary {
                root: node.span,
                start: spans[node.span].start,
                end: node.end,
                calls: under.len(),
                failed: all.iter().filter(|&&i| spans[i].error).count(),
                agents: under.iter().filter(|&&i| spans[i].child_agent.is_some()).count(),
                commits: trace
                    .instants()
                    .iter()
                    .filter(|x| matches!(x.kind, super::InstantKind::Commit { .. }) && (spans[node.span].start..=node.end).contains(&x.t))
                    .count(),
            })
        })
        .collect()
}

/// The node for span `root`, wherever it sits in `tree`.
pub fn subtree(tree: &[Node], root: usize) -> Option<&Node> {
    tree.iter().find_map(|n| if n.span == root { Some(n) } else { subtree(&n.children, root) })
}

/// A span or an instant: what a row, or a selection, stands for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Item {
    Span(usize),
    Instant(usize),
}

/// One waterfall row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Row {
    pub item: Item,
    pub depth: usize,
    /// Spans nested under it, at any depth.
    pub descendants: usize,
    pub collapsed: bool,
    /// Its effective end: for a delegation, its subagent's last moment.
    pub end: u64,
}

/// The waterfall's rows: the span tree — or, `within` a trace, that
/// trace's subtree — under `filter`, depth first, collapsed subtrees shut,
/// instants among the roots in time order. A non-empty `query` lists
/// instead every span whose name contains it (case-insensitive), still
/// indented by depth.
pub fn waterfall(trace: &Trace, filter: Option<&str>, within: Option<usize>, collapsed: &HashSet<usize>, query: &str) -> Vec<Row> {
    let full = trace.tree();
    let tree: Vec<Node> = match within {
        Some(root) => subtree(&full, root).cloned().into_iter().collect(),
        None => full,
    };
    let roots = trace.roots(&tree, filter);
    let query = query.to_lowercase();
    let window = tree.first().filter(|_| within.is_some()).map(|n| (trace.spans()[n.span].start, n.end));

    let mut out = Vec::new();
    let mut instants: Vec<usize> = (0..trace.instants().len())
        .filter(|&i| filter.is_none_or(|a| trace.instants()[i].agent.as_deref().is_none_or(|ia| ia.starts_with(a))))
        .filter(|&i| window.is_none_or(|(s, e)| (s..=e).contains(&trace.instants()[i].t)))
        .collect();
    instants.sort_by_key(|&i| trace.instants()[i].t);
    let mut instants = instants.into_iter().peekable();

    fn walk(trace: &Trace, node: &Node, depth: usize, collapsed: &HashSet<usize>, query: &str, out: &mut Vec<Row>) {
        let matches = query.is_empty() || trace.spans()[node.span].name().to_lowercase().contains(query);
        let shut = query.is_empty() && collapsed.contains(&node.span);
        if matches {
            out.push(Row { item: Item::Span(node.span), depth, descendants: count(node), collapsed: shut, end: node.end });
        }
        if !shut {
            for child in &node.children {
                walk(trace, child, depth + 1, collapsed, query, out);
            }
        }
    }
    fn count(node: &Node) -> usize {
        node.children.iter().map(|c| 1 + count(c)).sum()
    }

    if within.is_some() {
        // One trace: its moments among the calls under its root, in time
        // order, where they happened.
        for root in &roots {
            walk(trace, root, 0, collapsed, &query, &mut out);
        }
        for i in instants {
            let t = trace.instants()[i].t;
            let at = out
                .iter()
                .enumerate()
                .skip(1)
                .find(|(_, r)| r.depth <= 1 && matches!(r.item, Item::Span(s) if trace.spans()[s].start > t))
                .map_or(out.len(), |(ix, _)| ix);
            out.insert(at, Row { depth: usize::from(!out.is_empty()), ..instant_row(trace, i) });
        }
    } else {
        for root in roots {
            let start = trace.spans()[root.span].start;
            while let Some(i) = instants.next_if(|&i| trace.instants()[i].t <= start) {
                out.push(instant_row(trace, i));
            }
            walk(trace, root, 0, collapsed, &query, &mut out);
        }
        out.extend(instants.map(|i| instant_row(trace, i)));
    }
    if !query.is_empty() {
        out.retain(|r| matches!(r.item, Item::Span(_)));
    }
    out
}

fn instant_row(trace: &Trace, i: usize) -> Row {
    Row { item: Item::Instant(i), depth: 0, descendants: 0, collapsed: false, end: trace.instants()[i].t }
}

/// One agent's track.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Track {
    pub agent: Arc<str>,
    /// How deep in the delegation tree the agent is.
    pub depth: usize,
    /// The delegation that started it, if any.
    pub delegation: Option<usize>,
    /// Its spans, stacked so none overlaps another in its lane.
    pub lanes: Vec<Vec<usize>>,
    /// Instants on this track: its own, and project-wide ones on the first.
    pub instants: Vec<usize>,
}

/// Every agent's track under `filter`, in delegation-tree order (an agent
/// after the one that started it, siblings by start). A span still running
/// is laid out as if it ended at `open_end`; a delegation lasts until its
/// subagent's last moment.
pub fn tracks(trace: &Trace, filter: Option<&str>, within: Option<usize>, open_end: u64) -> Vec<Track> {
    let spans = trace.spans();
    let tree = trace.tree();
    let ends = effective_ends(&tree);
    let end_of = |i: usize| ends.get(&i).copied().or(spans[i].end).unwrap_or(open_end).max(spans[i].start + 1);
    // Within one trace, only its spans. A prompt is the trace itself, not a
    // call on a track.
    let mut shown: Vec<usize> = match within.and_then(|r| subtree(&tree, r)) {
        Some(node) => node.spans(),
        None if within.is_some() => Vec::new(),
        None => (0..spans.len()).collect(),
    };
    shown.retain(|&i| spans[i].kind != SpanKind::Prompt);
    shown.sort_by_key(|&i| (spans[i].start, i));

    // Each agent's spans, in start order, and the delegation that started it.
    let mut by_agent: HashMap<&str, Vec<usize>> = HashMap::new();
    for &i in &shown {
        by_agent.entry(&spans[i].agent).or_default().push(i);
    }
    let started_by: HashMap<&str, usize> =
        spans.iter().enumerate().filter_map(|(i, s)| Some((s.child_agent.as_deref()?, i))).collect();
    let parent_of = |agent: &str| started_by.get(agent).map(|&d| &*spans[d].agent).filter(|p| *p != agent);

    let first = |agent: &str| by_agent.get(agent).and_then(|v| v.first()).map_or(u64::MAX, |&i| spans[i].start);
    let mut roots: Vec<&str> = by_agent.keys().copied().filter(|a| parent_of(a).is_none_or(|p| !by_agent.contains_key(p))).collect();
    roots.sort_by_key(|a| (first(a), *a));
    let mut children: HashMap<&str, Vec<&str>> = HashMap::new();
    for agent in by_agent.keys().copied() {
        if let Some(parent) = parent_of(agent).filter(|p| by_agent.contains_key(p)) {
            children.entry(parent).or_default().push(agent);
        }
    }
    for kids in children.values_mut() {
        kids.sort_by_key(|a| (started_by.get(a).map(|&d| spans[d].start), *a));
    }

    let mut order: Vec<(&str, usize)> = Vec::new();
    let mut stack: Vec<(&str, usize)> = roots.iter().rev().map(|a| (*a, 0)).collect();
    let mut seen = HashSet::new();
    while let Some((agent, depth)) = stack.pop() {
        if !seen.insert(agent) {
            continue;
        }
        order.push((agent, depth));
        for kid in children.get(agent).into_iter().flatten().rev() {
            stack.push((kid, depth + 1));
        }
    }
    // Under a filter, the filtered agent's subtree, re-rooted.
    if let Some(f) = filter {
        if let Some(pos) = order.iter().position(|(a, _)| a.starts_with(f)) {
            let base = order[pos].1;
            let len = order[pos + 1..].iter().take_while(|(_, d)| *d > base).count();
            order = order[pos..=pos + len].iter().map(|&(a, d)| (a, d - base)).collect();
        } else {
            order.clear();
        }
    }

    order
        .iter()
        .enumerate()
        .map(|(n, &(agent, depth))| {
            let mut lanes: Vec<Vec<usize>> = Vec::new();
            let mut lane_ends: Vec<u64> = Vec::new();
            for &i in &by_agent[agent] {
                match lane_ends.iter().position(|&e| e <= spans[i].start) {
                    Some(l) => {
                        lanes[l].push(i);
                        lane_ends[l] = end_of(i);
                    }
                    None => {
                        lanes.push(vec![i]);
                        lane_ends.push(end_of(i));
                    }
                }
            }
            let instants = (0..trace.instants().len())
                .filter(|&i| match trace.instants()[i].agent.as_deref() {
                    Some(a) => a == agent,
                    None => n == 0,
                })
                .collect();
            Track { agent: spans[by_agent[agent][0]].agent.clone(), depth, delegation: started_by.get(agent).copied(), lanes, instants }
        })
        .collect()
}

/// Span index → effective end, for every node of the tree.
pub fn effective_ends(tree: &[Node]) -> HashMap<usize, u64> {
    let mut out = HashMap::new();
    let mut stack: Vec<&Node> = tree.iter().collect();
    while let Some(node) = stack.pop() {
        out.insert(node.span, node.end);
        stack.extend(node.children.iter());
    }
    out
}

/// A screen row of the tracks layout: one lane of a track, or a collapsed
/// track's single summary row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrackRow {
    pub track: usize,
    /// `None` for a collapsed track.
    pub lane: Option<usize>,
}

pub fn track_rows(tracks: &[Track], collapsed: &HashSet<Arc<str>>) -> Vec<TrackRow> {
    tracks
        .iter()
        .enumerate()
        .flat_map(|(t, track)| {
            let lanes: Vec<Option<usize>> =
                if collapsed.contains(&track.agent) { vec![None] } else { (0..track.lanes.len().max(1)).map(Some).collect() };
            lanes.into_iter().map(move |lane| TrackRow { track: t, lane })
        })
        .collect()
}

impl TrackRow {
    /// The spans this row shows, in start order.
    pub fn spans(&self, tracks: &[Track]) -> Vec<usize> {
        let track = &tracks[self.track];
        match self.lane {
            Some(l) => track.lanes.get(l).cloned().unwrap_or_default(),
            None => {
                let mut all: Vec<usize> = track.lanes.iter().flatten().copied().collect();
                all.sort_unstable();
                all
            }
        }
    }
}

/// How many of `spans` cover each of `cols` columns: a collapsed track's
/// density row.
pub fn density(trace: &Trace, spans: &[usize], vp: &Viewport, cols: usize, open_end: u64) -> Vec<u32> {
    let mut out = vec![0; cols];
    for &i in spans {
        let s = &trace.spans()[i];
        for (c, _) in bar(vp.col(s.start, cols), vp.col(s.end.unwrap_or(open_end), cols), cols) {
            out[c] += 1;
        }
    }
    out
}

/// Which waterfall layout, or the tracks.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Layout {
    #[default]
    Waterfall,
    Tracks,
}

/// Everything the trace view remembers between frames. Selection is by
/// span, not row, so it holds still while live spans arrive.
#[derive(Debug, Default)]
pub struct TraceView {
    pub open: bool,
    /// The trace whose timeline is shown (its root span); `None` shows the
    /// list of traces.
    pub focus: Option<usize>,
    /// The trace selected in the list (its root span); `None` is the latest.
    pub list: Option<usize>,
    pub layout: Layout,
    /// `None` fits the whole session, growing as it does.
    pub viewport: Option<Viewport>,
    pub selected: Option<Item>,
    /// The selected lane row in the tracks layout.
    pub track_row: usize,
    pub collapsed_spans: HashSet<usize>,
    pub collapsed_agents: HashSet<Arc<str>>,
    /// The waterfall's name filter, and whether it is being typed.
    pub query: String,
    pub typing: bool,
}

impl TraceView {
    pub fn reset(&mut self) {
        *self = Self { open: self.open, layout: self.layout, ..Self::default() };
    }

    /// The window on screen: the focused trace's, until zoomed.
    pub fn viewport(&self, trace: &Trace) -> Viewport {
        self.viewport.unwrap_or_else(|| Viewport::fit(self.range(trace).unwrap_or((0, 1000))))
    }

    /// The focused trace's first and last moment, else the session's.
    pub fn range(&self, trace: &Trace) -> Option<(u64, u64)> {
        match self.focus {
            Some(root) => subtree(&trace.tree(), root).map(|n| (trace.spans()[root].start, n.end)),
            None => trace.range(),
        }
    }

    /// Show `root`'s timeline, from the whole of it, nothing selected.
    pub fn open_trace(&mut self, root: usize) {
        self.focus = Some(root);
        self.list = Some(root);
        self.viewport = None;
        self.selected = None;
        self.track_row = 0;
        self.query.clear();
    }

    /// Back to the list, on the trace that was open.
    pub fn close_trace(&mut self) {
        self.focus = None;
        self.viewport = None;
        self.selected = None;
    }

    /// Move the list selection by `delta`.
    pub fn move_list(&mut self, traces: &[TraceSummary], delta: isize) {
        let Some(last) = traces.len().checked_sub(1) else { return };
        let at = self.list.and_then(|r| traces.iter().position(|t| t.root == r)).unwrap_or(last);
        self.list = Some(traces[at.saturating_add_signed(delta).min(last)].root);
    }

    /// The list row selected: the chosen trace, else the latest.
    pub fn list_index(&self, traces: &[TraceSummary]) -> Option<usize> {
        let last = traces.len().checked_sub(1)?;
        Some(self.list.and_then(|r| traces.iter().position(|t| t.root == r)).unwrap_or(last))
    }

    /// Where zoom centres: the selection when it is on screen, else the middle.
    fn focus_time(&self, trace: &Trace, vp: &Viewport) -> u64 {
        let t = match self.selected {
            Some(Item::Span(i)) => trace.spans().get(i).map(|s| s.start),
            Some(Item::Instant(i)) => trace.instants().get(i).map(|x| x.t),
            None => None,
        };
        t.filter(|t| (vp.start..vp.end).contains(t)).unwrap_or(vp.start + vp.width() / 2)
    }

    pub fn zoom(&mut self, trace: &Trace, factor: f64, at: Option<u64>) {
        let mut vp = self.viewport(trace);
        let at = at.unwrap_or_else(|| self.focus_time(trace, &vp));
        vp.zoom(at, factor);
        self.viewport = Some(vp);
    }

    pub fn pan(&mut self, trace: &Trace, fraction: f64) {
        let mut vp = self.viewport(trace);
        vp.pan(fraction);
        self.viewport = Some(vp);
    }

    pub fn fit(&mut self) {
        self.viewport = None;
    }

    /// Move the waterfall selection by `delta` rows.
    pub fn move_row(&mut self, rows: &[Row], delta: isize) {
        let Some(last) = rows.len().checked_sub(1) else { return };
        let at = self.selected.and_then(|s| rows.iter().position(|r| r.item == s));
        // From no selection, a step down lands on the first row, a jump on
        // the row it reaches; a step up on the last.
        let next = match at {
            Some(i) => i.saturating_add_signed(delta).min(last),
            None if delta < 0 => last,
            None => (delta.unsigned_abs() - 1).min(last),
        };
        self.selected = Some(rows[next].item);
    }

    /// The waterfall row index of the selection.
    pub fn row_index(&self, rows: &[Row]) -> Option<usize> {
        self.selected.and_then(|s| rows.iter().position(|r| r.item == s))
    }

    /// Collapse or expand the selected span's subtree.
    pub fn toggle_span(&mut self, open: Option<bool>) {
        if let Some(Item::Span(i)) = self.selected {
            let shut = self.collapsed_spans.contains(&i);
            match open {
                Some(true) | None if shut => {
                    self.collapsed_spans.remove(&i);
                }
                Some(false) | None if !shut => {
                    self.collapsed_spans.insert(i);
                }
                _ => {}
            }
        }
    }

    /// The next span in row order after the selection that failed.
    pub fn next_error(&mut self, trace: &Trace, rows: &[Row]) {
        let from = self.row_index(rows).map_or(0, |i| i + 1);
        let failed = |r: &Row| matches!(r.item, Item::Span(i) if trace.spans()[i].error);
        if let Some(r) = rows[from.min(rows.len())..].iter().chain(&rows[..from.min(rows.len())]).find(|r| failed(r)) {
            self.selected = Some(r.item);
        }
    }

    /// Move to another lane row, selecting the span on it nearest in time
    /// to the current selection.
    pub fn move_track_row(&mut self, trace: &Trace, tracks: &[Track], rows: &[TrackRow], delta: isize) {
        let Some(last) = rows.len().checked_sub(1) else { return };
        self.track_row = self.track_row.min(last).saturating_add_signed(delta).min(last);
        let at = match self.selected {
            Some(Item::Span(i)) => trace.spans().get(i).map(|s| s.start),
            _ => None,
        };
        let on_row = rows[self.track_row].spans(tracks);
        let nearest = on_row.iter().copied().min_by_key(|&i| at.map_or(0, |t| trace.spans()[i].start.abs_diff(t)));
        if let Some(i) = nearest {
            self.selected = Some(Item::Span(i));
        }
    }

    /// The previous (`-1`) or next (`1`) span on the selected lane row.
    pub fn step_span(&mut self, tracks: &[Track], rows: &[TrackRow], dir: isize) {
        let Some(row) = rows.get(self.track_row) else { return };
        let on_row = row.spans(tracks);
        let at = match self.selected {
            Some(Item::Span(i)) => on_row.iter().position(|&s| s == i),
            _ => None,
        };
        let next = match (at, dir < 0) {
            (Some(p), true) => p.checked_sub(1),
            (Some(p), false) => Some(p + 1).filter(|&n| n < on_row.len()),
            (None, _) => (!on_row.is_empty()).then_some(0),
        };
        if let Some(n) = next {
            self.selected = Some(Item::Span(on_row[n]));
        }
    }

    /// Put the tracks selection on `agent`'s first lane and first span.
    pub fn select_track(&mut self, tracks: &[Track], rows: &[TrackRow], agent: &str) -> bool {
        let Some(t) = tracks.iter().position(|t| &*t.agent == agent) else { return false };
        let Some(r) = rows.iter().position(|r| r.track == t) else { return false };
        self.track_row = r;
        self.selected = rows[r].spans(tracks).first().map(|&i| Item::Span(i));
        true
    }
}

/// The span under column `col` on a row showing `spans`, if any: the one
/// whose bar covers it, else the nearest within a cell.
pub fn span_at(trace: &Trace, spans: &[usize], vp: &Viewport, cols: usize, col: usize, open_end: u64) -> Option<usize> {
    let c = col as f64 + 0.5;
    spans
        .iter()
        .copied()
        .map(|i| {
            let s = &trace.spans()[i];
            let (from, to) = (vp.col(s.start, cols), vp.col(s.end.unwrap_or(open_end), cols).max(vp.col(s.start, cols) + 0.125));
            let gap = if c < from { from - c } else if c > to { c - to } else { 0.0 };
            (i, gap)
        })
        .filter(|(_, gap)| *gap <= 1.0)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::ToolFinished;
    use crate::trace::InstantKind;
    use std::path::Path;

    const BASE: u64 = 1_790_000_000_000;

    fn at(ms: u64) -> String {
        let t = BASE + ms;
        format!("{}.{:03}Z", crate::time::rfc3339(t / 1000).trim_end_matches('Z'), t % 1000)
    }

    /// `(agent, id, tool, start, end, child)`, times in milliseconds from BASE.
    type Fixture<'a> = (&'a str, &'a str, &'a str, u64, Option<u64>, Option<&'a str>);

    fn trace(spans: &[Fixture<'_>]) -> Trace {
        let mut t = Trace::default();
        for &(agent, id, tool, start, end, child) in spans {
            let mut c = crate::helpers::tool_call(tool, "/p/src/a.rs", crate::tracking::ReadDepth::FullBody);
            c.agent_id = Arc::from(agent);
            c.tool_use_id = Some(Arc::from(id));
            c.timestamp_str = at(start);
            t.start(&c, Path::new("/p"));
            if let Some(end) = end {
                t.finish(&ToolFinished {
                    id: Arc::from(id),
                    agent_id: Arc::from(agent),
                    timestamp: at(end),
                    error: tool == "Edit",
                    child_agent: child.map(Arc::from),
                });
            }
        }
        t
    }

    /// main reads, delegates to `x` (which reads twice, overlapping), reads again.
    fn sample() -> Trace {
        trace(&[
            ("main", "m1", "Read", 0, Some(1_000), None),
            ("main", "d1", "Agent", 2_000, Some(2_100), Some("x")),
            ("x", "x1", "Read", 3_000, Some(6_000), None),
            ("x", "x2", "Edit", 4_000, Some(5_000), None),
            ("main", "m2", "Read", 3_500, Some(4_000), None),
        ])
    }

    fn items(rows: &[Row]) -> Vec<(Item, usize)> {
        rows.iter().map(|r| (r.item, r.depth)).collect()
    }

    #[test]
    fn durations_read_compactly() {
        assert_eq!(duration(250), "250ms");
        assert_eq!(duration(1_500), "1.5s");
        assert_eq!(duration(42_000), "42s");
        assert_eq!(duration(185_000), "3m05s");
        assert_eq!(duration(3_720_000), "1h02m");
        assert_eq!(duration(7_200_000), "2h");
    }

    #[test]
    fn zoom_keeps_its_anchor_and_pan_keeps_the_width() {
        let mut vp = Viewport { start: 0, end: 1_000 };
        vp.zoom(250, 0.5);
        assert_eq!(vp, Viewport { start: 125, end: 625 });
        assert_eq!(vp.col(250, 100).round(), 25.0, "the anchor stays in its column");
        vp.pan(-1.0);
        assert_eq!(vp, Viewport { start: 0, end: 500 }, "never before the epoch");
        vp.pan(0.5);
        assert_eq!(vp, Viewport { start: 250, end: 750 });
        vp.zoom(300, 0.0);
        assert_eq!(vp.width(), MIN_WIDTH_MS);
    }

    #[test]
    fn fit_shows_the_first_and_last_moment() {
        let vp = Viewport::fit((10_000, 70_000));
        assert!(vp.start <= 10_000 && vp.end > 70_000);
        assert!(vp.col(70_000, 100) < 100.0);
    }

    #[test]
    fn ticks_adapt_from_milliseconds_to_hours() {
        let label = |width: u64| ticks(&Viewport { start: BASE, end: BASE + width }, 100, BASE, 10)[1].1.clone();
        assert_eq!(label(500), "50ms");
        assert_eq!(label(60_000), "10s");
        assert_eq!(label(3_600_000), "10m");
        assert_eq!(label(36_000_000), "1h");
        let t = ticks(&Viewport { start: BASE, end: BASE + 60_000 }, 100, BASE, 10);
        assert_eq!(t[0], (0, "0".into()));
        assert!(t.windows(2).all(|w| w[1].0 - w[0].0 >= 10));
    }

    #[test]
    fn bars_have_sub_cell_edges_and_never_vanish() {
        assert_eq!(bar(1.0, 3.0, 10), vec![(1, '█'), (2, '█')]);
        assert_eq!(bar(1.5, 3.25, 10), vec![(1, '▐'), (2, '█'), (3, '▎')]);
        assert_eq!(bar(4.0, 4.01, 10), vec![(4, '▏')]);
        assert_eq!(bar(-5.0, 2.0, 10), vec![(0, '█'), (1, '█')], "clipped on the left");
        assert_eq!(bar(8.0, 50.0, 10), vec![(8, '█'), (9, '█')], "and on the right");
        assert!(bar(11.0, 12.0, 10).is_empty());
    }

    #[test]
    fn the_waterfall_nests_a_delegations_calls_and_collapses() {
        let t = sample();
        let rows = waterfall(&t, None, None, &HashSet::new(), "");
        assert_eq!(
            items(&rows),
            vec![(Item::Span(0), 0), (Item::Span(1), 0), (Item::Span(2), 1), (Item::Span(3), 1), (Item::Span(4), 0)]
        );
        assert_eq!(rows[1].descendants, 2);
        assert_eq!(rows[1].end, BASE + 6_000, "a delegation lasts until its agent's last call");

        let rows = waterfall(&t, None, None, &HashSet::from([1]), "");
        assert_eq!(items(&rows), vec![(Item::Span(0), 0), (Item::Span(1), 0), (Item::Span(4), 0)]);
        assert!(rows[1].collapsed);
    }

    #[test]
    fn the_waterfall_filter_reroots_and_the_query_lists_matches() {
        let t = sample();
        assert_eq!(items(&waterfall(&t, Some("x"), None, &HashSet::new(), "")), vec![(Item::Span(2), 0), (Item::Span(3), 0)]);
        let rows = waterfall(&t, None, None, &HashSet::from([1]), "edit");
        assert_eq!(items(&rows), vec![(Item::Span(3), 1)], "a match inside a collapsed subtree still shows");
    }

    #[test]
    fn an_unmatched_subagent_hangs_off_the_root() {
        let t = trace(&[("main", "m1", "Read", 0, Some(10), None), ("orphan", "o1", "Read", 5, Some(9), None)]);
        assert_eq!(items(&waterfall(&t, None, None, &HashSet::new(), "")), vec![(Item::Span(0), 0), (Item::Span(1), 0)]);
    }

    #[test]
    fn instants_sit_among_the_roots_in_time_order() {
        let mut t = sample();
        t.instant(BASE + 2_500, Some(Arc::from("main")), InstantKind::Compaction);
        let rows = waterfall(&t, None, None, &HashSet::new(), "");
        let pos = rows.iter().position(|r| r.item == Item::Instant(0)).unwrap();
        assert_eq!(rows[pos + 1].item, Item::Span(4), "before the read at 3.5s");
    }

    /// Inside one trace, a commit sits among its calls where it happened,
    /// and the list counts it.
    #[test]
    fn a_commit_lands_among_the_calls_of_its_trace() {
        let mut t = sample();
        t.prompt(&crate::ingest::Prompt { agent_id: Arc::from("main"), timestamp: at(0), text: "do it".into() });
        t.set_commits(&[crate::git::Commit { sha: "a".repeat(40), t: BASE + 3_200, subject: "Fix".into() }]);
        let rows = waterfall(&t, None, Some(5), &HashSet::new(), "");
        assert_eq!(
            items(&rows),
            vec![
                (Item::Span(5), 0),
                (Item::Span(0), 1),
                (Item::Span(1), 1),
                (Item::Span(2), 2),
                (Item::Span(3), 2),
                (Item::Instant(0), 1),
                (Item::Span(4), 1),
            ],
            "after the delegation began at 2s, before the read at 3.5s"
        );
        assert_eq!(traces(&t, None)[0].commits, 1);
        assert_eq!(t.instants()[0].kind.label(), format!("commit {} Fix", "a".repeat(7)));

        t.set_commits(&[]);
        assert!(t.instants().is_empty(), "a rescan replaces what the last one found");
    }

    #[test]
    fn tracks_follow_the_delegation_tree_and_stack_overlaps() {
        let t = sample();
        let tr = tracks(&t, None, None, BASE + 10_000);
        assert_eq!(tr.iter().map(|t| (&*t.agent, t.depth)).collect::<Vec<_>>(), vec![("main", 0), ("x", 1)]);
        // main: its read at 3.5s overlaps the delegation, which lasts to 6s.
        assert_eq!(tr[0].lanes, vec![vec![0, 1], vec![4]]);
        assert_eq!(tr[1].lanes, vec![vec![2], vec![3]], "two overlapping calls take two lanes");
        assert_eq!(tr[1].delegation, Some(1));

        let only = tracks(&t, Some("x"), None, BASE + 10_000);
        assert_eq!(only.iter().map(|t| (&*t.agent, t.depth)).collect::<Vec<_>>(), vec![("x", 0)]);
    }

    #[test]
    fn a_collapsed_track_is_one_density_row() {
        let t = sample();
        let tr = tracks(&t, None, None, BASE + 10_000);
        let rows = track_rows(&tr, &HashSet::from([Arc::from("x")]));
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[2], TrackRow { track: 1, lane: None });
        let vp = Viewport { start: BASE, end: BASE + 10_000 };
        let d = density(&t, &rows[2].spans(&tr), &vp, 10, BASE + 10_000);
        assert_eq!(d[3], 1);
        assert_eq!(d[4], 2, "both of x's calls cover 4s-5s");
    }

    #[test]
    fn instants_land_on_their_agents_track() {
        let mut t = sample();
        t.instant(BASE + 4_500, Some(Arc::from("x")), InstantKind::Compaction);
        t.instant(BASE + 4_600, None, InstantKind::Commit { sha: "abc".into(), subject: "s".into() });
        let tr = tracks(&t, None, None, BASE + 10_000);
        assert_eq!((tr[0].instants.clone(), tr[1].instants.clone()), (vec![1], vec![0]));
    }

    #[test]
    fn navigation_moves_by_row_span_and_error() {
        let t = sample();
        let rows = waterfall(&t, None, None, &HashSet::new(), "");
        let mut view = TraceView::default();
        view.move_row(&rows, isize::MAX / 2);
        assert_eq!(view.selected, Some(Item::Span(4)), "G from nothing selected is the last row");
        view.selected = None;
        view.move_row(&rows, 1);
        assert_eq!(view.selected, Some(Item::Span(0)));
        view.move_row(&rows, 2);
        assert_eq!(view.selected, Some(Item::Span(2)));
        view.next_error(&t, &rows);
        assert_eq!(view.selected, Some(Item::Span(3)), "the failed edit");

        let tr = tracks(&t, None, None, BASE + 10_000);
        let trows = track_rows(&tr, &HashSet::new());
        assert!(view.select_track(&tr, &trows, "x"));
        assert_eq!((view.track_row, view.selected), (2, Some(Item::Span(2))));
        view.move_track_row(&t, &tr, &trows, 1);
        assert_eq!(view.selected, Some(Item::Span(3)));
        view.move_track_row(&t, &tr, &trows, -3);
        assert_eq!((view.track_row, view.selected), (0, Some(Item::Span(1))), "nearest in time on main's first lane");
        view.step_span(&tr, &trows, -1);
        assert_eq!(view.selected, Some(Item::Span(0)));
    }

    #[test]
    fn a_click_finds_the_span_under_it() {
        let t = sample();
        let vp = Viewport { start: BASE, end: BASE + 10_000 };
        assert_eq!(span_at(&t, &[0, 1], &vp, 100, 5, BASE), Some(0));
        assert_eq!(span_at(&t, &[0, 1], &vp, 100, 50, BASE), None);
    }
}
