use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState};

use ambits::app::{App, FocusPanel};
use ambits::tracking::ReadDepth;
use ambits::writes::Status;

use super::colors;

pub fn render(f: &mut Frame, app: &App, area: Rect) {
    let border_style = if app.focus == FocusPanel::Tree {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let block = Block::default()
        .title(" Files ")
        .borders(Borders::ALL)
        .border_style(border_style);

    // File bars line up in a column after the longest path, within reason.
    let align = app.tree_rows.iter().filter(|r| r.is_file()).map(|r| super::width(&r.display_name)).max().unwrap_or(0).min(48);
    let items: Vec<ListItem> = app.tree_rows.iter().map(|row| ListItem::new(row_line(row, align))).collect();

    let mut state = ListState::default();
    state.select(Some(app.selected_index));

    let list = List::new(items)
        .block(block)
        .highlight_style(
            Style::default()
                .bg(colors::HIGHLIGHT_BG)
                .fg(colors::HIGHLIGHT_FG)
                .add_modifier(Modifier::BOLD),
        );

    f.render_stateful_widget(list, area, &mut state);
}

/// One tree row. A symbol reads left to right as a gutter of states —
/// read depth, freshness, write — then its name:
///
/// `  ●!✎ fn render   L12-80 ~310 tok`
///
/// A file shows a bar of its symbols (full, partly read, unseen) and counts:
///
/// `▾ src/app.rs  ███▓▓░░░░░  42/120  !4  ✎3`
pub(super) fn row_line(row: &ambits::app::TreeRow, align: usize) -> Line<'static> {
    let dim = Style::default().fg(Color::DarkGray);
    let indent = "  ".repeat(row.depth);
    let fold = match (row.is_file(), row.has_children, row.is_expanded) {
        (true, _, true) => "▼ ",
        (true, _, false) => "▶ ",
        (false, true, true) => "▾ ",
        (false, true, false) => "▸ ",
        _ => "  ",
    };
    let mut spans = vec![Span::raw(indent), Span::styled(fold, dim)];

    if row.is_file() {
        spans.push(Span::styled(row.display_name.clone(), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)));
        if row.coverage_total > 0 {
            spans.push(Span::raw(" ".repeat(2 + align.saturating_sub(super::width(&row.display_name)))));
            spans.extend(coverage_bar(row.coverage_full, row.coverage_seen, row.coverage_total, 10));
            spans.push(Span::styled(format!("  {}/{}", row.coverage_seen, row.coverage_total), Style::default().fg(Color::Gray)));
        }
        spans.extend(counts(row));
        spans.push(Span::styled(format!("  ({})", row.line_range), dim));
        return Line::from(spans);
    }

    let depth = if row.read_depth.is_seen() { row.read_depth } else { ReadDepth::Unseen };
    let freshness = if row.stale && depth.is_seen() {
        Span::styled("!", Style::default().fg(colors::DEPTH_STALE))
    } else if row.restored && depth.is_seen() {
        Span::styled("◌", Style::default().fg(colors::ACCENT_MUTED))
    } else {
        Span::raw(" ")
    };
    let write = match &row.write {
        Some(mark) => Span::styled("✎", Style::default().fg(write_color(mark.status))),
        None => Span::raw(" "),
    };
    spans.push(Span::styled(depth_glyph(depth).to_string(), Style::default().fg(depth_color(depth, false))));
    spans.push(freshness);
    spans.push(write);
    // `impl Status` already says what it is.
    let label = if row.display_name.starts_with(&format!("{} ", row.label)) { String::from(" ") } else { format!(" {} ", row.label) };
    spans.push(Span::styled(label, dim));
    let name_color = if depth.is_seen() { Color::White } else { Color::Gray };
    spans.push(Span::styled(row.display_name.clone(), Style::default().fg(name_color)));
    // A folded symbol summarizes what it hides, as a file does.
    if row.coverage_status.is_some() && row.coverage_total > 0 {
        spans.push(Span::styled(format!("  {}/{}", row.coverage_seen, row.coverage_total), Style::default().fg(Color::Gray)));
        spans.extend(counts(row).into_iter().filter(|s| s.content.contains('!')));
    }
    spans.push(Span::styled(format!("  {} ~{} tok", row.line_range, row.token_count), dim));
    Line::from(spans)
}

/// `  !4  ✎3`: changed-since-read and write counts, when there are any.
fn counts(row: &ambits::app::TreeRow) -> Vec<Span<'static>> {
    let mut out = Vec::new();
    if row.stale_count > 0 {
        out.push(Span::styled(format!("  !{}", row.stale_count), Style::default().fg(colors::DEPTH_STALE)));
    }
    if let Some(mark) = &row.write {
        out.push(Span::styled(format!("  ✎{}", mark.count), Style::default().fg(write_color(mark.status))));
    }
    out
}

/// `width` cells in proportion: `█` read in full, `▓` partly read, `░` unseen.
pub(super) fn coverage_bar(full: usize, seen: usize, total: usize, width: usize) -> Vec<Span<'static>> {
    let cells = |n: usize| (n * width + total / 2).checked_div(total).unwrap_or(0);
    // Anything read shows at least one cell.
    let full_cells = if full > 0 { cells(full).max(1) } else { 0 };
    let seen_cells = if seen > 0 { cells(seen).max(full_cells + usize::from(seen > full)).min(width) } else { 0 };
    vec![
        Span::styled("█".repeat(full_cells), Style::default().fg(colors::DEPTH_FULL_BODY)),
        Span::styled("▓".repeat(seen_cells - full_cells), Style::default().fg(colors::DEPTH_OVERVIEW)),
        Span::styled("░".repeat(width - seen_cells), Style::default().fg(colors::DEPTH_UNSEEN)),
    ]
}

/// `●` full, `◕` signature, `◑` overview, `◔` name, `·` unseen: fuller
/// is deeper.
pub(super) fn depth_glyph(depth: ReadDepth) -> char {
    match depth {
        ReadDepth::Unseen => '·',
        ReadDepth::NameOnly => '◔',
        ReadDepth::Overview => '◑',
        ReadDepth::Signature => '◕',
        ReadDepth::FullBody => '●',
    }
}

/// Stale symbols get the stale color regardless of how deeply they were read —
/// "what you know is out of date" is the more urgent fact than "how much of it
/// you read".
pub(super) fn depth_color(depth: ReadDepth, stale: bool) -> Color {
    if stale && depth.is_seen() {
        return colors::DEPTH_STALE;
    }
    match depth {
        ReadDepth::Unseen => colors::DEPTH_UNSEEN,
        ReadDepth::NameOnly => colors::DEPTH_NAME_ONLY,
        ReadDepth::Overview => colors::DEPTH_OVERVIEW,
        ReadDepth::Signature => colors::DEPTH_SIGNATURE,
        ReadDepth::FullBody => colors::DEPTH_FULL_BODY,
    }
}

/// Green while the agent's version is still there, amber once it changed,
/// red once it is gone.
pub(super) fn write_color(status: Status) -> Color {
    match status {
        Status::Current => colors::WRITE_CURRENT,
        Status::Changed => colors::WRITE_CHANGED,
        Status::Removed => colors::WRITE_REMOVED,
        Status::Unknown => colors::WRITE_UNKNOWN,
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
            label: "fn", file_path: std::sync::Arc::new(PathBuf::new()),
            byte_range: 0..100, line_range: 1..10, content_hash: hash,
            merkle_hash: hash, children: Vec::new(), estimated_tokens: 30,
        }
    }

    fn test_app() -> App {
        let tree = ProjectTree {
            root: PathBuf::from("/test"),
            files: vec![
                FileSymbols { file_path: "mock/a.rs".into(), symbols: vec![sym("a1", "alpha"), sym("a2", "beta")], total_lines: 50 },
                FileSymbols { file_path: "mock/b.rs".into(), symbols: vec![sym("b1", "gamma")], total_lines: 30 },
            ],
        };
        App::new(tree, PathBuf::from("/test"))
    }

    /// Find the foreground color of the first cell in `row` that contains part of `text`.
    fn fg_color_of(backend: &TestBackend, row: u16, text: &str) -> Option<Color> {
        let buf = backend.buffer();
        let row_str = crate::ui::test_render::row(backend, row);
        // A column, not a byte offset: the border and the expand icons are
        // multi-byte.
        let col = row_str[..row_str.find(text)?].chars().count() as u16;
        Some(buf[(col, row)].fg)
    }

    #[test]
    fn depth_color_variants() {
        assert_eq!(depth_color(ReadDepth::Unseen, false), colors::DEPTH_UNSEEN);
        assert_eq!(depth_color(ReadDepth::NameOnly, false), colors::DEPTH_NAME_ONLY);
        assert_eq!(depth_color(ReadDepth::Overview, false), colors::DEPTH_OVERVIEW);
        assert_eq!(depth_color(ReadDepth::Signature, false), colors::DEPTH_SIGNATURE);
        assert_eq!(depth_color(ReadDepth::FullBody, false), colors::DEPTH_FULL_BODY);
    }

    #[test]
    fn stale_overrides_depth_color_for_seen_symbols() {
        assert_eq!(depth_color(ReadDepth::FullBody, true), colors::DEPTH_STALE);
        assert_eq!(depth_color(ReadDepth::NameOnly, true), colors::DEPTH_STALE);
        // An unseen symbol can't be stale; don't let a bad flag recolor it.
        assert_eq!(depth_color(ReadDepth::Unseen, true), colors::DEPTH_UNSEEN);
    }

    /// A written symbol and its file carry a green `✎` while the agent's
    /// version is there; the file's counts its writes.
    #[test]
    fn render_marks_written_rows() {
        let tree = ProjectTree {
            root: PathBuf::from("/test"),
            files: vec![FileSymbols { file_path: "mock/w.rs".into(), symbols: vec![sym("mock/w.rs::alpha", "alpha"), sym("mock/w.rs::beta", "beta")], total_lines: 5 }],
        };
        let mut app = App::new(tree, PathBuf::from("/test"));
        app.set_session_id(Some("sess".into()));
        app.set_expanded("mock/w.rs", ambits::expansion::RowKind::File, true);
        let alpha = &app.project_tree.files[0].symbols[0];
        let write = ambits::writes::WriteRecord {
            op: "toolu_1".into(),
            file: "mock/w.rs".into(),
            level: ambits::writes::Level::Symbol,
            syms: vec![(alpha.id.clone(), ambits::journal::encode_hash(&alpha.content_hash))],
            ..Default::default()
        };
        app.record_write("sess", write);
        app.selected_index = 2;

        let mut terminal = Terminal::new(TestBackend::new(80, 6)).unwrap();
        terminal.draw(|f| render(f, &app, f.area())).unwrap();
        assert_eq!(fg_color_of(terminal.backend(), 1, "✎1"), Some(colors::WRITE_CURRENT));
        assert_eq!(fg_color_of(terminal.backend(), 2, "✎"), Some(colors::WRITE_CURRENT));
        assert_eq!(fg_color_of(terminal.backend(), 3, "✎"), None, "beta was not written");
    }

    fn text_of(line: &Line) -> String {
        line.spans.iter().map(|s| s.content.as_ref()).collect()
    }

    /// The bar is in proportion, and anything read shows.
    #[test]
    fn the_file_bar_shows_full_partial_and_unseen() {
        let bar = |full, seen, total| coverage_bar(full, seen, total, 10).iter().map(|s| s.content.to_string()).collect::<String>();
        assert_eq!(bar(0, 0, 10), "░░░░░░░░░░");
        assert_eq!(bar(5, 8, 10), "█████▓▓▓░░");
        assert_eq!(bar(10, 10, 10), "██████████");
        assert_eq!(bar(1, 2, 1000), "█▓░░░░░░░░", "one read in a thousand still shows");
        assert_eq!(bar(0, 0, 0), "░░░░░░░░░░");
    }

    /// Each state is a glyph in its own column: depth, freshness, write.
    #[test]
    fn a_symbol_row_spells_its_states_in_the_gutter() {
        let mut app = test_app();
        app.set_expanded("mock/a.rs", ambits::expansion::RowKind::File, true);
        app.ledger.record("a1".into(), ReadDepth::FullBody, [0; 32], "ag".into(), 10);
        app.ledger.mark_stale("a1");
        app.ledger.record("a2".into(), ReadDepth::NameOnly, [0; 32], "ag".into(), 10);
        app.rebuild_tree_rows();
        let rows: Vec<String> = app.tree_rows.iter().map(|r| text_of(&row_line(r, 0))).collect();
        assert!(rows[0].starts_with("▼ mock/a.rs  ") && rows[0].contains("2/2") && rows[0].contains("!1"), "{}", rows[0]);
        assert!(rows[1].starts_with("    ●!  fn alpha"), "{}", rows[1]);
        assert!(rows[2].starts_with("    ◔   fn beta"), "{}", rows[2]);
        assert!(rows[3].starts_with("▶ mock/b.rs  ░░░░░░░░░░  0/1"), "{}", rows[3]);

        app.ledger.mark_all_restored();
        app.rebuild_tree_rows();
        assert!(text_of(&row_line(&app.tree_rows[2], 0)).starts_with("    ◔◌  fn beta"));
        assert!(text_of(&row_line(&app.tree_rows[3], 12)).starts_with("▶ mock/b.rs     ░"), "bars align after the longest path");
    }

    /// Row text of `row` in the rendered buffer.
    fn row_text(backend: &TestBackend, row: u16) -> String {
        crate::ui::test_render::row(backend, row)
    }

    /// An app whose only file holds `Parent` with two children, file
    /// expanded, `Parent` collapsed, one child read.
    fn app_with_a_read_child() -> App {
        let mut parent = sym("p", "Parent");
        parent.children = vec![sym("p/a", "a"), sym("p/b", "b")];
        let tree = ProjectTree {
            root: PathBuf::from("/test"),
            files: vec![FileSymbols { file_path: "mock/p.rs".into(), symbols: vec![parent], total_lines: 20 }],
        };
        let mut app = App::new(tree, PathBuf::from("/test"));
        app.set_expanded("mock/p.rs", ambits::expansion::RowKind::File, true);
        app.set_expanded("p", ambits::expansion::RowKind::Symbol, false);
        app.ledger.record("p/a".into(), ReadDepth::FullBody, [0; 32], "ag".into(), 10);
        app.rebuild_tree_rows();
        app.selected_index = 0;
        app
    }

    fn draw(app: &App) -> Terminal<TestBackend> {
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal.draw(|f| render(f, app, f.area())).unwrap();
        terminal
    }

    /// Reads land on innermost symbols. A collapsed parent shows how many
    /// of its children were read, or an unread parent would hide them.
    #[test]
    fn a_collapsed_parent_shows_its_read_children() {
        let app = app_with_a_read_child();
        let terminal = draw(&app);
        let row = row_text(terminal.backend(), 2);
        assert!(row.contains("▸ ·   fn Parent  1/2"), "{row}");
    }

    /// Expanded, the children speak for themselves.
    #[test]
    fn an_expanded_parent_leaves_it_to_its_children() {
        let mut app = app_with_a_read_child();
        app.set_expanded("p", ambits::expansion::RowKind::Symbol, true);
        app.rebuild_tree_rows();
        let terminal = draw(&app);
        assert!(!row_text(terminal.backend(), 2).contains("1/2"));
        assert!(row_text(terminal.backend(), 3).contains("●   fn a"), "{}", row_text(terminal.backend(), 3));
        assert_eq!(fg_color_of(terminal.backend(), 3, "●"), Some(colors::DEPTH_FULL_BODY));
    }

    /// A name-only read must not look unread.
    #[test]
    fn name_only_is_distinguishable_from_unseen() {
        assert_ne!(colors::DEPTH_NAME_ONLY, colors::DEPTH_UNSEEN);
        assert_ne!(depth_color(ReadDepth::NameOnly, false), depth_color(ReadDepth::Unseen, false));
    }
}
