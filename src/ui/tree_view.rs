use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState};

use ambits::app::{App, FileCoverageStatus, FocusPanel};
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
        .title(" Symbol Tree ")
        .borders(Borders::ALL)
        .border_style(border_style);

    let items: Vec<ListItem> = app
        .tree_rows
        .iter()
        .map(|row| {
            let indent = "  ".repeat(row.depth);
            let icon = if row.is_file() {
                if row.is_expanded { "▼ " } else { "▶ " }
            } else if row.has_children {
                if row.is_expanded { "▾ " } else { "▸ " }
            } else {
                "  "
            };

            let color = depth_color(row.read_depth, row.stale);

            let mut spans = vec![
                Span::raw(indent),
                Span::styled(icon, Style::default().fg(Color::DarkGray)),
            ];

            if row.is_file() {
                let file_color = file_coverage_color(row.coverage_status);
                spans.push(Span::styled(
                    &row.display_name,
                    Style::default().fg(file_color).add_modifier(Modifier::BOLD),
                ));
                if row.coverage_total > 0 {
                    spans.push(coverage_count(row, file_color));
                }
                if let Some(mark) = row.write {
                    spans.push(Span::styled(format!(" ✎{}", mark.count), Style::default().fg(write_color(mark.status))));
                }
                spans.push(Span::styled(
                    format!("  ({})", row.line_range),
                    Style::default().fg(Color::DarkGray),
                ));
            } else {
                spans.push(Span::styled(
                    format!("{} ", row.label),
                    Style::default().fg(Color::DarkGray),
                ));
                // A collapsed symbol with read descendants: color an unread
                // name by their coverage, and show how many, like a file row.
                let inner = row
                    .coverage_status
                    .filter(|_| row.coverage_seen > 0)
                    .map(|status| file_coverage_color(Some(status)));
                let name_color = match inner {
                    Some(c) if !row.read_depth.is_seen() => c,
                    _ => color,
                };
                spans.push(Span::styled(&row.display_name, symbol_style(name_color, row.restored)));
                if let Some(c) = inner {
                    spans.push(coverage_count(row, c));
                }
                if let Some(mark) = row.write {
                    spans.push(Span::styled(" ✎", Style::default().fg(write_color(mark.status))));
                }
                spans.push(Span::styled(
                    format!("  [{}] ~{} tok", row.line_range, row.token_count),
                    Style::default().fg(Color::DarkGray),
                ));
            }

            ListItem::new(Line::from(spans))
        })
        .collect();

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

/// `  seen/total` for a row that summarizes symbols it does not show.
fn coverage_count(row: &ambits::app::TreeRow, color: Color) -> Span<'static> {
    Span::styled(format!("  {}/{}", row.coverage_seen, row.coverage_total), Style::default().fg(color))
}

/// Restored reads keep their depth color but render dimmed, so pre-compaction
/// coverage stays visible without looking like it is still in the model's
/// context.
fn symbol_style(color: Color, restored: bool) -> Style {
    let style = Style::default().fg(color);
    if restored {
        style.add_modifier(Modifier::DIM)
    } else {
        style
    }
}

/// Stale symbols get the stale color regardless of how deeply they were read —
/// "what you know is out of date" is the more urgent fact than "how much of it
/// you read".
fn depth_color(depth: ReadDepth, stale: bool) -> Color {
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

fn file_coverage_color(status: Option<FileCoverageStatus>) -> Color {
    match status {
        Some(FileCoverageStatus::FullyCovered) => colors::FILE_FULLY_COVERED,
        Some(FileCoverageStatus::AllSeen) => colors::FILE_ALL_SEEN,
        Some(FileCoverageStatus::PartiallyCovered) => colors::FILE_PARTIALLY_COVERED,
        _ => colors::FILE_NOT_COVERED,
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
        let row_str: String = (0..buf.area.width)
            .map(|x| buf[(x, row)].symbol().to_string())
            .collect::<String>();
        // A column, not a byte offset: the border and the expand icons are
        // multi-byte.
        let col = row_str[..row_str.find(text)?].chars().count() as u16;
        Some(buf[(col, row)].fg)
    }

    #[test]
    fn file_coverage_color_variants() {
        assert_eq!(file_coverage_color(Some(FileCoverageStatus::FullyCovered)), colors::FILE_FULLY_COVERED);
        assert_eq!(file_coverage_color(Some(FileCoverageStatus::AllSeen)), colors::FILE_ALL_SEEN);
        assert_eq!(file_coverage_color(Some(FileCoverageStatus::PartiallyCovered)), colors::FILE_PARTIALLY_COVERED);
        assert_eq!(file_coverage_color(Some(FileCoverageStatus::NotCovered)), colors::FILE_NOT_COVERED);
        assert_eq!(file_coverage_color(None), colors::FILE_NOT_COVERED);
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

    #[test]
    fn render_uncovered_files_are_white() {
        let mut app = test_app();
        // Select row 2 so row 1 (a.rs) isn't highlighted.
        app.selected_index = 1;
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| render(f, &app, f.area())).unwrap();

        let color = fg_color_of(terminal.backend(), 1, "mock/a.rs").unwrap();
        assert_eq!(color, colors::FILE_NOT_COVERED);
    }

    #[test]
    fn render_fully_covered_file_is_green() {
        let mut app = test_app();
        app.selected_index = 1;
        app.ledger.record("a1".into(), ReadDepth::FullBody, [0; 32], "ag".into(), 10);
        app.ledger.record("a2".into(), ReadDepth::FullBody, [0; 32], "ag".into(), 10);
        app.rebuild_tree_rows();

        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| render(f, &app, f.area())).unwrap();

        let color = fg_color_of(terminal.backend(), 1, "mock/a.rs").unwrap();
        assert_eq!(color, colors::FILE_FULLY_COVERED);
    }

    #[test]
    fn render_all_seen_file_is_yellow_green() {
        let mut app = test_app();
        app.ledger.record("b1".into(), ReadDepth::NameOnly, [0; 32], "ag".into(), 10);
        app.rebuild_tree_rows();

        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| render(f, &app, f.area())).unwrap();

        let color = fg_color_of(terminal.backend(), 2, "mock/b.rs").unwrap();
        assert_eq!(color, colors::FILE_ALL_SEEN);
    }

    #[test]
    fn render_partially_covered_file_is_amber() {
        let mut app = test_app();
        app.selected_index = 1;
        app.ledger.record("a1".into(), ReadDepth::NameOnly, [0; 32], "ag".into(), 10);
        app.rebuild_tree_rows();

        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| render(f, &app, f.area())).unwrap();

        let color = fg_color_of(terminal.backend(), 1, "mock/a.rs").unwrap();
        assert_eq!(color, colors::FILE_PARTIALLY_COVERED);
    }

    #[test]
    fn render_expanded_symbol_has_depth_color() {
        let mut app = test_app();
        app.selected_index = 2;
        app.set_expanded("mock/a.rs", ambits::expansion::RowKind::File, true);
        app.ledger.record("a1".into(), ReadDepth::FullBody, [0; 32], "ag".into(), 10);
        app.rebuild_tree_rows();

        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| render(f, &app, f.area())).unwrap();

        let color = fg_color_of(terminal.backend(), 2, "alpha").unwrap();
        assert_eq!(color, colors::DEPTH_FULL_BODY);
    }

    /// Row text of `row` in the rendered buffer.
    fn row_text(backend: &TestBackend, row: u16) -> String {
        let buf = backend.buffer();
        (0..buf.area.width).map(|x| buf[(x, row)].symbol().to_string()).collect()
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

    /// Reads land on innermost symbols. A collapsed parent must show that
    /// something inside it was read, or an amber file expands to all grey.
    #[test]
    fn a_collapsed_parent_shows_its_read_children() {
        let app = app_with_a_read_child();
        let terminal = draw(&app);
        assert_eq!(fg_color_of(terminal.backend(), 2, "Parent"), Some(colors::FILE_PARTIALLY_COVERED));
        assert!(row_text(terminal.backend(), 2).contains("1/2"), "{}", row_text(terminal.backend(), 2));
    }

    /// Expanded, the children speak for themselves.
    #[test]
    fn an_expanded_parent_keeps_its_own_color() {
        let mut app = app_with_a_read_child();
        app.set_expanded("p", ambits::expansion::RowKind::Symbol, true);
        app.rebuild_tree_rows();
        let terminal = draw(&app);
        assert_eq!(fg_color_of(terminal.backend(), 2, "Parent"), Some(colors::DEPTH_UNSEEN));
        assert!(!row_text(terminal.backend(), 2).contains("1/2"));
        assert_eq!(fg_color_of(terminal.backend(), 3, "a  [L"), Some(colors::DEPTH_FULL_BODY));
    }

    /// A name-only read must not look unread.
    #[test]
    fn name_only_is_distinguishable_from_unseen() {
        assert_ne!(colors::DEPTH_NAME_ONLY, colors::DEPTH_UNSEEN);
        assert_ne!(depth_color(ReadDepth::NameOnly, false), depth_color(ReadDepth::Unseen, false));
    }
}
