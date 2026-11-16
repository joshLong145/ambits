//! The README's pictures of the TUI.
//!
//! Writes a scripted Claude Code session over this repo's own `src/`, runs
//! the `ambits` binary on it in a pseudo-terminal, presses the keys a user
//! would, and writes each screen to `images/` as SVG.
//!
//! ```bash
//! cargo build --release
//! cargo run --manifest-path tools/readme-images/Cargo.toml
//! ```
//!
//! The first argument overrides the binary (default `target/release/ambits`).

use std::fmt::Write as _;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use portable_pty::{native_pty_system, CommandBuilder, PtySize};
use serde_json::{json, Value};

const SESSION: &str = "7c1f2a9e-4b3d-4e8a-9f60-2d5b8c1e3a47";
const AGENT: &str = "a3f9c1d";

const COLS: u16 = 132;
const ROWS: u16 = 34;

/// One cell, in pixels.
const CELL_W: f64 = 8.4;
const CELL_H: f64 = 18.0;
const FONT_SIZE: f64 = 14.0;
const PAD: f64 = 12.0;
const BAR: f64 = 28.0;

const BACKGROUND: &str = "#0d1117";
const FOREGROUND: &str = "#c9d1d9";

/// The 16 ANSI colours, as GitHub's dark theme draws them.
const ANSI: [&str; 16] = [
    "#484f58", "#ff7b72", "#3fb950", "#d29922", "#58a6ff", "#bc8cff", "#39c5cf", "#b1bac4",
    "#6e7681", "#ffa198", "#56d364", "#e3b341", "#79c0ff", "#d2a8ff", "#56d4dd", "#f0f6fc",
];

fn hex(color: vt100::Color) -> Option<String> {
    match color {
        vt100::Color::Default => None,
        vt100::Color::Rgb(r, g, b) => Some(format!("#{r:02x}{g:02x}{b:02x}")),
        vt100::Color::Idx(i) => Some(match i {
            0..=15 => ANSI[i as usize].to_string(),
            16..=231 => {
                let level = |v: u8| if v == 0 { 0 } else { 55 + v * 40 };
                let i = i - 16;
                format!(
                    "#{:02x}{:02x}{:02x}",
                    level(i / 36),
                    level(i / 6 % 6),
                    level(i % 6)
                )
            }
            _ => {
                let v = 8 + (i - 232) * 10;
                format!("#{v:02x}{v:02x}{v:02x}")
            }
        }),
    }
}

/// A cell's foreground and background, inversion applied.
fn colors(cell: &vt100::Cell) -> (String, Option<String>) {
    let fg = hex(cell.fgcolor());
    let bg = hex(cell.bgcolor());
    if cell.inverse() {
        (
            bg.unwrap_or_else(|| BACKGROUND.into()),
            Some(fg.unwrap_or_else(|| FOREGROUND.into())),
        )
    } else {
        (fg.unwrap_or_else(|| FOREGROUND.into()), bg)
    }
}

/// The arms (up, down, left, right) of a light box-drawing character.
fn arms(c: char) -> Option<[bool; 4]> {
    Some(match c {
        '─' => [false, false, true, true],
        '│' => [true, true, false, false],
        '┌' | '╭' => [false, true, false, true],
        '┐' | '╮' => [false, true, true, false],
        '└' | '╰' => [true, false, false, true],
        '┘' | '╯' => [true, false, true, false],
        '├' => [true, true, false, true],
        '┤' => [true, true, true, false],
        '┬' => [false, true, true, true],
        '┴' => [true, false, true, true],
        '┼' => [true, true, true, true],
        _ => return None,
    })
}

/// A block character as the part of its cell it fills — (x, y, w, h) as
/// fractions of the cell — and its opacity.
fn block(c: char) -> Option<(f64, f64, f64, f64, f64)> {
    let eighths = |n: u32| f64::from(n) / 8.0;
    Some(match c {
        '█' => (0.0, 0.0, 1.0, 1.0, 1.0),
        '▓' => (0.0, 0.0, 1.0, 1.0, 0.75),
        '▒' => (0.0, 0.0, 1.0, 1.0, 0.5),
        '░' => (0.0, 0.0, 1.0, 1.0, 0.25),
        '▐' => (0.5, 0.0, 0.5, 1.0, 1.0),
        '▀' => (0.0, 0.0, 1.0, 0.5, 1.0),
        '▁'..='▇' => {
            let h = eighths(c as u32 - '▁' as u32 + 1);
            (0.0, 1.0 - h, 1.0, h, 1.0)
        }
        '▉'..='▏' => (0.0, 0.0, eighths('▏' as u32 - c as u32 + 1), 1.0, 1.0),
        _ => return None,
    })
}

/// The single character a cell holds, when it is one.
fn single(cell: &vt100::Cell) -> Option<char> {
    let mut chars = cell.contents().chars();
    let c = chars.next()?;
    chars.next().is_none().then_some(c)
}

fn is_shape(cell: &vt100::Cell) -> bool {
    single(cell).is_some_and(|c| arms(c).is_some() || block(c).is_some())
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// `screen` as an SVG of a terminal window: backgrounds, borders and blocks
/// drawn as shapes so they join up, and text in runs of one style stretched
/// to their cells so columns line up in any monospace font.
fn to_svg(screen: &vt100::Screen, title: &str) -> String {
    let (rows, cols) = screen.size();
    let (rows, cols) = (rows.get(), cols.get());
    let width = f64::from(cols) * CELL_W + 2.0 * PAD;
    let height = f64::from(rows) * CELL_H + 2.0 * PAD + BAR;
    let x_of = |col: u16| PAD + f64::from(col) * CELL_W;
    let cell = |row: u16, col: u16| screen.cell(row, col).expect("cell on screen");
    let mut shapes = String::new();
    let mut text = String::new();

    for row in 0..rows {
        let y = BAR + PAD + f64::from(row) * CELL_H;
        let mut col = 0;
        while col < cols {
            let bg = colors(cell(row, col)).1;
            let start = col;
            while col < cols && colors(cell(row, col)).1 == bg {
                col += 1;
            }
            if let Some(bg) = bg {
                let _ = write!(
                    shapes,
                    r#"<rect x="{:.1}" y="{y:.1}" width="{:.1}" height="{CELL_H}" fill="{bg}"/>"#,
                    x_of(start),
                    f64::from(col - start) * CELL_W
                );
            }
        }

        let mut col = 0;
        while col < cols {
            let c = cell(row, col);
            let fg = colors(c).0;
            if let Some([up, down, left, right]) = single(c).and_then(arms) {
                let (x, cx, cy) = (x_of(col), x_of(col) + CELL_W / 2.0, y + CELL_H / 2.0);
                let mut d = String::new();
                if up {
                    let _ = write!(d, "M{cx:.1} {y:.1}V{cy:.1}");
                }
                if down {
                    let _ = write!(d, "M{cx:.1} {cy:.1}V{:.1}", y + CELL_H);
                }
                if left {
                    let _ = write!(d, "M{x:.1} {cy:.1}H{cx:.1}");
                }
                if right {
                    let _ = write!(d, "M{cx:.1} {cy:.1}H{:.1}", x + CELL_W);
                }
                let _ = write!(
                    shapes,
                    r#"<path d="{d}" stroke="{fg}" stroke-width="1.2" fill="none"/>"#
                );
                col += 1;
                continue;
            }
            if let Some((bx, by, bw, bh, opacity)) = single(c).and_then(block) {
                // A run of the same full-width block is one rect, without
                // seams between its cells.
                let start = col;
                col += 1;
                while bw == 1.0
                    && col < cols
                    && cell(row, col).contents() == c.contents()
                    && colors(cell(row, col)).0 == fg
                {
                    col += 1;
                }
                let _ = write!(
                    shapes,
                    r#"<rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" fill="{fg}" fill-opacity="{opacity}"/>"#,
                    x_of(start) + bx * CELL_W,
                    y + by * CELL_H,
                    (f64::from(col - start - 1) + bw) * CELL_W,
                    bh * CELL_H
                );
                continue;
            }
            let blank = |c: &vt100::Cell| !c.has_contents() || c.contents() == " ";
            if blank(c) {
                col += 1;
                continue;
            }
            // A word in one style: each placed at its own column, so no
            // glyph is stretched across the spaces between them.
            let style = (fg, c.bold(), c.dim());
            let start = col;
            let mut run = String::new();
            while col < cols {
                let c = cell(row, col);
                if blank(c) || is_shape(c) || (colors(c).0, c.bold(), c.dim()) != style {
                    break;
                }
                if !c.is_wide_continuation() {
                    run.push_str(c.contents());
                }
                col += 1;
            }
            let _ = write!(
                text,
                r#"<text x="{:.1}" y="{:.1}" textLength="{:.1}" lengthAdjust="spacingAndGlyphs" fill="{}"{}{}>{}</text>"#,
                x_of(start),
                y + CELL_H * 0.75,
                f64::from(col - start) * CELL_W,
                style.0,
                if style.1 {
                    r#" font-weight="bold""#
                } else {
                    ""
                },
                if style.2 {
                    r#" fill-opacity="0.6""#
                } else {
                    ""
                },
                escape(&run)
            );
        }
    }

    let title = escape(title);
    let font = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'DejaVu Sans Mono', monospace";
    format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0}" height="{height:.0}" viewBox="0 0 {width:.0} {height:.0}" role="img" aria-label="{title}">
<rect width="100%" height="100%" rx="8" fill="{BACKGROUND}"/>
<circle cx="18" cy="14" r="5" fill="#ff5f56"/><circle cx="34" cy="14" r="5" fill="#ffbd2e"/><circle cx="50" cy="14" r="5" fill="#27c93f"/>
<text x="{:.0}" y="18" fill="#8b949e" font-family="{font}" font-size="12" text-anchor="middle">{title}</text>
<g>{shapes}</g>
<g font-family="{font}" font-size="{FONT_SIZE}" xml:space="preserve">{text}</g>
</svg>
"##,
        width / 2.0
    )
}

fn assistant(agent: &str, at: &str, id: &str, name: &str, input: Value) -> Value {
    json!({"type": "assistant", "sessionId": SESSION, "agentId": agent, "timestamp": at,
           "message": {"role": "assistant", "content": [{"type": "tool_use", "id": id, "name": name, "input": input}]}})
}

fn result(at: &str, id: &str, detail: Value) -> Value {
    json!({"type": "user", "sessionId": SESSION, "timestamp": at, "toolUseResult": detail,
           "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": id, "content": "ok"}]}})
}

fn prompt(at: &str, text: &str) -> Value {
    json!({"type": "user", "sessionId": SESSION, "timestamp": at,
           "message": {"role": "user", "content": text}})
}

fn write_log(path: &Path, lines: &[Value]) {
    std::fs::create_dir_all(path.parent().expect("a log has a directory")).expect("log dir");
    let body: String = lines.iter().map(|l| format!("{l}\n")).collect();
    std::fs::write(path, body).expect("log written");
}

/// The scripted session: a question about trace span ids, answered by
/// reading the trace code, a search, a subagent that maps the report's
/// side, and a test run.
fn session_logs(root: &Path, logs: &Path) {
    let file = |p: &str| root.join(p).to_string_lossy().into_owned();
    let t = |s: &str| format!("2026-11-16T10:{s}Z");
    write_log(
        &logs.join(format!("{SESSION}.jsonl")),
        &[
            prompt(
                &t("00:00.000"),
                "Why do OTLP span ids change when a trace gains new spans?",
            ),
            assistant(
                SESSION,
                &t("00:04.100"),
                "toolu_01",
                "Read",
                json!({"file_path": file("src/trace/export.rs")}),
            ),
            result(&t("00:04.300"), "toolu_01", json!({})),
            assistant(
                SESSION,
                &t("00:09.800"),
                "toolu_02",
                "Grep",
                json!({"pattern": "fn numbering|span_id", "path": file("src"), "output_mode": "content"}),
            ),
            result(&t("00:10.050"), "toolu_02", json!({})),
            assistant(
                SESSION,
                &t("00:15.400"),
                "toolu_03",
                "Read",
                json!({"file_path": file("src/trace/mod.rs"), "offset": 1, "limit": 220}),
            ),
            result(&t("00:15.520"), "toolu_03", json!({})),
            assistant(
                SESSION,
                &t("00:21.000"),
                "toolu_04",
                "Agent",
                json!({"description": "Map the report's span ids", "prompt": "Find where report.rs numbers spans and who reads those numbers.", "subagent_type": "Explore"}),
            ),
            result(
                &t("00:58.600"),
                "toolu_04",
                json!({"agentId": AGENT, "status": "completed"}),
            ),
            assistant(
                SESSION,
                &t("01:04.200"),
                "toolu_05",
                "Read",
                json!({"file_path": file("src/trace/view.rs"), "offset": 1, "limit": 160}),
            ),
            result(&t("01:04.330"), "toolu_05", json!({})),
            assistant(
                SESSION,
                &t("01:11.000"),
                "toolu_06",
                "Bash",
                json!({"command": "cargo test --test trace", "description": "Run the trace export tests"}),
            ),
            result(
                &t("01:39.400"),
                "toolu_06",
                json!({"stdout": "test result: ok. 9 passed", "stderr": "", "interrupted": false}),
            ),
        ],
    );
    write_log(
        &logs
            .join(SESSION)
            .join(format!("subagents/agent-{AGENT}.jsonl")),
        &[
            assistant(
                AGENT,
                &t("00:23.000"),
                "toolu_11",
                "Grep",
                json!({"pattern": "SpanDto", "path": file("src"), "output_mode": "files_with_matches"}),
            ),
            result(&t("00:23.200"), "toolu_11", json!({})),
            assistant(
                AGENT,
                &t("00:28.700"),
                "toolu_12",
                "Read",
                json!({"file_path": file("src/report.rs")}),
            ),
            result(&t("00:28.950"), "toolu_12", json!({})),
            assistant(
                AGENT,
                &t("00:41.300"),
                "toolu_13",
                "Read",
                json!({"file_path": file("src/trace/summary.rs")}),
            ),
            result(&t("00:41.500"), "toolu_13", json!({})),
            assistant(
                AGENT,
                &t("00:52.000"),
                "toolu_14",
                "Read",
                json!({"file_path": file("src/trace/mod.rs"), "offset": 220, "limit": 200}),
            ),
            result(&t("00:52.100"), "toolu_14", json!({})),
        ],
    );
}

/// The TUI in a pseudo-terminal, its output fed to a terminal emulator.
struct Tui {
    parser: vt100::Parser,
    output: mpsc::Receiver<Vec<u8>>,
    input: Box<dyn Write + Send>,
    child: Box<dyn portable_pty::Child + Send + Sync>,
}

impl Tui {
    fn start(binary: &Path, args: &[String]) -> Self {
        let pty = native_pty_system()
            .openpty(PtySize {
                rows: ROWS,
                cols: COLS,
                pixel_width: 0,
                pixel_height: 0,
            })
            .expect("a pseudo-terminal");
        let mut command = CommandBuilder::new(binary);
        command.args(args);
        command.env("TERM", "xterm-256color");
        command.env("COLORTERM", "truecolor");
        let child = pty.slave.spawn_command(command).expect("ambits starts");
        let mut reader = pty.master.try_clone_reader().expect("pty reader");
        let input = pty.master.take_writer().expect("pty writer");
        let (tx, output) = mpsc::channel();
        std::thread::spawn(move || {
            let mut buf = [0u8; 8192];
            while let Ok(n) = reader.read(&mut buf) {
                if n == 0 || tx.send(buf[..n].to_vec()).is_err() {
                    break;
                }
            }
        });
        Self {
            parser: vt100::Parser::new(
                std::num::NonZeroU16::new(ROWS).expect("rows"),
                std::num::NonZeroU16::new(COLS).expect("cols"),
                0,
            ),
            output,
            input,
            child,
        }
    }

    /// Feed the emulator everything drawn over the next few seconds: the TUI
    /// redraws on a timer, so it never falls silent to say it is done.
    fn settle(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(3);
        while let Some(left) = deadline.checked_duration_since(Instant::now()) {
            match self.output.recv_timeout(left) {
                Ok(bytes) => self.parser.process(&bytes),
                Err(mpsc::RecvTimeoutError::Timeout) => break,
                Err(mpsc::RecvTimeoutError::Disconnected) => panic!("ambits exited early"),
            }
        }
    }

    fn press(&mut self, keys: &[u8]) {
        self.input.write_all(keys).expect("key sent");
        self.input.flush().expect("key flushed");
        self.settle();
    }

    fn save(&self, path: &Path, title: &str) {
        std::fs::write(path, to_svg(self.parser.screen(), title)).expect("svg written");
        println!("wrote {}", path.display());
    }
}

impl Drop for Tui {
    fn drop(&mut self) {
        let _ = self.input.write_all(b"q");
        let _ = self.child.kill();
    }
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("the repo root");
    let binary = std::env::args()
        .nth(1)
        .map_or_else(|| root.join("target/release/ambits"), PathBuf::from);
    assert!(
        binary.exists(),
        "no binary at {}; run `cargo build --release`",
        binary.display()
    );

    let scratch = std::env::temp_dir().join(format!("ambits-readme-{}", std::process::id()));
    let logs = scratch.join("logs");
    session_logs(&root, &logs);
    let path = |p: &Path| p.to_string_lossy().into_owned();
    let args = [
        "-p".into(),
        path(&root),
        "--log-dir".into(),
        path(&logs),
        "--session".into(),
        SESSION.into(),
        "--filter".into(),
        "src".into(),
        "--no-journal".into(),
        "--state-dir".into(),
        path(&scratch.join("state")),
    ];

    let images = root.join("images");
    let mut tui = Tui::start(&binary, &args);
    tui.settle();
    // Sorted by coverage, the most-read file opened to its symbols.
    tui.press(b"s");
    tui.press(b"l");
    tui.save(&images.join("tui.svg"), "ambits -p .");
    tui.press(b"t");
    tui.press(b"\r");
    tui.save(&images.join("trace.svg"), "ambits -p .  —  t: traces");
    drop(tui);
    let _ = std::fs::remove_dir_all(&scratch);
}
