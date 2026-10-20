//! Shared color palette for the TUI.

use ratatui::style::Color;

// ── Read-depth colors (symbol level) ────────────────────────────────
pub const DEPTH_UNSEEN: Color = Color::Rgb(100, 100, 100);
// A hue of its own: a grey name-only read was indistinguishable from unseen.
pub const DEPTH_NAME_ONLY: Color = Color::Rgb(170, 140, 200);
pub const DEPTH_OVERVIEW: Color = Color::Rgb(120, 160, 220);
pub const DEPTH_SIGNATURE: Color = Color::Rgb(80, 140, 255);
pub const DEPTH_FULL_BODY: Color = Color::Rgb(80, 220, 120);
pub const DEPTH_STALE: Color = Color::Rgb(230, 160, 60);

// ── Alignment overlay ───────────────────────────────────────────────
pub const FILE_FULLY_COVERED: Color = Color::Rgb(80, 220, 120);
pub const FILE_PARTIALLY_COVERED: Color = Color::Rgb(255, 180, 50);

// ── Write marks (✎): whether the agent's version is still there ─────
pub const WRITE_CURRENT: Color = Color::Rgb(80, 220, 120);
pub const WRITE_CHANGED: Color = Color::Rgb(230, 160, 60);
pub const WRITE_REMOVED: Color = Color::Rgb(220, 80, 80);
/// A write with nothing in memory to compare (file-level).
pub const WRITE_UNKNOWN: Color = Color::Rgb(120, 200, 220);

// ── Coverage percentage gradient ────────────────────────────────────
pub const PCT_LOW: Color = Color::Rgb(180, 60, 60);
pub const PCT_MID_LOW: Color = Color::Rgb(230, 160, 60);
pub const PCT_MID_HIGH: Color = Color::Rgb(200, 200, 80);
pub const PCT_HIGH: Color = Color::Rgb(80, 220, 120);

// ── Accent / chrome ─────────────────────────────────────────────────
pub const ACCENT_MUTED: Color = Color::Rgb(120, 120, 180);
pub const HIGHLIGHT_BG: Color = Color::Rgb(60, 55, 50);
pub const HIGHLIGHT_FG: Color = Color::Rgb(255, 220, 150);
