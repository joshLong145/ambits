//! Shared color palette for the TUI.

use ratatui::style::Color;

// ── Read-depth colors (symbol level) ────────────────────────────────
pub const DEPTH_UNSEEN: Color = Color::Rgb(100, 100, 100);
pub const DEPTH_NAME_ONLY: Color = Color::Rgb(160, 160, 160);
pub const DEPTH_OVERVIEW: Color = Color::Rgb(120, 160, 220);
pub const DEPTH_SIGNATURE: Color = Color::Rgb(80, 140, 255);
pub const DEPTH_FULL_BODY: Color = Color::Rgb(80, 220, 120);
pub const DEPTH_STALE: Color = Color::Rgb(230, 160, 60);

// ── File coverage colors (file header level) ────────────────────────
pub const FILE_FULLY_COVERED: Color = Color::Rgb(80, 220, 120);
pub const FILE_ALL_SEEN: Color = Color::Rgb(180, 220, 80);
pub const FILE_PARTIALLY_COVERED: Color = Color::Rgb(255, 180, 50);
pub const FILE_NOT_COVERED: Color = Color::White;

// ── Coverage percentage gradient ────────────────────────────────────
pub const PCT_LOW: Color = Color::Rgb(180, 60, 60);
pub const PCT_MID_LOW: Color = Color::Rgb(230, 160, 60);
pub const PCT_MID_HIGH: Color = Color::Rgb(200, 200, 80);
pub const PCT_HIGH: Color = Color::Rgb(80, 220, 120);

// ── Accent / chrome ─────────────────────────────────────────────────
pub const ACCENT_MUTED: Color = Color::Rgb(120, 120, 180);
pub const HIGHLIGHT_BG: Color = Color::Rgb(60, 55, 50);
pub const HIGHLIGHT_FG: Color = Color::Rgb(255, 220, 150);
