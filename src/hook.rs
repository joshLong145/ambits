//! Register ambit's `SessionStart` hook with Claude Code.
//!
//! Closes the loop the rest of the cache work opens. Everything else produces
//! a digest that only helps if somebody remembers to run it; this makes Claude
//! Code run it automatically the moment a compaction lands, and inject the
//! result as context.
//!
//! ## Why a hook rather than writing to the session log
//!
//! The obvious-looking alternative — appending to the session JSONL so the
//! reads "come back" — is a trap. That file is Claude Code's private state,
//! held open and appended concurrently, in an undocumented versioned format.
//! Worse, ambit's own `LogTailer` reads it and tracks byte offsets into it, so
//! synthetic records would be re-ingested as genuine agent activity: we would
//! mark symbols read that nobody read, then journal those fabricated reads.
//! The cache would poison itself. A hook has Claude Code do the injection, and
//! we write nothing.

use std::path::{Path, PathBuf};

use color_eyre::eyre::{Result, WrapErr};
use serde_json::{json, Value};

/// Marker used to recognize an entry we previously installed, so re-running
/// the installer updates rather than duplicating.
const HOOK_MARKER: &str = "restore-context --format hook";

/// Build the command Claude Code should run.
///
/// The project path is baked in as an absolute path rather than relying on
/// `--project .`: hooks run in the session's cwd, which is not guaranteed to
/// stay at the project root.
fn hook_command(project_root: &Path) -> String {
    format!(
        "ambits --project {} restore-context --format hook",
        project_root.display()
    )
}

/// The hook group we install under `hooks.SessionStart`.
///
/// `matcher: "compact"` restricts it to the post-compaction start. Matchers
/// are exact and case-sensitive.
fn hook_entry(project_root: &Path) -> Value {
    json!({
        "matcher": "compact",
        "hooks": [{
            "type": "command",
            "command": hook_command(project_root),
        }]
    })
}

/// Does this hook group look like one of ours?
fn is_ambit_entry(entry: &Value) -> bool {
    entry
        .get("hooks")
        .and_then(Value::as_array)
        .map(|hooks| {
            hooks.iter().any(|h| {
                h.get("command")
                    .and_then(Value::as_str)
                    .is_some_and(|c| c.contains(HOOK_MARKER))
            })
        })
        .unwrap_or(false)
}

/// Install (or update) the hook in a settings file.
///
/// Merges into existing settings rather than overwriting: this file is the
/// user's, and may hold unrelated configuration we must not touch. A settings
/// file we cannot parse is left strictly alone — we print the snippet instead
/// of risking clobbering it.
pub fn install(global: bool, project: Option<PathBuf>) -> Result<()> {
    let project_root = project
        .clone()
        .unwrap_or_else(|| PathBuf::from("."))
        .canonicalize()
        .wrap_err("could not resolve the project directory")?;

    let settings_path = if global {
        let home = std::env::var("HOME").wrap_err("HOME environment variable not set")?;
        PathBuf::from(home).join(".claude/settings.json")
    } else {
        project_root.join(".claude/settings.json")
    };

    let mut settings = match std::fs::read_to_string(&settings_path) {
        Ok(raw) if raw.trim().is_empty() => json!({}),
        Ok(raw) => match serde_json::from_str::<Value>(&raw) {
            Ok(v) => v,
            Err(e) => {
                // Refuse to guess at a malformed file. Losing the user's
                // settings is far worse than making them paste four lines.
                eprintln!("[ambit] {} is not valid JSON: {e}", settings_path.display());
                eprintln!("[ambit] Not modifying it. Add this to \"hooks\" yourself:\n");
                println!("{}", serde_json::to_string_pretty(&hook_entry(&project_root))?);
                return Ok(());
            }
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => json!({}),
        Err(e) => return Err(e).wrap_err_with(|| format!("reading {}", settings_path.display())),
    };

    if !settings.is_object() {
        eprintln!(
            "[ambit] {} is valid JSON but not an object; not modifying it.",
            settings_path.display()
        );
        return Ok(());
    }

    let entry = hook_entry(&project_root);
    let session_start = settings
        .as_object_mut()
        .expect("checked above")
        .entry("hooks")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .ok_or_else(|| color_eyre::eyre::eyre!("\"hooks\" in {} is not an object", settings_path.display()))?
        .entry("SessionStart")
        .or_insert_with(|| json!([]))
        .as_array_mut()
        .ok_or_else(|| {
            color_eyre::eyre::eyre!("\"hooks.SessionStart\" in {} is not an array", settings_path.display())
        })?;

    // Replace a previous ambit entry rather than appending a second one, so
    // re-running after moving the project doesn't leave a stale command
    // behind.
    let existing = session_start.iter().position(is_ambit_entry);
    let action = match existing {
        Some(i) if session_start[i] == entry => "already installed",
        Some(i) => {
            session_start[i] = entry;
            "updated"
        }
        None => {
            session_start.push(entry);
            "installed"
        }
    };

    if action != "already installed" {
        if let Some(dir) = settings_path.parent() {
            std::fs::create_dir_all(dir)
                .wrap_err_with(|| format!("creating {}", dir.display()))?;
        }
        std::fs::write(&settings_path, serde_json::to_string_pretty(&settings)? + "\n")
            .wrap_err_with(|| format!("writing {}", settings_path.display()))?;
    }

    println!("SessionStart hook {action}: {}", settings_path.display());
    println!("  {}", hook_command(&project_root));
    println!();
    println!("After a compaction, Claude Code will run this and inject the symbols");
    println!("this session has already read.");
    println!();
    println!("Requires `ambits` on PATH. Coverage must have been journaled by the");
    println!("TUI for this session, otherwise the digest is reconstructed from");
    println!("session logs and says so.");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings_after(initial: Option<&str>) -> (Value, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let claude = dir.path().join(".claude");
        std::fs::create_dir_all(&claude).unwrap();
        if let Some(raw) = initial {
            std::fs::write(claude.join("settings.json"), raw).unwrap();
        }
        install(false, Some(dir.path().to_path_buf())).unwrap();
        let raw = std::fs::read_to_string(claude.join("settings.json")).unwrap();
        (serde_json::from_str(&raw).unwrap(), dir)
    }

    #[test]
    fn creates_settings_when_absent() {
        let (v, _d) = settings_after(None);
        let entry = &v["hooks"]["SessionStart"][0];
        assert_eq!(entry["matcher"], "compact");
        assert!(entry["hooks"][0]["command"]
            .as_str()
            .unwrap()
            .contains("restore-context --format hook"));
    }

    /// The settings file is the user's; unrelated keys must survive.
    #[test]
    fn preserves_unrelated_settings() {
        let (v, _d) = settings_after(Some(
            r#"{"model":"opus","env":{"FOO":"bar"},"hooks":{"PreToolUse":[{"matcher":"Bash"}]}}"#,
        ));
        assert_eq!(v["model"], "opus");
        assert_eq!(v["env"]["FOO"], "bar");
        assert_eq!(v["hooks"]["PreToolUse"][0]["matcher"], "Bash");
        assert_eq!(v["hooks"]["SessionStart"][0]["matcher"], "compact");
    }

    /// Someone else's SessionStart hook must not be displaced.
    #[test]
    fn appends_alongside_foreign_session_start_hooks() {
        let (v, _d) = settings_after(Some(
            r#"{"hooks":{"SessionStart":[{"matcher":"startup","hooks":[{"type":"command","command":"echo hi"}]}]}}"#,
        ));
        let arr = v["hooks"]["SessionStart"].as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["hooks"][0]["command"], "echo hi");
        assert!(arr[1]["hooks"][0]["command"]
            .as_str()
            .unwrap()
            .contains("restore-context"));
    }

    #[test]
    fn reinstalling_does_not_duplicate() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join(".claude")).unwrap();
        install(false, Some(dir.path().to_path_buf())).unwrap();
        install(false, Some(dir.path().to_path_buf())).unwrap();

        let raw = std::fs::read_to_string(dir.path().join(".claude/settings.json")).unwrap();
        let v: Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(v["hooks"]["SessionStart"].as_array().unwrap().len(), 1);
    }

    /// A stale command (project moved) is replaced, not accumulated.
    #[test]
    fn updates_a_stale_entry_in_place() {
        let (v, _d) = settings_after(Some(
            r#"{"hooks":{"SessionStart":[{"matcher":"compact","hooks":[{"type":"command","command":"ambits --project /old/path restore-context --format hook"}]}]}}"#,
        ));
        let arr = v["hooks"]["SessionStart"].as_array().unwrap();
        assert_eq!(arr.len(), 1, "replaced rather than appended");
        assert!(!arr[0]["hooks"][0]["command"]
            .as_str()
            .unwrap()
            .contains("/old/path"));
    }

    /// Never rewrite a file we could not parse — losing real settings is far
    /// worse than making the user paste a snippet.
    #[test]
    fn leaves_malformed_settings_untouched() {
        let dir = tempfile::tempdir().unwrap();
        let claude = dir.path().join(".claude");
        std::fs::create_dir_all(&claude).unwrap();
        let broken = "{ this is not json";
        std::fs::write(claude.join("settings.json"), broken).unwrap();

        install(false, Some(dir.path().to_path_buf())).unwrap();

        assert_eq!(
            std::fs::read_to_string(claude.join("settings.json")).unwrap(),
            broken
        );
    }
}
