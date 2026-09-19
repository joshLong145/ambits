//! Resolve and invoke an external editor to jump to a symbol's declaration.
//!
//! Precedence for *which* editor to use is CLI flag > project/user
//! `tools.toml` `[editor]` stanza > `$VISUAL` > `$EDITOR` > none. What that
//! resolves to is a command *template*: if it already names `{file}`/`{line}`
//! placeholders it is used verbatim, otherwise a small built-in table supplies
//! the right line-jump syntax for common editors so a bare `EDITOR=vim` still
//! lands on the correct line.

use std::path::Path;

/// Resolve the editor command template from the layered sources, highest
/// precedence first. Env values are passed in explicitly rather than read
/// here so resolution stays a pure function and is deterministic to test.
pub fn resolve_editor_template(
    cli: Option<&str>,
    cfg_command: Option<&str>,
    visual: Option<&str>,
    editor: Option<&str>,
) -> Option<String> {
    [cli, cfg_command, visual, editor]
        .into_iter()
        .find_map(|v| v.map(str::to_string))
}

/// Expand a resolved template into argv, substituting `{file}`/`{line}`.
///
/// If `template` contains neither placeholder, it is treated as a bare editor
/// command and expanded via the built-in per-basename default below —
/// otherwise the substitution happens verbatim, giving full control to a
/// template that already names them (e.g. `"code -g {file}:{line}"`).
pub fn build_editor_argv(template: &str, file: &Path, line: u32) -> Vec<String> {
    let file_str = file.to_string_lossy();
    let line_str = line.to_string();

    let expanded = if template.contains("{file}") || template.contains("{line}") {
        template.to_string()
    } else {
        default_template_for(template).replace("{editor}", template)
    };

    expanded
        .replace("{file}", &file_str)
        .replace("{line}", &line_str)
        .split_whitespace()
        .map(str::to_string)
        .collect()
}

/// The line-jump syntax for a bare editor command, keyed off the basename of
/// its first whitespace-separated token (so `"code --wait"` still matches
/// `"code"`). Unrecognized editors just get the file, no line jump — opening
/// the right file is still strictly better than not resolving at all.
fn default_template_for(command: &str) -> &'static str {
    let first_token = command.split_whitespace().next().unwrap_or(command);
    let basename = first_token
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(first_token);

    match basename {
        "vim" | "nvim" | "vi" | "emacs" | "emacsclient" | "nano" => "{editor} +{line} {file}",
        "code" | "code-insiders" => "{editor} --goto {file}:{line}",
        "subl" | "sublime_text" | "zed" => "{editor} {file}:{line}",
        _ => "{editor} {file}",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // resolve_editor_template
    // -----------------------------------------------------------------------

    #[test]
    fn cli_wins_over_everything() {
        assert_eq!(
            resolve_editor_template(Some("cli-editor"), Some("cfg"), Some("visual"), Some("editor")),
            Some("cli-editor".to_string())
        );
    }

    #[test]
    fn config_wins_over_env() {
        assert_eq!(
            resolve_editor_template(None, Some("cfg"), Some("visual"), Some("editor")),
            Some("cfg".to_string())
        );
    }

    #[test]
    fn visual_wins_over_editor() {
        assert_eq!(
            resolve_editor_template(None, None, Some("visual"), Some("editor")),
            Some("visual".to_string())
        );
    }

    #[test]
    fn editor_is_the_last_fallback() {
        assert_eq!(
            resolve_editor_template(None, None, None, Some("editor")),
            Some("editor".to_string())
        );
    }

    #[test]
    fn nothing_resolved_is_none() {
        assert_eq!(resolve_editor_template(None, None, None, None), None);
    }

    // -----------------------------------------------------------------------
    // build_editor_argv
    // -----------------------------------------------------------------------

    #[test]
    fn vim_gets_plus_line_prefix() {
        let argv = build_editor_argv("vim", Path::new("a.rs"), 42);
        assert_eq!(argv, vec!["vim", "+42", "a.rs"]);
    }

    #[test]
    fn nvim_matches_the_same_default_as_vim() {
        let argv = build_editor_argv("nvim", Path::new("a.rs"), 7);
        assert_eq!(argv, vec!["nvim", "+7", "a.rs"]);
    }

    #[test]
    fn code_uses_goto_syntax() {
        let argv = build_editor_argv("code", Path::new("a.rs"), 42);
        assert_eq!(argv, vec!["code", "--goto", "a.rs:42"]);
    }

    #[test]
    fn subl_uses_colon_syntax() {
        let argv = build_editor_argv("subl", Path::new("a.rs"), 42);
        assert_eq!(argv, vec!["subl", "a.rs:42"]);
    }

    #[test]
    fn zed_uses_colon_syntax() {
        let argv = build_editor_argv("zed", Path::new("a.rs"), 42);
        assert_eq!(argv, vec!["zed", "a.rs:42"]);
    }

    #[test]
    fn emacs_gets_plus_line_prefix() {
        let argv = build_editor_argv("emacs", Path::new("a.rs"), 42);
        assert_eq!(argv, vec!["emacs", "+42", "a.rs"]);
    }

    #[test]
    fn nano_gets_plus_line_prefix() {
        let argv = build_editor_argv("nano", Path::new("a.rs"), 42);
        assert_eq!(argv, vec!["nano", "+42", "a.rs"]);
    }

    #[test]
    fn unknown_editor_gets_file_only() {
        let argv = build_editor_argv("mystery-editor", Path::new("a.rs"), 42);
        assert_eq!(argv, vec!["mystery-editor", "a.rs"]);
    }

    #[test]
    fn extra_flags_still_resolve_the_basename() {
        let argv = build_editor_argv("code --wait", Path::new("a.rs"), 42);
        assert_eq!(argv, vec!["code", "--wait", "--goto", "a.rs:42"]);
    }

    #[test]
    fn explicit_placeholders_are_used_verbatim() {
        let argv = build_editor_argv("code -g {file}:{line}", Path::new("a.rs"), 42);
        assert_eq!(argv, vec!["code", "-g", "a.rs:42"]);
    }

    #[test]
    fn a_path_basename_is_still_recognized() {
        let argv = build_editor_argv("/usr/local/bin/vim", Path::new("a.rs"), 42);
        assert_eq!(argv, vec!["/usr/local/bin/vim", "+42", "a.rs"]);
    }
}
