//! Internal diagnostics: session ingest, tool/depth resolution, symbol
//! updates, and symbol parsing, all behind the `log` facade at debug level.
//!
//! Deliberately file-only, never stderr. The TUI runs in an alternate screen
//! with raw mode enabled, and anything written to stderr outside
//! `terminal.draw()` corrupts the display — and it isn't just a startup
//! concern, since symbol parsing also runs mid-session (the file watcher
//! re-parses a changed file through the same code path). So when no
//! `--log-output` directory is given, [`init`] does not install a logger at
//! all: with no global logger registered, every `log::debug!`/`log::warn!`
//! call is a no-op, which is a stronger guarantee than "level filtered to
//! off" — there is no path by which output could ever reach stderr.

use std::fs::OpenOptions;
use std::path::Path;

/// File name under `--log-output <dir>` for the debug log. Process-scoped
/// rather than session-scoped: several subcommands (`--coverage`, `rg`,
/// `show`, ...) haven't resolved a session id at the point logging must
/// initialize, and several don't ingest a session at all.
const DEBUG_LOG_NAME: &str = "ambits.debug.log";

/// Install the logger if `log_output_dir` is `Some`; otherwise do nothing.
///
/// `RUST_LOG` (e.g. `RUST_LOG=ambits=debug`) controls the level threshold
/// once logging is active; with no `--log-output`, `RUST_LOG` has no effect
/// since no logger is ever registered.
pub fn init(log_output_dir: Option<&Path>) {
    let Some(dir) = log_output_dir else {
        return;
    };

    if std::fs::create_dir_all(dir).is_err() {
        return;
    }

    let Ok(file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join(DEBUG_LOG_NAME))
    else {
        return;
    };

    // Errors here mean a logger is already installed (e.g. called twice in a
    // test) — never fatal, so `try_init` rather than `init`.
    let _ = env_logger::Builder::from_default_env()
        .target(env_logger::Target::Pipe(Box::new(file)))
        .try_init();
}
