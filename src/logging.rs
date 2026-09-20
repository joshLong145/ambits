//! One structured (JSON-lines) log file per session: session-log ingest,
//! tool/depth resolution, symbol updates, symbol parsing, and the per-tool-
//! call activity line, all behind the `log` facade.
//!
//! Deliberately file-only, never stderr. The TUI runs in an alternate screen
//! with raw mode enabled, and anything written to stderr outside
//! `terminal.draw()` corrupts the display — and it isn't just a startup
//! concern, since symbol parsing also runs mid-session (the file watcher
//! re-parses a changed file through the same code path). So when no
//! `--log-output` directory is given, [`init`] does not install a logger at
//! all: with no global logger registered, every `log::info!`/`debug!`/`warn!`
//! call is a no-op, which is a stronger guarantee than "level filtered to
//! off" — there is no path by which output could ever reach stderr.
//!
//! Verbosity: `Info` by default (the one-line-per-tool-call activity record),
//! `RUST_LOG=ambits=debug` additionally unlocks the five noisier internal-
//! diagnostics subsystems (each scoped under its own `target: "ambits::..."`).
//! `RUST_LOG` can still override in either direction as usual.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use log::kv::{Error as KvError, Key, Value, VisitSource, VisitValue};

/// Install the logger if `log_output_dir` is `Some`; otherwise do nothing.
///
/// The file is named after `session_id` — the same id this codebase already
/// treats as the root/top-level agent id (it seeds the root of the agent
/// tree) — so every subsystem's output for one `ambits` session lands in one
/// place: `<dir>/<session_id>.log`. Falls back to `<dir>/no-session.log` for
/// the rare case nothing resolves (a project with no Claude Code session
/// history at all — `--dump`/`show`/`callers` still work standalone).
pub fn init(log_output_dir: Option<&Path>, session_id: Option<&str>) {
    let Some(dir) = log_output_dir else {
        return;
    };

    if std::fs::create_dir_all(dir).is_err() {
        return;
    }

    let file_name = format!("{}.log", session_id.unwrap_or("no-session"));
    let Ok(file) = OpenOptions::new().create(true).append(true).open(dir.join(file_name)) else {
        return;
    };

    // Errors here mean a logger is already installed (e.g. called twice in a
    // test) — never fatal, so `try_init` rather than `init`.
    let _ = env_logger::Builder::new()
        .filter_level(log::LevelFilter::Info)
        .parse_default_env() // RUST_LOG, if set, overrides the Info baseline above.
        .format(format_json_line)
        .target(env_logger::Target::Pipe(Box::new(file)))
        .try_init();
}

/// Render one `log::Record` as a single JSON object per line.
///
/// Fixed envelope keys are `ts`, `level`, `target` (the module-path-style log
/// target, e.g. `ambits::ingest`), and `message`; every structured field
/// attached via the `key = value` macro syntax merges in alongside those.
/// Callers are responsible for not naming a field `target`, `level`, `ts`, or
/// `message` themselves — call sites in this codebase use `symbol_target`
/// for a read's symbol/selector, precisely to avoid colliding with the
/// envelope's own `target`.
fn format_json_line(buf: &mut env_logger::fmt::Formatter, record: &log::Record) -> std::io::Result<()> {
    let mut map = serde_json::Map::new();
    map.insert("ts".to_string(), serde_json::Value::String(now_rfc3339()));
    map.insert("level".to_string(), serde_json::Value::String(record.level().to_string()));
    map.insert("target".to_string(), serde_json::Value::String(record.target().to_string()));
    map.insert("message".to_string(), serde_json::Value::String(record.args().to_string()));

    let mut visitor = JsonMapVisitor { map: &mut map };
    let _ = record.key_values().visit(&mut visitor);

    writeln!(buf, "{}", serde_json::Value::Object(map))
}

/// Wall-clock time as RFC 3339 (UTC), with no extra time-crate dependency —
/// `SystemTime` plus a small manual calendar conversion.
fn now_rfc3339() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let (days, secs_of_day) = (secs / 86_400, secs % 86_400);
    let (hour, minute, second) = (secs_of_day / 3600, (secs_of_day / 60) % 60, secs_of_day % 60);

    // Civil-from-days (Howard Hinnant's algorithm), proleptic Gregorian.
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };

    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

struct JsonMapVisitor<'a> {
    map: &'a mut serde_json::Map<String, serde_json::Value>,
}

impl<'a, 'kvs> VisitSource<'kvs> for JsonMapVisitor<'a> {
    fn visit_pair(&mut self, key: Key<'kvs>, value: Value<'kvs>) -> Result<(), KvError> {
        let mut collector = JsonValueVisitor(None);
        value.visit(&mut collector)?;
        self.map.insert(key.to_string(), collector.0.unwrap_or(serde_json::Value::Null));
        Ok(())
    }
}

/// Collects one `log::kv::Value` into its natural JSON type — numbers and
/// bools as real JSON types, everything else (strings, and anything captured
/// via the macro's `:?` Debug sigil) as a JSON string.
struct JsonValueVisitor(Option<serde_json::Value>);

impl<'v> VisitValue<'v> for JsonValueVisitor {
    fn visit_any(&mut self, value: Value) -> Result<(), KvError> {
        self.0 = Some(serde_json::Value::String(format!("{value:?}")));
        Ok(())
    }

    fn visit_u64(&mut self, value: u64) -> Result<(), KvError> {
        self.0 = Some(serde_json::Value::from(value));
        Ok(())
    }

    fn visit_i64(&mut self, value: i64) -> Result<(), KvError> {
        self.0 = Some(serde_json::Value::from(value));
        Ok(())
    }

    fn visit_f64(&mut self, value: f64) -> Result<(), KvError> {
        self.0 = Some(serde_json::json!(value));
        Ok(())
    }

    fn visit_bool(&mut self, value: bool) -> Result<(), KvError> {
        self.0 = Some(serde_json::Value::Bool(value));
        Ok(())
    }

    fn visit_str(&mut self, value: &str) -> Result<(), KvError> {
        self.0 = Some(serde_json::Value::String(value.to_string()));
        Ok(())
    }

    fn visit_null(&mut self) -> Result<(), KvError> {
        self.0 = Some(serde_json::Value::Null);
        Ok(())
    }
}
