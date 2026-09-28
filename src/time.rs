//! Wall-clock time without a date crate: seconds since the epoch, RFC 3339
//! UTC strings, and day counts. One copy of the calendar arithmetic for the
//! logs, the reflog, notes and linkage.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub const SECS_PER_DAY: u64 = 24 * 60 * 60;

/// `n` days, or `None` if that overflows — for CLI flags given in days.
pub fn days(n: u64) -> Option<Duration> {
    n.checked_mul(SECS_PER_DAY).map(Duration::from_secs)
}

/// Seconds since the Unix epoch, or 0 if the clock reads before it.
pub fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// Now, as RFC 3339 UTC to the second.
pub fn now_rfc3339() -> String {
    rfc3339(now_secs())
}

/// Seconds since the epoch of an RFC 3339 UTC time as Claude Code writes
/// them (`2026-09-27T13:39:43.103Z`; fractions ignored). `None` for
/// anything else, including a non-`Z` offset.
pub fn parse_rfc3339(s: &str) -> Option<u64> {
    let b = s.as_bytes();
    if b.len() < 20 || b[4] != b'-' || b[7] != b'-' || b[10] != b'T' || b[13] != b':' || b[16] != b':' || !s.ends_with('Z') {
        return None;
    }
    let num = |r: std::ops::Range<usize>| s.get(r)?.parse::<i64>().ok();
    let (y, mo, d, h, mi, se) = (num(0..4)?, num(5..7)?, num(8..10)?, num(11..13)?, num(14..16)?, num(17..19)?);
    if !(1..=12).contains(&mo) || !(1..=31).contains(&d) || h > 23 || mi > 59 || se > 60 {
        return None;
    }
    // Howard Hinnant's days-from-civil.
    let y = if mo <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (mo + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    u64::try_from(days * 86_400 + h * 3600 + mi * 60 + se).ok()
}

/// Milliseconds since the epoch of an RFC 3339 UTC time, keeping the
/// fraction [`parse_rfc3339`] drops: spans are often shorter than a second.
pub fn parse_rfc3339_millis(s: &str) -> Option<u64> {
    let (whole, fraction) = match s.split_once('.') {
        Some((whole, rest)) => (format!("{whole}Z"), rest.strip_suffix('Z')?),
        None => (s.to_string(), ""),
    };
    let millis: u64 = format!("{fraction:0<3}").get(..3)?.parse().ok()?;
    Some(parse_rfc3339(&whole)? * 1000 + millis)
}

/// `secs` since the epoch as `YYYY-MM-DDTHH:MM:SSZ` (Howard Hinnant's
/// civil-from-days, so no date dependency).
pub fn rfc3339(secs: u64) -> String {
    let days = (secs / SECS_PER_DAY) as i64;
    let rem = secs % SECS_PER_DAY;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z", rem / 3600, rem % 3600 / 60, rem % 60)
}

/// `MM-DD HH:MM`, UTC, for milliseconds since the epoch: a compact
/// when-was-it for lists.
pub fn day_minute(ms: u64) -> String {
    let t = rfc3339(ms / 1000);
    format!("{} {}", &t[5..10], &t[11..16])
}

/// `HH:MM:SS.d`, UTC, for milliseconds since the epoch: a time of day to
/// the tenth of a second.
pub fn clock(ms: u64) -> String {
    let t = rfc3339(ms / 1000);
    format!("{}.{}", &t[11..19], ms % 1000 / 100)
}

/// A recorded RFC 3339 time (`2026-09-27T10:05:33.103Z`) as
/// `2026-09-27 10:05Z`; anything else as it is.
pub fn short(t: &str) -> String {
    match t.get(..16) {
        Some(minute) if t.len() > 16 => format!("{}Z", minute.replace('T', " ")),
        _ => t.to_string(),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn display_forms() {
        let ms = super::parse_rfc3339_millis("2026-09-27T10:05:33.450Z").unwrap();
        assert_eq!(super::day_minute(ms), "09-27 10:05");
        assert_eq!(super::clock(ms), "10:05:33.4");
        assert_eq!(super::short("2026-09-27T10:05:33.103Z"), "2026-09-27 10:05Z");
        assert_eq!(super::short("yesterday"), "yesterday");
    }

    use super::*;

    #[test]
    fn days_convert_and_refuse_overflow() {
        assert_eq!(days(2), Some(Duration::from_secs(2 * SECS_PER_DAY)));
        assert_eq!(days(u64::MAX), None);
    }

    #[test]
    fn rfc3339_parses_what_it_formats_and_what_claude_code_writes() {
        for secs in [0, 951_782_400, 1_790_000_000] {
            assert_eq!(parse_rfc3339(&rfc3339(secs)), Some(secs));
        }
        assert_eq!(parse_rfc3339("2026-09-27T13:39:43.103Z"), parse_rfc3339("2026-09-27T13:39:43Z"));
        assert_eq!(parse_rfc3339("2026-09-27T13:39:43+02:00"), None);
        assert_eq!(parse_rfc3339("yesterday"), None);
    }

    #[test]
    fn millisecond_precision_is_kept() {
        let base = parse_rfc3339("2026-09-27T13:39:43Z").unwrap() * 1000;
        assert_eq!(parse_rfc3339_millis("2026-09-27T13:39:43.103Z"), Some(base + 103));
        assert_eq!(parse_rfc3339_millis("2026-09-27T13:39:43.1Z"), Some(base + 100));
        assert_eq!(parse_rfc3339_millis("2026-09-27T13:39:43Z"), Some(base));
        assert_eq!(parse_rfc3339_millis("nonsense"), None);
    }

    #[test]
    fn rfc3339_formats_known_instants() {
        assert_eq!(rfc3339(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(rfc3339(1_790_000_000), "2026-09-21T14:13:20Z");
    }
}
