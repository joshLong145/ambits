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

#[cfg(test)]
mod tests {
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
    fn rfc3339_formats_known_instants() {
        assert_eq!(rfc3339(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(rfc3339(1_790_000_000), "2026-09-21T14:13:20Z");
    }
}
