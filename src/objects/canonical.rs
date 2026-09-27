//! Canonical JSON (spec §5.2, D3): the RFC 8785 rules, restricted to what
//! ambits objects contain — integers only, no floats.
//!
//! Hand-written rather than relying on `serde_json`'s output: key order there
//! depends on whether *any* crate in the build enables `preserve_order`, and
//! an object id must not change because a dependency did.

use color_eyre::eyre::{bail, Result};
use serde_json::Value;

/// Serialize `value` canonically: no whitespace, object keys sorted by UTF-16
/// code units, strings escaped as ECMAScript's `JSON.stringify` does.
pub fn to_bytes(value: &Value) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    write(value, &mut out)?;
    Ok(out)
}

/// Parse `bytes`, refusing anything that is not already canonical.
///
/// Readers reject non-canonical input rather than normalizing it, so one
/// object can never have two encodings (§5.2).
pub fn from_bytes(bytes: &[u8]) -> Result<Value> {
    let value: Value = serde_json::from_slice(bytes)?;
    if to_bytes(&value)? != bytes {
        bail!("not canonical JSON");
    }
    Ok(value)
}

fn write(value: &Value, out: &mut Vec<u8>) -> Result<()> {
    match value {
        Value::Null => out.extend_from_slice(b"null"),
        Value::Bool(b) => out.extend_from_slice(if *b { b"true" } else { b"false" }),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.extend_from_slice(i.to_string().as_bytes());
            } else if let Some(u) = n.as_u64() {
                out.extend_from_slice(u.to_string().as_bytes());
            } else {
                bail!("non-integer number {n} in an object");
            }
        }
        Value::String(s) => write_string(s, out),
        Value::Array(items) => {
            out.push(b'[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write(item, out)?;
            }
            out.push(b']');
        }
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by(|a, b| a.0.encode_utf16().cmp(b.0.encode_utf16()));
            out.push(b'{');
            for (i, (key, item)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_string(key, out);
                out.push(b':');
                write(item, out)?;
            }
            out.push(b'}');
        }
    }
    Ok(())
}

fn write_string(s: &str, out: &mut Vec<u8>) {
    out.push(b'"');
    for c in s.chars() {
        match c {
            '"' => out.extend_from_slice(b"\\\""),
            '\\' => out.extend_from_slice(b"\\\\"),
            '\u{08}' => out.extend_from_slice(b"\\b"),
            '\u{0c}' => out.extend_from_slice(b"\\f"),
            '\n' => out.extend_from_slice(b"\\n"),
            '\r' => out.extend_from_slice(b"\\r"),
            '\t' => out.extend_from_slice(b"\\t"),
            c if (c as u32) < 0x20 => out.extend_from_slice(format!("\\u{:04x}", c as u32).as_bytes()),
            c => {
                let mut buf = [0u8; 4];
                out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            }
        }
    }
    out.push(b'"');
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn canon(v: Value) -> String {
        String::from_utf8(to_bytes(&v).unwrap()).unwrap()
    }

    #[test]
    fn keys_sort_and_whitespace_goes() {
        assert_eq!(canon(json!({"b": 1, "a": [true, null], "c": {"z": 0, "y": -2}})), r#"{"a":[true,null],"b":1,"c":{"y":-2,"z":0}}"#);
    }

    /// RFC 8785 orders keys by UTF-16 code units, which differs from UTF-8
    /// byte order above the BMP.
    #[test]
    fn keys_sort_by_utf16_code_units() {
        // U+FF61 is one unit (0xFF61); U+1F600 is a surrogate pair (0xD83D …),
        // so it sorts first in UTF-16 though last in UTF-8.
        assert_eq!(canon(json!({"\u{ff61}": 1, "\u{1f600}": 2})), "{\"\u{1f600}\":2,\"\u{ff61}\":1}");
    }

    #[test]
    fn strings_escape_like_json_stringify() {
        assert_eq!(canon(json!("a\"b\\c\n\u{1}/é")), r#""a\"b\\c\n\u0001/é""#);
    }

    #[test]
    fn floats_are_refused() {
        assert!(to_bytes(&json!({"x": 1.5})).is_err());
    }

    #[test]
    fn readers_reject_non_canonical_input() {
        assert!(from_bytes(br#"{"a":1,"b":2}"#).is_ok());
        assert!(from_bytes(br#"{"b":2,"a":1}"#).is_err(), "unsorted");
        assert!(from_bytes(br#"{"a": 1}"#).is_err(), "whitespace");
        assert!(from_bytes(b"\"\\u0041\"").is_err(), "needless escape");
    }
}
