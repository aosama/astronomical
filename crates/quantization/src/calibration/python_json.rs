//! JSON text exactly as Python's `json.dumps(value, ensure_ascii=False)`
//! writes it.
//!
//! A chat template writes tool definitions into the prompt through its
//! `tojson` filter, and the Python reference backs that filter with
//! `json.dumps`: `", "` and `": "` separators, keys in insertion order,
//! non-ASCII characters left as they are. A compact or key-sorted encoder
//! produces a different prompt, different tokens, and a calibration set that
//! no longer matches the reference's.

use minijinja::value::{Value, ValueKind};

#[cfg(test)]
#[path = "python_json_tests.rs"]
mod python_json_tests;

/// The value as Python-style JSON text.
#[must_use]
pub fn to_python_json(value: &Value) -> String {
    let mut json_text = String::new();
    write_value(value, &mut json_text);
    json_text
}

fn write_value(value: &Value, json_text: &mut String) {
    match value.kind() {
        ValueKind::Undefined | ValueKind::None => json_text.push_str("null"),
        ValueKind::Bool => json_text.push_str(if value.is_true() { "true" } else { "false" }),
        ValueKind::Number => json_text.push_str(&value.to_string()),
        ValueKind::Map => write_map(value, json_text),
        ValueKind::Seq | ValueKind::Iterable => write_sequence(value, json_text),
        _ => match value.as_str() {
            Some(text) => write_string(text, json_text),
            None => write_string(&value.to_string(), json_text),
        },
    }
}

fn write_map(map_value: &Value, json_text: &mut String) {
    json_text.push('{');
    if let Ok(keys) = map_value.try_iter() {
        for (key_position, key) in keys.enumerate() {
            if key_position > 0 {
                json_text.push_str(", ");
            }
            match key.as_str() {
                Some(key_text) => write_string(key_text, json_text),
                None => write_string(&key.to_string(), json_text),
            }
            json_text.push_str(": ");
            let entry_value = map_value.get_item(&key).unwrap_or(Value::UNDEFINED);
            write_value(&entry_value, json_text);
        }
    }
    json_text.push('}');
}

fn write_sequence(sequence_value: &Value, json_text: &mut String) {
    json_text.push('[');
    if let Ok(elements) = sequence_value.try_iter() {
        for (element_position, element) in elements.enumerate() {
            if element_position > 0 {
                json_text.push_str(", ");
            }
            write_value(&element, json_text);
        }
    }
    json_text.push(']');
}

/// Python escapes the quote, the backslash and control characters, and with
/// `ensure_ascii=False` nothing else.
fn write_string(text: &str, json_text: &mut String) {
    json_text.push('"');
    for character in text.chars() {
        match character {
            '"' => json_text.push_str("\\\""),
            '\\' => json_text.push_str("\\\\"),
            '\n' => json_text.push_str("\\n"),
            '\r' => json_text.push_str("\\r"),
            '\t' => json_text.push_str("\\t"),
            '\u{08}' => json_text.push_str("\\b"),
            '\u{0c}' => json_text.push_str("\\f"),
            control_character if (control_character as u32) < 0x20 => {
                json_text.push_str(&format!("\\u{:04x}", control_character as u32));
            }
            plain_character => json_text.push(plain_character),
        }
    }
    json_text.push('"');
}
