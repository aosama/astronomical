//! JSON text written exactly the way the reference decision models write it.
//!
//! Every reference implementation turns a structured state or criterion into
//! model input with Python's `json.dumps(value, ensure_ascii=False)`. The
//! model then reads that text token by token, so a different separator, key
//! order, escape or float spelling is a different prompt. `serde_json` writes
//! compact separators and its own float text, so this module reproduces the
//! Python output instead: separators, escapes, and `float.__repr__`.

use crate::ordered_json::{OrderedJsonMember, OrderedJsonValue};

/// Whether an object's members are written in arrival order or sorted by key.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JsonKeyOrder {
    /// The order the request gave them, as `json.dumps` does by default.
    Insertion,
    /// Sorted by Unicode code point, as `json.dumps(sort_keys=True)` does.
    Sorted,
}

/// The separators and key order one model family writes JSON with.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct JsonTextStyle {
    pub item_separator: &'static str,
    pub key_separator: &'static str,
    pub key_order: JsonKeyOrder,
}

impl JsonTextStyle {
    /// `json.dumps(value, ensure_ascii=False)`: `", "` and `": "`, arrival order.
    pub const PYTHON_DEFAULT: Self = Self {
        item_separator: ", ",
        key_separator: ": ",
        key_order: JsonKeyOrder::Insertion,
    };

    /// `json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)`.
    pub const COMPACT_SORTED_KEYS: Self = Self {
        item_separator: ",",
        key_separator: ":",
        key_order: JsonKeyOrder::Sorted,
    };
}

/// Writes `value` as the JSON text Python's `json.dumps` produces in `style`.
#[must_use]
pub fn python_json_text(value: &OrderedJsonValue, style: JsonTextStyle) -> String {
    let mut json_text = String::new();
    append_value(&mut json_text, value, style);
    json_text
}

fn append_value(json_text: &mut String, value: &OrderedJsonValue, style: JsonTextStyle) {
    match value {
        OrderedJsonValue::Null => json_text.push_str("null"),
        OrderedJsonValue::Boolean(true) => json_text.push_str("true"),
        OrderedJsonValue::Boolean(false) => json_text.push_str("false"),
        OrderedJsonValue::SignedInteger(integer) => json_text.push_str(&integer.to_string()),
        OrderedJsonValue::UnsignedInteger(integer) => json_text.push_str(&integer.to_string()),
        OrderedJsonValue::Float(float) => json_text.push_str(&python_float_repr(*float)),
        OrderedJsonValue::Text(text) => append_quoted_text(json_text, text),
        OrderedJsonValue::Array(elements) => append_array(json_text, elements, style),
        OrderedJsonValue::Object(members) => append_object(json_text, members, style),
    }
}

fn append_array(json_text: &mut String, elements: &[OrderedJsonValue], style: JsonTextStyle) {
    json_text.push('[');
    for (element_index, element) in elements.iter().enumerate() {
        if element_index > 0 {
            json_text.push_str(style.item_separator);
        }
        append_value(json_text, element, style);
    }
    json_text.push(']');
}

fn append_object(json_text: &mut String, members: &[OrderedJsonMember], style: JsonTextStyle) {
    let mut ordered_members: Vec<&OrderedJsonMember> = members.iter().collect();
    if style.key_order == JsonKeyOrder::Sorted {
        // Rust compares strings by UTF-8 bytes, which orders them by code
        // point exactly as Python compares `str` keys.
        ordered_members.sort_by(compare_member_keys);
    }
    json_text.push('{');
    for (member_index, object_member) in ordered_members.iter().enumerate() {
        if member_index > 0 {
            json_text.push_str(style.item_separator);
        }
        append_quoted_text(json_text, &object_member.key);
        json_text.push_str(style.key_separator);
        append_value(json_text, &object_member.value, style);
    }
    json_text.push('}');
}

fn compare_member_keys(
    left_member: &&OrderedJsonMember,
    right_member: &&OrderedJsonMember,
) -> std::cmp::Ordering {
    left_member.key.cmp(&right_member.key)
}

/// Quotes `text` with the escapes `json.dumps(ensure_ascii=False)` uses.
///
/// Only the quote, the backslash and the control characters below U+0020 are
/// escaped; every other character, including non-ASCII, is written as itself.
fn append_quoted_text(json_text: &mut String, text: &str) {
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
            control_character if u32::from(control_character) < 0x20 => {
                json_text.push_str(&format!("\\u{:04x}", u32::from(control_character)));
            }
            ordinary_character => json_text.push(ordinary_character),
        }
    }
    json_text.push('"');
}

/// The lowest decimal exponent `float.__repr__` still writes without an exponent.
const LOWEST_FIXED_NOTATION_EXPONENT: i32 = -4;
/// The first decimal exponent `float.__repr__` writes in scientific notation.
const FIRST_SCIENTIFIC_NOTATION_EXPONENT: i32 = 16;

/// Writes `float` exactly as Python's `float.__repr__` does.
///
/// Both languages print the shortest digits that round-trip, so the digits
/// agree; only the layout differs. Python uses fixed notation for a decimal
/// exponent from -4 up to 15 and always keeps a fractional part (`1250.0`),
/// and otherwise writes a signed exponent of at least two digits (`1e+16`,
/// `1e-05`).
#[must_use]
pub fn python_float_repr(float: f64) -> String {
    if float.is_nan() {
        return "NaN".to_owned();
    }
    if float.is_infinite() {
        return if float > 0.0 { "Infinity" } else { "-Infinity" }.to_owned();
    }
    let sign_text = if float.is_sign_negative() { "-" } else { "" };
    if float == 0.0 {
        return format!("{sign_text}0.0");
    }
    let (significant_digits, decimal_exponent) = shortest_digits_and_exponent(float.abs());
    let magnitude_text = if (LOWEST_FIXED_NOTATION_EXPONENT..FIRST_SCIENTIFIC_NOTATION_EXPONENT)
        .contains(&decimal_exponent)
    {
        fixed_notation(&significant_digits, decimal_exponent)
    } else {
        scientific_notation(&significant_digits, decimal_exponent)
    };
    format!("{sign_text}{magnitude_text}")
}

/// The shortest round-trip digits of a positive finite float and its decimal exponent.
///
/// Rust's `{:e}` already prints the shortest digits as `d.ddde<exponent>`, so
/// this only takes that text apart.
fn shortest_digits_and_exponent(positive_float: f64) -> (String, i32) {
    let exponent_form = format!("{positive_float:e}");
    let (mantissa_text, exponent_text) = exponent_form
        .split_once('e')
        .unwrap_or((exponent_form.as_str(), "0"));
    let significant_digits = mantissa_text.replace('.', "");
    let decimal_exponent = exponent_text.parse::<i32>().unwrap_or(0);
    (significant_digits, decimal_exponent)
}

fn fixed_notation(significant_digits: &str, decimal_exponent: i32) -> String {
    if decimal_exponent < 0 {
        let leading_zero_count = usize::try_from(-decimal_exponent - 1).unwrap_or(0);
        return format!("0.{}{significant_digits}", "0".repeat(leading_zero_count));
    }
    let integer_digit_count = usize::try_from(decimal_exponent).unwrap_or(0) + 1;
    if significant_digits.len() <= integer_digit_count {
        let trailing_zero_count = integer_digit_count - significant_digits.len();
        return format!("{significant_digits}{}.0", "0".repeat(trailing_zero_count));
    }
    let (integer_digits, fraction_digits) = significant_digits.split_at(integer_digit_count);
    format!("{integer_digits}.{fraction_digits}")
}

fn scientific_notation(significant_digits: &str, decimal_exponent: i32) -> String {
    let (leading_digit, remaining_digits) = significant_digits.split_at(1);
    let mantissa_text = if remaining_digits.is_empty() {
        leading_digit.to_owned()
    } else {
        format!("{leading_digit}.{remaining_digits}")
    };
    let exponent_sign = if decimal_exponent < 0 { '-' } else { '+' };
    format!(
        "{mantissa_text}e{exponent_sign}{:02}",
        decimal_exponent.unsigned_abs()
    )
}

#[cfg(test)]
#[path = "python_json_text_tests.rs"]
mod tests;
