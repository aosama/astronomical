//! Unit tests for Python-compatible JSON text and float spelling.

use super::*;
use crate::test_support::ordered_json;

#[test]
fn should_write_python_default_separators_in_arrival_order() {
    let value = ordered_json(r#"{"b":1,"a":[true,null]}"#);
    assert_eq!(
        python_json_text(&value, JsonTextStyle::PYTHON_DEFAULT),
        r#"{"b": 1, "a": [true, null]}"#
    );
}

#[test]
fn should_write_compact_text_with_sorted_keys() {
    let value = ordered_json(r#"{"b":1,"a":{"z":2,"y":3}}"#);
    assert_eq!(
        python_json_text(&value, JsonTextStyle::COMPACT_SORTED_KEYS),
        r#"{"a":{"y":3,"z":2},"b":1}"#
    );
}

#[test]
fn should_write_empty_containers_without_inner_space() {
    assert_eq!(
        python_json_text(&ordered_json("[]"), JsonTextStyle::PYTHON_DEFAULT),
        "[]"
    );
    assert_eq!(
        python_json_text(&ordered_json("{}"), JsonTextStyle::PYTHON_DEFAULT),
        "{}"
    );
}

#[test]
fn should_escape_only_quotes_backslashes_and_control_characters() {
    let value = OrderedJsonValue::Text("a\"b\\c\nd\te\u{01}f\u{7f}é☕".to_owned());
    assert_eq!(
        python_json_text(&value, JsonTextStyle::PYTHON_DEFAULT),
        "\"a\\\"b\\\\c\\nd\\te\\u0001f\u{7f}é☕\""
    );
}

#[test]
fn should_spell_floats_in_fixed_notation_inside_the_fixed_range() {
    assert_eq!(python_float_repr(1250.0), "1250.0");
    assert_eq!(python_float_repr(0.1), "0.1");
    assert_eq!(python_float_repr(123_456_789.125), "123456789.125");
    assert_eq!(python_float_repr(-2.5), "-2.5");
}

#[test]
fn should_switch_to_scientific_notation_at_both_range_boundaries() {
    // 1e15 is the last fixed-notation power of ten and 1e16 the first scientific one.
    assert_eq!(python_float_repr(1e15), "1000000000000000.0");
    assert_eq!(python_float_repr(1e16), "1e+16");
    // 1e-4 is the last fixed-notation power of ten and 1e-5 the first scientific one.
    assert_eq!(python_float_repr(0.0001), "0.0001");
    assert_eq!(python_float_repr(0.00001), "1e-05");
    assert_eq!(python_float_repr(1.5e-7), "1.5e-07");
}

#[test]
fn should_spell_the_extreme_finite_floats() {
    assert_eq!(python_float_repr(f64::MAX), "1.7976931348623157e+308");
    assert_eq!(python_float_repr(5e-324), "5e-324");
}

#[test]
fn should_spell_zero_with_its_sign() {
    assert_eq!(python_float_repr(0.0), "0.0");
    assert_eq!(python_float_repr(-0.0), "-0.0");
}

#[test]
fn should_spell_non_finite_floats_as_python_does() {
    assert_eq!(python_float_repr(f64::NAN), "NaN");
    assert_eq!(python_float_repr(f64::INFINITY), "Infinity");
    assert_eq!(python_float_repr(f64::NEG_INFINITY), "-Infinity");
}

#[test]
fn should_write_integers_beyond_the_signed_range() {
    assert_eq!(
        python_json_text(
            &ordered_json("18446744073709551615"),
            JsonTextStyle::PYTHON_DEFAULT
        ),
        "18446744073709551615"
    );
}

#[test]
fn should_sort_non_ascii_keys_by_code_point() {
    let value = ordered_json(r#"{"é":1,"z":2,"Z":3,"☕":4}"#);
    assert_eq!(
        python_json_text(&value, JsonTextStyle::COMPACT_SORTED_KEYS),
        r#"{"Z":3,"z":2,"é":1,"☕":4}"#
    );
}

#[test]
fn should_write_negative_integers() {
    assert_eq!(
        python_json_text(
            &ordered_json("[-1,-9223372036854775808]"),
            JsonTextStyle::PYTHON_DEFAULT
        ),
        "[-1, -9223372036854775808]"
    );
}

/// Seventeen-digit decimals are where a fast float parser can land one unit
/// in the last place away from the nearest double. Python's `json.loads` is
/// correctly rounded, so the text a model reads must survive a round trip
/// exactly, or the request is tokenized differently from the reference.
#[test]
fn should_write_seventeen_digit_floats_back_exactly_as_python_reads_them() {
    for float_text in [
        "0.10058280825614929",
        "1.7601518630981445",
        "1.0000158548355103",
        "2.2250738585072014e-308",
        "0.30000000000000004",
    ] {
        let value = ordered_json(&format!("[{float_text}]"));

        assert_eq!(
            python_json_text(&value, JsonTextStyle::PYTHON_DEFAULT),
            format!("[{float_text}]"),
            "{float_text} must be written back as Python writes it"
        );
    }
}
