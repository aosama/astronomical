//! Boundary tests for duplicate-aware JSON parsing: every scalar and
//! container shape, plus the duplicate-key rejection at every nesting depth.

use crate::strict_json::{DUPLICATE_JSON_FIELD_MARKER, DuplicateAwareJsonValue};

fn parse(json_text: &str) -> Result<serde_json::Value, String> {
    serde_json::from_str::<DuplicateAwareJsonValue>(json_text)
        .map(|parsed_value| parsed_value.0)
        .map_err(|parse_error| parse_error.to_string())
}

#[test]
fn should_parse_every_scalar_shape() {
    assert_eq!(parse("true"), Ok(serde_json::Value::Bool(true)));
    assert_eq!(parse("-5"), Ok(serde_json::Value::Number((-5_i64).into())));
    assert_eq!(
        parse("18446744073709551615"),
        Ok(serde_json::Value::Number(u64::MAX.into()))
    );
    assert_eq!(
        parse("1.5"),
        Ok(serde_json::Value::Number(
            serde_json::Number::from_f64(1.5).expect("1.5 is finite")
        ))
    );
    assert_eq!(
        parse("\"safetensors\""),
        Ok(serde_json::Value::String("safetensors".to_owned()))
    );
    assert_eq!(parse("null"), Ok(serde_json::Value::Null));
}

#[test]
fn should_parse_arrays_and_nested_unique_objects() {
    let parsed_array =
        parse("[1,{\"dtype\":\"BF16\"},[true,null]]").expect("unique nesting should parse");
    assert_eq!(
        parsed_array,
        serde_json::json!([1, {"dtype": "BF16"}, [true, null]])
    );

    let parsed_object = parse("{\"t\":{\"shape\":[2,3]},\"__metadata__\":{\"format\":\"pt\"}}")
        .expect("unique keys should parse");
    assert_eq!(
        parsed_object,
        serde_json::json!({"t": {"shape": [2, 3]}, "__metadata__": {"format": "pt"}})
    );
}

#[test]
fn should_reject_a_duplicate_key_at_the_top_level() {
    let parse_error = parse("{\"t\":1,\"t\":2}").expect_err("duplicate keys must fail");
    assert!(
        parse_error.contains(DUPLICATE_JSON_FIELD_MARKER),
        "error must name the duplicate field: {parse_error}"
    );
    assert!(parse_error.contains("'t'"));
}

#[test]
fn should_reject_a_duplicate_key_nested_inside_an_object() {
    let parse_error = parse("{\"outer\":{\"a\":1,\"a\":2}}")
        .expect_err("nested duplicate keys must fail before replacement can hide them");
    assert!(parse_error.contains(DUPLICATE_JSON_FIELD_MARKER));
}

#[test]
fn should_reject_a_duplicate_key_inside_an_array_element() {
    let parse_error = parse("[{\"a\":1,\"a\":2}]").expect_err("duplicates inside arrays must fail");
    assert!(parse_error.contains(DUPLICATE_JSON_FIELD_MARKER));
}

#[test]
fn should_bound_the_field_name_quoted_in_the_duplicate_error() {
    // A 500-character duplicated key must not be echoed whole into the
    // error: untruncated it would exceed 500 characters, truncated to the
    // bounded field name it stays under 400.
    let oversized_field_name = "k".repeat(500);
    let json_with_oversized_duplicate =
        format!("{{\"{oversized_field_name}\":1,\"{oversized_field_name}\":2}}");
    let parse_error = parse(&json_with_oversized_duplicate)
        .expect_err("an oversized duplicate key must still fail");
    assert!(parse_error.contains(DUPLICATE_JSON_FIELD_MARKER));
    assert!(
        parse_error.chars().count() < 400,
        "the quoted field name must be truncated, error was {} characters",
        parse_error.chars().count()
    );
}

#[test]
fn should_fail_closed_on_an_out_of_range_float_literal() {
    // 1e400 overflows f64; whether serde_json rejects the literal or the
    // visitor sees a non-finite value, the parse must fail closed rather
    // than produce an infinite JSON number.
    let parse_error = parse("1e400").expect_err("an overflowing float must not parse");
    assert!(
        !parse_error.is_empty(),
        "the failure must carry a reason: {parse_error}"
    );
}

#[test]
fn should_parse_negative_and_integral_floats() {
    let parsed_negative_float = parse("-2.5").expect("negative floats are valid JSON");
    assert_eq!(parsed_negative_float, serde_json::json!(-2.5));
    let parsed_integral_float = parse("4.0").expect("integral floats are valid JSON");
    assert_eq!(parsed_integral_float, serde_json::json!(4.0));
}
