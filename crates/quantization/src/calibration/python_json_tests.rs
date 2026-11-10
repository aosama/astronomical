use minijinja::Value;

use super::to_python_json;

fn parsed(json_text: &str) -> Value {
    serde_json::from_str(json_text).expect("the test JSON parses")
}

#[test]
fn should_write_python_separators_and_keep_keys_in_file_order() {
    let tool_definition = parsed(
        r#"{"type":"function","function":{"name":"read","parameters":{"required":["path"]}}}"#,
    );

    assert_eq!(
        to_python_json(&tool_definition),
        r#"{"type": "function", "function": {"name": "read", "parameters": {"required": ["path"]}}}"#
    );
}

#[test]
fn should_leave_non_ascii_text_unescaped_and_escape_quotes_and_controls() {
    let value = parsed(r#"{"text": "caf\u00e9 \"quoted\"\n\ttab \u0001"}"#);

    assert_eq!(
        to_python_json(&value),
        "{\"text\": \"caf\u{e9} \\\"quoted\\\"\\n\\ttab \\u0001\"}"
    );
}

#[test]
fn should_write_scalars_as_python_does() {
    assert_eq!(to_python_json(&parsed("null")), "null");
    assert_eq!(to_python_json(&parsed("true")), "true");
    assert_eq!(to_python_json(&parsed("false")), "false");
    assert_eq!(to_python_json(&parsed("42")), "42");
    assert_eq!(to_python_json(&parsed("-7")), "-7");
    assert_eq!(to_python_json(&parsed("1.5")), "1.5");
    assert_eq!(to_python_json(&parsed("2.0")), "2.0");
}

#[test]
fn should_write_empty_containers_without_inner_spaces() {
    assert_eq!(to_python_json(&parsed("{}")), "{}");
    assert_eq!(to_python_json(&parsed("[]")), "[]");
    assert_eq!(to_python_json(&parsed(r#"{"a": []}"#)), r#"{"a": []}"#);
}

#[test]
fn should_write_an_undefined_value_as_null() {
    assert_eq!(to_python_json(&Value::UNDEFINED), "null");
}
