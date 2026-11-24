//! Unit tests for the order-preserving JSON value.

use super::*;
use crate::test_support::ordered_json;

#[test]
fn should_keep_object_members_in_arrival_order() {
    let parsed_value = ordered_json(r#"{"zebra": 1, "apple": 2, "mango": 3}"#);
    let mut member_keys = Vec::new();
    for object_member in parsed_value.as_object().expect("an object") {
        member_keys.push(object_member.key.as_str());
    }
    assert_eq!(member_keys, ["zebra", "apple", "mango"]);
}

#[test]
fn should_keep_the_first_position_and_last_value_of_a_repeated_key() {
    let parsed_value = ordered_json(r#"{"first": 1, "second": 2, "first": 3}"#);
    let members = parsed_value.as_object().expect("an object");
    assert_eq!(members.len(), 2);
    assert_eq!(members[0].key, "first");
    assert_eq!(members[0].value, OrderedJsonValue::SignedInteger(3));
    assert_eq!(members[1].key, "second");
}

#[test]
fn should_give_each_number_one_representation() {
    assert_eq!(ordered_json("7"), OrderedJsonValue::SignedInteger(7));
    assert_eq!(ordered_json("-7"), OrderedJsonValue::SignedInteger(-7));
    assert_eq!(
        ordered_json("9223372036854775807"),
        OrderedJsonValue::SignedInteger(i64::MAX)
    );
    assert_eq!(
        ordered_json("9223372036854775808"),
        OrderedJsonValue::UnsignedInteger(9_223_372_036_854_775_808)
    );
    assert_eq!(ordered_json("7.0"), OrderedJsonValue::Float(7.0));
}

#[test]
fn should_parse_every_json_kind() {
    let parsed_value = ordered_json(r#"[null, true, "text", [], {}]"#);
    assert_eq!(
        parsed_value,
        OrderedJsonValue::Array(vec![
            OrderedJsonValue::Null,
            OrderedJsonValue::Boolean(true),
            OrderedJsonValue::Text("text".to_owned()),
            OrderedJsonValue::Array(Vec::new()),
            OrderedJsonValue::Object(Vec::new()),
        ])
    );
}

#[test]
fn should_find_a_member_only_in_an_object() {
    let object_value = ordered_json(r#"{"present": true}"#);
    assert_eq!(
        object_value.member("present"),
        Some(&OrderedJsonValue::Boolean(true))
    );
    assert_eq!(object_value.member("absent"), None);
    assert_eq!(ordered_json("[1]").member("present"), None);
}

#[test]
fn should_reject_malformed_json() {
    let parse_result = serde_json::from_str::<OrderedJsonValue>(r#"{"unterminated": "#);
    assert!(parse_result.is_err());
}

#[test]
fn should_name_each_json_type_for_error_messages() {
    assert_eq!(ordered_json("null").type_name(), "null");
    assert_eq!(ordered_json("false").type_name(), "boolean");
    assert_eq!(ordered_json("1.5").type_name(), "number");
    assert_eq!(ordered_json(r#""s""#).type_name(), "string");
    assert_eq!(ordered_json("[]").type_name(), "array");
    assert_eq!(ordered_json("{}").type_name(), "object");
}

#[test]
fn should_write_object_members_back_in_their_kept_order() {
    let json_text =
        r#"{"zebra":1,"apple":{"y":2,"x":[true,null,1.5,"s"]},"big":9223372036854775808}"#;
    let parsed_value = ordered_json(json_text);
    assert_eq!(
        serde_json::to_string(&parsed_value).expect("an ordered value serializes"),
        json_text
    );
}

#[test]
fn should_survive_a_round_trip_through_json() {
    let parsed_value = ordered_json(r#"{"b":[1,-2,0.1,1e16],"a":"é☕","empty":{},"none":[]}"#);
    let json_text = serde_json::to_string(&parsed_value).expect("an ordered value serializes");
    assert_eq!(ordered_json(&json_text), parsed_value);
}
