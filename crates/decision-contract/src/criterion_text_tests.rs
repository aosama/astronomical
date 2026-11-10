//! Unit tests for criterion text and the missing-description rule.

use super::*;
use crate::test_support::ordered_json;

#[test]
fn should_use_a_text_criterion_as_written() {
    assert_eq!(
        criterion_text(&ordered_json(r#""as written""#)),
        "as written"
    );
}

#[test]
fn should_render_a_structured_criterion_as_single_line_json() {
    assert_eq!(
        criterion_text(&ordered_json(r#"{"desc":"calm","weight":0.5}"#)),
        r#"{"desc": "calm", "weight": 0.5}"#
    );
    assert_eq!(criterion_text(&ordered_json("[1,2]")), "[1, 2]");
    assert_eq!(criterion_text(&ordered_json("7")), "7");
}

#[test]
fn should_treat_absent_null_and_empty_text_as_missing() {
    assert!(is_missing_description(None));
    assert!(is_missing_description(Some(&ordered_json("null"))));
    assert!(is_missing_description(Some(&ordered_json(r#""""#))));
}

#[test]
fn should_treat_zero_false_and_whitespace_as_real_descriptions() {
    assert!(!is_missing_description(Some(&ordered_json("0"))));
    assert!(!is_missing_description(Some(&ordered_json("false"))));
    assert!(!is_missing_description(Some(&ordered_json(r#"" ""#))));
}
