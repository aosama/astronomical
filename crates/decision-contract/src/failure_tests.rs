//! Unit tests for decision failures.

use super::*;

#[test]
fn should_keep_a_reason_of_exactly_the_kept_length() {
    let reason = "r".repeat(MAXIMUM_FAILURE_REASON_CHARACTERS);
    assert_eq!(
        DecisionFailure::invalid_request(&reason),
        DecisionFailure::InvalidRequest { reason }
    );
}

#[test]
fn should_cut_a_reason_one_character_over_the_kept_length() {
    let reason = "r".repeat(MAXIMUM_FAILURE_REASON_CHARACTERS + 1);
    let DecisionFailure::FatalExecution {
        reason: kept_reason,
    } = DecisionFailure::fatal_execution(&reason)
    else {
        panic!("a fatal execution failure");
    };
    assert_eq!(
        kept_reason.chars().count(),
        MAXIMUM_FAILURE_REASON_CHARACTERS
    );
}

#[test]
fn should_cut_on_characters_so_multibyte_text_is_never_split() {
    let reason = "é".repeat(MAXIMUM_FAILURE_REASON_CHARACTERS + 10);
    let DecisionFailure::InvalidRequest {
        reason: kept_reason,
    } = DecisionFailure::invalid_request(&reason)
    else {
        panic!("an invalid request failure");
    };
    assert_eq!(
        kept_reason.chars().count(),
        MAXIMUM_FAILURE_REASON_CHARACTERS
    );
}

#[test]
fn should_display_its_reason() {
    assert_eq!(
        DecisionFailure::invalid_request("use fewer options").to_string(),
        "use fewer options"
    );
}

#[test]
fn should_survive_a_round_trip_through_json() {
    for failure in [
        DecisionFailure::invalid_request("use fewer options"),
        DecisionFailure::fatal_execution("the weights are unreadable"),
    ] {
        let json_text = serde_json::to_string(&failure).expect("a failure serializes");
        let restored: DecisionFailure = serde_json::from_str(&json_text).expect("it parses back");
        assert_eq!(restored, failure);
    }
}
