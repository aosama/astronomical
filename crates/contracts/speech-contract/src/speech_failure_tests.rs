use super::SpeechFailure;

#[test]
fn should_name_whose_problem_a_failure_is() {
    assert!(
        SpeechFailure::invalid_request("no voice")
            .to_string()
            .starts_with("invalid speech request")
    );
    assert!(
        SpeechFailure::fatal_execution("out of memory")
            .to_string()
            .starts_with("speech failed")
    );
}

// Boundary: a reason past the limit is cut and marked.
#[test]
fn should_cut_an_overlong_reason() {
    let SpeechFailure::FatalExecution { reason } = SpeechFailure::fatal_execution("x".repeat(600))
    else {
        panic!("a fatal failure");
    };
    assert_eq!(reason.chars().count(), 513);
    assert!(reason.ends_with('…'));
}

// Positive: failures cross process boundaries as snake_case JSON.
#[test]
fn should_round_trip_every_failure_through_json() {
    for failure in [
        SpeechFailure::invalid_request("there is no voice \"alloy\""),
        SpeechFailure::fatal_execution("the model failed"),
        SpeechFailure::EngineBusy,
        SpeechFailure::Cancelled,
    ] {
        let failure_json = serde_json::to_string(&failure).expect("a failure serialises");
        let read_back: SpeechFailure = serde_json::from_str(&failure_json).expect("and reads back");
        assert_eq!(read_back, failure);
    }
    assert_eq!(
        serde_json::to_string(&SpeechFailure::EngineBusy).expect("a failure serialises"),
        "\"engine_busy\""
    );
}
