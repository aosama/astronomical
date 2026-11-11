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
