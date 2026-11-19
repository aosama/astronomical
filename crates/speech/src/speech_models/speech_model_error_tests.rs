use astronomical_speech_contract::SpeechFailure;

use super::SpeechModelError;

// Positive: an input the model cannot speak is the caller's to fix.
#[test]
fn should_turn_an_input_error_into_an_invalid_request() {
    let failure = SpeechFailure::from(SpeechModelError::Input(
        "there is no voice \"juliet\"".to_owned(),
    ));

    assert!(
        matches!(&failure, SpeechFailure::InvalidRequest { reason } if reason == "there is no voice \"juliet\""),
        "{failure:?}"
    );
}

// Negative: a broken artifact is the server's failure, not the caller's.
#[test]
fn should_turn_an_artifact_error_into_a_fatal_execution() {
    let failure = SpeechFailure::from(SpeechModelError::Artifact(
        "tensor bert.embeddings: missing".to_owned(),
    ));

    assert!(
        matches!(&failure, SpeechFailure::FatalExecution { reason } if reason.contains("tensor bert.embeddings")),
        "{failure:?}"
    );
}

// Edge: a very long reason is bounded by the contract's constructor.
#[test]
fn should_bound_a_long_input_reason() {
    let long_reason = "O Romeo, Romeo, wherefore art thou Romeo? ".repeat(40);

    let SpeechFailure::InvalidRequest { reason } =
        SpeechFailure::from(SpeechModelError::Input(long_reason.clone()))
    else {
        panic!("an input error must be an invalid request");
    };

    assert!(reason.chars().count() < long_reason.chars().count());
}
