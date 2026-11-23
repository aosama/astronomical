use super::error_outcome;
use crate::{InferenceEngineError, PerformanceAttributionOutcome};

const REJECTION: &str = "the request was refused";
const FAILURE: &str = "the engine failed";

// Positive: a refused request is recorded as a rejection, with its reason.
#[test]
fn should_record_an_invalid_request_as_a_rejection() {
    let (outcome, description) = error_outcome(
        &InferenceEngineError::InvalidRequest {
            reason: "too long".to_owned(),
        },
        REJECTION,
        FAILURE,
    );

    assert!(matches!(outcome, PerformanceAttributionOutcome::Rejected));
    assert_eq!(description, REJECTION);
}

// Edge: a busy engine turned the request away, so it is a rejection too.
#[test]
fn should_record_a_busy_engine_as_a_rejection() {
    let (outcome, description) =
        error_outcome(&InferenceEngineError::EngineBusy, REJECTION, FAILURE);

    assert!(matches!(outcome, PerformanceAttributionOutcome::Rejected));
    assert_eq!(description, REJECTION);
}

// Negative: an engine failure is never recorded as a rejection.
#[test]
fn should_record_a_fatal_error_as_a_failure() {
    let (outcome, description) = error_outcome(
        &InferenceEngineError::Fatal {
            reason: "lost the model".to_owned(),
        },
        REJECTION,
        FAILURE,
    );

    assert!(matches!(outcome, PerformanceAttributionOutcome::Failed));
    assert_eq!(description, FAILURE);
}
