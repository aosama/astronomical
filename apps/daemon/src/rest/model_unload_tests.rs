use axum::http::StatusCode;

use super::unload_response;
use crate::ModelUnloadOutcome;

#[test]
fn should_answer_each_outcome_with_its_status_code() {
    for (outcome, expected_status) in [
        (
            ModelUnloadOutcome::Unloaded {
                model_id: "Ornith-1.5-9B".to_owned(),
            },
            StatusCode::OK,
        ),
        (ModelUnloadOutcome::NothingLoaded, StatusCode::OK),
        (ModelUnloadOutcome::Busy, StatusCode::CONFLICT),
        (
            ModelUnloadOutcome::Unavailable,
            StatusCode::SERVICE_UNAVAILABLE,
        ),
        (
            ModelUnloadOutcome::UnloadedWithoutReadyWorker {
                model_id: "Ornith-1.5-9B".to_owned(),
            },
            StatusCode::SERVICE_UNAVAILABLE,
        ),
    ] {
        assert_eq!(
            unload_response(outcome.clone()).status(),
            expected_status,
            "{outcome:?}"
        );
    }
}
