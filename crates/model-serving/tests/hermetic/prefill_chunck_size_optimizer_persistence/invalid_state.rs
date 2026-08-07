use std::fs;

use super::PrefillChunckSizeOptimizer;
use super::support::*;

#[test]
fn should_return_none_when_optimizer_state_file_does_not_exist() {
    let temporary_directory = temporary_directory();
    let load_outcome = PrefillChunckSizeOptimizer::load_from_path(
        temporary_directory.path().join("missing.json"),
        "test-model".to_owned(),
        "rev-1".to_owned(),
        DEFAULT_CANDIDATES.to_vec(),
        SLIDING_WINDOW_OBSERVATION_COUNT,
    )
    .expect("missing state should not fail");
    assert!(load_outcome.is_none());
}

#[test]
fn should_return_none_for_corrupt_empty_or_truncated_state() {
    for serialized_state in ["", "not json", "{\"format_version\":4"] {
        let temporary_directory = temporary_directory();
        let state_file_path = temporary_directory.path().join(OPTIMIZER_STATE_FILE_NAME);
        fs::write(&state_file_path, serialized_state).expect("state fixture should be written");
        load_expect_none(
            &state_file_path,
            "test-model",
            "rev-1",
            &DEFAULT_CANDIDATES,
            SLIDING_WINDOW_OBSERVATION_COUNT,
        );
    }
}

#[test]
fn should_return_none_for_previous_format_without_migration() {
    let temporary_directory = temporary_directory();
    let state_file_path = temporary_directory.path().join(OPTIMIZER_STATE_FILE_NAME);
    let previous_state = r#"{"format_version":3,"model_id":"test-model","model_revision":"rev-1","candidate_prefill_chunck_tokens":[128,256,512,1024,2048],"trusted_observation_count":3,"sliding_window_observation_count":5,"drift_trigger_factor":2,"context_buckets":{}}"#;
    fs::write(&state_file_path, previous_state).expect("previous state should be written");
    load_expect_none(
        &state_file_path,
        "test-model",
        "rev-1",
        &DEFAULT_CANDIDATES,
        SLIDING_WINDOW_OBSERVATION_COUNT,
    );
}

#[test]
fn should_return_none_for_mismatched_model_revision_candidates_or_window() {
    let temporary_directory = temporary_directory();
    let optimizer_directory = temporary_directory.path().join("optimizer");
    create_optimizer_with_default_candidates()
        .save_to_directory(&optimizer_directory, "test-model", "rev-1")
        .expect("save should succeed");
    let state_file_path = optimizer_directory.join(OPTIMIZER_STATE_FILE_NAME);
    load_expect_none(
        &state_file_path,
        "test-model",
        "rev-2",
        &DEFAULT_CANDIDATES,
        SLIDING_WINDOW_OBSERVATION_COUNT,
    );
    load_expect_none(
        &state_file_path,
        "test-model",
        "rev-1",
        &[128, 256],
        SLIDING_WINDOW_OBSERVATION_COUNT,
    );
    load_expect_none(
        &state_file_path,
        "test-model",
        "rev-1",
        &DEFAULT_CANDIDATES,
        3,
    );
}
