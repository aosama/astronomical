use std::fs;

use super::PromptProcessingChunkSizeOptimizer;
use super::support::*;

#[test]
fn should_return_none_when_optimizer_state_file_does_not_exist() {
    let temporary_directory = temporary_directory();
    let load_outcome = PromptProcessingChunkSizeOptimizer::load_from_path(
        temporary_directory.path().join("missing.json"),
        "test-model".to_owned(),
        "rev-1".to_owned(),
        DEFAULT_CANDIDATES.to_vec(),
        MAXIMUM_RETAINED_MEASUREMENTS,
    )
    .expect("missing state should not fail");
    assert!(load_outcome.is_none());
}

#[test]
fn should_return_none_for_corrupt_empty_or_truncated_state() {
    for serialized_state in ["", "not json", "{\"format_version\":6"] {
        let temporary_directory = temporary_directory();
        let state_file_path = temporary_directory.path().join(OPTIMIZER_STATE_FILE_NAME);
        fs::write(&state_file_path, serialized_state).expect("state fixture should be written");
        load_expect_none(
            &state_file_path,
            "test-model",
            "rev-1",
            &DEFAULT_CANDIDATES,
            MAXIMUM_RETAINED_MEASUREMENTS,
        );
    }
}

#[test]
fn should_return_none_for_previous_format_without_migration() {
    let temporary_directory = temporary_directory();
    let state_file_path = temporary_directory.path().join(OPTIMIZER_STATE_FILE_NAME);
    let previous_state = r#"{"format_version":5,"model_id":"test-model","model_revision":"rev-1","candidate_chunk_size_tokens":[128,256,512,1024,2048],"maximum_retained_measurements_per_candidate_and_context":5,"measurement_sequence":0,"execution_profile_buckets":[]}"#;
    fs::write(&state_file_path, previous_state).expect("previous state should be written");
    load_expect_none(
        &state_file_path,
        "test-model",
        "rev-1",
        &DEFAULT_CANDIDATES,
        MAXIMUM_RETAINED_MEASUREMENTS,
    );
}

#[test]
fn should_return_none_for_mismatched_model_revision_candidates_or_window() {
    let temporary_directory = temporary_directory();
    let optimizer_directory = temporary_directory.path().join("optimizer");
    create_optimizer_with_default_candidates()
        .save_to_directory(&optimizer_directory, "test-model", "rev-1")
        .expect("save should succeed");
    let state_file_path = state_file_path_for_model(&optimizer_directory, "test-model", "rev-1");
    load_expect_none(
        &state_file_path,
        "test-model",
        "rev-2",
        &DEFAULT_CANDIDATES,
        MAXIMUM_RETAINED_MEASUREMENTS,
    );
    load_expect_none(
        &state_file_path,
        "test-model",
        "rev-1",
        &[128, 256],
        MAXIMUM_RETAINED_MEASUREMENTS,
    );
    load_expect_none(
        &state_file_path,
        "test-model",
        "rev-1",
        &DEFAULT_CANDIDATES,
        3,
    );
}
