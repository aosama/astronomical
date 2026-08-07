use std::fs;

use super::support::{
    DEFAULT_CANDIDATES, OPTIMIZER_STATE_FILE_NAME, SLIDING_WINDOW_OBSERVATION_COUNT,
    context_at_bucket, create_optimizer_with_default_candidates, load_expect_some,
    observe_all_candidates, temporary_directory,
};

#[test]
fn should_create_optimizer_directory_on_save_if_it_does_not_exist() {
    let temporary_directory = temporary_directory();
    let optimizer_directory = temporary_directory
        .path()
        .join("deeply")
        .join("nested")
        .join("optimizer");
    let optimizer = create_optimizer_with_default_candidates();

    assert!(
        !optimizer_directory.exists(),
        "directory should not exist before save"
    );

    optimizer
        .save_to_directory(&optimizer_directory, "test-model", "rev-1")
        .expect("save should create nested directories");

    assert!(
        optimizer_directory.exists(),
        "directory should exist after save"
    );
    assert!(
        optimizer_directory.join(OPTIMIZER_STATE_FILE_NAME).exists(),
        "state file should exist after save"
    );
}

#[test]
fn should_update_persisted_file_after_each_save() {
    let temporary_directory = temporary_directory();
    let optimizer_directory = temporary_directory.path().join("optimizer");
    let mut optimizer = create_optimizer_with_default_candidates();
    let context = context_at_bucket(0);

    // Save with no observations
    optimizer
        .save_to_directory(&optimizer_directory, "test-model", "rev-1")
        .expect("first save should succeed");

    // Add observations and save again
    observe_all_candidates(&mut optimizer, context, 1000);
    optimizer
        .save_to_directory(&optimizer_directory, "test-model", "rev-1")
        .expect("second save should succeed");

    // Load should reflect the observations
    let mut loaded_optimizer = load_expect_some(
        &optimizer_directory.join(OPTIMIZER_STATE_FILE_NAME),
        "test-model",
        "rev-1",
        &DEFAULT_CANDIDATES,
        SLIDING_WINDOW_OBSERVATION_COUNT,
    );

    let original_decision = optimizer.ask(context);
    let loaded_decision = loaded_optimizer.ask(context);
    assert_eq!(
        original_decision.candidate_prefill_chunck_tokens,
        loaded_decision.candidate_prefill_chunck_tokens,
        "reloaded optimizer should reflect updated observations"
    );
}

#[test]
fn should_write_valid_json_that_can_be_parsed_independently() {
    let temporary_directory = temporary_directory();
    let optimizer_directory = temporary_directory.path().join("optimizer");
    let mut optimizer = create_optimizer_with_default_candidates();

    let context = context_at_bucket(0);
    observe_all_candidates(&mut optimizer, context, 1500);

    optimizer
        .save_to_directory(&optimizer_directory, "test-model", "rev-1")
        .expect("save should succeed");

    let state_file_path = optimizer_directory.join(OPTIMIZER_STATE_FILE_NAME);
    let state_file_content =
        fs::read_to_string(&state_file_path).expect("state file should be readable");

    // Verify the file is valid JSON with expected top-level structure
    let parsed_state_file_json: serde_json::Value =
        serde_json::from_str(&state_file_content).expect("saved file should be valid JSON");
    assert_eq!(
        parsed_state_file_json["format_version"], 4,
        "format version 4 should invalidate the retired median-throughput policy"
    );
    assert_eq!(
        parsed_state_file_json["model_id"], "test-model",
        "model_id should match"
    );
    assert_eq!(
        parsed_state_file_json["model_revision"], "rev-1",
        "model_revision should match"
    );
    assert!(
        parsed_state_file_json["context_buckets"].is_array(),
        "context_buckets should be a JSON array"
    );
}
