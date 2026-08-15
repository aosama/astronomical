use std::fs;

use super::support::{
    DEFAULT_CANDIDATES, MAXIMUM_RETAINED_MEASUREMENTS, OPTIMIZER_STATE_FILE_NAME,
    context_at_position_range, create_optimizer_with_default_candidates, load_expect_some,
    measure_all_candidates, state_file_path_for_model, temporary_directory,
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
        state_file_path_for_model(&optimizer_directory, "test-model", "rev-1").exists(),
        "state file should exist after save"
    );
}

#[test]
fn should_update_persisted_file_after_each_save() {
    let temporary_directory = temporary_directory();
    let optimizer_directory = temporary_directory.path().join("optimizer");
    let mut optimizer = create_optimizer_with_default_candidates();
    let measurement_context = context_at_position_range(0);

    // Save with no measurements.
    optimizer
        .save_to_directory(&optimizer_directory, "test-model", "rev-1")
        .expect("first save should succeed");

    // Add measurements and save again.
    measure_all_candidates(&mut optimizer, measurement_context, 1000);
    optimizer
        .save_to_directory(&optimizer_directory, "test-model", "rev-1")
        .expect("second save should succeed");

    // Load should reflect the measurements.
    let mut loaded_optimizer = load_expect_some(
        &state_file_path_for_model(&optimizer_directory, "test-model", "rev-1"),
        "test-model",
        "rev-1",
        &DEFAULT_CANDIDATES,
        MAXIMUM_RETAINED_MEASUREMENTS,
    );

    let original_selection = optimizer.select_candidate_chunk_size(measurement_context);
    let loaded_selection = loaded_optimizer.select_candidate_chunk_size(measurement_context);
    assert_eq!(
        original_selection.selected_candidate_chunk_size_tokens,
        loaded_selection.selected_candidate_chunk_size_tokens,
        "reloaded optimizer should reflect updated measurements"
    );
}

#[test]
fn should_preserve_independent_measurements_for_two_model_revisions() {
    let temporary_directory = temporary_directory();
    let optimizer_directory = temporary_directory.path().join("optimizer");
    let measurement_context = context_at_position_range(0);
    let mut first_model_optimizer = create_optimizer_with_default_candidates();
    measure_all_candidates(&mut first_model_optimizer, measurement_context, 900);
    first_model_optimizer
        .save_to_directory(&optimizer_directory, "first-model", "first-revision")
        .expect("the first model state should save");

    let mut second_model_optimizer = create_optimizer_with_default_candidates();
    measure_all_candidates(&mut second_model_optimizer, measurement_context, 1_400);
    second_model_optimizer
        .save_to_directory(&optimizer_directory, "second-model", "second-revision")
        .expect("the second model state should save without replacing the first");

    let first_model_state_file_path =
        state_file_path_for_model(&optimizer_directory, "first-model", "first-revision");
    let second_model_state_file_path =
        state_file_path_for_model(&optimizer_directory, "second-model", "second-revision");
    assert_ne!(first_model_state_file_path, second_model_state_file_path);
    assert!(first_model_state_file_path.is_file());
    assert!(second_model_state_file_path.is_file());

    let first_model_optimizer = load_expect_some(
        &first_model_state_file_path,
        "first-model",
        "first-revision",
        &DEFAULT_CANDIDATES,
        MAXIMUM_RETAINED_MEASUREMENTS,
    );
    let first_model_measurements =
        first_model_optimizer.candidate_measurement_summaries(measurement_context);
    assert!(first_model_measurements.all_candidates_have_measurements);
    assert!(
        first_model_measurements
            .candidate_measurement_summaries
            .iter()
            .all(|candidate_measurement| candidate_measurement.measurement_count == 1)
    );
}

#[test]
fn should_write_valid_json_that_can_be_parsed_independently() {
    let temporary_directory = temporary_directory();
    let optimizer_directory = temporary_directory.path().join("optimizer");
    let mut optimizer = create_optimizer_with_default_candidates();

    let measurement_context = context_at_position_range(0);
    measure_all_candidates(&mut optimizer, measurement_context, 1500);

    optimizer
        .save_to_directory(&optimizer_directory, "test-model", "rev-1")
        .expect("save should succeed");

    let state_file_path = state_file_path_for_model(&optimizer_directory, "test-model", "rev-1");
    let state_file_content =
        fs::read_to_string(&state_file_path).expect("state file should be readable");

    // Verify the file is valid JSON with expected top-level structure
    let parsed_state_file_json: serde_json::Value =
        serde_json::from_str(&state_file_content).expect("saved file should be valid JSON");
    assert_eq!(
        parsed_state_file_json["format_version"], 5,
        "format version 5 identifies the renamed measurement schema"
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

#[test]
fn should_ignore_the_retired_state_filename() {
    let temporary_directory = temporary_directory();
    let retired_state_file_path = temporary_directory.path().join("prefill-chunck-size.json");
    fs::write(&retired_state_file_path, "retired optimizer state")
        .expect("retired state fixture should be written");

    let load_outcome = super::PromptProcessingChunkSizeOptimizer::load_from_path(
        temporary_directory.path().join(OPTIMIZER_STATE_FILE_NAME),
        "test-model".to_owned(),
        "rev-1".to_owned(),
        DEFAULT_CANDIDATES.to_vec(),
        MAXIMUM_RETAINED_MEASUREMENTS,
    )
    .expect("missing new state should not fail");

    assert!(load_outcome.is_none());
    assert!(retired_state_file_path.exists());
}
