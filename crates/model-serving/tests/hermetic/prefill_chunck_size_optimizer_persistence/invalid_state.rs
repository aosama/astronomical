use std::fs;

use super::PrefillChunckSizeOptimizer;
use super::support::{
    DEFAULT_CANDIDATES, DRIFT_TRIGGER_FACTOR, OPTIMIZER_STATE_FILE_NAME,
    SLIDING_WINDOW_OBSERVATION_COUNT, TRUSTED_OBSERVATION_COUNT,
    create_optimizer_with_default_candidates, load_expect_none, temporary_directory,
};

#[test]
fn should_return_none_when_optimizer_state_file_does_not_exist() {
    let temporary_directory = temporary_directory();
    let nonexistent_state_file_path = temporary_directory
        .path()
        .join("nonexistent")
        .join(OPTIMIZER_STATE_FILE_NAME);

    let load_outcome = PrefillChunckSizeOptimizer::load_from_path(
        nonexistent_state_file_path,
        "test-model".to_string(),
        "rev-1".to_string(),
        DEFAULT_CANDIDATES.to_vec(),
        TRUSTED_OBSERVATION_COUNT,
        SLIDING_WINDOW_OBSERVATION_COUNT,
        DRIFT_TRIGGER_FACTOR,
    );

    assert!(
        load_outcome.is_ok(),
        "missing file should not error, should return Ok(None)"
    );
    assert!(
        load_outcome
            .expect("missing state file load should not error")
            .is_none(),
        "missing file should return None"
    );
}

#[test]
fn should_return_none_for_corrupt_json_file() {
    let temporary_directory = temporary_directory();
    let optimizer_directory = temporary_directory.path().join("optimizer");
    fs::create_dir_all(&optimizer_directory).expect("directory should be created");
    let state_file_path = optimizer_directory.join(OPTIMIZER_STATE_FILE_NAME);
    fs::write(&state_file_path, "this is not json at all!!!")
        .expect("corrupt file should be written");

    load_expect_none(
        &state_file_path,
        "test-model",
        "rev-1",
        &DEFAULT_CANDIDATES,
        TRUSTED_OBSERVATION_COUNT,
        SLIDING_WINDOW_OBSERVATION_COUNT,
        DRIFT_TRIGGER_FACTOR,
    );
}

#[test]
fn should_return_none_for_empty_file() {
    let temporary_directory = temporary_directory();
    let optimizer_directory = temporary_directory.path().join("optimizer");
    fs::create_dir_all(&optimizer_directory).expect("directory should be created");
    let state_file_path = optimizer_directory.join(OPTIMIZER_STATE_FILE_NAME);
    fs::write(&state_file_path, "").expect("empty file should be written");

    load_expect_none(
        &state_file_path,
        "test-model",
        "rev-1",
        &DEFAULT_CANDIDATES,
        TRUSTED_OBSERVATION_COUNT,
        SLIDING_WINDOW_OBSERVATION_COUNT,
        DRIFT_TRIGGER_FACTOR,
    );
}

#[test]
fn should_return_none_for_wrong_model_id() {
    let temporary_directory = temporary_directory();
    let optimizer_directory = temporary_directory.path().join("optimizer");
    let optimizer = create_optimizer_with_default_candidates();

    optimizer
        .save_to_directory(&optimizer_directory, "correct-model", "rev-1")
        .expect("save should succeed");

    load_expect_none(
        &optimizer_directory.join(OPTIMIZER_STATE_FILE_NAME),
        "wrong-model",
        "rev-1",
        &DEFAULT_CANDIDATES,
        TRUSTED_OBSERVATION_COUNT,
        SLIDING_WINDOW_OBSERVATION_COUNT,
        DRIFT_TRIGGER_FACTOR,
    );
}

#[test]
fn should_return_none_for_wrong_model_revision() {
    let temporary_directory = temporary_directory();
    let optimizer_directory = temporary_directory.path().join("optimizer");
    let optimizer = create_optimizer_with_default_candidates();

    optimizer
        .save_to_directory(&optimizer_directory, "test-model", "rev-1")
        .expect("save should succeed");

    load_expect_none(
        &optimizer_directory.join(OPTIMIZER_STATE_FILE_NAME),
        "test-model",
        "rev-2",
        &DEFAULT_CANDIDATES,
        TRUSTED_OBSERVATION_COUNT,
        SLIDING_WINDOW_OBSERVATION_COUNT,
        DRIFT_TRIGGER_FACTOR,
    );
}

#[test]
fn should_return_none_for_mismatched_candidate_prefill_chunck_tokens() {
    let temporary_directory = temporary_directory();
    let optimizer_directory = temporary_directory.path().join("optimizer");
    let optimizer = create_optimizer_with_default_candidates();

    optimizer
        .save_to_directory(&optimizer_directory, "test-model", "rev-1")
        .expect("save should succeed");

    load_expect_none(
        &optimizer_directory.join(OPTIMIZER_STATE_FILE_NAME),
        "test-model",
        "rev-1",
        &[256, 1024, 4096],
        TRUSTED_OBSERVATION_COUNT,
        SLIDING_WINDOW_OBSERVATION_COUNT,
        DRIFT_TRIGGER_FACTOR,
    );
}

#[test]
fn should_return_none_for_mismatched_trusted_observation_count() {
    let temporary_directory = temporary_directory();
    let optimizer_directory = temporary_directory.path().join("optimizer");
    let optimizer = create_optimizer_with_default_candidates();

    optimizer
        .save_to_directory(&optimizer_directory, "test-model", "rev-1")
        .expect("save should succeed");

    load_expect_none(
        &optimizer_directory.join(OPTIMIZER_STATE_FILE_NAME),
        "test-model",
        "rev-1",
        &DEFAULT_CANDIDATES,
        5, // mismatched: saved with 3, loading with 5
        SLIDING_WINDOW_OBSERVATION_COUNT,
        DRIFT_TRIGGER_FACTOR,
    );
}

#[test]
fn should_return_none_for_mismatched_drift_trigger_factor() {
    let temporary_directory = temporary_directory();
    let optimizer_directory = temporary_directory.path().join("optimizer");
    let optimizer = create_optimizer_with_default_candidates();

    optimizer
        .save_to_directory(&optimizer_directory, "test-model", "rev-1")
        .expect("save should succeed");

    load_expect_none(
        &optimizer_directory.join(OPTIMIZER_STATE_FILE_NAME),
        "test-model",
        "rev-1",
        &DEFAULT_CANDIDATES,
        TRUSTED_OBSERVATION_COUNT,
        SLIDING_WINDOW_OBSERVATION_COUNT,
        3, // mismatched: saved with 2, loading with 3
    );
}

#[test]
fn should_return_none_for_unknown_format_version() {
    let temporary_directory = temporary_directory();
    let optimizer_directory = temporary_directory.path().join("optimizer");
    fs::create_dir_all(&optimizer_directory).expect("directory should be created");
    let state_file_path = optimizer_directory.join(OPTIMIZER_STATE_FILE_NAME);
    let unsupported_format_state = r#"{"format_version":999,"model_id":"test-model","model_revision":"rev-1","candidate_prefill_chunck_tokens":[128,256,512,1024,2048],"trusted_observation_count":3,"sliding_window_observation_count":5,"drift_trigger_factor":2,"context_buckets":{}}"#;
    fs::write(&state_file_path, unsupported_format_state)
        .expect("future format file should be written");

    load_expect_none(
        &state_file_path,
        "test-model",
        "rev-1",
        &DEFAULT_CANDIDATES,
        TRUSTED_OBSERVATION_COUNT,
        SLIDING_WINDOW_OBSERVATION_COUNT,
        DRIFT_TRIGGER_FACTOR,
    );
}

#[test]
fn should_return_none_for_optimizer_state_from_pre_nax_attention_performance_generation() {
    let temporary_directory = temporary_directory();
    let optimizer_directory = temporary_directory.path().join("optimizer");
    fs::create_dir_all(&optimizer_directory).expect("directory should be created");
    let state_file_path = optimizer_directory.join(OPTIMIZER_STATE_FILE_NAME);
    let pre_nax_attention_state = r#"{"format_version":2,"model_id":"test-model","model_revision":"rev-1","candidate_prefill_chunck_tokens":[128,256,512,1024,2048],"trusted_observation_count":3,"sliding_window_observation_count":5,"drift_trigger_factor":2,"context_buckets":{}}"#;
    fs::write(&state_file_path, pre_nax_attention_state)
        .expect("pre-NAX attention optimizer state should be written");

    load_expect_none(
        &state_file_path,
        "test-model",
        "rev-1",
        &DEFAULT_CANDIDATES,
        TRUSTED_OBSERVATION_COUNT,
        SLIDING_WINDOW_OBSERVATION_COUNT,
        DRIFT_TRIGGER_FACTOR,
    );
}

#[test]
fn should_return_none_for_truncated_json_file() {
    let temporary_directory = temporary_directory();
    let optimizer_directory = temporary_directory.path().join("optimizer");
    fs::create_dir_all(&optimizer_directory).expect("directory should be created");
    let state_file_path = optimizer_directory.join(OPTIMIZER_STATE_FILE_NAME);
    // Valid start of JSON but truncated mid-field — simulates a crash during write
    let truncated_json =
        r#"{"format_version":1,"model_id":"test-model","model_revision":"rev-1","candidate_pref"#;
    fs::write(&state_file_path, truncated_json).expect("truncated file should be written");

    load_expect_none(
        &state_file_path,
        "test-model",
        "rev-1",
        &DEFAULT_CANDIDATES,
        TRUSTED_OBSERVATION_COUNT,
        SLIDING_WINDOW_OBSERVATION_COUNT,
        DRIFT_TRIGGER_FACTOR,
    );
}
