use std::fs;

use astronomical_experimental_aligned_expert_packs::{
    STREAMING_MODEL_IDENTITY_SUFFIX, StreamingModelManifest, StreamingModelPreparer,
    streaming_model_id_for,
};

use super::aligned_expert_pack::write_synthetic_expert_source_for_layer;

#[test]
fn should_publish_a_self_sufficient_per_expert_streaming_model() {
    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let source_model_directory = temporary_directory.path().join("synthetic-model");
    fs::create_dir(&source_model_directory)
        .expect("the synthetic source model directory should be creatable");
    fs::write(
        source_model_directory.join("config.json"),
        b"{\"model_type\":\"qwen3_5_moe\"}",
    )
    .expect("the synthetic config should be writable");
    let first_layer_plan = write_synthetic_expert_source_for_layer(
        &source_model_directory.join("model-00001-of-00002.safetensors"),
        0,
    );
    let second_layer_plan = write_synthetic_expert_source_for_layer(
        &source_model_directory.join("model-00002-of-00002.safetensors"),
        1,
    );
    let preparer = StreamingModelPreparer::from_layer_plans(
        &source_model_directory,
        "synthetic-model",
        streaming_model_id_for("synthetic-model"),
        "revision-1",
        vec![first_layer_plan, second_layer_plan],
    )
    .expect("the synthetic streaming model should plan");
    let output_directory = temporary_directory
        .path()
        .join(format!("synthetic-model{STREAMING_MODEL_IDENTITY_SUFFIX}"));
    let mut progress_events = Vec::new();

    let preparation_report = preparer
        .prepare(&output_directory, false, |progress_event| {
            progress_events.push(progress_event);
        })
        .expect("the streaming model should prepare");

    assert_eq!(
        preparation_report.streaming_model_id,
        "synthetic-model-expert-streaming"
    );
    assert_eq!(preparation_report.completed_expert_file_count, 8);
    assert_eq!(progress_events.len(), 8);
    assert!(!preparation_report.reused_existing_revision);
    assert!(output_directory.join("config.json").is_file());
    assert!(output_directory.join("layers/0/0.apack").is_file());
    assert!(output_directory.join("layers/1/3.apack").is_file());
    let streaming_model_manifest =
        StreamingModelManifest::read_from_revision_directory(&output_directory)
            .expect("the streaming model manifest should parse");
    assert_eq!(streaming_model_manifest.expert_files.len(), 8);
    assert!(
        streaming_model_manifest
            .resident_files
            .iter()
            .any(|resident_file| resident_file.file_name == "config.json")
    );
    // Single-copy layout: the resident bundle replaces the source shards.
    assert!(
        streaming_model_manifest
            .resident_files
            .iter()
            .any(|resident_file| resident_file.file_name == "resident.safetensors"),
        "the revision must carry the resident weight bundle"
    );
    assert!(
        fs::read_dir(&output_directory)
            .expect("the revision directory should be readable")
            .filter_map(|entry| entry.ok())
            .all(|entry| {
                let file_name = entry.file_name().to_string_lossy().into_owned();
                entry.path().is_dir()
                    || (!file_name.ends_with(".safetensors")
                        && file_name != "model.safetensors.index.json")
                    || file_name == "resident.safetensors"
            }),
        "the revision must not carry source shards"
    );
    let resident_header = astronomical_model_serving::parse_safetensors_header(
        &output_directory.join("resident.safetensors"),
    )
    .expect("the resident bundle should be a valid safetensors file");
    let resident_tensor_names: Vec<&str> = resident_header
        .tensor_entries
        .iter()
        .map(|tensor_entry| tensor_entry.tensor_name.as_str())
        .collect();
    assert_eq!(
        resident_tensor_names,
        vec!["layer.0.attention.weight", "layer.1.attention.weight",],
        "the resident bundle must hold exactly the non-expert tensors"
    );
}

#[test]
fn should_reuse_a_valid_streaming_model_revision() {
    let temporary_directory =
        tempfile::tempdir().expect("the test should create a temporary directory");
    let source_model_directory = temporary_directory.path().join("synthetic-model");
    fs::create_dir(&source_model_directory)
        .expect("the synthetic source model directory should be creatable");
    let layer_plan = write_synthetic_expert_source_for_layer(
        &source_model_directory.join("model.safetensors"),
        0,
    );
    let preparer = StreamingModelPreparer::from_layer_plans(
        &source_model_directory,
        "synthetic-model",
        streaming_model_id_for("synthetic-model"),
        "revision-1",
        vec![layer_plan],
    )
    .expect("the synthetic streaming model should plan");
    let output_directory = temporary_directory
        .path()
        .join("synthetic-model-expert-streaming");
    preparer
        .prepare(&output_directory, false, |_| {})
        .expect("the first streaming-model preparation should succeed");

    let reuse_report = preparer
        .prepare(&output_directory, false, |_| {
            panic!("a valid revision must not rebuild expert files");
        })
        .expect("the valid streaming model should be reused");

    assert!(reuse_report.reused_existing_revision);
}
