//! The shared image folder checks every image family's discovery uses.

use std::fs;
use std::path::Path;

use super::*;
use test_harness::fixtures::image_pipeline_fixture::qwen_image_21_scheduler_config;

fn write_safetensors(weight_path: &Path, header_size_bytes: u64, payload_size_bytes: usize) {
    let mut file_bytes = header_size_bytes.to_le_bytes().to_vec();
    file_bytes.extend(vec![b' '; usize::try_from(header_size_bytes).unwrap()]);
    file_bytes.extend(vec![7_u8; payload_size_bytes]);
    fs::write(weight_path, file_bytes).unwrap();
}

/// The scheduler profile Qwen-Image-2.1 ships with, which the shared
/// fixture document below is written for.
const SHIPPED_PROFILE: FlowMatchSchedulerProfile = FlowMatchSchedulerProfile {
    max_image_seq_len: 8192,
    max_shift: 0.9,
    shift: 1.0,
    shift_terminal: Some(0.02),
};

fn write_scheduler(model_directory: &Path, max_shift: f64) {
    fs::create_dir_all(model_directory.join("scheduler")).unwrap();
    let mut scheduler_document = qwen_image_21_scheduler_config();
    scheduler_document["max_shift"] = serde_json::json!(max_shift);
    fs::write(
        model_directory.join("scheduler/scheduler_config.json"),
        scheduler_document.to_string(),
    )
    .unwrap();
}

#[test]
fn should_size_only_a_weight_file_that_exists_and_is_not_empty() {
    let model_directory = tempfile::tempdir().unwrap();
    let weight_path = model_directory.path().join("model.safetensors");
    assert_eq!(nonempty_file_size(&weight_path), None);
    fs::write(&weight_path, b"").unwrap();
    assert_eq!(nonempty_file_size(&weight_path), None);
    assert_eq!(nonempty_file_size(model_directory.path()), None);
    fs::write(&weight_path, b"romeo").unwrap();
    assert_eq!(nonempty_file_size(&weight_path), Some(5));
}

#[test]
fn should_measure_the_tensor_data_after_a_safetensors_header() {
    let model_directory = tempfile::tempdir().unwrap();
    let weight_path = model_directory.path().join("model.safetensors");
    write_safetensors(&weight_path, 16, 24);
    assert_eq!(safetensors_payload_size(&weight_path), Some(24));
}

#[test]
fn should_find_no_tensor_data_in_a_short_or_header_only_file() {
    let model_directory = tempfile::tempdir().unwrap();
    let weight_path = model_directory.path().join("model.safetensors");
    fs::write(&weight_path, [1_u8, 2, 3]).unwrap();
    assert_eq!(safetensors_payload_size(&weight_path), None);
    write_safetensors(&weight_path, 16, 0);
    assert_eq!(safetensors_payload_size(&weight_path), None);
    fs::write(&weight_path, 4_096_u64.to_le_bytes()).unwrap();
    assert_eq!(safetensors_payload_size(&weight_path), None);
}

#[test]
fn should_follow_only_relative_safetensors_shard_paths_inside_the_folder() {
    assert!(is_safe_safetensors_path("model-00001-of-00002.safetensors"));
    assert!(is_safe_safetensors_path("transformer/model.safetensors"));
    for unsafe_shard_path in [
        "",
        "/etc/model.safetensors",
        "../model.safetensors",
        "transformer/../../model.safetensors",
        "transformer\\model.safetensors",
        "model.bin",
    ] {
        assert!(
            !is_safe_safetensors_path(unsafe_shard_path),
            "{unsafe_shard_path}"
        );
    }
}

#[test]
fn should_accept_only_the_reviewed_scheduler_profile() {
    let model_directory = tempfile::tempdir().unwrap();
    assert!(!is_reviewed_flow_match_scheduler(
        model_directory.path(),
        4 * 1024 * 1024,
        &SHIPPED_PROFILE
    ));
    write_scheduler(model_directory.path(), 0.9);
    assert!(is_reviewed_flow_match_scheduler(
        model_directory.path(),
        4 * 1024 * 1024,
        &SHIPPED_PROFILE
    ));
    write_scheduler(model_directory.path(), 1.15);
    assert!(!is_reviewed_flow_match_scheduler(
        model_directory.path(),
        4 * 1024 * 1024,
        &SHIPPED_PROFILE
    ));
}

#[test]
fn should_read_the_pipeline_revision_hugging_face_recorded_or_report_none() {
    let model_directory = tempfile::tempdir().unwrap();
    assert_eq!(
        pipeline_provenance(model_directory.path(), "provider/pipeline"),
        None
    );
    let download_metadata_directory = model_directory.path().join(".cache/huggingface/download");
    fs::create_dir_all(&download_metadata_directory).unwrap();
    fs::write(
        download_metadata_directory.join("model_index.json.metadata"),
        "4db4e8c0c0e7a1debf0320415bec8388e888494c\nignored\n",
    )
    .unwrap();
    assert_eq!(
        pipeline_provenance(model_directory.path(), "provider/pipeline"),
        Some((
            "provider/pipeline".to_owned(),
            "4db4e8c0c0e7a1debf0320415bec8388e888494c".to_owned()
        ))
    );
}
