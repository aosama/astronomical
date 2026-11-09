//! Contracts for reading one stored tensor's payload size from a validated
//! SafeTensors header: present, absent, and zero-length tensors.

use std::io::Write;

use super::ValidatedSafetensorsSource;
use crate::artifact_validation::{RequiredFileProfile, TensorSourceId, validate_required_file};

/// Writes a shard with a 4096-byte `U8` tensor and a zero-length tensor, and
/// validates it as one source.
fn validated_two_tensor_source(shard_directory: &std::path::Path) -> ValidatedSafetensorsSource {
    let header = serde_json::json!({
        "language_model.model.norm.weight": {
            "dtype": "U8", "shape": [4096], "data_offsets": [0, 4096]
        },
        "language_model.model.empty": {
            "dtype": "U8", "shape": [0], "data_offsets": [4096, 4096]
        }
    });
    let header_bytes = serde_json::to_vec(&header).expect("the header should serialize");
    let mut shard_file = std::fs::File::create(shard_directory.join("model.safetensors"))
        .expect("the shard should be created");
    shard_file
        .write_all(&(header_bytes.len() as u64).to_le_bytes())
        .and_then(|()| shard_file.write_all(&header_bytes))
        .and_then(|()| shard_file.write_all(&[7_u8; 4096]))
        .expect("the shard should be written");
    let required_file = validate_required_file(
        shard_directory,
        &RequiredFileProfile {
            file_name: "model.safetensors".to_owned(),
            size_bytes: 0,
        },
    )
    .expect("the shard should validate as a required file");
    ValidatedSafetensorsSource::parse(TensorSourceId::new(0), required_file)
        .expect("the shard header should parse")
}

#[test]
fn should_read_a_stored_tensor_payload_size_from_its_header_offsets() {
    let shard_directory = tempfile::tempdir().expect("a shard folder should exist");
    let validated_source = validated_two_tensor_source(shard_directory.path());

    assert_eq!(
        validated_source.stored_tensor_payload_bytes("language_model.model.norm.weight"),
        Some(4096)
    );
}

#[test]
fn should_read_a_zero_length_tensor_as_zero_bytes() {
    let shard_directory = tempfile::tempdir().expect("a shard folder should exist");
    let validated_source = validated_two_tensor_source(shard_directory.path());

    assert_eq!(
        validated_source.stored_tensor_payload_bytes("language_model.model.empty"),
        Some(0)
    );
}

#[test]
fn should_answer_none_for_a_tensor_the_source_does_not_hold() {
    let shard_directory = tempfile::tempdir().expect("a shard folder should exist");
    let validated_source = validated_two_tensor_source(shard_directory.path());

    assert_eq!(
        validated_source.stored_tensor_payload_bytes("language_model.lm_head.weight"),
        None
    );
}
