//! Contracts for reading one stored tensor's payload size from a validated
//! SafeTensors header: present, absent, and zero-length tensors.

use std::io::Write;

use super::{ValidatedSafetensorsSource, parse_tensor_metadata};
use crate::{ArtifactValidationError, RequiredFileProfile, TensorSourceId, validate_required_file};

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

#[cfg(feature = "direct-mlx")]
#[test]
fn should_make_a_stored_tensor_range_absolute_by_adding_the_data_section_start() {
    let shard_directory = tempfile::tempdir().expect("a shard folder should exist");
    let validated_source = validated_two_tensor_source(shard_directory.path());
    let header_length = u64::from_le_bytes(
        std::fs::read(shard_directory.path().join("model.safetensors"))
            .expect("the shard should read")[..8]
            .try_into()
            .expect("the shard should start with its header length"),
    );

    let tensor_file_range = validated_source
        .stored_tensor_file_range("language_model.model.norm.weight")
        .expect("the tensor should have a range");

    assert_eq!(tensor_file_range.file_start_offset, 8 + header_length);
    assert_eq!(tensor_file_range.file_end_offset, 8 + header_length + 4096);
    assert_eq!(tensor_file_range.dtype, "U8");
    assert_eq!(tensor_file_range.shape, [4096]);
    assert!(
        validated_source
            .stored_tensor_file_range("language_model.lm_head.weight")
            .is_none()
    );
}

fn header_entries(header_json: &str) -> serde_json::Map<String, serde_json::Value> {
    serde_json::from_str(header_json).expect("the test header should be a JSON object")
}

#[test]
fn should_parse_each_tensor_entry_under_its_stored_name() {
    let tensor_metadata = parse_tensor_metadata(
        header_entries(
            r#"{
                "embed.weight": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]},
                "norm.weight": {"dtype": "F32", "shape": [3], "data_offsets": [12, 24]}
            }"#,
        ),
        "model.safetensors",
    )
    .expect("well-formed entries should parse");

    assert_eq!(tensor_metadata.len(), 2);
    let embedding = &tensor_metadata["embed.weight"];
    assert_eq!(embedding.dtype, "BF16");
    assert_eq!(embedding.shape, [2, 3]);
    assert_eq!(embedding.data_offsets, [0, 12]);
    assert_eq!(tensor_metadata["norm.weight"].data_start_offset(), 12);
}

#[test]
fn should_parse_a_header_with_no_tensors_to_an_empty_map() {
    let tensor_metadata = parse_tensor_metadata(header_entries("{}"), "model.safetensors")
        .expect("an empty header should parse");

    assert!(tensor_metadata.is_empty());
}

#[test]
fn should_name_the_file_when_a_tensor_entry_lacks_its_dtype() {
    let parse_failure = parse_tensor_metadata(
        header_entries(r#"{"embed.weight": {"shape": [2], "data_offsets": [0, 4]}}"#),
        "model-00002.safetensors",
    )
    .expect_err("an entry without a dtype should not parse");

    assert!(matches!(
        parse_failure,
        ArtifactValidationError::InvalidSafetensorsHeader { file_name, .. }
            if file_name == "model-00002.safetensors"
    ));
}

#[test]
fn should_reject_a_tensor_entry_that_is_not_an_object() {
    let parse_failure = parse_tensor_metadata(
        header_entries(r#"{"embed.weight": 7}"#),
        "model.safetensors",
    );

    assert!(matches!(
        parse_failure,
        Err(ArtifactValidationError::InvalidSafetensorsHeader { .. })
    ));
}
