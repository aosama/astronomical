use std::collections::HashSet;
use std::io::Write;

use astronomical_model_serving::{
    ArtifactValidationError, RequiredFileProfile, TensorDtype, TensorProfile, ValidatedWeightsFile,
    validate_bounded_safetensors_with_partial_profiles, validate_required_file_for_tests,
};

const WEIGHTS_FILE_NAME: &str = "model.safetensors";

#[test]
fn should_inventory_the_retained_descriptor_after_the_artifact_path_is_replaced() {
    let header = concat!(
        r#"{"z.tensor":{"dtype":"F16","shape":[2],"data_offsets":[0,4]},"#,
        r#""__metadata__":{"format":"mlx"},"#,
        r#""a.tensor":{"dtype":"I32","shape":[1],"data_offsets":[4,8]}}"#,
    );
    let bytes = safetensors_bytes(header, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    let weights_path = model_directory.path().join(WEIGHTS_FILE_NAME);
    std::fs::write(&weights_path, &bytes).expect("the test should write the original shard");
    let validated_weights_file = validate_weights_file(model_directory.path(), bytes.len() as u64);

    std::fs::rename(
        &weights_path,
        model_directory.path().join("retained-model.safetensors"),
    )
    .expect("the test should retain the original inode");
    std::fs::write(&weights_path, b"replacement").expect("the test should replace the path");

    let inventory = validated_weights_file
        .read_raw_safetensors_inventory_for_tests()
        .expect("the retained descriptor should produce an inventory");
    let payload_start = 8 + header.len() as u64;

    assert_eq!(inventory.shard_payload_bytes, 8);
    assert_eq!(inventory.tensor_descriptors.len(), 2);
    assert_eq!(inventory.tensor_descriptors[0].tensor_name, "a.tensor");
    assert_eq!(inventory.tensor_descriptors[0].dtype.to_string(), "I32");
    assert_eq!(inventory.tensor_descriptors[0].shape, vec![1]);
    assert_eq!(
        inventory.tensor_descriptors[0].data_start_offset_bytes,
        payload_start + 4
    );
    assert_eq!(
        inventory.tensor_descriptors[0].data_end_offset_bytes,
        payload_start + 8
    );
    assert_eq!(inventory.tensor_descriptors[1].tensor_name, "z.tensor");
    assert!(
        inventory
            .tensor_descriptors
            .iter()
            .all(|descriptor| descriptor.tensor_name != "__metadata__")
    );
}

#[test]
fn should_accept_every_dtype_supported_by_the_safetensors_format() {
    let dtype_cases = [
        ("BOOL", 8_usize),
        ("F4", 4),
        ("F6_E2M3", 6),
        ("F6_E3M2", 6),
        ("U8", 8),
        ("I8", 8),
        ("F8_E5M2", 8),
        ("F8_E4M3", 8),
        ("F8_E8M0", 8),
        ("F8_E4M3FNUZ", 8),
        ("F8_E5M2FNUZ", 8),
        ("I16", 16),
        ("U16", 16),
        ("F16", 16),
        ("BF16", 16),
        ("I32", 32),
        ("U32", 32),
        ("F32", 32),
        ("C64", 64),
        ("F64", 64),
        ("I64", 64),
        ("U64", 64),
    ];

    for (dtype_name, payload_length_bytes) in dtype_cases {
        // Eight elements make all sub-byte dtypes byte-aligned.
        let header = format!(
            r#"{{"tensor":{{"dtype":"{dtype_name}","shape":[8],"data_offsets":[0,{payload_length_bytes}]}}}}"#
        );
        let (model_directory, validated_weights_file) =
            validated_weights_file_for_header(&header, &vec![0; payload_length_bytes]);
        let inventory = validated_weights_file
            .read_raw_safetensors_inventory_for_tests()
            .unwrap_or_else(|error| panic!("{dtype_name} should be accepted: {error}"));
        assert_eq!(
            inventory.tensor_descriptors[0].dtype.to_string(),
            dtype_name
        );
        assert_eq!(
            inventory.tensor_descriptors[0].tensor_payload_bytes,
            payload_length_bytes as u64
        );
        drop(model_directory);
    }
}

#[test]
fn should_reject_duplicate_tensor_and_nested_object_keys() {
    let duplicate_cases = [
        concat!(
            r#"{"tensor":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"#,
            r#""tensor":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
        ),
        r#"{"tensor":{"dtype":"U8","dtype":"F16","shape":[1],"data_offsets":[0,1]}}"#,
        r#"{"__metadata__":{"format":"mlx","format":"pt"},"tensor":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
    ];

    for (case_index, header) in duplicate_cases.into_iter().enumerate() {
        let (_model_directory, validated_weights_file) =
            validated_weights_file_for_header(header, &[0]);
        let error = validated_weights_file
            .read_raw_safetensors_inventory_for_tests()
            .expect_err("duplicate object keys must fail before replacement");
        assert!(matches!(
            error,
            ArtifactValidationError::InvalidSafetensorsHeader { source, .. }
                if if case_index == 0 {
                    source.to_string().contains("duplicate safetensors header key")
                } else {
                    source.to_string().contains("duplicate JSON object field")
                }
        ));
    }
}

#[test]
fn should_reject_invalid_dtypes_shapes_and_intervals() {
    let cases = [
        (
            r#"{"tensor":{"dtype":"UNKNOWN","shape":[1],"data_offsets":[0,1]}}"#,
            &[0][..],
            "dtype",
        ),
        (
            r#"{"tensor":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#,
            &[0; 4][..],
            "width",
        ),
        (
            r#"{"tensor":{"dtype":"F4","shape":[1],"data_offsets":[0,1]}}"#,
            &[0][..],
            "alignment",
        ),
        (
            r#"{"tensor":{"dtype":"U8","shape":[1],"data_offsets":[1,0]}}"#,
            &[0][..],
            "reversed",
        ),
    ];

    for (header, payload, case_name) in cases {
        let (_model_directory, validated_weights_file) =
            validated_weights_file_for_header(header, payload);
        assert!(
            validated_weights_file
                .read_raw_safetensors_inventory_for_tests()
                .is_err(),
            "{case_name} must fail closed"
        );
    }
}

#[test]
fn should_reject_overflow_gaps_overlaps_out_of_bounds_and_trailing_bytes() {
    let maximum_dimension = usize::MAX;
    let aggregate_overflow = format!(
        r#"{{"first":{{"dtype":"U8","shape":[{maximum_dimension}],"data_offsets":[0,{maximum_dimension}]}},"second":{{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}}}"#
    );
    let cases: Vec<(String, Vec<u8>)> = vec![
        (r#"{"tensor":{"dtype":"U64","shape":[18446744073709551615,2],"data_offsets":[0,0]}}"#.to_owned(), vec![]),
        (aggregate_overflow, vec![]),
        (r#"{"first":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"second":{"dtype":"U8","shape":[1],"data_offsets":[2,3]}}"#.to_owned(), vec![0; 3]),
        (r#"{"first":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"second":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}"#.to_owned(), vec![0; 2]),
        (r#"{"tensor":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}"#.to_owned(), vec![0]),
        (r#"{"tensor":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#.to_owned(), vec![0; 2]),
    ];

    for (header, payload) in cases {
        let (_model_directory, validated_weights_file) =
            validated_weights_file_for_header(&header, &payload);
        validated_weights_file
            .read_raw_safetensors_inventory_for_tests()
            .expect_err("invalid aggregate or interval accounting must fail closed");
    }
}

#[test]
fn should_reject_an_empty_tensor_name_and_an_over_bounded_header() {
    let (_model_directory, validated_weights_file) = validated_weights_file_for_header(
        r#"{"":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
        &[0],
    );
    assert!(matches!(
        validated_weights_file.read_raw_safetensors_inventory_for_tests(),
        Err(ArtifactValidationError::InvalidSafetensorsTensorName {
            tensor_name_length_bytes: 0,
            ..
        })
    ));

    let declared_header_length_bytes = 16_u64 * 1024 * 1024 + 1;
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    std::fs::write(
        model_directory.path().join(WEIGHTS_FILE_NAME),
        declared_header_length_bytes.to_le_bytes(),
    )
    .expect("the test should write the framing fixture");
    let validated_weights_file = validate_weights_file(model_directory.path(), 8);
    assert!(matches!(
        validated_weights_file.read_raw_safetensors_inventory_for_tests(),
        Err(ArtifactValidationError::SafetensorsHeaderLengthTooLarge { .. })
    ));
}

#[test]
fn should_continue_excluding_metadata_from_existing_profile_validation() {
    let header = concat!(
        r#"{"__metadata__":{"format":"mlx"},"#,
        r#""tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
    );
    let bytes = safetensors_bytes(header, &[0; 4]);
    let mut weights_file = tempfile::tempfile().expect("the test should create a file");
    weights_file
        .write_all(&bytes)
        .expect("the test should write the fixture");
    let metadata = validate_bounded_safetensors_with_partial_profiles(
        &weights_file,
        bytes.len() as u64,
        WEIGHTS_FILE_NAME,
        &[TensorProfile {
            name: "tensor".to_owned(),
            dtype: TensorDtype::Float32,
            shape: vec![1],
        }],
        &HashSet::new(),
    )
    .expect("metadata must remain excluded from tensor profiles");
    assert_eq!(metadata.total_payload_bytes, 4);
}

fn validated_weights_file_for_header(
    header: &str,
    payload: &[u8],
) -> (tempfile::TempDir, ValidatedWeightsFile) {
    let model_directory = tempfile::tempdir().expect("the test should create a model directory");
    let bytes = safetensors_bytes(header, payload);
    std::fs::write(model_directory.path().join(WEIGHTS_FILE_NAME), &bytes)
        .expect("the test should write the fixture");
    let validated_weights_file = validate_weights_file(model_directory.path(), bytes.len() as u64);
    (model_directory, validated_weights_file)
}

fn validate_weights_file(
    model_directory: &std::path::Path,
    size_bytes: u64,
) -> ValidatedWeightsFile {
    validate_required_file_for_tests(
        model_directory,
        &RequiredFileProfile {
            file_name: WEIGHTS_FILE_NAME.to_owned(),
            size_bytes,
        },
    )
    .expect("the fixture should retain its validated descriptor")
}

fn safetensors_bytes(header: &str, payload: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(payload);
    bytes
}
