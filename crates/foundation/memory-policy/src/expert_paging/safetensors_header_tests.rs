//! Boundary, negative, and happy-path tests for bounded safetensors header
//! parsing, built on real scratch files so the framing bytes are exercised.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::expert_paging::safetensors_header::{SafetensorsHeaderError, parse_safetensors_header};
use crate::safetensors::SAFETENSORS_HEADER_LENGTH_PREFIX_BYTES;

static SCRATCH_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// A scratch safetensors file removed when the guard drops.
struct ScratchSafetensorsFile(PathBuf);

impl Drop for ScratchSafetensorsFile {
    fn drop(&mut self) {
        let _removed = std::fs::remove_file(&self.0);
    }
}

fn scratch_directory() -> PathBuf {
    let directory = std::env::temp_dir().join(format!(
        "astronomical-memory-policy-safetensors-header-tests-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&directory).expect("scratch directory could be created");
    directory
}

fn write_safetensors_file(header_json: &str, payload_byte_count: usize) -> ScratchSafetensorsFile {
    let sequence = SCRATCH_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let file_path = scratch_directory().join(format!("header-{sequence}.safetensors"));
    let mut file_contents = Vec::new();
    file_contents.extend_from_slice(
        &u64::try_from(header_json.len())
            .expect("test header length fits u64")
            .to_le_bytes(),
    );
    file_contents.extend_from_slice(header_json.as_bytes());
    file_contents.resize(file_contents.len() + payload_byte_count, 0);
    std::fs::write(&file_path, file_contents).expect("scratch safetensors file could be written");
    ScratchSafetensorsFile(file_path)
}

fn write_raw_bytes(raw_bytes: &[u8]) -> ScratchSafetensorsFile {
    let sequence = SCRATCH_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let file_path = scratch_directory().join(format!("raw-{sequence}.safetensors"));
    std::fs::write(&file_path, raw_bytes).expect("scratch raw file could be written");
    ScratchSafetensorsFile(file_path)
}

#[test]
fn should_parse_a_minimal_valid_header_with_exact_payload_boundaries() {
    let scratch_file = write_safetensors_file(
        r#"{"weights":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#,
        12,
    );
    let header = parse_safetensors_header(&scratch_file.0).expect("valid header should parse");

    let entry = header
        .tensor_entry_for_name("weights")
        .expect("tensor entry should exist");
    let expected_payload_start = SAFETENSORS_HEADER_LENGTH_PREFIX_BYTES
        + r#"{"weights":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#.len() as u64;
    assert_eq!(entry.data_start_offset, expected_payload_start);
    assert_eq!(entry.data_end_offset, expected_payload_start + 12);
    assert_eq!(
        header.total_file_size_bytes,
        expected_payload_start + 12,
        "a tensor ending exactly at the file end is the accepted boundary"
    );
}

#[test]
fn should_parse_a_scalar_shaped_tensor_as_one_element() {
    let scratch_file = write_safetensors_file(
        r#"{"scalar":{"dtype":"F32","shape":[],"data_offsets":[0,4]}}"#,
        4,
    );
    let header = parse_safetensors_header(&scratch_file.0).expect("scalar header should parse");
    assert!(header.tensor_entry_for_name("scalar").is_some());
}

#[test]
fn should_reject_a_missing_file() {
    let missing_path = scratch_directory().join("does-not-exist.safetensors");
    assert!(matches!(
        parse_safetensors_header(&missing_path),
        Err(SafetensorsHeaderError::FileNotFound { .. })
    ));
}

#[test]
fn should_reject_a_file_smaller_than_the_length_prefix() {
    let scratch_file = write_raw_bytes(&[0_u8; 7]);
    assert!(matches!(
        parse_safetensors_header(&scratch_file.0),
        Err(SafetensorsHeaderError::FileTooSmall { file_size_bytes: 7 })
    ));
}

#[test]
fn should_reject_a_header_length_prefix_declared_beyond_the_file() {
    // A prefix claiming a 100-byte header when only a 2-byte header follows:
    // the declared header end passes the file boundary.
    let mut file_contents = 100_u64.to_le_bytes().to_vec();
    file_contents.extend_from_slice(b"{}");
    let scratch_file = write_raw_bytes(&file_contents);
    assert!(matches!(
        parse_safetensors_header(&scratch_file.0),
        Err(SafetensorsHeaderError::HeaderBeyondFile { .. })
    ));
}

#[test]
fn should_reject_an_oversized_header_length_prefix() {
    let scratch_file = write_raw_bytes(&u64::MAX.to_le_bytes());
    assert!(matches!(
        parse_safetensors_header(&scratch_file.0),
        Err(SafetensorsHeaderError::HeaderTooLarge { .. })
    ));
}

#[test]
fn should_reject_an_invalid_header_json_body() {
    let scratch_file = write_safetensors_file("not json at all", 0);
    assert!(matches!(
        parse_safetensors_header(&scratch_file.0),
        Err(SafetensorsHeaderError::HeaderNotJson(_))
    ));
}

#[test]
fn should_reject_duplicate_tensor_keys() {
    let scratch_file = write_safetensors_file(
        r#"{"t":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"t":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
        1,
    );
    assert!(matches!(
        parse_safetensors_header(&scratch_file.0),
        Err(SafetensorsHeaderError::HeaderNotJson(_))
    ));
}

#[test]
fn should_reject_an_unsupported_dtype() {
    let scratch_file = write_safetensors_file(
        r#"{"t":{"dtype":"F64","shape":[1],"data_offsets":[0,1]}}"#,
        1,
    );
    assert!(matches!(
        parse_safetensors_header(&scratch_file.0),
        Err(SafetensorsHeaderError::UnsupportedDtype { .. })
    ));
}

#[test]
fn should_reject_a_non_object_tensor_entry() {
    let scratch_file = write_safetensors_file(r#"{"t":5}"#, 0);
    assert!(matches!(
        parse_safetensors_header(&scratch_file.0),
        Err(SafetensorsHeaderError::HeaderNotObject { .. })
    ));
}

#[test]
fn should_reject_data_offsets_that_invert_start_and_end() {
    let scratch_file = write_safetensors_file(
        r#"{"t":{"dtype":"U8","shape":[1],"data_offsets":[2,1]}}"#,
        4,
    );
    assert!(matches!(
        parse_safetensors_header(&scratch_file.0),
        Err(SafetensorsHeaderError::InvalidDataOffsets { .. })
    ));
}

#[test]
fn should_reject_data_offsets_outside_the_payload() {
    let scratch_file = write_safetensors_file(
        r#"{"t":{"dtype":"U8","shape":[1],"data_offsets":[0,5]}}"#,
        4,
    );
    assert!(matches!(
        parse_safetensors_header(&scratch_file.0),
        Err(SafetensorsHeaderError::DataOffsetsOutsidePayload {
            payload_byte_count: 4,
            ..
        })
    ));
}

#[test]
fn should_reject_a_declared_byte_count_that_disagrees_with_shape_and_dtype() {
    let scratch_file = write_safetensors_file(
        r#"{"t":{"dtype":"U8","shape":[3],"data_offsets":[0,2]}}"#,
        4,
    );
    assert!(matches!(
        parse_safetensors_header(&scratch_file.0),
        Err(SafetensorsHeaderError::ByteCountMismatch { .. })
    ));
}

#[test]
fn should_reject_a_shape_whose_byte_count_overflows_the_usize_domain() {
    // 2^62 elements of a 8-byte dtype need 2^65 bytes; the validated
    // declaration wraps to zero unless the byte-count arithmetic is checked,
    // letting a corrupt header pass with a zero-byte payload claim.
    let scratch_file = write_safetensors_file(
        r#"{"t":{"dtype":"U64","shape":[4611686018427387904],"data_offsets":[0,0]}}"#,
        0,
    );
    assert!(matches!(
        parse_safetensors_header(&scratch_file.0),
        Err(SafetensorsHeaderError::ByteCountMismatch { .. })
    ));
}

#[test]
fn should_reject_a_directory_path_as_not_a_regular_file() {
    let scratch_directory_path = scratch_directory();
    assert!(matches!(
        parse_safetensors_header(&scratch_directory_path),
        Err(SafetensorsHeaderError::NotARegularFile { .. })
    ));
}

#[test]
fn should_reject_a_tensor_entry_with_a_missing_dtype_field() {
    let scratch_file = write_safetensors_file(r#"{"t":{"shape":[1],"data_offsets":[0,1]}}"#, 1);
    assert!(matches!(
        parse_safetensors_header(&scratch_file.0),
        Err(SafetensorsHeaderError::UnsupportedDtype { dtype, .. }) if dtype == "missing"
    ));
}

#[test]
fn should_reject_every_invalid_shape_entry() {
    // Negative, fractional, and absent shapes are all structural defects.
    for header_json in [
        r#"{"t":{"dtype":"U8","shape":[-1],"data_offsets":[0,1]}}"#,
        r#"{"t":{"dtype":"U8","shape":[1.5],"data_offsets":[0,1]}}"#,
        r#"{"t":{"dtype":"U8","data_offsets":[0,1]}}"#,
        r#"{"t":{"dtype":"U8","shape":"wide","data_offsets":[0,1]}}"#,
    ] {
        let scratch_file = write_safetensors_file(header_json, 1);
        assert!(
            matches!(
                parse_safetensors_header(&scratch_file.0),
                Err(SafetensorsHeaderError::InvalidShape { tensor_name }) if tensor_name == "t"
            ),
            "shape entry in {header_json} must fail closed"
        );
    }
}

#[test]
fn should_reject_data_offsets_that_are_not_exactly_two_entries() {
    for header_json in [
        r#"{"t":{"dtype":"U8","shape":[1],"data_offsets":[0]}}"#,
        r#"{"t":{"dtype":"U8","shape":[1],"data_offsets":[0,1,2]}}"#,
        r#"{"t":{"dtype":"U8","shape":[1]}}"#,
        r#"{"t":{"dtype":"U8","shape":[1],"data_offsets":{"start":0,"end":1}}}"#,
    ] {
        let scratch_file = write_safetensors_file(header_json, 1);
        assert!(
            matches!(
                parse_safetensors_header(&scratch_file.0),
                Err(SafetensorsHeaderError::InvalidDataOffsets { tensor_name }) if tensor_name == "t"
            ),
            "offsets entry in {header_json} must fail closed"
        );
    }
}

#[test]
fn should_reject_a_negative_data_offset_start() {
    let scratch_file = write_safetensors_file(
        r#"{"t":{"dtype":"U8","shape":[1],"data_offsets":[-1,1]}}"#,
        1,
    );
    assert!(matches!(
        parse_safetensors_header(&scratch_file.0),
        Err(SafetensorsHeaderError::InvalidDataOffsets { .. })
    ));
}

#[test]
fn should_report_the_exact_byte_width_and_name_of_every_supported_dtype() {
    use crate::expert_paging::safetensors_header::SafetensorsDtype as Dtype;
    let every_dtype_with_width_and_name = [
        (Dtype::Bool, 1, "BOOL"),
        (Dtype::Int8, 1, "I8"),
        (Dtype::Uint8, 1, "U8"),
        (Dtype::Float8E4M3, 1, "F8_E4M3"),
        (Dtype::Float8E5M2, 1, "F8_E5M2"),
        (Dtype::Int16, 2, "I16"),
        (Dtype::Uint16, 2, "U16"),
        (Dtype::Float16, 2, "F16"),
        (Dtype::BFloat16, 2, "BF16"),
        (Dtype::Int32, 4, "I32"),
        (Dtype::Uint32, 4, "U32"),
        (Dtype::Float32, 4, "F32"),
        (Dtype::Int64, 8, "I64"),
        (Dtype::Uint64, 8, "U64"),
    ];
    for (dtype, expected_byte_width, expected_name) in every_dtype_with_width_and_name {
        assert_eq!(
            dtype.byte_width(),
            expected_byte_width,
            "{expected_name} byte width"
        );
        assert_eq!(dtype.to_string(), expected_name);
        assert_eq!(dtype.as_str(), expected_name);
    }
}

#[test]
fn should_answer_none_for_an_unknown_tensor_name() {
    let scratch_file = write_safetensors_file(
        r#"{"known":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
        1,
    );
    let header = parse_safetensors_header(&scratch_file.0).expect("valid header should parse");
    assert!(header.tensor_entry_for_name("known").is_some());
    assert!(
        header.tensor_entry_for_name("absent").is_none(),
        "an unknown tensor name must answer None, never a wrong entry"
    );
}
