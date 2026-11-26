//! Unit contracts for the steps of header parsing that need no file: one
//! tensor entry, its dtype field, and the declared byte count.

use super::{
    SafetensorsDtype, SafetensorsHeaderError, parse_tensor_dtype, parse_tensor_header_entry,
    validate_declared_byte_count,
};

fn tensor_header_object(header_json: &str) -> serde_json::Map<String, serde_json::Value> {
    serde_json::from_str(header_json).expect("the test header should be a JSON object")
}

#[test]
fn should_make_a_tensor_entry_file_relative_by_adding_the_payload_start() {
    let tensor_value: serde_json::Value =
        serde_json::from_str(r#"{"dtype":"BF16","shape":[2,3],"data_offsets":[4,16]}"#)
            .expect("the entry should be valid JSON");

    let entry = parse_tensor_header_entry("weights", &tensor_value, 100, 16)
        .expect("a consistent entry should parse");

    assert_eq!(entry.tensor_name, "weights");
    assert_eq!(entry.dtype, SafetensorsDtype::BFloat16);
    assert_eq!(entry.shape, [2, 3]);
    assert_eq!(entry.data_start_offset, 104);
    assert_eq!(entry.data_end_offset, 116);
}

#[test]
fn should_accept_a_zero_length_tensor_at_the_payload_end() {
    let tensor_value: serde_json::Value =
        serde_json::from_str(r#"{"dtype":"U8","shape":[0],"data_offsets":[16,16]}"#)
            .expect("the entry should be valid JSON");

    let entry = parse_tensor_header_entry("empty", &tensor_value, 100, 16)
        .expect("a zero-length tensor should parse");

    assert_eq!(entry.data_start_offset, 116);
    assert_eq!(entry.data_end_offset, 116);
}

#[test]
fn should_reject_a_tensor_entry_that_is_not_a_json_object() {
    let tensor_value = serde_json::json!(["BF16", [2, 3]]);

    assert!(matches!(
        parse_tensor_header_entry("weights", &tensor_value, 100, 16),
        Err(SafetensorsHeaderError::HeaderNotObject { .. })
    ));
}

#[test]
fn should_read_a_supported_dtype_field() {
    let tensor_header = tensor_header_object(r#"{"dtype":"F32"}"#);

    assert_eq!(
        parse_tensor_dtype("weights", &tensor_header).expect("F32 is supported"),
        SafetensorsDtype::Float32
    );
}

#[test]
fn should_report_a_missing_dtype_field_as_the_dtype_missing() {
    let tensor_header = tensor_header_object(r#"{"shape":[1]}"#);

    assert!(matches!(
        parse_tensor_dtype("weights", &tensor_header),
        Err(SafetensorsHeaderError::UnsupportedDtype { dtype, .. }) if dtype == "missing"
    ));
}

#[test]
fn should_report_an_unknown_dtype_by_its_own_name() {
    let tensor_header = tensor_header_object(r#"{"dtype":"F64"}"#);

    assert!(matches!(
        parse_tensor_dtype("weights", &tensor_header),
        Err(SafetensorsHeaderError::UnsupportedDtype { dtype, .. }) if dtype == "F64"
    ));
}

#[test]
fn should_accept_a_declared_byte_count_equal_to_shape_times_dtype_width() {
    assert!(
        validate_declared_byte_count("weights", &[2, 3], SafetensorsDtype::Float16, 12).is_ok()
    );
}

#[test]
fn should_accept_a_scalar_as_one_element() {
    assert!(validate_declared_byte_count("scalar", &[], SafetensorsDtype::Float32, 4).is_ok());
}

#[test]
fn should_name_both_byte_counts_when_the_declared_one_disagrees() {
    let mismatch = validate_declared_byte_count("weights", &[2, 3], SafetensorsDtype::Float16, 11)
        .expect_err("eleven bytes cannot hold six 16-bit elements");

    assert!(matches!(
        mismatch,
        SafetensorsHeaderError::ByteCountMismatch {
            declared_bytes: 11,
            expected_bytes: 12,
            ..
        }
    ));
}

#[test]
fn should_fail_closed_when_the_element_count_alone_overflows() {
    let overflow =
        validate_declared_byte_count("weights", &[usize::MAX, 2], SafetensorsDtype::Uint8, 0)
            .expect_err("an overflowing shape must not pass");

    assert!(matches!(
        overflow,
        SafetensorsHeaderError::ByteCountMismatch {
            expected_bytes: 0,
            ..
        }
    ));
}

#[test]
fn should_fail_closed_when_only_the_dtype_width_multiply_overflows() {
    let overflow = validate_declared_byte_count(
        "weights",
        &[usize::MAX / 2 + 1],
        SafetensorsDtype::Float32,
        0,
    )
    .expect_err("an overflowing byte count must not wrap to a small value");

    assert!(matches!(
        overflow,
        SafetensorsHeaderError::ByteCountMismatch {
            expected_bytes: 0,
            ..
        }
    ));
}
