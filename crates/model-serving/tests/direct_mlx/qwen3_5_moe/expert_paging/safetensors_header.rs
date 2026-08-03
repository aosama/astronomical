//! Hermetic tests for safetensors header parsing and validation.
//!
//! These tests create temporary safetensors files and verify that
//! `parse_safetensors_header` correctly parses, validates, and rejects
//! various header conditions.

use std::io::Write;
use std::path::PathBuf;

use astronomical_model_serving::{
    SafetensorsDtype, SafetensorsHeaderError, parse_safetensors_header,
};

/// Creates a minimal valid safetensors file at the given path.
/// The file contains a single BF16 tensor named `test_tensor` with shape [2, 3].
fn create_minimal_safetensors_file(directory: &std::path::Path) -> (PathBuf, usize) {
    let header_json = serde_json::json!({
        "test_tensor": {
            "dtype": "BF16",
            "shape": [2, 3],
            "data_offsets": [0, 12]
        }
    });
    let header_bytes = serde_json::to_vec(&header_json).unwrap();
    let header_length = header_bytes.len();
    let payload_bytes = vec![0u8; 12]; // 2 * 3 * 2 bytes per BF16 element

    let file_path = directory.join("test.safetensors");
    let mut file = std::fs::File::create(&file_path).unwrap();
    file.write_all(&(header_length as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header_bytes).unwrap();
    file.write_all(&payload_bytes).unwrap();

    (file_path, header_length)
}

#[test]
fn should_parse_minimal_safetensors_header() {
    let temp_dir = tempfile::tempdir().unwrap();
    let (file_path, _header_length) = create_minimal_safetensors_file(temp_dir.path());

    let result = parse_safetensors_header(&file_path);
    assert!(
        result.is_ok(),
        "should parse a valid safetensors file: {:?}",
        result
    );

    let header = result.unwrap();
    assert_eq!(header.tensor_entries.len(), 1);
    let entry = &header.tensor_entries[0];
    assert_eq!(entry.tensor_name, "test_tensor");
    assert_eq!(entry.dtype, SafetensorsDtype::BFloat16);
    assert_eq!(entry.shape, vec![2, 3]);
    // data_offsets should be file-relative, not payload-relative
    assert!(
        entry.data_start_offset > 0,
        "data_start_offset should be file-relative"
    );
    assert!(entry.data_end_offset > entry.data_start_offset);
    assert_eq!(entry.data_end_offset - entry.data_start_offset, 12);
}

#[test]
fn should_parse_safetensors_header_with_multiple_tensors() {
    let temp_dir = tempfile::tempdir().unwrap();
    // Two tensors: weight (U32, [2, 128]) and scales (BF16, [1, 128])
    let header_json = serde_json::json!({
        "layer.weight": {
            "dtype": "U32",
            "shape": [2, 128],
            "data_offsets": [0, 1024]
        },
        "layer.scales": {
            "dtype": "BF16",
            "shape": [1, 128],
            "data_offsets": [1024, 1280]
        }
    });
    let header_bytes = serde_json::to_vec(&header_json).unwrap();
    let header_length = header_bytes.len();
    let payload_bytes = vec![0u8; 1280]; // enough for both tensors

    let file_path = temp_dir.path().join("multi_tensor.safetensors");
    let mut file = std::fs::File::create(&file_path).unwrap();
    file.write_all(&(header_length as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header_bytes).unwrap();
    file.write_all(&payload_bytes).unwrap();

    let result = parse_safetensors_header(&file_path);
    assert!(
        result.is_ok(),
        "should parse multi-tensor file: {:?}",
        result
    );
    let header = result.unwrap();
    assert_eq!(header.tensor_entries.len(), 2);

    let weight_entry = header.tensor_entry_for_name("layer.weight").unwrap();
    assert_eq!(weight_entry.dtype, SafetensorsDtype::Uint32);
    assert_eq!(weight_entry.shape, vec![2, 128]);

    let scales_entry = header.tensor_entry_for_name("layer.scales").unwrap();
    assert_eq!(scales_entry.dtype, SafetensorsDtype::BFloat16);
    assert_eq!(scales_entry.shape, vec![1, 128]);
}

#[test]
fn should_ignore_safetensors_metadata_entry_when_parsing_tensor_headers() {
    eprintln!("[safetensors-header] status=start case=ignore_metadata_entry");
    let temp_dir = tempfile::tempdir().unwrap();
    let header_json = serde_json::json!({
        "__metadata__": {
            "format": "pt"
        },
        "layer.weight": {
            "dtype": "U32",
            "shape": [1, 4],
            "data_offsets": [0, 16]
        }
    });
    let header_bytes = serde_json::to_vec(&header_json).unwrap();
    let header_length = header_bytes.len();
    let payload_bytes = vec![0u8; 16];
    let file_path = temp_dir.path().join("metadata_entry.safetensors");
    let mut file = std::fs::File::create(&file_path).unwrap();
    file.write_all(&(header_length as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header_bytes).unwrap();
    file.write_all(&payload_bytes).unwrap();

    let header = parse_safetensors_header(&file_path)
        .expect("the metadata entry is not a tensor and must be ignored");

    eprintln!(
        "[safetensors-header] status=success tensor_entry_count={}",
        header.tensor_entries.len()
    );
    assert_eq!(header.tensor_entries.len(), 1);
    assert!(header.tensor_entry_for_name("__metadata__").is_none());
    assert!(header.tensor_entry_for_name("layer.weight").is_some());
}

#[test]
fn should_reject_missing_safetensors_file() {
    let result = parse_safetensors_header(&PathBuf::from("/nonexistent/path.safetensors"));
    assert!(
        matches!(result, Err(SafetensorsHeaderError::FileNotFound { .. })),
        "should reject missing file: {:?}",
        result
    );
}

#[test]
fn should_reject_header_exceeding_safety_limit() {
    let temp_dir = tempfile::tempdir().unwrap();
    let file_path = temp_dir.path().join("oversized_header.safetensors");

    // Write a 64MB+ header length prefix (exceeds the 64MB safety limit)
    let mut file = std::fs::File::create(&file_path).unwrap();
    let oversized_length: u64 = 64 * 1024 * 1024 + 1; // 64MB + 1 byte
    file.write_all(&oversized_length.to_le_bytes()).unwrap();
    // Write enough payload to avoid "header beyond file" error
    let dummy = vec![0u8; 128];
    file.write_all(&dummy).unwrap();

    let result = parse_safetensors_header(&file_path);
    assert!(
        matches!(result, Err(SafetensorsHeaderError::HeaderTooLarge { .. })),
        "should reject oversized header: {:?}",
        result
    );
}

#[test]
fn should_reject_truncated_header() {
    let temp_dir = tempfile::tempdir().unwrap();
    let file_path = temp_dir.path().join("truncated_header.safetensors");

    // Write a length prefix claiming 1000 bytes of header, but only provide 100
    let mut file = std::fs::File::create(&file_path).unwrap();
    let declared_length: u64 = 1000;
    file.write_all(&declared_length.to_le_bytes()).unwrap();
    let short_header = vec![b'{'; 100]; // starts with valid JSON open brace
    file.write_all(&short_header).unwrap();

    let result = parse_safetensors_header(&file_path);
    // This could be HeaderBeyondFile or Io depending on file size.
    // The key assertion is that it fails.
    assert!(
        result.is_err(),
        "should reject truncated header: {:?}",
        result
    );
}

#[test]
fn should_reject_invalid_json_header() {
    let temp_dir = tempfile::tempdir().unwrap();
    let file_path = temp_dir.path().join("invalid_json.safetensors");

    // Write a valid length prefix but garbage JSON
    let garbage_json = b"this is not valid json!!!";
    let header_length = garbage_json.len() as u64;
    let mut file = std::fs::File::create(&file_path).unwrap();
    file.write_all(&header_length.to_le_bytes()).unwrap();
    file.write_all(garbage_json).unwrap();
    // Add some payload bytes
    file.write_all(&[0u8; 16]).unwrap();

    let result = parse_safetensors_header(&file_path);
    assert!(
        matches!(result, Err(SafetensorsHeaderError::HeaderNotJson(_))),
        "should reject invalid JSON: {:?}",
        result
    );
}

#[test]
fn should_reject_data_offsets_exceeding_payload() {
    let temp_dir = tempfile::tempdir().unwrap();
    let file_path = temp_dir.path().join("offsets_beyond_payload.safetensors");

    // Tensor claims data_offsets [0, 9999] but only 100 bytes of payload exist
    let header_json = serde_json::json!({
        "bad_tensor": {
            "dtype": "F32",
            "shape": [10],
            "data_offsets": [0, 9999]
        }
    });
    let header_bytes = serde_json::to_vec(&header_json).unwrap();
    let header_length = header_bytes.len() as u64;
    let mut file = std::fs::File::create(&file_path).unwrap();
    file.write_all(&header_length.to_le_bytes()).unwrap();
    file.write_all(&header_bytes).unwrap();
    file.write_all(&[0u8; 100]).unwrap(); // Only 100 bytes of payload

    let result = parse_safetensors_header(&file_path);
    assert!(
        matches!(
            result,
            Err(SafetensorsHeaderError::DataOffsetsOutsidePayload { .. })
        ),
        "should reject offsets beyond payload: {:?}",
        result
    );
}

#[test]
fn should_parse_bf16_uint32_float16_dtypes() {
    let temp_dir = tempfile::tempdir().unwrap();

    // Create a file with BF16, U32, and F16 tensors
    let header_json = serde_json::json!({
        "bf16_tensor": {
            "dtype": "BF16",
            "shape": [1, 4],
            "data_offsets": [0, 8]
        },
        "u32_tensor": {
            "dtype": "U32",
            "shape": [1, 4],
            "data_offsets": [8, 24]
        },
        "f16_tensor": {
            "dtype": "F16",
            "shape": [1, 4],
            "data_offsets": [24, 32]
        }
    });
    let header_bytes = serde_json::to_vec(&header_json).unwrap();
    let header_length = header_bytes.len() as u64;
    let payload_bytes = vec![0u8; 32]; // 8 + 16 + 8 = 32 bytes

    let file_path = temp_dir.path().join("multi_dtype.safetensors");
    let mut file = std::fs::File::create(&file_path).unwrap();
    file.write_all(&header_length.to_le_bytes()).unwrap();
    file.write_all(&header_bytes).unwrap();
    file.write_all(&payload_bytes).unwrap();

    let result = parse_safetensors_header(&file_path);
    assert!(
        result.is_ok(),
        "should parse multi-dtype file: {:?}",
        result
    );
    let header = result.unwrap();

    assert_eq!(header.tensor_entries.len(), 3);

    let bf16 = header.tensor_entry_for_name("bf16_tensor").unwrap();
    assert_eq!(bf16.dtype, SafetensorsDtype::BFloat16);
    assert_eq!(bf16.dtype.byte_width(), 2);

    let u32_entry = header.tensor_entry_for_name("u32_tensor").unwrap();
    assert_eq!(u32_entry.dtype, SafetensorsDtype::Uint32);
    assert_eq!(u32_entry.dtype.byte_width(), 4);

    let f16 = header.tensor_entry_for_name("f16_tensor").unwrap();
    assert_eq!(f16.dtype, SafetensorsDtype::Float16);
    assert_eq!(f16.dtype.byte_width(), 2);
}

#[test]
fn should_reject_unsupported_dtype() {
    let temp_dir = tempfile::tempdir().unwrap();
    let file_path = temp_dir.path().join("bad_dtype.safetensors");

    let header_json = serde_json::json!({
        "weird_tensor": {
            "dtype": "FP24",
            "shape": [1, 4],
            "data_offsets": [0, 12]
        }
    });
    let header_bytes = serde_json::to_vec(&header_json).unwrap();
    let header_length = header_bytes.len() as u64;
    let mut file = std::fs::File::create(&file_path).unwrap();
    file.write_all(&header_length.to_le_bytes()).unwrap();
    file.write_all(&header_bytes).unwrap();
    file.write_all(&[0u8; 12]).unwrap();

    let result = parse_safetensors_header(&file_path);
    assert!(
        matches!(result, Err(SafetensorsHeaderError::UnsupportedDtype { .. })),
        "should reject unsupported dtype: {:?}",
        result
    );
}

#[test]
fn should_reject_byte_count_mismatch() {
    let temp_dir = tempfile::tempdir().unwrap();
    let file_path = temp_dir.path().join("byte_count_mismatch.safetensors");

    // U32 tensor with shape [2, 3] should be 2*3*4=24 bytes,
    // but data_offsets says [0, 999]
    let header_json = serde_json::json!({
        "mismatched_tensor": {
            "dtype": "U32",
            "shape": [2, 3],
            "data_offsets": [0, 999]
        }
    });
    let header_bytes = serde_json::to_vec(&header_json).unwrap();
    let header_length = header_bytes.len() as u64;
    let payload_bytes = vec![0u8; 999]; // Enough payload for data_offsets

    let mut file = std::fs::File::create(&file_path).unwrap();
    file.write_all(&header_length.to_le_bytes()).unwrap();
    file.write_all(&header_bytes).unwrap();
    file.write_all(&payload_bytes).unwrap();

    let result = parse_safetensors_header(&file_path);
    assert!(
        matches!(
            result,
            Err(SafetensorsHeaderError::ByteCountMismatch { .. })
        ),
        "should reject byte count mismatch: {:?}",
        result
    );
}
