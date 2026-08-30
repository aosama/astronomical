use std::fs::File;
use std::io::Write;
use std::path::Path;

use astronomical_model_serving::{
    PersistentPromptCacheBlockError, PersistentPromptCacheBlockHeader,
    PersistentPromptCacheModelContract,
};
use serde_json::{Map, Value, json};

use crate::common::qwen3_5_moe::persistent_prompt_cache_model_contract;

#[test]
fn should_accept_a_contract_generated_sequence_state_header() {
    let persistent_prompt_cache_model_contract = persistent_prompt_cache_model_contract();
    let temporary_directory = tempfile::tempdir().expect("the test directory should exist");
    let sequence_state_file_path = temporary_directory.path().join("sequence.safetensors");
    write_contract_generated_file(
        &sequence_state_file_path,
        &persistent_prompt_cache_model_contract,
        false,
        None,
        None,
    );

    let sequence_state_file = File::open(&sequence_state_file_path)
        .expect("the generated sequence-state file should reopen");
    let sequence_state_header = PersistentPromptCacheBlockHeader::read_kv_block_from_file(
        &sequence_state_file,
        &sequence_state_file_path,
        &persistent_prompt_cache_model_contract,
    )
    .expect("the contract-generated sequence state should validate");

    assert_eq!(
        sequence_state_header.block_token_count(),
        persistent_prompt_cache_model_contract.block_token_count()
    );
    assert_eq!(
        sequence_state_header.tensor_count(),
        persistent_prompt_cache_model_contract
            .decoder_cache_layout()
            .sequence_tensor_count()
    );
    assert_eq!(
        sequence_state_header.storage_contract_fingerprint(),
        persistent_prompt_cache_model_contract.storage_contract_fingerprint_hex()
    );
}

#[test]
fn should_accept_a_contract_generated_boundary_state_header() {
    let persistent_prompt_cache_model_contract = persistent_prompt_cache_model_contract();
    let temporary_directory = tempfile::tempdir().expect("the test directory should exist");
    let boundary_state_file_path = temporary_directory.path().join("boundary.safetensors");
    write_contract_generated_file(
        &boundary_state_file_path,
        &persistent_prompt_cache_model_contract,
        true,
        None,
        None,
    );

    let boundary_state_file =
        File::open(&boundary_state_file_path).expect("the generated boundary file should reopen");
    let boundary_state_header =
        PersistentPromptCacheBlockHeader::read_recurrent_snapshot_from_file(
            &boundary_state_file,
            &boundary_state_file_path,
            &persistent_prompt_cache_model_contract,
        )
        .expect("the contract-generated boundary state should validate");

    assert_eq!(
        boundary_state_header.tensor_count(),
        persistent_prompt_cache_model_contract
            .decoder_cache_layout()
            .boundary_tensor_count()
    );
}

#[test]
fn should_reject_a_declared_dtype_that_differs_from_the_model_contract() {
    let persistent_prompt_cache_model_contract = persistent_prompt_cache_model_contract();
    let first_sequence_tensor_name = persistent_prompt_cache_model_contract
        .decoder_cache_layout()
        .sequence_tensor_layouts()
        .first()
        .expect("the frozen model should have sequence state")
        .persistent_tensor_name();
    let temporary_directory = tempfile::tempdir().expect("the test directory should exist");
    let sequence_state_file_path = temporary_directory.path().join("wrong-dtype.safetensors");
    write_contract_generated_file(
        &sequence_state_file_path,
        &persistent_prompt_cache_model_contract,
        false,
        Some(first_sequence_tensor_name.as_str()),
        None,
    );

    let sequence_state_file = File::open(&sequence_state_file_path)
        .expect("the generated wrong-dtype file should reopen");
    let rejection = PersistentPromptCacheBlockHeader::read_kv_block_from_file(
        &sequence_state_file,
        &sequence_state_file_path,
        &persistent_prompt_cache_model_contract,
    )
    .expect_err("a wrong execution dtype must be rejected");

    assert!(matches!(
        rejection,
        PersistentPromptCacheBlockError::TensorDtypeMismatch { .. }
    ));
}

#[test]
fn should_reject_a_missing_declared_tensor() {
    let persistent_prompt_cache_model_contract = persistent_prompt_cache_model_contract();
    let first_sequence_tensor_name = persistent_prompt_cache_model_contract
        .decoder_cache_layout()
        .sequence_tensor_layouts()
        .first()
        .expect("the frozen model should have sequence state")
        .persistent_tensor_name();
    let temporary_directory = tempfile::tempdir().expect("the test directory should exist");
    let sequence_state_file_path = temporary_directory
        .path()
        .join("missing-tensor.safetensors");
    write_contract_generated_file(
        &sequence_state_file_path,
        &persistent_prompt_cache_model_contract,
        false,
        None,
        Some(first_sequence_tensor_name.as_str()),
    );

    let sequence_state_file = File::open(&sequence_state_file_path)
        .expect("the generated missing-tensor file should reopen");
    let rejection = PersistentPromptCacheBlockHeader::read_kv_block_from_file(
        &sequence_state_file,
        &sequence_state_file_path,
        &persistent_prompt_cache_model_contract,
    );

    assert!(rejection.is_err());
}

#[test]
fn should_reject_a_foreign_storage_contract_fingerprint_before_payload_use() {
    let persistent_prompt_cache_model_contract = persistent_prompt_cache_model_contract();
    let temporary_directory = tempfile::tempdir().expect("the test directory should exist");
    let sequence_state_file_path = temporary_directory
        .path()
        .join("foreign-contract.safetensors");
    write_contract_generated_file(
        &sequence_state_file_path,
        &persistent_prompt_cache_model_contract,
        false,
        None,
        None,
    );
    replace_metadata_value(
        &sequence_state_file_path,
        "storage_contract_fingerprint",
        "00".repeat(32),
    );

    let sequence_state_file = File::open(&sequence_state_file_path)
        .expect("the generated foreign-contract file should reopen");
    let rejection = PersistentPromptCacheBlockHeader::read_kv_block_from_file(
        &sequence_state_file,
        &sequence_state_file_path,
        &persistent_prompt_cache_model_contract,
    );

    assert!(rejection.is_err());
}

#[test]
fn should_reject_a_previous_disposable_format_version() {
    let persistent_prompt_cache_model_contract = persistent_prompt_cache_model_contract();
    let temporary_directory = tempfile::tempdir().expect("the test directory should exist");
    let sequence_state_file_path = temporary_directory.path().join("old-format.safetensors");
    write_contract_generated_file(
        &sequence_state_file_path,
        &persistent_prompt_cache_model_contract,
        false,
        None,
        None,
    );
    replace_metadata_value(&sequence_state_file_path, "format_version", "11".to_owned());

    let sequence_state_file =
        File::open(&sequence_state_file_path).expect("the generated old-format file should reopen");
    let rejection = PersistentPromptCacheBlockHeader::read_kv_block_from_file(
        &sequence_state_file,
        &sequence_state_file_path,
        &persistent_prompt_cache_model_contract,
    )
    .expect_err("the old disposable format must be rejected");

    assert!(rejection.to_string().contains("expected 12"));
}

fn write_contract_generated_file(
    persistent_prompt_cache_file_path: &Path,
    persistent_prompt_cache_model_contract: &PersistentPromptCacheModelContract,
    should_write_boundary_state: bool,
    wrong_dtype_tensor_name: Option<&str>,
    omitted_tensor_name: Option<&str>,
) {
    let persisted_tensor_layouts = if should_write_boundary_state {
        persistent_prompt_cache_model_contract
            .decoder_cache_layout()
            .boundary_tensor_layouts()
    } else {
        persistent_prompt_cache_model_contract
            .decoder_cache_layout()
            .sequence_tensor_layouts()
    };
    let mut header_entries = Map::new();
    let mut payload_offset_bytes = 0_u64;
    for persisted_tensor_layout in persisted_tensor_layouts {
        let tensor_name = persisted_tensor_layout.persistent_tensor_name();
        if omitted_tensor_name == Some(tensor_name.as_str()) {
            continue;
        }
        let tensor_layout = persisted_tensor_layout.tensor_layout();
        let tensor_shape = tensor_layout
            .dimensions()
            .iter()
            .enumerate()
            .map(|(dimension_index, tensor_dimension)| {
                if Some(dimension_index) == tensor_layout.sequence_axis() {
                    persistent_prompt_cache_model_contract.block_token_count()
                } else {
                    *tensor_dimension
                }
            })
            .collect::<Vec<_>>();
        let tensor_element_size_bytes = tensor_layout.dtype().scalar_byte_count() as u64;
        let tensor_payload_byte_count = tensor_shape.iter().fold(
            tensor_element_size_bytes,
            |payload_byte_count, tensor_dimension| {
                payload_byte_count.saturating_mul(*tensor_dimension as u64)
            },
        );
        let payload_end_bytes = payload_offset_bytes.saturating_add(tensor_payload_byte_count);
        header_entries.insert(
            tensor_name.clone(),
            json!({
                // Pick a dtype different from the contract rather than assuming the frozen
                // sequence tensors use one particular precision. The header validator must
                // reject the mismatch before it considers the synthetic payload contents.
                "dtype": if wrong_dtype_tensor_name == Some(tensor_name.as_str()) {
                    match tensor_layout.dtype().safetensors_dtype_name() {
                        "BF16" => "F32",
                        _ => "BF16",
                    }
                } else { tensor_layout.dtype().safetensors_dtype_name() },
                "shape": tensor_shape,
                "data_offsets": [payload_offset_bytes, payload_end_bytes],
            }),
        );
        payload_offset_bytes = payload_end_bytes;
    }
    header_entries.insert(
        "__metadata__".to_owned(),
        json!({
            "format_version": "12",
            "block_token_count": persistent_prompt_cache_model_contract.block_token_count().to_string(),
            "storage_contract_fingerprint": persistent_prompt_cache_model_contract.storage_contract_fingerprint_hex(),
        }),
    );
    let header_bytes = serde_json::to_vec(&Value::Object(header_entries))
        .expect("the generated header should serialize");
    let mut persistent_prompt_cache_file =
        File::create(persistent_prompt_cache_file_path).expect("the generated file should open");
    persistent_prompt_cache_file
        .write_all(&(header_bytes.len() as u64).to_le_bytes())
        .expect("the generated header length should write");
    persistent_prompt_cache_file
        .write_all(&header_bytes)
        .expect("the generated header should write");
    persistent_prompt_cache_file
        .write_all(&vec![0_u8; payload_offset_bytes as usize])
        .expect("the generated payload should write");
}

fn replace_metadata_value(
    persistent_prompt_cache_file_path: &Path,
    metadata_name: &str,
    metadata_value: String,
) {
    let file_bytes =
        std::fs::read(persistent_prompt_cache_file_path).expect("the metadata fixture should read");
    let header_length_bytes = u64::from_le_bytes(
        file_bytes[..8]
            .try_into()
            .expect("the metadata fixture should have a length prefix"),
    ) as usize;
    let mut header_document: Value =
        serde_json::from_slice(&file_bytes[8..8 + header_length_bytes])
            .expect("the metadata fixture header should parse");
    header_document["__metadata__"][metadata_name] = Value::String(metadata_value);
    let header_bytes = serde_json::to_vec(&header_document)
        .expect("the modified metadata fixture should serialize");
    let payload_bytes = &file_bytes[8 + header_length_bytes..];
    let mut replacement_file = File::create(persistent_prompt_cache_file_path)
        .expect("the metadata fixture should reopen for replacement");
    replacement_file
        .write_all(&(header_bytes.len() as u64).to_le_bytes())
        .expect("the replacement header length should write");
    replacement_file
        .write_all(&header_bytes)
        .expect("the replacement header should write");
    replacement_file
        .write_all(payload_bytes)
        .expect("the replacement payload should write");
}
