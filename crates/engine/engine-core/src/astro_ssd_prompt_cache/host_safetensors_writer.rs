//! Writes host-held decoder state as a safetensors file without MLX, so the
//! background `astro_ssd_prompt_cache` writer runs on a thread that never
//! touches MLX.
//!
//! The header is the one `model_contract_storage_geometry::exact_state_file_bytes`
//! predicts and MLX's own writer produced: compact JSON with sorted keys, the
//! `__metadata__` object, and each tensor's `data_offsets`, `dtype` and
//! `shape`, payloads laid out largest first and then by name. Matching it
//! keeps every staged file at the contract's exact size, so the quota and
//! the file-size checks are unchanged, and MLX's loader reads the files as
//! before. The file is synchronized before returning, as MLX's writer did, so
//! a published block survives a crash.

use std::fs::File;
use std::io::Write;

use super::disk_store_error::AstroSsdPromptCacheDiskStoreError;
use super::host_state_tensor::{AstroSsdPromptCacheHostTensor, AstroSsdPromptCacheHostTensors};

/// Elements converted to bytes per write: large enough that a 64 MiB tensor
/// is a few dozen writes, small enough that the conversion buffer stays a
/// few MiB beside the block's own host copy.
const ELEMENTS_PER_WRITE: usize = 1 << 20;

/// Writes `host_tensors` and `metadata_entries` to `output_file` and
/// synchronizes it, returning the bytes written.
///
/// # Errors
/// Fails when the header cannot be serialized or the file cannot be written
/// or synchronized.
pub fn write_host_safetensors(
    mut output_file: File,
    host_tensors: &AstroSsdPromptCacheHostTensors,
    metadata_entries: &[(&str, &str)],
) -> Result<u64, AstroSsdPromptCacheDiskStoreError> {
    let tensors_in_payload_order = tensors_in_payload_order(host_tensors);
    let header_bytes = safetensors_header_bytes(&tensors_in_payload_order, metadata_entries)?;
    let header_length_bytes = u64::try_from(header_bytes.len()).unwrap_or(u64::MAX);
    let mut written_byte_count = 0_u64;
    write_counted(
        &mut output_file,
        &header_length_bytes.to_le_bytes(),
        &mut written_byte_count,
    )?;
    write_counted(&mut output_file, &header_bytes, &mut written_byte_count)?;
    let mut payload_bytes = Vec::with_capacity(ELEMENTS_PER_WRITE * 4);
    for (_, host_tensor) in &tensors_in_payload_order {
        write_tensor_payload(
            &mut output_file,
            host_tensor,
            &mut payload_bytes,
            &mut written_byte_count,
        )?;
    }
    output_file.sync_all().map_err(|source| {
        AstroSsdPromptCacheDiskStoreError::WriteSafetensorsDescriptor { source }
    })?;
    Ok(written_byte_count)
}

/// Largest payload first, then by name: the order the contract's exact-size
/// prediction assigns offsets in.
fn tensors_in_payload_order(
    host_tensors: &AstroSsdPromptCacheHostTensors,
) -> Vec<(&str, &AstroSsdPromptCacheHostTensor)> {
    let mut ordered_tensors: Vec<(&str, &AstroSsdPromptCacheHostTensor)> = host_tensors
        .iter()
        .map(|(tensor_name, host_tensor)| (tensor_name.as_str(), host_tensor))
        .collect();
    ordered_tensors.sort_by(|(left_name, left_tensor), (right_name, right_tensor)| {
        right_tensor
            .payload_byte_count()
            .cmp(&left_tensor.payload_byte_count())
            .then_with(|| left_name.cmp(right_name))
    });
    ordered_tensors
}

fn safetensors_header_bytes(
    tensors_in_payload_order: &[(&str, &AstroSsdPromptCacheHostTensor)],
    metadata_entries: &[(&str, &str)],
) -> Result<Vec<u8>, AstroSsdPromptCacheDiskStoreError> {
    let mut metadata = serde_json::Map::new();
    for (metadata_key, metadata_value) in metadata_entries {
        metadata.insert(
            (*metadata_key).to_owned(),
            serde_json::Value::String((*metadata_value).to_owned()),
        );
    }
    let mut header = serde_json::Map::new();
    header.insert(
        "__metadata__".to_owned(),
        serde_json::Value::Object(metadata),
    );
    let mut payload_offset_bytes = 0_u64;
    for (tensor_name, host_tensor) in tensors_in_payload_order {
        let payload_end_bytes =
            payload_offset_bytes + u64::try_from(host_tensor.payload_byte_count()).unwrap_or(0);
        header.insert(
            (*tensor_name).to_owned(),
            serde_json::json!({
                "data_offsets": [payload_offset_bytes, payload_end_bytes],
                "dtype": host_tensor.safetensors_dtype_name,
                "shape": host_tensor.shape,
            }),
        );
        payload_offset_bytes = payload_end_bytes;
    }
    serde_json::to_vec(&header)
        .map_err(|source| AstroSsdPromptCacheDiskStoreError::SerializeSafetensorsHeader { source })
}

fn write_tensor_payload(
    output_file: &mut File,
    host_tensor: &AstroSsdPromptCacheHostTensor,
    payload_bytes: &mut Vec<u8>,
    written_byte_count: &mut u64,
) -> Result<(), AstroSsdPromptCacheDiskStoreError> {
    let element_count = host_tensor.element_count();
    let mut first_element = 0;
    while first_element < element_count {
        let end_element = (first_element + ELEMENTS_PER_WRITE).min(element_count);
        payload_bytes.clear();
        host_tensor.append_little_endian_bytes(first_element..end_element, payload_bytes);
        write_counted(output_file, payload_bytes, written_byte_count)?;
        first_element = end_element;
    }
    Ok(())
}

fn write_counted(
    output_file: &mut File,
    bytes: &[u8],
    written_byte_count: &mut u64,
) -> Result<(), AstroSsdPromptCacheDiskStoreError> {
    output_file.write_all(bytes).map_err(|source| {
        AstroSsdPromptCacheDiskStoreError::WriteSafetensorsDescriptor { source }
    })?;
    *written_byte_count += u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    Ok(())
}

#[cfg(test)]
#[path = "host_safetensors_writer_tests.rs"]
mod host_safetensors_writer_tests;
