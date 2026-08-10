//! Exact on-disk size projection for model-bound prompt-cache artifacts.
//!
//! Quota admission happens before publication, so estimates cannot be vague.
//! These formulas reproduce the safetensors header and payload layout using the
//! same deterministic largest-tensor-first order as the native writer.

use crate::DecoderCachePersistedTensorLayout;

use super::block_format::PERSISTENT_PROMPT_CACHE_FORMAT_VERSION;
use super::model_contract_error::PersistentPromptCacheModelContractError;

pub(super) fn exact_state_file_bytes(
    block_token_count: usize,
    persisted_tensor_layouts: Vec<DecoderCachePersistedTensorLayout>,
) -> Result<u64, PersistentPromptCacheModelContractError> {
    if persisted_tensor_layouts.is_empty() {
        return Ok(0);
    }
    // The writer materializes the largest tensor first to bound peak workspace.
    // Sorting here the same way also makes JSON insertion and payload offsets
    // deterministic, which is required for exact byte prediction.
    let mut tensor_geometry = persisted_tensor_layouts
        .into_iter()
        .map(|persisted_tensor_layout| {
            let tensor_layout = persisted_tensor_layout.tensor_layout();
            let dimensions = tensor_layout
                .dimensions()
                .iter()
                .enumerate()
                .map(|(dimension_index, dimension)| {
                    if tensor_layout.sequence_axis() == Some(dimension_index) {
                        block_token_count
                    } else {
                        *dimension
                    }
                })
                .collect::<Vec<_>>();
            let payload_bytes = if tensor_layout.sequence_axis().is_some() {
                tensor_layout
                    .sequence_payload_byte_count_per_token()?
                    .checked_mul(block_token_count)
                    .ok_or(crate::DecoderCacheLayoutError::SequenceTensorPayloadByteCountOverflow)?
            } else {
                tensor_layout.fixed_payload_byte_count()?
            };
            Ok::<_, crate::DecoderCacheLayoutError>((
                persisted_tensor_layout.persistent_tensor_name(),
                tensor_layout.dtype().safetensors_dtype_name(),
                dimensions,
                payload_bytes,
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    tensor_geometry.sort_by(|left_tensor, right_tensor| {
        right_tensor
            .3
            .cmp(&left_tensor.3)
            .then_with(|| left_tensor.0.cmp(&right_tensor.0))
    });
    // Fingerprint contents do not affect length: every SHA-256 hexadecimal
    // encoding occupies exactly 64 bytes, so zeroes model the final header.
    let metadata = serde_json::json!({
        "block_token_count": block_token_count.to_string(),
        "format_version": PERSISTENT_PROMPT_CACHE_FORMAT_VERSION,
        "storage_contract_fingerprint": "0".repeat(64),
    });
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".to_owned(), metadata);
    let mut payload_offset_bytes = 0_u64;
    for (tensor_name, dtype_name, dimensions, payload_bytes) in tensor_geometry {
        let payload_bytes = u64::try_from(payload_bytes).unwrap_or(u64::MAX);
        let payload_end_bytes = payload_offset_bytes
            .checked_add(payload_bytes)
            .ok_or(PersistentPromptCacheModelContractError::CapturePayloadByteCountOverflow)?;
        header.insert(
            tensor_name,
            serde_json::json!({
                "data_offsets": [payload_offset_bytes, payload_end_bytes],
                "dtype": dtype_name,
                "shape": dimensions,
            }),
        );
        payload_offset_bytes = payload_end_bytes;
    }
    let header_bytes = serde_json::to_vec(&header)
        .map_err(PersistentPromptCacheModelContractError::SerializeStorageGeometry)?;
    8_u64
        .checked_add(u64::try_from(header_bytes.len()).unwrap_or(u64::MAX))
        .and_then(|header_section_bytes| header_section_bytes.checked_add(payload_offset_bytes))
        .ok_or(PersistentPromptCacheModelContractError::CapturePayloadByteCountOverflow)
}

pub(super) fn maximum_block_manifest_file_bytes(
    maximum_context_token_count: usize,
) -> Result<u64, PersistentPromptCacheModelContractError> {
    // Use the widest possible values and all optional fields so the transaction
    // can reject an unexpectedly larger manifest before consuming global quota.
    let maximum_block_index = u32::try_from(maximum_context_token_count).unwrap_or(u32::MAX);
    let manifest = serde_json::json!({
        "format_version": PERSISTENT_PROMPT_CACHE_FORMAT_VERSION,
        "block_hash": "0".repeat(64),
        "block_index": maximum_block_index,
        "parent_block_hash": "0".repeat(64),
        "storage_contract_fingerprint": "0".repeat(64),
        "has_sequence_state": true,
        "has_boundary_state": true,
    });
    serde_json::to_vec(&manifest)
        .map(|manifest_bytes| u64::try_from(manifest_bytes.len()).unwrap_or(u64::MAX))
        .map_err(PersistentPromptCacheModelContractError::SerializeStorageGeometry)
}
