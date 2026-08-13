//! Exact SafeTensors layout validation for one persisted SpecPrefill selection.

use std::path::Path;

use crate::safetensors::SafetensorsTensorView;

use super::{
    PERSISTENT_SPECULATIVE_PREFILL_SELECTION_TENSOR_NAME,
    PersistentSpeculativePrefillSelectionFileError,
};
use crate::persistent_cache::persistent_safetensors_header::PersistentSafetensorsHeader;

pub(super) fn validate_selection_tensor_layout(
    parsed_header: &PersistentSafetensorsHeader,
    selection_file_path: &Path,
) -> Result<(), PersistentSpeculativePrefillSelectionFileError> {
    if parsed_header.tensor_views.len() != 1 {
        return Err(invalid_layout(selection_file_path));
    }
    let selected_token_positions_tensor = parsed_header
        .tensor_views
        .get(PERSISTENT_SPECULATIVE_PREFILL_SELECTION_TENSOR_NAME)
        .ok_or_else(|| invalid_layout(selection_file_path))?;
    if selected_token_positions_tensor.dtype != "U32"
        || selected_token_positions_tensor.shape.len() != 1
        || selected_token_positions_tensor.shape[0] == 0
    {
        return Err(invalid_layout(selection_file_path));
    }
    validate_tensor_offsets(
        selected_token_positions_tensor,
        parsed_header,
        selection_file_path,
    )
}

pub(super) fn selected_token_position_count(
    parsed_header: &PersistentSafetensorsHeader,
    selection_file_path: &Path,
) -> Result<usize, PersistentSpeculativePrefillSelectionFileError> {
    parsed_header
        .tensor_views
        .get(PERSISTENT_SPECULATIVE_PREFILL_SELECTION_TENSOR_NAME)
        .map(|selected_token_positions_tensor| selected_token_positions_tensor.shape[0])
        .ok_or_else(|| invalid_layout(selection_file_path))
}

fn validate_tensor_offsets(
    selected_token_positions_tensor: &SafetensorsTensorView,
    parsed_header: &PersistentSafetensorsHeader,
    selection_file_path: &Path,
) -> Result<(), PersistentSpeculativePrefillSelectionFileError> {
    let [data_start_offset, data_end_offset] = selected_token_positions_tensor.data_offsets;
    let absolute_data_end_offset = parsed_header
        .data_section_start_bytes
        .checked_add(data_end_offset)
        .ok_or_else(|| outside_file(selection_file_path))?;
    if data_start_offset > data_end_offset
        || absolute_data_end_offset > parsed_header.file_size_bytes
    {
        return Err(outside_file(selection_file_path));
    }
    Ok(())
}

fn invalid_layout(selection_file_path: &Path) -> PersistentSpeculativePrefillSelectionFileError {
    PersistentSpeculativePrefillSelectionFileError::InvalidTensorLayout {
        selection_file_path: selection_file_path.to_path_buf(),
    }
}

fn outside_file(selection_file_path: &Path) -> PersistentSpeculativePrefillSelectionFileError {
    PersistentSpeculativePrefillSelectionFileError::TensorDataOutsideFile {
        selection_file_path: selection_file_path.to_path_buf(),
    }
}
