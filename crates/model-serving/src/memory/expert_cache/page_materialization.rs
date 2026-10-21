//! Peak allocation estimates for pages materialized into a retained tensor form.

/// Estimates the peak bytes for a sequential page materialization pass.
///
/// Every decoded tensor remains resident after its conversion, while only the
/// current tensor's temporary decode workspace is live. The decoder supplies
/// the maximum number and element width of simultaneous full-sized temporary
/// arrays and the shapes and widths of its smaller index tensors; this owner
/// converts those facts to bytes and adds both workspaces to persistent output
/// storage.
///
/// `decode_workspace_tensor_shapes` describes one staged decoder block's
/// shapes, not the page: a decoder that walks the page in expert blocks keeps
/// only one block of temporaries live at a time, so the workspace follows the
/// block size. `output_assembly_copy_count` covers the final assembly step
/// for multi-block decodes: while block outputs concatenate into the page
/// tensor, the largest projection's output exists both as block inputs and as
/// the page being written, so a count of two prices that transient second
/// copy; single-block decodes never assemble and pass one.
///
/// Returns `None` when a tensor shape is empty, has a zero dimension, or any
/// element-count or byte-count arithmetic overflows.
#[must_use]
pub fn expert_page_peak_allocation_bytes(
    loaded_source_payload_bytes: u64,
    persisted_tensor_shapes: &[Vec<usize>],
    decode_workspace_tensor_shapes: &[Vec<usize>],
    output_assembly_copy_count: u64,
    bytes_per_persisted_element: usize,
    maximum_simultaneous_temporary_array_count: usize,
    bytes_per_temporary_element: usize,
    decoder_index_tensor_shapes: &[Vec<usize>],
    simultaneous_decoder_index_array_count: usize,
    bytes_per_decoder_index_element: usize,
) -> Option<u64> {
    if bytes_per_persisted_element == 0
        || maximum_simultaneous_temporary_array_count == 0
        || bytes_per_temporary_element == 0
        || simultaneous_decoder_index_array_count == 0
        || bytes_per_decoder_index_element == 0
        || output_assembly_copy_count == 0
    {
        return None;
    }

    let temporary_bytes_per_element =
        maximum_simultaneous_temporary_array_count.checked_mul(bytes_per_temporary_element)?;
    let decoder_index_bytes_per_element =
        simultaneous_decoder_index_array_count.checked_mul(bytes_per_decoder_index_element)?;
    let persisted_output_bytes =
        total_tensor_storage_bytes(persisted_tensor_shapes, bytes_per_persisted_element)?;
    let largest_persisted_tensor_bytes =
        maximum_tensor_storage_bytes(persisted_tensor_shapes, bytes_per_persisted_element)?;
    // Only the second and further copies of the largest projection are extra:
    // the first copy is already part of the persisted total above.
    let output_assembly_transient_bytes =
        largest_persisted_tensor_bytes.checked_mul(output_assembly_copy_count.saturating_sub(1))?;
    let largest_single_tensor_workspace_bytes =
        maximum_tensor_storage_bytes(decode_workspace_tensor_shapes, temporary_bytes_per_element)?;
    let largest_decoder_index_workspace_bytes =
        maximum_tensor_storage_bytes(decoder_index_tensor_shapes, decoder_index_bytes_per_element)?;
    loaded_source_payload_bytes
        .checked_add(persisted_output_bytes)?
        .checked_add(output_assembly_transient_bytes)?
        .checked_add(largest_single_tensor_workspace_bytes)?
        .checked_add(largest_decoder_index_workspace_bytes)
}

/// Calculates the packed source bytes for a page from one expert's payload.
#[must_use]
pub fn source_expert_page_payload_bytes(
    source_bytes_per_expert: u64,
    selected_expert_count: usize,
) -> Option<u64> {
    source_bytes_per_expert.checked_mul(u64::try_from(selected_expert_count).ok()?)
}

/// Calculates retained page bytes after selected packed tensors become dense.
///
/// `source_tensor_bytes_per_expert` describes every source tensor in one
/// expert; `replaced_source_tensor_bytes_per_expert` is the subset released
/// after decode. Dense output shapes describe the replacement storage. This
/// keeps expert encoding knowledge with the family while this module owns the
/// subtraction, multiplication, and sum.
#[must_use]
pub fn retained_expert_page_payload_bytes(
    source_tensor_bytes_per_expert: &[usize],
    replaced_source_tensor_bytes_per_expert: &[usize],
    selected_expert_count: usize,
    decoded_tensor_shapes: &[Vec<usize>],
    bytes_per_decoded_element: usize,
) -> Option<u64> {
    if selected_expert_count == 0 || bytes_per_decoded_element == 0 {
        return None;
    }
    let source_bytes_per_expert = source_tensor_bytes_per_expert.iter().try_fold(
        0_u64,
        |source_bytes, tensor_source_bytes| {
            source_bytes.checked_add(u64::try_from(*tensor_source_bytes).ok()?)
        },
    )?;
    let replaced_source_bytes_per_expert = replaced_source_tensor_bytes_per_expert
        .iter()
        .try_fold(0_u64, |replaced_source_bytes, tensor_source_bytes| {
            replaced_source_bytes.checked_add(u64::try_from(*tensor_source_bytes).ok()?)
        })?;
    let unreplaced_source_bytes_per_expert =
        source_bytes_per_expert.checked_sub(replaced_source_bytes_per_expert)?;
    let unreplaced_source_payload_bytes = unreplaced_source_bytes_per_expert
        .checked_mul(u64::try_from(selected_expert_count).ok()?)?;
    let decoded_output_bytes =
        total_tensor_storage_bytes(decoded_tensor_shapes, bytes_per_decoded_element)?;
    unreplaced_source_payload_bytes.checked_add(decoded_output_bytes)
}

fn total_tensor_storage_bytes(
    tensor_shapes: &[Vec<usize>],
    bytes_per_element: usize,
) -> Option<u64> {
    tensor_shapes
        .iter()
        .try_fold(0_u64, |total_bytes, tensor_shape| {
            let tensor_element_count = tensor_element_count(tensor_shape)?;
            let tensor_storage_bytes =
                tensor_element_count.checked_mul(u64::try_from(bytes_per_element).ok()?)?;
            total_bytes.checked_add(tensor_storage_bytes)
        })
}

fn maximum_tensor_storage_bytes(
    tensor_shapes: &[Vec<usize>],
    bytes_per_element: usize,
) -> Option<u64> {
    tensor_shapes
        .iter()
        .try_fold(0_u64, |maximum_bytes, tensor_shape| {
            let tensor_element_count = tensor_element_count(tensor_shape)?;
            let tensor_storage_bytes =
                tensor_element_count.checked_mul(u64::try_from(bytes_per_element).ok()?)?;
            Some(maximum_bytes.max(tensor_storage_bytes))
        })
}

fn tensor_element_count(tensor_shape: &[usize]) -> Option<u64> {
    if tensor_shape.is_empty() || tensor_shape.contains(&0) {
        return None;
    }
    tensor_shape
        .iter()
        .try_fold(1_u64, |element_count, dimension| {
            element_count.checked_mul(u64::try_from(*dimension).ok()?)
        })
}
