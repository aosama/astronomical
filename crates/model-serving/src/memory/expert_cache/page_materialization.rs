//! Peak allocation estimates for pages materialized into a retained tensor form.

/// Estimates the peak bytes of one decoded expert page.
///
/// The packed source stays live while the page decodes and every decoded
/// tensor remains resident afterwards, so the peak is the loaded source
/// payload plus the persisted decoded outputs. The decoder itself is one
/// Metal launch per projection that reads packed words and writes decoded
/// weights with no intermediate arrays, so there is no workspace term.
///
/// Returns `None` when a tensor shape is empty, has a zero dimension, or any
/// element-count or byte-count arithmetic overflows.
#[must_use]
pub fn expert_page_peak_allocation_bytes(
    loaded_source_payload_bytes: u64,
    persisted_tensor_shapes: &[Vec<usize>],
    bytes_per_persisted_element: usize,
) -> Option<u64> {
    if bytes_per_persisted_element == 0 {
        return None;
    }
    let persisted_output_bytes =
        total_tensor_storage_bytes(persisted_tensor_shapes, bytes_per_persisted_element)?;
    loaded_source_payload_bytes.checked_add(persisted_output_bytes)
}

/// Calculates the packed source bytes for a page from one expert's payload.
#[must_use]
pub fn source_expert_page_payload_bytes(
    source_bytes_per_expert: u64,
    selected_expert_count: usize,
) -> Option<u64> {
    source_bytes_per_expert.checked_mul(u64::try_from(selected_expert_count).ok()?)
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
