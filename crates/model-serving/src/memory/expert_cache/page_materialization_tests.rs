use super::page_materialization::{
    expert_page_peak_allocation_bytes, source_expert_page_payload_bytes,
};

#[test]
fn should_sum_loaded_sources_and_persistent_decoded_outputs() {
    let decoded_tensor_shapes = vec![vec![2, 3], vec![1, 4]];

    // 10 source bytes plus (6 + 4) decoded elements × 2 bytes.
    let peak_allocation_bytes = expert_page_peak_allocation_bytes(10, &decoded_tensor_shapes, 2);

    assert_eq!(peak_allocation_bytes, Some(30));
}

#[test]
fn should_return_loaded_source_bytes_when_no_tensor_needs_materialization() {
    assert_eq!(expert_page_peak_allocation_bytes(123, &[], 2), Some(123));
}

#[test]
fn should_reject_empty_or_zero_sized_tensor_shapes() {
    assert_eq!(expert_page_peak_allocation_bytes(0, &[vec![]], 2), None);
    assert_eq!(
        expert_page_peak_allocation_bytes(0, &[vec![2, 0, 4]], 2),
        None
    );
}

#[test]
fn should_reject_zero_width_persisted_elements() {
    assert_eq!(expert_page_peak_allocation_bytes(0, &[vec![2, 3]], 0), None);
}

#[test]
fn should_reject_tensor_byte_counts_that_overflow() {
    assert_eq!(
        expert_page_peak_allocation_bytes(0, &[vec![usize::MAX, 2]], 2),
        None
    );
    assert_eq!(
        expert_page_peak_allocation_bytes(u64::MAX, &[vec![2, 3]], 2),
        None
    );
}

#[test]
fn should_calculate_source_page_bytes_with_checked_expert_multiplication() {
    assert_eq!(source_expert_page_payload_bytes(24, 3), Some(72));
    assert_eq!(source_expert_page_payload_bytes(u64::MAX, 2), None);
}
