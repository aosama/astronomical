use super::page_materialization::{
    expert_page_peak_allocation_bytes, retained_expert_page_payload_bytes,
    source_expert_page_payload_bytes,
};

#[test]
fn should_sum_loaded_sources_persistent_outputs_and_largest_sequential_workspace() {
    let decoded_tensor_shapes = vec![vec![2, 3], vec![1, 4]];
    let decoder_index_tensor_shapes = vec![vec![3], vec![4]];

    let peak_allocation_bytes = expert_page_peak_allocation_bytes(
        10,
        &decoded_tensor_shapes,
        &decoded_tensor_shapes,
        1,
        2,
        3,
        4,
        &decoder_index_tensor_shapes,
        9,
        4,
    );

    assert_eq!(peak_allocation_bytes, Some(246));
}

#[test]
fn should_follow_block_workspace_shapes_and_charge_one_assembly_copy_for_multi_block_pages() {
    let decoded_tensor_shapes = vec![vec![2, 3], vec![1, 4]];
    let block_workspace_tensor_shapes = vec![vec![1, 3], vec![1, 4]];
    let decoder_index_tensor_shapes = vec![vec![3], vec![4]];

    // Persisted outputs stay page-sized (10 elements × 2 bytes = 20), the
    // workspace follows one block (largest block 4 elements × 12 bytes = 48),
    // and assembly adds one extra copy of the largest persisted output (12).
    let peak_allocation_bytes = expert_page_peak_allocation_bytes(
        10,
        &decoded_tensor_shapes,
        &block_workspace_tensor_shapes,
        2,
        2,
        3,
        4,
        &decoder_index_tensor_shapes,
        9,
        4,
    );

    assert_eq!(peak_allocation_bytes, Some(234));
}

#[test]
fn should_return_loaded_source_bytes_when_no_tensor_needs_materialization() {
    assert_eq!(
        expert_page_peak_allocation_bytes(123, &[], &[], 1, 2, 3, 4, &[], 6, 4),
        Some(123)
    );
}

#[test]
fn should_reject_empty_or_zero_sized_tensor_shapes() {
    assert_eq!(
        expert_page_peak_allocation_bytes(0, &[vec![]], &[], 1, 2, 3, 4, &[], 6, 4),
        None
    );
    assert_eq!(
        expert_page_peak_allocation_bytes(
            0,
            &[vec![2, 0, 4]],
            &[],
            1,
            2,
            3,
            4,
            &[],
            6,
            4
        ),
        None
    );
}

#[test]
fn should_reject_tensor_byte_counts_that_overflow() {
    assert_eq!(
        expert_page_peak_allocation_bytes(
            0,
            &[vec![usize::MAX, 2]],
            &[],
            1,
            2,
            3,
            4,
            &[],
            6,
            4
        ),
        None
    );
}

#[test]
fn should_reject_temporary_workspace_byte_counts_that_overflow() {
    assert_eq!(
        expert_page_peak_allocation_bytes(
            0,
            &[vec![2, 3]],
            &[vec![2, 3]],
            1,
            2,
            usize::MAX,
            4,
            &[],
            6,
            4
        ),
        None
    );
}

#[test]
fn should_reject_zero_output_assembly_copy_counts() {
    assert_eq!(
        expert_page_peak_allocation_bytes(0, &[vec![2, 3]], &[vec![2, 3]], 0, 2, 3, 4, &[], 6, 4),
        None
    );
}

#[test]
fn should_count_decoder_auxiliary_arrays_even_for_a_single_output_row() {
    let decoded_tensor_shapes = vec![vec![1, 1, 32]];
    // The reserved auxiliary shape is the larger of the input width (32) and
    // the page's output-row count (1 × 1), so 32 here.
    let decoder_index_tensor_shapes = vec![vec![32]];

    let peak_allocation_bytes = expert_page_peak_allocation_bytes(
        0,
        &decoded_tensor_shapes,
        &decoded_tensor_shapes,
        1,
        2,
        3,
        4,
        &decoder_index_tensor_shapes,
        9,
        4,
    );

    assert_eq!(peak_allocation_bytes, Some(1_600));
}

#[test]
fn should_reject_decoder_index_workspace_byte_counts_that_overflow() {
    assert_eq!(
        expert_page_peak_allocation_bytes(
            0,
            &[vec![2, 3]],
            &[vec![2, 3]],
            1,
            2,
            3,
            4,
            &[vec![3]],
            usize::MAX,
            4
        ),
        None
    );
}

#[test]
fn should_replace_packed_expert_sources_with_decoded_output_storage() {
    let source_tensor_bytes_per_expert = [10, 20];
    let replaced_codebook_source_bytes_per_expert = [10];
    let decoded_tensor_shapes = vec![vec![2, 3]];

    let retained_page_payload_bytes = retained_expert_page_payload_bytes(
        &source_tensor_bytes_per_expert,
        &replaced_codebook_source_bytes_per_expert,
        2,
        &decoded_tensor_shapes,
        2,
    );

    assert_eq!(retained_page_payload_bytes, Some(52));
}

#[test]
fn should_reject_replaced_source_bytes_above_the_page_source_bytes() {
    assert_eq!(
        retained_expert_page_payload_bytes(&[10], &[11], 1, &[vec![2, 3]], 2),
        None
    );
}

#[test]
fn should_calculate_source_page_bytes_with_checked_expert_multiplication() {
    assert_eq!(source_expert_page_payload_bytes(24, 3), Some(72));
    assert_eq!(source_expert_page_payload_bytes(u64::MAX, 2), None);
}
