use astronomical_model_serving::{
    QuantizedExpertPageManifest, qwen3_5_moe_combine_experts, qwen3_5_moe_remap_expert_page_slots,
    qwen3_5_moe_restore_expert_assignment_order, qwen3_5_moe_route_experts,
    qwen3_5_moe_sort_expert_assignments, qwen3_5_moe_sorted_expert_weighted_sum,
    qwen3_5_moe_sorted_expert_weighted_sum_kernel,
};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

#[tokio::test]
async fn should_route_normalize_and_combine_selected_and_shared_experts() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the MoE test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    let router_logits = runtime
        .array_from_f32(&[0.0, 2.0_f32.ln(), 3.0_f32.ln(), 4.0_f32.ln()], &[1, 1, 4])
        .expect("the router logits should be valid");
    let expert_outputs = runtime
        .array_from_f32(&[0.0, 0.0, 0.0, 0.0, 7.0, 14.0, 14.0, 21.0], &[1, 1, 4, 2])
        .expect("the expert outputs should be valid");
    let shared_expert_output = runtime
        .array_from_f32(&[2.0, 4.0], &[1, 1, 2])
        .expect("the shared expert output should be valid");
    let shared_expert_gate_logits = runtime
        .array_from_f32(&[0.0], &[1, 1, 1])
        .expect("the shared expert gate should be valid");

    let (selected_indices, selected_scores) =
        qwen3_5_moe_route_experts(&runtime, &router_logits, 2, true)
            .expect("expert routing should build a valid graph");
    let expanded_indices = runtime
        .expand_dims(&selected_indices, -1)
        .expect("selected indices should gain an output dimension");
    let repeated_indices = runtime
        .repeat_axis(&expanded_indices, 2, -1)
        .expect("selected indices should cover each expert output dimension");
    let selected_expert_outputs = runtime
        .take_along_axis(&expert_outputs, &repeated_indices, 2)
        .expect("selected expert outputs should follow routing indices");
    let combined_output = qwen3_5_moe_combine_experts(
        &runtime,
        &selected_expert_outputs,
        &selected_scores,
        &shared_expert_output,
        &shared_expert_gate_logits,
    )
    .expect("expert combination should build a valid graph");

    let mut actual_selected_scores = selected_scores
        .to_vec_f32()
        .expect("selected scores should evaluate as float32");
    actual_selected_scores.sort_by(f32::total_cmp);
    assert_f32_close(&actual_selected_scores, &[3.0 / 7.0, 4.0 / 7.0]);
    assert_f32_close(
        &combined_output
            .to_vec_f32()
            .expect("combined experts should evaluate as float32"),
        &[12.0, 20.0],
    );
}

#[tokio::test]
async fn should_sort_expert_assignments_and_restore_their_original_order() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the MoE test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    let hidden_states = runtime
        .array_from_f32(&[10.0, 11.0, 20.0, 21.0], &[1, 2, 2])
        .expect("the hidden states should be valid");
    let expanded_states = runtime
        .expand_dims(&hidden_states, -2)
        .and_then(|states| runtime.expand_dims(&states, -3))
        .expect("the hidden states should gain gather-qmm dimensions");
    let selected_indices = runtime
        .array_from_u32(&[5, 0, 4, 1, 3, 2], &[1, 2, 3])
        .expect("the expert assignments should be valid");

    let (sorted_states, sorted_indices, inverse_order) =
        qwen3_5_moe_sort_expert_assignments(&runtime, &expanded_states, &selected_indices)
            .expect("expert assignments should sort into contiguous expert groups");

    assert_eq!(sorted_states.shape(), vec![6, 1, 2]);
    assert_f32_close(
        &sorted_states
            .to_vec_f32()
            .expect("the sorted states should evaluate as float32"),
        &[
            10.0, 11.0, 20.0, 21.0, 20.0, 21.0, 20.0, 21.0, 10.0, 11.0, 10.0, 11.0,
        ],
    );
    assert_eq!(
        runtime
            .copy_u32_values(&sorted_indices)
            .expect("the sorted expert indices should evaluate"),
        vec![0, 1, 2, 3, 4, 5]
    );
    assert_eq!(
        runtime
            .copy_u32_values(&inverse_order)
            .expect("the inverse expert order should evaluate"),
        vec![5, 0, 4, 1, 3, 2]
    );

    let sorted_expert_outputs = runtime
        .array_from_f32(
            &[
                100.0, 101.0, 110.0, 111.0, 120.0, 121.0, 130.0, 131.0, 140.0, 141.0, 150.0, 151.0,
            ],
            &[6, 1, 2],
        )
        .expect("the sorted expert outputs should be valid");
    let restored_outputs = qwen3_5_moe_restore_expert_assignment_order(
        &runtime,
        &sorted_expert_outputs,
        &inverse_order,
        &selected_indices.shape(),
    )
    .expect("sorted expert outputs should return to token assignment order");

    assert_eq!(restored_outputs.shape(), vec![1, 2, 3, 2]);
    assert_f32_close(
        &restored_outputs
            .to_vec_f32()
            .expect("the restored outputs should evaluate as float32"),
        &[
            150.0, 151.0, 100.0, 101.0, 140.0, 141.0, 110.0, 111.0, 130.0, 131.0, 120.0, 121.0,
        ],
    );
}

#[tokio::test]
async fn should_weight_sorted_expert_outputs_without_restoring_the_expanded_assignment_tensor() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the MoE test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    let weighted_sum_kernel = qwen3_5_moe_sorted_expert_weighted_sum_kernel()
        .expect("the sorted expert weighted-sum kernel should initialize");
    let sorted_expert_outputs = runtime
        .array_from_f32(
            &[
                100.0, 101.0, 110.0, 111.0, 120.0, 121.0, 130.0, 131.0, 140.0, 141.0, 150.0, 151.0,
            ],
            &[6, 1, 2],
        )
        .expect("the sorted expert outputs should be valid");
    let inverse_order = runtime
        .array_from_u32(&[5, 0, 4, 1, 3, 2], &[6])
        .expect("the inverse expert order should be valid");
    let selected_scores = runtime
        .array_from_f32(&[0.1, 0.2, 0.7, 0.25, 0.25, 0.5], &[1, 2, 3])
        .expect("the selected expert scores should be valid");

    let weighted_outputs = qwen3_5_moe_sorted_expert_weighted_sum(
        &runtime,
        &weighted_sum_kernel,
        &sorted_expert_outputs,
        &inverse_order,
        &selected_scores,
    )
    .expect("sorted expert outputs should combine directly into token outputs");

    assert_eq!(weighted_outputs.shape(), vec![1, 2, 2]);
    assert_f32_close(
        &weighted_outputs
            .to_vec_f32()
            .expect("the weighted expert outputs should evaluate as float32"),
        &[133.0, 134.0, 120.0, 121.0],
    );
}

#[tokio::test]
async fn should_remap_repeated_non_contiguous_global_expert_ids_on_the_gpu() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the page-slot remapping test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    let selected_global_expert_indices = runtime
        .array_from_u32(&[5, 1, 3, 5], &[1, 2, 2])
        .expect("the global expert assignments should be valid");
    let page_manifest = QuantizedExpertPageManifest {
        expert_ids: vec![1, 3, 5],
        page_slot_by_global_expert_id: vec![u32::MAX, 0, u32::MAX, 1, u32::MAX, 2],
        source_manifests: Vec::new(),
        payload_byte_count: 0,
    };

    let page_slot_indices = qwen3_5_moe_remap_expert_page_slots(
        &runtime,
        &selected_global_expert_indices,
        &[1, 3, 5],
        &page_manifest,
    )
    .expect("the GPU should build compact page-slot assignments");

    assert_eq!(page_slot_indices.shape(), vec![1, 2, 2]);
    assert_eq!(
        runtime
            .copy_u32_values(&page_slot_indices)
            .expect("the compact page-slot assignments should evaluate"),
        vec![2, 0, 1, 2]
    );
}

#[tokio::test]
async fn should_reject_page_slot_remapping_when_the_manifest_does_not_match_the_routed_experts() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the page-slot mismatch test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    let selected_global_expert_indices = runtime
        .array_from_u32(&[5, 1], &[1, 1, 2])
        .expect("the global expert assignments should be valid");
    let page_manifest = QuantizedExpertPageManifest {
        expert_ids: vec![1, 3, 5],
        page_slot_by_global_expert_id: vec![u32::MAX, 0, u32::MAX, 1, u32::MAX, 2],
        source_manifests: Vec::new(),
        payload_byte_count: 0,
    };

    let remapping_error = qwen3_5_moe_remap_expert_page_slots(
        &runtime,
        &selected_global_expert_indices,
        &[1, 5],
        &page_manifest,
    )
    .expect_err("a mismatched compact page must be rejected before GPU execution");

    assert!(
        remapping_error
            .to_string()
            .contains("routed expert IDs do not match the compact page manifest"),
        "unexpected remapping error: {remapping_error}"
    );
}

fn assert_f32_close(actual_values: &[f32], expected_values: &[f32]) {
    assert_eq!(actual_values.len(), expected_values.len());
    for (actual_value, expected_value) in actual_values.iter().zip(expected_values) {
        let comparison_scale = expected_value.abs().max(1.0);
        assert!(
            (*actual_value - *expected_value).abs() <= 1e-6 * comparison_scale,
            "expected {actual_value} to be close to {expected_value}"
        );
    }
}
