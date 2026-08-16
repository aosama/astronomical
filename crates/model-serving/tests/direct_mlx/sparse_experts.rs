use astronomical_model_serving::{
    PerformanceAttribution, PerformanceOperation, gathered_indices_use_sorted_contract,
    restore_expert_assignment_order, router_weighted_expert_inputs, sort_expert_assignments,
    sorted_expert_weighted_sum, sorted_expert_weighted_sum_kernel, unsorted_expert_weighted_sum,
};
use astronomical_runtime_integration::{MlxDtype, MlxMemoryLimits, MlxRuntime};

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("sparse-expert test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

fn assert_f32_close(actual_values: &[f32], expected_values: &[f32]) {
    assert_eq!(actual_values.len(), expected_values.len());
    for (actual_value, expected_value) in actual_values.iter().zip(expected_values) {
        let comparison_scale = expected_value.abs().max(1.0);
        assert!(
            (*actual_value - *expected_value).abs() <= 1e-5 * comparison_scale,
            "expected {actual_value} to be close to {expected_value}"
        );
    }
}

#[tokio::test]
async fn should_preserve_bfloat16_expert_output_dtype_after_float32_weighted_reduction() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let selected_expert_outputs = runtime
        .array_from_f32(&[2.0, 4.0, 6.0, 8.0], &[1, 2, 2])
        .and_then(|outputs| runtime.astype(&outputs, MlxDtype::BFloat16))
        .expect("BF16 expert outputs should be valid");
    let float32_selected_scores = runtime
        .array_from_f32(&[0.25, 0.75], &[1, 2])
        .expect("Float32 router scores should be valid");

    let reduced_output = unsorted_expert_weighted_sum(
        &runtime,
        &selected_expert_outputs,
        &float32_selected_scores,
        &mut PerformanceAttribution::disabled(),
    )
    .expect("mixed-dtype expert reduction should succeed");

    assert_eq!(reduced_output.dtype(), MlxDtype::BFloat16);
    let float32_verification_output = runtime
        .astype(&reduced_output, MlxDtype::Float32)
        .expect("the BF16 output should cast for host verification");
    assert_f32_close(
        &float32_verification_output
            .to_vec_f32()
            .expect("reduced BF16 output should evaluate"),
        &[5.0, 7.0],
    );
}

#[tokio::test]
async fn should_preserve_bfloat16_activation_dtype_when_router_weights_expert_inputs() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let hidden_states = runtime
        .array_from_f32(&[2.0, 4.0], &[1, 1, 2])
        .and_then(|states| runtime.astype(&states, MlxDtype::BFloat16))
        .expect("BF16 hidden states should be valid");
    let float32_selected_scores = runtime
        .array_from_f32(&[0.25, 0.75], &[1, 1, 2])
        .expect("Float32 router scores should be valid");

    let weighted_inputs =
        router_weighted_expert_inputs(&runtime, &hidden_states, &float32_selected_scores)
            .expect("router-weighted expert inputs should succeed");

    assert_eq!(weighted_inputs.dtype(), MlxDtype::BFloat16);
    assert_eq!(weighted_inputs.shape(), vec![1, 1, 2, 1, 2]);
}

#[tokio::test]
async fn should_sort_assignments_and_reduce_without_restoring_the_expanded_tensor() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let hidden_states = runtime
        .array_from_f32(&[10.0, 11.0, 20.0, 21.0], &[1, 2, 2])
        .expect("hidden states should be valid");
    let expanded_states = runtime
        .expand_dims(&hidden_states, -2)
        .and_then(|states| runtime.expand_dims(&states, -3))
        .expect("gather dimensions should expand");
    let selected_indices = runtime
        .array_from_u32(&[5, 0, 4, 1, 3, 2], &[1, 2, 3])
        .expect("assignments should be valid");
    let mut performance_attribution = PerformanceAttribution::enabled();
    let sorted_assignments = sort_expert_assignments(
        &runtime,
        &expanded_states,
        &selected_indices,
        &mut performance_attribution,
    )
    .expect("assignments should sort");
    assert_eq!(
        runtime
            .copy_u32_values(&sorted_assignments.sorted_indices)
            .expect("sorted ids should evaluate"),
        vec![0, 1, 2, 3, 4, 5]
    );
    assert!(gathered_indices_use_sorted_contract(true));

    let sorted_expert_outputs = runtime
        .array_from_f32(
            &[
                100.0, 101.0, 110.0, 111.0, 120.0, 121.0, 130.0, 131.0, 140.0, 141.0, 150.0, 151.0,
            ],
            &[6, 1, 2],
        )
        .expect("sorted outputs should be valid");
    let selected_scores = runtime
        .array_from_f32(&[0.1, 0.2, 0.7, 0.25, 0.25, 0.5], &[1, 2, 3])
        .expect("scores should be valid");
    let kernel = sorted_expert_weighted_sum_kernel().expect("kernel should initialize");
    let weighted = sorted_expert_weighted_sum(
        &runtime,
        &kernel,
        &sorted_expert_outputs,
        &sorted_assignments.inverse_order,
        &selected_scores,
        &mut performance_attribution,
    )
    .expect("sorted reduction should succeed");
    assert_f32_close(
        &weighted
            .to_vec_f32()
            .expect("weighted outputs should evaluate"),
        &[133.0, 134.0, 120.0, 121.0],
    );
    let restored = restore_expert_assignment_order(
        &runtime,
        &sorted_expert_outputs,
        &sorted_assignments.inverse_order,
        &selected_indices.shape(),
    )
    .expect("restore remains available for the operations reference");
    let restored_weighted = unsorted_expert_weighted_sum(
        &runtime,
        &restored,
        &selected_scores,
        &mut PerformanceAttribution::disabled(),
    )
    .expect("restored unsorted reduction should match");
    assert_f32_close(
        &restored_weighted
            .to_vec_f32()
            .expect("restored reduction should evaluate"),
        &[133.0, 134.0, 120.0, 121.0],
    );
    assert!(
        performance_attribution
            .operation_measurement(PerformanceOperation::ExpertAssignmentPreparation)
            .is_some()
    );
    assert!(
        performance_attribution
            .operation_measurement(PerformanceOperation::ExpertWeightedReduction)
            .is_some()
    );
}

#[tokio::test]
async fn should_complete_empty_assignments_without_a_family_branch() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let expanded_states = runtime
        .array_from_f32(&[1.0, 2.0], &[1, 1, 1, 2])
        .expect("one token of hidden state should be valid");
    let selected_indices = runtime
        .array_from_u32(&[], &[1, 0])
        .expect("zero experts per token should be representable");
    let mut performance_attribution = PerformanceAttribution::disabled();
    let sorted_assignments = sort_expert_assignments(
        &runtime,
        &expanded_states,
        &selected_indices,
        &mut performance_attribution,
    )
    .expect("empty assignments should sort");
    assert_eq!(sorted_assignments.sorted_states.shape(), vec![0, 1, 2]);
    let selected_scores = runtime
        .array_from_f32(&[], &[1, 0])
        .expect("empty scores should be valid");
    let reduced = unsorted_expert_weighted_sum(
        &runtime,
        &runtime
            .array_from_f32(&[], &[1, 0, 2])
            .expect("empty outputs should be valid"),
        &selected_scores,
        &mut performance_attribution,
    )
    .expect("empty reduction should succeed");
    assert_eq!(reduced.shape(), vec![1, 2]);
    assert_eq!(
        reduced
            .to_vec_f32()
            .expect("empty reduction should evaluate"),
        vec![0.0, 0.0]
    );
}

#[tokio::test]
async fn should_preserve_activation_dtype_for_empty_sorted_assignments() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let expanded_states = runtime
        .array_from_f32(&[], &[1, 0, 1, 2])
        .and_then(|states| runtime.astype(&states, MlxDtype::BFloat16))
        .expect("empty BF16 hidden states should be valid");
    let selected_indices = runtime
        .array_from_u32(&[], &[1, 0])
        .expect("empty assignments should be valid");

    let sorted_assignments = sort_expert_assignments(
        &runtime,
        &expanded_states,
        &selected_indices,
        &mut PerformanceAttribution::disabled(),
    )
    .expect("empty assignments should retain activation dtype");

    assert_eq!(sorted_assignments.sorted_states.dtype(), MlxDtype::BFloat16);
}

#[tokio::test]
async fn should_accept_named_xs_and_s_assignment_geometries() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let named_rows = [("xs_routed", 8_i32, 4_i32), ("s_routed", 10, 4)];
    for (row_name, experts_per_token, hidden_dimension) in named_rows {
        let token_count = 2_i32;
        let hidden_values = vec![1.0_f32; (token_count * hidden_dimension) as usize];
        let hidden_states = runtime
            .array_from_f32(&hidden_values, &[1, token_count, hidden_dimension])
            .unwrap_or_else(|_| panic!("{row_name} hidden states"));
        let expanded_states = runtime
            .expand_dims(&hidden_states, -2)
            .and_then(|states| runtime.expand_dims(&states, -3))
            .unwrap_or_else(|_| panic!("{row_name} expand"));
        let assignment_ids = (0..(token_count * experts_per_token) as u32).collect::<Vec<_>>();
        let selected_indices = runtime
            .array_from_u32(&assignment_ids, &[1, token_count, experts_per_token])
            .unwrap_or_else(|_| panic!("{row_name} assignments"));
        let mut performance_attribution = PerformanceAttribution::disabled();
        let sorted_assignments = sort_expert_assignments(
            &runtime,
            &expanded_states,
            &selected_indices,
            &mut performance_attribution,
        )
        .unwrap_or_else(|_| panic!("{row_name} should sort"));
        assert_eq!(
            sorted_assignments.sorted_indices.shape()[0],
            token_count * experts_per_token,
            "{row_name}"
        );
    }
}
