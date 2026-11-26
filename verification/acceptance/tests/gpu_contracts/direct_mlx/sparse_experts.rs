use model_serving::{
    PerformanceAttribution, PerformanceOperation, SortedExpertAssignments,
    restore_expert_assignment_order, router_weighted_expert_inputs, sort_expert_assignments,
    sorted_expert_weighted_sum, sorted_expert_weighted_sum_kernel, unsorted_expert_weighted_sum,
};
use runtime_integration::{MlxMemoryLimits, MlxRuntime};

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_MLX_BUFFER_CACHE_LIMIT_BYTES,
};
use mlx_bridge::{MlxArray, MlxDtype};

use super::synthetic_values::{
    largest_relative_difference, swept_unit_interval_values, swept_values,
};

fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_MLX_BUFFER_CACHE_LIMIT_BYTES,
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
    let (sorted_expert_outputs, selected_scores) = sorted_outputs_and_selected_scores(&runtime);
    assert_sorted_reduction_skips_restoration(
        &runtime,
        &sorted_assignments,
        &sorted_expert_outputs,
        &selected_scores,
        &mut performance_attribution,
    );
    assert_restored_reduction_matches_sorted_reduction(
        &runtime,
        &sorted_assignments,
        &sorted_expert_outputs,
        &selected_scores,
        &selected_indices.shape(),
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

fn sorted_outputs_and_selected_scores(runtime: &MlxRuntime) -> (MlxArray, MlxArray) {
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
    (sorted_expert_outputs, selected_scores)
}

fn assert_sorted_reduction_skips_restoration(
    runtime: &MlxRuntime,
    sorted_assignments: &SortedExpertAssignments,
    sorted_expert_outputs: &MlxArray,
    selected_scores: &MlxArray,
    performance_attribution: &mut PerformanceAttribution,
) {
    let kernel = sorted_expert_weighted_sum_kernel().expect("kernel should initialize");
    let weighted = sorted_expert_weighted_sum(
        runtime,
        &kernel,
        sorted_expert_outputs,
        &sorted_assignments.inverse_order,
        selected_scores,
        performance_attribution,
    )
    .expect("sorted reduction should succeed");
    assert_f32_close(
        &weighted
            .to_vec_f32()
            .expect("weighted outputs should evaluate"),
        &[133.0, 134.0, 120.0, 121.0],
    );
}

fn assert_restored_reduction_matches_sorted_reduction(
    runtime: &MlxRuntime,
    sorted_assignments: &SortedExpertAssignments,
    sorted_expert_outputs: &MlxArray,
    selected_scores: &MlxArray,
    selected_indices_shape: &[i32],
) {
    let restored = restore_expert_assignment_order(
        runtime,
        sorted_expert_outputs,
        &sorted_assignments.inverse_order,
        selected_indices_shape,
    )
    .expect("restore remains available for the operations reference");
    let restored_weighted = unsorted_expert_weighted_sum(
        runtime,
        &restored,
        selected_scores,
        &mut PerformanceAttribution::disabled(),
    )
    .expect("restored unsorted reduction should match");
    assert_f32_close(
        &restored_weighted
            .to_vec_f32()
            .expect("restored reduction should evaluate"),
        &[133.0, 134.0, 120.0, 121.0],
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

#[tokio::test]
async fn should_reject_sorted_reduction_when_scores_do_not_match_assignment_geometry() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let sorted_expert_outputs = runtime
        .array_from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 1, 2])
        .expect("sorted outputs should be valid");
    let inverse_order = runtime
        .array_from_u32(&[0, 1], &[2])
        .expect("inverse order should be valid");
    let mismatched_scores = runtime
        .array_from_f32(&[0.2, 0.3, 0.5], &[1, 3])
        .expect("mismatched scores should be representable");
    let weighted_sum_kernel =
        sorted_expert_weighted_sum_kernel().expect("kernel should initialize");

    let reduction_error = sorted_expert_weighted_sum(
        &runtime,
        &weighted_sum_kernel,
        &sorted_expert_outputs,
        &inverse_order,
        &mismatched_scores,
        &mut PerformanceAttribution::disabled(),
    )
    .expect_err("mismatched routing scores must fail before kernel execution");

    assert!(
        reduction_error
            .to_string()
            .contains("must match selected scores")
    );
}

#[tokio::test]
async fn should_reject_sorted_reduction_when_scores_are_not_floating_point() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let sorted_expert_outputs = runtime
        .array_from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 1, 2])
        .expect("sorted outputs should be valid");
    let inverse_order = runtime
        .array_from_u32(&[0, 1], &[2])
        .expect("inverse order should be valid");
    let integer_scores = runtime
        .array_from_u32(&[1, 1], &[1, 2])
        .expect("integer scores should be representable");
    let weighted_sum_kernel =
        sorted_expert_weighted_sum_kernel().expect("kernel should initialize");

    sorted_expert_weighted_sum(
        &runtime,
        &weighted_sum_kernel,
        &sorted_expert_outputs,
        &inverse_order,
        &integer_scores,
        &mut PerformanceAttribution::disabled(),
    )
    .expect_err("integer routing scores must fail before kernel execution");
}

const SWEPT_TOKEN_COUNT: i32 = 16;
const SWEPT_EXPERTS_A_TOKEN: i32 = 8;
const SWEPT_EXPERT_COUNT: u64 = 256;
const SWEPT_HIDDEN_WIDTH: i32 = 512;
const SWEPT_OUTPUT_MAGNITUDES: [f32; 4] = [0.05, 0.5, 3.0, 20.0];
/// The kernel adds a token's eight weighted outputs in float32 in its own
/// order; a wrong score or a wrong expert moves the sum by its own size.
const LARGEST_SWEPT_RELATIVE_DIFFERENCE: f32 = 1e-5;

/// The sorted weighted-sum kernel against restoring the assignment order
/// and reducing with MLX's own multiply and sum, on pseudo-random expert
/// outputs near zero and far from it (issue #2599's lesson: patterned
/// inputs near zero are not a proof), at a 256-expert layer's top eight.
#[tokio::test]
async fn should_match_the_unsorted_reduction_to_rounding_on_outputs_far_from_zero() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let kernel = sorted_expert_weighted_sum_kernel().expect("kernel should initialize");
    let mut generator_state = 0xD1B5_4A32_D192_ED03_u64;
    let assignment_count = (SWEPT_TOKEN_COUNT * SWEPT_EXPERTS_A_TOKEN) as usize;
    for output_magnitude in SWEPT_OUTPUT_MAGNITUDES {
        let expert_ids = swept_unit_interval_values(assignment_count, &mut generator_state)
            .into_iter()
            .map(|unit_value| (f64::from(unit_value) * SWEPT_EXPERT_COUNT as f64) as u32)
            .collect::<Vec<_>>();
        let selected_indices = runtime
            .array_from_u32(&expert_ids, &[1, SWEPT_TOKEN_COUNT, SWEPT_EXPERTS_A_TOKEN])
            .expect("assignments should be valid");
        let selected_scores = runtime
            .array_from_f32(
                &swept_unit_interval_values(assignment_count, &mut generator_state),
                &[1, SWEPT_TOKEN_COUNT, SWEPT_EXPERTS_A_TOKEN],
            )
            .expect("scores should be valid");
        let sorted_expert_outputs = runtime
            .array_from_f32(
                &swept_values(
                    assignment_count * SWEPT_HIDDEN_WIDTH as usize,
                    output_magnitude,
                    &mut generator_state,
                ),
                &[assignment_count as i32, 1, SWEPT_HIDDEN_WIDTH],
            )
            .expect("sorted outputs should be valid");
        let (relative_difference, differing_element_count) = swept_reduction_difference(
            &runtime,
            &kernel,
            (&selected_indices, &selected_scores, &sorted_expert_outputs),
        );
        eprintln!(
            "[sorted-weighted-sum-sweep] magnitude={output_magnitude} relative_difference={relative_difference:e} differing_elements={differing_element_count}"
        );
        assert!(
            relative_difference <= LARGEST_SWEPT_RELATIVE_DIFFERENCE,
            "at magnitude {output_magnitude} the sorted reduction parts from the unsorted one by {relative_difference:e}"
        );
    }
}

/// The sorted kernel's sum against the unsorted reduction of the same
/// outputs put back in assignment order.
fn swept_reduction_difference(
    runtime: &MlxRuntime,
    kernel: &mlx_bridge::MlxMetalKernel,
    (selected_indices, selected_scores, sorted_expert_outputs): (&MlxArray, &MlxArray, &MlxArray),
) -> (f32, usize) {
    let hidden_states = runtime
        .zeros(
            &[1, SWEPT_TOKEN_COUNT, SWEPT_HIDDEN_WIDTH],
            MlxDtype::Float32,
        )
        .expect("hidden states should be valid");
    let expanded_states = runtime
        .expand_dims(&hidden_states, -2)
        .and_then(|states| runtime.expand_dims(&states, -3))
        .expect("gather dimensions should expand");
    let sorted_assignments = sort_expert_assignments(
        runtime,
        &expanded_states,
        selected_indices,
        &mut PerformanceAttribution::disabled(),
    )
    .expect("assignments should sort");
    let sorted_sum = sorted_expert_weighted_sum(
        runtime,
        kernel,
        sorted_expert_outputs,
        &sorted_assignments.inverse_order,
        selected_scores,
        &mut PerformanceAttribution::disabled(),
    )
    .expect("sorted reduction should succeed");
    let restored = restore_expert_assignment_order(
        runtime,
        sorted_expert_outputs,
        &sorted_assignments.inverse_order,
        &selected_indices.shape(),
    )
    .expect("the assignment order should restore");
    let unsorted_sum = unsorted_expert_weighted_sum(
        runtime,
        &restored,
        selected_scores,
        &mut PerformanceAttribution::disabled(),
    )
    .expect("unsorted reduction should succeed");
    largest_relative_difference(
        &sorted_sum.to_vec_f32().expect("the sorted sum evaluates"),
        &unsorted_sum
            .to_vec_f32()
            .expect("the unsorted sum evaluates"),
    )
}
