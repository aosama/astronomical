//! The fused combination against the two-step path it replaces: the sorted
//! weighted-sum kernel followed by MLX's compiled sigmoid-multiply-add graph.
//! Every equality case demands identical bits, not a tolerance.

use mlx_bridge::{MlxArray, MlxCompiledElementwiseGraphs, MlxDtype};
use runtime_integration::{MlxMemoryLimits, MlxRuntime};
use serial_test::file_serial;

use super::output_combination::combine_sparse_and_shared_experts;
use super::routing::{
    qwen3_5_moe_sorted_expert_weighted_sum, qwen3_5_moe_sorted_expert_weighted_sum_kernel,
};
use super::sorted_expert_shared_combination::{
    SortedExpertSharedCombinationInputs, can_fuse_sorted_expert_shared_combination,
    qwen3_5_sorted_expert_shared_combination, qwen3_5_sorted_expert_shared_combination_kernel,
};

/// Ornith-1.5-35B-A3B routes each token to eight experts.
const ORNITH_35B_A3B_SELECTED_EXPERT_COUNT: i32 = 8;
/// Ornith-1.5-35B-A3B's hidden width.
const ORNITH_35B_A3B_HIDDEN_DIMENSION: i32 = 2_048;

fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(512 * 1024 * 1024, 512 * 1024 * 1024)
            .expect("the fused combination test memory limits are valid"),
    )
    .expect("the direct MLX runtime initializes")
}

/// Owned inputs for one combination. `token_shape` is the router scores'
/// leading axes: `[1, tokens]` is what the resident MoE layer passes.
struct CombinationArrays {
    sorted_expert_outputs: MlxArray,
    inverse_order: MlxArray,
    selected_scores: MlxArray,
    shared_expert_output: MlxArray,
    shared_expert_gate_logits: MlxArray,
}

impl CombinationArrays {
    fn as_inputs(&self) -> SortedExpertSharedCombinationInputs<'_> {
        SortedExpertSharedCombinationInputs {
            sorted_expert_outputs: &self.sorted_expert_outputs,
            inverse_order: &self.inverse_order,
            selected_scores: &self.selected_scores,
            shared_expert_output: &self.shared_expert_output,
            shared_expert_gate_logits: &self.shared_expert_gate_logits,
        }
    }
}

/// A seeded permutation of `0..assignment_count`; a linear congruential walk
/// keeps the test deterministic without another random source.
fn shuffled_inverse_order(assignment_count: u32, seed: u64) -> Vec<u32> {
    let mut assignment_rows: Vec<u32> = (0..assignment_count).collect();
    let mut generator_state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    for swap_position in (1..assignment_rows.len()).rev() {
        generator_state = generator_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let swap_partner = (generator_state >> 33) as usize % (swap_position + 1);
        assignment_rows.swap(swap_position, swap_partner);
    }
    assignment_rows
}

fn seeded_normal(
    runtime: &MlxRuntime,
    shape: &[i32],
    dtype: MlxDtype,
    standard_deviation: f32,
    seed: u64,
) -> MlxArray {
    let random_key = runtime.random_key(seed).expect("a random key builds");
    runtime
        .random_normal(shape, dtype, 0.0, standard_deviation, &random_key)
        .expect("seeded normal inputs build")
}

fn with_trailing_axis(token_shape: &[i32], trailing_axis_length: i32) -> Vec<i32> {
    let mut extended_shape = token_shape.to_vec();
    extended_shape.push(trailing_axis_length);
    extended_shape
}

fn random_combination_arrays(
    runtime: &MlxRuntime,
    token_shape: &[i32],
    output_dimension: i32,
    seed: u64,
) -> CombinationArrays {
    let token_count: i32 = token_shape.iter().product();
    let assignment_count = token_count * ORNITH_35B_A3B_SELECTED_EXPERT_COUNT;
    let score_shape = with_trailing_axis(token_shape, ORNITH_35B_A3B_SELECTED_EXPERT_COUNT);
    let score_logits = seeded_normal(runtime, &score_shape, MlxDtype::Float32, 1.0, seed + 1);
    let inverse_order = shuffled_inverse_order(assignment_count as u32, seed);
    CombinationArrays {
        sorted_expert_outputs: seeded_normal(
            runtime,
            &[assignment_count, 1, output_dimension],
            MlxDtype::BFloat16,
            0.5,
            seed + 2,
        ),
        inverse_order: runtime
            .array_from_u32(&inverse_order, &[assignment_count])
            .expect("the inverse order builds"),
        selected_scores: runtime
            .softmax_axis(&score_logits, -1)
            .expect("router scores build"),
        shared_expert_output: seeded_normal(
            runtime,
            &with_trailing_axis(token_shape, output_dimension),
            MlxDtype::BFloat16,
            0.5,
            seed + 3,
        ),
        // A wide spread reaches saturated gates on both sides of zero.
        shared_expert_gate_logits: seeded_normal(
            runtime,
            &with_trailing_axis(token_shape, 1),
            MlxDtype::BFloat16,
            6.0,
            seed + 4,
        ),
    }
}

fn two_step_sparse_expert_output(runtime: &MlxRuntime, arrays: &CombinationArrays) -> MlxArray {
    let weighted_sum_kernel =
        qwen3_5_moe_sorted_expert_weighted_sum_kernel().expect("the weighted-sum kernel declares");
    qwen3_5_moe_sorted_expert_weighted_sum(
        runtime,
        &weighted_sum_kernel,
        &arrays.sorted_expert_outputs,
        &arrays.inverse_order,
        &arrays.selected_scores,
    )
    .expect("the sorted weighted sum runs")
}

fn two_step_compiled_combination(runtime: &MlxRuntime, arrays: &CombinationArrays) -> MlxArray {
    let compiled_elementwise_graphs =
        MlxCompiledElementwiseGraphs::new().expect("the compiled graphs build");
    runtime
        .apply_compiled_sparse_shared_expert_combination(
            &compiled_elementwise_graphs,
            &two_step_sparse_expert_output(runtime, arrays),
            &arrays.shared_expert_output,
            &arrays.shared_expert_gate_logits,
        )
        .expect("the compiled combination runs")
}

fn fused_combination(
    runtime: &MlxRuntime,
    arrays: &CombinationArrays,
) -> Result<MlxArray, runtime_integration::MlxRuntimeError> {
    let combination_kernel =
        qwen3_5_sorted_expert_shared_combination_kernel().expect("the fused kernel declares");
    qwen3_5_sorted_expert_shared_combination(runtime, &combination_kernel, &arrays.as_inputs())
}

/// Bfloat16 widens to Float32 exactly, so equal Float32 bit patterns mean
/// equal bfloat16 bits.
fn float32_bit_patterns(runtime: &MlxRuntime, combined_expert_output: &MlxArray) -> Vec<u32> {
    let widened_output = runtime
        .astype(combined_expert_output, MlxDtype::Float32)
        .expect("the output widens");
    let widened_values = runtime
        .array_to_vec_f32(&widened_output)
        .expect("the output reads back");
    let mut bit_patterns = Vec::with_capacity(widened_values.len());
    for widened_value in widened_values {
        bit_patterns.push(widened_value.to_bits());
    }
    bit_patterns
}

fn assert_fused_matches_two_step(runtime: &MlxRuntime, arrays: &CombinationArrays, case: &str) {
    let fused_output = fused_combination(runtime, arrays).expect("the fused combination runs");
    let two_step_output = two_step_compiled_combination(runtime, arrays);
    assert_eq!(fused_output.shape(), two_step_output.shape(), "{case}");
    assert_eq!(fused_output.dtype(), MlxDtype::BFloat16, "{case}");
    assert_eq!(
        float32_bit_patterns(runtime, &fused_output),
        float32_bit_patterns(runtime, &two_step_output),
        "{case}"
    );
}

fn assert_fused_combination_refused(runtime: &MlxRuntime, arrays: &CombinationArrays, case: &str) {
    assert!(fused_combination(runtime, arrays).is_err(), "{case}");
}

fn narrowed_to_float16(runtime: &MlxRuntime, bfloat16_activation: &MlxArray) -> MlxArray {
    runtime
        .astype(bfloat16_activation, MlxDtype::Float16)
        .expect("the activation narrows to Float16")
}

#[test]
#[file_serial]
fn should_match_the_two_step_combination_bit_for_bit_in_the_resident_layer_shape() {
    if test_harness::skip_without(
        test_harness::PlatformRequirement::MetalGpu,
        "should_match_the_two_step_combination_bit_for_bit_in_the_resident_layer_shape",
    ) {
        return;
    }
    let runtime = test_runtime();
    for seed in [3_u64, 17, 2_026] {
        let arrays =
            random_combination_arrays(&runtime, &[1, 64], ORNITH_35B_A3B_HIDDEN_DIMENSION, seed);
        assert!(can_fuse_sorted_expert_shared_combination(
            &arrays.as_inputs()
        ));
        assert_fused_matches_two_step(&runtime, &arrays, &format!("seed {seed}"));
    }
}

#[test]
#[file_serial]
fn should_match_the_uncompiled_combination_bit_for_bit() {
    if test_harness::skip_without(
        test_harness::PlatformRequirement::MetalGpu,
        "should_match_the_uncompiled_combination_bit_for_bit",
    ) {
        return;
    }
    let runtime = test_runtime();
    let arrays = random_combination_arrays(&runtime, &[32], ORNITH_35B_A3B_HIDDEN_DIMENSION, 41);
    let uncompiled_output = combine_sparse_and_shared_experts(
        &runtime,
        &two_step_sparse_expert_output(&runtime, &arrays),
        &arrays.shared_expert_output,
        &arrays.shared_expert_gate_logits,
    )
    .expect("the uncompiled combination runs");
    let fused_output = fused_combination(&runtime, &arrays).expect("the fused combination runs");
    assert_eq!(
        float32_bit_patterns(&runtime, &fused_output),
        float32_bit_patterns(&runtime, &uncompiled_output)
    );
}

#[test]
#[file_serial]
fn should_match_bit_for_bit_for_one_token_and_widths_off_the_threadgroup_grid() {
    if test_harness::skip_without(
        test_harness::PlatformRequirement::MetalGpu,
        "should_match_bit_for_bit_for_one_token_and_widths_off_the_threadgroup_grid",
    ) {
        return;
    }
    let runtime = test_runtime();
    for (token_count, output_dimension) in [(1, ORNITH_35B_A3B_HIDDEN_DIMENSION), (3, 200), (5, 1)]
    {
        let arrays = random_combination_arrays(&runtime, &[token_count], output_dimension, 7);
        assert_fused_matches_two_step(
            &runtime,
            &arrays,
            &format!("{token_count} tokens of width {output_dimension}"),
        );
    }
}

#[test]
#[file_serial]
fn should_match_bit_for_bit_at_zero_saturated_and_infinite_gate_logits() {
    if test_harness::skip_without(
        test_harness::PlatformRequirement::MetalGpu,
        "should_match_bit_for_bit_at_zero_saturated_and_infinite_gate_logits",
    ) {
        return;
    }
    let runtime = test_runtime();
    let mut arrays = random_combination_arrays(&runtime, &[8], 64, 11);
    let edge_gate_logits = runtime
        .array_from_f32(
            &[
                0.0,
                -0.0,
                40.0,
                -40.0,
                1.0e-3,
                -100.0,
                f32::INFINITY,
                f32::NEG_INFINITY,
            ],
            &[8, 1],
        )
        .expect("edge gate logits build");
    arrays.shared_expert_gate_logits = runtime
        .astype(&edge_gate_logits, MlxDtype::BFloat16)
        .expect("edge gate logits narrow");
    assert_fused_matches_two_step(&runtime, &arrays, "edge gate logits");
}

#[test]
#[file_serial]
fn should_match_bit_for_bit_with_bfloat16_router_scores() {
    if test_harness::skip_without(
        test_harness::PlatformRequirement::MetalGpu,
        "should_match_bit_for_bit_with_bfloat16_router_scores",
    ) {
        return;
    }
    let runtime = test_runtime();
    let mut arrays = random_combination_arrays(&runtime, &[1, 16], 256, 23);
    arrays.selected_scores = runtime
        .astype(&arrays.selected_scores, MlxDtype::BFloat16)
        .expect("router scores narrow");
    assert_fused_matches_two_step(&runtime, &arrays, "bfloat16 router scores");
}

#[test]
#[file_serial]
fn should_refuse_float16_activations_the_contract_is_not_proven_for() {
    let runtime = test_runtime();
    let mut arrays = random_combination_arrays(&runtime, &[4], 64, 5);
    arrays.sorted_expert_outputs = narrowed_to_float16(&runtime, &arrays.sorted_expert_outputs);
    arrays.shared_expert_output = narrowed_to_float16(&runtime, &arrays.shared_expert_output);
    arrays.shared_expert_gate_logits =
        narrowed_to_float16(&runtime, &arrays.shared_expert_gate_logits);
    assert!(!can_fuse_sorted_expert_shared_combination(
        &arrays.as_inputs()
    ));
    assert_fused_combination_refused(&runtime, &arrays, "Float16 activations");
}

#[test]
#[file_serial]
fn should_refuse_a_shared_expert_dtype_that_differs_from_the_routed_outputs() {
    let runtime = test_runtime();
    let mut arrays = random_combination_arrays(&runtime, &[4], 64, 5);
    arrays.shared_expert_output = runtime
        .astype(&arrays.shared_expert_output, MlxDtype::Float32)
        .expect("the shared output widens");
    assert!(!can_fuse_sorted_expert_shared_combination(
        &arrays.as_inputs()
    ));
    assert_fused_combination_refused(&runtime, &arrays, "Float32 shared output");
}

#[test]
#[file_serial]
fn should_refuse_a_gate_with_more_than_one_logit_per_token() {
    let runtime = test_runtime();
    let mut arrays = random_combination_arrays(&runtime, &[4], 64, 9);
    arrays.shared_expert_gate_logits = runtime
        .copy(&arrays.shared_expert_output)
        .expect("the shared output copies");
    assert_fused_combination_refused(&runtime, &arrays, "a gate as wide as the hidden axis");
}

#[test]
#[file_serial]
fn should_refuse_a_shared_expert_output_of_another_width() {
    let runtime = test_runtime();
    let mut arrays = random_combination_arrays(&runtime, &[4], 64, 13);
    arrays.shared_expert_output = seeded_normal(&runtime, &[4, 32], MlxDtype::BFloat16, 0.5, 13);
    assert_fused_combination_refused(
        &runtime,
        &arrays,
        "a 32-wide shared output for 64-wide experts",
    );
}

#[test]
#[file_serial]
fn should_refuse_an_inverse_order_missing_one_assignment() {
    let runtime = test_runtime();
    let mut arrays = random_combination_arrays(&runtime, &[4], 64, 19);
    let short_inverse_order = shuffled_inverse_order(31, 19);
    arrays.inverse_order = runtime
        .array_from_u32(&short_inverse_order, &[31])
        .expect("the short inverse order builds");
    assert_fused_combination_refused(&runtime, &arrays, "31 of 32 assignments");
}

#[test]
#[file_serial]
fn should_refuse_sorted_outputs_without_the_projection_axis() {
    let runtime = test_runtime();
    let mut arrays = random_combination_arrays(&runtime, &[4], 64, 29);
    arrays.sorted_expert_outputs = seeded_normal(&runtime, &[32, 64], MlxDtype::BFloat16, 0.5, 29);
    assert_fused_combination_refused(&runtime, &arrays, "[assignments, hidden] sorted outputs");
}

#[test]
#[file_serial]
fn should_refuse_router_scores_without_a_top_k_axis() {
    let runtime = test_runtime();
    let mut arrays = random_combination_arrays(&runtime, &[4], 64, 31);
    arrays.selected_scores = seeded_normal(&runtime, &[32], MlxDtype::Float32, 1.0, 31);
    assert_fused_combination_refused(&runtime, &arrays, "one-axis router scores");
}
