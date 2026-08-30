//! End-to-end operations references for family-neutral sparse-expert projection.
//!
//! These tests deliberately avoid any Qwen or Laguna router. They start from the
//! canonical arrays that every family must produce, exercise the real MLX GPU
//! operations, and compare sorted, unsorted, dense, and affine paths. This proves
//! that the neutral layer shares math without silently sharing family policy.

use astronomical_model_serving::{
    ExpertAssignmentOrder, PerformanceAttribution, PerformanceOperation, StackedExpertProjection,
    gather_expert_projection, restore_expert_assignment_order, sort_expert_assignments,
    sorted_expert_weighted_sum, sorted_expert_weighted_sum_kernel, unsorted_expert_weighted_sum,
};
use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxMemoryLimits, MlxRuntime};

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

fn test_runtime() -> MlxRuntime {
    // Use the same bounded runtime configuration as the rest of direct-MLX tests.
    // A test must not inherit whatever memory limits happen to exist on one Mac.
    MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("sparse-expert projection test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

fn assert_f32_close(actual_values: &[f32], expected_values: &[f32], tolerance: f32) {
    // Relative tolerance handles both values near zero and larger accumulated
    // matrix products. `max(1)` prevents a zero reference from allowing no error.
    assert_eq!(actual_values.len(), expected_values.len());
    for (actual_value, expected_value) in actual_values.iter().zip(expected_values) {
        let comparison_scale = expected_value.abs().max(1.0);
        assert!(
            (*actual_value - *expected_value).abs() <= tolerance * comparison_scale,
            "expected {actual_value} to be close to {expected_value}"
        );
    }
}

fn values_as_float32(runtime: &MlxRuntime, array: &MlxArray) -> Vec<f32> {
    // Float16 and BFloat16 are production cases, but host comparison is simplest
    // in f32. This cast is test-only and occurs after the operation under test.
    runtime
        .astype(array, MlxDtype::Float32)
        .and_then(|float32_array| float32_array.to_vec_f32())
        .expect("the operation result should evaluate as float32")
}

#[tokio::test]
async fn should_match_dense_operations_reference_across_generic_assignment_rows_and_dtypes() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    // Every tuple is (diagnostic name, expert count, token count, top-K, output
    // width). The rows grow independently so the implementation cannot assume a
    // particular model geometry or that one token always means one assignment.
    let matrix_rows = [
        ("one_token", 1_i32, 1_i32, 1_i32, 3_i32),
        ("two_tokens", 3, 2, 2, 5),
        ("multi_token", 5, 3, 4, 7),
    ];
    let activation_dtypes = [MlxDtype::Float16, MlxDtype::BFloat16, MlxDtype::Float32];
    // Production retains one reduction kernel with the model. Do the same here
    // instead of rebuilding its owner inside every matrix cell.
    let sorted_reduction_kernel =
        sorted_expert_weighted_sum_kernel().expect("the sorted reduction kernel should build");

    for (row_name, expert_count, token_count, top_k, output_width) in matrix_rows {
        let input_width = 4_i32;
        // Non-constant activations and weights make incorrect expert selection,
        // token selection, or axis order observable in the resulting numbers.
        let hidden_values = (0..token_count * input_width)
            .map(|value_index| 0.25 + value_index as f32 * 0.125)
            .collect::<Vec<_>>();
        let dense_weight_values = (0..expert_count * input_width * output_width)
            .map(|value_index| ((value_index % 11) as f32 - 5.0) * 0.0625)
            .collect::<Vec<_>>();
        // The modular stride creates valid but generally non-sorted expert IDs.
        // That gives the original-order path real scattered assignments to test.
        let selected_expert_ids = (0..token_count * top_k)
            .map(|assignment_index| {
                u32::try_from((assignment_index * 2 + 1) % expert_count)
                    .expect("expert id should fit u32")
            })
            .collect::<Vec<_>>();

        for activation_dtype in activation_dtypes {
            // Shape legend:
            // hidden_states       = [batch=1, tokens, input]
            // selected_indices    = [batch=1, tokens, top_k]
            // transposed_weights  = [experts, input, output]
            let hidden_states = runtime
                .array_from_f32(&hidden_values, &[1, token_count, input_width])
                .and_then(|array| runtime.astype(&array, activation_dtype))
                .unwrap_or_else(|_| panic!("{row_name} hidden states should be valid"));
            let transposed_weights = runtime
                .array_from_f32(
                    &dense_weight_values,
                    &[expert_count, input_width, output_width],
                )
                .and_then(|array| runtime.astype(&array, activation_dtype))
                .unwrap_or_else(|_| panic!("{row_name} dense weights should be valid"));
            let selected_indices = runtime
                .array_from_u32(&selected_expert_ids, &[1, token_count, top_k])
                .unwrap_or_else(|_| panic!("{row_name} expert ids should be valid"));
            // Two singleton matrix axes make token rows broadcast over top-K
            // assignments without physically copying one row per selected expert.
            let expanded_states = runtime
                .expand_dims(&hidden_states, -2)
                .and_then(|array| runtime.expand_dims(&array, -3))
                .unwrap_or_else(|_| panic!("{row_name} gather states should expand"));

            let mut attribution = PerformanceAttribution::enabled();
            // First build the straightforward reference: preserve router order,
            // gather the requested expert for each assignment, then later reduce.
            let unsorted_output = gather_expert_projection(
                &runtime,
                &expanded_states,
                StackedExpertProjection::Dense {
                    transposed_weights: &transposed_weights,
                },
                &selected_indices,
                ExpertAssignmentOrder::Original,
                &mut attribution,
            )
            .unwrap_or_else(|_| panic!("{row_name} unsorted gathered projection should succeed"));
            let unsorted_output = runtime
                .squeeze_axis(&unsorted_output, -2)
                .expect("the singleton matrix row should squeeze");

            // Now build the optimized path. Sorting moves expert IDs and their
            // matching activation rows together and records an inverse map back
            // to the original [token, top-K] assignment positions.
            let sorted_assignments = sort_expert_assignments(
                &runtime,
                &expanded_states,
                &selected_indices,
                &mut attribution,
            )
            .unwrap_or_else(|_| panic!("{row_name} assignments should sort"));
            let sorted_output = gather_expert_projection(
                &runtime,
                &sorted_assignments.sorted_states,
                StackedExpertProjection::Dense {
                    transposed_weights: &transposed_weights,
                },
                &sorted_assignments.sorted_indices,
                ExpertAssignmentOrder::SortedByExpert,
                &mut attribution,
            )
            .unwrap_or_else(|_| panic!("{row_name} sorted gathered projection should succeed"));
            // Restoration exists only as a transparent test oracle. Production
            // sorted reduction consumes inverse_order directly and intentionally
            // avoids materializing this expanded [token, top-K, output] tensor.
            let restored_output = restore_expert_assignment_order(
                &runtime,
                &sorted_output,
                &sorted_assignments.inverse_order,
                &selected_indices.shape(),
            )
            .expect("the sorted output should restore to router order");

            let tolerance = if activation_dtype == MlxDtype::Float32 {
                1e-5
            } else {
                2e-2
            };
            assert_f32_close(
                &values_as_float32(&runtime, &restored_output),
                &values_as_float32(&runtime, &unsorted_output),
                tolerance,
            );
            // Uniform scores isolate assignment ordering from router mathematics.
            // Router score formulas remain model-family responsibilities.
            let selected_scores = runtime
                .array_from_f32(
                    &vec![1.0 / top_k as f32; (token_count * top_k) as usize],
                    &[1, token_count, top_k],
                )
                .expect("the deterministic router scores should be valid");
            // Compare the production optimized reduction with the obvious
            // original-order multiply-and-sum reference.
            let sorted_reduction = sorted_expert_weighted_sum(
                &runtime,
                &sorted_reduction_kernel,
                &sorted_output,
                &sorted_assignments.inverse_order,
                &selected_scores,
                &mut attribution,
            )
            .expect("the sorted projection should reduce without an expanded restoration");
            let unsorted_reduction = unsorted_expert_weighted_sum(
                &runtime,
                &unsorted_output,
                &selected_scores,
                &mut attribution,
            )
            .expect("the original-order projection should reduce");
            assert_f32_close(
                &values_as_float32(&runtime, &sorted_reduction),
                &values_as_float32(&runtime, &unsorted_reduction),
                tolerance,
            );
            // One journey must expose all three critical performance boundaries:
            // assignment preparation, gathered projection, and weighted reduction.
            assert!(
                attribution
                    .operation_measurement(PerformanceOperation::ExpertAssignmentPreparation)
                    .is_some(),
                "{row_name} must attribute assignment preparation"
            );
            assert!(
                attribution
                    .operation_measurement(PerformanceOperation::GatheredExpertExecution)
                    .is_some(),
                "{row_name} must attribute gathered execution"
            );
            assert!(
                attribution
                    .operation_measurement(PerformanceOperation::ExpertWeightedReduction)
                    .is_some(),
                "{row_name} must attribute weighted reduction"
            );
        }
    }
}

#[tokio::test]
async fn should_match_dequantized_reference_for_every_affine_bit_and_group_size() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    // These are exactly the affine profiles accepted by the pinned MLX runtime.
    // The nested loops cover the complete 6 × 3 compatibility matrix.
    let supported_bits = [2_i32, 3, 4, 5, 6, 8];
    let supported_group_sizes = [32_i32, 64, 128];

    for group_size in supported_group_sizes {
        for bits in supported_bits {
            let expert_count = 3_i32;
            let output_width = 2_i32;
            // Start from deterministic floating-point rows, then ask MLX itself
            // to pack them. Hand-encoding bits here would test our fixture writer
            // rather than the gathered operation used by real model weights.
            let source_weight_values = (0..expert_count * output_width * group_size)
                .map(|value_index| {
                    let expert_offset = (value_index / (output_width * group_size)) as f32;
                    let repeating_value = (value_index % 13) as f32 - 6.0;
                    expert_offset * 0.2 + repeating_value * 0.03
                })
                .collect::<Vec<_>>();
            let source_weights = runtime
                .array_from_f32(
                    &source_weight_values,
                    &[expert_count, output_width, group_size],
                )
                .expect("affine source weights should be valid");
            let (packed_weights, scales, biases) = runtime
                .quantize_affine(&source_weights, group_size, bits)
                .expect("the supported affine row should quantize");
            let activation_values = (0..expert_count * group_size)
                .map(|value_index| 0.1 + (value_index % 17) as f32 * 0.015)
                .collect::<Vec<_>>();
            let activations = runtime
                .array_from_f32(&activation_values, &[expert_count, 1, group_size])
                .expect("affine gathered activations should be valid");
            // Alternate sorted and unsorted cells across bit widths. Sorted cells
            // receive ascending IDs; unsorted cells receive deliberately shuffled
            // IDs. We never assert MLX's sorted flag for a shuffled index vector.
            let uses_sorted_order = bits % 2 == 0;
            let selected_expert_ids = if uses_sorted_order {
                vec![0_u32, 1, 2]
            } else {
                vec![2_u32, 0, 1]
            };
            let selected_indices = runtime
                .array_from_u32(&selected_expert_ids, &[expert_count])
                .expect("affine selected expert ids should be valid");
            let assignment_order = if uses_sorted_order {
                ExpertAssignmentOrder::SortedByExpert
            } else {
                ExpertAssignmentOrder::Original
            };

            // This is the production packed path: x @ dequantize(selected W)ᵀ is
            // fused inside MLX gather_qmm without a full dequantized weight tensor.
            let gathered_affine_output = gather_expert_projection(
                &runtime,
                &activations,
                StackedExpertProjection::Affine {
                    packed_weights: &packed_weights,
                    scales: &scales,
                    biases: &biases,
                    group_size,
                    bits,
                },
                &selected_indices,
                assignment_order,
                &mut PerformanceAttribution::disabled(),
            )
            .expect("the affine gathered operation should build");
            // Build a clear operations reference from stock MLX primitives:
            // dequantize every test weight, transpose it, then use dense gather_mm.
            // This is intentionally slower and exists only to establish parity.
            let dequantized_weights = runtime
                .dequantize_affine(&packed_weights, &scales, &biases, group_size, bits)
                .expect("the affine weights should dequantize for the reference");
            let transposed_reference_weights = runtime
                .transpose_axes(&dequantized_weights, &[0, 2, 1])
                .expect("reference weights should transpose");
            let gathered_reference_output = gather_expert_projection(
                &runtime,
                &activations,
                StackedExpertProjection::Dense {
                    transposed_weights: &transposed_reference_weights,
                },
                &selected_indices,
                assignment_order,
                &mut PerformanceAttribution::disabled(),
            )
            .expect("the dequantized operations reference should build");

            assert_f32_close(
                &values_as_float32(&runtime, &gathered_affine_output),
                &values_as_float32(&runtime, &gathered_reference_output),
                // Stock gather_qmm uses its packed quantized accumulation path,
                // while the reference dequantizes before dense accumulation.
                // Their expected floating-point ordering difference stays below
                // three tenths of one percent across every supported profile.
                3e-3,
            );
        }
    }
}

#[tokio::test]
async fn should_execute_empty_dense_and_affine_assignment_sets() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let source_weights = runtime
        .array_from_f32(&[0.25; 2 * 2 * 32], &[2, 2, 32])
        .expect("source weights should be valid");
    let transposed_weights = runtime
        .transpose_axes(&source_weights, &[0, 2, 1])
        .expect("dense weights should transpose");
    let (packed_weights, scales, biases) = runtime
        .quantize_affine(&source_weights, 32, 4)
        .expect("affine weights should quantize");
    // Zero assignments are represented by a genuine zero-length leading axis,
    // not a fake expert ID and not a model-family special case.
    let empty_activations = runtime
        .array_from_f32(&[], &[0, 1, 32])
        .expect("empty activations should be valid");
    let empty_indices = runtime
        .array_from_u32(&[], &[0])
        .expect("empty expert ids should be valid");

    // Both storage variants must preserve the same empty shape contract. Running
    // them in one loop prevents one path from gaining a hidden empty fast-path.
    for projection in [
        StackedExpertProjection::Dense {
            transposed_weights: &transposed_weights,
        },
        StackedExpertProjection::Affine {
            packed_weights: &packed_weights,
            scales: &scales,
            biases: &biases,
            group_size: 32,
            bits: 4,
        },
    ] {
        let empty_output = gather_expert_projection(
            &runtime,
            &empty_activations,
            projection,
            &empty_indices,
            ExpertAssignmentOrder::SortedByExpert,
            &mut PerformanceAttribution::disabled(),
        )
        .expect("an empty gathered projection should build");
        assert_eq!(empty_output.shape(), vec![0, 1, 2]);
        assert!(values_as_float32(&runtime, &empty_output).is_empty());
    }
}

#[tokio::test]
async fn should_execute_named_xs_and_s_projection_geometries_without_defaults() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    // These are named acceptance evidence from issue #100. They are local
    // test rows, not constants or defaults that production routing may inherit.
    let named_rows = [
        ("xs_routed", 8_i32, 256_i32, 512_i32),
        ("s_routed", 10, 256, 1_024),
    ];

    for (row_name, top_k, expert_count, routed_width) in named_rows {
        // Keep the contracted input tiny so the test remains cheap while still
        // exercising the exact expert count, top-K, and routed output width.
        let input_width = 4_i32;
        let weight_values = vec![0.125_f32; (expert_count * input_width * routed_width) as usize];
        let transposed_weights = runtime
            .array_from_f32(&weight_values, &[expert_count, input_width, routed_width])
            .unwrap_or_else(|_| panic!("{row_name} weights should be valid"));
        let activations = runtime
            .array_from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 1, input_width])
            .unwrap_or_else(|_| panic!("{row_name} activations should be valid"));
        let selected_expert_ids = (0..top_k)
            .map(|expert_id| expert_id as u32)
            .collect::<Vec<_>>();
        let selected_indices = runtime
            .array_from_u32(&selected_expert_ids, &[1, 1, top_k])
            .unwrap_or_else(|_| panic!("{row_name} expert ids should be valid"));

        // IDs 0..top_k are already ascending, so this row may truthfully opt into
        // MLX's sorted-index contract without invoking a sort solely for the test.
        let named_output = gather_expert_projection(
            &runtime,
            &activations,
            StackedExpertProjection::Dense {
                transposed_weights: &transposed_weights,
            },
            &selected_indices,
            ExpertAssignmentOrder::SortedByExpert,
            &mut PerformanceAttribution::disabled(),
        )
        .unwrap_or_else(|_| panic!("{row_name} gathered projection should build"));

        assert_eq!(
            named_output.shape(),
            vec![1, 1, top_k, 1, routed_width],
            "{row_name}"
        );
        let output_values = values_as_float32(&runtime, &named_output);
        assert_eq!(
            output_values.len(),
            (top_k * routed_width) as usize,
            "{row_name}"
        );
        assert_f32_close(&output_values, &vec![1.25; output_values.len()], 1e-5);
    }
}
