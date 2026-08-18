//! Regression coverage for MLX sorted gathered-matmul activation row strides.
//!
//! Sorted Mixture-of-Experts execution flattens assignment axes while retaining
//! one singleton matrix-row axis. This test protects that exact production shape
//! against an MLX regression that previously read later activation rows from the
//! singleton axis stride instead of the contracted input width.

use astronomical_model_serving::{
    ExpertAssignmentOrder, PerformanceAttribution, StackedExpertProjection,
    gather_expert_projection,
};
use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxMemoryLimits, MlxRuntime};

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("sorted gathered-matmul test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

fn evaluated_float32_values(runtime: &MlxRuntime, array: &MlxArray) -> Vec<f32> {
    runtime
        .astype(array, MlxDtype::Float32)
        .and_then(|float32_array| float32_array.to_vec_f32())
        .expect("the gathered projection should evaluate as float32")
}

#[tokio::test]
async fn should_match_unsorted_reference_for_singleton_row_sorted_dense_gather() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let assignment_count = 3_i32;
    let input_width = 64_i32;
    let output_width = 3_i32;

    let activation_values = (0..assignment_count * input_width)
        .map(|value_index| 0.125 + value_index as f32 * 0.0078125)
        .collect::<Vec<_>>();
    let activation_rows = runtime
        .array_from_f32(&activation_values, &[assignment_count, input_width])
        .expect("activation rows should be valid");
    // The inserted matrix-row axis used to retain stride one in MLX. For sorted
    // gather, treating that stride as the row width made every row after the first
    // read from the wrong activation offset.
    let singleton_row_activations = runtime
        .expand_dims(&activation_rows, -2)
        .expect("activation rows should gain the matrix-row axis");
    assert_eq!(
        singleton_row_activations.shape(),
        vec![assignment_count, 1, input_width]
    );

    let transposed_weight_values = (0..assignment_count * input_width * output_width)
        .map(|value_index| ((value_index % 29) as f32 - 14.0) * 0.015625)
        .collect::<Vec<_>>();
    let transposed_weights = runtime
        .array_from_f32(
            &transposed_weight_values,
            &[assignment_count, input_width, output_width],
        )
        .expect("stacked dense expert weights should be valid");
    let selected_expert_indices = runtime
        .array_from_u32(&[0, 1, 2], &[assignment_count])
        .expect("ascending expert indices should be valid");

    let sorted_output = gather_expert_projection(
        &runtime,
        &singleton_row_activations,
        StackedExpertProjection::Dense {
            transposed_weights: &transposed_weights,
        },
        &selected_expert_indices,
        ExpertAssignmentOrder::SortedByExpert,
        &mut PerformanceAttribution::disabled(),
    )
    .expect("the sorted gathered projection should build");
    let unsorted_reference_output = gather_expert_projection(
        &runtime,
        &singleton_row_activations,
        StackedExpertProjection::Dense {
            transposed_weights: &transposed_weights,
        },
        &selected_expert_indices,
        ExpertAssignmentOrder::Original,
        &mut PerformanceAttribution::disabled(),
    )
    .expect("the unsorted gathered reference should build");

    let sorted_values = evaluated_float32_values(&runtime, &sorted_output);
    let unsorted_reference_values = evaluated_float32_values(&runtime, &unsorted_reference_output);
    assert_eq!(sorted_values.len(), unsorted_reference_values.len());
    for (sorted_value, unsorted_reference_value) in
        sorted_values.iter().zip(&unsorted_reference_values)
    {
        assert!(
            (sorted_value - unsorted_reference_value).abs() <= 1e-5,
            "sorted singleton-row gather produced {sorted_value}, expected {unsorted_reference_value}"
        );
    }
}
