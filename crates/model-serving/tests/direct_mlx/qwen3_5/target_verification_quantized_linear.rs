use astronomical_model_serving::{
    Qwen3_5TargetVerificationProjectionDispatch, four_row_split_k_quantized_linear_kernel,
    qwen3_5_target_verification_quantized_linear, target_verification_quantized_linear_kernel,
};
use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxMemoryLimits, MlxRuntime};

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

#[derive(Clone, Copy)]
struct ProjectionGeometry {
    token_count: i32,
    input_dimension: i32,
    output_dimension: i32,
    activation_dtype: MlxDtype,
    quantization_bits: i32,
    quantization_group_size: i32,
}

#[tokio::test]
async fn should_match_repeated_one_token_projection_across_supported_geometries() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let target_verification_kernel = target_verification_quantized_linear_kernel()
        .expect("the target-verification kernel should initialize");
    let four_row_split_k_kernel = four_row_split_k_quantized_linear_kernel()
        .expect("the four-row split-K kernel should initialize");
    let supported_geometries = [
        ProjectionGeometry {
            token_count: 2,
            input_dimension: 512,
            output_dimension: 8,
            activation_dtype: MlxDtype::BFloat16,
            quantization_bits: 4,
            quantization_group_size: 32,
        },
        ProjectionGeometry {
            token_count: 3,
            input_dimension: 1_024,
            output_dimension: 16,
            activation_dtype: MlxDtype::Float16,
            quantization_bits: 5,
            quantization_group_size: 64,
        },
    ];

    for projection_geometry in supported_geometries {
        let projection_inputs = projection_inputs(&runtime, projection_geometry);
        let optimized_projection = qwen3_5_target_verification_quantized_linear(
            &runtime,
            &target_verification_kernel,
            &four_row_split_k_kernel,
            &projection_inputs.activations,
            &projection_inputs.packed_weight,
            &projection_inputs.quantization_scales,
            &projection_inputs.quantization_biases,
            projection_geometry.quantization_group_size,
            projection_geometry.quantization_bits,
        )
        .expect("supported target-verification geometry should project");
        assert_eq!(
            optimized_projection.dispatch(),
            Qwen3_5TargetVerificationProjectionDispatch::OptimizedMetal
        );
        let token_local_projection =
            repeated_one_token_projection(&runtime, &projection_inputs, projection_geometry);

        assert_eq!(
            float32_values(
                &runtime,
                &optimized_projection.into_projected_activations(),
                "optimized projection",
            ),
            float32_values(&runtime, &token_local_projection, "token-local projection"),
            "optimized target verification must preserve token-local arithmetic for tokens={}, input={}, output={}, dtype={:?}, bits={}, group_size={}",
            projection_geometry.token_count,
            projection_geometry.input_dimension,
            projection_geometry.output_dimension,
            projection_geometry.activation_dtype,
            projection_geometry.quantization_bits,
            projection_geometry.quantization_group_size,
        );
    }
}

#[tokio::test]
async fn should_keep_four_row_split_k_argmax_aligned_with_token_local_projection() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let target_verification_kernel = target_verification_quantized_linear_kernel()
        .expect("the target-verification kernel should initialize");
    let four_row_split_k_kernel = four_row_split_k_quantized_linear_kernel()
        .expect("the four-row split-K kernel should initialize");
    let projection_geometry = ProjectionGeometry {
        token_count: 4,
        input_dimension: 512,
        output_dimension: 24,
        activation_dtype: MlxDtype::BFloat16,
        quantization_bits: 4,
        quantization_group_size: 64,
    };
    let projection_inputs = projection_inputs(&runtime, projection_geometry);
    let four_row_projection = qwen3_5_target_verification_quantized_linear(
        &runtime,
        &target_verification_kernel,
        &four_row_split_k_kernel,
        &projection_inputs.activations,
        &projection_inputs.packed_weight,
        &projection_inputs.quantization_scales,
        &projection_inputs.quantization_biases,
        projection_geometry.quantization_group_size,
        projection_geometry.quantization_bits,
    )
    .expect("four-row 4-bit geometry should project");
    assert_eq!(
        four_row_projection.dispatch(),
        Qwen3_5TargetVerificationProjectionDispatch::FourRowSplitK
    );
    let token_local_projection =
        repeated_one_token_projection(&runtime, &projection_inputs, projection_geometry);
    let four_row_values = float32_values(
        &runtime,
        &four_row_projection.into_projected_activations(),
        "four-row projection",
    );
    let token_local_values =
        float32_values(&runtime, &token_local_projection, "token-local projection");
    assert_eq!(four_row_values.len(), token_local_values.len());
    let output_dimension = projection_geometry.output_dimension as usize;
    for token_position_index in 0..projection_geometry.token_count as usize {
        let row_start = token_position_index * output_dimension;
        let row_end = row_start + output_dimension;
        let four_row_argmax = argmax(&four_row_values[row_start..row_end]);
        let token_local_argmax = argmax(&token_local_values[row_start..row_end]);
        assert_eq!(
            four_row_argmax, token_local_argmax,
            "four-row split-K must preserve token-local argmax at token {token_position_index}"
        );
    }
}

fn argmax(row_values: &[f32]) -> usize {
    row_values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(row_index, _)| row_index)
        .expect("a projection row should contain values")
}

#[tokio::test]
async fn should_use_token_local_mlx_for_each_unsupported_dispatch_geometry() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();
    let target_verification_kernel = target_verification_quantized_linear_kernel()
        .expect("the target-verification kernel should initialize");
    let four_row_split_k_kernel = four_row_split_k_quantized_linear_kernel()
        .expect("the four-row split-K kernel should initialize");
    let unsupported_geometries = [
        ProjectionGeometry {
            token_count: 3,
            input_dimension: 512,
            output_dimension: 8,
            activation_dtype: MlxDtype::BFloat16,
            quantization_bits: 6,
            quantization_group_size: 32,
        },
        ProjectionGeometry {
            token_count: 3,
            input_dimension: 256,
            output_dimension: 8,
            activation_dtype: MlxDtype::Float16,
            quantization_bits: 4,
            quantization_group_size: 64,
        },
        ProjectionGeometry {
            token_count: 3,
            input_dimension: 512,
            output_dimension: 10,
            activation_dtype: MlxDtype::BFloat16,
            quantization_bits: 5,
            quantization_group_size: 128,
        },
        ProjectionGeometry {
            token_count: 3,
            input_dimension: 512,
            output_dimension: 8,
            activation_dtype: MlxDtype::Float32,
            quantization_bits: 4,
            quantization_group_size: 32,
        },
    ];

    for projection_geometry in unsupported_geometries {
        let projection_inputs = projection_inputs(&runtime, projection_geometry);
        let fallback_projection = qwen3_5_target_verification_quantized_linear(
            &runtime,
            &target_verification_kernel,
            &four_row_split_k_kernel,
            &projection_inputs.activations,
            &projection_inputs.packed_weight,
            &projection_inputs.quantization_scales,
            &projection_inputs.quantization_biases,
            projection_geometry.quantization_group_size,
            projection_geometry.quantization_bits,
        )
        .expect("unsupported custom-kernel geometry should use the MLX fallback");
        assert_eq!(
            fallback_projection.dispatch(),
            Qwen3_5TargetVerificationProjectionDispatch::TokenLocalMlxFallback
        );
        let token_local_projection =
            repeated_one_token_projection(&runtime, &projection_inputs, projection_geometry);
        assert_eq!(
            float32_values(
                &runtime,
                &fallback_projection.into_projected_activations(),
                "fallback projection",
            ),
            float32_values(&runtime, &token_local_projection, "token-local projection"),
        );
    }
}

struct ProjectionInputs {
    activations: MlxArray,
    packed_weight: MlxArray,
    quantization_scales: MlxArray,
    quantization_biases: MlxArray,
}

fn projection_inputs(
    runtime: &MlxRuntime,
    projection_geometry: ProjectionGeometry,
) -> ProjectionInputs {
    let activation_element_count =
        (projection_geometry.token_count * projection_geometry.input_dimension) as usize;
    let activation_values = (0..activation_element_count)
        .map(|activation_index| ((activation_index % 29) as f32 - 14.0) / 16.0)
        .collect::<Vec<_>>();
    let activations = runtime
        .array_from_f32(
            &activation_values,
            &[
                1,
                projection_geometry.token_count,
                projection_geometry.input_dimension,
            ],
        )
        .and_then(|array| runtime.astype(&array, projection_geometry.activation_dtype))
        .expect("target-verification activations should be valid");

    let weight_element_count =
        (projection_geometry.output_dimension * projection_geometry.input_dimension) as usize;
    let weight_values = (0..weight_element_count)
        .map(|weight_index| ((weight_index % 31) as f32 - 15.0) / 32.0)
        .collect::<Vec<_>>();
    let weights = runtime
        .array_from_f32(
            &weight_values,
            &[
                projection_geometry.output_dimension,
                projection_geometry.input_dimension,
            ],
        )
        .and_then(|array| runtime.astype(&array, projection_geometry.activation_dtype))
        .expect("target-verification weights should be valid");
    let (packed_weight, quantization_scales, quantization_biases) = runtime
        .quantize_affine(
            &weights,
            projection_geometry.quantization_group_size,
            projection_geometry.quantization_bits,
        )
        .expect("target-verification weights should quantize");

    ProjectionInputs {
        activations,
        packed_weight,
        quantization_scales,
        quantization_biases,
    }
}

fn repeated_one_token_projection(
    runtime: &MlxRuntime,
    projection_inputs: &ProjectionInputs,
    projection_geometry: ProjectionGeometry,
) -> MlxArray {
    let mut token_projection_outputs = Vec::with_capacity(projection_geometry.token_count as usize);
    for token_position_index in 0..projection_geometry.token_count {
        let token_activations = runtime
            .slice(
                &projection_inputs.activations,
                &[0, token_position_index, 0],
                &[
                    1,
                    token_position_index + 1,
                    projection_geometry.input_dimension,
                ],
                &[1, 1, 1],
            )
            .expect("one target-verification token should slice");
        token_projection_outputs.push(
            runtime
                .quantized_matmul_affine(
                    &token_activations,
                    &projection_inputs.packed_weight,
                    &projection_inputs.quantization_scales,
                    &projection_inputs.quantization_biases,
                    true,
                    projection_geometry.quantization_group_size,
                    projection_geometry.quantization_bits,
                )
                .expect("one target-verification token should project"),
        );
    }
    let token_projection_output_references = token_projection_outputs.iter().collect::<Vec<_>>();
    runtime
        .concatenate_axis(&token_projection_output_references, 1)
        .expect("token-local projections should concatenate")
}

fn float32_values(runtime: &MlxRuntime, projected_activations: &MlxArray, name: &str) -> Vec<f32> {
    runtime
        .astype(projected_activations, MlxDtype::Float32)
        .and_then(|float32_activations| float32_activations.to_vec_f32())
        .unwrap_or_else(|projection_error| panic!("{name} should evaluate: {projection_error}"))
}

fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("target-verification test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}
