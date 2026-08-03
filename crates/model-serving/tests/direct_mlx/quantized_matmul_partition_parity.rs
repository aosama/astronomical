use std::time::Duration;

use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxMemoryLimits, MlxRuntime};
use tokio::time::timeout;

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

const FULL_PREFILL_TOKEN_COUNT: i32 = 4_096;
const PREFILL_PARTITION_TOKEN_COUNT: i32 = FULL_PREFILL_TOKEN_COUNT / 2;
const HIDDEN_STATE_DIMENSION: i32 = 2_048;
const GATED_DELTA_PARAMETER_COUNT: i32 = 32;
const QUANTIZATION_GROUP_SIZE: i32 = 64;
const QUANTIZATION_BIT_WIDTH: i32 = 4;
const PACKED_WEIGHT_WORD_COUNT_PER_OUTPUT: i32 =
    HIDDEN_STATE_DIMENSION * QUANTIZATION_BIT_WIDTH / 32;
const QUANTIZATION_GROUP_COUNT_PER_OUTPUT: i32 = HIDDEN_STATE_DIMENSION / QUANTIZATION_GROUP_SIZE;
const QUANTIZED_MATMUL_PARTITION_PARITY_TIMEOUT: Duration = Duration::from_secs(115);

#[tokio::test]
async fn should_preserve_small_output_quantized_matmul_across_equivalent_prefill_partitions() {
    timeout(
        QUANTIZED_MATMUL_PARTITION_PARITY_TIMEOUT,
        compare_small_output_quantized_matmul_across_prefill_partitions(),
    )
    .await
    .expect("the quantized matmul partition parity test must finish within 115 seconds");
}

async fn compare_small_output_quantized_matmul_across_prefill_partitions() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the quantized matmul partition parity memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize for quantized matmul partition parity");
    let full_prefill_hidden_states = bfloat16_full_prefill_hidden_states(&runtime);
    let quantized_weights = affine_quantized_weights(&runtime);
    let quantization_scales = bfloat16_affine_quantization_parameters(&runtime, 0.03125, 0.125);
    let quantization_biases = bfloat16_affine_quantization_parameters(&runtime, 0.015625, -0.0625);

    eprintln!(
        "[quantized-matmul-partition-parity] status=progress phase=full_prefill token_count={FULL_PREFILL_TOKEN_COUNT}"
    );
    let full_prefill_projection = runtime
        .quantized_matmul_affine(
            &full_prefill_hidden_states,
            &quantized_weights,
            &quantization_scales,
            &quantization_biases,
            true,
            QUANTIZATION_GROUP_SIZE,
            QUANTIZATION_BIT_WIDTH,
        )
        .expect("the full prefill quantized projection should build");
    let full_prefill_suffix_projection_values = slice_prefill_suffix_projection_values(
        &runtime,
        &full_prefill_projection,
        PREFILL_PARTITION_TOKEN_COUNT,
        FULL_PREFILL_TOKEN_COUNT,
    );

    let partitioned_prefill_hidden_states = runtime
        .slice(
            &full_prefill_hidden_states,
            &[0, PREFILL_PARTITION_TOKEN_COUNT, 0],
            &[1, FULL_PREFILL_TOKEN_COUNT, HIDDEN_STATE_DIMENSION],
            &[1, 1, 1],
        )
        .expect("the second prefill partition hidden states should slice");
    eprintln!(
        "[quantized-matmul-partition-parity] status=progress phase=partitioned_prefill token_count={PREFILL_PARTITION_TOKEN_COUNT}"
    );
    let partitioned_prefill_projection = runtime
        .quantized_matmul_affine(
            &partitioned_prefill_hidden_states,
            &quantized_weights,
            &quantization_scales,
            &quantization_biases,
            true,
            QUANTIZATION_GROUP_SIZE,
            QUANTIZATION_BIT_WIDTH,
        )
        .expect("the partitioned prefill quantized projection should build");
    let partitioned_prefill_projection_values =
        projection_values_as_float32(&runtime, &partitioned_prefill_projection);
    let maximum_absolute_delta = maximum_absolute_difference(
        &full_prefill_suffix_projection_values,
        &partitioned_prefill_projection_values,
    );
    eprintln!(
        "[quantized-matmul-partition-parity] status=success maximum_absolute_delta={maximum_absolute_delta:.6}"
    );
    assert_eq!(
        maximum_absolute_delta, 0.0,
        "quantized projection output must not change when equivalent prefill work is partitioned"
    );
}

fn bfloat16_full_prefill_hidden_states(runtime: &MlxRuntime) -> MlxArray {
    let hidden_state_element_count =
        usize::try_from(FULL_PREFILL_TOKEN_COUNT * HIDDEN_STATE_DIMENSION)
            .expect("the full prefill hidden-state element count should fit usize");
    let float32_hidden_states = runtime
        .array_from_f32(
            &deterministic_float_values(hidden_state_element_count, 0.0078125, -1.0),
            &[1, FULL_PREFILL_TOKEN_COUNT, HIDDEN_STATE_DIMENSION],
        )
        .expect("the full prefill float32 hidden states should allocate");
    runtime
        .astype(&float32_hidden_states, MlxDtype::BFloat16)
        .expect("the full prefill hidden states should cast to bfloat16")
}

fn affine_quantized_weights(runtime: &MlxRuntime) -> MlxArray {
    let packed_weight_element_count =
        usize::try_from(GATED_DELTA_PARAMETER_COUNT * PACKED_WEIGHT_WORD_COUNT_PER_OUTPUT)
            .expect("the packed weight element count should fit usize");
    let packed_weight_words = (0..packed_weight_element_count)
        .map(|packed_weight_index| {
            let packed_nibble_pattern = (packed_weight_index as u32).wrapping_mul(0x9e37_79b9);
            packed_nibble_pattern ^ packed_nibble_pattern.rotate_left(13)
        })
        .collect::<Vec<_>>();
    runtime
        .array_from_u32(
            &packed_weight_words,
            &[
                GATED_DELTA_PARAMETER_COUNT,
                PACKED_WEIGHT_WORD_COUNT_PER_OUTPUT,
            ],
        )
        .expect("the affine quantized weights should allocate")
}

fn bfloat16_affine_quantization_parameters(
    runtime: &MlxRuntime,
    value_multiplier: f32,
    value_offset: f32,
) -> MlxArray {
    let quantization_parameter_element_count =
        usize::try_from(GATED_DELTA_PARAMETER_COUNT * QUANTIZATION_GROUP_COUNT_PER_OUTPUT)
            .expect("the affine quantization parameter element count should fit usize");
    let float32_quantization_parameters = runtime
        .array_from_f32(
            &deterministic_float_values(
                quantization_parameter_element_count,
                value_multiplier,
                value_offset,
            ),
            &[
                GATED_DELTA_PARAMETER_COUNT,
                QUANTIZATION_GROUP_COUNT_PER_OUTPUT,
            ],
        )
        .expect("the float32 affine quantization parameters should allocate");
    runtime
        .astype(&float32_quantization_parameters, MlxDtype::BFloat16)
        .expect("the affine quantization parameters should cast to bfloat16")
}

fn slice_prefill_suffix_projection_values(
    runtime: &MlxRuntime,
    full_prefill_projection: &MlxArray,
    suffix_start_token: i32,
    suffix_end_token: i32,
) -> Vec<f32> {
    let full_prefill_suffix_projection = runtime
        .slice(
            full_prefill_projection,
            &[0, suffix_start_token, 0],
            &[1, suffix_end_token, GATED_DELTA_PARAMETER_COUNT],
            &[1, 1, 1],
        )
        .expect("the full prefill suffix projection should slice");
    projection_values_as_float32(runtime, &full_prefill_suffix_projection)
}

fn projection_values_as_float32(runtime: &MlxRuntime, projection_values: &MlxArray) -> Vec<f32> {
    runtime
        .astype(projection_values, MlxDtype::Float32)
        .expect("the bfloat16 projection values should cast to float32")
        .to_vec_f32()
        .expect("the projection values should materialize")
}

fn deterministic_float_values(
    value_count: usize,
    value_multiplier: f32,
    value_offset: f32,
) -> Vec<f32> {
    (0..value_count)
        .map(|value_index| {
            let bounded_pattern = (value_index.wrapping_mul(37) % 257) as f32 - 128.0;
            bounded_pattern * value_multiplier + value_offset
        })
        .collect()
}

fn maximum_absolute_difference(
    full_prefill_projection_values: &[f32],
    partitioned_prefill_projection_values: &[f32],
) -> f32 {
    assert_eq!(
        full_prefill_projection_values.len(),
        partitioned_prefill_projection_values.len(),
        "equivalent projection outputs must have equal element counts"
    );
    full_prefill_projection_values
        .iter()
        .zip(partitioned_prefill_projection_values)
        .map(
            |(full_prefill_projection_value, partitioned_prefill_projection_value)| {
                (full_prefill_projection_value - partitioned_prefill_projection_value).abs()
            },
        )
        .fold(0.0_f32, f32::max)
}
