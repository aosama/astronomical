//! Quantized KV state round-trip parity against the bfloat16 slab.

use astronomical_model_serving::{FullAttentionKeyValueState, QuantizedFullAttentionKeyValueState};
use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxMemoryLimits, MlxRuntime};

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

const QUERY_HEAD_COUNT: i32 = 32;
const KEY_VALUE_HEAD_COUNT: i32 = 8;
const REPEATS: i32 = QUERY_HEAD_COUNT / KEY_VALUE_HEAD_COUNT;
const HEAD_DIMENSION: i32 = 128;
const CONTEXT_TOKENS: i32 = 256;
const GROWTH_TOKENS: i32 = 256;
const GROUP_SIZE: i32 = 64;
const BITS: i32 = 8;

#[tokio::test]
async fn should_round_trip_quantized_kv_within_bounded_tolerance() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();

    let mut full_precision =
        FullAttentionKeyValueState::empty_with_growth_tokens(GROWTH_TOKENS).expect("bf16 state");
    let mut quantized = QuantizedFullAttentionKeyValueState::empty_with_growth_tokens(
        GROWTH_TOKENS,
        GROUP_SIZE,
        BITS,
    )
    .expect("quantized state");

    // Two updates exercise the initial allocation and the in-place splice.
    let first_token_count = 100_i32;
    let first_keys = sin_wave_values(0.0, first_token_count as usize);
    let first_values = sin_wave_values(1000.0, first_token_count as usize);
    let first_keys_array = kv_array(&runtime, &first_keys, first_token_count);
    let first_values_array = kv_array(&runtime, &first_values, first_token_count);
    full_precision
        .update_and_fetch(&runtime, &first_keys_array, &first_values_array, 0)
        .expect("bf16 update");
    quantized
        .update_and_fetch(&runtime, &first_keys_array, &first_values_array, 0)
        .expect("quantized update");

    let second_token_count = 200_i32;
    let second_keys = sin_wave_values(0.25, second_token_count as usize);
    let second_values = sin_wave_values(1000.25, second_token_count as usize);
    let second_keys_array = kv_array(&runtime, &second_keys, second_token_count);
    let second_values_array = kv_array(&runtime, &second_values, second_token_count);
    full_precision
        .update_and_fetch(
            &runtime,
            &second_keys_array,
            &second_values_array,
            first_token_count,
        )
        .expect("bf16 update 2");
    quantized
        .update_and_fetch(
            &runtime,
            &second_keys_array,
            &second_values_array,
            first_token_count,
        )
        .expect("quantized update 2");

    assert_eq!(full_precision.offset_tokens(), 300);
    assert_eq!(quantized.offset_tokens(), 300);

    // The quantized slab must hold meaningfully less payload than bf16:
    // 8-bit values plus group scales/biases, versus two bytes per element.
    let bf16_payload = full_precision.payload_byte_count();
    let quantized_payload = quantized.payload_byte_count();
    let actual_ratio = quantized_payload as f64 / bf16_payload as f64;
    eprintln!("payload ratio: {actual_ratio:.3}");
    assert!(
        actual_ratio < 0.75,
        "quantized payload must be meaningfully smaller than bf16: {quantized_payload} vs {bf16_payload}"
    );

    // Round-trip: the dequantized range must track the bf16 input closely.
    let dequantized_keys = quantized
        .dequantized_token_range(&runtime, true, 0, first_token_count)
        .expect("dequantize keys");
    assert_dequantized_within_tolerance(&first_keys_array, &dequantized_keys, "keys");
    let dequantized_values = quantized
        .dequantized_token_range(&runtime, false, 0, first_token_count)
        .expect("dequantize values");
    assert_dequantized_within_tolerance(&first_values_array, &dequantized_values, "values");
}

fn assert_dequantized_within_tolerance(original: &MlxArray, dequantized: &MlxArray, label: &str) {
    let original_values = original.to_vec_f32().expect("original values");
    let dequantized_values = dequantized.to_vec_f32().expect("dequantized values");
    let maximum_error = original_values
        .iter()
        .zip(dequantized_values.iter())
        .map(|(original, dequantized)| (original - dequantized).abs())
        .fold(0.0_f32, f32::max);
    eprintln!("{label} max dequantize error: {maximum_error:.5}");
    assert!(
        maximum_error < 0.15,
        "the {label} quantized KV round trip must stay within a bounded tolerance, got {maximum_error}"
    );
}

fn sin_wave_values(seed_offset: f32, token_count: usize) -> Vec<f32> {
    (0..KEY_VALUE_HEAD_COUNT as usize * token_count * HEAD_DIMENSION as usize)
        .map(|index| {
            let position = index as f32 + seed_offset;
            (position * 0.01).sin() * 0.5
        })
        .collect()
}

fn query_wave_values(seed_offset: f32, token_count: usize) -> Vec<f32> {
    (0..QUERY_HEAD_COUNT as usize * token_count * HEAD_DIMENSION as usize)
        .map(|index| {
            let position = index as f32 + seed_offset;
            (position * 0.01).sin() * 0.5
        })
        .collect()
}

fn kv_array(runtime: &MlxRuntime, values: &[f32], token_count: i32) -> MlxArray {
    runtime
        .array_from_f32(
            values,
            &[1, KEY_VALUE_HEAD_COUNT, token_count, HEAD_DIMENSION],
        )
        .expect("kv array")
}

fn bf16_array(
    runtime: &MlxRuntime,
    values: &[f32],
    head_count: i32,
    token_count: i32,
    head_dim: i32,
) -> MlxArray {
    runtime
        .array_from_f32(values, &[1, head_count, token_count, head_dim])
        .and_then(|array| runtime.astype(&array, MlxDtype::BFloat16))
        .expect("bf16 array")
}

fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("test memory limits"),
    )
    .expect("direct MLX runtime")
}

#[tokio::test]
async fn should_match_bf16_attention_within_tolerance_at_production_shapes() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();

    // K2 production geometry: 32 query heads over 8 KV heads, head_dim 128.
    let query_head_count = 32_i32;
    let key_value_head_count = 8_i32;
    let head_dim = 128_i32;
    let context_tokens = 256_i32;

    // bf16 keys and values with the GQA-aware causal attention as reference.
    let keys_f32 = sin_wave_values(0.5, context_tokens as usize);
    let values_f32 = sin_wave_values(7.5, context_tokens as usize);
    let keys_bf16 = bf16_array(
        &runtime,
        &keys_f32,
        key_value_head_count,
        context_tokens,
        head_dim,
    );
    let values_bf16 = bf16_array(
        &runtime,
        &values_f32,
        key_value_head_count,
        context_tokens,
        head_dim,
    );

    // Decode: one token per head.
    let queries_f32 = query_wave_values(3.0, 1);
    let queries_bf16 = bf16_array(&runtime, &queries_f32, query_head_count, 1, head_dim);
    let scale = (head_dim as f32).sqrt().recip();
    let reference = runtime
        .causal_scaled_dot_product_attention(&queries_bf16, &keys_bf16, &values_bf16, scale)
        .expect("bf16 causal attention");

    // Quantized path: quantize-on-append then the two quantized matmul passes.
    let mut quantized =
        QuantizedFullAttentionKeyValueState::empty_with_growth_tokens(256, GROUP_SIZE, BITS)
            .expect("quantized state");
    let views = quantized
        .update_and_fetch(&runtime, &keys_bf16, &values_bf16, 0)
        .expect("quantized update");
    let output =
        quantized_attention_probe(&runtime, &queries_bf16, &views).expect("quantized attention");

    let reference_values = to_f32(&runtime, &reference);
    let output_values = to_f32(&runtime, &output);
    let maximum_error = reference_values
        .iter()
        .zip(output_values.iter())
        .map(|(reference, output)| (reference - output).abs())
        .fold(0.0_f32, f32::max);
    eprintln!("decode max attention error: {maximum_error:.5}");
    assert!(
        maximum_error < 0.05,
        "quantized attention must match the bfloat16 reference within tolerance, got {maximum_error}"
    );
}

/// Mirrors the production quantized attention passes at one probe shape.
fn quantized_attention_probe(
    runtime: &MlxRuntime,
    queries: &MlxArray,
    views: &astronomical_model_serving::QuantizedKeyValueViews,
) -> Result<MlxArray, Box<dyn std::error::Error>> {
    let scale = 128_f32.sqrt().recip();
    let scaled_queries = runtime.multiply_scalar(queries, scale)?;
    let query_shape = scaled_queries.shape();
    let query_token_count = query_shape[2];
    let reshaped = runtime.reshape(
        &scaled_queries,
        &[
            1,
            KEY_VALUE_HEAD_COUNT,
            REPEATS,
            query_token_count,
            HEAD_DIMENSION,
        ],
    )?;
    let keys = expanded_views(runtime, &views.keys)?;
    let values = expanded_views(runtime, &views.values)?;
    let scores = runtime.quantized_matmul_affine(
        &reshaped,
        &keys.packed,
        &keys.scales,
        &keys.biases,
        true,
        GROUP_SIZE,
        BITS,
    )?;
    let query_indices = runtime.arange_i32(CONTEXT_TOKENS - query_token_count, CONTEXT_TOKENS)?;
    let key_indices = runtime.arange_i32(0, CONTEXT_TOKENS)?;
    let query_row = runtime.expand_dims(&query_indices, -1)?;
    let key_row = runtime.expand_dims(&key_indices, 0)?;
    let mask = runtime.greater_equal(&query_row, &key_row)?;
    let negative_limit = runtime.full(&[1], -3.4e38, scores.dtype())?;
    let masked = runtime.where_select(&mask, &scores, &negative_limit)?;
    let probabilities = runtime.softmax_axis(&masked, -1)?;
    let output = runtime.quantized_matmul_affine(
        &probabilities,
        &values.packed,
        &values.scales,
        &values.biases,
        false,
        GROUP_SIZE,
        BITS,
    )?;
    let output_shape = output.shape();
    let reshaped_output = runtime.reshape(
        &output,
        &[
            1,
            QUERY_HEAD_COUNT,
            output_shape[output_shape.len() - 2],
            HEAD_DIMENSION,
        ],
    )?;
    Ok(reshaped_output)
}

fn expanded_views(
    runtime: &MlxRuntime,
    views: &astronomical_model_serving::QuantizedTensorViews,
) -> Result<astronomical_model_serving::QuantizedTensorViews, Box<dyn std::error::Error>> {
    Ok(astronomical_model_serving::QuantizedTensorViews {
        packed: runtime.expand_dims(&views.packed, -3)?,
        scales: runtime.expand_dims(&views.scales, -3)?,
        biases: runtime.expand_dims(&views.biases, -3)?,
    })
}

fn to_f32(runtime: &MlxRuntime, array: &MlxArray) -> Vec<f32> {
    runtime
        .astype(array, MlxDtype::Float32)
        .expect("cast to float32")
        .to_vec_f32()
        .expect("read float32 values")
}
