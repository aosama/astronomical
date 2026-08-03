use std::time::Instant;

use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxRuntime};

use crate::common::runtime_test_support::runtime;

const ORNITH_BATCH_SIZE: i32 = 1;
const ORNITH_QUERY_HEAD_COUNT: i32 = 16;
const ORNITH_KEY_VALUE_HEAD_COUNT: i32 = 2;
const ORNITH_ATTENTION_HEAD_DIMENSION: i32 = 256;
const WARMUP_ITERATIONS: usize = 3;
const MEASUREMENT_ITERATIONS: usize = 10;
const MEBIBYTE: usize = 1024 * 1024;
const BF16_ATTENTION_ABSOLUTE_TOLERANCE: f32 = 5e-3;
const BF16_ATTENTION_RELATIVE_TOLERANCE: f32 = 5e-3;

#[test]
#[ignore = "compares fused NAX head_dim=256 attention with independent MLX primitives"]
fn should_match_unfused_reference_for_head_dim_256_causal_attention() {
    let runtime = runtime();
    for attention_benchmark_shape in [
        AttentionBenchmarkShape::new(1_024, 1_024),
        AttentionBenchmarkShape::new(1_025, 1_301),
    ] {
        eprintln!(
            "[attention-correctness:mlx] q_len={} kv_len={} ETA<=30s",
            attention_benchmark_shape.query_token_count,
            attention_benchmark_shape.key_value_token_count,
        );
        let attention_benchmark_inputs =
            AttentionBenchmarkInputs::new(&runtime, attention_benchmark_shape);
        let fused_attention_output = mlx_attention(&runtime, &attention_benchmark_inputs);
        let unfused_attention_output = unfused_causal_attention_reference(
            &runtime,
            &attention_benchmark_inputs,
            attention_benchmark_shape,
        );

        assert_attention_output_samples_close(
            &runtime,
            &fused_attention_output,
            &unfused_attention_output,
            attention_benchmark_shape,
        );
    }
}

#[test]
#[ignore = "measures MLX causal attention for Ornith-shaped BF16 prefill tensors"]
fn should_measure_mlx_attention_for_ornith_prefill_shapes() {
    let runtime = runtime();
    for attention_benchmark_shape in [
        AttentionBenchmarkShape::new(1_024, 2_048),
        AttentionBenchmarkShape::new(1_024, 4_096),
        AttentionBenchmarkShape::new(1_024, 8_192),
        AttentionBenchmarkShape::new(2_048, 8_192),
        AttentionBenchmarkShape::new(4_096, 4_096),
    ] {
        eprintln!(
            "[attention-benchmark:mlx] preparing q_len={} kv_len={} ETA<=30s",
            attention_benchmark_shape.query_token_count,
            attention_benchmark_shape.key_value_token_count,
        );
        let attention_benchmark_inputs =
            AttentionBenchmarkInputs::new(&runtime, attention_benchmark_shape);
        materialize_inputs(&runtime, &attention_benchmark_inputs);
        let fused_median_elapsed_millis = measure_attention_implementation(
            &runtime,
            attention_benchmark_shape,
            "fused-nax",
            || mlx_attention(&runtime, &attention_benchmark_inputs),
        );
        let unfused_median_elapsed_millis = measure_attention_implementation(
            &runtime,
            attention_benchmark_shape,
            "unfused-reference",
            || {
                unfused_causal_attention_reference(
                    &runtime,
                    &attention_benchmark_inputs,
                    attention_benchmark_shape,
                )
            },
        );
        eprintln!(
            "[attention-benchmark:comparison] q_len={} kv_len={} fused_median_ms={fused_median_elapsed_millis:.3} unfused_median_ms={unfused_median_elapsed_millis:.3} speedup={:.2}x",
            attention_benchmark_shape.query_token_count,
            attention_benchmark_shape.key_value_token_count,
            unfused_median_elapsed_millis / fused_median_elapsed_millis,
        );
    }
}

#[test]
#[ignore = "proves fused head_dim=256 attention does not materialize the quadratic score tensor"]
fn should_keep_head_dim_256_attention_peak_below_the_score_tensor_size() {
    let runtime = runtime();
    let attention_benchmark_inputs =
        AttentionBenchmarkInputs::new(&runtime, AttentionBenchmarkShape::new(1_024, 4_096));
    materialize_inputs(&runtime, &attention_benchmark_inputs);
    runtime
        .clear_allocator_cache()
        .expect("the memory contract should start without reclaimable allocations");
    runtime
        .reset_peak_memory()
        .expect("the memory contract should reset the MLX peak counter");

    let attention_output = mlx_attention(&runtime, &attention_benchmark_inputs);
    runtime
        .evaluate_arrays(&[&attention_output])
        .expect("head_dim=256 attention should evaluate");
    let peak_memory_bytes = runtime
        .memory_snapshot()
        .expect("the attention peak should be readable")
        .peak_memory_bytes();
    eprintln!(
        "[attention-memory:mlx] q_len=1024 kv_len=4096 peak_mib={:.1}",
        peak_memory_bytes as f64 / MEBIBYTE as f64,
    );

    assert!(
        peak_memory_bytes < 128 * MEBIBYTE,
        "fused attention should stay below the 128 MiB BF16 score tensor; peak={peak_memory_bytes} bytes"
    );
}

fn materialize_inputs(runtime: &MlxRuntime, attention_benchmark_inputs: &AttentionBenchmarkInputs) {
    runtime
        .evaluate_arrays(&[
            &attention_benchmark_inputs.query_states,
            &attention_benchmark_inputs.key_states,
            &attention_benchmark_inputs.value_states,
        ])
        .expect("attention inputs should materialize before measurement");
}

fn measure_attention_implementation(
    runtime: &MlxRuntime,
    attention_benchmark_shape: AttentionBenchmarkShape,
    implementation_label: &str,
    mut build_attention_output: impl FnMut() -> MlxArray,
) -> f64 {
    for warmup_iteration in 1..=WARMUP_ITERATIONS {
        eprintln!(
            "[attention-benchmark:{implementation_label}] warmup {warmup_iteration}/{WARMUP_ITERATIONS} q_len={} kv_len={}",
            attention_benchmark_shape.query_token_count,
            attention_benchmark_shape.key_value_token_count,
        );
        evaluate_attention_output(runtime, build_attention_output());
    }
    let mut elapsed_millis = Vec::with_capacity(MEASUREMENT_ITERATIONS);
    for measurement_iteration in 1..=MEASUREMENT_ITERATIONS {
        eprintln!(
            "[attention-benchmark:{implementation_label}] measurement {measurement_iteration}/{MEASUREMENT_ITERATIONS} q_len={} kv_len={}",
            attention_benchmark_shape.query_token_count,
            attention_benchmark_shape.key_value_token_count,
        );
        let attention_output = build_attention_output();
        let started_at = Instant::now();
        evaluate_attention_output(runtime, attention_output);
        elapsed_millis.push(started_at.elapsed().as_secs_f64() * 1_000.0);
    }
    runtime
        .clear_allocator_cache()
        .expect("the benchmark should release reclaimable MLX allocations");
    elapsed_millis.sort_by(f64::total_cmp);
    let middle_index = elapsed_millis.len() / 2;
    (elapsed_millis[middle_index - 1] + elapsed_millis[middle_index]) / 2.0
}

fn evaluate_attention_output(runtime: &MlxRuntime, attention_output: MlxArray) {
    runtime
        .evaluate_arrays(&[&attention_output])
        .expect("attention implementation should evaluate");
}

fn mlx_attention(
    runtime: &MlxRuntime,
    attention_benchmark_inputs: &AttentionBenchmarkInputs,
) -> MlxArray {
    runtime
        .causal_scaled_dot_product_attention(
            &attention_benchmark_inputs.query_states,
            &attention_benchmark_inputs.key_states,
            &attention_benchmark_inputs.value_states,
            (ORNITH_ATTENTION_HEAD_DIMENSION as f32).sqrt().recip(),
        )
        .expect("MLX causal attention should build a valid graph")
}

fn unfused_causal_attention_reference(
    runtime: &MlxRuntime,
    attention_benchmark_inputs: &AttentionBenchmarkInputs,
    attention_benchmark_shape: AttentionBenchmarkShape,
) -> MlxArray {
    let grouped_query_repeat_count = ORNITH_QUERY_HEAD_COUNT / ORNITH_KEY_VALUE_HEAD_COUNT;
    let repeated_key_states = runtime
        .repeat_axis(
            &attention_benchmark_inputs.key_states,
            grouped_query_repeat_count,
            1,
        )
        .expect("reference keys should repeat across grouped query heads");
    let repeated_value_states = runtime
        .repeat_axis(
            &attention_benchmark_inputs.value_states,
            grouped_query_repeat_count,
            1,
        )
        .expect("reference values should repeat across grouped query heads");
    let transposed_key_states = runtime
        .transpose_axes(&repeated_key_states, &[0, 1, 3, 2])
        .expect("reference keys should transpose for QK multiplication");
    let attention_scores = runtime
        .matmul(
            &attention_benchmark_inputs.query_states,
            &transposed_key_states,
        )
        .expect("reference QK multiplication should build");
    let scaled_attention_scores = runtime
        .multiply_scalar(
            &attention_scores,
            (ORNITH_ATTENTION_HEAD_DIMENSION as f32).sqrt().recip(),
        )
        .expect("reference attention scores should scale");

    let query_position_offset = attention_benchmark_shape.key_value_token_count
        - attention_benchmark_shape.query_token_count;
    let query_positions = runtime
        .arange_i32(
            query_position_offset,
            attention_benchmark_shape.key_value_token_count,
        )
        .expect("reference query positions should be valid");
    let key_positions = runtime
        .arange_i32(0, attention_benchmark_shape.key_value_token_count)
        .expect("reference key positions should be valid");
    let query_position_column = runtime
        .expand_dims(&query_positions, 1)
        .expect("reference query positions should form a column");
    let key_position_row = runtime
        .expand_dims(&key_positions, 0)
        .expect("reference key positions should form a row");
    let causal_mask = runtime
        .greater_equal(&query_position_column, &key_position_row)
        .expect("reference causal positions should compare");
    let negative_infinity = runtime
        .array_from_f32(&[f32::NEG_INFINITY], &[])
        .expect("negative infinity should form a scalar");
    let negative_infinity = runtime
        .astype(&negative_infinity, MlxDtype::BFloat16)
        .expect("negative infinity should use the attention dtype");
    let masked_attention_scores = runtime
        .where_select(&causal_mask, &scaled_attention_scores, &negative_infinity)
        .expect("reference causal mask should apply");
    let attention_probabilities = runtime
        .softmax_axis(&masked_attention_scores, -1)
        .expect("reference attention probabilities should normalize");
    runtime
        .matmul(&attention_probabilities, &repeated_value_states)
        .expect("reference probability-value multiplication should build")
}

fn assert_attention_output_samples_close(
    runtime: &MlxRuntime,
    actual_output: &MlxArray,
    expected_output: &MlxArray,
    attention_benchmark_shape: AttentionBenchmarkShape,
) {
    assert_eq!(actual_output.shape(), expected_output.shape());
    let sampled_query_positions = [
        0,
        attention_benchmark_shape.query_token_count / 2,
        attention_benchmark_shape.query_token_count - 1,
    ];
    let sampled_query_heads = [0, 7, 8, ORNITH_QUERY_HEAD_COUNT - 1];
    let mut maximum_absolute_error = 0.0_f32;
    for query_head_index in sampled_query_heads {
        for query_position in sampled_query_positions {
            let slice_starts = [0, query_head_index, query_position, 0];
            let slice_stops = [
                1,
                query_head_index + 1,
                query_position + 1,
                ORNITH_ATTENTION_HEAD_DIMENSION,
            ];
            let actual_values =
                attention_output_slice_values(runtime, actual_output, &slice_starts, &slice_stops);
            let expected_values = attention_output_slice_values(
                runtime,
                expected_output,
                &slice_starts,
                &slice_stops,
            );
            assert_attention_values_close(
                &actual_values,
                &expected_values,
                &mut maximum_absolute_error,
            );
        }
    }
    eprintln!("[attention-correctness:mlx] max_abs_error={maximum_absolute_error:.6}");
}

fn attention_output_slice_values(
    runtime: &MlxRuntime,
    attention_output: &MlxArray,
    slice_starts: &[i32],
    slice_stops: &[i32],
) -> Vec<f32> {
    let attention_output_slice = runtime
        .slice(attention_output, slice_starts, slice_stops, &[1, 1, 1, 1])
        .expect("attention output sample should slice");
    runtime
        .astype(&attention_output_slice, MlxDtype::Float32)
        .expect("attention output sample should cast to float32")
        .to_vec_f32()
        .expect("attention output sample should evaluate")
}

fn assert_attention_values_close(
    actual_values: &[f32],
    expected_values: &[f32],
    maximum_absolute_error: &mut f32,
) {
    assert_eq!(actual_values.len(), expected_values.len());
    for (actual_value, expected_value) in actual_values.iter().zip(expected_values) {
        let absolute_error = (*actual_value - *expected_value).abs();
        *maximum_absolute_error = (*maximum_absolute_error).max(absolute_error);
        assert!(
            absolute_error
                <= BF16_ATTENTION_ABSOLUTE_TOLERANCE
                    + BF16_ATTENTION_RELATIVE_TOLERANCE * expected_value.abs(),
            "expected fused value {actual_value} to match unfused value {expected_value}; absolute_error={absolute_error}"
        );
    }
}

#[derive(Clone, Copy)]
struct AttentionBenchmarkShape {
    query_token_count: i32,
    key_value_token_count: i32,
}

impl AttentionBenchmarkShape {
    const fn new(query_token_count: i32, key_value_token_count: i32) -> Self {
        Self {
            query_token_count,
            key_value_token_count,
        }
    }
}

struct AttentionBenchmarkInputs {
    query_states: MlxArray,
    key_states: MlxArray,
    value_states: MlxArray,
}

impl AttentionBenchmarkInputs {
    fn new(runtime: &MlxRuntime, attention_benchmark_shape: AttentionBenchmarkShape) -> Self {
        let query_shape = [
            ORNITH_BATCH_SIZE,
            ORNITH_QUERY_HEAD_COUNT,
            attention_benchmark_shape.query_token_count,
            ORNITH_ATTENTION_HEAD_DIMENSION,
        ];
        let key_value_shape = [
            ORNITH_BATCH_SIZE,
            ORNITH_KEY_VALUE_HEAD_COUNT,
            attention_benchmark_shape.key_value_token_count,
            ORNITH_ATTENTION_HEAD_DIMENSION,
        ];
        Self {
            query_states: deterministic_bf16_array(runtime, &query_shape, 17),
            key_states: deterministic_bf16_array(runtime, &key_value_shape, 29),
            value_states: deterministic_bf16_array(runtime, &key_value_shape, 43),
        }
    }
}

fn deterministic_bf16_array(
    runtime: &MlxRuntime,
    shape: &[i32],
    sequence_multiplier: usize,
) -> MlxArray {
    let element_count = shape
        .iter()
        .try_fold(1_usize, |product, dimension_size| {
            usize::try_from(*dimension_size)
                .ok()
                .and_then(|dimension_size| product.checked_mul(dimension_size))
        })
        .expect("attention shape should fit usize");
    let deterministic_values = (0..element_count)
        .map(|element_index| {
            let bounded_pattern = (element_index.wrapping_mul(sequence_multiplier) % 257) as f32;
            (bounded_pattern - 128.0) / 256.0
        })
        .collect::<Vec<_>>();
    let float32_array = runtime
        .array_from_f32(&deterministic_values, shape)
        .expect("deterministic attention values should match their shape");
    runtime
        .astype(&float32_array, MlxDtype::BFloat16)
        .expect("attention inputs should cast to BF16")
}
