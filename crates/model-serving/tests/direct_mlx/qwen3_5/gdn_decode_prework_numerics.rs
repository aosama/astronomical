//! Fused GDN decode prework kernel numerics: bit-exact against the composed ops path.
//!
//! The composed one-token decode path between the projections and the
//! recurrence rolls the convolution state, runs the depthwise conv1d, applies
//! SiLU, splits q/k/v, applies two ones-weight RMS normalizations, and applies
//! two scalar scales — roughly thirteen small dispatches per gated-delta layer
//! that are launch-bound at decode shapes. The fused prework kernel replaces
//! that chain with one Metal launch. Because it changes the production decode
//! arithmetic path, this test proves the kernel reproduces the composed
//! reference bit for bit before any engagement decision is made.
//!
//! bf16/fp16 values convert to f32 losslessly, so comparing the f32 vectors
//! recovered through `to_vec_f32` is a bitwise comparison of the underlying
//! low-precision payloads. Values stay finite on purpose: the kernel's SiLU
//! arm claims parity over all finite inputs, matching the donor kernel's
//! sweep.

use std::time::Duration;

use astronomical_model_serving::{
    is_gdn_decode_prework_eligible, qwen3_5_gdn_decode_prework, qwen3_5_gdn_decode_prework_kernel,
};
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use tokio::time::timeout;

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};
use astronomical_mlx_c_rust::{MlxArray, MlxCompiledElementwiseGraphs, MlxDtype};

const KEY_HEAD_COUNT: i32 = 16;
const VALUE_HEAD_COUNT: i32 = 32;
const HEAD_DIMENSION: i32 = 128;
const KEY_DIMENSION: i32 = KEY_HEAD_COUNT * HEAD_DIMENSION;
const CONVOLUTION_DIMENSION: i32 = 2 * KEY_DIMENSION + VALUE_HEAD_COUNT * HEAD_DIMENSION;
const CONVOLUTION_KERNEL_DIMENSION: i32 = 4;
const KEPT_STATE_ROW_COUNT: i32 = CONVOLUTION_KERNEL_DIMENSION - 1;
const RMS_NORM_EPSILON: f32 = 1e-6;
const TEST_TIMEOUT: Duration = Duration::from_secs(115);

#[tokio::test]
async fn should_match_the_composed_decode_path_bit_for_bit_at_one_token() {
    timeout(
        TEST_TIMEOUT,
        compare_fused_prework_with_composed_path(1, MlxDtype::BFloat16),
    )
    .await
    .expect("the one-token prework numerics test must finish within 115 seconds");
}

#[tokio::test]
async fn should_match_the_composed_verify_path_bit_for_bit_at_two_tokens() {
    timeout(
        TEST_TIMEOUT,
        compare_fused_prework_with_composed_path(2, MlxDtype::BFloat16),
    )
    .await
    .expect("the two-token prework numerics test must finish within 115 seconds");
}

#[tokio::test]
async fn should_match_the_composed_verify_path_bit_for_bit_at_four_tokens() {
    timeout(
        TEST_TIMEOUT,
        compare_fused_prework_with_composed_path(4, MlxDtype::BFloat16),
    )
    .await
    .expect("the four-token prework numerics test must finish within 115 seconds");
}

/// The fused kernel engages only where bit parity is proven. bfloat16 (the
/// production dtype) has no native Metal arithmetic and is verified bit-exact;
/// float16 uses native half arithmetic whose `exp` intrinsic a separate fused
/// launch does not reproduce, so it must fall back to the composed path.
#[tokio::test]
async fn should_only_engage_the_fused_prework_for_the_verified_dtype() {
    let kernel = qwen3_5_gdn_decode_prework_kernel(RMS_NORM_EPSILON)
        .expect("the fused decode prework kernel should compile");
    assert!(
        is_gdn_decode_prework_eligible(Some(&kernel), 1, MlxDtype::BFloat16, HEAD_DIMENSION),
        "bfloat16 is the production dtype and must engage the fused prework"
    );
    assert!(
        !is_gdn_decode_prework_eligible(Some(&kernel), 1, MlxDtype::Float16, HEAD_DIMENSION),
        "float16 lacks proven bit parity and must fall back to the composed path"
    );
}

#[tokio::test]
async fn should_roll_two_consecutive_one_token_steps_like_the_composed_path() {
    timeout(TEST_TIMEOUT, compare_two_consecutive_decode_steps())
        .await
        .expect("the consecutive-step prework numerics test must finish within 115 seconds");
}

#[tokio::test]
async fn should_match_the_composed_decay_chain_bit_for_bit_at_one_token() {
    timeout(TEST_TIMEOUT, compare_compiled_decay_with_composed_chain())
        .await
        .expect("the one-token decay numerics test must finish within 115 seconds");
}

async fn compare_fused_prework_with_composed_path(token_count: i32, dtype: MlxDtype) {
    let (runtime, graphs, kernel) = prepare_runtime_and_kernel().await;
    let mixed_queries_keys_values = patterned_array(
        &runtime,
        &[1, token_count, CONVOLUTION_DIMENSION],
        0.004,
        -0.37,
        dtype,
    );
    let convolution_state = patterned_array(
        &runtime,
        &[1, KEPT_STATE_ROW_COUNT, CONVOLUTION_DIMENSION],
        0.0025,
        0.11,
        dtype,
    );
    let convolution_weight = convolution_weight_array(&runtime, dtype);
    let query_scale = scale_array(&runtime, HEAD_DIMENSION as f32, dtype);
    let key_scale = scale_array(&runtime, (HEAD_DIMENSION as f32).sqrt(), dtype);

    let fused = qwen3_5_gdn_decode_prework(
        &runtime,
        &kernel,
        KEY_HEAD_COUNT,
        VALUE_HEAD_COUNT,
        HEAD_DIMENSION,
        &mixed_queries_keys_values,
        &convolution_state,
        &convolution_weight,
        &query_scale,
        &key_scale,
    )
    .expect("the fused decode prework launch should succeed");
    let composed = composed_prework_reference(
        &runtime,
        &graphs,
        token_count,
        &mixed_queries_keys_values,
        &convolution_state,
        &convolution_weight,
        &query_scale,
        &key_scale,
    );

    let mut mismatch_report = Vec::new();
    collect_bitwise_mismatches(
        &runtime,
        &fused.queries,
        &composed.queries,
        "queries",
        &mut mismatch_report,
    );
    collect_bitwise_mismatches(
        &runtime,
        &fused.keys,
        &composed.keys,
        "keys",
        &mut mismatch_report,
    );
    collect_bitwise_mismatches(
        &runtime,
        &fused.values,
        &composed.values,
        "values",
        &mut mismatch_report,
    );
    collect_bitwise_mismatches(
        &runtime,
        &fused.next_convolution_state,
        &composed.next_convolution_state,
        "next convolution state",
        &mut mismatch_report,
    );
    assert!(
        mismatch_report.is_empty(),
        "the fused prework must match the composed path bitwise: {mismatch_report:#?}"
    );
}

async fn compare_two_consecutive_decode_steps() {
    let (runtime, graphs, kernel) = prepare_runtime_and_kernel().await;
    let dtype = MlxDtype::BFloat16;
    let convolution_weight = convolution_weight_array(&runtime, dtype);
    let query_scale = scale_array(&runtime, HEAD_DIMENSION as f32, dtype);
    let key_scale = scale_array(&runtime, (HEAD_DIMENSION as f32).sqrt(), dtype);
    let mut fused_state = patterned_array(
        &runtime,
        &[1, KEPT_STATE_ROW_COUNT, CONVOLUTION_DIMENSION],
        0.0025,
        0.11,
        dtype,
    );
    let mut composed_state = patterned_array(
        &runtime,
        &[1, KEPT_STATE_ROW_COUNT, CONVOLUTION_DIMENSION],
        0.0025,
        0.11,
        dtype,
    );

    for step_index in 0..2 {
        let step_queries_keys_values = patterned_array(
            &runtime,
            &[1, 1, CONVOLUTION_DIMENSION],
            0.004,
            -0.37 + step_index as f32 * 0.2,
            dtype,
        );
        let fused = qwen3_5_gdn_decode_prework(
            &runtime,
            &kernel,
            KEY_HEAD_COUNT,
            VALUE_HEAD_COUNT,
            HEAD_DIMENSION,
            &step_queries_keys_values,
            &fused_state,
            &convolution_weight,
            &query_scale,
            &key_scale,
        )
        .expect("the fused decode prework launch should succeed");
        let composed = composed_prework_reference(
            &runtime,
            &graphs,
            1,
            &step_queries_keys_values,
            &composed_state,
            &convolution_weight,
            &query_scale,
            &key_scale,
        );
        assert_bitwise_equal(&runtime, &fused.queries, &composed.queries, "queries");
        assert_bitwise_equal(&runtime, &fused.keys, &composed.keys, "keys");
        assert_bitwise_equal(&runtime, &fused.values, &composed.values, "values");
        assert_bitwise_equal(
            &runtime,
            &fused.next_convolution_state,
            &composed.next_convolution_state,
            "next convolution state",
        );
        fused_state = fused.next_convolution_state;
        composed_state = composed.next_convolution_state;
    }
}

async fn compare_compiled_decay_with_composed_chain() {
    let (runtime, graphs, _kernel) = prepare_runtime_and_kernel().await;
    let decay_inputs = patterned_array(
        &runtime,
        &[1, 1, VALUE_HEAD_COUNT],
        0.03,
        -0.4,
        MlxDtype::BFloat16,
    );
    let decay_rate_logarithm = runtime
        .astype(
            &runtime
                .array_from_f32(&[-0.2; VALUE_HEAD_COUNT as usize], &[VALUE_HEAD_COUNT])
                .expect("the decay rate logarithm should be valid"),
            MlxDtype::Float32,
        )
        .expect("the decay rate logarithm should stay float32");
    let decay_interval_bias = patterned_array(
        &runtime,
        &[VALUE_HEAD_COUNT],
        0.002,
        0.05,
        MlxDtype::BFloat16,
    );

    let compiled = runtime
        .apply_compiled_gated_delta_decay(
            &graphs,
            &decay_rate_logarithm,
            &decay_inputs,
            &decay_interval_bias,
        )
        .expect("the compiled decay graph should apply");

    let composed_bias_inputs = runtime
        .add(&decay_inputs, &decay_interval_bias)
        .expect("the biased decay inputs should be valid");
    let composed_intervals = runtime
        .softplus(&composed_bias_inputs)
        .expect("the composed softplus should be valid");
    let composed_rates = runtime
        .exp(&decay_rate_logarithm)
        .expect("the composed exponential should be valid");
    let composed_products = runtime
        .multiply(&composed_rates, &composed_intervals)
        .expect("the composed decay products should be valid");
    let composed = runtime
        .exp(
            &runtime
                .negative(&composed_products)
                .expect("the composed negation should be valid"),
        )
        .expect("the composed decay should be valid");

    assert_bitwise_equal(&runtime, &compiled, &composed, "decays");
}

async fn prepare_runtime_and_kernel() -> (
    MlxRuntime,
    MlxCompiledElementwiseGraphs,
    astronomical_mlx_c_rust::MlxMetalKernel,
) {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the prework numerics test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    let compiled_elementwise_graphs =
        MlxCompiledElementwiseGraphs::new().expect("the compiled elementwise graphs should build");
    let kernel = qwen3_5_gdn_decode_prework_kernel(RMS_NORM_EPSILON)
        .expect("the fused decode prework kernel should compile");
    (runtime, compiled_elementwise_graphs, kernel)
}

/// The composed production arithmetic the kernel must reproduce: rolling
/// concat, depthwise conv1d, SiLU, q/k/v split, ones-weight RMS norms, the
/// two scalar scales, and the trailing-state slice that becomes the next
/// rolling buffer.
fn composed_prework_reference(
    runtime: &MlxRuntime,
    compiled_elementwise_graphs: &MlxCompiledElementwiseGraphs,
    token_count: i32,
    mixed_queries_keys_values: &MlxArray,
    convolution_state: &MlxArray,
    convolution_weight: &MlxArray,
    query_scale: &MlxArray,
    key_scale: &MlxArray,
) -> ComposedPreworkReference {
    let concatenated = runtime
        .concatenate_axis(&[convolution_state, mixed_queries_keys_values], 1)
        .expect("the concatenated convolution input should be valid");
    let convolution_output = runtime
        .conv1d(
            &concatenated,
            convolution_weight,
            1,
            0,
            1,
            CONVOLUTION_DIMENSION,
        )
        .expect("the depthwise convolution should be valid");
    let activated = runtime
        .apply_compiled_silu(compiled_elementwise_graphs, &convolution_output)
        .expect("the compiled SiLU should be valid");
    let queries = slice_reshape(
        runtime,
        &activated,
        0,
        KEY_DIMENSION,
        token_count,
        KEY_HEAD_COUNT,
        HEAD_DIMENSION,
    );
    let keys = slice_reshape(
        runtime,
        &activated,
        KEY_DIMENSION,
        KEY_DIMENSION * 2,
        token_count,
        KEY_HEAD_COUNT,
        HEAD_DIMENSION,
    );
    let values = slice_reshape(
        runtime,
        &activated,
        KEY_DIMENSION * 2,
        CONVOLUTION_DIMENSION,
        token_count,
        VALUE_HEAD_COUNT,
        HEAD_DIMENSION,
    );
    let queries = normalized_and_scaled(runtime, &queries, query_scale);
    let keys = normalized_and_scaled(runtime, &keys, key_scale);
    let next_convolution_state = runtime
        .slice(
            &concatenated,
            &[0, token_count, 0],
            &[1, token_count + KEPT_STATE_ROW_COUNT, CONVOLUTION_DIMENSION],
            &[1, 1, 1],
        )
        .expect("the next convolution state slice should be valid");
    ComposedPreworkReference {
        queries,
        keys,
        values,
        next_convolution_state,
    }
}

struct ComposedPreworkReference {
    queries: MlxArray,
    keys: MlxArray,
    values: MlxArray,
    next_convolution_state: MlxArray,
}

fn slice_reshape(
    runtime: &MlxRuntime,
    activated: &MlxArray,
    channel_start: i32,
    channel_stop: i32,
    token_count: i32,
    head_count: i32,
    head_dimension: i32,
) -> MlxArray {
    let sliced = runtime
        .slice(
            activated,
            &[0, 0, channel_start],
            &[1, token_count, channel_stop],
            &[1, 1, 1],
        )
        .expect("the activated slice should be valid");
    runtime
        .reshape(&sliced, &[1, token_count, head_count, head_dimension])
        .expect("the activated slice should reshape to heads")
}

fn normalized_and_scaled(
    runtime: &MlxRuntime,
    head_values: &MlxArray,
    scale: &MlxArray,
) -> MlxArray {
    let normalized = runtime
        .rms_norm_without_weight(head_values, RMS_NORM_EPSILON)
        .expect("the ones-weight RMS normalization should be valid");
    runtime
        .multiply(&normalized, scale)
        .expect("the normalized head values should scale")
}

/// The bf16 scalar the production model bakes from the head dimension —
/// shared by both paths so the comparison cannot diverge on scale bits.
fn scale_array(runtime: &MlxRuntime, scale_divisor: f32, dtype: MlxDtype) -> MlxArray {
    runtime
        .astype(
            &runtime
                .array_from_f32(&[scale_divisor.recip()], &[])
                .expect("the scale scalar should be valid"),
            dtype,
        )
        .expect("the scale scalar should cast to the activation dtype")
}

/// Records where two arrays' low-precision payloads diverge. bf16/fp16 values
/// convert to f32 losslessly, so f32 vector equality is bitwise equality of
/// the underlying payload.
fn collect_bitwise_mismatches(
    runtime: &MlxRuntime,
    fused: &MlxArray,
    composed: &MlxArray,
    label: &str,
    mismatch_report: &mut Vec<(String, usize, u32, u32)>,
) {
    let fused_values = runtime
        .astype(fused, MlxDtype::Float32)
        .expect("the fused output should cast to float32 for comparison")
        .to_vec_f32()
        .expect("the fused output should evaluate as float32");
    let composed_values = runtime
        .astype(composed, MlxDtype::Float32)
        .expect("the composed output should cast to float32 for comparison")
        .to_vec_f32()
        .expect("the composed output should evaluate as float32");
    for (value_index, (fused_value, composed_value)) in
        fused_values.iter().zip(composed_values.iter()).enumerate()
    {
        if fused_value.to_bits() != composed_value.to_bits() {
            mismatch_report.push((
                label.to_owned(),
                value_index,
                fused_value.to_bits(),
                composed_value.to_bits(),
            ));
        }
    }
}

/// bf16/fp16 payloads convert to f32 losslessly, so f32 vector equality is
/// bitwise equality of the low-precision payload.
fn assert_bitwise_equal(runtime: &MlxRuntime, fused: &MlxArray, composed: &MlxArray, label: &str) {
    assert_eq!(
        fused.shape(),
        composed.shape(),
        "the fused and composed {label} shapes must match"
    );
    assert_eq!(
        fused.dtype(),
        composed.dtype(),
        "the fused and composed {label} dtypes must match"
    );
    let fused_values = runtime
        .astype(fused, MlxDtype::Float32)
        .expect("the fused output should cast to float32 for comparison")
        .to_vec_f32()
        .expect("the fused output should evaluate as float32");
    let composed_values = runtime
        .astype(composed, MlxDtype::Float32)
        .expect("the composed output should cast to float32 for comparison")
        .to_vec_f32()
        .expect("the composed output should evaluate as float32");
    let mismatch_index = fused_values.iter().zip(composed_values.iter()).position(
        |(fused_value, composed_value)| fused_value.to_bits() != composed_value.to_bits(),
    );
    assert!(
        mismatch_index.is_none(),
        "the fused and composed {label} must be bitwise identical; first mismatch at index {mismatch_index:?}: fused {:#x} versus composed {:#x}",
        mismatch_index
            .map(|index| fused_values[index].to_bits())
            .unwrap_or(0),
        mismatch_index
            .map(|index| composed_values[index].to_bits())
            .unwrap_or(0),
    );
}

/// Deterministic mixed-magnitude finite pattern covering small magnitudes,
/// unit values, large magnitudes, and sign flips.
fn patterned_array(
    runtime: &MlxRuntime,
    shape: &[i32],
    base_step: f32,
    base_offset: f32,
    dtype: MlxDtype,
) -> MlxArray {
    let element_count: usize = shape.iter().map(|dimension| *dimension as usize).product();
    let sample_values: Vec<f32> = (0..element_count)
        .map(|element_index| {
            let phase = (element_index % 97) as f32;
            let magnitude = base_offset + phase * base_step;
            let signed = if element_index % 3 == 0 {
                -magnitude
            } else {
                magnitude
            };
            let large = (element_index % 211 == 0).then(|| signed * 3000.0);
            let tiny = (element_index % 43 == 0).then(|| signed * 1e-5);
            tiny.or(large).unwrap_or(signed)
        })
        .collect();
    runtime
        .astype(
            &runtime
                .array_from_f32(&sample_values, shape)
                .expect("the patterned array should be valid"),
            dtype,
        )
        .expect("the patterned array should cast to the requested dtype")
}

fn convolution_weight_array(runtime: &MlxRuntime, dtype: MlxDtype) -> MlxArray {
    let weight_values: Vec<f32> = (0..(CONVOLUTION_DIMENSION * CONVOLUTION_KERNEL_DIMENSION)
        as usize)
        .map(|element_index| ((element_index % 17) as f32 - 8.0) * 0.1)
        .collect();
    runtime
        .astype(
            &runtime
                .array_from_f32(
                    &weight_values,
                    &[CONVOLUTION_DIMENSION, CONVOLUTION_KERNEL_DIMENSION, 1],
                )
                .expect("the convolution weight should be valid"),
            dtype,
        )
        .expect("the convolution weight should cast to the requested dtype")
}
