//! Linear-attention prefill norm-scale folding numerics: bit-exact against the
//! composed norm-then-multiply pair.
//!
//! The prefill composed path in `convolution_to_normalized_heads` folds each
//! scalar scale into `fast_rms_norm`'s per-channel weight, replacing the norm
//! kernel plus the separate broadcast-multiply kernel with one fused launch
//! (issue #915 item 5). MLX's own reference for the weighted normalization
//! computes `round(normalize(x)) * weight`, the same arithmetic and rounding
//! order as the former pair, so this test pins bit-for-bit parity across the
//! production bfloat16 dtype and the real prefill chunk shapes before the
//! folded dispatch engages.
//!
//! bf16 values convert to f32 losslessly, so comparing the f32 vectors
//! recovered through `to_vec_f32` is a bitwise comparison of the underlying
//! low-precision payloads. Values stay finite on purpose.

use std::time::Duration;

use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxMemoryLimits, MlxRuntime};
use tokio::time::timeout;

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

const KEY_HEAD_COUNT: i32 = 16;
const HEAD_DIMENSION: i32 = 128;
const RMS_NORM_EPSILON: f32 = 1e-6;
const TEST_TIMEOUT: Duration = Duration::from_secs(60);

#[tokio::test]
async fn should_match_the_norm_then_multiply_pair_bit_for_bit_at_production_shapes() {
    timeout(
        TEST_TIMEOUT,
        compare_folded_weight_with_norm_then_multiply_pair(),
    )
    .await
    .expect("the norm-scale numerics test must finish within its timeout");
}

async fn compare_folded_weight_with_norm_then_multiply_pair() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the norm-scale test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");

    // The production scalar scales from the head dimension, bfloat16-rounded
    // exactly as the model loader bakes them.
    let head_dimension = HEAD_DIMENSION as f32;
    let query_scalar_scale = scalar_scale_array(&runtime, head_dimension.recip());
    let key_scalar_scale = scalar_scale_array(&runtime, head_dimension.sqrt().recip());
    // The folded per-channel weights: every channel carries the same f32 value
    // through the same bfloat16 rounding as the scalar scale.
    let query_scale_weight = per_channel_scale_weight(&runtime, head_dimension.recip());
    let key_scale_weight = per_channel_scale_weight(&runtime, head_dimension.sqrt().recip());
    assert_weight_channels_equal_scalar(
        &runtime,
        &query_scale_weight,
        &query_scalar_scale,
        "query weight channels versus scalar scale",
    );
    assert_weight_channels_equal_scalar(
        &runtime,
        &key_scale_weight,
        &key_scalar_scale,
        "key weight channels versus scalar scale",
    );

    for token_count in [1, 3, 2_048] {
        let queries = patterned_head_array(&runtime, token_count, 0.004, -0.37);
        let keys = patterned_head_array(&runtime, token_count, 0.0025, 0.11);

        let folded_queries = runtime
            .rms_norm(&queries, &query_scale_weight, RMS_NORM_EPSILON)
            .expect("the folded query normalization should be valid");
        let folded_keys = runtime
            .rms_norm(&keys, &key_scale_weight, RMS_NORM_EPSILON)
            .expect("the folded key normalization should be valid");
        let composed_queries = normalized_then_scaled(&runtime, &queries, &query_scalar_scale);
        let composed_keys = normalized_then_scaled(&runtime, &keys, &key_scalar_scale);

        assert_bitwise_equal(
            &runtime,
            &folded_queries,
            &composed_queries,
            &format!("queries at {token_count} tokens"),
        );
        assert_bitwise_equal(
            &runtime,
            &folded_keys,
            &composed_keys,
            &format!("keys at {token_count} tokens"),
        );
    }
}

/// The former composed pair: unweighted RMS normalization, then the separate
/// scalar-scale multiply.
fn normalized_then_scaled(
    runtime: &MlxRuntime,
    head_values: &MlxArray,
    scale: &MlxArray,
) -> MlxArray {
    let normalized = runtime
        .rms_norm_without_weight(head_values, RMS_NORM_EPSILON)
        .expect("the unweighted RMS normalization should be valid");
    runtime
        .multiply(&normalized, scale)
        .expect("the normalized head values should scale")
}

/// The production bfloat16 scalar the model loader bakes from the head
/// dimension.
fn scalar_scale_array(runtime: &MlxRuntime, scale_value: f32) -> MlxArray {
    runtime
        .astype(
            &runtime
                .array_from_f32(&[scale_value], &[])
                .expect("the scale scalar should be valid"),
            MlxDtype::BFloat16,
        )
        .expect("the scale scalar should cast to bfloat16")
}

/// The folded per-channel weight: the same f32 scale value in every channel of
/// one `[head_dimension]` bfloat16 vector, cast exactly like the scalar.
fn per_channel_scale_weight(runtime: &MlxRuntime, scale_value: f32) -> MlxArray {
    runtime
        .astype(
            &runtime
                .array_from_f32(
                    &vec![scale_value; HEAD_DIMENSION as usize],
                    &[HEAD_DIMENSION],
                )
                .expect("the per-channel scale weight should be valid"),
            MlxDtype::BFloat16,
        )
        .expect("the per-channel scale weight should cast to bfloat16")
}

/// Proves every folded weight channel carries the scalar scale's exact bits,
/// so the folded launch cannot diverge from the pair on scale values.
fn assert_weight_channels_equal_scalar(
    runtime: &MlxRuntime,
    weight: &MlxArray,
    scalar_scale: &MlxArray,
    label: &str,
) {
    let scalar_values = runtime
        .astype(scalar_scale, MlxDtype::Float32)
        .expect("the scalar scale should cast to float32 for comparison")
        .to_vec_f32()
        .expect("the scalar scale should evaluate as float32");
    assert_eq!(
        scalar_values.len(),
        1,
        "the {label} scalar scale must hold exactly one value"
    );
    let weight_values = runtime
        .astype(weight, MlxDtype::Float32)
        .expect("the scale weight should cast to float32 for comparison")
        .to_vec_f32()
        .expect("the scale weight should evaluate as float32");
    assert_eq!(
        weight_values.len(),
        HEAD_DIMENSION as usize,
        "the {label} weight must carry one channel per head dimension"
    );
    let mismatch_index = weight_values
        .iter()
        .position(|weight_value| weight_value.to_bits() != scalar_values[0].to_bits());
    assert!(
        mismatch_index.is_none(),
        "the {label} weight channels must equal the scalar scale bitwise; first mismatch at channel index {mismatch_index:?}"
    );
}

/// Deterministic mixed-magnitude finite pattern covering small magnitudes,
/// unit values, large magnitudes, and sign flips.
fn patterned_head_array(
    runtime: &MlxRuntime,
    token_count: i32,
    base_step: f32,
    base_offset: f32,
) -> MlxArray {
    let shape = [1, token_count, KEY_HEAD_COUNT, HEAD_DIMENSION];
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
            if element_index % 11 == 0 {
                signed * 41.0
            } else {
                signed
            }
        })
        .collect();
    runtime
        .astype(
            &runtime
                .array_from_f32(&sample_values, &shape)
                .expect("the patterned head array should be valid"),
            MlxDtype::BFloat16,
        )
        .expect("the patterned head array should cast to bfloat16")
}

/// bf16 payloads convert to f32 losslessly, so f32 vector equality is bitwise
/// equality of the low-precision payload.
fn assert_bitwise_equal(runtime: &MlxRuntime, folded: &MlxArray, composed: &MlxArray, label: &str) {
    assert_eq!(
        folded.shape(),
        composed.shape(),
        "the folded and composed {label} shapes must match"
    );
    assert_eq!(
        folded.dtype(),
        composed.dtype(),
        "the folded and composed {label} dtypes must match"
    );
    let folded_values = runtime
        .astype(folded, MlxDtype::Float32)
        .expect("the folded output should cast to float32 for comparison")
        .to_vec_f32()
        .expect("the folded output should evaluate as float32");
    let composed_values = runtime
        .astype(composed, MlxDtype::Float32)
        .expect("the composed output should cast to float32 for comparison")
        .to_vec_f32()
        .expect("the composed output should evaluate as float32");
    let mismatch_index = folded_values.iter().zip(composed_values.iter()).position(
        |(folded_value, composed_value)| folded_value.to_bits() != composed_value.to_bits(),
    );
    assert!(
        mismatch_index.is_none(),
        "the folded and composed {label} must be bitwise identical; first mismatch at index {mismatch_index:?}: folded {:#x} versus composed {:#x}",
        mismatch_index
            .map(|index| folded_values[index].to_bits())
            .unwrap_or(0),
        mismatch_index
            .map(|index| composed_values[index].to_bits())
            .unwrap_or(0),
    );
}
