//! Vision attention padding-zero cache contract: bit-exact against per-call zeros.
//!
//! The vision tower pads Q/K/V head dimensions from 72 to the fused SDPA width
//! in every one of its 27 blocks, and the padded tensors differ only by the
//! attention states — the appended zeros are identical every time (issue #915
//! item 4i). The retained cache serves those zeros from one array instead of
//! re-allocating them per pad call. This contract proves the cached padding is
//! bitwise identical to the former per-call zeros-plus-concatenate path and
//! that a changed padding shape replaces the retained array correctly.
//!
//! bf16 values convert to f32 losslessly, so comparing the f32 vectors
//! recovered through `to_vec_f32` is a bitwise comparison of the underlying
//! low-precision payloads. Values stay finite on purpose.

use std::time::Duration;

use astronomical_model_serving::Qwen3_5VisionPaddingZeroCache;
use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use tokio::time::timeout;

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};
use astronomical_mlx_c_rust::{MlxArray, MlxDtype};

const HEAD_COUNT: i32 = 16;
const HEAD_DIMENSION: i32 = 72;
const FUSED_HEAD_DIMENSION: i32 = 80;
const WIDER_FUSED_HEAD_DIMENSION: i32 = 128;
const RMS_PAD_TEST_TIMEOUT: Duration = Duration::from_secs(60);

#[tokio::test]
async fn should_pad_bitwise_identically_to_the_per_call_zeros_path() {
    timeout(
        RMS_PAD_TEST_TIMEOUT,
        compare_cached_padding_with_per_call_zeros(),
    )
    .await
    .expect("the padding cache contract test must finish within its timeout");
}

async fn compare_cached_padding_with_per_call_zeros() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("the padding cache test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize");
    let padding_zero_cache = Qwen3_5VisionPaddingZeroCache::new();

    for patch_count in [64, 129] {
        for component_index in 0..3 {
            let attention_states = patterned_head_array(
                &runtime,
                patch_count,
                0.004,
                -0.37 + component_index as f32 * 0.11,
            );
            let cached_padded = padding_zero_cache
                .pad_attention_head_dimension(
                    &runtime,
                    &attention_states,
                    HEAD_COUNT,
                    patch_count,
                    FUSED_HEAD_DIMENSION,
                )
                .expect("the cached padding should build");
            let per_call_padded = per_call_zeros_padding(
                &runtime,
                &attention_states,
                HEAD_COUNT,
                patch_count,
                FUSED_HEAD_DIMENSION,
            );
            assert_bitwise_equal(
                &runtime,
                &cached_padded,
                &per_call_padded,
                &format!("padded component {component_index} at {patch_count} patches"),
            );
        }
    }

    // A different fused width must replace the retained zeros, not reuse them.
    let attention_states = patterned_head_array(&runtime, 64, 0.004, -0.37);
    let cached_wider_padded = padding_zero_cache
        .pad_attention_head_dimension(
            &runtime,
            &attention_states,
            HEAD_COUNT,
            64,
            WIDER_FUSED_HEAD_DIMENSION,
        )
        .expect("the wider cached padding should build");
    let per_call_wider_padded = per_call_zeros_padding(
        &runtime,
        &attention_states,
        HEAD_COUNT,
        64,
        WIDER_FUSED_HEAD_DIMENSION,
    );
    assert_bitwise_equal(
        &runtime,
        &cached_wider_padded,
        &per_call_wider_padded,
        "padded component at the wider fused width",
    );

    // An already-fused head dimension takes the identity path without padding.
    let fused_states =
        patterned_head_array_with_dimension(&runtime, 64, WIDER_FUSED_HEAD_DIMENSION, 0.004, -0.37);
    let identity_padded = padding_zero_cache
        .pad_attention_head_dimension(
            &runtime,
            &fused_states,
            HEAD_COUNT,
            64,
            WIDER_FUSED_HEAD_DIMENSION,
        )
        .expect("the identity padding should build");
    assert_bitwise_equal(
        &runtime,
        &identity_padded,
        &fused_states,
        "identity-shaped attention states",
    );
}

/// The former production sequence: a fresh zeros tensor per pad call, then the
/// head-dimension concatenate.
fn per_call_zeros_padding(
    runtime: &MlxRuntime,
    attention_states: &MlxArray,
    head_count: i32,
    patch_count: i32,
    fused_head_dimension: i32,
) -> MlxArray {
    let head_dimension = attention_states.shape()[3];
    if head_dimension == fused_head_dimension {
        return runtime
            .reshape(attention_states, &attention_states.shape())
            .expect("the identity reshape should be valid");
    }
    let padding_states = runtime
        .zeros(
            &[
                1,
                head_count,
                patch_count,
                fused_head_dimension - head_dimension,
            ],
            attention_states.dtype(),
        )
        .expect("the per-call padding zeros should be valid");
    runtime
        .concatenate_axis(&[attention_states, &padding_states], 3)
        .expect("the per-call padding should concatenate")
}

/// Deterministic mixed-magnitude finite pattern covering small magnitudes,
/// unit values, large magnitudes, and sign flips.
fn patterned_head_array(
    runtime: &MlxRuntime,
    patch_count: i32,
    base_step: f32,
    base_offset: f32,
) -> MlxArray {
    patterned_head_array_with_dimension(
        runtime,
        patch_count,
        HEAD_DIMENSION,
        base_step,
        base_offset,
    )
}

fn patterned_head_array_with_dimension(
    runtime: &MlxRuntime,
    patch_count: i32,
    head_dimension: i32,
    base_step: f32,
    base_offset: f32,
) -> MlxArray {
    let shape = [1, HEAD_COUNT, patch_count, head_dimension];
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
fn assert_bitwise_equal(runtime: &MlxRuntime, cached: &MlxArray, per_call: &MlxArray, label: &str) {
    assert_eq!(
        cached.shape(),
        per_call.shape(),
        "the cached and per-call {label} shapes must match"
    );
    assert_eq!(
        cached.dtype(),
        per_call.dtype(),
        "the cached and per-call {label} dtypes must match"
    );
    let cached_values = runtime
        .astype(cached, MlxDtype::Float32)
        .expect("the cached output should cast to float32 for comparison")
        .to_vec_f32()
        .expect("the cached output should evaluate as float32");
    let per_call_values = runtime
        .astype(per_call, MlxDtype::Float32)
        .expect("the per-call output should cast to float32 for comparison")
        .to_vec_f32()
        .expect("the per-call output should evaluate as float32");
    let mismatch_index = cached_values.iter().zip(per_call_values.iter()).position(
        |(cached_value, per_call_value)| cached_value.to_bits() != per_call_value.to_bits(),
    );
    assert!(
        mismatch_index.is_none(),
        "the cached and per-call {label} must be bitwise identical; first mismatch at index {mismatch_index:?}"
    );
}
