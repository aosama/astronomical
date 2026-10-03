//! Incremental SSD block-restore journeys for K2 Horizon MoVA KV states.
//!
//! Proves the restore peak stays at the final destination plus one block:
//! active memory must not accumulate loaded blocks across absorbs, and the
//! seated slabs must match a bulk concatenation reference bit-for-bit (full
//! precision) or within the affine quantization tolerance (quantized).

use astronomical_model_serving::K2HorizonMoVAKvState;
use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxMemoryLimits, MlxRuntime};

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
};

const BLOCK_TOKEN_COUNT: i32 = 128;
const COMPLETE_BLOCK_COUNT: usize = 4;
const RESTORED_TOKEN_COUNT: i32 = BLOCK_TOKEN_COUNT * COMPLETE_BLOCK_COUNT as i32;
const KEY_VALUE_HEAD_COUNT: i32 = 8;
const HEAD_DIMENSION: i32 = 128;
const GROWTH_TOKENS: i32 = 128;
const DEQUANTIZE_TOLERANCE: f32 = 0.15;

fn test_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_ALLOCATOR_CACHE_MEMORY_LIMIT_BYTES,
        )
        .expect("K2 restore test memory limits should be valid"),
    )
    .expect("the direct MLX runtime should initialize")
}

/// Builds one distinct bfloat16 sequence block; the phase offset makes every
/// block's payload unique so absorb-order mistakes cannot cancel out.
fn bf16_block(runtime: &MlxRuntime, block_index: usize) -> (MlxArray, MlxArray) {
    let element_count = (KEY_VALUE_HEAD_COUNT * BLOCK_TOKEN_COUNT * HEAD_DIMENSION) as usize;
    let key_values: Vec<f32> = (0..element_count)
        .map(|element_index| ((element_index as f32) * 0.01 + block_index as f32).sin() * 0.5)
        .collect();
    let value_values: Vec<f32> = (0..element_count)
        .map(|element_index| ((element_index as f32) * 0.02 + block_index as f32 + 0.5).cos() * 0.5)
        .collect();
    let keys = bf16_tensor(runtime, &key_values, "keys");
    let values = bf16_tensor(runtime, &value_values, "values");
    (keys, values)
}

fn bf16_tensor(runtime: &MlxRuntime, host_values: &[f32], tensor_role: &'static str) -> MlxArray {
    runtime
        .array_from_f32(
            host_values,
            &[1, KEY_VALUE_HEAD_COUNT, BLOCK_TOKEN_COUNT, HEAD_DIMENSION],
        )
        .and_then(|float_array| runtime.astype(&float_array, MlxDtype::BFloat16))
        .unwrap_or_else(|build_error| {
            panic!("the {tensor_role} block tensor should build: {build_error}")
        })
}

fn full_precision_cache() -> K2HorizonMoVAKvState {
    K2HorizonMoVAKvState::build(false, GROWTH_TOKENS as u32)
        .expect("the full-precision K2 cache should build")
}

fn quantized_cache() -> K2HorizonMoVAKvState {
    K2HorizonMoVAKvState::build(true, GROWTH_TOKENS as u32)
        .expect("the quantized K2 cache should build")
}

fn restore_all_blocks(runtime: &MlxRuntime, cache: &mut K2HorizonMoVAKvState) {
    for block_index in 0..COMPLETE_BLOCK_COUNT {
        let (block_keys, block_values) = bf16_block(runtime, block_index);
        let block_start_tokens = block_index as i32 * BLOCK_TOKEN_COUNT;
        if block_index == 0 {
            cache
                .begin_incremental_block_restore(
                    runtime,
                    block_keys,
                    block_values,
                    RESTORED_TOKEN_COUNT,
                )
                .expect("the first block should begin the incremental restore");
        } else {
            cache
                .absorb_incremental_block_restore(
                    runtime,
                    block_keys,
                    block_values,
                    block_start_tokens,
                )
                .expect("each later block should absorb at its token range");
        }
    }
    cache
        .finish_incremental_block_restore(RESTORED_TOKEN_COUNT)
        .expect("the restore should finish at the restored token count");
}

fn active_memory_bytes(runtime: &MlxRuntime) -> u64 {
    runtime
        .memory_snapshot()
        .expect("the MLX memory snapshot should sample")
        .active_memory_bytes() as u64
}

fn evaluate_seated_slabs(runtime: &MlxRuntime, cache: &K2HorizonMoVAKvState) {
    let (seated_keys, seated_values) = match cache {
        K2HorizonMoVAKvState::FullPrecision(state) => (state.keys_state(), state.values_state()),
        K2HorizonMoVAKvState::Quantized(state) => {
            (state.quantized_keys_state(), state.quantized_values_state())
        }
    };
    if let (Some(seated_keys), Some(seated_values)) = (seated_keys, seated_values) {
        runtime
            .evaluate_arrays(&[seated_keys, seated_values])
            .expect("the seated slabs should materialize");
    }
}

fn bf16_reference_vec(runtime: &MlxRuntime, array: &MlxArray) -> Vec<f32> {
    runtime
        .astype(array, MlxDtype::Float32)
        .and_then(|float_array| float_array.to_vec_f32())
        .expect("the bfloat16 slab should read back as float32")
}

/// Reads back only the restored prefix of a seated slab: the slab keeps
/// growth-headroom capacity beyond the restored tokens, and that tail would
/// misalign a flat element-wise comparison against the restored-length
/// reference.
fn seated_reference_vec(
    runtime: &MlxRuntime,
    seated_slab: &MlxArray,
    restored_token_count: i32,
) -> Vec<f32> {
    let slab_shape = seated_slab.shape();
    let restored_slab = runtime
        .slice(
            seated_slab,
            &[0, 0, 0, 0],
            &[
                slab_shape[0],
                slab_shape[1],
                restored_token_count,
                slab_shape[3],
            ],
            &[1, 1, 1, 1],
        )
        .expect("the seated slab should slice to the restored prefix");
    bf16_reference_vec(runtime, &restored_slab)
}

#[tokio::test]
async fn should_seat_incrementally_restored_full_precision_slabs_matching_the_bulk_reference() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();

    let mut cache = full_precision_cache();
    restore_all_blocks(&runtime, &mut cache);

    let K2HorizonMoVAKvState::FullPrecision(seated_state) = &cache else {
        panic!("the full-precision cache should stay full precision");
    };
    assert_eq!(
        seated_state.offset_tokens(),
        RESTORED_TOKEN_COUNT,
        "the restored slab must hold exactly the restored prefix"
    );
    assert_eq!(
        seated_state.capacity_tokens(),
        RESTORED_TOKEN_COUNT + GROWTH_TOKENS,
        "the restored slab must keep exactly one growth step of headroom"
    );

    // The seated slabs must equal one bulk concatenation of the same blocks.
    let block_arrays: Vec<(MlxArray, MlxArray)> = (0..COMPLETE_BLOCK_COUNT)
        .map(|block_index| bf16_block(&runtime, block_index))
        .collect();
    let key_refs: Vec<&MlxArray> = block_arrays.iter().map(|(keys, _)| keys).collect();
    let value_refs: Vec<&MlxArray> = block_arrays.iter().map(|(_, values)| values).collect();
    let reference_keys = runtime
        .concatenate_axis(&key_refs, 2)
        .expect("the reference keys should concatenate");
    let reference_values = runtime
        .concatenate_axis(&value_refs, 2)
        .expect("the reference values should concatenate");
    let seated_keys = seated_state
        .keys_state()
        .expect("the seated keys should be present");
    let seated_values = seated_state
        .values_state()
        .expect("the seated values should be present");
    assert_eq!(
        seated_reference_vec(&runtime, seated_keys, RESTORED_TOKEN_COUNT),
        bf16_reference_vec(&runtime, &reference_keys),
        "the seated keys must match the bulk concatenation bit-for-bit"
    );
    assert_eq!(
        seated_reference_vec(&runtime, seated_values, RESTORED_TOKEN_COUNT),
        bf16_reference_vec(&runtime, &reference_values),
        "the seated values must match the bulk concatenation bit-for-bit"
    );
}

#[tokio::test]
async fn should_keep_incremental_restore_active_memory_at_one_block_above_the_destination() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();

    let block_byte_count = 2_u64
        * (KEY_VALUE_HEAD_COUNT as u64)
        * (BLOCK_TOKEN_COUNT as u64)
        * (HEAD_DIMENSION as u64)
        * 2;
    let mut cache = full_precision_cache();
    runtime
        .reset_peak_memory()
        .expect("the MLX peak counter should reset so earlier tests do not inflate it");
    let (first_block_keys, first_block_values) = bf16_block(&runtime, 0);
    cache
        .begin_incremental_block_restore(
            &runtime,
            first_block_keys,
            first_block_values,
            RESTORED_TOKEN_COUNT,
        )
        .expect("the first block should begin the incremental restore");
    evaluate_seated_slabs(&runtime, &cache);
    let destination_active_bytes = active_memory_bytes(&runtime);

    for block_index in 1..COMPLETE_BLOCK_COUNT {
        let (block_keys, block_values) = bf16_block(&runtime, block_index);
        cache
            .absorb_incremental_block_restore(
                &runtime,
                block_keys,
                block_values,
                block_index as i32 * BLOCK_TOKEN_COUNT,
            )
            .expect("each later block should absorb at its token range");
        evaluate_seated_slabs(&runtime, &cache);
        let after_absorb_bytes = active_memory_bytes(&runtime);
        assert!(
            after_absorb_bytes
                <= destination_active_bytes + block_byte_count + block_byte_count / 2,
            "absorbing block {block_index} must not accumulate loaded blocks: \
             destination={destination_active_bytes} after_absorb={after_absorb_bytes} \
             block_bytes={block_byte_count}"
        );
    }
    cache
        .finish_incremental_block_restore(RESTORED_TOKEN_COUNT)
        .expect("the restore should finish at the restored token count");

    let peak_bytes = runtime
        .memory_snapshot()
        .expect("the MLX memory snapshot should sample")
        .peak_memory_bytes() as u64;
    let bulk_restore_bound_bytes = destination_active_bytes + 3 * block_byte_count;
    assert!(
        peak_bytes <= bulk_restore_bound_bytes,
        "the incremental restore peak must stay below the bulk all-blocks bound: \
         peak={peak_bytes} bulk_bound={bulk_restore_bound_bytes}"
    );
}

#[tokio::test]
async fn should_seat_incrementally_restored_quantized_slabs_within_affine_tolerance() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();

    let mut cache = quantized_cache();
    restore_all_blocks(&runtime, &mut cache);

    let K2HorizonMoVAKvState::Quantized(seated_state) = &cache else {
        panic!("the quantized cache should stay quantized");
    };
    assert_eq!(
        seated_state.offset_tokens(),
        RESTORED_TOKEN_COUNT,
        "the restored slabs must hold exactly the restored prefix"
    );

    // Block-wise quantization must match whole-prefix quantization within the
    // affine dequantization tolerance.
    let block_arrays: Vec<(MlxArray, MlxArray)> = (0..COMPLETE_BLOCK_COUNT)
        .map(|block_index| bf16_block(&runtime, block_index))
        .collect();
    let key_refs: Vec<&MlxArray> = block_arrays.iter().map(|(keys, _)| keys).collect();
    let value_refs: Vec<&MlxArray> = block_arrays.iter().map(|(_, values)| values).collect();
    let reference_keys = runtime
        .concatenate_axis(&key_refs, 2)
        .expect("the reference keys should concatenate");
    let reference_values = runtime
        .concatenate_axis(&value_refs, 2)
        .expect("the reference values should concatenate");
    let dequantized_keys = seated_state
        .dequantized_token_range(&runtime, true, 0, RESTORED_TOKEN_COUNT)
        .expect("the seated keys should dequantize");
    let dequantized_values = seated_state
        .dequantized_token_range(&runtime, false, 0, RESTORED_TOKEN_COUNT)
        .expect("the seated values should dequantize");
    assert_within_tolerance(
        &bf16_reference_vec(&runtime, &reference_keys),
        &bf16_reference_vec(&runtime, &dequantized_keys),
        "keys",
    );
    assert_within_tolerance(
        &bf16_reference_vec(&runtime, &reference_values),
        &bf16_reference_vec(&runtime, &dequantized_values),
        "values",
    );
}

fn assert_within_tolerance(reference_values: &[f32], restored_values: &[f32], label: &str) {
    let maximum_error = reference_values
        .iter()
        .zip(restored_values.iter())
        .map(|(reference, restored)| (reference - restored).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        maximum_error < DEQUANTIZE_TOLERANCE,
        "the {label} incremental quantized restore must stay within a bounded tolerance, got {maximum_error}"
    );
}
