//! SSD block-restore journeys for K2 Horizon MoVA KV states, one
//! concatenation per tensor.
//!
//! Proves the concat restore seats slabs matching a bulk concatenation
//! reference bit-for-bit (full precision) or within the affine quantization
//! tolerance (quantized), keeps exactly one growth step of headroom, and
//! holds the restore peak at the destination plus the complete block set.

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
/// Tiny fixtures round through MLX allocation granularity, so byte-bound
/// asserts carry explicit headroom instead of exact equality.
const ALLOCATION_GRANULARITY_HEADROOM_BYTES: u64 = 2 * 1024 * 1024;

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
/// block's payload unique so ordering mistakes cannot cancel out.
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
    let block_arrays: Vec<(MlxArray, MlxArray)> = (0..COMPLETE_BLOCK_COUNT)
        .map(|block_index| bf16_block(runtime, block_index))
        .collect();
    let key_references: Vec<&MlxArray> = block_arrays.iter().map(|(keys, _)| keys).collect();
    let value_references: Vec<&MlxArray> = block_arrays.iter().map(|(_, values)| values).collect();
    cache
        .restore_block_slices(
            runtime,
            &key_references,
            &value_references,
            RESTORED_TOKEN_COUNT,
        )
        .expect("the complete block set should restore in one concat per tensor");
}

fn active_memory_bytes(runtime: &MlxRuntime) -> u64 {
    runtime
        .memory_snapshot()
        .expect("the MLX memory snapshot should sample")
        .active_memory_bytes() as u64
}

fn peak_memory_bytes(runtime: &MlxRuntime) -> u64 {
    runtime
        .memory_snapshot()
        .expect("the MLX memory snapshot should sample")
        .peak_memory_bytes() as u64
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
async fn should_seat_full_precision_restored_slabs_matching_the_bulk_reference() {
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
async fn should_keep_concat_restore_peak_within_destination_plus_the_complete_block_set() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let runtime = test_runtime();

    let block_byte_count = 2_u64
        * (KEY_VALUE_HEAD_COUNT as u64)
        * (BLOCK_TOKEN_COUNT as u64)
        * (HEAD_DIMENSION as u64)
        * 2;
    let complete_block_set_bytes = block_byte_count * COMPLETE_BLOCK_COUNT as u64;
    let destination_byte_count = 2_u64
        * (KEY_VALUE_HEAD_COUNT as u64)
        * (RESTORED_TOKEN_COUNT + GROWTH_TOKENS) as u64
        * (HEAD_DIMENSION as u64)
        * 2;

    let mut cache = full_precision_cache();
    let baseline_active_bytes = active_memory_bytes(&runtime);
    runtime
        .reset_peak_memory()
        .expect("the MLX peak counter should reset so earlier tests do not inflate it");
    restore_all_blocks(&runtime, &mut cache);

    // The concat restore materializes the destination while the complete
    // source block set is consumed in one pass; the peak must stay inside
    // destination plus the complete block set plus concat scratch headroom.
    let peak_bytes = peak_memory_bytes(&runtime);
    let concat_restore_bound_bytes =
        destination_byte_count + complete_block_set_bytes + 3 * block_byte_count;
    assert!(
        peak_bytes <= concat_restore_bound_bytes,
        "the concat restore peak must stay within destination plus the complete \
         block set: peak={peak_bytes} bound={concat_restore_bound_bytes} \
         destination={destination_byte_count} blocks={complete_block_set_bytes}"
    );
    // The transient block set must release back to the destination. Tiny
    // fixtures round through MLX allocation granularity, so the bound carries
    // an explicit headroom instead of asserting exact byte equality.
    let after_restore_active_bytes = active_memory_bytes(&runtime);
    assert!(
        after_restore_active_bytes
            <= baseline_active_bytes
                + destination_byte_count
                + ALLOCATION_GRANULARITY_HEADROOM_BYTES,
        "after the restore only the destination slabs stay resident: \
         baseline={baseline_active_bytes} after={after_restore_active_bytes} \
         destination={destination_byte_count}"
    );
}

#[tokio::test]
async fn should_seat_quantized_restored_slabs_within_affine_tolerance() {
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

    // Whole-prefix quantization must match a bulk dequantization reference
    // within the affine dequantization tolerance.
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
        "the {label} quantized restore must stay within a bounded tolerance, got {maximum_error}"
    );
}
