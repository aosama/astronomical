//! The drafter's kept context, held in a slab that grows in place, against
//! the same rows joined by concatenation: a sliding-window layer must show
//! a block the same latest context rows, followed by the block's own,
//! round after round, including after its slab is cut back to the window.
//!
//! Random bfloat16 rows; a window of 8 so a few dozen rounds cross the
//! cut-back more than once. No model on disk. Run with
//! `cargo test-dflash-drafter-context-slab`. Seconds.

use mlx_bridge::{MlxArray, MlxDtype};
use model_serving::speculative_decoding::dflash_drafter::{
    DFlashDrafterConfiguration, DrafterContextStates,
};
use runtime_integration::{MlxMemoryLimits, MlxRuntime};
use serde_json::json;
use serial_test::serial;

const HEAD_COUNT: i32 = 2;
const HEAD_DIMENSION: i32 = 8;
const BLOCK_ROW_COUNT: i32 = 4;
const ROUND_COUNT: usize = 40;

fn slab_runtime() -> MlxRuntime {
    let one_gigabyte = 1024 * 1024 * 1024;
    let memory_limits = MlxMemoryLimits::new(one_gigabyte, one_gigabyte / 4);
    MlxRuntime::initialize(memory_limits.expect("the limits are valid")).expect("MLX starts")
}

/// One sliding-window layer of window 8 and one full layer.
fn two_layer_configuration() -> DFlashDrafterConfiguration {
    DFlashDrafterConfiguration::from_config_document(&json!({
        "architectures": ["DFlashDraftModel"],
        "dflash_config": {"block_size": 4, "mask_token_id": 9, "target_layer_ids": [1]},
        "head_dim": HEAD_DIMENSION, "hidden_size": 16, "intermediate_size": 32,
        "layer_types": ["sliding_attention", "full_attention"],
        "num_attention_heads": HEAD_COUNT, "num_hidden_layers": 2,
        "num_key_value_heads": HEAD_COUNT, "num_target_layers": 4, "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10000, "rope_type": "default"},
        "sliding_window": 8, "vocab_size": 16
    }))
    .expect("the configuration reads")
}

fn random_rows(runtime: &MlxRuntime, random_key: &MlxArray, row_count: i32) -> MlxArray {
    runtime
        .random_normal(
            &[1, HEAD_COUNT, row_count, HEAD_DIMENSION],
            MlxDtype::BFloat16,
            0.0,
            1.0,
            random_key,
        )
        .expect("rows are drawn")
}

/// The kept rows a layer shows, by concatenation: the latest `row_limit`
/// (or all), then the block's.
fn joined_rows(
    runtime: &MlxRuntime,
    every_kept_row: &[MlxArray],
    row_limit: Option<i32>,
    block_rows: &MlxArray,
) -> (MlxArray, i32) {
    let kept_references: Vec<&MlxArray> = every_kept_row.iter().collect();
    let kept = runtime
        .concatenate_axis(&kept_references, 2)
        .expect("kept rows join");
    let kept_row_count = kept.shape()[2];
    let first_shown = row_limit.map_or(0, |limit| (kept_row_count - limit).max(0));
    let shown = runtime
        .slice(
            &kept,
            &[0, 0, first_shown, 0],
            &[1, HEAD_COUNT, kept_row_count, HEAD_DIMENSION],
            &[1, 1, 1, 1],
        )
        .expect("the shown rows slice");
    let joined = runtime
        .concatenate_axis(&[&shown, block_rows], 2)
        .expect("the block joins");
    (joined, kept_row_count - first_shown)
}

fn float_values(runtime: &MlxRuntime, rows: &MlxArray) -> Vec<f32> {
    runtime
        .array_to_vec_f32(&runtime.astype(rows, MlxDtype::Float32).expect("widen"))
        .expect("rows read back")
}

#[test]
#[ignore = "runs MLX on the local GPU"]
#[serial]
fn should_show_a_block_the_rows_concatenation_shows_round_after_round() {
    let runtime = slab_runtime();
    let mut context_states = DrafterContextStates::new(&two_layer_configuration(), 0);
    for layer_index in 0..2 {
        let random_key = runtime.random_key(40 + layer_index as u64).expect("a key");
        let mut every_kept_key = Vec::new();
        let mut every_kept_value = Vec::new();
        let layer_state = &mut context_states.layers[layer_index];
        let row_limit = layer_state.retained_row_limit();
        for round_index in 0..ROUND_COUNT {
            let context_row_count = 1 + (round_index % 3) as i32;
            let context_keys = random_rows(&runtime, &random_key, context_row_count);
            let context_values = random_rows(&runtime, &random_key, context_row_count);
            let block_keys = random_rows(&runtime, &random_key, BLOCK_ROW_COUNT);
            let block_values = random_rows(&runtime, &random_key, BLOCK_ROW_COUNT);
            runtime
                .evaluate_arrays(&[&context_keys, &context_values, &block_keys, &block_values])
                .expect("rows evaluate");
            every_kept_key.push(context_keys.retain().expect("retain"));
            every_kept_value.push(context_values.retain().expect("retain"));
            let attended = layer_state
                .attended_rows(
                    &runtime,
                    (&context_keys, &context_values),
                    (&block_keys, &block_values),
                )
                .expect("the slab attends");
            let (expected_keys, expected_kept) =
                joined_rows(&runtime, &every_kept_key, row_limit, &block_keys);
            let (expected_values, _) =
                joined_rows(&runtime, &every_kept_value, row_limit, &block_values);
            assert_eq!(
                attended.kept_row_count, expected_kept,
                "layer {layer_index} round {round_index}"
            );
            assert_eq!(
                float_values(&runtime, &attended.keys),
                float_values(&runtime, &expected_keys),
                "layer {layer_index} round {round_index} keys"
            );
            assert_eq!(
                float_values(&runtime, &attended.values),
                float_values(&runtime, &expected_values),
                "layer {layer_index} round {round_index} values"
            );
        }
    }
}

#[test]
#[ignore = "runs MLX on the local GPU"]
#[serial]
fn should_advance_the_context_position_by_the_rows_kept() {
    let runtime = slab_runtime();
    let mut context_states = DrafterContextStates::new(&two_layer_configuration(), 100);
    let random_key = runtime.random_key(7).expect("a key");
    let layer_state = &mut context_states.layers[0];
    let rows = random_rows(&runtime, &random_key, 3);
    let block = random_rows(&runtime, &random_key, BLOCK_ROW_COUNT);
    layer_state
        .attended_rows(&runtime, (&rows, &rows), (&block, &block))
        .expect("the slab attends");
    assert_eq!(layer_state.next_context_position(), 103);
}

/// What the memory ledger is told the kept context holds (issue #2572):
/// nothing before a round, then the slabs as allocated, which is a whole
/// number of growth steps of two bfloat16 arrays a layer and never less
/// than the rows kept.
#[test]
#[ignore = "runs MLX on the local GPU"]
#[serial]
fn should_report_the_bytes_its_slabs_hold_for_the_memory_ledger() {
    const BFLOAT16_BYTES: u64 = 2;
    let runtime = slab_runtime();
    let mut context_states = DrafterContextStates::new(&two_layer_configuration(), 0);
    // Edge: no round has run, so nothing is held.
    assert_eq!(context_states.payload_byte_count(), 0);

    let random_key = runtime.random_key(72).expect("a key");
    let kept_row_count = 3;
    let mut every_layer_bytes = 0;
    for layer_state in &mut context_states.layers {
        let context_keys = random_rows(&runtime, &random_key, kept_row_count);
        let context_values = random_rows(&runtime, &random_key, kept_row_count);
        let block_keys = random_rows(&runtime, &random_key, BLOCK_ROW_COUNT);
        let block_values = random_rows(&runtime, &random_key, BLOCK_ROW_COUNT);
        layer_state
            .attended_rows(
                &runtime,
                (&context_keys, &context_values),
                (&block_keys, &block_values),
            )
            .expect("the slab attends");
        let layer_bytes = layer_state.payload_byte_count();
        let kept_row_bytes =
            2 * BFLOAT16_BYTES * (HEAD_COUNT * kept_row_count * HEAD_DIMENSION) as u64;
        let one_row_bytes = 2 * BFLOAT16_BYTES * (HEAD_COUNT * HEAD_DIMENSION) as u64;
        assert!(
            layer_bytes >= kept_row_bytes,
            "a layer holds at least its kept rows: {layer_bytes} against {kept_row_bytes}"
        );
        assert_eq!(
            layer_bytes % one_row_bytes,
            0,
            "a slab is a whole number of key and value rows"
        );
        every_layer_bytes += layer_bytes;
    }
    // Positive: the request's figure is the sum over its layers.
    assert_eq!(context_states.payload_byte_count(), every_layer_bytes);
}
