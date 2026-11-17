//! The lowest rung under a draft-tree round's growth with context: MLX's
//! attention alone at the 35B-A3B target's shape (16 query heads over 2
//! key-value heads, head size 256), timed over 1,024 to 65,536
//! stored keys for one row unmasked (a plain decode step), four rows
//! unmasked, and four rows with the additive mask a tree verify passes.
//! A plain step's cost barely grows with the context while a 4-row tree
//! round's attention grows several times faster; this says whether the
//! mask is what moves MLX off its kernel that reads the history once.
//! It reports; it asserts nothing.

use std::time::Instant;

use astronomical_runtime_integration::{MlxMemoryLimits, MlxRuntime};
use mlx_bridge::{MlxArray, MlxDtype};
use serial_test::serial;

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_MLX_BUFFER_CACHE_LIMIT_BYTES,
};

const QUERY_HEAD_COUNT: i32 = 16;
const KEY_VALUE_HEAD_COUNT: i32 = 2;
const HEAD_SIZE: i32 = 256;
const STORED_KEY_COUNTS: [i32; 5] = [1_024, 4_096, 10_240, 32_768, 65_536];
const TREE_ROW_COUNT: i32 = 4;
const TIMED_REPETITIONS: usize = 30;
/// The target's full-attention layers a round passes through.
const FULL_ATTENTION_LAYER_COUNT: f64 = 10.0;

#[test]
#[ignore = "runs MLX kernels on the GPU; run through cargo measure-tree-attention-context-cost"]
#[serial]
fn should_time_tree_attention_against_the_plain_step_as_the_context_grows() {
    let runtime = direct_mlx_runtime();
    let scale = 1.0 / (HEAD_SIZE as f32).sqrt();
    for stored_key_count in STORED_KEY_COUNTS {
        let keys = normal(
            &runtime,
            &[1, KEY_VALUE_HEAD_COUNT, stored_key_count, HEAD_SIZE],
            3,
        );
        let values = normal(
            &runtime,
            &[1, KEY_VALUE_HEAD_COUNT, stored_key_count, HEAD_SIZE],
            5,
        );
        let one_row = normal(&runtime, &[1, QUERY_HEAD_COUNT, 1, HEAD_SIZE], 7);
        let tree_rows = normal(
            &runtime,
            &[1, QUERY_HEAD_COUNT, TREE_ROW_COUNT, HEAD_SIZE],
            9,
        );
        let tree_mask = chain_mask(&runtime, stored_key_count);
        let plain_step = median_milliseconds(&runtime, &|| {
            runtime
                .scaled_dot_product_attention(&one_row, &keys, &values, scale)
                .expect("the one-row attention builds")
        });
        let unmasked_rows = median_milliseconds(&runtime, &|| {
            runtime
                .scaled_dot_product_attention(&tree_rows, &keys, &values, scale)
                .expect("the four-row attention builds")
        });
        let masked_rows = median_milliseconds(&runtime, &|| {
            runtime
                .masked_scaled_dot_product_attention(&tree_rows, &keys, &values, scale, &tree_mask)
                .expect("the masked attention builds")
        });
        eprintln!(
            "[tree-attention-context-cost] {stored_key_count:>6} keys: 1 row {plain_step:.3} ms | 4 rows unmasked {unmasked_rows:.3} ms | 4 rows masked {masked_rows:.3} ms (x{FULL_ATTENTION_LAYER_COUNT} layers: {:.1} / {:.1} / {:.1} ms)",
            plain_step * FULL_ATTENTION_LAYER_COUNT,
            unmasked_rows * FULL_ATTENTION_LAYER_COUNT,
            masked_rows * FULL_ATTENTION_LAYER_COUNT,
        );
    }
}

/// The additive mask a 4-row chain verify passes: every row sees the whole
/// context; among the tree rows, each sees itself and the rows before it.
fn chain_mask(runtime: &MlxRuntime, stored_key_count: i32) -> MlxArray {
    let context_count = (stored_key_count - TREE_ROW_COUNT) as usize;
    let column_count = stored_key_count as usize;
    let mut additive = vec![0.0_f32; TREE_ROW_COUNT as usize * column_count];
    for row_index in 0..TREE_ROW_COUNT as usize {
        for tree_column in (row_index + 1)..TREE_ROW_COUNT as usize {
            additive[row_index * column_count + context_count + tree_column] = f32::NEG_INFINITY;
        }
    }
    let mask = runtime
        .array_from_f32(&additive, &[TREE_ROW_COUNT, stored_key_count])
        .expect("the mask uploads");
    runtime
        .astype(&mask, MlxDtype::BFloat16)
        .expect("the mask converts")
}

fn median_milliseconds(runtime: &MlxRuntime, build: &dyn Fn() -> MlxArray) -> f64 {
    let warm_up = build();
    runtime.evaluate_arrays(&[&warm_up]).expect("evaluates");
    let mut timings = Vec::with_capacity(TIMED_REPETITIONS);
    for _ in 0..TIMED_REPETITIONS {
        let started_at = Instant::now();
        let output = build();
        runtime.evaluate_arrays(&[&output]).expect("evaluates");
        timings.push(started_at.elapsed().as_secs_f64() * 1000.0);
    }
    timings.sort_by(f64::total_cmp);
    timings[timings.len() / 2]
}

fn normal(runtime: &MlxRuntime, shape: &[i32], seed: u64) -> MlxArray {
    let key = runtime.random_key(seed).expect("a random key");
    runtime
        .random_normal(shape, MlxDtype::BFloat16, 0.0, 1.0, &key)
        .expect("normal values")
}

fn direct_mlx_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_MLX_BUFFER_CACHE_LIMIT_BYTES,
        )
        .expect("the memory limits are valid"),
    )
    .expect("the direct MLX runtime initializes")
}
