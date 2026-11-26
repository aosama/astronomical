//! The row-exact tree attention kernel at a dense target's shape: the dense
//! 27B's 6 query heads a key-value head and its 16-row draft trees, which
//! fill six matrix tiles of query vectors where a 35B-A3B round fills two.
//!
//! A dense target attends its trees and its one-token steps through this
//! kernel, so its proof is the one that matters for a drafted reply equal
//! to the plain one (issue #2599), and its timings are what replaced MLX's
//! attention in exact groups of rows (issue #2601).
//!
//! Run with `cargo measure-row-exact-tree-attention`. No model; seconds.

use mlx_bridge::MlxDtype;
use model_serving::RowExactTreeAttentionKernels;
use serial_test::serial;

use super::row_exact_tree_attention::{
    AttentionCase, AttentionShape, HEAD_SIZE, MAXIMUM_DIFFERENCE_FROM_MLX, attend, attention_scale,
    bf16_array, direct_mlx_runtime, largest_difference, median_milliseconds, row_of,
};

/// The dense 27B: 6 query heads a key-value head, trees of 16 rows.
const DENSE_TARGET_SHAPE: AttentionShape = AttentionShape {
    query_head_count: 24,
    key_value_head_count: 4,
};
/// A 16-row draft tree as a dense target verifies it: a trunk with
/// siblings at several depths.
const DENSE_TREE: [Option<usize>; 16] = [
    None,
    Some(0),
    Some(0),
    Some(1),
    Some(1),
    Some(2),
    Some(3),
    Some(3),
    Some(4),
    Some(6),
    Some(6),
    Some(7),
    Some(9),
    Some(9),
    Some(12),
    Some(12),
];

/// A dense target's 16-row tree (6 query heads a key-value head, 6 matrix
/// tiles of query vectors): every row must get its one-row step's bits,
/// after a short and a long context and with attention sharpened until the
/// kernel's exponential runs far from zero, and stay as close to float32
/// attention as MLX's own bf16 attention does.
#[test]
#[ignore = "runs MLX kernels on the GPU; run through cargo measure-row-exact-tree-attention"]
#[serial]
fn should_give_every_row_of_a_dense_targets_tree_its_one_row_steps_bits() {
    let runtime = direct_mlx_runtime();
    let kernels = RowExactTreeAttentionKernels::declare().expect("the kernels declare");
    for (context_length, query_sharpness) in [(1_000, 1.0_f32), (9_000, 1.0), (3_000, 24.0)] {
        let mut case =
            AttentionCase::random_of_shape(DENSE_TARGET_SHAPE, context_length, &DENSE_TREE);
        for query_value in &mut case.queries {
            *query_value *= query_sharpness;
        }
        let tree_output = case.tree_output(&runtime, &kernels);
        for row in 0..DENSE_TREE.len() {
            let tree_row = row_of(&runtime, &tree_output, row);
            assert_eq!(
                tree_row,
                case.one_row_step_output(&runtime, &kernels, row),
                "dense tree after {context_length} keys, queries x{query_sharpness}: row {row} must equal its one-row step"
            );
            let exact = case.reference_row(&runtime, row, MlxDtype::Float32);
            let kernel_error = largest_difference(&exact, &tree_row);
            let mlx_error = largest_difference(
                &exact,
                &case.reference_row(&runtime, row, MlxDtype::BFloat16),
            );
            assert!(
                kernel_error <= mlx_error + MAXIMUM_DIFFERENCE_FROM_MLX,
                "dense tree after {context_length} keys, queries x{query_sharpness}: row {row} is {kernel_error} from float32 attention where MLX's bf16 is {mlx_error}"
            );
        }
        eprintln!(
            "[row-exact-tree-attention] dense 16-row tree after {context_length} keys, queries x{query_sharpness}: every row equals its one-row step"
        );
    }
}

/// What a dense target's full-attention layers cost a draft round and a
/// plain token each way, over the 27B's 16 such layers: MLX's attention for
/// one row, for 16 rows in one masked call (not row-exact), and in the
/// exact groups of 5 rows it is served in today, against the kernel for one
/// row and for the tree.
#[test]
#[ignore = "runs MLX kernels on the GPU; run through cargo measure-row-exact-tree-attention"]
#[serial]
fn should_time_a_dense_targets_tree_against_mlx_one_call_and_exact_groups() {
    const DENSE_FULL_ATTENTION_LAYER_COUNT: f64 = 16.0;
    const MOST_ROWS_AN_EXACT_MLX_CALL_TAKES: usize = 5;
    let runtime = direct_mlx_runtime();
    let kernels = RowExactTreeAttentionKernels::declare().expect("the kernels declare");
    let row_count = DENSE_TREE.len();
    for context_length in [2_048, 10_240, 32_768] {
        let case = AttentionCase::random_of_shape(DENSE_TARGET_SHAPE, context_length, &DENSE_TREE);
        let token_count = context_length + row_count as i32;
        let slab_shape = [
            1,
            DENSE_TARGET_SHAPE.key_value_head_count,
            token_count,
            HEAD_SIZE,
        ];
        let keys = bf16_array(&runtime, &case.keys, &slab_shape);
        let values = bf16_array(&runtime, &case.values, &slab_shape);
        let query_heads = DENSE_TARGET_SHAPE.query_head_count;
        let tree_queries = bf16_array(
            &runtime,
            &case.queries,
            &[1, query_heads, row_count as i32, HEAD_SIZE],
        );
        let one_query = bf16_array(
            &runtime,
            &case.row_queries(0),
            &[1, query_heads, 1, HEAD_SIZE],
        );
        let mut group_queries = Vec::new();
        let mut first_row = 0;
        while first_row < row_count {
            let end_row = (first_row + MOST_ROWS_AN_EXACT_MLX_CALL_TAKES).min(row_count);
            group_queries.push(
                runtime
                    .slice(
                        &tree_queries,
                        &[0, 0, first_row as i32, 0],
                        &[1, query_heads, end_row as i32, HEAD_SIZE],
                        &[1, 1, 1, 1],
                    )
                    .expect("a group of rows slices"),
            );
            first_row = end_row;
        }
        runtime
            .evaluate_arrays(&group_queries.iter().collect::<Vec<_>>())
            .expect("the groups evaluate");
        let tree_mask = case.tree_mask(&runtime, 0, row_count);
        let mut group_masks = Vec::new();
        for group_index in 0..group_queries.len() {
            let first_group_row = group_index * MOST_ROWS_AN_EXACT_MLX_CALL_TAKES;
            let end_group_row = first_group_row + group_queries[group_index].shape()[2] as usize;
            group_masks.push(case.tree_mask(&runtime, first_group_row, end_group_row));
        }
        let scale = attention_scale();
        let mlx_one_row = median_milliseconds(&runtime, &|| {
            runtime
                .scaled_dot_product_attention(&one_query, &keys, &values, scale)
                .expect("MLX attends")
        });
        let mlx_one_call = median_milliseconds(&runtime, &|| {
            runtime
                .masked_scaled_dot_product_attention(
                    &tree_queries,
                    &keys,
                    &values,
                    scale,
                    &tree_mask,
                )
                .expect("MLX attends")
        });
        let mlx_exact_groups = median_milliseconds(&runtime, &|| {
            let mut attended_groups = Vec::with_capacity(group_queries.len());
            for (queries, mask) in group_queries.iter().zip(&group_masks) {
                attended_groups.push(
                    runtime
                        .masked_scaled_dot_product_attention(queries, &keys, &values, scale, mask)
                        .expect("MLX attends"),
                );
            }
            runtime
                .concatenate_axis(&attended_groups.iter().collect::<Vec<_>>(), 2)
                .expect("the groups join")
        });
        let kernel_one_row = median_milliseconds(&runtime, &|| {
            attend(
                &runtime,
                &kernels,
                &one_query,
                &keys,
                &values,
                token_count - 1,
                &[None],
            )
        });
        let kernel_tree = median_milliseconds(&runtime, &|| {
            attend(
                &runtime,
                &kernels,
                &tree_queries,
                &keys,
                &values,
                context_length,
                &DENSE_TREE,
            )
        });
        eprintln!(
            "[row-exact-tree-attention] dense {context_length:>6} keys, x{DENSE_FULL_ATTENTION_LAYER_COUNT} layers: one row MLX {:.1} ms, kernel {:.1} ms | 16-row tree MLX one call {:.1} ms, MLX exact groups {:.1} ms, kernel {:.1} ms",
            mlx_one_row * DENSE_FULL_ATTENTION_LAYER_COUNT,
            kernel_one_row * DENSE_FULL_ATTENTION_LAYER_COUNT,
            mlx_one_call * DENSE_FULL_ATTENTION_LAYER_COUNT,
            mlx_exact_groups * DENSE_FULL_ATTENTION_LAYER_COUNT,
            kernel_tree * DENSE_FULL_ATTENTION_LAYER_COUNT,
        );
    }
}
