//! The row-exact tree attention kernel at the 35B-A3B target's shape (16
//! query heads over 2 key-value heads, head size 256), held to two rules:
//!
//! - Exact: every row verified inside a 4-row tree gives the bits the same
//!   row gives as a one-row step, where its ancestors' keys sit right after
//!   the context as plain decoding would have stored them. A chain and a
//!   branching tree, over a context inside one chunk and one across many.
//! - Correct: every row agrees with MLX's masked attention within bf16
//!   rounding.
//!
//! Then a timing ladder over 1,024 to 65,536 keys against MLX, run through
//! `cargo measure-row-exact-tree-attention`.

use std::time::Instant;

use mlx_bridge::{MlxArray, MlxDtype};
use model_serving::{
    RowExactTreeAttentionInputs, RowExactTreeAttentionKernels, TreeRowChains,
    row_exact_tree_attention,
};
use runtime_integration::{MlxMemoryLimits, MlxRuntime};
use serial_test::serial;

use crate::common::{
    DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES, DIRECT_MLX_TEST_MLX_BUFFER_CACHE_LIMIT_BYTES,
};

pub(super) const HEAD_SIZE: i32 = 256;
const TREE_ROW_COUNT: usize = 4;

/// A target's attention heads: how many query heads share a key-value head
/// decides how many matrix tiles a tree's rows fill.
#[derive(Clone, Copy)]
pub(super) struct AttentionShape {
    pub(super) query_head_count: i32,
    pub(super) key_value_head_count: i32,
}

/// The 35B-A3B: 8 query heads a key-value head, rounds of 4 rows.
const EXPERT_TARGET_SHAPE: AttentionShape = AttentionShape {
    query_head_count: 16,
    key_value_head_count: 2,
};
/// Bf16 rounding of attention outputs near 1 in magnitude.
pub(super) const MAXIMUM_DIFFERENCE_FROM_MLX: f32 = 0.02;
/// Query scales from flat attention to one where a few keys take all of it.
const SWEPT_QUERY_SHARPNESS: [f32; 4] = [1.0, 8.0, 24.0, 96.0];
const SHARPENED_CONTEXT_LENGTH: i32 = 3_000;
const TIMED_CONTEXTS: [i32; 4] = [1_024, 10_240, 32_768, 65_536];
const TIMED_REPETITIONS: usize = 30;
const FULL_ATTENTION_LAYER_COUNT: f64 = 10.0;

/// A chain, and a tree whose rows 1 and 2 are siblings under the anchor and
/// row 3 sits under row 1.
const TREE_SHAPES: [(&str, [Option<usize>; TREE_ROW_COUNT]); 2] = [
    ("chain", [None, Some(0), Some(1), Some(2)]),
    ("branching tree", [None, Some(0), Some(0), Some(1)]),
];

#[test]
#[ignore = "runs MLX kernels on the GPU; run through cargo measure-row-exact-tree-attention"]
#[serial]
fn should_give_every_tree_row_its_one_row_steps_bits_and_agree_with_mlx() {
    let runtime = direct_mlx_runtime();
    let kernels = RowExactTreeAttentionKernels::declare().expect("the kernels declare");
    for context_length in [1_000, 9_000] {
        for (shape_label, parent_rows) in TREE_SHAPES {
            let case = AttentionCase::random(context_length, &parent_rows);
            let tree_output = case.tree_output(&runtime, &kernels);
            for row in 0..TREE_ROW_COUNT {
                let step_output = case.one_row_step_output(&runtime, &kernels, row);
                let tree_row = row_of(&runtime, &tree_output, row);
                assert_eq!(
                    tree_row, step_output,
                    "{shape_label} after {context_length} keys: row {row} must equal its one-row step"
                );
                let reference = case.mlx_reference_row(&runtime, row);
                let difference = largest_difference(&reference, &tree_row);
                assert!(
                    difference <= MAXIMUM_DIFFERENCE_FROM_MLX,
                    "{shape_label} after {context_length} keys: row {row} differs from MLX by {difference}"
                );
            }
            eprintln!(
                "[row-exact-tree-attention] {shape_label} after {context_length} keys: every row equals its one-row step"
            );
        }
    }
}

/// How far the kernel and MLX's bf16 attention each sit from attention
/// computed in float32 over the same bf16 values, after about 11,000 keys,
/// from flat to sharply peaked attention (queries scaled up), so a precision
/// loss in either shows apart from the other's.
#[test]
#[ignore = "runs MLX kernels on the GPU; run through cargo measure-row-exact-tree-attention"]
#[serial]
fn should_report_accuracy_against_float32_attention_as_attention_sharpens() {
    let runtime = direct_mlx_runtime();
    let kernels = RowExactTreeAttentionKernels::declare().expect("the kernels declare");
    let context_length = 11_292;
    for query_sharpness in [1.0_f32, 8.0, 24.0] {
        let mut case = AttentionCase::random(context_length, &[None, Some(0), Some(1), Some(2)]);
        for query_value in &mut case.queries {
            *query_value *= query_sharpness;
        }
        let tree_output = case.tree_output(&runtime, &kernels);
        let (mut kernel_error, mut mlx_error) = (0.0_f32, 0.0_f32);
        for row in 0..TREE_ROW_COUNT {
            let exact = case.reference_row(&runtime, row, MlxDtype::Float32);
            kernel_error = kernel_error.max(largest_difference(
                &exact,
                &row_of(&runtime, &tree_output, row),
            ));
            mlx_error = mlx_error.max(largest_difference(
                &exact,
                &case.reference_row(&runtime, row, MlxDtype::BFloat16),
            ));
        }
        eprintln!(
            "[row-exact-tree-attention] {context_length} keys, queries x{query_sharpness}: largest error against float32, kernel {kernel_error:.5}, MLX bf16 {mlx_error:.5}"
        );
    }
}

/// Queries scaled up until attention is sharply peaked: the kernel's
/// exponential then runs far from zero, where an exponential that only
/// agrees with another near zero parts from it (the one-token DeltaNet
/// kernel's did, issue #2599). The kernel does not promise MLX's bits, so
/// what is held is its own promise, a tree row's bits equal to its one-row
/// step's, and an error against float32 attention no worse than MLX's own
/// bf16 attention by more than bf16 rounding.
#[test]
#[ignore = "runs MLX kernels on the GPU; run through cargo measure-row-exact-tree-attention"]
#[serial]
fn should_keep_every_tree_row_its_one_row_steps_bits_as_attention_sharpens() {
    let runtime = direct_mlx_runtime();
    let kernels = RowExactTreeAttentionKernels::declare().expect("the kernels declare");
    for query_sharpness in SWEPT_QUERY_SHARPNESS {
        for (shape_label, parent_rows) in TREE_SHAPES {
            let mut case = AttentionCase::random(SHARPENED_CONTEXT_LENGTH, &parent_rows);
            for query_value in &mut case.queries {
                *query_value *= query_sharpness;
            }
            let tree_output = case.tree_output(&runtime, &kernels);
            for row in 0..TREE_ROW_COUNT {
                let tree_row = row_of(&runtime, &tree_output, row);
                assert_eq!(
                    tree_row,
                    case.one_row_step_output(&runtime, &kernels, row),
                    "{shape_label}, queries x{query_sharpness}: row {row} must equal its one-row step"
                );
                let exact = case.reference_row(&runtime, row, MlxDtype::Float32);
                let kernel_error = largest_difference(&exact, &tree_row);
                let mlx_error = largest_difference(
                    &exact,
                    &case.reference_row(&runtime, row, MlxDtype::BFloat16),
                );
                assert!(
                    kernel_error <= mlx_error + MAXIMUM_DIFFERENCE_FROM_MLX,
                    "{shape_label}, queries x{query_sharpness}: row {row} is {kernel_error} from float32 attention where MLX's bf16 is {mlx_error}"
                );
            }
            eprintln!(
                "[row-exact-tree-attention] {shape_label}, queries x{query_sharpness}: every row equals its one-row step"
            );
        }
    }
}

/// One random attention problem: host keys and values for the context and
/// the tree's rows, queries for every row, and the tree's shape.
pub(super) struct AttentionCase {
    shape: AttentionShape,
    context_length: i32,
    parent_rows: Vec<Option<usize>>,
    /// `[key-value head][token][dimension]`, context then tree rows.
    pub(super) keys: Vec<f32>,
    pub(super) values: Vec<f32>,
    /// `[query head][row][dimension]`.
    pub(super) queries: Vec<f32>,
}

impl AttentionCase {
    /// A 35B-A3B round's case.
    fn random(context_length: i32, parent_rows: &[Option<usize>]) -> Self {
        Self::random_of_shape(EXPERT_TARGET_SHAPE, context_length, parent_rows)
    }

    pub(super) fn random_of_shape(
        shape: AttentionShape,
        context_length: i32,
        parent_rows: &[Option<usize>],
    ) -> Self {
        let token_count = context_length as usize + parent_rows.len();
        let slab_values = shape.key_value_head_count as usize * token_count * HEAD_SIZE as usize;
        let query_values = shape.query_head_count as usize * parent_rows.len() * HEAD_SIZE as usize;
        let mut stream = SplitMix64(context_length as u64);
        Self {
            shape,
            context_length,
            parent_rows: parent_rows.to_vec(),
            keys: (0..slab_values).map(|_| stream.next_signed()).collect(),
            values: (0..slab_values).map(|_| stream.next_signed()).collect(),
            queries: (0..query_values).map(|_| stream.next_signed()).collect(),
        }
    }

    fn row_count(&self) -> usize {
        self.parent_rows.len()
    }

    pub(super) fn tree_output(
        &self,
        runtime: &MlxRuntime,
        kernels: &RowExactTreeAttentionKernels,
    ) -> MlxArray {
        let token_count = self.context_length + self.row_count() as i32;
        let keys = bf16_array(
            runtime,
            &self.keys,
            &[1, self.shape.key_value_head_count, token_count, HEAD_SIZE],
        );
        let values = bf16_array(
            runtime,
            &self.values,
            &[1, self.shape.key_value_head_count, token_count, HEAD_SIZE],
        );
        let queries = bf16_array(
            runtime,
            &self.queries,
            &[
                1,
                self.shape.query_head_count,
                self.row_count() as i32,
                HEAD_SIZE,
            ],
        );
        attend(
            runtime,
            kernels,
            &queries,
            &keys,
            &values,
            self.context_length,
            &self.parent_rows,
        )
    }

    /// Row `row` as a one-row step: its ancestors' keys and values right
    /// after the context, then attention for its query alone.
    pub(super) fn one_row_step_output(
        &self,
        runtime: &MlxRuntime,
        kernels: &RowExactTreeAttentionKernels,
        row: usize,
    ) -> Vec<f32> {
        let chain = chain_of(&self.parent_rows, row);
        let step_context = self.context_length + chain.len() as i32 - 1;
        let (keys, values) = self.step_slabs(&chain);
        let shape = [
            1,
            self.shape.key_value_head_count,
            step_context + 1,
            HEAD_SIZE,
        ];
        let query = self.row_queries(row);
        let output = attend(
            runtime,
            kernels,
            &bf16_array(
                runtime,
                &query,
                &[1, self.shape.query_head_count, 1, HEAD_SIZE],
            ),
            &bf16_array(runtime, &keys, &shape),
            &bf16_array(runtime, &values, &shape),
            step_context,
            &[None],
        );
        row_of(runtime, &output, 0)
    }

    /// Slabs holding the context, then the tokens of `chain` in order.
    fn step_slabs(&self, chain: &[usize]) -> (Vec<f32>, Vec<f32>) {
        let head_size = HEAD_SIZE as usize;
        let context = self.context_length as usize;
        let source_tokens = context + self.row_count();
        let (mut keys, mut values) = (Vec::new(), Vec::new());
        for head in 0..self.shape.key_value_head_count as usize {
            let head_start = head * source_tokens * head_size;
            keys.extend_from_slice(&self.keys[head_start..head_start + context * head_size]);
            values.extend_from_slice(&self.values[head_start..head_start + context * head_size]);
            for tree_row in chain {
                let token_start = head_start + (context + tree_row) * head_size;
                keys.extend_from_slice(&self.keys[token_start..token_start + head_size]);
                values.extend_from_slice(&self.values[token_start..token_start + head_size]);
            }
        }
        (keys, values)
    }

    pub(super) fn row_queries(&self, row: usize) -> Vec<f32> {
        let head_size = HEAD_SIZE as usize;
        let mut query = Vec::with_capacity(self.shape.query_head_count as usize * head_size);
        for head in 0..self.shape.query_head_count as usize {
            let start = (head * self.row_count() + row) * head_size;
            query.extend_from_slice(&self.queries[start..start + head_size]);
        }
        query
    }

    /// The additive mask of tree rows `first_row..end_row` over the context
    /// and every tree row, `[1, 1, rows, tokens]` bf16: a row sees the
    /// context and its own chain.
    pub(super) fn tree_mask(
        &self,
        runtime: &MlxRuntime,
        first_row: usize,
        end_row: usize,
    ) -> MlxArray {
        let token_count = self.context_length as usize + self.row_count();
        let mut additive = vec![0.0_f32; (end_row - first_row) * token_count];
        for row in first_row..end_row {
            let chain = chain_of(&self.parent_rows, row);
            for tree_row in 0..self.row_count() {
                if !chain.contains(&tree_row) {
                    additive[(row - first_row) * token_count
                        + self.context_length as usize
                        + tree_row] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = runtime
            .array_from_f32(
                &additive,
                &[1, 1, (end_row - first_row) as i32, token_count as i32],
            )
            .expect("the mask uploads");
        runtime
            .astype(&mask, MlxDtype::BFloat16)
            .expect("the mask converts")
    }

    /// MLX's masked attention over the context and the row's ancestors.
    fn mlx_reference_row(&self, runtime: &MlxRuntime, row: usize) -> Vec<f32> {
        self.reference_row(runtime, row, MlxDtype::BFloat16)
    }

    /// MLX's masked attention over the row's keys, its inputs the case's bf16
    /// values cast to `dtype`: float32 is the exact reference.
    pub(super) fn reference_row(
        &self,
        runtime: &MlxRuntime,
        row: usize,
        dtype: MlxDtype,
    ) -> Vec<f32> {
        let token_count = self.context_length + self.row_count() as i32;
        let shape = [1, self.shape.key_value_head_count, token_count, HEAD_SIZE];
        let chain = chain_of(&self.parent_rows, row);
        let mut additive = vec![0.0_f32; token_count as usize];
        for tree_row in 0..self.row_count() {
            if !chain.contains(&tree_row) {
                additive[self.context_length as usize + tree_row] = f32::NEG_INFINITY;
            }
        }
        let mask = runtime
            .astype(
                &runtime
                    .array_from_f32(&additive, &[1, token_count])
                    .expect("mask"),
                MlxDtype::BFloat16,
            )
            .expect("mask converts");
        let output = runtime
            .masked_scaled_dot_product_attention(
                &bf16_array(
                    runtime,
                    &self.row_queries(row),
                    &[1, self.shape.query_head_count, 1, HEAD_SIZE],
                ),
                &cast(runtime, &bf16_array(runtime, &self.keys, &shape), dtype),
                &cast(runtime, &bf16_array(runtime, &self.values, &shape), dtype),
                attention_scale(),
                &mask,
            )
            .expect("MLX attends");
        row_of(runtime, &output, 0)
    }
}

pub(super) fn attend(
    runtime: &MlxRuntime,
    kernels: &RowExactTreeAttentionKernels,
    queries: &MlxArray,
    key_slab: &MlxArray,
    value_slab: &MlxArray,
    context_length: i32,
    parent_rows: &[Option<usize>],
) -> MlxArray {
    let row_chains = TreeRowChains::from_parent_rows(parent_rows)
        .upload(runtime)
        .expect("the chains upload");
    row_exact_tree_attention(
        runtime,
        kernels,
        &RowExactTreeAttentionInputs {
            queries,
            key_slab,
            value_slab,
            context_length,
            row_chains: &row_chains,
            scale: attention_scale(),
        },
    )
    .expect("the tree attention builds")
}

/// The tree rows from the anchor to `row`, in depth order.
fn chain_of(parent_rows: &[Option<usize>], row: usize) -> Vec<usize> {
    let mut chain = vec![row];
    let mut current = row;
    while let Some(parent) = parent_rows[current] {
        chain.push(parent);
        current = parent;
    }
    chain.reverse();
    chain
}

/// Every query head's output for `row` of a `[1, heads, rows, 256]` output.
pub(super) fn row_of(runtime: &MlxRuntime, output: &MlxArray, row: usize) -> Vec<f32> {
    let row_count = output.shape()[2] as usize;
    let values = host_values(runtime, output);
    let head_size = HEAD_SIZE as usize;
    let query_head_count = output.shape()[1] as usize;
    let mut row_values = Vec::with_capacity(query_head_count * head_size);
    for head in 0..query_head_count {
        let start = (head * row_count + row) * head_size;
        row_values.extend_from_slice(&values[start..start + head_size]);
    }
    row_values
}

pub(super) fn attention_scale() -> f32 {
    1.0 / (HEAD_SIZE as f32).sqrt()
}

fn cast(runtime: &MlxRuntime, array: &MlxArray, dtype: MlxDtype) -> MlxArray {
    runtime.astype(array, dtype).expect("values convert")
}

pub(super) fn bf16_array(runtime: &MlxRuntime, values: &[f32], shape: &[i32]) -> MlxArray {
    let array = runtime
        .array_from_f32(values, shape)
        .expect("values upload");
    runtime
        .astype(&array, MlxDtype::BFloat16)
        .expect("values convert")
}

fn host_values(runtime: &MlxRuntime, array: &MlxArray) -> Vec<f32> {
    let single_precision = runtime
        .astype(array, MlxDtype::Float32)
        .expect("values convert");
    runtime
        .array_to_vec_f32(&single_precision)
        .expect("values read back")
}

pub(super) fn largest_difference(reference: &[f32], candidate: &[f32]) -> f32 {
    let mut largest = 0.0_f32;
    for (reference_value, candidate_value) in reference.iter().zip(candidate) {
        largest = largest.max((reference_value - candidate_value).abs());
    }
    largest
}

/// A seeded stream of values in [-1, 1), so a failure reproduces.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_signed(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut mixed = self.0;
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        mixed ^= mixed >> 31;
        ((mixed >> 40) as f32 / (1_u64 << 24) as f32) * 2.0 - 1.0
    }
}

#[test]
#[ignore = "runs MLX kernels on the GPU; run through cargo measure-row-exact-tree-attention"]
#[serial]
fn should_time_row_exact_tree_attention_against_mlx_as_the_context_grows() {
    let runtime = direct_mlx_runtime();
    let kernels = RowExactTreeAttentionKernels::declare().expect("the kernels declare");
    let chain = [None, Some(0), Some(1), Some(2)];
    for context_length in TIMED_CONTEXTS {
        let case = AttentionCase::random(context_length, &chain);
        let token_count = context_length + TREE_ROW_COUNT as i32;
        let shape = [
            1,
            EXPERT_TARGET_SHAPE.key_value_head_count,
            token_count,
            HEAD_SIZE,
        ];
        let keys = bf16_array(&runtime, &case.keys, &shape);
        let values = bf16_array(&runtime, &case.values, &shape);
        let tree_queries = bf16_array(
            &runtime,
            &case.queries,
            &[1, EXPERT_TARGET_SHAPE.query_head_count, 4, HEAD_SIZE],
        );
        let one_query = bf16_array(
            &runtime,
            &case.row_queries(0),
            &[1, EXPERT_TARGET_SHAPE.query_head_count, 1, HEAD_SIZE],
        );
        let scale = attention_scale();
        let mlx_one_row = median_milliseconds(&runtime, &|| {
            runtime
                .scaled_dot_product_attention(&one_query, &keys, &values, scale)
                .expect("MLX attends")
        });
        let mlx_four_rows = median_milliseconds(&runtime, &|| {
            runtime
                .scaled_dot_product_attention(&tree_queries, &keys, &values, scale)
                .expect("MLX attends")
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
        let kernel_four_rows = median_milliseconds(&runtime, &|| {
            attend(
                &runtime,
                &kernels,
                &tree_queries,
                &keys,
                &values,
                context_length,
                &chain,
            )
        });
        eprintln!(
            "[row-exact-tree-attention] {context_length:>6} keys, x{FULL_ATTENTION_LAYER_COUNT} layers: MLX 1 row {:.1} ms, MLX 4 rows {:.1} ms | kernel 1 row {:.1} ms, kernel 4 rows {:.1} ms",
            mlx_one_row * FULL_ATTENTION_LAYER_COUNT,
            mlx_four_rows * FULL_ATTENTION_LAYER_COUNT,
            kernel_one_row * FULL_ATTENTION_LAYER_COUNT,
            kernel_four_rows * FULL_ATTENTION_LAYER_COUNT,
        );
    }
}

pub(super) fn median_milliseconds(runtime: &MlxRuntime, build: &dyn Fn() -> MlxArray) -> f64 {
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

pub(super) fn direct_mlx_runtime() -> MlxRuntime {
    MlxRuntime::initialize(
        MlxMemoryLimits::new(
            DIRECT_MLX_TEST_ACTIVE_MEMORY_LIMIT_BYTES,
            DIRECT_MLX_TEST_MLX_BUFFER_CACHE_LIMIT_BYTES,
        )
        .expect("the memory limits are valid"),
    )
    .expect("the direct MLX runtime initializes")
}
