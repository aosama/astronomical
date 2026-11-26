//! The dense 27B's full attention, the row-exact kernel against MLX's, with
//! a profiler verdict (issue #2601): the one part of a drafted reply that
//! changed kernel when a dense model's drafted reply was made to equal its
//! plain one.
//!
//! Two cells a context length: a reply token's one row (MLX's vector kernel
//! against the row-exact kernel) and a 16-row draft tree (one masked MLX
//! call, which is not row-exact, against the kernel). Sixteen layers' worth
//! of distinct key and value slabs a job, so a pass reads memory as a
//! forward over the 27B's 16 attention layers does. Twelve rounds with the
//! order flipped every round; "faster" or "slower" only when the interval
//! for the median difference excludes zero.
//!
//! Run with `cargo measure-row-exact-attention-verdict`. No model; seconds.

use std::path::Path;

use mlx_bridge::{MlxArray, MlxDtype};
use mlx_performance_profiler::{JobComparison, MeasuredJob, ReportLocation, compare_jobs};
use model_serving::{
    RowExactTreeAttentionInputs, RowExactTreeAttentionKernels, TreeRowChains,
    UploadedTreeRowChains, row_exact_tree_attention,
};
use runtime_integration::MlxRuntime;
use serial_test::serial;

use super::k_row_verify_cost::initialize_runtime;
use super::verdict_text::metric_text;

const QUERY_HEAD_COUNT: i32 = 24;
const KEY_VALUE_HEAD_COUNT: i32 = 4;
const HEAD_SIZE: i32 = 256;
const ATTENTION_LAYER_COUNT: usize = 16;
const TREE_ROW_COUNT: i32 = 16;
/// The ledger's long prompt for this model, and a longer conversation.
const CONTEXT_LENGTHS: [i32; 4] = [120, 1_000, 2_200, 10_240];
const REPORT_SUBDIRECTORY: &str = "row-exact-attention-verdict";

/// One layer's stored keys and values with the tree's rows after the
/// context, and the queries a step attends with.
struct AttentionLayer {
    keys: MlxArray,
    values: MlxArray,
}

/// The inputs a cell shares between its two jobs.
struct AttentionCell<'cell> {
    runtime: &'cell MlxRuntime,
    kernels: &'cell RowExactTreeAttentionKernels,
    layers: &'cell [AttentionLayer],
    queries: &'cell MlxArray,
    context_length: i32,
    tree_mask: Option<&'cell MlxArray>,
    row_chains: &'cell UploadedTreeRowChains,
}

/// Every layer's attention one way.
struct AttentionPass<'cell> {
    name: &'static str,
    is_row_exact: bool,
    cell: &'cell AttentionCell<'cell>,
}

impl MeasuredJob for AttentionPass<'_> {
    fn name(&self) -> &str {
        self.name
    }

    fn fresh_outputs(&mut self) -> Result<Vec<MlxArray>, String> {
        let cell = self.cell;
        let scale = (HEAD_SIZE as f32).sqrt().recip();
        let mut attended_layers = Vec::with_capacity(cell.layers.len());
        for layer in cell.layers {
            let attended = if self.is_row_exact {
                row_exact_tree_attention(
                    cell.runtime,
                    cell.kernels,
                    &RowExactTreeAttentionInputs {
                        queries: cell.queries,
                        key_slab: &layer.keys,
                        value_slab: &layer.values,
                        context_length: cell.context_length,
                        row_chains: cell.row_chains,
                        scale,
                    },
                )
                .map_err(|attention_error| attention_error.to_string())
            } else {
                mlx_attention(cell, layer, scale)
            };
            attended_layers.push(attended?);
        }
        Ok(attended_layers)
    }
}

/// MLX's attention as serving calls it: unmasked for one row, one masked
/// call for a tree.
fn mlx_attention(
    cell: &AttentionCell<'_>,
    layer: &AttentionLayer,
    scale: f32,
) -> Result<MlxArray, String> {
    let attended = match cell.tree_mask {
        Some(tree_mask) => cell.runtime.masked_scaled_dot_product_attention(
            cell.queries,
            &layer.keys,
            &layer.values,
            scale,
            tree_mask,
        ),
        None => cell.runtime.scaled_dot_product_attention(
            cell.queries,
            &layer.keys,
            &layer.values,
            scale,
        ),
    };
    attended.map_err(|attention_error| attention_error.to_string())
}

#[test]
#[ignore = "performance measurement on the GPU; run through cargo measure-row-exact-attention-verdict"]
#[serial]
fn should_give_a_verdict_on_row_exact_attention_against_mlx_at_the_dense_targets_shape() {
    let runtime = initialize_runtime();
    let kernels = RowExactTreeAttentionKernels::declare().expect("the kernels declare");
    let random_key = runtime.random_key(2_601).expect("a key builds");
    let report_directory = Path::new(env!("CARGO_TARGET_TMPDIR")).join(REPORT_SUBDIRECTORY);
    std::fs::create_dir_all(&report_directory).expect("the report directory is made");
    for context_length in CONTEXT_LENGTHS {
        let layers = attention_layers(&runtime, &random_key, context_length + TREE_ROW_COUNT);
        for row_count in [1, TREE_ROW_COUNT] {
            let file_base = format!("context_{context_length}_rows_{row_count}");
            let comparison = compare_cell(
                (&runtime, &kernels, &random_key),
                (&layers, context_length, row_count),
                (&report_directory, &file_base),
            );
            eprintln!(
                "[row-exact-attention-verdict] {file_base}: job {} | GPU {}",
                metric_text(&comparison.job_wall, "row-exact kernel"),
                metric_text(&comparison.gpu_span, "row-exact kernel"),
            );
        }
    }
    eprintln!(
        "[row-exact-attention-verdict] status=success pages={}",
        report_directory.display()
    );
}

/// MLX's attention as the baseline and the row-exact kernel as the
/// candidate, for `row_count` rows after `context_length` stored tokens.
fn compare_cell(
    (runtime, kernels, random_key): (&MlxRuntime, &RowExactTreeAttentionKernels, &MlxArray),
    (layers, context_length, row_count): (&[AttentionLayer], i32, i32),
    (report_directory, file_base): (&Path, &str),
) -> JobComparison {
    let queries = bf16_normal(
        runtime,
        random_key,
        &[1, QUERY_HEAD_COUNT, row_count, HEAD_SIZE],
    );
    let parent_rows = chain_parent_rows(row_count);
    let row_chains = TreeRowChains::from_parent_rows(&parent_rows)
        .upload(runtime)
        .expect("the chains upload");
    let tree_mask = (row_count > 1).then(|| chain_mask(runtime, context_length, row_count));
    // A one-row step attends over everything stored, its own row included.
    let step_context = if row_count == 1 {
        context_length + TREE_ROW_COUNT - 1
    } else {
        context_length
    };
    let cell = AttentionCell {
        runtime,
        kernels,
        layers,
        queries: &queries,
        context_length: step_context,
        tree_mask: tree_mask.as_ref(),
        row_chains: &row_chains,
    };
    let mut mlx_pass = AttentionPass {
        name: "mlx_attention",
        is_row_exact: false,
        cell: &cell,
    };
    let mut kernel_pass = AttentionPass {
        name: "row_exact_kernel",
        is_row_exact: true,
        cell: &cell,
    };
    compare_jobs(
        &mut mlx_pass,
        &mut kernel_pass,
        ReportLocation {
            title: "Row-exact attention against MLX's",
            directory: report_directory,
            file_base,
        },
    )
    .expect("the two attentions are compared")
}

/// A chain: each row's parent is the row before it.
fn chain_parent_rows(row_count: i32) -> Vec<Option<usize>> {
    let mut parent_rows = Vec::with_capacity(row_count as usize);
    for row in 0..row_count as usize {
        parent_rows.push(row.checked_sub(1));
    }
    parent_rows
}

/// A causal chain's additive mask over the context and the tree's rows,
/// `[1, 1, rows, tokens]` bf16.
fn chain_mask(runtime: &MlxRuntime, context_length: i32, row_count: i32) -> MlxArray {
    let token_count = (context_length + row_count) as usize;
    let mut additive = vec![0.0_f32; row_count as usize * token_count];
    for row in 0..row_count as usize {
        for later_row in row + 1..row_count as usize {
            additive[row * token_count + context_length as usize + later_row] = f32::NEG_INFINITY;
        }
    }
    let mask = runtime
        .array_from_f32(&additive, &[1, 1, row_count, token_count as i32])
        .expect("the mask uploads");
    runtime
        .astype(&mask, MlxDtype::BFloat16)
        .expect("the mask converts")
}

fn attention_layers(
    runtime: &MlxRuntime,
    random_key: &MlxArray,
    token_count: i32,
) -> Vec<AttentionLayer> {
    let slab_shape = [1, KEY_VALUE_HEAD_COUNT, token_count, HEAD_SIZE];
    let mut layers = Vec::with_capacity(ATTENTION_LAYER_COUNT);
    for _layer_index in 0..ATTENTION_LAYER_COUNT {
        let layer = AttentionLayer {
            keys: bf16_normal(runtime, random_key, &slab_shape),
            values: bf16_normal(runtime, random_key, &slab_shape),
        };
        runtime
            .evaluate_arrays(&[&layer.keys, &layer.values])
            .expect("the slabs evaluate");
        layers.push(layer);
    }
    layers
}

fn bf16_normal(runtime: &MlxRuntime, random_key: &MlxArray, shape: &[i32]) -> MlxArray {
    let values = runtime
        .random_normal(shape, MlxDtype::BFloat16, 0.0, 1.0, random_key)
        .expect("values are drawn");
    runtime
        .evaluate_arrays(&[&values])
        .expect("values evaluate");
    values
}
