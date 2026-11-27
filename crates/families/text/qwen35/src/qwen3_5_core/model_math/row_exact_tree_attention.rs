//! Full attention for the rows of a draft tree that gives every row the
//! bits its one-row step gives, on the M5 matrix units.
//!
//! A draft-tree round verifies several rows, and a row verified there must
//! give the bits its one-row step gives, or a near-tied greedy pick flips.
//! MLX's attention changes kernels with the number of query rows (exact up
//! to 4 rows at this shape, then not), so this kernel is the same code for
//! one row and for a tree.
//!
//! The arithmetic follows TensorFold's tree attention (`attention_mpp.metal`,
//! `q27_attn_prefix` and `q27_attn_tail`, release 1.2.9, Apache-2.0): a row
//! sees its keys as one sequence (the context, then its ancestors in depth
//! order ending with itself), cut into tiles and chunks by key index alone;
//! the matrix units score 16 query vectors against a 32-key tile, the running
//! softmax works on each lane's own elements, and chunk partials merge in
//! chunk order. A key-value head's query vectors pack head by head
//! (`vector = head * rows + row`), 16 to a matrix tile, a pair of simdgroups
//! a tile, each owning half of the 256 dimensions of the output.
//!
//! Every tile is staged in threadgroup memory. A tile holding only context
//! keys is staged once for every row; a tile reaching past the context is
//! staged once per tree row along that row's chain, and in that pass every
//! other row is masked so its state is left exactly as it was. Either way a
//! row's arithmetic is the same instructions on the same values, so a row in
//! a tree equals its one-row step.
//!
//! A mixture-of-experts target verifies a tree through this kernel once the
//! context reaches `ROW_EXACT_TREE_ATTENTION_MINIMUM_CONTEXT_TOKENS` (below
//! it MLX's attention is as cheap). A dense target, whose drafted reply
//! must equal its plain one, attends through it at every context length,
//! its one-token steps included (`one_row_exact_attention`). Either falls
//! back to MLX's attention for inputs the kernel does not take
//! (`row_exact_tree_attention_accepts`).
//!
//! The query vectors of a key-value head are its query heads times the
//! tree's rows, 16 to a matrix tile: 4 rows of the 35B-A3B's 8 heads fill
//! two tiles, 16 rows of the dense 27B's 6 fill six, and a threadgroup
//! holds eight.
//!
//! Earlier versions, all exact, were set aside on speed, measured with
//! `cargo measure-row-exact-tree-attention` at 65,536 keys over 10 layers:
//! a simdgroup per query head on the vector units matched MLX for one row
//! (11.5 ms) but cost 2.7 times that for four; 8x8 `simdgroup_matrix`
//! fragments ran 105 ms for four rows, their multiplies waiting on each
//! other and their fragments rebuilt element by element.
//!
//! The Metal is in `metal/astro_family_qwen35_row_exact_tree_attention_*`.

use mlx_bridge::{
    MlxArray, MlxDtype, MlxMetalKernel, MlxMetalKernelOutput, MlxMetalKernelTemplateArgument,
};
use runtime_integration::{MlxRuntime, MlxRuntimeError};

use super::gated_delta_tree::UploadedTreeRowChains;
use crate::metal_kernels::{
    ASTRO_FAMILY_QWEN35_ROW_EXACT_TREE_ATTENTION_MERGE,
    ASTRO_FAMILY_QWEN35_ROW_EXACT_TREE_ATTENTION_PARTIALS,
};
use model_primitives::metal_kernel_declaration::declare_metal_kernel;

/// Keys a chunk holds. A fixed width keeps a row's chunks the same in any
/// round; 256 gives 256 chunks of work at 65,536 keys.
const CHUNK_KEY_COUNT: i32 = 256;
pub const HEAD_SIZE: i32 = 256;
const SIMDGROUP_WIDTH: i32 = 32;
/// Query vectors a matrix tile holds.
const TILE_VECTOR_COUNT: i32 = 16;
/// Query heads a key-value head serves, for the targets the kernel is
/// proven on: 8 on the 35B-A3B and 6 on the dense 27B. A row's vectors are
/// `rows * group`, packed 16 to a matrix tile, and nothing in a lane's
/// arithmetic depends on where in a tile its vector sits, so the group only
/// decides how many tiles a tree fills.
pub const SUPPORTED_QUERY_GROUPS: [i32; 2] = [6, 8];
/// Tree rows one dispatch verifies: a dense target's 16-row tree.
pub const MAXIMUM_ROW_COUNT: i32 = 16;
/// Matrix tiles of query vectors one dispatch holds: a threadgroup has 64
/// threads a tile and Metal allows 1,024. Sixteen rows of 8 heads fill 8.
const MAXIMUM_ROW_TILE_COUNT: i32 = 8;

/// Whether a target's attention has a shape the kernel is proven on: head
/// size 256 and a whole, supported number of query heads a key-value head.
#[must_use]
pub fn fits_attention_shape(
    head_dimension: u32,
    query_head_count: u32,
    key_value_head_count: u32,
) -> bool {
    if head_dimension != HEAD_SIZE as u32 || key_value_head_count == 0 {
        return false;
    }
    query_head_count.is_multiple_of(key_value_head_count)
        && SUPPORTED_QUERY_GROUPS.contains(&((query_head_count / key_value_head_count) as i32))
}

/// The two passes, declared once per model.
#[derive(Debug)]
pub struct RowExactTreeAttentionKernels {
    partials: MlxMetalKernel,
    merge: MlxMetalKernel,
}

impl RowExactTreeAttentionKernels {
    pub fn declare() -> Result<Self, MlxRuntimeError> {
        Ok(Self {
            partials: declare_metal_kernel(
                &ASTRO_FAMILY_QWEN35_ROW_EXACT_TREE_ATTENTION_PARTIALS,
                &[],
            )?,
            merge: declare_metal_kernel(&ASTRO_FAMILY_QWEN35_ROW_EXACT_TREE_ATTENTION_MERGE, &[])?,
        })
    }
}

/// One tree's attention over a full-attention layer's stored state.
pub struct RowExactTreeAttentionInputs<'inputs> {
    /// `[1, query heads, rows, 256]` bf16, rotated.
    pub queries: &'inputs MlxArray,
    /// The layer's whole key and value slabs, `[1, key-value heads,
    /// capacity, 256]` bf16, with the tree's rows already written after the
    /// context. The slabs are passed whole because a kernel input must be
    /// contiguous, and a view of the first tokens would be copied.
    pub key_slab: &'inputs MlxArray,
    pub value_slab: &'inputs MlxArray,
    /// Tokens stored before the tree's anchor.
    pub context_length: i32,
    pub row_chains: &'inputs UploadedTreeRowChains,
    pub scale: f32,
}

/// Whether the kernel takes these inputs: bf16 queries and slabs, head size
/// 256, 6 or 8 query heads a key-value head, at most 16 rows in at most 8
/// matrix tiles, and the rows stored after the context. A caller falls back to MLX's attention otherwise.
#[must_use]
pub fn row_exact_tree_attention_accepts(inputs: &RowExactTreeAttentionInputs<'_>) -> bool {
    validated_shape(inputs).is_ok()
}

/// `[1, query heads, rows, 256]`: every row's attention over the context and
/// its own ancestors, in the queries' dtype.
pub fn row_exact_tree_attention(
    runtime: &MlxRuntime,
    kernels: &RowExactTreeAttentionKernels,
    inputs: &RowExactTreeAttentionInputs<'_>,
) -> Result<MlxArray, MlxRuntimeError> {
    let shape = validated_shape(inputs)?;
    let packed_queries = packed_queries(runtime, inputs.queries, &shape)?;
    let context_length = runtime.array_from_i32(&[inputs.context_length], &[1])?;
    let scale = runtime.array_from_f32(&[inputs.scale], &[1])?;
    let partials = run_partials(
        runtime,
        kernels,
        inputs,
        &shape,
        &[&packed_queries, &context_length, &scale],
    )?;
    run_merge(runtime, kernels, inputs, &shape, &partials, &context_length)
}

struct TreeAttentionShape {
    key_value_head_count: i32,
    /// Query heads a key-value head serves.
    query_group: i32,
    row_count: i32,
    padded_vector_count: i32,
    chunk_count: i32,
}

impl TreeAttentionShape {
    fn row_tile_count(&self) -> i32 {
        self.padded_vector_count / TILE_VECTOR_COUNT
    }

    /// Query vectors a key-value head attends for: every row of each of its
    /// query heads.
    fn vector_count(&self) -> i32 {
        self.row_count * self.query_group
    }
}

/// Query vectors padded to whole matrix tiles.
fn padded_to_whole_tiles(vector_count: i32) -> i32 {
    (vector_count + TILE_VECTOR_COUNT - 1) / TILE_VECTOR_COUNT * TILE_VECTOR_COUNT
}

/// The query heads a key-value head serves, when it is a whole number the
/// kernel is proven on and the tree's vectors fit the matrix tiles a
/// threadgroup holds.
fn supported_query_group(query_shape: &[i32], slab_shape: &[i32], row_count: i32) -> Option<i32> {
    let (query_head_count, key_value_head_count) = (*query_shape.get(1)?, *slab_shape.get(1)?);
    if key_value_head_count <= 0 || query_head_count % key_value_head_count != 0 {
        return None;
    }
    let query_group = query_head_count / key_value_head_count;
    let fits_the_tiles = padded_to_whole_tiles(row_count * query_group)
        <= MAXIMUM_ROW_TILE_COUNT * TILE_VECTOR_COUNT;
    (SUPPORTED_QUERY_GROUPS.contains(&query_group) && fits_the_tiles).then_some(query_group)
}

fn validated_shape(
    inputs: &RowExactTreeAttentionInputs<'_>,
) -> Result<TreeAttentionShape, MlxRuntimeError> {
    let query_shape = inputs.queries.shape();
    let slab_shape = inputs.key_slab.shape();
    let row_count = inputs.row_chains.row_count();
    let query_group = supported_query_group(&query_shape, &slab_shape, row_count).unwrap_or(0);
    let is_valid = query_shape.len() == 4
        && slab_shape.len() == 4
        && query_shape[3] == HEAD_SIZE
        && slab_shape[3] == HEAD_SIZE
        && query_shape[2] == row_count
        && row_count <= MAXIMUM_ROW_COUNT
        && inputs.value_slab.shape() == slab_shape
        && slab_shape[1] > 0
        && query_group > 0
        && inputs.queries.dtype() == MlxDtype::BFloat16
        && inputs.key_slab.dtype() == MlxDtype::BFloat16
        && inputs.value_slab.dtype() == MlxDtype::BFloat16
        // A chain is never longer than the tree's rows, and the rows are
        // what the slab holds after the context; `maximum_chain` is padded
        // to a power of two and may pass the slab's end.
        && inputs.context_length + row_count <= slab_shape[2];
    if !is_valid {
        return Err(attention_error(format!(
            "queries {query_shape:?} and slabs {slab_shape:?} do not fit {row_count} rows \
             after {} context tokens: bf16, head size {HEAD_SIZE}, one of \
             {SUPPORTED_QUERY_GROUPS:?} query heads a key-value head and at most \
             {MAXIMUM_ROW_COUNT} rows",
            inputs.context_length
        )));
    }
    let longest_sequence = inputs.context_length + row_count;
    Ok(TreeAttentionShape {
        key_value_head_count: slab_shape[1],
        query_group,
        row_count,
        padded_vector_count: padded_to_whole_tiles(row_count * query_group),
        chunk_count: (longest_sequence + CHUNK_KEY_COUNT - 1) / CHUNK_KEY_COUNT,
    })
}

/// `[key-value heads, padded vectors, 256]`: each key-value head's vectors
/// row by row, every query head of a row together, padded with zero vectors
/// to whole matrix tiles. Packing a row's vectors together lets a tile sit
/// out the passes of rows it does not hold: with heads one after another, a
/// 16-row tree put every row in every tile, and each of the 16 passes over a
/// key tile past the context multiplied all 96 vectors to use 6.
fn packed_queries(
    runtime: &MlxRuntime,
    queries: &MlxArray,
    shape: &TreeAttentionShape,
) -> Result<MlxArray, MlxRuntimeError> {
    let vector_count = shape.vector_count();
    let packed = queries_row_by_row(runtime, queries, shape)?;
    if vector_count == shape.padded_vector_count {
        return Ok(packed);
    }
    let padding = runtime.zeros(
        &[
            shape.key_value_head_count,
            shape.padded_vector_count - vector_count,
            HEAD_SIZE,
        ],
        MlxDtype::BFloat16,
    )?;
    Ok(runtime.concatenate_axis(&[&packed, &padding], 1)?)
}

/// `[1, heads, rows, 256]` as each key-value head's vectors row by row,
/// `[key-value heads, rows * group, 256]` with `vector = row * group + head`.
fn queries_row_by_row(
    runtime: &MlxRuntime,
    queries: &MlxArray,
    shape: &TreeAttentionShape,
) -> Result<MlxArray, MlxRuntimeError> {
    let by_group = runtime.reshape(
        queries,
        &[
            shape.key_value_head_count,
            shape.query_group,
            shape.row_count,
            HEAD_SIZE,
        ],
    )?;
    let row_by_row = runtime.transpose_axes(&by_group, &[0, 2, 1, 3])?;
    Ok(runtime.reshape(
        &row_by_row,
        &[shape.key_value_head_count, shape.vector_count(), HEAD_SIZE],
    )?)
}

fn template_arguments(
    shape: &TreeAttentionShape,
    maximum_chain: i32,
) -> Vec<MlxMetalKernelTemplateArgument> {
    vec![
        MlxMetalKernelTemplateArgument::Dtype {
            name: "T",
            dtype: MlxDtype::BFloat16,
        },
        integer_argument("QUERY_GROUP", shape.query_group),
        integer_argument("ROW_TILES", shape.row_tile_count()),
        integer_argument("CHUNK_KEYS", CHUNK_KEY_COUNT),
        integer_argument("MAXIMUM_CHAIN", maximum_chain),
    ]
}

fn integer_argument(name: &'static str, value: i32) -> MlxMetalKernelTemplateArgument {
    MlxMetalKernelTemplateArgument::Integer {
        name,
        integer_template_argument: value,
    }
}

/// `per_call_inputs` are the packed queries, the context length and the scale.
fn run_partials(
    runtime: &MlxRuntime,
    kernels: &RowExactTreeAttentionKernels,
    inputs: &RowExactTreeAttentionInputs<'_>,
    shape: &TreeAttentionShape,
    per_call_inputs: &[&MlxArray; 3],
) -> Result<Vec<MlxArray>, MlxRuntimeError> {
    let [packed_queries, context_length, scale] = *per_call_inputs;
    let partial_rows = vec![
        shape.key_value_head_count,
        shape.chunk_count,
        shape.padded_vector_count,
    ];
    let mut output_rows = partial_rows.clone();
    output_rows.push(HEAD_SIZE);
    let threadgroup_width = 2 * SIMDGROUP_WIDTH * shape.row_tile_count();
    Ok(runtime.apply_metal_kernel(
        &kernels.partials,
        &[
            packed_queries,
            inputs.key_slab,
            inputs.value_slab,
            context_length,
            scale,
            inputs.row_chains.row_chains(),
            inputs.row_chains.chain_lengths(),
            &runtime.array_from_i32(&[shape.chunk_count], &[1])?,
        ],
        &[
            MlxMetalKernelOutput::new(output_rows, MlxDtype::Float32),
            MlxMetalKernelOutput::new(partial_rows.clone(), MlxDtype::Float32),
            MlxMetalKernelOutput::new(partial_rows, MlxDtype::Float32),
        ],
        [
            threadgroup_width * shape.key_value_head_count,
            shape.chunk_count,
            1,
        ],
        [threadgroup_width, 1, 1],
        &template_arguments(shape, inputs.row_chains.maximum_chain()),
    )?)
}

fn run_merge(
    runtime: &MlxRuntime,
    kernels: &RowExactTreeAttentionKernels,
    inputs: &RowExactTreeAttentionInputs<'_>,
    shape: &TreeAttentionShape,
    partials: &[MlxArray],
    context_length: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    let [chunk_outputs, chunk_maxima, chunk_sums] = partials else {
        return Err(attention_error(
            "the partials pass returned the wrong number of outputs".to_owned(),
        ));
    };
    let vector_count = shape.vector_count();
    let mut outputs = runtime.apply_metal_kernel(
        &kernels.merge,
        &[
            chunk_outputs,
            chunk_maxima,
            chunk_sums,
            context_length,
            inputs.row_chains.chain_lengths(),
        ],
        &[MlxMetalKernelOutput::new(
            inputs.queries.shape(),
            MlxDtype::BFloat16,
        )],
        [
            SIMDGROUP_WIDTH * vector_count,
            shape.key_value_head_count,
            1,
        ],
        [SIMDGROUP_WIDTH, 1, 1],
        &template_arguments(shape, inputs.row_chains.maximum_chain()),
    )?;
    outputs
        .pop()
        .ok_or_else(|| attention_error("the merge pass returned no output".to_owned()))
}

fn attention_error(description: String) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: "attend over a draft tree row-exactly",
        description,
    }
}
