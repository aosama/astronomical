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
//! Serving verifies a tree through this kernel once the context reaches
//! `ROW_EXACT_TREE_ATTENTION_MINIMUM_CONTEXT_TOKENS` (below it MLX's attention
//! is as cheap), and falls back to MLX's attention for inputs it does not
//! take (`row_exact_tree_attention_accepts`).
//!
//! Earlier versions, all exact, were set aside on speed, measured with
//! `cargo measure-row-exact-tree-attention` at 65,536 keys over 10 layers:
//! a simdgroup per query head on the vector units matched MLX for one row
//! (11.5 ms) but cost 2.7 times that for four; 8x8 `simdgroup_matrix`
//! fragments ran 105 ms for four rows, their multiplies waiting on each
//! other and their fragments rebuilt element by element.

use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};
use mlx_bridge::{
    MlxArray, MlxDtype, MlxMetalKernel, MlxMetalKernelOutput, MlxMetalKernelTemplateArgument,
};

use super::gated_delta_tree::UploadedTreeRowChains;
use crate::row_batched_matmul::kernel_source::ROW_BATCHED_MATMUL_HEADER;

/// Keys a chunk holds. A fixed width keeps a row's chunks the same in any
/// round; 256 gives 256 chunks of work at 65,536 keys.
const CHUNK_KEY_COUNT: i32 = 256;
pub const HEAD_SIZE: i32 = 256;
const SIMDGROUP_WIDTH: i32 = 32;
/// Query vectors a matrix tile holds.
const TILE_VECTOR_COUNT: i32 = 16;
/// Query heads a key-value head serves. The 35B-A3B target has 8.
pub const SUPPORTED_QUERY_GROUP: i32 = 8;
/// Tree rows one dispatch verifies: the 4-row round a mixture-of-experts
/// target drafts, two matrix tiles of 16 vectors.
pub const MAXIMUM_ROW_COUNT: i32 = 4;

const PARTIALS_SOURCE: &str = r#"
  constexpr int head_size = 256;
  constexpr int tile_keys = 32;
  // The weights enter the value product in half precision, while their
  // sum stays float32. Below about 6e-5 of the row's largest weight a half
  // is subnormal, so scaling by 2^14 keeps half's 11 bits down to about
  // 4e-9 of it; the largest, 16,384, stays inside half's range, and the
  // output is scaled back on its way out. Powers of two make both scalings
  // exact. On the 35B-A3B target's real attention it changed no measured
  // output; it keeps a context of many small weights from losing mass.
  constexpr float probability_scale = 16384.0f;
  const ushort lane = thread_index_in_simdgroup;
  const ushort simdgroup = simdgroup_index_in_threadgroup;
  const int thread_index = int(thread_position_in_threadgroup.x);
  const int thread_count = ROW_TILES * 64;
  const int row_tile = simdgroup >> 1;
  const bool upper_half = (simdgroup & 1) != 0;
  const int key_value_head = threadgroup_position_in_grid.x;
  const int chunk = threadgroup_position_in_grid.y;
  const int row_count = chain_lengths_shape[0];
  const int padded_vectors = packed_queries_shape[1];
  const int vector_count = row_count * 8;
  const int capacity = key_slab_shape[2];
  // An input, not a template argument: the count grows with the context,
  // and a template argument would compile the kernel again at every size.
  const int chunk_count = chunk_count_input[0];
  const int context = context_length[0];
  const short quad = lane >> 2;
  const short fragment_row = (quad & 4) | ((lane >> 1) & 3);
  const short fragment_column = ((quad & 2) | (lane & 1)) * 4;
  // This lane's elements belong to two query vectors, 8 apart in the tile.
  const int first_vector = row_tile * 16 + fragment_row;
  const int second_vector = first_vector + 8;
  const int first_tree_row = first_vector < vector_count ? first_vector % row_count : -1;
  const int second_tree_row = second_vector < vector_count ? second_vector % row_count : -1;
  const int first_seen = first_tree_row >= 0 ? context + chain_lengths[first_tree_row] : 0;
  const int second_seen = second_tree_row >= 0 ? context + chain_lengths[second_tree_row] : 0;
  int longest_seen = context;
  for (int tree_row = 0; tree_row < row_count; ++tree_row) {
    longest_seen = max(longest_seen, context + chain_lengths[tree_row]);
  }
  const int first_key = chunk * CHUNK_KEYS;
  const int chunk_end = min(first_key + CHUNK_KEYS, longest_seen);

  threadgroup bfloat staged_tile[tile_keys * head_size];
  threadgroup half probability_type_holder[16 * tile_keys];
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> query_tensor(
      (device bfloat*)packed_queries + (int64_t)key_value_head * padded_vectors * head_size,
      dextents<int32_t, 2>(head_size, padded_vectors));
  tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline> tile_tensor(
      staged_tile, dextents<int32_t, 2>(head_size, tile_keys));
  tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> probability_tensor(
      probability_type_holder, dextents<int32_t, 2>(tile_keys, 16));
  constexpr auto score_descriptor = matmul2d_descriptor(
      16, tile_keys, head_size, false, true, false, matmul2d_descriptor::mode::multiply);
  constexpr auto output_descriptor = matmul2d_descriptor(
      16, 128, tile_keys, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<score_descriptor, execution_simdgroup> score_operation;
  matmul2d<output_descriptor, execution_simdgroup> output_operation;
  auto query_rows = query_tensor.slice(0, row_tile * 16);
  auto value_half = tile_tensor.slice(upper_half ? 128 : 0, 0);
  auto output = output_operation.template get_destination_cooperative_tensor<
      decltype(probability_tensor), decltype(value_half), float>();
  for (int element = 0; element < 64; ++element) output[element] = 0.0f;
  float first_maximum = -INFINITY, second_maximum = -INFINITY;
  float first_sum = 0.0f, second_sum = 0.0f;
  const size_t head_offset = (size_t)key_value_head * capacity * head_size;

  for (int tile_start = first_key; tile_start < chunk_end; tile_start += tile_keys) {
    // Context keys are the same for every row; a tile reaching past them is
    // staged for one tree row at a time along its chain.
    const bool shared_tile = tile_start + tile_keys <= context;
    const int pass_count = shared_tile ? 1 : row_count;
    for (int pass = 0; pass < pass_count; ++pass) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      stage_tile(staged_tile, key_slab, head_offset, tile_start, context, shared_tile,
          pass, chain_lengths, row_chains, MAXIMUM_CHAIN, thread_index, thread_count);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      auto scores = score_operation.template get_destination_cooperative_tensor<
          decltype(query_rows), decltype(tile_tensor), float>();
      score_operation.run(query_rows, tile_tensor, scores);
      const bool first_in_pass = shared_tile || first_tree_row == pass;
      const bool second_in_pass = shared_tile || second_tree_row == pass;
      float tile_scores[tile_keys / 2];
      for (int element = 0; element < tile_keys / 2; ++element) {
        const int key = tile_start + (element >> 3) * 16 + fragment_column + (element & 3);
        const bool is_second = (element & 4) != 0;
        const bool is_seen = is_second ? (second_in_pass && key < second_seen)
                                       : (first_in_pass && key < first_seen);
        tile_scores[element] = is_seen ? scores[element] * scale[0] : -INFINITY;
      }
      float first_tile_maximum = -INFINITY, second_tile_maximum = -INFINITY;
      for (int element = 0; element < tile_keys / 2; ++element) {
        if (element & 4) second_tile_maximum = max(second_tile_maximum, tile_scores[element]);
        else first_tile_maximum = max(first_tile_maximum, tile_scores[element]);
      }
      first_tile_maximum = max(first_tile_maximum, simd_shuffle_xor(first_tile_maximum, 1));
      first_tile_maximum = max(first_tile_maximum, simd_shuffle_xor(first_tile_maximum, 8));
      second_tile_maximum = max(second_tile_maximum, simd_shuffle_xor(second_tile_maximum, 1));
      second_tile_maximum = max(second_tile_maximum, simd_shuffle_xor(second_tile_maximum, 8));
      const float first_next = max(first_maximum, first_tile_maximum);
      const float second_next = max(second_maximum, second_tile_maximum);
      const float first_rescale = first_tile_maximum == -INFINITY ? 1.0f : fast::exp(first_maximum - first_next);
      const float second_rescale = second_tile_maximum == -INFINITY ? 1.0f : fast::exp(second_maximum - second_next);
      float weights[tile_keys / 2];
      for (int element = 0; element < tile_keys / 2; ++element) {
        weights[element] = tile_scores[element] == -INFINITY ? 0.0f
            : fast::exp(tile_scores[element] - ((element & 4) ? second_next : first_next));
      }
      float first_tile_sum = 0.0f, second_tile_sum = 0.0f;
      for (int block = 0; block < tile_keys / 16; ++block) {
        first_tile_sum += (weights[block * 8] + weights[block * 8 + 1]) + (weights[block * 8 + 2] + weights[block * 8 + 3]);
        second_tile_sum += (weights[block * 8 + 4] + weights[block * 8 + 5]) + (weights[block * 8 + 6] + weights[block * 8 + 7]);
      }
      first_tile_sum += simd_shuffle_xor(first_tile_sum, 1);
      first_tile_sum += simd_shuffle_xor(first_tile_sum, 8);
      second_tile_sum += simd_shuffle_xor(second_tile_sum, 1);
      second_tile_sum += simd_shuffle_xor(second_tile_sum, 8);
      if (first_tile_maximum != -INFINITY) { first_sum = first_sum * first_rescale + first_tile_sum; first_maximum = first_next; }
      if (second_tile_maximum != -INFINITY) { second_sum = second_sum * second_rescale + second_tile_sum; second_maximum = second_next; }
      auto probabilities = score_operation.template get_destination_cooperative_tensor<
          decltype(query_rows), decltype(tile_tensor), half>();
      for (int element = 0; element < tile_keys / 2; ++element) {
        probabilities[element] = half(weights[element] * probability_scale);
      }
      auto probability_input = output_operation.template get_left_input_cooperative_tensor<
          half, bfloat, float>(probabilities);
      for (int element = 0; element < 64; ++element) output[element] *= (element & 4) ? second_rescale : first_rescale;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      stage_tile(staged_tile, value_slab, head_offset, tile_start, context, shared_tile,
          pass, chain_lengths, row_chains, MAXIMUM_CHAIN, thread_index, thread_count);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      output_operation.run(probability_input, value_half, output);
    }
  }
  const size_t partial_base = ((size_t)key_value_head * chunk_count + chunk) * padded_vectors;
  for (int pair = 0; pair < 16; ++pair) {
    const int vector = row_tile * 16 + fragment_row + (pair & 1) * 8;
    if (vector >= vector_count) continue;
    device float* destination = chunk_outputs + (partial_base + vector) * head_size
        + (upper_half ? 128 : 0) + (pair >> 1) * 16 + fragment_column;
    *(device float4*)destination = float4(output[4 * pair], output[4 * pair + 1],
        output[4 * pair + 2], output[4 * pair + 3]) * (1.0f / probability_scale);
  }
  if (!upper_half && (lane & 9) == 0) {
    if (first_vector < vector_count) {
      chunk_maxima[partial_base + first_vector] = first_maximum;
      chunk_sums[partial_base + first_vector] = first_sum;
    }
    if (second_vector < vector_count) {
      chunk_maxima[partial_base + second_vector] = second_maximum;
      chunk_sums[partial_base + second_vector] = second_sum;
    }
  }
"#;

/// Stages one 32-key tile of a key or value slab in threadgroup memory: the
/// context's own keys, or, past the context, `source_row`'s ancestors in
/// depth order; keys that row does not see are zero.
const STAGE_TILE_HEADER: &str = r#"
// Pointer types are template parameters: MLX may place a small input in the
// constant address space and a large one in device memory.
template <typename SlabPointer, typename LengthsPointer, typename ChainsPointer>
inline void stage_tile(threadgroup bfloat* staged_tile, SlabPointer slab,
    size_t head_offset, int tile_start, int context, bool shared_tile, int source_row,
    LengthsPointer chain_lengths, ChainsPointer row_chains, int maximum_chain,
    int thread_index, int thread_count) {
  constexpr int head_size = 256;
  constexpr int tile_keys = 32;
  for (int element = thread_index; element < tile_keys * head_size / 8; element += thread_count) {
    const int tile_key = element / (head_size / 8);
    const int column = (element % (head_size / 8)) * 8;
    const int logical_key = tile_start + tile_key;
    int physical_key = -1;
    if (logical_key < context) {
      physical_key = logical_key;
    } else if (!shared_tile && logical_key < context + chain_lengths[source_row]) {
      physical_key = context + row_chains[source_row * maximum_chain + (logical_key - context)];
    }
    vec<bfloat, 8> values = vec<bfloat, 8>(bfloat(0.0f));
    if (physical_key >= 0) {
      values = *(const device vec<bfloat, 8>*)(slab + head_offset + (size_t)physical_key * head_size + column);
    }
    *(threadgroup vec<bfloat, 8>*)(staged_tile + tile_key * head_size + column) = values;
  }
}
"#;

const MERGE_SOURCE: &str = r#"
    constexpr int lane_dimensions = 8;
    const uint lane = thread_index_in_simdgroup;
    const int query_vector = threadgroup_position_in_grid.x;
    const int key_value_head = threadgroup_position_in_grid.y;
    const int row_count = chain_lengths_shape[0];
    const int chunk_count = chunk_maxima_shape[1];
    const int padded_vectors = chunk_maxima_shape[2];
    const int tree_row = query_vector % row_count;
    const int head = key_value_head * QUERY_GROUP + query_vector / row_count;
    const int seen_keys = context_length[0] + chain_lengths[tree_row];
    const int own_chunk_count = (seen_keys + CHUNK_KEYS - 1) / CHUNK_KEYS;
    float merged_maximum = -INFINITY;
    float merged_sum = 0.0f;
    float merged[lane_dimensions];
    for (int dimension = 0; dimension < lane_dimensions; ++dimension) {
        merged[dimension] = 0.0f;
    }
    for (int chunk = 0; chunk < own_chunk_count; ++chunk) {
        const size_t partial_row = ((size_t)key_value_head * chunk_count + chunk) * padded_vectors + query_vector;
        const float chunk_maximum = chunk_maxima[partial_row];
        if (chunk_maximum == -INFINITY) {
            continue;
        }
        const float next_maximum = max(merged_maximum, chunk_maximum);
        const float merged_scale = fast::exp(merged_maximum - next_maximum);
        const float chunk_scale = fast::exp(chunk_maximum - next_maximum);
        merged_sum = merged_sum * merged_scale + chunk_sums[partial_row] * chunk_scale;
        for (int dimension = 0; dimension < lane_dimensions; ++dimension) {
            merged[dimension] = merged[dimension] * merged_scale
                + chunk_outputs[partial_row * 256 + lane * lane_dimensions + dimension] * chunk_scale;
        }
        merged_maximum = next_maximum;
    }
    device T* destination = attended + ((size_t)head * row_count + tree_row) * 256 + lane * lane_dimensions;
    for (int dimension = 0; dimension < lane_dimensions; ++dimension) {
        destination[dimension] = T(merged[dimension] / merged_sum);
    }
"#;

/// The two passes, declared once per model.
#[derive(Debug)]
pub struct RowExactTreeAttentionKernels {
    partials: MlxMetalKernel,
    merge: MlxMetalKernel,
}

impl RowExactTreeAttentionKernels {
    pub fn declare() -> Result<Self, MlxRuntimeError> {
        let partials_header = format!("{ROW_BATCHED_MATMUL_HEADER}{STAGE_TILE_HEADER}");
        Ok(Self {
            partials: MlxMetalKernel::new_with_header(
                "astronomical_row_exact_tree_attention_partials",
                &[
                    "packed_queries",
                    "key_slab",
                    "value_slab",
                    "context_length",
                    "scale",
                    "row_chains",
                    "chain_lengths",
                    "chunk_count_input",
                ],
                &["chunk_outputs", "chunk_maxima", "chunk_sums"],
                &partials_header,
                PARTIALS_SOURCE,
            )?,
            merge: MlxMetalKernel::new(
                "astronomical_row_exact_tree_attention_merge",
                &[
                    "chunk_outputs",
                    "chunk_maxima",
                    "chunk_sums",
                    "context_length",
                    "chain_lengths",
                ],
                &["attended"],
                MERGE_SOURCE,
            )?,
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
/// 256, 8 query heads a key-value head, at most 4 rows, and the rows stored
/// after the context. A caller falls back to MLX's attention otherwise.
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
    row_count: i32,
    padded_vector_count: i32,
    chunk_count: i32,
}

impl TreeAttentionShape {
    fn row_tile_count(&self) -> i32 {
        self.padded_vector_count / TILE_VECTOR_COUNT
    }
}

fn validated_shape(
    inputs: &RowExactTreeAttentionInputs<'_>,
) -> Result<TreeAttentionShape, MlxRuntimeError> {
    let query_shape = inputs.queries.shape();
    let slab_shape = inputs.key_slab.shape();
    let row_count = inputs.row_chains.row_count();
    let is_valid = query_shape.len() == 4
        && slab_shape.len() == 4
        && query_shape[3] == HEAD_SIZE
        && slab_shape[3] == HEAD_SIZE
        && query_shape[2] == row_count
        && row_count <= MAXIMUM_ROW_COUNT
        && inputs.value_slab.shape() == slab_shape
        && slab_shape[1] > 0
        && query_shape[1] == slab_shape[1] * SUPPORTED_QUERY_GROUP
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
             after {} context tokens: bf16, head size {HEAD_SIZE}, {SUPPORTED_QUERY_GROUP} \
             query heads a key-value head and at most {MAXIMUM_ROW_COUNT} rows",
            inputs.context_length
        )));
    }
    let vector_count = row_count * SUPPORTED_QUERY_GROUP;
    let longest_sequence = inputs.context_length + row_count;
    Ok(TreeAttentionShape {
        key_value_head_count: slab_shape[1],
        row_count,
        padded_vector_count: (vector_count + TILE_VECTOR_COUNT - 1) / TILE_VECTOR_COUNT
            * TILE_VECTOR_COUNT,
        chunk_count: (longest_sequence + CHUNK_KEY_COUNT - 1) / CHUNK_KEY_COUNT,
    })
}

/// `[key-value heads, padded vectors, 256]`: each key-value head's query
/// heads one after another, every row of a head together, padded with zero
/// vectors to whole matrix tiles. The queries' own layout already orders
/// them so; only a 1-row tree needs padding.
fn packed_queries(
    runtime: &MlxRuntime,
    queries: &MlxArray,
    shape: &TreeAttentionShape,
) -> Result<MlxArray, MlxRuntimeError> {
    let vector_count = shape.row_count * SUPPORTED_QUERY_GROUP;
    let packed = runtime.reshape(
        queries,
        &[shape.key_value_head_count, vector_count, HEAD_SIZE],
    )?;
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

fn template_arguments(
    shape: &TreeAttentionShape,
    maximum_chain: i32,
) -> Vec<MlxMetalKernelTemplateArgument> {
    vec![
        MlxMetalKernelTemplateArgument::Dtype {
            name: "T",
            dtype: MlxDtype::BFloat16,
        },
        integer_argument("QUERY_GROUP", SUPPORTED_QUERY_GROUP),
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
    let vector_count = shape.row_count * SUPPORTED_QUERY_GROUP;
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
