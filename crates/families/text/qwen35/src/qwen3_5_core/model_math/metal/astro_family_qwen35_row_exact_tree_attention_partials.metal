// The first pass of row-exact tree attention: every 256-key chunk's partial
// output, running maximum and sum for every packed query vector of a
// key-value head. The query vectors of a key-value head pack head by head
// (vector = head * rows + row), 16 to a matrix tile, a pair of simdgroups a
// tile, each owning half of the 256 dimensions of the output; the matrix
// units score 16 query vectors against a 32-key tile and the running softmax
// works on each lane's own elements. The launcher's module documents the
// whole design and why a row in a tree equals its one-row step.
//
// @header astro_primitive_matrix_unit_tensor_ops.h astro_family_qwen35_row_exact_tree_attention_stage_tile.h
// @input packed_queries T
// @input key_slab T
// @input value_slab T
// @input context_length int32_t always-small
// @input scale float always-small
// @input row_chains int32_t may-be-small
// @input chain_lengths int32_t may-be-small
// @input chunk_count_input int32_t always-small
// @output chunk_outputs float
// @output chunk_maxima float
// @output chunk_sums float
// @template T typename
// @template QUERY_GROUP int
// @template ROW_TILES int
// @template CHUNK_KEYS int
// @template MAXIMUM_CHAIN int
// @compile-with T=bfloat16_t QUERY_GROUP=8 ROW_TILES=2 CHUNK_KEYS=256 MAXIMUM_CHAIN=4
// @compile-with T=bfloat16_t QUERY_GROUP=6 ROW_TILES=6 CHUNK_KEYS=256 MAXIMUM_CHAIN=1
// @contract direct_mlx::qwen3_5::row_exact_tree_attention
// @probe none, no capability probe exists for it yet

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
  const int vector_count = row_count * QUERY_GROUP;
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
  // Query vectors pack row by row (`vector = row * QUERY_GROUP + head`), so
  // a matrix tile holds a few rows' vectors and sits out the passes of the
  // rest.
  const int first_tree_row = first_vector < vector_count ? first_vector / QUERY_GROUP : -1;
  const int second_tree_row = second_vector < vector_count ? second_vector / QUERY_GROUP : -1;
  const int tile_first_row = (row_tile * 16) / QUERY_GROUP;
  const int tile_last_row = (row_tile * 16 + 15) / QUERY_GROUP;
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
      // A pass staged for a row this tile holds no vector of would leave
      // every one of its lanes exactly as it was (no key seen: rescale 1,
      // weights 0, nothing added), so the tile does no arithmetic for it.
      const bool tile_in_pass = shared_tile || (pass >= tile_first_row && pass <= tile_last_row);
      auto probabilities = score_operation.template get_destination_cooperative_tensor<
          decltype(query_rows), decltype(tile_tensor), half>();
      if (tile_in_pass) {
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
        for (int element = 0; element < tile_keys / 2; ++element) {
          probabilities[element] = half(weights[element] * probability_scale);
        }
        for (int element = 0; element < 64; ++element) output[element] *= (element & 4) ? second_rescale : first_rescale;
      }
      auto probability_input = output_operation.template get_left_input_cooperative_tensor<
          half, bfloat, float>(probabilities);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      stage_tile(staged_tile, value_slab, head_offset, tile_start, context, shared_tile,
          pass, chain_lengths, row_chains, MAXIMUM_CHAIN, thread_index, thread_count);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tile_in_pass) output_operation.run(probability_input, value_half, output);
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
