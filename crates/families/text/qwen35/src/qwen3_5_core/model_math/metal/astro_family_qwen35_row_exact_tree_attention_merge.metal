// The second pass of row-exact tree attention: a row's chunk partials merged
// in chunk order into its attended output, so the merge order depends on key
// index alone and never on how many rows share the call.
//
// @input chunk_outputs float
// @input chunk_maxima float
// @input chunk_sums float
// @input context_length int32_t always-small
// @input chain_lengths int32_t may-be-small
// @output attended T
// @template T typename
// @template QUERY_GROUP int
// @template ROW_TILES int
// @template CHUNK_KEYS int
// @template MAXIMUM_CHAIN int
// @compile-with T=bfloat16_t QUERY_GROUP=8 ROW_TILES=2 CHUNK_KEYS=256 MAXIMUM_CHAIN=4
// @compile-with T=bfloat16_t QUERY_GROUP=6 ROW_TILES=6 CHUNK_KEYS=256 MAXIMUM_CHAIN=1
// @contract direct_mlx::qwen3_5::row_exact_tree_attention
// @probe none, no capability probe exists for it yet

    constexpr int lane_dimensions = 8;
    const uint lane = thread_index_in_simdgroup;
    const int query_vector = threadgroup_position_in_grid.x;
    const int key_value_head = threadgroup_position_in_grid.y;
    const int row_count = chain_lengths_shape[0];
    const int chunk_count = chunk_maxima_shape[1];
    const int padded_vectors = chunk_maxima_shape[2];
    const int tree_row = query_vector / QUERY_GROUP;
    const int head = key_value_head * QUERY_GROUP + query_vector % QUERY_GROUP;
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
