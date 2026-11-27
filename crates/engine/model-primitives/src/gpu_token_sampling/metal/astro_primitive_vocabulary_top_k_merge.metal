// The merge pass of a chunked top-k: every chunk's best K of a row in,
// the row's best K token ids out, [1, rows, K]. One threadgroup a row.
//
// @body astro_primitive_vocabulary_top_k_body.metal
// @input chunk_values float
// @input chunk_ids uint32_t
// @output token_ids uint32_t
// @template K int
// @template V int
// @template C int
// @compile-with K=20 V=248320 C=32
// @compile-with K=1 V=8 C=1
// @contract model-primitives gpu_token_sampling::vocabulary_top_k_tests
// @probe none, it uses no matrix units or other chip feature a probe would need to prove

    const uint row = threadgroup_position_in_grid.y;
    constexpr uint CANDIDATES = C * K;
    const auto row_values = chunk_values + (size_t)row * CANDIDATES;
    const auto row_ids = chunk_ids + (size_t)row * CANDIDATES;
    #define CANDIDATE_VALUE(index) row_values[index]
    #define CANDIDATE_ID(index) row_ids[index]
    #define WRITE_BEST(round, value, id) { token_ids[(size_t)row * K + (round)] = (id); }
