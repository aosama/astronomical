// A row's best K token ids in one launch, one threadgroup a row: logits
// [1, rows, V] in, token_ids [1, rows, K] out, best first. What one or two
// rows run.
//
// @body astro_primitive_vocabulary_top_k_body.metal
// @input logits bfloat16_t|float16_t|float
// @output token_ids uint32_t
// @template K int
// @template V int
// @compile-with K=20 V=248320
// @compile-with K=1 V=8
// @contract model-primitives gpu_token_sampling::vocabulary_top_k_tests
// @probe none, it uses no matrix units or other chip feature a probe would need to prove

    const uint row = threadgroup_position_in_grid.y;
    constexpr uint CANDIDATES = V;
    const device auto* row_logits = logits + (size_t)row * V;
    #define CANDIDATE_VALUE(index) float(row_logits[index])
    #define CANDIDATE_ID(index) (index)
    #define WRITE_BEST(round, value, id) { token_ids[(size_t)row * K + (round)] = (id); }
