// The chunk pass of a chunked top-k: logits [1, rows, V] in, each of a
// row's C chunks' best K logits and token ids out, [rows, C, K] each.
// Threadgroup x is the chunk, y the row, so a round's few rows still fill
// the GPU.
//
// @body astro_primitive_vocabulary_top_k_body.metal
// @input logits bfloat16_t|float16_t|float
// @output chunk_values float
// @output chunk_ids uint32_t
// @template K int
// @template V int
// @template C int
// @compile-with K=20 V=248320 C=32
// @compile-with K=1 V=8 C=1
// @contract model-primitives gpu_token_sampling::vocabulary_top_k_tests
// @probe none, it uses no matrix units or other chip feature a probe would need to prove

    const uint row = threadgroup_position_in_grid.y;
    const uint chunk = threadgroup_position_in_grid.x;
    constexpr uint SPAN = (V + C - 1) / C;
    const uint chunk_start = chunk * SPAN;
    const uint chunk_end = min(chunk_start + SPAN, uint(V));
    const uint CANDIDATES = chunk_end > chunk_start ? chunk_end - chunk_start : 0;
    const device auto* chunk_logits = logits + (size_t)row * V + chunk_start;
    const size_t list_base = ((size_t)row * C + chunk) * K;
    #define CANDIDATE_VALUE(index) float(chunk_logits[index])
    #define CANDIDATE_ID(index) (chunk_start + (index))
    #define WRITE_BEST(round, value, id) \
        { chunk_values[list_base + (round)] = (value); chunk_ids[list_base + (round)] = (id); }
