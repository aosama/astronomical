// The sequence kernel with boundary checkpoints: the fused Qwen 3.5
// gated-delta recurrence over a sequence, software-pipelined. Inputs: queries
// and keys [B, T, Hk, Dk], values [B, T, Hv, Dv], decays and update_rates
// [B, T, Hv], recurrent_state [B, Hv, Dv, Dk] float, and the token count T as
// a scalar. Outputs: [B, T, Hv, Dv] and the next recurrent_state. One
// threadgroup owns 16 value rows of one value head, and eight lanes cooperate
// on each row.
//
// Why the text repeats: the kernel stages twelve tokens per block in
// threadgroup memory and prefetches the next block's keys, values, decays and
// update rates into registers while the current block computes, hiding
// global-memory latency behind the recurrence arithmetic. A full block is
// written as twelve unrolled steps, one per token, each with its literal token
// index. A step publishes its query-dot partial into a four-slot array. At the
// steps with block-local token index 4 and 8, and once more after the last
// step, the previous four steps' partials are reduce-scattered onto the
// staged outputs in one pass, bit-identical to reducing each step on its own.
// A block with fewer than twelve tokens (the tail) runs one rolled step in a
// loop instead: it guards the refill of the next step's operands, carries the
// previous step's partial in a single slot and reduces it at the next step.
// The algorithm mirrors an upstream open-source software-pipelined
// gated-delta kernel (see third-party/THIRD_PARTY_NOTICES for the provenance
// and license).
//
// Boundary checkpoints: at completed-token count first_checkpoint_token_count,
// and every checkpoint_interval_token_count tokens after it, checkpoint_count
// times in all, the state is written to recurrent_boundary_states
// [checkpoint_count, B, Hv, Dv, Dk] float. The write sits at a fixed point
// inside every step body, after the state update, with the step's own token
// index (the literal in an unrolled step, the loop variable in the rolled
// tail). The batch size B is a template integer because that write indexes
// the boundary states with it, while the token count stays a run-time scalar.
//
// @body astro_family_qwen35_gated_delta_sequence_checkpoints_middle_steps_body.metal astro_family_qwen35_gated_delta_sequence_checkpoints_later_steps_body.metal
// @input queries InT
// @input keys InT
// @input values InT
// @input decays bfloat16_t|float16_t|float may-be-small
// @input update_rates bfloat16_t|float16_t|float may-be-small
// @input recurrent_state float
// @input token_count int32_t scalar
// @input first_checkpoint_token_count int32_t scalar
// @input checkpoint_interval_token_count int32_t scalar
// @input checkpoint_count int32_t scalar
// @output outputs InT
// @output next_recurrent_state float
// @output recurrent_boundary_states float
// @template InT typename
// @template StT typename
// @template Dk int
// @template Dv int
// @template Hk int
// @template Hv int
// @template B int
// @compile-with InT=bfloat16_t StT=float Dk=128 Dv=128 Hk=16 Hv=48 B=1
// @compile-with InT=float16_t StT=float Dk=128 Dv=16 Hk=1 Hv=2 B=2
// @contract verification/acceptance/tests/gpu_contracts/direct_mlx/qwen3_5/gated_delta_checkpoints.rs
// @probe gated_delta_boundary_checkpoint

    constexpr int time_block_size = 12;
    constexpr int value_row_block_size = 16;
    constexpr int threadgroup_thread_count = 128;
    auto thread_index = thread_position_in_threadgroup.x;
    auto value_head_index = threadgroup_position_in_grid.y;
    auto batch_index = threadgroup_position_in_grid.z;
    auto key_head_index = value_head_index / (Hv / Hk);
    auto first_value_row = threadgroup_position_in_grid.x * value_row_block_size;
    auto value_row_in_block = thread_index / 8;
    auto key_segment_index = thread_index % 8;

    threadgroup InT staged_keys[time_block_size][Dk];
    threadgroup InT staged_queries[time_block_size][Dk];
    threadgroup float staged_values[time_block_size][value_row_block_size];
    threadgroup float staged_decays[time_block_size];
    threadgroup float staged_update_rates[time_block_size];
    threadgroup InT staged_outputs[time_block_size][value_row_block_size];

    auto key_row_stride = (size_t)Hk * Dk;
    auto value_row_stride = (size_t)Hv * Dv;
    auto key_base = keys +
        ((size_t)batch_index * token_count * Hk + key_head_index) * Dk;
    auto query_base = queries +
        ((size_t)batch_index * token_count * Hk + key_head_index) * Dk;
    auto value_base = values +
        ((size_t)batch_index * token_count * Hv + value_head_index) * Dv +
        first_value_row;
    auto output_base = outputs +
        ((size_t)batch_index * token_count * Hv + value_head_index) * Dv +
        first_value_row;

    // Each lane owns the strided float4 state granules
    // key_segment_index + 8 * fragment_index of its row; the stride keeps
    // threadgroup accesses bank-conflict-free and the four granules cover
    // the full row.
    float4 state_fragment[4];
    {
        auto input_state = (const device float4*)(recurrent_state +
            (((size_t)batch_index * Hv + value_head_index) * Dv +
             first_value_row + value_row_in_block) * Dk);
        for (int fragment_index = 0; fragment_index < 4; ++fragment_index) {
            state_fragment[fragment_index] =
                input_state[key_segment_index + 8 * fragment_index];
        }
    }

    int checkpoint_index = 0;
    int next_checkpoint_token_count = first_checkpoint_token_count;

    // Register prefetch of one block: raw key/query vec4 granules, the value
    // slice, and the decay/update-rate scalars of the NEXT block, loaded
    // while the current block computes.
    constexpr int key_granule_count = time_block_size * Dk / 4;
    constexpr int key_prefetch_slots =
        (key_granule_count + threadgroup_thread_count - 1) /
        threadgroup_thread_count;
    constexpr int value_granule_count =
        time_block_size * value_row_block_size / 4;
    constexpr int value_prefetch_slots =
        (value_granule_count + threadgroup_thread_count - 1) /
        threadgroup_thread_count;
    vec<InT, 4> prefetched_keys[key_prefetch_slots];
    vec<InT, 4> prefetched_queries[key_prefetch_slots];
    vec<InT, 4> prefetched_values[value_prefetch_slots];
    float prefetched_decay = 0.0f;
    float prefetched_update_rate = 0.0f;
#define ASTRONOMICAL_GDN_PIPE_FETCH(first_fetch_token) { \
        const int prefetch_token_count = \
            min(time_block_size, token_count - (first_fetch_token)); \
        for (int prefetch_slot = 0; prefetch_slot < key_prefetch_slots; \
             ++prefetch_slot) { \
            const int prefetch_position = \
                thread_index + prefetch_slot * threadgroup_thread_count; \
            if (prefetch_position < prefetch_token_count * (Dk / 4)) { \
                const int prefetch_token = prefetch_position / (Dk / 4); \
                const int prefetch_granule = prefetch_position % (Dk / 4); \
                const size_t key_offset = \
                    (size_t)((first_fetch_token) + prefetch_token) * \
                    key_row_stride + 4 * prefetch_granule; \
                prefetched_keys[prefetch_slot] = \
                    *(const device vec<InT, 4>*)(key_base + key_offset); \
                prefetched_queries[prefetch_slot] = \
                    *(const device vec<InT, 4>*)(query_base + key_offset); \
            } \
        } \
        for (int prefetch_slot = 0; prefetch_slot < value_prefetch_slots; \
             ++prefetch_slot) { \
            const int prefetch_position = \
                thread_index + prefetch_slot * threadgroup_thread_count; \
            if (prefetch_position < \
                prefetch_token_count * (value_row_block_size / 4)) { \
                const int prefetch_token = \
                    prefetch_position / (value_row_block_size / 4); \
                const int prefetch_granule = \
                    prefetch_position % (value_row_block_size / 4); \
                prefetched_values[prefetch_slot] = \
                    *(const device vec<InT, 4>*)(value_base + \
                        (size_t)((first_fetch_token) + prefetch_token) * \
                        value_row_stride + 4 * prefetch_granule); \
            } \
        } \
        if (thread_index < prefetch_token_count) { \
            prefetched_decay = decays[((size_t)batch_index * token_count + \
                (first_fetch_token) + thread_index) * Hv + value_head_index]; \
            prefetched_update_rate = update_rates[ \
                ((size_t)batch_index * token_count + (first_fetch_token) + \
                 thread_index) * Hv + value_head_index]; \
        } \
    }
    ASTRONOMICAL_GDN_PIPE_FETCH(0)

    float4 next_keys[4];
    float next_decay = 0.0f;
    float next_update_rate = 0.0f;
    float next_value = 0.0f;
    float previous_step_query_partial = 0.0f;
    float step_query_partials[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int first_token = 0; first_token < token_count;
         first_token += time_block_size) {
        auto tokens_in_block = min(time_block_size, token_count - first_token);
        for (int stage_slot = 0; stage_slot < key_prefetch_slots; ++stage_slot) {
            const int stage_position =
                thread_index + stage_slot * threadgroup_thread_count;
            if (stage_position < tokens_in_block * (Dk / 4)) {
                const int stage_token = stage_position / (Dk / 4);
                const int stage_granule = stage_position % (Dk / 4);
                *(threadgroup vec<InT, 4>*)(
                    &staged_keys[stage_token][4 * stage_granule]) =
                    prefetched_keys[stage_slot];
                *(threadgroup vec<InT, 4>*)(
                    &staged_queries[stage_token][4 * stage_granule]) =
                    prefetched_queries[stage_slot];
            }
        }
        for (int stage_slot = 0; stage_slot < value_prefetch_slots; ++stage_slot) {
            const int stage_position =
                thread_index + stage_slot * threadgroup_thread_count;
            if (stage_position < tokens_in_block * (value_row_block_size / 4)) {
                const int stage_token =
                    stage_position / (value_row_block_size / 4);
                const int stage_granule =
                    stage_position % (value_row_block_size / 4);
                *(threadgroup float4*)(
                    &staged_values[stage_token][4 * stage_granule]) =
                    float4(prefetched_values[stage_slot]);
            }
        }
        if (thread_index < tokens_in_block) {
            staged_decays[thread_index] = prefetched_decay;
            staged_update_rates[thread_index] = prefetched_update_rate;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (first_token + time_block_size < token_count) {
            ASTRONOMICAL_GDN_PIPE_FETCH(first_token + time_block_size)
        }
        for (int fragment_index = 0; fragment_index < 4; ++fragment_index) {
            next_keys[fragment_index] = float4(
                *(const threadgroup vec<InT, 4>*)(&staged_keys[0]
                    [4 * (key_segment_index + 8 * fragment_index)]));
        }
        next_decay = staged_decays[0];
        next_update_rate = staged_update_rates[0];
        next_value = staged_values[0][value_row_in_block];
        if (tokens_in_block == time_block_size) {
            {
                float4 current_keys[4];
                for (int fragment_index = 0; fragment_index < 4;
                     ++fragment_index) {
                    current_keys[fragment_index] = next_keys[fragment_index];
                }
                const float current_decay = next_decay;
                const float current_update_rate = next_update_rate;
                const float current_value = next_value;
                for (int fragment_index = 0; fragment_index < 4; ++fragment_index) {
                    next_keys[fragment_index] = float4(
                        *(const threadgroup vec<InT, 4>*)(
                            &staged_keys[1]
                                [4 * (key_segment_index + 8 * fragment_index)]));
                }
                next_decay = staged_decays[1];
                next_update_rate = staged_update_rates[1];
                next_value = staged_values[1][value_row_in_block];
                float2 state_key_dot_partials = 0.0f;
                for (int fragment_index = 0; fragment_index < 4;
                     ++fragment_index) {
                    state_fragment[fragment_index] *= current_decay;
                    state_key_dot_partials = fma(
                        state_fragment[fragment_index].xy,
                        current_keys[fragment_index].xy,
                        state_key_dot_partials);
                    state_key_dot_partials = fma(
                        state_fragment[fragment_index].zw,
                        current_keys[fragment_index].zw,
                        state_key_dot_partials);
                }
                float remembered_value = state_key_dot_partials.x +
                    state_key_dot_partials.y;
                remembered_value += simd_shuffle_xor(remembered_value, 4);
                remembered_value += simd_shuffle_xor(remembered_value, 2);
                remembered_value += simd_shuffle_xor(remembered_value, 1);
                float4 current_queries[4];
                for (int fragment_index = 0; fragment_index < 4;
                     ++fragment_index) {
                    current_queries[fragment_index] = float4(
                        *(const threadgroup vec<InT, 4>*)(
                            &staged_queries[0]
                                [4 * (key_segment_index + 8 * fragment_index)]));
                }
                const float delta = (current_value - remembered_value) *
                    current_update_rate;
                float2 state_query_dot_partials = 0.0f;
                for (int fragment_index = 0; fragment_index < 4;
                     ++fragment_index) {
                    state_fragment[fragment_index] = fma(
                        current_keys[fragment_index], float4(delta),
                        state_fragment[fragment_index]);
                    state_query_dot_partials = fma(
                        state_fragment[fragment_index].xy,
                        current_queries[fragment_index].xy,
                        state_query_dot_partials);
                    state_query_dot_partials = fma(
                        state_fragment[fragment_index].zw,
                        current_queries[fragment_index].zw,
                        state_query_dot_partials);
                }

                const int completed_token_count = first_token +
                    0 + 1;
                if (checkpoint_index < checkpoint_count &&
                    completed_token_count == next_checkpoint_token_count) {
                    auto checkpoint_state = (device float4*)(
                        recurrent_boundary_states +
                        (((((size_t)checkpoint_index * B + batch_index) * Hv +
                           value_head_index) * Dv + first_value_row +
                           value_row_in_block) * Dk));
                    for (int fragment_index = 0; fragment_index < 4;
                         ++fragment_index) {
                        checkpoint_state[key_segment_index +
                            8 * fragment_index] =
                            state_fragment[fragment_index];
                    }
                    ++checkpoint_index;
                    next_checkpoint_token_count +=
                        checkpoint_interval_token_count;
                }
                step_query_partials[0] = state_query_dot_partials.x +
                    state_query_dot_partials.y;
            }
            {
                float4 current_keys[4];
                for (int fragment_index = 0; fragment_index < 4;
                     ++fragment_index) {
                    current_keys[fragment_index] = next_keys[fragment_index];
                }
                const float current_decay = next_decay;
                const float current_update_rate = next_update_rate;
                const float current_value = next_value;
                for (int fragment_index = 0; fragment_index < 4; ++fragment_index) {
                    next_keys[fragment_index] = float4(
                        *(const threadgroup vec<InT, 4>*)(
                            &staged_keys[2]
                                [4 * (key_segment_index + 8 * fragment_index)]));
                }
                next_decay = staged_decays[2];
                next_update_rate = staged_update_rates[2];
                next_value = staged_values[2][value_row_in_block];
                float2 state_key_dot_partials = 0.0f;
                for (int fragment_index = 0; fragment_index < 4;
                     ++fragment_index) {
                    state_fragment[fragment_index] *= current_decay;
                    state_key_dot_partials = fma(
                        state_fragment[fragment_index].xy,
                        current_keys[fragment_index].xy,
                        state_key_dot_partials);
                    state_key_dot_partials = fma(
                        state_fragment[fragment_index].zw,
                        current_keys[fragment_index].zw,
                        state_key_dot_partials);
                }
                float remembered_value = state_key_dot_partials.x +
                    state_key_dot_partials.y;
                remembered_value += simd_shuffle_xor(remembered_value, 4);
                remembered_value += simd_shuffle_xor(remembered_value, 2);
                remembered_value += simd_shuffle_xor(remembered_value, 1);
                float4 current_queries[4];
                for (int fragment_index = 0; fragment_index < 4;
                     ++fragment_index) {
                    current_queries[fragment_index] = float4(
                        *(const threadgroup vec<InT, 4>*)(
                            &staged_queries[1]
                                [4 * (key_segment_index + 8 * fragment_index)]));
                }
                const float delta = (current_value - remembered_value) *
                    current_update_rate;
                float2 state_query_dot_partials = 0.0f;
                for (int fragment_index = 0; fragment_index < 4;
                     ++fragment_index) {
                    state_fragment[fragment_index] = fma(
                        current_keys[fragment_index], float4(delta),
                        state_fragment[fragment_index]);
                    state_query_dot_partials = fma(
                        state_fragment[fragment_index].xy,
                        current_queries[fragment_index].xy,
                        state_query_dot_partials);
                    state_query_dot_partials = fma(
                        state_fragment[fragment_index].zw,
                        current_queries[fragment_index].zw,
                        state_query_dot_partials);
                }

                const int completed_token_count = first_token +
                    1 + 1;
                if (checkpoint_index < checkpoint_count &&
                    completed_token_count == next_checkpoint_token_count) {
                    auto checkpoint_state = (device float4*)(
                        recurrent_boundary_states +
                        (((((size_t)checkpoint_index * B + batch_index) * Hv +
                           value_head_index) * Dv + first_value_row +
                           value_row_in_block) * Dk));
                    for (int fragment_index = 0; fragment_index < 4;
                         ++fragment_index) {
                        checkpoint_state[key_segment_index +
                            8 * fragment_index] =
                            state_fragment[fragment_index];
                    }
                    ++checkpoint_index;
                    next_checkpoint_token_count +=
                        checkpoint_interval_token_count;
                }
                step_query_partials[1] = state_query_dot_partials.x +
                    state_query_dot_partials.y;
            }
            {
                float4 current_keys[4];
                for (int fragment_index = 0; fragment_index < 4;
                     ++fragment_index) {
                    current_keys[fragment_index] = next_keys[fragment_index];
                }
                const float current_decay = next_decay;
                const float current_update_rate = next_update_rate;
                const float current_value = next_value;
                for (int fragment_index = 0; fragment_index < 4; ++fragment_index) {
                    next_keys[fragment_index] = float4(
                        *(const threadgroup vec<InT, 4>*)(
                            &staged_keys[3]
                                [4 * (key_segment_index + 8 * fragment_index)]));
                }
                next_decay = staged_decays[3];
                next_update_rate = staged_update_rates[3];
                next_value = staged_values[3][value_row_in_block];
                float2 state_key_dot_partials = 0.0f;
                for (int fragment_index = 0; fragment_index < 4;
                     ++fragment_index) {
                    state_fragment[fragment_index] *= current_decay;
                    state_key_dot_partials = fma(
                        state_fragment[fragment_index].xy,
                        current_keys[fragment_index].xy,
                        state_key_dot_partials);
                    state_key_dot_partials = fma(
                        state_fragment[fragment_index].zw,
                        current_keys[fragment_index].zw,
                        state_key_dot_partials);
                }
                float remembered_value = state_key_dot_partials.x +
                    state_key_dot_partials.y;
                remembered_value += simd_shuffle_xor(remembered_value, 4);
                remembered_value += simd_shuffle_xor(remembered_value, 2);
                remembered_value += simd_shuffle_xor(remembered_value, 1);
                float4 current_queries[4];
                for (int fragment_index = 0; fragment_index < 4;
                     ++fragment_index) {
                    current_queries[fragment_index] = float4(
                        *(const threadgroup vec<InT, 4>*)(
                            &staged_queries[2]
                                [4 * (key_segment_index + 8 * fragment_index)]));
                }
                const float delta = (current_value - remembered_value) *
                    current_update_rate;
                float2 state_query_dot_partials = 0.0f;
                for (int fragment_index = 0; fragment_index < 4;
                     ++fragment_index) {
                    state_fragment[fragment_index] = fma(
                        current_keys[fragment_index], float4(delta),
                        state_fragment[fragment_index]);
                    state_query_dot_partials = fma(
                        state_fragment[fragment_index].xy,
                        current_queries[fragment_index].xy,
                        state_query_dot_partials);
                    state_query_dot_partials = fma(
                        state_fragment[fragment_index].zw,
                        current_queries[fragment_index].zw,
                        state_query_dot_partials);
                }

                const int completed_token_count = first_token +
                    2 + 1;
                if (checkpoint_index < checkpoint_count &&
                    completed_token_count == next_checkpoint_token_count) {
                    auto checkpoint_state = (device float4*)(
                        recurrent_boundary_states +
                        (((((size_t)checkpoint_index * B + batch_index) * Hv +
                           value_head_index) * Dv + first_value_row +
                           value_row_in_block) * Dk));
                    for (int fragment_index = 0; fragment_index < 4;
                         ++fragment_index) {
                        checkpoint_state[key_segment_index +
                            8 * fragment_index] =
                            state_fragment[fragment_index];
                    }
                    ++checkpoint_index;
                    next_checkpoint_token_count +=
                        checkpoint_interval_token_count;
                }
                step_query_partials[2] = state_query_dot_partials.x +
                    state_query_dot_partials.y;
            }
