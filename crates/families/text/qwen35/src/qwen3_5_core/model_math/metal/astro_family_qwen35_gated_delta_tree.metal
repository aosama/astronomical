// The gated-delta recurrence over the rows of a draft tree: every row starts
// from its parent's state, not the previous row's.
//
// TensorFold's tree kernel (`lane_tree.py`, release 0.3.5.1, Apache-2.0)
// keeps every row's state in registers, which fits its 32 lanes per value
// row. The sequence kernel (astro_family_qwen35_gated_delta_sequence) spreads
// a row over 8 lanes, so 16 rows of state would be 256 floats a lane and
// spill. Instead each row replays its own ancestor chain from the round's
// starting state: at most a block's depth (8 steps) of register arithmetic,
// while the starting state, the only large read, is loaded once a thread. Each
// step is the sequence kernel's step, operation for operation, so a row's
// output is bit-identical to feeding its chain alone.
//
// Inputs as the sequence kernel takes them, [1, rows, ...], plus row_chains
// [rows, MAXIMUM_CHAIN] (each row's chain from the anchor to itself) and
// chain_lengths [rows]. Output outputs [1, rows, Hv, Dv]. The thread layout
// opens the kernel from the shared file. The row step, from the key load to
// the reduction of remembered_value, is the same text in the replay kernel,
// and a unit test compares the two.
//
// @prologue astro_family_qwen35_gated_delta_thread_layout_body.metal
// @input queries InT
// @input keys InT
// @input values InT
// @input decays bfloat16_t|float16_t|float may-be-small
// @input update_rates bfloat16_t|float16_t|float may-be-small
// @input recurrent_state float
// @input row_chains int32_t may-be-small
// @input chain_lengths int32_t may-be-small
// @output outputs InT
// @template InT typename
// @template StT typename
// @template Dk int
// @template Dv int
// @template Hk int
// @template Hv int
// @template MAXIMUM_CHAIN int
// @compile-with InT=bfloat16_t StT=float Dk=128 Dv=128 Hk=16 Hv=48 MAXIMUM_CHAIN=8
// @compile-with InT=float16_t StT=float Dk=128 Dv=16 Hk=1 Hv=2 MAXIMUM_CHAIN=1
// @contract crates/families/text/qwen35/src/qwen3_5_core/model_math/gated_delta_tree_tests.rs
// @probe none, declared only when draft tree verification is enabled, so no worker-start probe runs it; its tests compare it with the sequence kernel bit for bit

    const int tree_row_count = chain_lengths_shape[0];

    float4 starting_fragment[4];
    {
        auto input_state = (const device float4*)(recurrent_state +
            ((size_t)value_head_index * Dv + value_row) * Dk);
        for (int fragment_index = 0; fragment_index < 4; ++fragment_index) {
            starting_fragment[fragment_index] =
                input_state[key_segment_index + 8 * fragment_index];
        }
    }
    // The state after the previous row: a row that continues it (a chain,
    // such as a copied run) takes one step from there instead of replaying.
    float4 carried_fragment[4];
    for (int tree_row = 0; tree_row < tree_row_count; ++tree_row) {
        const int chain_length = chain_lengths[tree_row];
        const bool continues_previous_row = tree_row > 0 && chain_length > 1 &&
            row_chains[tree_row * MAXIMUM_CHAIN + chain_length - 2] == tree_row - 1;
        float4 state_fragment[4];
        for (int fragment_index = 0; fragment_index < 4; ++fragment_index) {
            state_fragment[fragment_index] = continues_previous_row
                ? carried_fragment[fragment_index] : starting_fragment[fragment_index];
        }
        float reduced_query_partial = 0.0f;
        for (int chain_step = continues_previous_row ? chain_length - 1 : 0;
             chain_step < chain_length; ++chain_step) {
            const int row = row_chains[tree_row * MAXIMUM_CHAIN + chain_step];
            auto key_row = keys + ((size_t)row * Hk + key_head_index) * Dk;
            auto query_row = queries + ((size_t)row * Hk + key_head_index) * Dk;

        float4 current_keys[4];
        for (int fragment_index = 0; fragment_index < 4; ++fragment_index) {
            current_keys[fragment_index] = float4(*(const device vec<InT, 4>*)(
                key_row + 4 * (key_segment_index + 8 * fragment_index)));
        }
        const float current_decay = decays[(size_t)row * Hv + value_head_index];
        const float current_update_rate = update_rates[(size_t)row * Hv + value_head_index];
        const float current_value =
            float(values[((size_t)row * Hv + value_head_index) * Dv + value_row]);
        float2 state_key_dot_partials = 0.0f;
        for (int fragment_index = 0; fragment_index < 4; ++fragment_index) {
            state_fragment[fragment_index] *= current_decay;
            state_key_dot_partials = fma(state_fragment[fragment_index].xy,
                current_keys[fragment_index].xy, state_key_dot_partials);
            state_key_dot_partials = fma(state_fragment[fragment_index].zw,
                current_keys[fragment_index].zw, state_key_dot_partials);
        }
        float remembered_value = state_key_dot_partials.x + state_key_dot_partials.y;
        remembered_value += simd_shuffle_xor(remembered_value, 4);
        remembered_value += simd_shuffle_xor(remembered_value, 2);
        remembered_value += simd_shuffle_xor(remembered_value, 1);

            float4 current_queries[4];
            for (int fragment_index = 0; fragment_index < 4; ++fragment_index) {
                current_queries[fragment_index] = float4(*(const device vec<InT, 4>*)(
                    query_row + 4 * (key_segment_index + 8 * fragment_index)));
            }
            const float delta = (current_value - remembered_value) * current_update_rate;
            float2 state_query_dot_partials = 0.0f;
            for (int fragment_index = 0; fragment_index < 4; ++fragment_index) {
                state_fragment[fragment_index] = fma(current_keys[fragment_index],
                    float4(delta), state_fragment[fragment_index]);
                state_query_dot_partials = fma(state_fragment[fragment_index].xy,
                    current_queries[fragment_index].xy, state_query_dot_partials);
                state_query_dot_partials = fma(state_fragment[fragment_index].zw,
                    current_queries[fragment_index].zw, state_query_dot_partials);
            }
            reduced_query_partial = state_query_dot_partials.x + state_query_dot_partials.y;
        }
        // The sequence kernel's reduction of a step's query partials across
        // the row's 8 lanes, in the same order.
        reduced_query_partial += simd_shuffle_down(reduced_query_partial, 4);
        reduced_query_partial += simd_shuffle_down(reduced_query_partial, 2);
        reduced_query_partial += simd_shuffle_down(reduced_query_partial, 1);
        if (key_segment_index == 0) {
            outputs[((size_t)tree_row * Hv + value_head_index) * Dv + value_row] =
                (InT)reduced_query_partial;
        }
        for (int fragment_index = 0; fragment_index < 4; ++fragment_index) {
            carried_fragment[fragment_index] = state_fragment[fragment_index];
        }
    }
