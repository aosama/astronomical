// The recurrent state after the kept rows of a verified draft tree, from the
// round's starting state: the starting state advanced over the kept rows in
// order, by the same step as the tree kernel, so it equals the sequence
// kernel's final state over those rows without gathering them first (five
// gathers a layer, 240 a round on the 27B).
//
// Inputs: keys, values, decays, update_rates and recurrent_state as the tree
// kernel takes them, plus kept_rows [n] int32. Output next_recurrent_state.
// The thread layout opens the kernel from the shared file. The row step, from
// the key load to the reduction of remembered_value, is the same text in the
// tree kernel, and a unit test compares the two.
//
// @prologue astro_family_qwen35_gated_delta_thread_layout_body.metal
// @input keys InT
// @input values InT
// @input decays bfloat16_t|float16_t|float may-be-small
// @input update_rates bfloat16_t|float16_t|float may-be-small
// @input recurrent_state float
// @input kept_rows int32_t may-be-small
// @output next_recurrent_state float
// @template InT typename
// @template StT typename
// @template Dk int
// @template Dv int
// @template Hk int
// @template Hv int
// @compile-with InT=bfloat16_t StT=float Dk=128 Dv=128 Hk=16 Hv=48
// @compile-with InT=float16_t StT=float Dk=128 Dv=16 Hk=1 Hv=2
// @contract crates/families/text/qwen35/src/qwen3_5_core/model_math/gated_delta_tree_tests.rs
// @probe none, declared only when draft tree verification is enabled, so no worker-start probe runs it; its test compares it with the sequence kernel's final state bit for bit

    const int kept_row_count = kept_rows_shape[0];
    const size_t state_offset = ((size_t)value_head_index * Dv + value_row) * Dk;

    float4 state_fragment[4];
    {
        auto input_state = (const device float4*)(recurrent_state + state_offset);
        for (int fragment_index = 0; fragment_index < 4; ++fragment_index) {
            state_fragment[fragment_index] =
                input_state[key_segment_index + 8 * fragment_index];
        }
    }
    for (int kept_index = 0; kept_index < kept_row_count; ++kept_index) {
        const int row = kept_rows[kept_index];
        auto key_row = keys + ((size_t)row * Hk + key_head_index) * Dk;

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

        const float delta = (current_value - remembered_value) * current_update_rate;
        for (int fragment_index = 0; fragment_index < 4; ++fragment_index) {
            state_fragment[fragment_index] = fma(current_keys[fragment_index],
                float4(delta), state_fragment[fragment_index]);
        }
    }
    {
        auto output_state = (device float4*)(next_recurrent_state + state_offset);
        for (int fragment_index = 0; fragment_index < 4; ++fragment_index) {
            output_state[key_segment_index + 8 * fragment_index] =
                state_fragment[fragment_index];
        }
    }
