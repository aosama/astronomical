// The checkpoint kernel's unrolled steps from block-local token 8 on, the tail and the epilogue, split from its file only to keep each file readable.
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
                            &staged_keys[9]
                                [4 * (key_segment_index + 8 * fragment_index)]));
                }
                next_decay = staged_decays[9];
                next_update_rate = staged_update_rates[9];
                next_value = staged_values[9][value_row_in_block];
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
            {
                const bool high_half_lane = (key_segment_index & 4) != 0;
                const bool high_pair_lane = (key_segment_index & 2) != 0;
                float first_reduced_partial = high_half_lane
                    ? step_query_partials[2]
                    : step_query_partials[0];
                float second_reduced_partial = high_half_lane
                    ? step_query_partials[3]
                    : step_query_partials[1];
                first_reduced_partial += simd_shuffle_xor(
                    high_half_lane ? step_query_partials[0]
                                   : step_query_partials[2], 4);
                second_reduced_partial += simd_shuffle_xor(
                    high_half_lane ? step_query_partials[1]
                                   : step_query_partials[3], 4);
                float reduced_query_partial = high_pair_lane
                    ? second_reduced_partial
                    : first_reduced_partial;
                reduced_query_partial += simd_shuffle_xor(
                    high_pair_lane ? first_reduced_partial
                                   : second_reduced_partial, 2);
                reduced_query_partial +=
                    simd_shuffle_xor(reduced_query_partial, 1);
                if ((key_segment_index & 1) == 0) {
                    staged_outputs[4 +
                        (high_half_lane ? 2 : 0) + (high_pair_lane ? 1 : 0)]
                        [value_row_in_block] = (InT)reduced_query_partial;
                }
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
                            &staged_queries[8]
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
                    8 + 1;
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
                            &staged_keys[10]
                                [4 * (key_segment_index + 8 * fragment_index)]));
                }
                next_decay = staged_decays[10];
                next_update_rate = staged_update_rates[10];
                next_value = staged_values[10][value_row_in_block];
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
                            &staged_queries[9]
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
                    9 + 1;
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
                            &staged_keys[11]
                                [4 * (key_segment_index + 8 * fragment_index)]));
                }
                next_decay = staged_decays[11];
                next_update_rate = staged_update_rates[11];
                next_value = staged_values[11][value_row_in_block];
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
                            &staged_queries[10]
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
                    10 + 1;
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
            {
                float4 current_keys[4];
                for (int fragment_index = 0; fragment_index < 4;
                     ++fragment_index) {
                    current_keys[fragment_index] = next_keys[fragment_index];
                }
                const float current_decay = next_decay;
                const float current_update_rate = next_update_rate;
                const float current_value = next_value;
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
                            &staged_queries[11]
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
                    11 + 1;
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
                step_query_partials[3] = state_query_dot_partials.x +
                    state_query_dot_partials.y;
            }
            {
                const bool high_half_lane = (key_segment_index & 4) != 0;
                const bool high_pair_lane = (key_segment_index & 2) != 0;
                float first_reduced_partial = high_half_lane
                    ? step_query_partials[2]
                    : step_query_partials[0];
                float second_reduced_partial = high_half_lane
                    ? step_query_partials[3]
                    : step_query_partials[1];
                first_reduced_partial += simd_shuffle_xor(
                    high_half_lane ? step_query_partials[0]
                                   : step_query_partials[2], 4);
                second_reduced_partial += simd_shuffle_xor(
                    high_half_lane ? step_query_partials[1]
                                   : step_query_partials[3], 4);
                float reduced_query_partial = high_pair_lane
                    ? second_reduced_partial
                    : first_reduced_partial;
                reduced_query_partial += simd_shuffle_xor(
                    high_pair_lane ? first_reduced_partial
                                   : second_reduced_partial, 2);
                reduced_query_partial +=
                    simd_shuffle_xor(reduced_query_partial, 1);
                if ((key_segment_index & 1) == 0) {
                    staged_outputs[8 +
                        (high_half_lane ? 2 : 0) + (high_pair_lane ? 1 : 0)]
                        [value_row_in_block] = (InT)reduced_query_partial;
                }
            }

        } else {
            for (int token_in_block = 0; token_in_block < tokens_in_block;
                 ++token_in_block) {
            {
                float4 current_keys[4];
                for (int fragment_index = 0; fragment_index < 4;
                     ++fragment_index) {
                    current_keys[fragment_index] = next_keys[fragment_index];
                }
                const float current_decay = next_decay;
                const float current_update_rate = next_update_rate;
                const float current_value = next_value;
                if (token_in_block + 1 < tokens_in_block) {
                    for (int fragment_index = 0; fragment_index < 4;
                         ++fragment_index) {
                        next_keys[fragment_index] = float4(
                            *(const threadgroup vec<InT, 4>*)(
                                &staged_keys[token_in_block + 1]
                                    [4 * (key_segment_index +
                                        8 * fragment_index)]));
                    }
                    next_decay = staged_decays[token_in_block + 1];
                    next_update_rate = staged_update_rates[token_in_block + 1];
                    next_value =
                        staged_values[token_in_block + 1][value_row_in_block];
                }
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
                if (token_in_block > 0) {
                    float reduced_query_partial = previous_step_query_partial;
                    reduced_query_partial +=
                        simd_shuffle_down(reduced_query_partial, 4);
                    reduced_query_partial +=
                        simd_shuffle_down(reduced_query_partial, 2);
                    reduced_query_partial +=
                        simd_shuffle_down(reduced_query_partial, 1);
                    if (key_segment_index == 0) {
                        staged_outputs[token_in_block - 1]
                            [value_row_in_block] =
                            (InT)reduced_query_partial;
                    }
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
                            &staged_queries[token_in_block]
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
                    token_in_block + 1;
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
                previous_step_query_partial = state_query_dot_partials.x +
                    state_query_dot_partials.y;
            }

            }
            {
                float reduced_query_partial = previous_step_query_partial;
                reduced_query_partial +=
                    simd_shuffle_down(reduced_query_partial, 4);
                reduced_query_partial +=
                    simd_shuffle_down(reduced_query_partial, 2);
                reduced_query_partial +=
                    simd_shuffle_down(reduced_query_partial, 1);
                if (key_segment_index == 0) {
                    staged_outputs[tokens_in_block - 1][value_row_in_block] =
                        (InT)reduced_query_partial;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int output_position = thread_index;
             output_position < tokens_in_block * (value_row_block_size / 4);
             output_position += threadgroup_thread_count) {
            const int output_token =
                output_position / (value_row_block_size / 4);
            const int output_granule =
                output_position % (value_row_block_size / 4);
            *(device vec<InT, 4>*)(output_base +
                (size_t)(first_token + output_token) * value_row_stride +
                4 * output_granule) =
                *(threadgroup vec<InT, 4>*)(
                    &staged_outputs[output_token][4 * output_granule]);
        }
    }
#undef ASTRONOMICAL_GDN_PIPE_FETCH

    {
        auto output_state = (device float4*)(next_recurrent_state +
            (((size_t)batch_index * Hv + value_head_index) * Dv +
             first_value_row + value_row_in_block) * Dk);
        for (int fragment_index = 0; fragment_index < 4; ++fragment_index) {
            output_state[key_segment_index + 8 * fragment_index] =
                state_fragment[fragment_index];
        }
    }
