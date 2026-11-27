// The checkpoint kernel's unrolled steps for block-local tokens 3 to 7, split from its file only to keep each file readable.
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
                            &staged_keys[4]
                                [4 * (key_segment_index + 8 * fragment_index)]));
                }
                next_decay = staged_decays[4];
                next_update_rate = staged_update_rates[4];
                next_value = staged_values[4][value_row_in_block];
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
                            &staged_queries[3]
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
                    3 + 1;
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
                            &staged_keys[5]
                                [4 * (key_segment_index + 8 * fragment_index)]));
                }
                next_decay = staged_decays[5];
                next_update_rate = staged_update_rates[5];
                next_value = staged_values[5][value_row_in_block];
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
                    staged_outputs[0 +
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
                            &staged_queries[4]
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
                    4 + 1;
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
                            &staged_keys[6]
                                [4 * (key_segment_index + 8 * fragment_index)]));
                }
                next_decay = staged_decays[6];
                next_update_rate = staged_update_rates[6];
                next_value = staged_values[6][value_row_in_block];
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
                            &staged_queries[5]
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
                    5 + 1;
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
                            &staged_keys[7]
                                [4 * (key_segment_index + 8 * fragment_index)]));
                }
                next_decay = staged_decays[7];
                next_update_rate = staged_update_rates[7];
                next_value = staged_values[7][value_row_in_block];
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
                            &staged_queries[6]
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
                    6 + 1;
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
                for (int fragment_index = 0; fragment_index < 4; ++fragment_index) {
                    next_keys[fragment_index] = float4(
                        *(const threadgroup vec<InT, 4>*)(
                            &staged_keys[8]
                                [4 * (key_segment_index + 8 * fragment_index)]));
                }
                next_decay = staged_decays[8];
                next_update_rate = staged_update_rates[8];
                next_value = staged_values[8][value_row_in_block];
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
                            &staged_queries[7]
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
                    7 + 1;
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
