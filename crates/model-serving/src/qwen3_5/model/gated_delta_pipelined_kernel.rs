//! Metal source generator for the software-pipelined Qwen3.5 gated-delta
//! prefill recurrence.
//!
//! The kernel stages twelve tokens per block in threadgroup memory and
//! prefetches the next block's keys, values, decays, and update rates into
//! registers while the current block computes, hiding global-memory latency
//! behind the recurrence arithmetic. The algorithm mirrors OMLX's production
//! `gated_delta_pipelined` kernel; naming follows this repository and the
//! boundary-checkpoint write is injected at a fixed point inside every step
//! body.

/// Placeholder inside the checkpoint write template for the block-local
/// token expression of each injection site: the literal step index in
/// unrolled full blocks, the rolled loop variable in the tail block.
pub(super) const CHECKPOINT_TOKEN_EXPRESSION_MARKER: &str = "/* ASTRONOMICAL_CHECKPOINT_TOKEN */";

/// Tokens processed by one kernel block before rolling to the next block.
pub const TIME_BLOCK_SIZE: usize = 12;
/// Value rows owned by one kernel block.
pub const VALUE_ROW_BLOCK_SIZE: usize = 16;
/// Eight lanes cooperate on each value row.
pub const THREADGROUP_THREAD_COUNT: usize = VALUE_ROW_BLOCK_SIZE * 8;

const KERNEL_PROLOGUE: &str = r#"
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
"#;

const KERNEL_PREFETCH_SECTION: &str = r#"
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
"#;

const KERNEL_FULL_TO_TAIL_SPLIT: &str = r#"
        } else {
            for (int token_in_block = 0; token_in_block < tokens_in_block;
                 ++token_in_block) {
"#;

const KERNEL_OUTPUT_STORE_AND_EPILOGUE: &str = r#"
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
"#;

/// How one pipelined step obtains its next-step operands and publishes its
/// query-dot partial.
enum StepPipelining {
    /// Unrolled full-block step: literal token index, unconditional refill
    /// from the next staged token when one exists, partial published into
    /// the four-slot scatter array.
    Unrolled {
        token_index: usize,
        next_staged_token: Option<usize>,
    },
    /// Rolled tail-block step: loop-variable token index, guarded refill,
    /// partial published into the single carry slot.
    Rolled,
}

/// Which earlier steps' query-dot partials a step reduces onto
/// `staged_outputs`.
enum PreviousStepReduction {
    /// Reduce-scatter the four steps `first_reduced_token..+4` from the
    /// scatter array; bit-identical to reducing each step on its own.
    ScatterFourSteps { first_reduced_token: usize },
    /// Reduce the previous step's partial carried in
    /// `previous_step_query_partial`; skipped at step zero.
    CarryPreviousStep,
    /// No earlier partials are ready at this step.
    NoPreviousSteps,
}

/// Builds the fused gated-delta kernel source with the supplied boundary
/// checkpoint injection sources. The ordinary kernel passes empty strings.
pub(super) fn gated_delta_pipelined_kernel_source(
    checkpoint_setup_source: &str,
    checkpoint_write_source_template: &str,
) -> String {
    let mut source = String::from(KERNEL_PROLOGUE);
    source.push_str(checkpoint_setup_source);
    source.push_str(KERNEL_PREFETCH_SECTION);
    for token_index in 0..TIME_BLOCK_SIZE {
        source.push_str(&pipelined_step_source(
            StepPipelining::Unrolled {
                token_index,
                next_staged_token: (token_index + 1 < TIME_BLOCK_SIZE).then_some(token_index + 1),
            },
            if token_index > 0 && token_index % 4 == 0 {
                PreviousStepReduction::ScatterFourSteps {
                    first_reduced_token: token_index - 4,
                }
            } else {
                PreviousStepReduction::NoPreviousSteps
            },
            checkpoint_write_source_template,
        ));
    }
    source.push_str(&reduce_scatter_four_steps_source(TIME_BLOCK_SIZE - 4));
    source.push_str(KERNEL_FULL_TO_TAIL_SPLIT);
    source.push_str(&pipelined_step_source(
        StepPipelining::Rolled,
        PreviousStepReduction::CarryPreviousStep,
        checkpoint_write_source_template,
    ));
    source.push_str(KERNEL_OUTPUT_STORE_AND_EPILOGUE);
    source
}

fn checkpoint_write_source_for_token(
    checkpoint_write_source_template: &str,
    token_expression: &str,
) -> String {
    checkpoint_write_source_template.replace(CHECKPOINT_TOKEN_EXPRESSION_MARKER, token_expression)
}

fn pipelined_step_source(
    step_pipelining: StepPipelining,
    previous_step_reduction: PreviousStepReduction,
    checkpoint_write_source_template: &str,
) -> String {
    let (token_expression, next_refill_source, partial_publication_source) = match step_pipelining
    {
        StepPipelining::Unrolled {
            token_index,
            next_staged_token,
        } => {
            let token_expression = token_index.to_string();
            let next_refill_source = match next_staged_token {
                Some(next_staged_token) => format!(
                    r#"                for (int fragment_index = 0; fragment_index < 4; ++fragment_index) {{
                    next_keys[fragment_index] = float4(
                        *(const threadgroup vec<InT, 4>*)(
                            &staged_keys[{next_staged_token}]
                                [4 * (key_segment_index + 8 * fragment_index)]));
                }}
                next_decay = staged_decays[{next_staged_token}];
                next_update_rate = staged_update_rates[{next_staged_token}];
                next_value = staged_values[{next_staged_token}][value_row_in_block];
"#
                ),
                None => String::new(),
            };
            let partial_publication_source = format!(
                "                step_query_partials[{}] = state_query_dot_partials.x +\n                    state_query_dot_partials.y;\n",
                token_index % 4
            );
            (
                token_expression,
                next_refill_source,
                partial_publication_source,
            )
        }
        StepPipelining::Rolled => (
            "token_in_block".to_owned(),
            r#"                if (token_in_block + 1 < tokens_in_block) {
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
"#
            .to_owned(),
            "                previous_step_query_partial = state_query_dot_partials.x +\n                    state_query_dot_partials.y;\n"
                .to_owned(),
        ),
    };
    let reduction_source = match &previous_step_reduction {
        PreviousStepReduction::ScatterFourSteps {
            first_reduced_token,
        } => reduce_scatter_four_steps_source(*first_reduced_token),
        PreviousStepReduction::CarryPreviousStep => format!(
            r#"                if ({token_expression} > 0) {{
                    float reduced_query_partial = previous_step_query_partial;
                    reduced_query_partial +=
                        simd_shuffle_down(reduced_query_partial, 4);
                    reduced_query_partial +=
                        simd_shuffle_down(reduced_query_partial, 2);
                    reduced_query_partial +=
                        simd_shuffle_down(reduced_query_partial, 1);
                    if (key_segment_index == 0) {{
                        staged_outputs[{token_expression} - 1]
                            [value_row_in_block] =
                            (InT)reduced_query_partial;
                    }}
                }}
"#,
            token_expression = token_expression
        ),
        PreviousStepReduction::NoPreviousSteps => String::new(),
    };
    let checkpoint_write_source =
        checkpoint_write_source_for_token(checkpoint_write_source_template, &token_expression);
    format!(
        r#"            {{
                float4 current_keys[4];
                for (int fragment_index = 0; fragment_index < 4;
                     ++fragment_index) {{
                    current_keys[fragment_index] = next_keys[fragment_index];
                }}
                const float current_decay = next_decay;
                const float current_update_rate = next_update_rate;
                const float current_value = next_value;
{next_refill_source}                float2 state_key_dot_partials = 0.0f;
                for (int fragment_index = 0; fragment_index < 4;
                     ++fragment_index) {{
                    state_fragment[fragment_index] *= current_decay;
                    state_key_dot_partials = fma(
                        state_fragment[fragment_index].xy,
                        current_keys[fragment_index].xy,
                        state_key_dot_partials);
                    state_key_dot_partials = fma(
                        state_fragment[fragment_index].zw,
                        current_keys[fragment_index].zw,
                        state_key_dot_partials);
                }}
{reduction_source}                float remembered_value = state_key_dot_partials.x +
                    state_key_dot_partials.y;
                remembered_value += simd_shuffle_xor(remembered_value, 4);
                remembered_value += simd_shuffle_xor(remembered_value, 2);
                remembered_value += simd_shuffle_xor(remembered_value, 1);
                float4 current_queries[4];
                for (int fragment_index = 0; fragment_index < 4;
                     ++fragment_index) {{
                    current_queries[fragment_index] = float4(
                        *(const threadgroup vec<InT, 4>*)(
                            &staged_queries[{token_expression}]
                                [4 * (key_segment_index + 8 * fragment_index)]));
                }}
                const float delta = (current_value - remembered_value) *
                    current_update_rate;
                float2 state_query_dot_partials = 0.0f;
                for (int fragment_index = 0; fragment_index < 4;
                     ++fragment_index) {{
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
                }}
{checkpoint_write_source}{partial_publication_source}            }}
"#,
        next_refill_source = next_refill_source,
        reduction_source = reduction_source,
        checkpoint_write_source = checkpoint_write_source,
        partial_publication_source = partial_publication_source,
        token_expression = token_expression
    )
}

fn reduce_scatter_four_steps_source(first_reduced_token: usize) -> String {
    format!(
        r#"            {{
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
                if ((key_segment_index & 1) == 0) {{
                    staged_outputs[{first_reduced_token} +
                        (high_half_lane ? 2 : 0) + (high_pair_lane ? 1 : 0)]
                        [value_row_in_block] = (InT)reduced_query_partial;
                }}
            }}
"#,
        first_reduced_token = first_reduced_token
    )
}
