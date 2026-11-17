//! The gated-delta recurrence over the rows of a draft tree: every row starts
//! from its parent's state, not the previous row's.
//!
//! TensorFold's tree kernel (`lane_tree.py`, release 0.3.5.1, Apache-2.0)
//! keeps every row's state in registers, which fits its 32 lanes per value
//! row. This engine's sequence kernel spreads a row over 8 lanes, so 16 rows
//! of state would be 256 floats a lane and spill. Instead each row replays
//! its own ancestor chain from the round's starting state: at most a block's
//! depth (8 steps) of register arithmetic, while the starting state, the
//! only large read, is loaded once a thread. Each step is the sequence
//! kernel's step, operation for operation (`gated_delta_pipelined_kernel`),
//! so a row's output is bit-identical to feeding its chain alone.

use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};
use mlx_bridge::{MlxArray, MlxMetalKernel, MlxMetalKernelOutput, MlxMetalKernelTemplateArgument};

use super::gated_delta_pipelined_kernel::{THREADGROUP_THREAD_COUNT, VALUE_ROW_BLOCK_SIZE};
use super::gated_delta_sequence_contract::{
    template_arguments, validate_gated_delta_sequence_shapes,
};

/// Inputs as the sequence kernel takes them, `[1, rows, ...]`, plus
/// `row_chains` [rows, MAXIMUM_CHAIN] (each row's chain from the anchor to
/// itself) and `chain_lengths` [rows]. Output `outputs` [1, rows, Hv, Dv].
const GATED_DELTA_TREE_SOURCE: &str = r#"
    constexpr int value_row_block_size = 16;
    auto thread_index = thread_position_in_threadgroup.x;
    auto value_head_index = threadgroup_position_in_grid.y;
    auto key_head_index = value_head_index / (Hv / Hk);
    auto value_row = threadgroup_position_in_grid.x * value_row_block_size + thread_index / 8;
    auto key_segment_index = thread_index % 8;
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
"#;

/// Inputs as the tree kernel's, plus `kept_rows` [n] int32. Output
/// `next_recurrent_state`: the starting state advanced over the kept rows in
/// order, by the same step, so it equals the sequence kernel's final state
/// over those rows without gathering them first (five gathers a layer, 240 a
/// round on the 27B).
const GATED_DELTA_REPLAY_SOURCE: &str = r#"
    constexpr int value_row_block_size = 16;
    auto thread_index = thread_position_in_threadgroup.x;
    auto value_head_index = threadgroup_position_in_grid.y;
    auto key_head_index = value_head_index / (Hv / Hk);
    auto value_row = threadgroup_position_in_grid.x * value_row_block_size + thread_index / 8;
    auto key_segment_index = thread_index % 8;
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
"#;

/// The replay kernel, declared once per model beside the tree kernel.
pub fn qwen3_5_gated_delta_replay_kernel() -> Result<MlxMetalKernel, MlxRuntimeError> {
    Ok(MlxMetalKernel::new(
        "astronomical_qwen3_5_gated_delta_replay",
        &[
            "keys",
            "values",
            "decays",
            "update_rates",
            "recurrent_state",
            "kept_rows",
        ],
        &["next_recurrent_state"],
        GATED_DELTA_REPLAY_SOURCE,
    )?)
}

/// The recurrent state after the kept rows (`kept_rows` int32 [n], in
/// order) of a verified tree, from the round's starting state.
#[allow(clippy::too_many_arguments)]
pub fn qwen3_5_gated_delta_replay(
    runtime: &MlxRuntime,
    replay_kernel: &MlxMetalKernel,
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    decays: &MlxArray,
    update_rates: &MlxArray,
    recurrent_state: &MlxArray,
    kept_rows: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    let sequence_shape = validate_gated_delta_sequence_shapes(
        queries,
        keys,
        values,
        decays,
        update_rates,
        recurrent_state,
    )?;
    let mut outputs = runtime.apply_metal_kernel(
        replay_kernel,
        &[
            keys,
            values,
            decays,
            update_rates,
            recurrent_state,
            kept_rows,
        ],
        &[MlxMetalKernelOutput::new(
            recurrent_state.shape(),
            recurrent_state.dtype(),
        )],
        [
            THREADGROUP_THREAD_COUNT as i32
                * (sequence_shape.value_head_dimension / VALUE_ROW_BLOCK_SIZE as i32),
            sequence_shape.value_head_count,
            1,
        ],
        [THREADGROUP_THREAD_COUNT as i32, 1, 1],
        &template_arguments(sequence_shape, queries.dtype(), recurrent_state.dtype()),
    )?;
    outputs
        .pop()
        .ok_or_else(|| MlxRuntimeError::RuntimeOperation {
            operation: "replay the kept rows of the gated-delta recurrence",
            description: "the kernel returned no output".to_owned(),
        })
}

/// The tree kernel, declared once per model.
pub fn qwen3_5_gated_delta_tree_kernel() -> Result<MlxMetalKernel, MlxRuntimeError> {
    Ok(MlxMetalKernel::new(
        "astronomical_qwen3_5_gated_delta_tree",
        &[
            "queries",
            "keys",
            "values",
            "decays",
            "update_rates",
            "recurrent_state",
            "row_chains",
            "chain_lengths",
        ],
        &["outputs"],
        GATED_DELTA_TREE_SOURCE,
    )?)
}

/// Each tree row's chain from the anchor, padded to one width: the
/// `row_chains` and `chain_lengths` the kernel reads.
pub struct TreeRowChains {
    pub row_chains: Vec<i32>,
    pub chain_lengths: Vec<i32>,
    pub maximum_chain: i32,
}

impl TreeRowChains {
    /// From each row's parent row (None for the anchor); parents precede
    /// their children.
    pub fn from_parent_rows(parent_rows: &[Option<usize>]) -> Self {
        let mut chains: Vec<Vec<i32>> = Vec::with_capacity(parent_rows.len());
        for (row, parent_row) in parent_rows.iter().enumerate() {
            let mut chain = match parent_row {
                Some(parent_row) => chains[*parent_row].clone(),
                None => Vec::new(),
            };
            chain.push(row as i32);
            chains.push(chain);
        }
        let mut maximum_chain = 1;
        for chain in &chains {
            maximum_chain = maximum_chain.max(chain.len());
        }
        // MLX compiles one kernel per template value; whole powers of two
        // keep that to a handful however deep copies reach.
        let maximum_chain = maximum_chain.next_power_of_two();
        let mut row_chains = Vec::with_capacity(chains.len() * maximum_chain);
        let mut chain_lengths = Vec::with_capacity(chains.len());
        for chain in &chains {
            chain_lengths.push(chain.len() as i32);
            row_chains.extend_from_slice(chain);
            row_chains.resize(row_chains.len() + maximum_chain - chain.len(), 0);
        }
        Self {
            row_chains,
            chain_lengths,
            maximum_chain: maximum_chain as i32,
        }
    }
}

/// A tree's chains on the GPU, uploaded once for every layer of a pass.
pub struct UploadedTreeRowChains {
    row_chains: MlxArray,
    chain_lengths: MlxArray,
    maximum_chain: i32,
    row_count: i32,
}

impl UploadedTreeRowChains {
    /// int32 [rows, maximum chain]: each row's chain from the anchor to itself.
    pub fn row_chains(&self) -> &MlxArray {
        &self.row_chains
    }

    /// int32 [rows]: how many rows each chain holds, the row itself included.
    pub fn chain_lengths(&self) -> &MlxArray {
        &self.chain_lengths
    }

    pub fn maximum_chain(&self) -> i32 {
        self.maximum_chain
    }

    pub fn row_count(&self) -> i32 {
        self.row_count
    }
}

impl TreeRowChains {
    pub fn upload(&self, runtime: &MlxRuntime) -> Result<UploadedTreeRowChains, MlxRuntimeError> {
        let row_count = self.chain_lengths.len() as i32;
        Ok(UploadedTreeRowChains {
            row_chains: runtime
                .array_from_i32(&self.row_chains, &[row_count, self.maximum_chain])?,
            chain_lengths: runtime.array_from_i32(&self.chain_lengths, &[row_count])?,
            maximum_chain: self.maximum_chain,
            row_count,
        })
    }
}

/// The recurrence's output for every tree row, `[1, rows, Hv, Dv]`. The
/// starting `recurrent_state` is only read: the round commits the kept chain
/// separately, through the sequence kernel over that chain's rows.
#[allow(clippy::too_many_arguments)]
pub fn qwen3_5_gated_delta_tree(
    runtime: &MlxRuntime,
    tree_kernel: &MlxMetalKernel,
    queries: &MlxArray,
    keys: &MlxArray,
    values: &MlxArray,
    decays: &MlxArray,
    update_rates: &MlxArray,
    recurrent_state: &MlxArray,
    row_chains: &UploadedTreeRowChains,
) -> Result<MlxArray, MlxRuntimeError> {
    let sequence_shape = validate_gated_delta_sequence_shapes(
        queries,
        keys,
        values,
        decays,
        update_rates,
        recurrent_state,
    )?;
    let row_count = row_chains.row_count;
    let mut template =
        template_arguments(sequence_shape, queries.dtype(), recurrent_state.dtype()).to_vec();
    template.push(MlxMetalKernelTemplateArgument::Integer {
        name: "MAXIMUM_CHAIN",
        integer_template_argument: row_chains.maximum_chain,
    });
    let mut outputs = runtime.apply_metal_kernel(
        tree_kernel,
        &[
            queries,
            keys,
            values,
            decays,
            update_rates,
            recurrent_state,
            &row_chains.row_chains,
            &row_chains.chain_lengths,
        ],
        &[MlxMetalKernelOutput::new(
            vec![
                1,
                row_count,
                sequence_shape.value_head_count,
                sequence_shape.value_head_dimension,
            ],
            queries.dtype(),
        )],
        [
            THREADGROUP_THREAD_COUNT as i32
                * (sequence_shape.value_head_dimension / VALUE_ROW_BLOCK_SIZE as i32),
            sequence_shape.value_head_count,
            1,
        ],
        [THREADGROUP_THREAD_COUNT as i32, 1, 1],
        &template,
    )?;
    outputs
        .pop()
        .ok_or_else(|| MlxRuntimeError::RuntimeOperation {
            operation: "run the gated-delta tree recurrence",
            description: "the kernel returned no output".to_owned(),
        })
}
