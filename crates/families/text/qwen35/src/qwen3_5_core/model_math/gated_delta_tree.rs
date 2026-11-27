//! The gated-delta recurrence over the rows of a draft tree: every row starts
//! from its parent's state, not the previous row's. The tree kernel
//! (`metal/astro_family_qwen35_gated_delta_tree.metal`) and the replay kernel
//! (`metal/astro_family_qwen35_gated_delta_replay.metal`) explain the
//! arithmetic; this file launches them and builds the chains they read.

use crate::metal_kernels::{
    ASTRO_FAMILY_QWEN35_GATED_DELTA_REPLAY, ASTRO_FAMILY_QWEN35_GATED_DELTA_TREE,
};
use mlx_bridge::{MlxArray, MlxMetalKernel, MlxMetalKernelOutput, MlxMetalKernelTemplateArgument};
use model_primitives::metal_kernel_declaration::declare_metal_kernel;
use runtime_integration::{MlxRuntime, MlxRuntimeError};

use super::gated_delta_inputs::GatedDeltaInputs;
use super::gated_delta_launch_geometry::{THREADGROUP_THREAD_COUNT, VALUE_ROW_BLOCK_SIZE};
use super::gated_delta_sequence_contract::{
    template_arguments, validate_gated_delta_sequence_shapes,
};

/// The replay kernel, declared once per model beside the tree kernel.
pub fn qwen3_5_gated_delta_replay_kernel() -> Result<MlxMetalKernel, MlxRuntimeError> {
    declare_metal_kernel(&ASTRO_FAMILY_QWEN35_GATED_DELTA_REPLAY, &[])
}

/// The recurrent state after the kept rows (`kept_rows` int32 [n], in
/// order) of a verified tree, from the round's starting state.
pub fn qwen3_5_gated_delta_replay(
    runtime: &MlxRuntime,
    replay_kernel: &MlxMetalKernel,
    inputs: &GatedDeltaInputs<'_>,
    kept_rows: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    let sequence_shape = validate_gated_delta_sequence_shapes(inputs)?;
    let GatedDeltaInputs {
        queries,
        keys,
        values,
        decays,
        update_rates,
        recurrent_state,
    } = *inputs;
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
    declare_metal_kernel(&ASTRO_FAMILY_QWEN35_GATED_DELTA_TREE, &[])
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
pub fn qwen3_5_gated_delta_tree(
    runtime: &MlxRuntime,
    tree_kernel: &MlxMetalKernel,
    inputs: &GatedDeltaInputs<'_>,
    row_chains: &UploadedTreeRowChains,
) -> Result<MlxArray, MlxRuntimeError> {
    let sequence_shape = validate_gated_delta_sequence_shapes(inputs)?;
    let GatedDeltaInputs {
        queries,
        keys,
        values,
        decays,
        update_rates,
        recurrent_state,
    } = *inputs;
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
