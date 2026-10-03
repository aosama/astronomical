//! Incremental persistent-cache restoration for K2 key/value state.

use astronomical_runtime_integration::{MlxArray, MlxRuntime, MlxRuntimeError};

use super::decoder::K2HorizonMoVAKvState;

impl K2HorizonMoVAKvState {
    /// Begins restore from the first block, reserving configured growth headroom.
    pub fn begin_incremental_block_restore(
        &mut self,
        runtime: &MlxRuntime,
        first_block_keys: MlxArray,
        first_block_values: MlxArray,
        restored_token_count: i32,
    ) -> Result<(), MlxRuntimeError> {
        match self {
            Self::FullPrecision(state) => state.begin_incremental_block_restore(
                runtime,
                first_block_keys,
                first_block_values,
                restored_token_count,
                state.growth_headroom_tokens(),
            ),
            Self::Quantized(state) => state.begin_incremental_bf16_block_restore(
                runtime,
                first_block_keys,
                first_block_values,
                restored_token_count,
                state.growth_headroom_tokens(),
            ),
        }
    }

    /// Writes the next block into the preallocated destination.
    pub fn absorb_incremental_block_restore(
        &mut self,
        runtime: &MlxRuntime,
        block_keys: MlxArray,
        block_values: MlxArray,
        block_start_tokens: i32,
    ) -> Result<(), MlxRuntimeError> {
        match self {
            Self::FullPrecision(state) => state.absorb_incremental_restore_block(
                runtime,
                block_keys,
                block_values,
                block_start_tokens,
            ),
            Self::Quantized(state) => state.absorb_incremental_bf16_restore_block(
                runtime,
                block_keys,
                block_values,
                block_start_tokens,
            ),
        }
    }

    /// Verifies that every restored token was absorbed.
    pub fn finish_incremental_block_restore(
        &mut self,
        restored_token_count: i32,
    ) -> Result<(), MlxRuntimeError> {
        match self {
            Self::FullPrecision(state) => {
                state.finish_incremental_block_restore(restored_token_count)
            }
            Self::Quantized(state) => {
                state.finish_incremental_bf16_block_restore(restored_token_count)
            }
        }
    }
}
