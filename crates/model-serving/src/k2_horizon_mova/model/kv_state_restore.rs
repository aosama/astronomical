//! Persistent-cache block restoration for K2 key/value state, one
//! concatenation per tensor.

use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};

use super::decoder::K2HorizonMoVAKvState;
use astronomical_mlx_c_rust::MlxArray;

impl K2HorizonMoVAKvState {
    /// Seats the restored K/V by concatenating every block slice along the
    /// token axis once per tensor, keeping the state's growth headroom.
    pub fn restore_block_slices(
        &mut self,
        runtime: &MlxRuntime,
        block_keys: &[&MlxArray],
        block_values: &[&MlxArray],
        restored_token_count: i32,
    ) -> Result<(), MlxRuntimeError> {
        match self {
            Self::FullPrecision(state) => state.restore_from_block_slices(
                runtime,
                block_keys,
                block_values,
                restored_token_count,
                state.growth_headroom_tokens(),
            ),
            Self::Quantized(state) => state.restore_from_bf16_block_slices(
                runtime,
                block_keys,
                block_values,
                restored_token_count,
                state.growth_headroom_tokens(),
            ),
        }
    }

    /// Rolls the logical offset back to an adopted prefix boundary; the
    /// stale KV tail is overwritten by subsequent in-place updates.
    pub fn truncate_to_offset(
        &mut self,
        checkpoint_offset_tokens: i32,
    ) -> Result<(), MlxRuntimeError> {
        match self {
            Self::FullPrecision(state) => state.truncate_to_offset(checkpoint_offset_tokens),
            Self::Quantized(state) => state.truncate_to_offset(checkpoint_offset_tokens),
        }
    }

    /// Logical payload bytes across both K and V slabs.
    pub fn payload_byte_count(&self) -> u64 {
        match self {
            Self::FullPrecision(state) => state.payload_byte_count(),
            Self::Quantized(state) => state.payload_byte_count(),
        }
    }
}
