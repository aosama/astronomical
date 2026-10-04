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
}
