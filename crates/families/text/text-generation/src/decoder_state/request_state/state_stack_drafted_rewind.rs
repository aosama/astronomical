//! Rewinding a request's `astro_decoder_state_cache` after a drafted block
//! keeps only its first rows.
//!
//! The verify pass already wrote every row into the live state. Attention
//! layers only appended keys and values, so moving their offset back drops the
//! refused rows. DeltaNet layers fold every row into one running state, which
//! cannot be undone, so the verify pass captured that state after each row
//! (`Qwen3_5StreamingModel::verify_drafted_rows`) and the kept row's capture is
//! put back. Nothing runs through the model again: the old rewind restored a
//! checkpoint and ran the kept rows a second time, 71 ms a round on the 27B.

use std::collections::HashMap;

use mlx_bridge::MlxArray;
use runtime_integration::MlxRuntimeError;

use super::state_stack::RequestDecoderStateStack;
use crate::astro_decoder_state_cache::AstroDecoderStateCacheState;

impl RequestDecoderStateStack {
    /// Leaves the state as if only the first kept rows had been forwarded:
    /// attention offsets move back to `kept_token_offset` (every token now in
    /// the state), and each DeltaNet layer takes its convolution and recurrent
    /// state from `kept_row_boundary`, the verify pass's capture after the last
    /// kept row. The tensors are taken out of the boundary map.
    pub fn rewind_to_kept_row_boundary(
        &mut self,
        kept_row_boundary: &mut HashMap<String, MlxArray>,
        kept_token_offset: i32,
    ) -> Result<(), MlxRuntimeError> {
        for layer_index in 0..self.layer_count() {
            match self.layer_mut(layer_index) {
                Some(AstroDecoderStateCacheState::AppendOnlyAttention { attention }) => {
                    attention.truncate_to_offset(kept_token_offset)?;
                }
                Some(AstroDecoderStateCacheState::Composite {
                    convolution,
                    recurrent,
                }) => {
                    convolution.restore_from_snapshot(take_boundary_tensor(
                        kept_row_boundary,
                        &format!("layer_{layer_index}_linear.convolution"),
                    )?);
                    recurrent.restore_from_snapshot(take_boundary_tensor(
                        kept_row_boundary,
                        &format!("layer_{layer_index}_linear.gated_delta_recurrent"),
                    )?);
                }
                None => {
                    return Err(rewind_error(
                        "a decoder layer is missing from the state stack",
                    ));
                }
            }
        }
        Ok(())
    }
}

fn take_boundary_tensor(
    kept_row_boundary: &mut HashMap<String, MlxArray>,
    tensor_name: &str,
) -> Result<MlxArray, MlxRuntimeError> {
    kept_row_boundary
        .remove(tensor_name)
        .ok_or_else(|| rewind_error("the kept row's boundary has no state for a DeltaNet layer"))
}

fn rewind_error(description: &str) -> MlxRuntimeError {
    MlxRuntimeError::RuntimeOperation {
        operation: "rewind a drafted block to its kept rows",
        description: description.to_owned(),
    }
}
