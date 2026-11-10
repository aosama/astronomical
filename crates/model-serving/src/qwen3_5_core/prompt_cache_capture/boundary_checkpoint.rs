//! Records the recurrent state at a prefill chunk that ends on a prompt-cache
//! block boundary, for block capture to publish after the chunk.

use super::block_capture::required_prompt_state_persistence_failure;
use crate::Qwen3_5PersistentPromptCacheBoundaryCheckpoint;
use crate::qwen3_5_core::engine_request::Qwen3_5EngineRequest;

/// Snapshots the recurrent state when this chunk completes a block boundary;
/// chunks that end anywhere else record nothing.
pub(crate) fn record_persistent_prompt_cache_boundary_checkpoint<RequestExtension>(
    active_request: &mut Qwen3_5EngineRequest<RequestExtension>,
    prefill_token_count: usize,
    all_completed_prefill_chunk_tokens: &[usize],
    boundary_checkpoints: &mut Vec<Qwen3_5PersistentPromptCacheBoundaryCheckpoint>,
) -> Result<(), crate::InferenceEngineError> {
    if all_completed_prefill_chunk_tokens.last().copied() != Some(prefill_token_count) {
        return Ok(());
    }
    let recurrent_snapshot_tensors = active_request
        .request_decoder_state
        .extract_persistent_prompt_cache_recurrent_snapshot_tensors();
    match recurrent_snapshot_tensors {
        Ok(recurrent_snapshot_tensors) => {
            boundary_checkpoints.push(Qwen3_5PersistentPromptCacheBoundaryCheckpoint {
                completed_prefill_chunk_tokens: prefill_token_count,
                recurrent_snapshot_tensors,
            });
            Ok(())
        }
        Err(error) => Err(required_prompt_state_persistence_failure(
            active_request,
            "exact target prompt-state extraction",
            error,
        )),
    }
}
