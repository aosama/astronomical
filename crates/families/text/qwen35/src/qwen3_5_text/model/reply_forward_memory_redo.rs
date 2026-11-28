//! Runs a reply forward again when a streamed expert page was refused for
//! room, as prompt reading already does for a refused span.
//!
//! A reply forward holds every page it streams until its evaluation ends,
//! and the ledger learns how much that is only from forwards it has
//! measured. An early forward over cold tables, or one whose reads come
//! slowly from a cold `macos_file_cache`, can hold more than was planned:
//! on the 35B-A3B at an 8 GiB limit the page of layer 39 was refused 11 MB
//! short and the reply ended, though the experts kept in memory could have
//! made that room many times over. The refusal comes while the forward is
//! still being built, before its evaluation, so the decoder state is
//! restored the way a refused in-graph lookup already restores it, the
//! governor leases the missing room, the `astro_expert_cache` gives back the
//! experts that room displaces, and the forward runs again.
//!
//! Only a forward with a checkpoint to return to is run again: the exact
//! rerun after a speculative attempt, which is where the refusals were seen.
//! A failure inside evaluation stays an error, since the attention state may
//! then still point into the failed graph.

use text_generation::RequestDecoderStateStackCheckpoint;

use super::Qwen3_5Model;
use super::{RequestDecoderStateStack, TextExecutionError};
use crate::MemoryPhase;

/// Bytes a refused page was short of, or `None` for any other error.
pub(super) const fn refused_page_shortfall_bytes(
    forward_error: &TextExecutionError,
) -> Option<u64> {
    match forward_error {
        TextExecutionError::MemoryAdmissionRejected(
            crate::MlxAllocationAdmissionError::Rejected {
                shortfall_bytes, ..
            },
        ) => Some(*shortfall_bytes),
        _ => None,
    }
}

impl Qwen3_5Model {
    /// The state to return to should the exact rerun of a reply forward be
    /// refused for room; `None` while a prompt is read, whose own recovery
    /// redoes the span.
    fn checkpoint_for_reply_redo(
        &self,
        request_decoder_state: &RequestDecoderStateStack,
    ) -> Result<Option<RequestDecoderStateStackCheckpoint>, TextExecutionError> {
        if self.current_ledger_phase(MemoryPhase::Decode).is_none() {
            return Ok(None);
        }
        Ok(Some(request_decoder_state.checkpoint()?))
    }

    /// Returns the decoder state to where a speculative attempt began, for
    /// its exact rerun, and keeps that state to return to should the rerun be
    /// refused room.
    pub(super) fn restore_for_exact_rerun(
        &self,
        request_decoder_state: &mut RequestDecoderStateStack,
        state_before_speculation: RequestDecoderStateStackCheckpoint,
    ) -> Result<Option<RequestDecoderStateStackCheckpoint>, TextExecutionError> {
        request_decoder_state.restore_checkpoint(state_before_speculation)?;
        self.checkpoint_for_reply_redo(request_decoder_state)
    }

    /// Makes room to run a refused reply forward again and returns the state
    /// to return to should that run be refused too. Returns `forward_error`
    /// itself when the forward cannot be run again: it was not a refusal for
    /// room, there is no state to return to, or the governor's attempts are
    /// spent.
    pub(super) fn room_to_redo_reply_forward(
        &self,
        forward_error: TextExecutionError,
        request_decoder_state: &mut RequestDecoderStateStack,
        state_before_forward: Option<RequestDecoderStateStackCheckpoint>,
    ) -> Result<RequestDecoderStateStackCheckpoint, TextExecutionError> {
        // Nothing is touched unless a reply is being written: a prompt's own
        // recovery rebuilds its state and must find it as the failure left it.
        let (Some(_), Some(shortfall_bytes), Some(state_before_forward)) = (
            self.current_ledger_phase(MemoryPhase::Decode),
            refused_page_shortfall_bytes(&forward_error),
            state_before_forward,
        ) else {
            return Err(forward_error);
        };
        self.discard_previous_forward_leftovers();
        let discarded_page_count = match self.astro_expert_cache.as_ref() {
            Some(astro_expert_cache) => astro_expert_cache.borrow_mut().discard_pending_inserts(),
            None => 0,
        };
        request_decoder_state.restore_checkpoint(state_before_forward)?;
        if !self.govern_redo_after_memory_failure(MemoryPhase::Decode, shortfall_bytes) {
            return Err(forward_error);
        }
        // The experts given back and the failed attempt's pages leave active
        // memory only once MLX's buffer cache lets them go; the peak starts
        // again so the failed attempt is not measured as this step's need.
        self.runtime
            .synchronize_gpu_stream_and_clear_mlx_buffer_cache()?;
        self.runtime.reset_peak_memory()?;
        tracing::warn!(
            shortfall_bytes,
            discarded_page_count,
            "a reply forward was refused room for an expert page; running it again"
        );
        Ok(request_decoder_state.checkpoint()?)
    }

    /// A forward of reply rows as its loop reads it: usable outputs, `None`
    /// for outputs that cannot be used, or an error. A forward refused room
    /// for a page becomes `None` with `state_before_in_graph_attempt` set to
    /// the state the loop then runs it again from, exactly as it does after a
    /// refused lookup.
    pub(super) fn reply_rows_or_room_to_redo<ForwardOutput>(
        &self,
        attempt: Result<Option<ForwardOutput>, TextExecutionError>,
        request_decoder_state: &mut RequestDecoderStateStack,
        state_before_in_graph_attempt: &mut Option<RequestDecoderStateStackCheckpoint>,
        state_before_exact_rerun: &mut Option<RequestDecoderStateStackCheckpoint>,
    ) -> Result<Option<ForwardOutput>, TextExecutionError> {
        let forward_error = match attempt {
            Ok(forward_output) => return Ok(forward_output),
            Err(forward_error) => forward_error,
        };
        let state_before_forward = state_before_in_graph_attempt
            .take()
            .or(state_before_exact_rerun.take());
        *state_before_in_graph_attempt = Some(self.room_to_redo_reply_forward(
            forward_error,
            request_decoder_state,
            state_before_forward,
        )?);
        Ok(None)
    }
}

#[cfg(test)]
#[path = "reply_forward_memory_redo_tests.rs"]
mod tests;
