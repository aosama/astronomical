//! Pre-forward decision logic for a single prompt-processing chunk.
//!
//! This module computes cache eligibility, checkpoint boundaries, workspace
//! projection without side effects.

use crate::persistent_prompt_cache_boundary_completed_prefill_chunk_tokens;

use super::Qwen3_5EngineState;
use super::engine_request::Qwen3_5EngineRequest;
use super::prompt_prefill_errors::PromptPrefillChunkAttemptError;
/// Immutable decisions computed before memory admission and forward execution.
///
/// Every field is a pure function of the request state and the model config.
/// Nothing here triggers GPU work or mutable engine state.
pub(super) struct PrefillChunkPlan {
    /// Token count of this chunk (`prefill_end - prefill_start`).
    pub(super) prefill_token_count: usize,

    /// Completed chunk-end offsets (relative to chunk start) where a prompt
    /// cache boundary checkpoint should be captured after the forward.
    pub(super) all_completed_prefill_chunk_tokens: Vec<usize>,

    /// Like `all_completed_prefill_chunk_tokens` but with the final boundary
    /// removed — intermediate checkpoints that need a snapshot during the
    /// forward.
    pub(super) intermediate_completed_prefill_chunk_tokens: Vec<usize>,

    /// Block token count from the persistent prompt-cache contract, if
    /// applicable.
    pub(super) persistent_prompt_cache_block_token_count: Option<usize>,

    /// Workspace bytes needed for direct prompt-cache publication.
    pub(super) direct_publication_workspace_bytes: usize,

    /// Combined temporary workspace reservation.
    pub(super) exact_temporary_workspace_bytes: usize,
}

impl Qwen3_5EngineState {
    /// Compute all pre-admission decisions for a prompt-processing chunk.
    ///
    /// Returns a plan that feeds into `execute_prompt_prefill_chunk`'s
    /// admission and forward phases. This method performs no GPU work and
    /// no mutable engine state changes.
    pub(super) fn plan_prompt_prefill_chunk(
        &self,
        active_request: &Qwen3_5EngineRequest,
        prefill_start: usize,
        prefill_end: usize,
    ) -> Result<PrefillChunkPlan, PromptPrefillChunkAttemptError> {
        let model = self
            .model
            .as_ref()
.ok_or_else(super::super::generation_session_lifecycle::generation_error_constructors::engine_lost_loaded_model_error)?;
        let prefill_token_count = prefill_end - prefill_start;
        let capture_is_eligible = self.persistent_prompt_cache.is_some()
            && active_request.can_use_persistent_prompt_cache;

        // Cache-disabled and request-ineligible paths stop here: they do not
        // plan checkpoint boundaries, derive a synthetic block length, or
        // reserve checkpoint/publication memory. Cache-enabled execution uses
        // the one contract owned by the open store.
        let (all_completed_prefill_chunk_tokens, persistent_prompt_cache_block_token_count) =
            if capture_is_eligible {
                let persistent_prompt_cache_block_token_count = self
                    .persistent_prompt_cache
                    .as_ref()
.ok_or_else(super::super::generation_session_lifecycle::generation_error_constructors::eligible_prompt_cache_capture_has_no_persistent_cache_owner_error)?
                    .model_contract_ref()
                    .block_token_count();
                (
                    persistent_prompt_cache_boundary_completed_prefill_chunk_tokens(
                        prefill_start,
                        prefill_end,
                        persistent_prompt_cache_block_token_count,
                    ),
                    Some(persistent_prompt_cache_block_token_count),
                )
            } else {
                (Vec::new(), None)
            };

        let mut intermediate_completed_prefill_chunk_tokens =
            all_completed_prefill_chunk_tokens.clone();
        if intermediate_completed_prefill_chunk_tokens.last().copied() == Some(prefill_token_count)
        {
            intermediate_completed_prefill_chunk_tokens.pop();
        }

        let boundary_checkpoint_workspace_bytes = if intermediate_completed_prefill_chunk_tokens
            .is_empty()
        {
            0
        } else {
            model
                    .decoder_cache_layout()
                    .boundary_snapshot_payload_byte_count()
                    .map_err(|error| {
                        super::fatal_engine_error(format!(
                            "failed to project boundary checkpoint workspace: {error}"
                        ))
                    })?
                    .checked_mul(intermediate_completed_prefill_chunk_tokens.len())
.ok_or_else(super::super::generation_session_lifecycle::generation_error_constructors::boundary_checkpoint_workspace_bytes_overflowed_error)?
        };

        let direct_publication_workspace_bytes = if !all_completed_prefill_chunk_tokens.is_empty() {
            self.persistent_prompt_cache
                .as_ref()
                .map(|persistent_prompt_cache| {
                    persistent_prompt_cache
                        .model_contract_ref()
                        .direct_publication_workspace_bytes()
                })
                .unwrap_or(0)
        } else {
            0
        };

        let exact_temporary_workspace_bytes = boundary_checkpoint_workspace_bytes
            .checked_add(direct_publication_workspace_bytes)
.ok_or_else(super::super::generation_session_lifecycle::generation_error_constructors::prompt_cache_publication_workspace_bytes_overflowed_error)?;

        Ok(PrefillChunkPlan {
            prefill_token_count,
            all_completed_prefill_chunk_tokens,
            intermediate_completed_prefill_chunk_tokens,
            persistent_prompt_cache_block_token_count,
            direct_publication_workspace_bytes,
            exact_temporary_workspace_bytes,
        })
    }
}
