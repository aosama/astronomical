//! Best-effort capture of the prompt's final partial block ("tail").
//!
//! A prompt whose token count is not a multiple of the block size leaves its
//! final tokens outside every published full block. Persisting that tail lets
//! the next turn restore up to the last prompt token instead of prefilling it
//! again. Tail capture is best-effort by contract: the request never depends
//! on tail reuse, so every failure is logged and skipped.

use crate::{
    PerformanceOperation, PersistentPromptCacheBlockCausalInput, PersistentPromptCacheBlockKey,
    PersistentPromptCacheDiskStore, PersistentPromptCachePublicationOutcome,
};

use super::Qwen3_5EngineState;
use super::engine_request::Qwen3_5EngineRequest;
use super::persistent_prompt_cache_capture::PromptBlockPublicationFailure;
use crate::qwen3_5_resident::model::Qwen3_5ResidentModel;

impl Qwen3_5EngineState {
    /// Captures the prompt's final partial block after the terminal prefill
    /// chunk, extending the last durable full block of this request's chain.
    /// A tail is a leaf: it is never recorded as a capture parent.
    pub(super) fn capture_persistent_prompt_cache_partial_tail_block(
        &self,
        persistent_prompt_cache: &PersistentPromptCacheDiskStore,
        model: &Qwen3_5ResidentModel,
        active_request: &mut Qwen3_5EngineRequest,
        prefill_end: usize,
    ) {
        if !active_request.can_use_persistent_prompt_cache {
            return;
        }
        let prompt_token_count = active_request.input_token_ids.len();
        // Only the terminal chunk holds every prompt token in decoder state;
        // prediction sessions reserve their final token for generation kickoff
        // and never reach a full-prompt prefill end.
        if prefill_end != prompt_token_count {
            return;
        }
        let persistent_prompt_cache_block_token_count =
            persistent_prompt_cache.model_contract.block_token_count();
        let tail_token_count = prompt_token_count % persistent_prompt_cache_block_token_count;
        if tail_token_count == 0 {
            return;
        }
        let tail_block_index = prompt_token_count / persistent_prompt_cache_block_token_count;
        let tail_block_start = tail_block_index * persistent_prompt_cache_block_token_count;
        let empty_block_causal_input = PersistentPromptCacheBlockCausalInput::empty();
        let tail_block_causal_input = if active_request
            .persistent_prompt_cache_block_causal_inputs
            .is_empty()
        {
            &empty_block_causal_input
        } else {
            let Some(tail_block_causal_input) = active_request
                .persistent_prompt_cache_block_causal_inputs
                .get(tail_block_index)
            else {
                tracing::warn!(
                    request_id = active_request.request_id.value(),
                    tail_block_index,
                    "prompt-cache causal input plan does not cover the partial tail; skipping tail capture"
                );
                return;
            };
            tail_block_causal_input
        };
        let tail_tokens = &active_request.input_token_ids[tail_block_start..prompt_token_count];
        let parent_block_key = active_request
            .last_restored_persistent_prompt_cache_block_key
            .clone();
        let tail_block_key = match parent_block_key.as_ref() {
            None => PersistentPromptCacheBlockKey::for_root_block_with_causal_input(
                &persistent_prompt_cache.model_contract,
                tail_tokens,
                tail_block_causal_input,
            ),
            Some(parent_block_key) => parent_block_key
                .for_child_block_with_causal_input(tail_tokens, tail_block_causal_input),
        };
        let Ok(tail_block_key) = tail_block_key else {
            tracing::warn!(
                request_id = active_request.request_id.value(),
                "prompt-cache partial tail identity construction failed; skipping tail capture"
            );
            return;
        };
        let kv_block_tensors = match active_request.measure_operation_with_request(
            PerformanceOperation::PersistentPromptCacheStateExtraction,
            |active_request| {
                active_request
                    .request_decoder_state
                    .extract_persistent_prompt_cache_kv_block_tensors(
                        model.runtime(),
                        tail_block_start,
                        prompt_token_count,
                        persistent_prompt_cache_block_token_count,
                    )
            },
        ) {
            Ok(kv_block_tensors) => kv_block_tensors,
            Err(error) => {
                tracing::warn!(
                    request_id = active_request.request_id.value(),
                    %error,
                    "prompt-cache partial tail KV extraction failed; skipping tail capture"
                );
                return;
            }
        };
        let recurrent_snapshot_tensors = match active_request.measure_operation_with_request(
            PerformanceOperation::PersistentPromptCacheStateExtraction,
            |active_request| {
                active_request
                    .request_decoder_state
                    .extract_persistent_prompt_cache_recurrent_snapshot_tensors()
            },
        ) {
            Ok(recurrent_snapshot_tensors) => recurrent_snapshot_tensors,
            Err(error) => {
                tracing::warn!(
                    request_id = active_request.request_id.value(),
                    %error,
                    "prompt-cache partial tail snapshot extraction failed; skipping tail capture"
                );
                return;
            }
        };
        match self.publish_block_with_reclamation_retry(
            model,
            active_request,
            persistent_prompt_cache,
            &tail_block_key,
            parent_block_key.as_ref(),
            &kv_block_tensors,
            &recurrent_snapshot_tensors,
        ) {
            Ok(publication_outcome) => {
                if publication_outcome == PersistentPromptCachePublicationOutcome::Published
                    && let Some(persistent_prompt_cache_diagnostics) =
                        active_request.persistent_prompt_cache_diagnostics.as_mut()
                {
                    persistent_prompt_cache_diagnostics.record_published_block();
                }
            }
            Err(PromptBlockPublicationFailure::Reclamation(error)) => {
                tracing::warn!(
                    request_id = active_request.request_id.value(),
                    %error,
                    "prompt-cache partial tail publication reclamation failed; skipping tail capture"
                );
            }
            Err(PromptBlockPublicationFailure::Publication(error)) => {
                tracing::warn!(
                    request_id = active_request.request_id.value(),
                    %error,
                    "prompt-cache partial tail publication failed; skipping tail capture"
                );
            }
        }
    }
}
