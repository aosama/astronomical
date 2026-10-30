//! Best-effort capture of the prompt's final partial block ("tail").
//!
//! A prompt whose token count is not a multiple of the block size leaves its
//! final tokens outside every published full block. Persisting that tail lets
//! the next turn restore everything except the final prompt token, which the
//! terminal forward always processes to produce the first logits. Tail capture
//! is best-effort by contract: the request never depends on tail reuse, so
//! every failure is logged and skipped.

use std::collections::HashMap;

use mlx_c_rust::MlxArray;

use crate::persistent_cache::block_causal_input::PersistentPromptCacheBlockCausalInput;
use crate::{
    PerformanceOperation, PersistentPromptCacheBlockKey, PersistentPromptCacheBlockKeyError,
    PersistentPromptCacheDiskStore, PersistentPromptCacheModelContract,
    PersistentPromptCachePublicationOutcome,
};

use super::engine_request::Qwen3_5EngineRequest;
use super::persistent_prompt_cache_capture::PromptBlockPublicationFailure;
use super::{Qwen3_5EngineState, Qwen3_5Model};

/// Token geometry of the partial tail one terminal prefill chunk can capture.
#[derive(Debug, Eq, PartialEq)]
struct PromptTailCaptureSpan {
    tail_capture_end: usize,
    tail_block_start: usize,
    tail_block_index: usize,
    persistent_prompt_cache_block_token_count: usize,
}

/// Outcome of resolving the causal input bound into the tail block identity.
#[derive(Debug)]
enum TailBlockCausalInputResolution<'a> {
    Planned(&'a PersistentPromptCacheBlockCausalInput),
    EmptyFallback,
    MissingFromPlan,
}

/// Everything publication needs about one captured tail block, grouped so the
/// publication step keeps a short signature.
struct TailBlockPublicationInputs<'a> {
    tail_block_key: &'a PersistentPromptCacheBlockKey,
    parent_block_key: Option<&'a PersistentPromptCacheBlockKey>,
    kv_block_tensors: &'a HashMap<String, MlxArray>,
    recurrent_snapshot_tensors: &'a HashMap<String, MlxArray>,
}

/// Computes the token span of the partial tail a terminal prefill chunk can
/// capture, or `None` when this chunk ends anywhere but the capture boundary.
fn prompt_tail_capture_span(
    can_use_persistent_prompt_cache: bool,
    prompt_token_count: usize,
    persistent_prompt_cache_block_token_count: usize,
    prefill_end: usize,
) -> Option<PromptTailCaptureSpan> {
    if !can_use_persistent_prompt_cache {
        return None;
    }
    // The generation-start lookup never consumes the final prompt token —
    // the terminal forward needs it to produce the first logits — so a tail
    // saved through that token could never be probed again. Capture stops
    // one token short, matching the partial-tail probe's retained-suffix
    // rule; the prefill clamp guarantees the chunk ending exactly here
    // carries the complete tail span in decoder state.
    let tail_capture_end = crate::persistent_prompt_cache_partial_tail_capture_end(
        prompt_token_count,
        persistent_prompt_cache_block_token_count,
    );
    if prefill_end != tail_capture_end {
        return None;
    }
    let tail_token_count = tail_capture_end % persistent_prompt_cache_block_token_count;
    if tail_token_count == 0 {
        return None;
    }
    let tail_block_index = tail_capture_end / persistent_prompt_cache_block_token_count;
    let tail_block_start = tail_block_index * persistent_prompt_cache_block_token_count;
    Some(PromptTailCaptureSpan {
        tail_capture_end,
        tail_block_start,
        tail_block_index,
        persistent_prompt_cache_block_token_count,
    })
}

/// Resolves the causal input bound into the tail block identity: the plan
/// entry for the tail block, the empty fallback when the request carries no
/// causal-input plan at all.
fn resolve_tail_block_causal_input<'a>(
    planned_block_causal_inputs: &'a [PersistentPromptCacheBlockCausalInput],
    tail_block_index: usize,
) -> TailBlockCausalInputResolution<'a> {
    if planned_block_causal_inputs.is_empty() {
        return TailBlockCausalInputResolution::EmptyFallback;
    }
    match planned_block_causal_inputs.get(tail_block_index) {
        Some(tail_block_causal_input) => {
            TailBlockCausalInputResolution::Planned(tail_block_causal_input)
        }
        None => TailBlockCausalInputResolution::MissingFromPlan,
    }
}

/// Builds the content-addressed identity of the tail block: a root identity
/// when the request restored no parent block, a chained child identity
/// otherwise.
fn build_tail_block_key(
    persistent_prompt_cache_model_contract: &PersistentPromptCacheModelContract,
    tail_tokens: &[u32],
    tail_block_causal_input: &PersistentPromptCacheBlockCausalInput,
    parent_block_key: Option<&PersistentPromptCacheBlockKey>,
) -> Result<PersistentPromptCacheBlockKey, PersistentPromptCacheBlockKeyError> {
    match parent_block_key {
        None => PersistentPromptCacheBlockKey::for_root_block_with_causal_input(
            persistent_prompt_cache_model_contract,
            tail_tokens,
            tail_block_causal_input,
        ),
        Some(parent_block_key) => {
            parent_block_key.for_child_block_with_causal_input(tail_tokens, tail_block_causal_input)
        }
    }
}

/// Extracts the tail span's full-attention KV tensors under one
/// performance-attribution span; `None` after logging a best-effort skip.
fn extract_tail_kv_block_tensors(
    active_request: &mut Qwen3_5EngineRequest,
    model: &Qwen3_5Model,
    tail_capture_span: &PromptTailCaptureSpan,
) -> Option<HashMap<String, MlxArray>> {
    let kv_block_tensors = match active_request.measure_operation_with_request(
        PerformanceOperation::PersistentPromptCacheStateExtraction,
        |active_request| {
            active_request
                .request_decoder_state
                .extract_persistent_prompt_cache_kv_block_tensors(
                    model.runtime(),
                    tail_capture_span.tail_block_start,
                    tail_capture_span.tail_capture_end,
                    tail_capture_span.persistent_prompt_cache_block_token_count,
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
            return None;
        }
    };
    Some(kv_block_tensors)
}

/// Extracts the gated-delta recurrent and convolution snapshots under one
/// performance-attribution span; `None` after logging a best-effort skip.
fn extract_tail_recurrent_snapshot_tensors(
    active_request: &mut Qwen3_5EngineRequest,
) -> Option<HashMap<String, MlxArray>> {
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
            return None;
        }
    };
    Some(recurrent_snapshot_tensors)
}

impl Qwen3_5EngineState {
    /// Captures the prompt's final partial block after the terminal prefill
    /// chunk, extending the last durable full block of this request's chain.
    /// A tail is a leaf: it is never recorded as a capture parent.
    pub(super) fn capture_persistent_prompt_cache_partial_tail_block(
        &self,
        persistent_prompt_cache: &PersistentPromptCacheDiskStore,
        model: &Qwen3_5Model,
        active_request: &mut Qwen3_5EngineRequest,
        prefill_end: usize,
    ) {
        let Some(tail_capture_span) = prompt_tail_capture_span(
            active_request.can_use_persistent_prompt_cache,
            active_request.input_token_ids.len(),
            persistent_prompt_cache.model_contract.block_token_count(),
            prefill_end,
        ) else {
            return;
        };
        let empty_block_causal_input = PersistentPromptCacheBlockCausalInput::empty();
        let tail_block_causal_input = match resolve_tail_block_causal_input(
            &active_request.persistent_prompt_cache_block_causal_inputs,
            tail_capture_span.tail_block_index,
        ) {
            TailBlockCausalInputResolution::Planned(planned_block_causal_input) => {
                planned_block_causal_input
            }
            TailBlockCausalInputResolution::EmptyFallback => &empty_block_causal_input,
            TailBlockCausalInputResolution::MissingFromPlan => {
                tracing::warn!(
                    request_id = active_request.request_id.value(),
                    tail_block_index = tail_capture_span.tail_block_index,
                    "prompt-cache causal input plan does not cover the partial tail; skipping tail capture"
                );
                return;
            }
        };
        let tail_tokens = &active_request.input_token_ids
            [tail_capture_span.tail_block_start..tail_capture_span.tail_capture_end];
        // The parent key is cloned so the chain ancestry outlives the mutable
        // request borrows taken by the extraction and publication steps below.
        let parent_block_key = active_request
            .last_restored_persistent_prompt_cache_block_key
            .clone();
        let Ok(tail_block_key) = build_tail_block_key(
            &persistent_prompt_cache.model_contract,
            tail_tokens,
            tail_block_causal_input,
            parent_block_key.as_ref(),
        ) else {
            tracing::warn!(
                request_id = active_request.request_id.value(),
                "prompt-cache partial tail identity construction failed; skipping tail capture"
            );
            return;
        };
        let Some(kv_block_tensors) =
            extract_tail_kv_block_tensors(active_request, model, &tail_capture_span)
        else {
            return;
        };
        let Some(recurrent_snapshot_tensors) =
            extract_tail_recurrent_snapshot_tensors(active_request)
        else {
            return;
        };
        self.publish_captured_tail_block(
            model,
            active_request,
            persistent_prompt_cache,
            &TailBlockPublicationInputs {
                tail_block_key: &tail_block_key,
                parent_block_key: parent_block_key.as_ref(),
                kv_block_tensors: &kv_block_tensors,
                recurrent_snapshot_tensors: &recurrent_snapshot_tensors,
            },
        );
    }

    /// Publishes one captured tail block with reclamation retry and records
    /// the outcome; every failure is logged and skipped because tail capture
    /// is best-effort by contract.
    fn publish_captured_tail_block(
        &self,
        model: &Qwen3_5Model,
        active_request: &mut Qwen3_5EngineRequest,
        persistent_prompt_cache: &PersistentPromptCacheDiskStore,
        tail_block_publication_inputs: &TailBlockPublicationInputs<'_>,
    ) {
        match self.publish_block_with_reclamation_retry(
            model,
            active_request,
            persistent_prompt_cache,
            tail_block_publication_inputs.tail_block_key,
            tail_block_publication_inputs.parent_block_key,
            tail_block_publication_inputs.kv_block_tensors,
            tail_block_publication_inputs.recurrent_snapshot_tensors,
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

#[cfg(test)]
#[path = "persistent_prompt_cache_tail_capture_tests.rs"]
mod persistent_prompt_cache_tail_capture_tests;
