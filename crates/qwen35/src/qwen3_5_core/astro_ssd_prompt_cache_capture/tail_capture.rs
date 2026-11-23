//! Best-effort capture of the prompt's final partial block ("tail").
//!
//! A prompt whose token count is not a multiple of the block size leaves its
//! final tokens outside every published full block. Persisting that tail lets
//! the next turn restore everything except the final prompt token, which the
//! terminal forward always processes to produce the first logits. Tail capture
//! is best-effort by contract: the request never depends on tail reuse, so
//! every failure is logged and skipped.

use engine_core::astro_ssd_prompt_cache::tail_capture_planning::{
    PromptTailCaptureSpan, TailBlockCausalInputResolution, build_tail_block_key,
    prompt_tail_capture_span, resolve_tail_block_causal_input,
};
use std::collections::HashMap;

use mlx_bridge::MlxArray;

use crate::{
    AstroSsdPromptCacheBlockKey, AstroSsdPromptCacheDiskStore,
    AstroSsdPromptCachePublicationOutcome, PerformanceOperation,
};
use engine_core::astro_ssd_prompt_cache::block_causal_input::AstroSsdPromptCacheBlockCausalInput;

use super::block_capture::{PromptBlockPublicationFailure, publish_block_with_reclamation_retry};
use super::capture_hook::Qwen3_5AstroSsdPromptCacheCapture;
use crate::qwen3_5_core::engine_request::Qwen3_5EngineRequest;
use crate::qwen3_5_core::model::Qwen3_5ModelBase;

/// Everything publication needs about one captured tail block, grouped so the
/// publication step keeps a short signature.
struct TailBlockPublicationInputs<'a> {
    tail_block_key: &'a AstroSsdPromptCacheBlockKey,
    parent_block_key: Option<&'a AstroSsdPromptCacheBlockKey>,
    kv_block_tensors: &'a HashMap<String, MlxArray>,
    recurrent_snapshot_tensors: &'a HashMap<String, MlxArray>,
}

/// Extracts the tail span's full-attention KV tensors under one
/// performance-attribution span; `None` after logging a best-effort skip.
fn extract_tail_kv_block_tensors<RequestExtension>(
    active_request: &mut Qwen3_5EngineRequest<RequestExtension>,
    model: &Qwen3_5ModelBase,
    tail_capture_span: &PromptTailCaptureSpan,
) -> Option<HashMap<String, MlxArray>> {
    let kv_block_tensors = match active_request.measure_operation_with_request(
        PerformanceOperation::AstroSsdPromptCacheStateExtraction,
        |active_request| {
            active_request
                .request_decoder_state
                .extract_astro_ssd_prompt_cache_kv_block_tensors(
                    model.runtime(),
                    tail_capture_span.tail_block_start,
                    tail_capture_span.tail_capture_end,
                    tail_capture_span.astro_ssd_prompt_cache_block_token_count,
                )
        },
    ) {
        Ok(kv_block_tensors) => kv_block_tensors,
        Err(error) => {
            tracing::warn!(
                request_id = active_request.request_id.value(),
                %error,
                "astro-ssd-prompt-cache partial tail KV extraction failed; skipping tail capture"
            );
            return None;
        }
    };
    Some(kv_block_tensors)
}

/// Extracts the gated-delta recurrent and convolution snapshots under one
/// performance-attribution span; `None` after logging a best-effort skip.
fn extract_tail_recurrent_snapshot_tensors<RequestExtension>(
    active_request: &mut Qwen3_5EngineRequest<RequestExtension>,
) -> Option<HashMap<String, MlxArray>> {
    let recurrent_snapshot_tensors = match active_request.measure_operation_with_request(
        PerformanceOperation::AstroSsdPromptCacheStateExtraction,
        |active_request| {
            active_request
                .request_decoder_state
                .extract_astro_ssd_prompt_cache_recurrent_snapshot_tensors()
        },
    ) {
        Ok(recurrent_snapshot_tensors) => recurrent_snapshot_tensors,
        Err(error) => {
            tracing::warn!(
                request_id = active_request.request_id.value(),
                %error,
                "astro-ssd-prompt-cache partial tail snapshot extraction failed; skipping tail capture"
            );
            return None;
        }
    };
    Some(recurrent_snapshot_tensors)
}

/// Captures the prompt's final partial block after the terminal prefill
/// chunk, extending the last durable full block of this request's chain.
/// A tail is a leaf: it is never recorded as a capture parent.
pub fn capture_astro_ssd_prompt_cache_partial_tail_block<
    CaptureEngine: Qwen3_5AstroSsdPromptCacheCapture,
>(
    capture_engine: &CaptureEngine,
    astro_ssd_prompt_cache: &AstroSsdPromptCacheDiskStore,
    model: &CaptureEngine::Model,
    active_request: &mut Qwen3_5EngineRequest<CaptureEngine::RequestExtension>,
    prefill_end: usize,
) {
    let Some(tail_capture_span) = prompt_tail_capture_span(
        active_request.can_use_astro_ssd_prompt_cache,
        active_request.input_token_ids.len(),
        astro_ssd_prompt_cache.model_contract.block_token_count(),
        prefill_end,
    ) else {
        return;
    };
    let empty_block_causal_input = AstroSsdPromptCacheBlockCausalInput::empty();
    let tail_block_causal_input = match resolve_tail_block_causal_input(
        &active_request.astro_ssd_prompt_cache_block_causal_inputs,
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
                "astro-ssd-prompt-cache causal input plan does not cover the partial tail; skipping tail capture"
            );
            return;
        }
    };
    let tail_tokens = &active_request.input_token_ids
        [tail_capture_span.tail_block_start..tail_capture_span.tail_capture_end];
    // The parent key is cloned so the chain ancestry outlives the mutable
    // request borrows taken by the extraction and publication steps below.
    let parent_block_key = active_request
        .last_restored_astro_ssd_prompt_cache_block_key
        .clone();
    let Ok(tail_block_key) = build_tail_block_key(
        &astro_ssd_prompt_cache.model_contract,
        tail_tokens,
        tail_block_causal_input,
        parent_block_key.as_ref(),
    ) else {
        tracing::warn!(
            request_id = active_request.request_id.value(),
            "astro-ssd-prompt-cache partial tail identity construction failed; skipping tail capture"
        );
        return;
    };
    let Some(kv_block_tensors) =
        extract_tail_kv_block_tensors(active_request, model, &tail_capture_span)
    else {
        return;
    };
    let Some(recurrent_snapshot_tensors) = extract_tail_recurrent_snapshot_tensors(active_request)
    else {
        return;
    };
    publish_captured_tail_block(
        capture_engine,
        model,
        active_request,
        astro_ssd_prompt_cache,
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
fn publish_captured_tail_block<CaptureEngine: Qwen3_5AstroSsdPromptCacheCapture>(
    capture_engine: &CaptureEngine,
    model: &CaptureEngine::Model,
    active_request: &mut Qwen3_5EngineRequest<CaptureEngine::RequestExtension>,
    astro_ssd_prompt_cache: &AstroSsdPromptCacheDiskStore,
    tail_block_publication_inputs: &TailBlockPublicationInputs<'_>,
) {
    match publish_block_with_reclamation_retry(
        capture_engine,
        model,
        active_request,
        astro_ssd_prompt_cache,
        tail_block_publication_inputs.tail_block_key,
        tail_block_publication_inputs.parent_block_key,
        tail_block_publication_inputs.kv_block_tensors,
        tail_block_publication_inputs.recurrent_snapshot_tensors,
    ) {
        Ok(publication_outcome) => {
            if publication_outcome == AstroSsdPromptCachePublicationOutcome::Published
                && let Some(astro_ssd_prompt_cache_diagnostics) =
                    active_request.astro_ssd_prompt_cache_diagnostics.as_mut()
            {
                astro_ssd_prompt_cache_diagnostics.record_published_block();
            }
        }
        Err(PromptBlockPublicationFailure::Reclamation(error)) => {
            tracing::warn!(
                request_id = active_request.request_id.value(),
                %error,
                "astro-ssd-prompt-cache partial tail publication reclamation failed; skipping tail capture"
            );
        }
        Err(PromptBlockPublicationFailure::Publication(error)) => {
            tracing::warn!(
                request_id = active_request.request_id.value(),
                %error,
                "astro-ssd-prompt-cache partial tail publication failed; skipping tail capture"
            );
        }
    }
}
