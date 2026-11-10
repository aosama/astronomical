//! Pure planning for the partial-tail prompt-cache capture, shared by the
//! streaming and resident engines: which tail span a terminal prefill chunk
//! can capture, which causal input the tail block binds, and the tail block's
//! identity. Extraction and publication of the tail live in
//! `qwen3_5_core::prompt_cache_capture::tail_capture`.

use crate::persistent_cache::block_causal_input::PersistentPromptCacheBlockCausalInput;
use crate::persistent_cache::block_key::{
    PersistentPromptCacheBlockKey, PersistentPromptCacheBlockKeyError,
};
use crate::persistent_cache::model_contract::PersistentPromptCacheModelContract;

/// Token geometry of the partial tail one terminal prefill chunk can capture.
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct PromptTailCaptureSpan {
    pub(crate) tail_capture_end: usize,
    pub(crate) tail_block_start: usize,
    pub(crate) tail_block_index: usize,
    pub(crate) persistent_prompt_cache_block_token_count: usize,
}

/// Outcome of resolving the causal input bound into the tail block identity.
#[derive(Debug)]
pub(crate) enum TailBlockCausalInputResolution<'a> {
    Planned(&'a PersistentPromptCacheBlockCausalInput),
    EmptyFallback,
    MissingFromPlan,
}

/// Computes the token span of the partial tail a terminal prefill chunk can
/// capture, or `None` when this chunk ends anywhere but the capture boundary.
pub(crate) fn prompt_tail_capture_span(
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
pub(crate) fn resolve_tail_block_causal_input<'a>(
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
pub(crate) fn build_tail_block_key(
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

#[cfg(test)]
#[path = "tail_capture_planning_tests.rs"]
mod tail_capture_planning_tests;
