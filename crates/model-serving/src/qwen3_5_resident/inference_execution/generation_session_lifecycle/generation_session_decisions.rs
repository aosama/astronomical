//! Pure admission decisions for one Qwen3.5 generation session.

use astronomical_ipc_protocol::WorkerPromptProcessingPhase;

/// Persistent snapshots contain target decoder state only, so target-only
/// execution is preferred whenever the cache is available; precomputed visual
/// embeddings would restore an incompatible shifted history and disable reuse.
pub(super) fn can_use_persistent_prompt_cache(
    persistent_prompt_cache_is_available: bool,
    request_has_precomputed_visual_embeddings: bool,
) -> bool {
    persistent_prompt_cache_is_available && !request_has_precomputed_visual_embeddings
}

/// Returns the Target prefill phase while prompt tokens remain beyond the
/// restored prefix; a fully restored prompt skips prefill entirely and starts
/// directly at decoding.
pub(super) fn initial_prompt_processing_phase(
    prefill_cursor: usize,
    prompt_prefill_end_exclusive: usize,
) -> Option<WorkerPromptProcessingPhase> {
    (prefill_cursor < prompt_prefill_end_exclusive).then_some(WorkerPromptProcessingPhase::Target)
}

#[cfg(test)]
#[path = "generation_session_decisions_tests.rs"]
mod generation_session_decisions_tests;
