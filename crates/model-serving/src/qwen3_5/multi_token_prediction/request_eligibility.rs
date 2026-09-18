//! Request-local eligibility for optional Qwen multi-token prediction.
//!
//! Configuration stays off until a model sets `acceleration.mtp.enabled` to
//! true. SSD-paged experts, vision, and a persistent-prompt-cache *restore*
//! that already replaced prompt work stay target-only even after that opt-in:
//! cache reuse and the decode-side head both live on the same prefill stream,
//! while the head never benefits from tokens the cache served wholesale.
//! Both sampling strategies are eligible: the greedy path verifies against
//! argmax tokens, and sampled requests verify through `min(1, p/q)` acceptance
//! with residual correction.

/// Returns whether this request may open an optional multi-token-prediction session.
#[must_use]
pub fn qwen3_5_mtp_request_is_eligible(
    mtp_enabled: bool,
    mtp_runtime_is_active: bool,
    model_has_mtp_weights: bool,
    has_precomputed_visual_embeddings: bool,
    has_processed_visual_images: bool,
    prompt_cache_restored_prompt_token_count: u32,
    prompt_token_count: usize,
    sparse_experts_are_paged: bool,
) -> bool {
    mtp_enabled
        && mtp_runtime_is_active
        && model_has_mtp_weights
        && !has_precomputed_visual_embeddings
        && !has_processed_visual_images
        && !sparse_experts_are_paged
        && usize::try_from(prompt_cache_restored_prompt_token_count).is_ok_and(
            |restored_token_count| restored_token_count < prompt_token_count.saturating_sub(1),
        )
        && prompt_token_count > 1
}
