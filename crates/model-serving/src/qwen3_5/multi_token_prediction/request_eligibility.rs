//! Request-local eligibility for optional Qwen multi-token prediction.
//!
//! Configuration stays off until a model sets `acceleration.mtp.enabled` to
//! true. SSD-paged experts, sampling, vision, and a live persistent prompt
//! cache still stay target-only even after that opt-in.

/// Returns whether this request may open an optional multi-token-prediction session.
#[must_use]
pub fn qwen3_5_mtp_request_is_eligible(
    mtp_enabled: bool,
    mtp_runtime_is_active: bool,
    model_has_mtp_weights: bool,
    sampling_selects_highest_logit: bool,
    has_precomputed_visual_embeddings: bool,
    has_processed_visual_images: bool,
    persistent_prompt_cache_is_available: bool,
    sparse_experts_are_paged: bool,
    prompt_token_count: usize,
    restored_prompt_token_count: u32,
) -> bool {
    mtp_enabled
        && mtp_runtime_is_active
        && model_has_mtp_weights
        && sampling_selects_highest_logit
        && !has_precomputed_visual_embeddings
        && !has_processed_visual_images
        && !persistent_prompt_cache_is_available
        && !sparse_experts_are_paged
        && usize::try_from(restored_prompt_token_count).is_ok_and(|restored_token_count| {
            restored_token_count < prompt_token_count.saturating_sub(1)
        })
        && prompt_token_count > 1
}
