//! Pure request-level eligibility policy for draft-assisted prompt processing.
//!
//! Model loading determines whether a compatible drafter exists. This module
//! applies the remaining request facts after target-state restoration, when the
//! engine finally knows how much prompt work is actually left.

/// The observable reason a request will or will not use draft-assisted prompt processing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Qwen3_5SpeculativePrefillRequestEligibility {
    /// Every model-level and request-level precondition is satisfied.
    Eligible,
    /// The user did not enable SpecPrefill for the worker.
    DisabledByConfiguration,
    /// Configuration enabled the policy but startup could not retain a compatible drafter.
    DraftModelUnavailable,
    /// Too little uncached prompt remains for drafter scoring to justify its setup cost.
    PromptBelowMinimum,
    /// Reusable target state already covers all tokens except generation kickoff.
    PromptAlreadyRestored,
    /// The caller supplied target-width embeddings that cannot be reused by another model.
    PrecomputedVisualEmbeddingsPresent,
    /// The request has images but the drafter cannot consume the target's processed pixels.
    DraftModelDoesNotSupportProcessedVisualImages,
}

impl Qwen3_5SpeculativePrefillRequestEligibility {
    #[must_use]
    pub(crate) const fn is_eligible(self) -> bool {
        // Keep call sites declarative and prevent them from duplicating the
        // growing list of ineligible reasons.
        matches!(self, Self::Eligible)
    }

    /// Stable telemetry identifier; changing a Rust variant must not silently
    /// alter the externally inspected log vocabulary.
    #[must_use]
    pub(crate) const fn identifier(self) -> &'static str {
        match self {
            Self::Eligible => "eligible",
            Self::DisabledByConfiguration => "disabled_by_configuration",
            Self::DraftModelUnavailable => "draft_model_unavailable",
            Self::PromptBelowMinimum => "prompt_below_minimum",
            Self::PromptAlreadyRestored => "prompt_already_restored",
            Self::PrecomputedVisualEmbeddingsPresent => "precomputed_visual_embeddings_present",
            Self::DraftModelDoesNotSupportProcessedVisualImages => {
                "draft_model_does_not_support_processed_visual_images"
            }
        }
    }
}

/// Evaluates request-specific conditions after target persistent-prefix restoration.
///
/// Checks are intentionally ordered. Configuration and model availability are
/// reported before request shape, an already-complete target restore wins over a
/// short-suffix report, and visual constraints are considered only when useful
/// uncached text remains.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub(crate) const fn qwen3_5_speculative_prefill_request_eligibility(
    speculative_prefill_enabled: bool,
    draft_model_is_loaded: bool,
    draft_model_supports_processed_visual_images: bool,
    prompt_token_count: usize,
    minimum_prompt_token_count: usize,
    restored_target_prompt_prefix_token_count: usize,
    has_precomputed_visual_embeddings: bool,
    has_processed_visual_images: bool,
) -> Qwen3_5SpeculativePrefillRequestEligibility {
    // Disabled is a normal policy choice, not a loading or request failure.
    if !speculative_prefill_enabled {
        return Qwen3_5SpeculativePrefillRequestEligibility::DisabledByConfiguration;
    }
    // Enabled-but-unavailable is kept distinct because the configured policy is
    // fail-closed at the caller rather than silently becoming target-only.
    if !draft_model_is_loaded {
        return Qwen3_5SpeculativePrefillRequestEligibility::DraftModelUnavailable;
    }
    // The last prompt token belongs to generation startup. If restoration has
    // reached it, there is no selectable suffix left for SpecPrefill.
    if restored_target_prompt_prefix_token_count >= prompt_token_count.saturating_sub(1) {
        return Qwen3_5SpeculativePrefillRequestEligibility::PromptAlreadyRestored;
    }
    let remaining_prompt_suffix_token_count =
        prompt_token_count.saturating_sub(restored_target_prompt_prefix_token_count);
    // Evaluate the threshold against remaining work, not the original prompt;
    // otherwise a large cached prompt could trigger a costly drafter for a tiny suffix.
    if remaining_prompt_suffix_token_count < minimum_prompt_token_count {
        return Qwen3_5SpeculativePrefillRequestEligibility::PromptBelowMinimum;
    }
    // Precomputed embeddings have the target hidden width. The drafter may have
    // a different width, so processed pixels—not target embeddings—are required.
    if has_precomputed_visual_embeddings {
        return Qwen3_5SpeculativePrefillRequestEligibility::PrecomputedVisualEmbeddingsPresent;
    }
    // Processed pixels are shareable only when startup validated compatible
    // vision geometry between the target and drafter towers.
    if has_processed_visual_images && !draft_model_supports_processed_visual_images {
        return Qwen3_5SpeculativePrefillRequestEligibility::DraftModelDoesNotSupportProcessedVisualImages;
    }
    Qwen3_5SpeculativePrefillRequestEligibility::Eligible
}
