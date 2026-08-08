/// The observable reason a request will or will not use draft-assisted prompt processing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Qwen3_5SpeculativePrefillRequestEligibility {
    Eligible,
    DisabledByConfiguration,
    DraftModelUnavailable,
    PromptBelowMinimum,
    PromptAlreadyRestored,
    PrecomputedVisualEmbeddingsPresent,
    DraftModelDoesNotSupportProcessedVisualImages,
}

impl Qwen3_5SpeculativePrefillRequestEligibility {
    #[must_use]
    pub(crate) const fn is_eligible(self) -> bool {
        matches!(self, Self::Eligible)
    }

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
    if !speculative_prefill_enabled {
        return Qwen3_5SpeculativePrefillRequestEligibility::DisabledByConfiguration;
    }
    if !draft_model_is_loaded {
        return Qwen3_5SpeculativePrefillRequestEligibility::DraftModelUnavailable;
    }
    if restored_target_prompt_prefix_token_count >= prompt_token_count.saturating_sub(1) {
        return Qwen3_5SpeculativePrefillRequestEligibility::PromptAlreadyRestored;
    }
    let remaining_prompt_suffix_token_count =
        prompt_token_count.saturating_sub(restored_target_prompt_prefix_token_count);
    if remaining_prompt_suffix_token_count < minimum_prompt_token_count {
        return Qwen3_5SpeculativePrefillRequestEligibility::PromptBelowMinimum;
    }
    if has_precomputed_visual_embeddings {
        return Qwen3_5SpeculativePrefillRequestEligibility::PrecomputedVisualEmbeddingsPresent;
    }
    if has_processed_visual_images && !draft_model_supports_processed_visual_images {
        return Qwen3_5SpeculativePrefillRequestEligibility::DraftModelDoesNotSupportProcessedVisualImages;
    }
    Qwen3_5SpeculativePrefillRequestEligibility::Eligible
}
