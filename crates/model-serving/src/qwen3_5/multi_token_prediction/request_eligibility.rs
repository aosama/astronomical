//! Pure request-level MTP eligibility and bounded target-only reasons.

/// One stable explanation for whether a loaded MTP runtime may serve a request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen3_5MtpRequestEligibility {
    Eligible,
    DisabledByConfiguration,
    RuntimeNotActive,
    WeightsUnavailable,
    SampledDecoding,
    PrecomputedVisionInput,
    ProcessedVisionInput,
    PersistentPromptCacheAvailable,
    PromptTooShort,
    InsufficientUncachedPromptHistory,
}

impl Qwen3_5MtpRequestEligibility {
    /// Returns a bounded identifier safe for status, logs, and attribution.
    #[must_use]
    pub const fn identifier(self) -> &'static str {
        match self {
            Self::Eligible => "eligible",
            Self::DisabledByConfiguration => "disabled_by_configuration",
            Self::RuntimeNotActive => "runtime_not_active",
            Self::WeightsUnavailable => "weights_unavailable",
            Self::SampledDecoding => "sampled_decoding",
            Self::PrecomputedVisionInput => "precomputed_vision_input",
            Self::ProcessedVisionInput => "processed_vision_input",
            Self::PersistentPromptCacheAvailable => "persistent_prompt_cache_available",
            Self::PromptTooShort => "prompt_too_short",
            Self::InsufficientUncachedPromptHistory => "insufficient_uncached_prompt_history",
        }
    }

    #[must_use]
    pub const fn is_eligible(self) -> bool {
        matches!(self, Self::Eligible)
    }
}

/// Complete request evidence evaluated before allocating private MTP state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen3_5MtpRequestEligibilityInputs {
    pub mtp_enabled: bool,
    pub mtp_runtime_is_active: bool,
    pub model_has_mtp_weights: bool,
    pub sampling_is_greedy: bool,
    pub has_precomputed_visual_embeddings: bool,
    pub has_processed_visual_images: bool,
    pub persistent_prompt_cache_is_available: bool,
    pub prompt_token_count: usize,
    pub restored_prompt_token_count: u32,
}

/// Selects one deterministic reason instead of silently treating every fallback alike.
#[must_use]
pub fn qwen3_5_mtp_request_eligibility(
    inputs: Qwen3_5MtpRequestEligibilityInputs,
) -> Qwen3_5MtpRequestEligibility {
    if !inputs.mtp_enabled {
        return Qwen3_5MtpRequestEligibility::DisabledByConfiguration;
    }
    if !inputs.mtp_runtime_is_active {
        return Qwen3_5MtpRequestEligibility::RuntimeNotActive;
    }
    if !inputs.model_has_mtp_weights {
        return Qwen3_5MtpRequestEligibility::WeightsUnavailable;
    }
    if !inputs.sampling_is_greedy {
        return Qwen3_5MtpRequestEligibility::SampledDecoding;
    }
    if inputs.has_precomputed_visual_embeddings {
        return Qwen3_5MtpRequestEligibility::PrecomputedVisionInput;
    }
    if inputs.has_processed_visual_images {
        return Qwen3_5MtpRequestEligibility::ProcessedVisionInput;
    }
    if inputs.persistent_prompt_cache_is_available {
        return Qwen3_5MtpRequestEligibility::PersistentPromptCacheAvailable;
    }
    if inputs.prompt_token_count <= 1 {
        return Qwen3_5MtpRequestEligibility::PromptTooShort;
    }
    if usize::try_from(inputs.restored_prompt_token_count).map_or(
        true,
        |restored_prompt_token_count| {
            restored_prompt_token_count >= inputs.prompt_token_count.saturating_sub(1)
        },
    ) {
        return Qwen3_5MtpRequestEligibility::InsufficientUncachedPromptHistory;
    }
    Qwen3_5MtpRequestEligibility::Eligible
}
