use super::AstronomicalConfigError;
use crate::config_file::SpeculativePrefillConfigFile;

/// Resolved user policy for optional draft-assisted speculative prefill.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpeculativePrefillConfig {
    enabled: bool,
    target_model_id: Option<String>,
    draft_model_id: Option<String>,
    minimum_prompt_tokens: u32,
    keep_percentage: u32,
    selection_chunck_token_count: u32,
    mandatory_trailing_token_count: u32,
    lookahead_token_count: u32,
    importance_pooling_kernel_token_count: u32,
}

impl SpeculativePrefillConfig {
    const DEFAULT_MINIMUM_PROMPT_TOKENS: u32 = 8_192;
    const DEFAULT_KEEP_PERCENTAGE: u32 = 20;
    const DEFAULT_SELECTION_CHUNCK_TOKEN_COUNT: u32 = 32;
    const DEFAULT_MANDATORY_TRAILING_TOKEN_COUNT: u32 = 512;
    const DEFAULT_LOOKAHEAD_TOKEN_COUNT: u32 = 8;
    const DEFAULT_IMPORTANCE_POOLING_KERNEL_TOKEN_COUNT: u32 = 13;

    /// Creates an explicitly resolved speculative-prefill policy.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        enabled: bool,
        target_model_id: Option<String>,
        draft_model_id: Option<String>,
        minimum_prompt_tokens: u32,
        keep_percentage: u32,
        selection_chunck_token_count: u32,
        mandatory_trailing_token_count: u32,
        lookahead_token_count: u32,
        importance_pooling_kernel_token_count: u32,
    ) -> Self {
        Self {
            enabled,
            target_model_id,
            draft_model_id,
            minimum_prompt_tokens,
            keep_percentage,
            selection_chunck_token_count,
            mandatory_trailing_token_count,
            lookahead_token_count,
            importance_pooling_kernel_token_count,
        }
    }

    /// Returns the safe default policy, which keeps the feature disabled.
    #[must_use]
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            target_model_id: None,
            draft_model_id: None,
            minimum_prompt_tokens: Self::DEFAULT_MINIMUM_PROMPT_TOKENS,
            keep_percentage: Self::DEFAULT_KEEP_PERCENTAGE,
            selection_chunck_token_count: Self::DEFAULT_SELECTION_CHUNCK_TOKEN_COUNT,
            mandatory_trailing_token_count: Self::DEFAULT_MANDATORY_TRAILING_TOKEN_COUNT,
            lookahead_token_count: Self::DEFAULT_LOOKAHEAD_TOKEN_COUNT,
            importance_pooling_kernel_token_count:
                Self::DEFAULT_IMPORTANCE_POOLING_KERNEL_TOKEN_COUNT,
        }
    }

    /// Returns whether draft-assisted speculative prefill is enabled.
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Returns the exact target model identity bound to speculative prefill.
    #[must_use]
    pub fn target_model_id(&self) -> Option<&str> {
        self.target_model_id.as_deref()
    }

    /// Returns the configured exact draft model identity, when present.
    #[must_use]
    pub fn draft_model_id(&self) -> Option<&str> {
        self.draft_model_id.as_deref()
    }

    /// Returns the minimum prompt length required for admission.
    #[must_use]
    pub const fn minimum_prompt_tokens(&self) -> u32 {
        self.minimum_prompt_tokens
    }

    /// Returns the percentage of scored prompt tokens retained by selection.
    #[must_use]
    pub const fn keep_percentage(&self) -> u32 {
        self.keep_percentage
    }

    /// Returns the number of tokens grouped into one selection chunk.
    #[must_use]
    pub const fn selection_chunck_token_count(&self) -> u32 {
        self.selection_chunck_token_count
    }

    /// Returns the trailing token count that selection must always retain.
    #[must_use]
    pub const fn mandatory_trailing_token_count(&self) -> u32 {
        self.mandatory_trailing_token_count
    }

    /// Returns the number of draft lookahead tokens used for importance scoring.
    #[must_use]
    pub const fn lookahead_token_count(&self) -> u32 {
        self.lookahead_token_count
    }

    /// Returns the smoothing kernel width used before chunk ranking.
    #[must_use]
    pub const fn importance_pooling_kernel_token_count(&self) -> u32 {
        self.importance_pooling_kernel_token_count
    }
}

pub(crate) fn resolve_speculative_prefill_config(
    configured_speculative_prefill: &SpeculativePrefillConfigFile,
) -> Result<SpeculativePrefillConfig, AstronomicalConfigError> {
    let enabled = configured_speculative_prefill.enabled.unwrap_or(false);
    let target_model_id = configured_speculative_prefill
        .target_model_id
        .as_ref()
        .map(|configured_target_model_id| configured_target_model_id.trim().to_owned());
    if target_model_id.as_deref() == Some("") {
        return Err(AstronomicalConfigError::SpeculativePrefillTargetModelIdMustNotBeEmpty);
    }
    if enabled && target_model_id.is_none() {
        return Err(AstronomicalConfigError::SpeculativePrefillTargetModelRequired);
    }
    let draft_model_id = configured_speculative_prefill
        .draft_model_id
        .as_ref()
        .map(|configured_draft_model_id| configured_draft_model_id.trim().to_owned());
    if draft_model_id.as_deref() == Some("") {
        return Err(AstronomicalConfigError::SpeculativePrefillDraftModelIdMustNotBeEmpty);
    }
    if enabled && draft_model_id.is_none() {
        return Err(AstronomicalConfigError::SpeculativePrefillDraftModelRequired);
    }

    let minimum_prompt_tokens = configured_speculative_prefill
        .minimum_prompt_tokens
        .unwrap_or(SpeculativePrefillConfig::DEFAULT_MINIMUM_PROMPT_TOKENS);
    if minimum_prompt_tokens == 0 {
        return Err(AstronomicalConfigError::SpeculativePrefillMinimumPromptTokensMustBePositive);
    }

    let keep_percentage = configured_speculative_prefill
        .keep_percentage
        .unwrap_or(SpeculativePrefillConfig::DEFAULT_KEEP_PERCENTAGE);
    if !(1..=100).contains(&keep_percentage) {
        return Err(AstronomicalConfigError::SpeculativePrefillKeepPercentageOutOfRange);
    }

    let selection_chunck_token_count = configured_speculative_prefill
        .selection_chunck_token_count
        .unwrap_or(SpeculativePrefillConfig::DEFAULT_SELECTION_CHUNCK_TOKEN_COUNT);
    if selection_chunck_token_count == 0 {
        return Err(
            AstronomicalConfigError::SpeculativePrefillSelectionChunckTokenCountMustBePositive,
        );
    }

    let mandatory_trailing_token_count = configured_speculative_prefill
        .mandatory_trailing_token_count
        .unwrap_or(SpeculativePrefillConfig::DEFAULT_MANDATORY_TRAILING_TOKEN_COUNT);
    if mandatory_trailing_token_count == 0 {
        return Err(
            AstronomicalConfigError::SpeculativePrefillMandatoryTrailingTokenCountMustBePositive,
        );
    }

    let lookahead_token_count = configured_speculative_prefill
        .lookahead_token_count
        .unwrap_or(SpeculativePrefillConfig::DEFAULT_LOOKAHEAD_TOKEN_COUNT);
    if lookahead_token_count == 0 {
        return Err(AstronomicalConfigError::SpeculativePrefillLookaheadTokenCountMustBePositive);
    }

    let importance_pooling_kernel_token_count = configured_speculative_prefill
        .importance_pooling_kernel_token_count
        .unwrap_or(SpeculativePrefillConfig::DEFAULT_IMPORTANCE_POOLING_KERNEL_TOKEN_COUNT);
    if importance_pooling_kernel_token_count == 0 {
        return Err(
            AstronomicalConfigError::SpeculativePrefillImportancePoolingKernelTokenCountMustBePositive,
        );
    }

    Ok(SpeculativePrefillConfig::new(
        enabled,
        target_model_id,
        draft_model_id,
        minimum_prompt_tokens,
        keep_percentage,
        selection_chunck_token_count,
        mandatory_trailing_token_count,
        lookahead_token_count,
        importance_pooling_kernel_token_count,
    ))
}
