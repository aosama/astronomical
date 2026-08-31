//! Exposes resolved per-target speculative-prefill policy while retaining internal tuning defaults.

/// Resolved user policy for optional draft-assisted speculative prefill.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpeculativePrefillConfig {
    target_model_id: String,
    draft_model_id: String,
    minimum_prompt_tokens: u32,
    keep_percentage: u32,
    selection_chunk_token_count: u32,
    mandatory_trailing_token_count: u32,
    lookahead_token_count: u32,
    importance_pooling_kernel_token_count: u32,
}

impl SpeculativePrefillConfig {
    pub(crate) const DEFAULT_MINIMUM_PROMPT_TOKENS: u32 = 8_192;
    pub(crate) const DEFAULT_KEEP_PERCENTAGE: u32 = 20;
    pub(crate) const DEFAULT_SELECTION_CHUNK_TOKEN_COUNT: u32 = 32;
    pub(crate) const DEFAULT_MANDATORY_TRAILING_TOKEN_COUNT: u32 = 512;
    pub(crate) const DEFAULT_LOOKAHEAD_TOKEN_COUNT: u32 = 8;
    pub(crate) const DEFAULT_IMPORTANCE_POOLING_KERNEL_TOKEN_COUNT: u32 = 13;

    pub(crate) fn for_target(
        target_model_id: &str,
        draft_model_id: &str,
        minimum_prompt_tokens: Option<u32>,
        keep_percentage: Option<u32>,
    ) -> Self {
        Self {
            target_model_id: target_model_id.to_owned(),
            draft_model_id: draft_model_id.to_owned(),
            minimum_prompt_tokens: minimum_prompt_tokens
                .unwrap_or(Self::DEFAULT_MINIMUM_PROMPT_TOKENS),
            keep_percentage: keep_percentage.unwrap_or(Self::DEFAULT_KEEP_PERCENTAGE),
            selection_chunk_token_count: Self::DEFAULT_SELECTION_CHUNK_TOKEN_COUNT,
            mandatory_trailing_token_count: Self::DEFAULT_MANDATORY_TRAILING_TOKEN_COUNT,
            lookahead_token_count: Self::DEFAULT_LOOKAHEAD_TOKEN_COUNT,
            importance_pooling_kernel_token_count:
                Self::DEFAULT_IMPORTANCE_POOLING_KERNEL_TOKEN_COUNT,
        }
    }

    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        true
    }

    #[must_use]
    pub fn target_model_id(&self) -> Option<&str> {
        Some(&self.target_model_id)
    }

    #[must_use]
    pub fn draft_model_id(&self) -> Option<&str> {
        Some(&self.draft_model_id)
    }

    #[must_use]
    pub const fn minimum_prompt_tokens(&self) -> u32 {
        self.minimum_prompt_tokens
    }

    #[must_use]
    pub const fn keep_percentage(&self) -> u32 {
        self.keep_percentage
    }

    #[must_use]
    pub const fn selection_chunk_token_count(&self) -> u32 {
        self.selection_chunk_token_count
    }

    #[must_use]
    pub const fn mandatory_trailing_token_count(&self) -> u32 {
        self.mandatory_trailing_token_count
    }

    #[must_use]
    pub const fn lookahead_token_count(&self) -> u32 {
        self.lookahead_token_count
    }

    #[must_use]
    pub const fn importance_pooling_kernel_token_count(&self) -> u32 {
        self.importance_pooling_kernel_token_count
    }
}
