const VISUAL_PREFILL_CONTEXT_FLAG: u64 = 1 << 34;
const ADDITIONAL_CONTEXT_STATE_PREFILL_CONTEXT_FLAG: u64 = 1 << 35;
const PAGED_EXPERTS_CONTEXT_FLAG: u64 = 1 << 36;
const PROMPT_CACHE_CAPTURE_ELIGIBLE_CONTEXT_FLAG: u64 = 1 << 37;
pub(super) const CAPACITY_REDUCED_CONTEXT_FLAG: u64 = 1 << 38;
pub(super) const SPECULATIVE_PREFILL_TARGET_ONLY_PREFIX_CONTEXT_FLAG: u64 = 1 << 39;
pub(super) const SPECULATIVE_PREFILL_SPARSE_TARGET_CONTEXT_FLAG: u64 = 1 << 40;

/// Known execution modes that materially change prompt-processing latency.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Qwen3_5PrefillExecutionContext {
    has_visual_embeddings: bool,
    has_optional_prediction_session: bool,
    are_sparse_experts_paged: bool,
    is_prompt_cache_capture_eligible: bool,
    is_target_only_prefix: bool,
    is_speculative_prefill_sparse_target: bool,
}

impl Qwen3_5PrefillExecutionContext {
    #[must_use]
    pub const fn new(
        has_visual_embeddings: bool,
        has_optional_prediction_session: bool,
        are_sparse_experts_paged: bool,
        is_prompt_cache_capture_eligible: bool,
    ) -> Self {
        Self {
            has_visual_embeddings,
            has_optional_prediction_session,
            are_sparse_experts_paged,
            is_prompt_cache_capture_eligible,
            is_target_only_prefix: false,
            is_speculative_prefill_sparse_target: false,
        }
    }

    pub(super) const fn with_target_only_prefix(mut self, is_target_only_prefix: bool) -> Self {
        self.is_target_only_prefix = is_target_only_prefix;
        self
    }

    pub(super) const fn with_speculative_prefill_sparse_target(
        mut self,
        is_speculative_prefill_sparse_target: bool,
    ) -> Self {
        self.is_speculative_prefill_sparse_target = is_speculative_prefill_sparse_target;
        self
    }

    pub(super) const fn context_identifier_flags(self) -> u64 {
        let mut context_identifier_flags = 0;
        if self.has_visual_embeddings {
            context_identifier_flags |= VISUAL_PREFILL_CONTEXT_FLAG;
        }
        if self.has_optional_prediction_session {
            context_identifier_flags |= ADDITIONAL_CONTEXT_STATE_PREFILL_CONTEXT_FLAG;
        }
        if self.are_sparse_experts_paged {
            context_identifier_flags |= PAGED_EXPERTS_CONTEXT_FLAG;
        }
        if self.is_prompt_cache_capture_eligible {
            context_identifier_flags |= PROMPT_CACHE_CAPTURE_ELIGIBLE_CONTEXT_FLAG;
        }
        if self.is_target_only_prefix {
            context_identifier_flags |= SPECULATIVE_PREFILL_TARGET_ONLY_PREFIX_CONTEXT_FLAG;
        }
        if self.is_speculative_prefill_sparse_target {
            context_identifier_flags |= SPECULATIVE_PREFILL_SPARSE_TARGET_CONTEXT_FLAG;
        }
        context_identifier_flags
    }

    #[must_use]
    pub const fn has_visual_embeddings(self) -> bool {
        self.has_visual_embeddings
    }

    #[must_use]
    pub const fn has_optional_prediction_session(self) -> bool {
        self.has_optional_prediction_session
    }

    #[must_use]
    pub const fn are_sparse_experts_paged(self) -> bool {
        self.are_sparse_experts_paged
    }

    #[must_use]
    pub const fn is_prompt_cache_capture_eligible(self) -> bool {
        self.is_prompt_cache_capture_eligible
    }

    #[must_use]
    pub(super) const fn is_speculative_prefill_sparse_target(self) -> bool {
        self.is_speculative_prefill_sparse_target
    }
}
