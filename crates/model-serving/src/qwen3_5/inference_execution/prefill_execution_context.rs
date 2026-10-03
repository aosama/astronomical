const VISUAL_PREFILL_CONTEXT_FLAG: u64 = 1 << 34;
const ADDITIONAL_CONTEXT_STATE_PREFILL_CONTEXT_FLAG: u64 = 1 << 35;
const PAGED_EXPERTS_CONTEXT_FLAG: u64 = 1 << 36;
const PROMPT_CACHE_CAPTURE_ELIGIBLE_CONTEXT_FLAG: u64 = 1 << 37;

/// Execution modes that partition adaptive memory-growth observations.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Qwen3_5PrefillExecutionContext {
    has_visual_embeddings: bool,
    has_optional_prediction_session: bool,
    are_sparse_experts_paged: bool,
    is_prompt_cache_capture_eligible: bool,
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
        }
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
        context_identifier_flags
    }
}
