use crate::AdaptiveRamGrowthExecutionProfile;

const VISUAL_PREFILL_CONTEXT_FLAG: u64 = 1 << 34;
const PAGED_EXPERTS_CONTEXT_FLAG: u64 = 1 << 36;
const PROMPT_CACHE_CAPTURE_ELIGIBLE_CONTEXT_FLAG: u64 = 1 << 37;

/// Execution profiles that partition adaptive memory-growth observations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen3_5PrefillExecutionContext {
    has_visual_embeddings: bool,
    execution_profile: AdaptiveRamGrowthExecutionProfile,
    is_prompt_cache_capture_eligible: bool,
}

impl Qwen3_5PrefillExecutionContext {
    #[must_use]
    pub const fn new(
        has_visual_embeddings: bool,
        execution_profile: AdaptiveRamGrowthExecutionProfile,
        is_prompt_cache_capture_eligible: bool,
    ) -> Self {
        Self {
            has_visual_embeddings,
            execution_profile,
            is_prompt_cache_capture_eligible,
        }
    }

    pub(super) const fn context_identifier_flags(self) -> u64 {
        let mut context_identifier_flags = 0;
        if self.has_visual_embeddings {
            context_identifier_flags |= VISUAL_PREFILL_CONTEXT_FLAG;
        }
        if matches!(
            self.execution_profile,
            AdaptiveRamGrowthExecutionProfile::Paged
        ) {
            context_identifier_flags |= PAGED_EXPERTS_CONTEXT_FLAG;
        }
        if self.is_prompt_cache_capture_eligible {
            context_identifier_flags |= PROMPT_CACHE_CAPTURE_ELIGIBLE_CONTEXT_FLAG;
        }
        context_identifier_flags
    }
}
