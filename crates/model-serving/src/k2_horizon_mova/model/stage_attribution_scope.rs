//! Per-stage attribution scope shared by the decode and prefill forwards.

/// Selects which attribution operation each optional per-stage completion
/// wait records. The waits synchronize the graphics processor several times
/// per layer, so they run only when the matching diagnostic opt-in is set.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StageAttributionScope {
    Decode,
    Prefill,
}

impl StageAttributionScope {
    #[must_use]
    pub(crate) const fn attention_completion_wait(self) -> crate::PerformanceOperation {
        match self {
            Self::Decode => {
                crate::PerformanceOperation::DecodeAttentionGraphicsProcessorCompletionWait
            }
            Self::Prefill => {
                crate::PerformanceOperation::PrefillFullAttentionGraphicsProcessorCompletionWait
            }
        }
    }

    #[must_use]
    pub(crate) const fn mixture_of_values_completion_wait(self) -> crate::PerformanceOperation {
        match self {
            Self::Decode => {
                crate::PerformanceOperation::DecodeMixtureOfValuesGraphicsProcessorCompletionWait
            }
            Self::Prefill => {
                crate::PerformanceOperation::PrefillMixtureOfValuesGraphicsProcessorCompletionWait
            }
        }
    }

    #[must_use]
    pub(crate) const fn feed_forward_completion_wait(self) -> crate::PerformanceOperation {
        match self {
            Self::Decode => {
                crate::PerformanceOperation::DecodeFeedForwardGraphicsProcessorCompletionWait
            }
            Self::Prefill => {
                crate::PerformanceOperation::PrefillFeedForwardGraphicsProcessorCompletionWait
            }
        }
    }

    #[must_use]
    pub(crate) const fn shared_expert_completion_wait(self) -> crate::PerformanceOperation {
        match self {
            Self::Decode => {
                crate::PerformanceOperation::DecodeSharedExpertGraphicsProcessorCompletionWait
            }
            Self::Prefill => {
                crate::PerformanceOperation::PrefillSharedExpertGraphicsProcessorCompletionWait
            }
        }
    }
}
