//! Typed choices for admitting resident and paged forward operations.
//!
//! One enum cannot span both engines: the resident engine can never reclaim
//! paged experts, and the paged engine can never demand a streaming fork. Separate enums keep every impossible
//! state unrepresentable, so callers match their own engine's decisions
//! exhaustively instead of carrying fallback arms for the other engine.

use crate::budget::AdaptiveRamGrowthProjection;

/// Why a resident forward must leave the complete-residency engine.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamingEngineRequirement {
    /// Persistent ownership alone exceeds the strict active-memory ceiling.
    StableShortfall { shortfall_bytes: usize },
    /// The configured prefill chunk's transient peak exceeds the ceiling.
    ///
    /// The chunk is never shrunk to make it fit: smaller chunks re-pay every
    /// per-chunk fixed cost (complete-layer decode and storage reads on the
    /// streaming side, publication and synchronization on both), so a
    /// halving ladder trades a clear engine decision for a slow request.
    TransientPeakShortfallAtPrefillChunk {
        shortfall_bytes: usize,
        prefill_chunk_token_count: usize,
    },
    /// A decode or other forward without a prefill chunk exceeds the peak allowance.
    TransientPeakShortfallWithoutPrefillChunk { shortfall_bytes: usize },
    /// The request's own context workspace cannot be admitted beside complete
    /// expert ownership. This is not a forward rejection: the forward never
    /// ran, and the SSD-streaming engine's smaller paged ownership may still
    /// admit the same context.
    RequestContextShortfall { shortfall_bytes: usize },
    /// A forward allocation exceeded the active-memory ceiling at runtime,
    /// after admission projected it to fit. The shortfall is exact.
    ForwardAllocationShortfall { shortfall_bytes: usize },
    /// A forward allocation exhausted graphics-processor memory at runtime.
    /// The failure is real but unquantified, so it carries no byte figure.
    ForwardGraphicsProcessorMemoryExhausted,
}

/// A concrete action after evaluating one resident forward's memory equation.
///
/// Allocator-cache reclamation is intentionally absent: MLX active-memory
/// admission excludes allocator-cache bytes, and no resident forward admission
/// path reclaims that cache to make a projection fit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidentForwardAdmissionDecision {
    /// Stable ownership and expected peak fit.
    Admit,
    /// The resident engine cannot admit this forward without changing engines.
    RequireStreamingEngine { cause: StreamingEngineRequirement },
}

/// A resident forward decision that did not admit the forward.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidentForwardAdmissionRejection {
    /// The resident engine cannot admit this forward without changing engines.
    RequireStreamingEngine { cause: StreamingEngineRequirement },
}

impl ResidentForwardAdmissionDecision {
    /// Chooses whether a resident forward may proceed or needs streaming.
    #[must_use]
    pub fn for_resident_forward(
        forward_projection: &AdaptiveRamGrowthProjection,
        prefill_chunk_token_count: Option<usize>,
    ) -> Self {
        if forward_projection.fits_stable_and_peak_limits() {
            return Self::Admit;
        }
        if !forward_projection.fits_stable_limit() {
            return Self::RequireStreamingEngine {
                cause: StreamingEngineRequirement::StableShortfall {
                    shortfall_bytes: forward_projection
                        .memory_equation()
                        .stable_shortfall_bytes(),
                },
            };
        }
        let shortfall_bytes = forward_projection.memory_equation().peak_shortfall_bytes();
        if let Some(prefill_chunk_token_count) = prefill_chunk_token_count {
            return Self::RequireStreamingEngine {
                cause: StreamingEngineRequirement::TransientPeakShortfallAtPrefillChunk {
                    shortfall_bytes,
                    prefill_chunk_token_count,
                },
            };
        }
        Self::RequireStreamingEngine {
            cause: StreamingEngineRequirement::TransientPeakShortfallWithoutPrefillChunk {
                shortfall_bytes,
            },
        }
    }

    #[must_use]
    pub const fn is_admitted(self) -> bool {
        matches!(self, Self::Admit)
    }

    /// Returns the rejection a caller must enact, or `None` when admitted.
    ///
    /// Rejection carriers hold this narrower type so no error path can carry
    /// `Admit` and no caller needs an unreachable "unexpected decision" arm.
    #[must_use]
    pub const fn rejection(self) -> Option<ResidentForwardAdmissionRejection> {
        match self {
            Self::Admit => None,
            Self::RequireStreamingEngine { cause } => {
                Some(ResidentForwardAdmissionRejection::RequireStreamingEngine { cause })
            }
        }
    }

    /// Returns the resident forward's established diagnostic decision label.
    #[must_use]
    pub const fn resident_log_label(self) -> &'static str {
        match self {
            Self::Admit => "admit_resident_forward",
            Self::RequireStreamingEngine { .. } => "request_streaming_engine",
        }
    }
}

/// A concrete action after evaluating one paged forward's memory equation.
///
/// Allocator-cache reclamation is intentionally absent: MLX active-memory
/// admission excludes allocator-cache bytes, and no paged forward admission
/// path reclaims that cache to make a projection fit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PagedForwardAdmissionDecision {
    /// Stable ownership and expected peak fit.
    Admit,
    /// Release retained paged expert payload before retrying the same forward.
    ReclaimRetainedExperts { target_bytes: usize },
    /// A paged forward cannot fit after the legal reclamation options are exhausted.
    Reject { unresolved_shortfall_bytes: usize },
}

impl PagedForwardAdmissionDecision {
    /// Chooses whether a paged forward fits, needs expert reclamation, or must stop.
    #[must_use]
    pub const fn for_paged_forward(
        forward_projection: &AdaptiveRamGrowthProjection,
        retained_expert_payload_bytes: usize,
        previous_pass_released_pages: Option<bool>,
    ) -> Self {
        if forward_projection.fits_stable_and_peak_limits() {
            return Self::Admit;
        }
        let reclamation_plan =
            forward_projection.expert_retention_reclamation_plan(retained_expert_payload_bytes);
        let target_bytes = reclamation_plan.reclamation_target_bytes();
        if reclamation_plan.can_satisfy_every_memory_boundary()
            && target_bytes > 0
            && !matches!(previous_pass_released_pages, Some(false))
        {
            return Self::ReclaimRetainedExperts { target_bytes };
        }
        let unresolved_shortfall_bytes = if reclamation_plan.unresolved_shortfall_bytes() > 0 {
            reclamation_plan.unresolved_shortfall_bytes()
        } else {
            let plan_shortfall_bytes = reclamation_plan.required_reclamation_bytes();
            let projection_shortfall_bytes =
                forward_projection.operation_reclamation_required_bytes();
            if plan_shortfall_bytes > projection_shortfall_bytes {
                plan_shortfall_bytes
            } else {
                projection_shortfall_bytes
            }
        };
        Self::Reject {
            unresolved_shortfall_bytes,
        }
    }

    #[must_use]
    pub const fn is_admitted(self) -> bool {
        matches!(self, Self::Admit)
    }

    /// Returns the paged forward's established diagnostic decision label.
    ///
    /// A non-admitted paged forward always enters the expert-reclamation
    /// ladder, and before this decision type existed every non-fitting paged
    /// forward was logged under the single ladder-entry label below. The
    /// ladder's own pressure logs ("admit"/"reject"/"reclaim_experts") carry
    /// the outcome, so a terminal rejection keeps that entry label instead of
    /// inventing a new one that log consumers have never seen.
    #[must_use]
    pub const fn paged_log_label(self) -> &'static str {
        match self {
            Self::Admit => "admit_paged_forward",
            Self::ReclaimRetainedExperts { .. } | Self::Reject { .. } => "reclaim_paged_experts",
        }
    }
}

#[cfg(test)]
#[path = "forward_admission_tests.rs"]
mod forward_admission_tests;
