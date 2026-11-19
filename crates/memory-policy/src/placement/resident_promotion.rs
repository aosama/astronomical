//! The bounded re-promotion gate from a demoted streaming engine back to the
//! complete-residency resident engine.
//!
//! A demotion is sticky by design: the SSD-streaming engine is markedly
//! slower, and a placement that flip-flopped per request would thrash GPU
//! memory. This gate re-promotes only when every pressure signal has
//! genuinely resolved:
//!
//! 1. the placement is a request demotion (not the startup streaming
//!    selection, whose ceiling was too low from the beginning, and not an
//!    already-promoted worker);
//! 2. the model family factory can build a resident engine mid-request;
//! 3. complete residency fits at the worker's *current effective* ceiling —
//!    the same byte arithmetic startup selection uses, so a re-promotion can
//!    never disagree with the engine a replacement worker would load; and
//! 4. the incoming request's total context is *strictly* below the total
//!    context that caused the demotion; and
//! 5. complete residency at the effective ceiling also leaves room for the
//!    incoming request's own persistent decoder state, so a smaller request
//!    that still cannot run resident does not pay a promote-then-demote
//!    pair of model swaps (and a strictly decreasing request sequence cannot
//!    ladder through repeated swaps).
//!
//! The strict inequality is the anti-thrash hysteresis: the demoting request
//! itself can never re-trigger a promotion (its own context compares equal,
//! not strictly below), so a promotion followed by re-admission of the same
//! request cannot loop. The worker additionally bounds swaps to at most one
//! per direction per request, so a promotion that is demoted again within
//! one request stays streaming until the request completes.

use crate::placement::engine_placement::EnginePlacement;
use crate::residency::StreamingStartupResidencyFacts;

/// Typed outcome of the re-promotion gate for one incoming request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidentPromotionDecision {
    /// Swap the worker to the resident engine before starting generation.
    PromoteToResident,
    /// Keep the streaming engine for this request.
    StayStreaming {
        /// The first gate that blocked the promotion.
        cause: ResidentPromotionBlocker,
    },
}

/// Why a demoted worker stayed on the streaming engine.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidentPromotionBlocker {
    /// The placement is not a request demotion: startup selected the
    /// streaming engine from the beginning, or the worker is already back on
    /// the resident engine.
    NotDemotedForRequest,
    /// The model family factory cannot build a resident engine mid-request.
    ResidentPromotionUnsupportedByFactory,
    /// Complete residency does not fit at the current effective ceiling, so
    /// the pressure that caused the demotion has not resolved.
    CompleteResidencyDoesNotFitAtEffectiveCeiling,
    /// The incoming request's total context is equal to or above the total
    /// context that caused the demotion, so admitting it resident would
    /// reproduce the demotion.
    IncomingContextNotBelowDemotionContext {
        incoming_total_context_token_count: usize,
        demoted_total_context_token_count: usize,
    },
    /// Complete residency fits empty, but not with the incoming request's
    /// persistent decoder state on top, so the resident engine would demote
    /// the request again on its first forward.
    IncomingRequestContextDoesNotFitCompleteResidency {
        incoming_total_context_token_count: usize,
    },
}

/// Decides whether a demoted streaming worker may swap back to the resident
/// engine for the incoming request.
///
/// `startup_residency_facts` are the facts captured when the streaming model
/// was loaded; the fit check replays startup selection's own arithmetic
/// against `effective_active_memory_ceiling_bytes`, so the gate and a
/// replacement worker's startup selection can never disagree.
#[must_use]
pub fn should_promote_demoted_streaming_engine_to_resident(
    engine_placement: EnginePlacement,
    factory_supports_resident_promotion: bool,
    startup_residency_facts: &StreamingStartupResidencyFacts,
    effective_active_memory_ceiling_bytes: u64,
    incoming_total_context_token_count: usize,
) -> ResidentPromotionDecision {
    if !engine_placement.is_demoted_for_request() {
        return ResidentPromotionDecision::StayStreaming {
            cause: ResidentPromotionBlocker::NotDemotedForRequest,
        };
    }
    if !factory_supports_resident_promotion {
        return ResidentPromotionDecision::StayStreaming {
            cause: ResidentPromotionBlocker::ResidentPromotionUnsupportedByFactory,
        };
    }
    if !startup_residency_facts
        .raised_ceiling_selects_complete_residency(effective_active_memory_ceiling_bytes)
    {
        return ResidentPromotionDecision::StayStreaming {
            cause: ResidentPromotionBlocker::CompleteResidencyDoesNotFitAtEffectiveCeiling,
        };
    }
    let Some(demoted_total_context_token_count) =
        engine_placement.demoted_total_context_token_count()
    else {
        // is_demoted_for_request guarantees the count; the else-arm exists
        // only because Rust cannot connect the two queries on a Copy value.
        return ResidentPromotionDecision::StayStreaming {
            cause: ResidentPromotionBlocker::NotDemotedForRequest,
        };
    };
    if incoming_total_context_token_count >= demoted_total_context_token_count {
        return ResidentPromotionDecision::StayStreaming {
            cause: ResidentPromotionBlocker::IncomingContextNotBelowDemotionContext {
                incoming_total_context_token_count,
                demoted_total_context_token_count,
            },
        };
    }
    if !startup_residency_facts.complete_residency_admits_request_context(
        effective_active_memory_ceiling_bytes,
        incoming_total_context_token_count,
    ) {
        return ResidentPromotionDecision::StayStreaming {
            cause: ResidentPromotionBlocker::IncomingRequestContextDoesNotFitCompleteResidency {
                incoming_total_context_token_count,
            },
        };
    }
    ResidentPromotionDecision::PromoteToResident
}
