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
//!    ladder through repeated swaps); and
//! 6. the bytes the smaller context frees, plus any raise of the ceiling
//!    since the demotion, cover the shortfall the resident engine measured
//!    when it demoted. The resident engine counts what the startup
//!    arithmetic cannot see (a prompt chunk's working buffers, a hosted
//!    drafter), so without this a request a few tokens smaller promoted and
//!    was demoted again on its first forward: two model loads a request.
//!
//! The strict inequality is the anti-thrash hysteresis: the demoting request
//! itself can never re-trigger a promotion (its own context compares equal,
//! not strictly below), so a promotion followed by re-admission of the same
//! request cannot loop. The worker additionally bounds swaps to at most one
//! per direction per request, so a promotion that is demoted again within
//! one request stays streaming until the request completes.

use crate::forward_admission::StreamingEngineRequirement;
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
    /// The incoming request's smaller context, with any raise of the
    /// ceiling since the demotion, frees fewer bytes than the resident engine
    /// was short when it demoted. The resident engine measured what the
    /// startup arithmetic cannot see (a prompt chunk's working buffers, a
    /// hosted drafter), so promoting would demote again on the first forward
    /// and cost two model loads for no resident token.
    IncomingRequestFreesLessThanTheDemotionShortfall {
        freed_bytes: u64,
        shortfall_bytes: u64,
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
    if let Some(cause) = incoming_request_blocker(
        engine_placement,
        startup_residency_facts,
        effective_active_memory_ceiling_bytes,
        incoming_total_context_token_count,
    ) {
        return ResidentPromotionDecision::StayStreaming { cause };
    }
    ResidentPromotionDecision::PromoteToResident
}

/// What, about the incoming request itself, keeps the worker streaming: a
/// context no smaller than the one that demoted it, a context the startup
/// arithmetic cannot fit, or a context too little smaller to free the bytes
/// the resident engine was short.
fn incoming_request_blocker(
    engine_placement: EnginePlacement,
    startup_residency_facts: &StreamingStartupResidencyFacts,
    effective_active_memory_ceiling_bytes: u64,
    incoming_total_context_token_count: usize,
) -> Option<ResidentPromotionBlocker> {
    // is_demoted_for_request guarantees the count; a missing one is read as
    // no demotion because Rust cannot connect the two queries on a Copy value.
    let Some(demoted_total_context_token_count) =
        engine_placement.demoted_total_context_token_count()
    else {
        return Some(ResidentPromotionBlocker::NotDemotedForRequest);
    };
    if incoming_total_context_token_count >= demoted_total_context_token_count {
        return Some(
            ResidentPromotionBlocker::IncomingContextNotBelowDemotionContext {
                incoming_total_context_token_count,
                demoted_total_context_token_count,
            },
        );
    }
    if !startup_residency_facts.complete_residency_admits_request_context(
        effective_active_memory_ceiling_bytes,
        incoming_total_context_token_count,
    ) {
        return Some(
            ResidentPromotionBlocker::IncomingRequestContextDoesNotFitCompleteResidency {
                incoming_total_context_token_count,
            },
        );
    }
    let freed_tokens = demoted_total_context_token_count - incoming_total_context_token_count;
    shortfall_blocker(
        engine_placement,
        startup_residency_facts
            .sequence_state_bytes_per_token
            .saturating_mul(u64::try_from(freed_tokens).unwrap_or(u64::MAX)),
        effective_active_memory_ceiling_bytes,
    )
}

/// Blocks the promotion when the bytes the incoming request frees, plus any
/// raise of the ceiling since the demotion, fall short of what the resident
/// engine was short. A cause without a measured shortfall is not judged.
fn shortfall_blocker(
    engine_placement: EnginePlacement,
    context_freed_bytes: u64,
    effective_active_memory_ceiling_bytes: u64,
) -> Option<ResidentPromotionBlocker> {
    let shortfall_bytes = engine_placement
        .demotion_cause()
        .and_then(StreamingEngineRequirement::shortfall_bytes)
        .and_then(|shortfall_bytes| u64::try_from(shortfall_bytes).ok())?;
    let ceiling_raise_bytes = engine_placement
        .effective_ceiling_bytes_at_demotion()
        .map_or(0, |ceiling_at_demotion| {
            effective_active_memory_ceiling_bytes.saturating_sub(ceiling_at_demotion)
        });
    let freed_bytes = context_freed_bytes.saturating_add(ceiling_raise_bytes);
    (freed_bytes < shortfall_bytes).then_some(
        ResidentPromotionBlocker::IncomingRequestFreesLessThanTheDemotionShortfall {
            freed_bytes,
            shortfall_bytes,
        },
    )
}
