//! Which serving phase demand evidence is recorded for, and the lifecycle
//! of a request's prefill demand window.
//!
//! The phase split is the load-bearing distinction of the demand domain:
//! prefill evidence answers "what does this prompt's route union need right
//! now" (admission planning for the current request), while decode evidence
//! answers "what does this conversation keep re-using" (cross-request
//! retention and persisted history). Recording both into one counter — the
//! design this module replaces — let a long prompt's wide, shallow route
//! union masquerade as conversational heat in the next session.

use super::WarmExpertSlotTableCache;

/// The serving phase demand evidence is currently recorded into. Every
/// generation opens in `Prefill` — the prompt must be processed before any
/// generated token exists — and switches to `Decode` at the first decode
/// forward, staying there until the next request opens its prefill window.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ExpertDemandPhase {
    Prefill,
    Decode,
}

impl WarmExpertSlotTableCache {
    /// The serving phase demand evidence is currently recorded for. This is
    /// engine state, exposed so the storage boundary files each expert read
    /// under the phase the engine is actually in (issue #1715).
    pub(crate) fn current_serving_phase(&self) -> crate::memory::SessionStreamingPhase {
        match self.demand_phase {
            ExpertDemandPhase::Prefill => crate::memory::SessionStreamingPhase::Prefill,
            ExpertDemandPhase::Decode => crate::memory::SessionStreamingPhase::Decode,
        }
    }

    /// Opens one request's prefill demand window.
    ///
    /// The previous request's prefill evidence is spent: it served its
    /// admission decisions and its breadth would only dilute the next
    /// prompt's ranking, so the window starts empty. The decode stream —
    /// lifetime evidence — is deliberately preserved: an identical or
    /// similar next ask serves from RAM instead of re-reading its experts
    /// (issue #955).
    pub fn begin_prefill_demand_window(&mut self) {
        self.demand_phase = ExpertDemandPhase::Prefill;
        // Chunks set their own weight before recording; the reset discards a
        // previous generation's final-chunk boost.
        self.demand_assignment_weight = 1;
        for prefill_layer_counts in &mut self.prefill_demand_counts_by_layer {
            prefill_layer_counts.fill(0);
        }
    }

    /// Scales prefill demand recorded by subsequent forwards until the next
    /// window or phase switch.
    ///
    /// The request's final prefill chunk records under
    /// `last_prefill_chunk_demand_weight` so prompt-tail routes — the best
    /// predictor of the first decode tokens' routing — keep decode-relevant
    /// coverage in the handoff residency plan. Decode evidence ignores this
    /// multiplier entirely; see `demand_recording`.
    pub fn set_prefill_demand_assignment_weight(&mut self, assignment_weight: u64) {
        self.demand_assignment_weight = assignment_weight.max(1);
    }
}
