//! One request's drafting state, for an engine whose decode loop emits one
//! token an advance: whether the session drafts, its draft session, and the
//! tokens a round produced that have not reached the caller yet.

use std::collections::VecDeque;

use super::draft_tree_round::DraftSession;

/// A request's drafting state from its start to its last token.
pub(crate) struct DraftRequestState {
    /// The drafter's context and the request's sampling key; `None` once the
    /// session stops drafting.
    pub(crate) draft_session: Option<DraftSession>,
    /// Tokens the last round chose, oldest first. Every entry but the last is
    /// already in the target's state; the last is the round's own next pick,
    /// which the next round's verify commits as its anchor.
    pub(crate) emission_queue: VecDeque<u32>,
    is_drafting: bool,
}

impl DraftRequestState {
    /// The state of a request that drafts when `is_admitted`, keyed by the
    /// request's seed so a seeded request samples the same reply every time.
    pub(crate) fn for_new_request(
        is_admitted: bool,
        sampling_strategy: crate::Qwen3_5SamplingStrategy,
    ) -> Self {
        let keyed_seed = match sampling_strategy {
            crate::Qwen3_5SamplingStrategy::HighestLogit => 0,
            crate::Qwen3_5SamplingStrategy::TopKTopP { seed, .. } => {
                crate::qwen3_5_core::text::resolve_sampling_seed(
                    seed,
                    crate::sampling_seed::current_time_millis_since_unix_epoch,
                )
            }
        };
        Self {
            draft_session: is_admitted.then_some(DraftSession::new(keyed_seed)),
            emission_queue: VecDeque::new(),
            is_drafting: is_admitted,
        }
    }

    /// Whether the session still decodes in draft rounds.
    pub(crate) const fn is_drafting(&self) -> bool {
        self.is_drafting
    }

    /// Stops drafting for the rest of the session, when a path the round
    /// cannot preserve takes over (a forced thinking boundary, or tokens fed
    /// in from outside). Queued tokens drop: the caller's classic path owns
    /// everything from here. Returns whether the session was drafting.
    pub(crate) fn stop_drafting(&mut self) -> bool {
        if !self.is_drafting {
            return false;
        }
        self.is_drafting = false;
        self.emission_queue.clear();
        self.draft_session = None;
        true
    }
}

#[cfg(test)]
#[path = "draft_request_state_tests.rs"]
mod draft_request_state_tests;
