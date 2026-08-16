use std::collections::VecDeque;

use crate::qwen3_5::decoder::Qwen3_5PersistentPromptCacheBoundaryCheckpoint;

/// Exact target frontier corresponding to one publicly emitted token prefix.
#[doc(hidden)]
pub struct VerifiedTargetFrontier {
    pub position_tokens: u32,
    pub boundary: Qwen3_5PersistentPromptCacheBoundaryCheckpoint,
}

impl VerifiedTargetFrontier {
    fn payload_byte_count(&self) -> u64 {
        self.boundary
            .recurrent_snapshot_tensors
            .values()
            .map(|snapshot| snapshot.byte_count() as u64)
            .fold(0, u64::saturating_add)
    }
}

struct VerifiedEmission {
    token_id: u32,
    frontier_after_emission: Option<VerifiedTargetFrontier>,
}

/// Accepted tokens plus the exact target frontier at the public stream boundary.
#[doc(hidden)]
pub struct VerifiedEmissionQueue {
    emissions: VecDeque<VerifiedEmission>,
    public_frontier: Option<VerifiedTargetFrontier>,
}

impl VerifiedEmissionQueue {
    pub fn new(public_frontier: VerifiedTargetFrontier) -> Self {
        Self {
            emissions: VecDeque::new(),
            public_frontier: Some(public_frontier),
        }
    }

    pub fn push(&mut self, token_id: u32, frontier_after_emission: Option<VerifiedTargetFrontier>) {
        self.emissions.push_back(VerifiedEmission {
            token_id,
            frontier_after_emission,
        });
    }

    pub fn pop_front(&mut self) -> Option<u32> {
        let emission = self.emissions.pop_front()?;
        self.public_frontier = emission.frontier_after_emission;
        Some(emission.token_id)
    }

    pub fn is_empty(&self) -> bool {
        self.emissions.is_empty()
    }

    /// Transfers the frontier that matches tokens already shown to the user.
    ///
    /// This is one-shot: a later injection must not restore unpublished drafts.
    /// The last accepted draft stores `None` because live target state already
    /// matches the public prefix after that token is emitted.
    pub fn take_public_frontier(&mut self) -> Option<VerifiedTargetFrontier> {
        self.public_frontier.take()
    }

    pub(crate) fn payload_byte_count(&self) -> u64 {
        let public_frontier_bytes = self
            .public_frontier
            .as_ref()
            .map_or(0, VerifiedTargetFrontier::payload_byte_count);
        self.emissions
            .iter()
            .filter_map(|emission| emission.frontier_after_emission.as_ref())
            .map(VerifiedTargetFrontier::payload_byte_count)
            .fold(public_frontier_bytes, u64::saturating_add)
    }
}
