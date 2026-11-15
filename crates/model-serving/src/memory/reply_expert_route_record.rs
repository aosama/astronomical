//! Where each token of one reply found its routed experts: in the
//! `astro_expert_cache` or on the SSD.
//!
//! The session counters already say how often experts came from storage
//! since the model loaded, but a total hides the shape of a reply: whether
//! reads were spread thinly over every token or bunched at its start.
//! Observatory draws one cell per token from this record, scaled by how many
//! experts a token routes to, so a token that read 22 of its 384 experts
//! from the SSD reads as mostly served from memory.

/// Per-token expert routes of one reply.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct ReplyExpertRouteRecord {
    storage_read_expert_counts_by_token: Vec<u16>,
    routed_expert_count_per_token: u16,
}

impl ReplyExpertRouteRecord {
    /// Records one reply token from the change in the session's route
    /// counters across its forward.
    pub(crate) fn record_token(
        &mut self,
        memory_served_route_count: u64,
        storage_read_route_count: u64,
    ) {
        let routed_expert_count =
            memory_served_route_count.saturating_add(storage_read_route_count);
        // Every token of a fixed-width router routes the same number of
        // experts. The largest seen is kept as the scale so that one token
        // that routed fewer, such as a model that streams no experts in some
        // layers, cannot shrink every other token's column.
        self.routed_expert_count_per_token = self
            .routed_expert_count_per_token
            .max(u16::try_from(routed_expert_count).unwrap_or(u16::MAX));
        self.storage_read_expert_counts_by_token
            .push(u16::try_from(storage_read_route_count).unwrap_or(u16::MAX));
    }

    /// How many routed experts one token uses, or zero before any token.
    #[must_use]
    pub(crate) const fn routed_expert_count_per_token(&self) -> u16 {
        self.routed_expert_count_per_token
    }

    #[must_use]
    pub(crate) fn into_storage_read_expert_counts_by_token(self) -> Vec<u16> {
        self.storage_read_expert_counts_by_token
    }
}

#[cfg(test)]
#[path = "reply_expert_route_record_tests.rs"]
mod reply_expert_route_record_tests;
