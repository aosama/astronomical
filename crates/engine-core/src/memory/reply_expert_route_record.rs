//! Where each token of one reply found its routed experts: in the
//! `astro_expert_cache` or on the SSD.
//!
//! The session counters already say how often experts came from storage
//! since the model loaded, but a total hides the shape of a reply: whether
//! reads were spread thinly over every token or bunched at its start.
//! Observatory draws one cell per token from this record, scaled by how many
//! experts a token routes to, so a token that read 22 of its 384 experts
//! from the SSD reads as mostly served from memory.
//!
//! The same record keeps the reply's totals (experts found in memory, experts
//! missed, bytes read and the time spent reading), the per-reply measure of
//! how well the `astro_expert_cache` chose what to keep. Every figure comes
//! from always-on session counters, so it costs no clock and no allocation per
//! token beyond the one per-token cell.

/// The session's always-on expert counters at one instant. A reply token's
/// own figures are the change across its forward (`change_since`).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SessionExpertRouteSample {
    /// Routed experts found in the `astro_expert_cache`.
    pub memory_served_expert_count: u64,
    /// Routed experts the `astro_expert_cache` did not hold: the misses.
    pub storage_read_expert_count: u64,
    /// Expert payload bytes decode read from storage.
    pub decode_storage_read_bytes: u64,
    /// Time decode spent waiting on those reads.
    pub decode_storage_read_wait_nanoseconds: u64,
}

impl SessionExpertRouteSample {
    /// What happened between `earlier` and this sample; counters only grow,
    /// so a smaller later value (a reset) reads as zero.
    #[must_use]
    pub const fn change_since(&self, earlier: &Self) -> Self {
        Self {
            memory_served_expert_count: self
                .memory_served_expert_count
                .saturating_sub(earlier.memory_served_expert_count),
            storage_read_expert_count: self
                .storage_read_expert_count
                .saturating_sub(earlier.storage_read_expert_count),
            decode_storage_read_bytes: self
                .decode_storage_read_bytes
                .saturating_sub(earlier.decode_storage_read_bytes),
            decode_storage_read_wait_nanoseconds: self
                .decode_storage_read_wait_nanoseconds
                .saturating_sub(earlier.decode_storage_read_wait_nanoseconds),
        }
    }

    const fn add(&self, other: &Self) -> Self {
        Self {
            memory_served_expert_count: self
                .memory_served_expert_count
                .saturating_add(other.memory_served_expert_count),
            storage_read_expert_count: self
                .storage_read_expert_count
                .saturating_add(other.storage_read_expert_count),
            decode_storage_read_bytes: self
                .decode_storage_read_bytes
                .saturating_add(other.decode_storage_read_bytes),
            decode_storage_read_wait_nanoseconds: self
                .decode_storage_read_wait_nanoseconds
                .saturating_add(other.decode_storage_read_wait_nanoseconds),
        }
    }
}

/// Per-token expert routes of one reply, and the reply's totals.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ReplyExpertRouteRecord {
    storage_read_expert_counts_by_token: Vec<u16>,
    routed_expert_count_per_token: u16,
    reply_totals: SessionExpertRouteSample,
}

impl ReplyExpertRouteRecord {
    /// Records one reply token from the change in the session's counters
    /// across its forward.
    pub fn record_token(&mut self, token_change: SessionExpertRouteSample) {
        let routed_expert_count = token_change
            .memory_served_expert_count
            .saturating_add(token_change.storage_read_expert_count);
        // Every token of a fixed-width router routes the same number of
        // experts. The largest seen is kept as the scale so that one token
        // that routed fewer, such as a model that streams no experts in some
        // layers, cannot shrink every other token's column.
        self.routed_expert_count_per_token = self
            .routed_expert_count_per_token
            .max(u16::try_from(routed_expert_count).unwrap_or(u16::MAX));
        self.storage_read_expert_counts_by_token
            .push(u16::try_from(token_change.storage_read_expert_count).unwrap_or(u16::MAX));
        self.reply_totals = self.reply_totals.add(&token_change);
    }

    /// How many routed experts one token uses, or zero before any token.
    #[must_use]
    pub const fn routed_expert_count_per_token(&self) -> u16 {
        self.routed_expert_count_per_token
    }

    /// Whether no reply token has been recorded yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.storage_read_expert_counts_by_token.is_empty()
    }

    /// The reply's totals over every recorded token.
    #[must_use]
    pub const fn reply_totals(&self) -> SessionExpertRouteSample {
        self.reply_totals
    }

    #[must_use]
    pub fn into_storage_read_expert_counts_by_token(self) -> Vec<u16> {
        self.storage_read_expert_counts_by_token
    }
}

#[cfg(test)]
#[path = "reply_expert_route_record_tests.rs"]
mod reply_expert_route_record_tests;
