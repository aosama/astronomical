//! Retained-expert ownership vocabulary shared across the paging surfaces.
//!
//! The whole-page-per-layer container that once lived beside these types was
//! removed: decode retention is owned by the per-layer expert slot tables
//! (`qwen3_5_streaming/expert_streaming/warm_expert_slot_table_cache.rs`),
//! which replaced page-granularity retention when individual expert caching
//! landed. Two ownership records outlived that container because other
//! surfaces consume them: the reclamation record memory-pressure paths
//! report through, and the demand weighting that keeps prompt-tail routes
//! competitive with earlier prompt frequency.

/// Exact class-specific ownership released under memory pressure.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RetainedExpertReclamation {
    pub released_partial_layer_count: usize,
    pub released_partial_payload_bytes: u64,
    pub released_complete_layer_count: usize,
    pub released_complete_payload_bytes: u64,
}

impl RetainedExpertReclamation {
    #[must_use]
    pub const fn released_payload_bytes(self) -> u64 {
        self.released_partial_payload_bytes
            .saturating_add(self.released_complete_payload_bytes)
    }
}

/// Scales last-chunk assignments so their token density matches earlier prefill.
///
/// Decode continues from the prompt tail. One last-chunk token therefore counts
/// as many assignments as `earlier / last_chunk`, with a floor of one.
#[must_use]
pub fn last_prefill_chunk_demand_weight(
    earlier_prefill_token_count: u64,
    last_prefill_chunk_token_count: u64,
) -> u64 {
    if earlier_prefill_token_count == 0 || last_prefill_chunk_token_count == 0 {
        return 1;
    }
    (earlier_prefill_token_count / last_prefill_chunk_token_count).max(1)
}
