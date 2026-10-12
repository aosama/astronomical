//! Retention decisions for the live decoder state kept between requests.
//!
//! In-memory prefix adoption (issue #1167) keeps the just-completed
//! generation's decoder state alive so the next turn of the same conversation
//! adopts it instead of re-reading the prefix KV from the disk store. The
//! engine measures the byte facts and enacts the decision; this module owns
//! the decision itself.

/// Exact caller-supplied facts about the just-completed generation's state.
pub struct RetainedPrefixRetentionRequirements {
    /// Whether the serving configuration enables in-memory prefix adoption.
    pub adoption_enabled: bool,
    /// Token count the retained state can adopt: the boundary of the last
    /// complete prompt-cache block the generation published. Zero means the
    /// generation published nothing adoptable.
    pub adoptable_token_count: usize,
    /// The retained decoder state's payload bytes, measured from the live state.
    pub retained_state_payload_bytes: u64,
    /// The engine's effective MLX memory ceiling.
    pub effective_mlx_memory_ceiling_bytes: u64,
}

/// The typed retention outcome; the engine enacts it by keeping or dropping
/// the live state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetainedPrefixRetentionDecision {
    Retain,
    Drop,
}

impl RetainedPrefixRetentionRequirements {
    /// Decides whether the completed generation's state stays live for the
    /// next request.
    ///
    /// A retained state above half the ceiling is dropped: an unrelated next
    /// conversation would otherwise start with half of its working memory
    /// already promised to a state it will never adopt. Smaller states are
    /// cheap to keep (the bytes are already active) and any pressure they
    /// cause is relieved by dropping them when a new request fails to adopt.
    pub fn decide(&self) -> RetainedPrefixRetentionDecision {
        if !self.adoption_enabled || self.adoptable_token_count == 0 {
            return RetainedPrefixRetentionDecision::Drop;
        }
        if self.retained_state_payload_bytes * 2 > self.effective_mlx_memory_ceiling_bytes {
            return RetainedPrefixRetentionDecision::Drop;
        }
        RetainedPrefixRetentionDecision::Retain
    }
}
