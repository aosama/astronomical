//! Typed commit, topology, and reclamation records for retained expert ownership.

use thiserror::Error;

/// Result of atomically offering one execution-materialized page to the cache.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetainedExpertLayerCommitOutcome {
    Committed(RetainedExpertLayerCommitDelta),
    PreservedExisting,
    RejectedByCurrentCeiling,
}

/// Commit decision plus the candidate owner returned when a live ceiling rejects it.
#[derive(Debug)]
pub struct RetainedExpertLayerCommit<ExpertPage> {
    pub outcome: RetainedExpertLayerCommitOutcome,
    pub uncommitted_page: Option<ExpertPage>,
}

/// Exact ownership bytes transferred by a successful atomic commit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetainedExpertLayerCommitDelta {
    pub released_payload_bytes: u64,
    pub committed_payload_bytes: u64,
}

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

/// Invalid commit metadata rejected before prior ownership can be mutated.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum RetainedExpertLayerCommitError {
    #[error("retained expert commit references out-of-range layer {layer_index}")]
    LayerOutOfRange { layer_index: usize },
    #[error("retained expert commit has zero expert capacity at layer {layer_index}")]
    ZeroExpertCapacity { layer_index: usize },
    #[error("retained expert commit has invalid expert identifiers at layer {layer_index}")]
    InvalidExpertIds { layer_index: usize },
    #[error("retained expert commit has zero payload at layer {layer_index}")]
    ZeroPayload { layer_index: usize },
    #[error("retained expert payload accounting overflowed at layer {layer_index}")]
    PayloadByteCountOverflow { layer_index: usize },
    #[error("retained expert payload accounting was inconsistent at layer {layer_index}")]
    InconsistentPayloadAccounting { layer_index: usize },
}
