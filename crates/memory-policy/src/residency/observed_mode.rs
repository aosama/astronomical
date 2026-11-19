//! Evidence-driven expert memory mode: residency observed, not just claimed.
//!
//! `classify_expert_memory_mode` labels ownership from structural facts
//! (which engine owns the experts and how many paged bytes are retained).
//! That claim can disagree with what a request actually did: a paged engine
//! whose every expert page was already warm read no expert storage at all,
//! and a claimed-resident engine that streamed even one expert page was not
//! fully resident for that request. This module derives the label from
//! observed expert storage-read bytes, so the user-facing residency label
//! cannot disagree with what actually happened — and it works with
//! performance attribution disabled, because the byte tally is always-on.

use astronomical_ipc_protocol::ExpertMemoryMode;

/// Classifies expert memory mode from the expert storage bytes actually read
/// during one request.
///
/// Zero observed bytes means every expert weight the request touched was
/// already resident, so the request was served resident regardless of which
/// engine owns the pager. Any observed byte defers to the ownership
/// classification, with one defensive correction: a claimed-resident engine
/// that still read expert storage was hybrid for that request, because some
/// expert weight arrived from storage mid-request.
#[must_use]
pub const fn observed_expert_memory_mode(
    engine_ownership_classification: ExpertMemoryMode,
    observed_expert_storage_read_bytes: u64,
) -> ExpertMemoryMode {
    if observed_expert_storage_read_bytes == 0 {
        return ExpertMemoryMode::Resident;
    }
    match engine_ownership_classification {
        ExpertMemoryMode::Resident => ExpertMemoryMode::Hybrid,
        classified_expert_memory_mode => classified_expert_memory_mode,
    }
}
