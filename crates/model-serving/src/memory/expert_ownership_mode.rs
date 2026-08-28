//! Single classification of expert ownership for status and execution.
//!
//! Family crates supply whether a complete owner exists, whether a pager exists,
//! and how many paged bytes are retained. They must not invent a second Hybrid /
//! Paged / Resident answer.

use astronomical_ipc_protocol::ExpertMemoryMode;

/// Classifies expert RAM ownership from structural facts.
#[must_use]
pub const fn classify_expert_memory_mode(
    complete_sparse_owner_is_installed: bool,
    sparse_expert_paging_is_configured: bool,
    retained_paged_expert_payload_bytes: u64,
) -> ExpertMemoryMode {
    if complete_sparse_owner_is_installed || !sparse_expert_paging_is_configured {
        ExpertMemoryMode::Resident
    } else if retained_paged_expert_payload_bytes > 0 {
        ExpertMemoryMode::Hybrid
    } else {
        ExpertMemoryMode::Paged
    }
}
