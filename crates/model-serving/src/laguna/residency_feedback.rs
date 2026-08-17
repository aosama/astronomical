//! Measured Laguna expert-retention feedback between sequential forwards.
//!
//! Startup remains conservative because no request has executed yet. Each
//! completed forward replaces that bootstrap ceiling with evidence from the
//! current active ownership and the forward's transient peak. The next forward
//! can therefore retain every expert that genuinely coexists with measured work
//! while naturally reclaiming experts as context state grows.

use crate::MlxRamBudget;

/// Extra room above one observed forward for allocator jitter and bookkeeping.
const COMPLETED_FORWARD_SAFETY_BUFFER_BYTES: u64 = 64_000_000;

/// Returns the retained-expert ceiling supported by one completed forward.
#[must_use]
pub fn laguna_retained_expert_budget_after_completed_forward(
    mlx_ram_budget: &MlxRamBudget,
    active_memory_bytes: u64,
    peak_memory_bytes: u64,
    retained_expert_payload_bytes: u64,
    complete_expert_payload_bytes: u64,
) -> u64 {
    let measured_next_forward_reserve_bytes = peak_memory_bytes
        .saturating_sub(active_memory_bytes)
        .saturating_add(COMPLETED_FORWARD_SAFETY_BUFFER_BYTES);
    mlx_ram_budget
        .retained_expert_budget_for_admitted_forward(
            active_memory_bytes,
            retained_expert_payload_bytes,
            measured_next_forward_reserve_bytes,
        )
        .min(complete_expert_payload_bytes)
}
