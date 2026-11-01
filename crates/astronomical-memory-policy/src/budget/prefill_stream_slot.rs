//! The prefill complete-layer streaming slot is only needed while a layer
//! stays nonresident.
//!
//! The slot is scratch for streaming one nonresident complete layer through a
//! multi-token forward. When the bytes left for experts without reserving it
//! already hold every expert, the plan seats every layer and prefill streams
//! nothing. Reserving the slot anyway shrinks the expert budget by exactly
//! the space those layers need, so prefill releases layers that decode just
//! warmed and then streams them back on every chunk — the slot evicts the
//! very layers it exists to stream. Measured at the 37-gigabyte sparse cell:
//! two layers released and 8.39 GB re-streamed per 8,192-token request, with
//! the real peak still 2.66 GB under the ceiling (issue #1716).

/// Pure: the prefill streaming slot once full fit is known.
///
/// Returns zero only when the expert budget without the slot holds the whole
/// expert payload; any shortfall keeps the slot, because at least one layer
/// will then be streamed. A model with no expert payload keeps the slot it
/// was given, so dense families are unaffected.
#[must_use]
pub(crate) const fn prefill_stream_slot_bytes_after_full_fit(
    mlx_active_memory_ceiling_bytes: u64,
    fixed_bytes_without_stream_slot: u64,
    complete_expert_payload_bytes: u64,
    stream_slot_bytes: u64,
) -> u64 {
    let expert_budget_without_stream_slot =
        mlx_active_memory_ceiling_bytes.saturating_sub(fixed_bytes_without_stream_slot);
    if complete_expert_payload_bytes > 0
        && expert_budget_without_stream_slot >= complete_expert_payload_bytes
    {
        return 0;
    }
    stream_slot_bytes
}

#[cfg(test)]
#[path = "prefill_stream_slot_tests.rs"]
mod prefill_stream_slot_tests;
