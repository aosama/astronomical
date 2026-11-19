//! Pure checked arithmetic for expert-memory ownership transitions.
//!
//! This module deliberately knows nothing about MLX, model identities, tensor
//! layouts, or hardware. Runtime code supplies measured byte counts; these
//! formulas decide only whether those counts are internally consistent and how
//! much elastic expert retention must yield.

use thiserror::Error;

/// Invalid byte accounting that must fail closed before changing ownership.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum ExpertMemoryAdmissionError {
    #[error("retained expert payload exceeds current active memory")]
    RetainedExpertPayloadExceedsActiveMemory,
    #[error("complete expert residency projection overflowed")]
    CompleteResidencyProjectionOverflow,
}

/// Projects active memory after replacing paged experts with complete experts.
///
/// `current_active_memory_bytes` already includes
/// `retained_paged_expert_payload_bytes`. Complete residency replaces that owner;
/// it does not coexist with it. Therefore:
///
/// `projected = current_active - retained_paged + complete`
pub fn projected_active_memory_after_complete_expert_replacement(
    current_active_memory_bytes: u64,
    retained_paged_expert_payload_bytes: u64,
    complete_expert_payload_bytes: u64,
) -> Result<u64, ExpertMemoryAdmissionError> {
    current_active_memory_bytes
        .checked_sub(retained_paged_expert_payload_bytes)
        .ok_or(ExpertMemoryAdmissionError::RetainedExpertPayloadExceedsActiveMemory)?
        .checked_add(complete_expert_payload_bytes)
        .ok_or(ExpertMemoryAdmissionError::CompleteResidencyProjectionOverflow)
}

/// Activation headroom required before complete expert residency may promote.
///
/// Complete residency only budgets static expert and non-expert payload. Serving
/// still needs temporary activations, key-value growth, and workspace memory.
/// Prefer the observed transient high-water from completed forwards; otherwise
/// reserve one complete layer. A tenth of the whole expert payload withheld
/// several gigabytes and forced SSD streaming when the user had already given
/// enough RAM to seat the model.
#[must_use]
pub const fn required_complete_residency_activation_headroom_bytes(
    startup_activation_floor_bytes: u64,
    observed_transient_high_water_bytes: u64,
) -> u64 {
    if observed_transient_high_water_bytes > startup_activation_floor_bytes {
        observed_transient_high_water_bytes
    } else {
        startup_activation_floor_bytes
    }
}

/// Returns true when static residency plus activation headroom exceeds the ceiling.
#[must_use]
pub const fn complete_residency_exceeds_ceiling_with_activation_headroom(
    projected_resident_active_memory_bytes: u64,
    stable_memory_ceiling_bytes: u64,
    required_activation_headroom_bytes: u64,
) -> bool {
    match projected_resident_active_memory_bytes.checked_add(required_activation_headroom_bytes) {
        Some(projected_with_headroom_bytes) => {
            projected_with_headroom_bytes > stable_memory_ceiling_bytes
        }
        // Overflow means the projection already cannot fit any finite ceiling.
        None => true,
    }
}
