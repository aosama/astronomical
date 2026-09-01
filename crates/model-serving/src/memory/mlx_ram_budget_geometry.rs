//! Pure RAM-budget geometry for measured expert-layer payload facts.
//!
//! Family measurement adapters read disk manifests and supply per-layer payload
//! facts here; this module owns the byte arithmetic that turns those facts into
//! the composed `MlxRamBudgetModelGeometry` and the startup complete-residency
//! headroom requirement. No artifact, SafeTensors, or family type appears here.

use thiserror::Error;

use super::expert_memory_admission::required_complete_residency_activation_headroom_bytes;
use super::mlx_ram_budget::MlxRamBudgetModelGeometry;

/// Why RAM-budget geometry could not be composed from measured layer facts.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum RamBudgetGeometryError {
    #[error("this artifact has no sparse expert payload")]
    NotSparseMixtureOfExperts,
    #[error("expert payload accounting overflowed")]
    ExpertPayloadOverflow,
}

/// One measured sparse-expert decoder layer's complete payload facts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MeasuredExpertLayerPayload {
    /// Total bytes of every expert weight in the layer when fully resident.
    complete_payload_bytes: u64,
    /// How many experts the layer holds.
    expert_capacity: usize,
}

impl MeasuredExpertLayerPayload {
    #[must_use]
    pub const fn new(complete_payload_bytes: u64, expert_capacity: usize) -> Self {
        Self {
            complete_payload_bytes,
            expert_capacity,
        }
    }

    #[must_use]
    pub const fn complete_payload_bytes(&self) -> u64 {
        self.complete_payload_bytes
    }

    #[must_use]
    pub const fn expert_capacity(&self) -> usize {
        self.expert_capacity
    }
}

/// Composes model geometry and startup headroom from measured expert layers.
///
/// `complete_residency_transient_bytes` is the transient payload the measurement
/// owner expects to coexist with complete residency (for example the largest
/// resident gate-up fusion); it feeds the activation-headroom requirement.
#[must_use]
pub fn mlx_ram_budget_model_geometry_from_measured_layer_facts(
    measured_layer_payloads: &[MeasuredExpertLayerPayload],
    total_model_payload_bytes: u64,
    experts_per_token: usize,
    complete_residency_transient_bytes: u64,
) -> Result<(MlxRamBudgetModelGeometry, u64), RamBudgetGeometryError> {
    let mut complete_expert_payload_bytes = 0_u64;
    let mut largest_complete_expert_layer_bytes = 0_u64;
    for layer_payload in measured_layer_payloads {
        complete_expert_payload_bytes = complete_expert_payload_bytes
            .checked_add(layer_payload.complete_payload_bytes)
            .ok_or(RamBudgetGeometryError::ExpertPayloadOverflow)?;
        largest_complete_expert_layer_bytes =
            largest_complete_expert_layer_bytes.max(layer_payload.complete_payload_bytes);
    }
    if complete_expert_payload_bytes == 0 {
        return Err(RamBudgetGeometryError::NotSparseMixtureOfExperts);
    }
    let largest_routed_expert_page_bytes =
        largest_routed_expert_page_bytes(measured_layer_payloads, experts_per_token)?;
    let model_core_payload_bytes =
        total_model_payload_bytes.saturating_sub(complete_expert_payload_bytes);
    let required_headroom_bytes = required_complete_residency_activation_headroom_bytes(
        complete_residency_transient_bytes,
        0,
    );
    Ok((
        MlxRamBudgetModelGeometry {
            model_core_payload_bytes,
            complete_expert_payload_bytes,
            largest_complete_expert_layer_bytes,
            largest_routed_expert_page_bytes,
            sequence_state_bytes_per_token: 0,
        },
        required_headroom_bytes,
    ))
}

/// Largest exact top-K routed expert page across the measured layers.
fn largest_routed_expert_page_bytes(
    measured_layer_payloads: &[MeasuredExpertLayerPayload],
    experts_per_token: usize,
) -> Result<u64, RamBudgetGeometryError> {
    measured_layer_payloads
        .iter()
        .try_fold(0_u64, |largest_page_bytes, layer_payload| {
            let routed_expert_count = experts_per_token.min(layer_payload.expert_capacity);
            let routed_page_bytes = u128::from(layer_payload.complete_payload_bytes)
                .saturating_mul(routed_expert_count as u128)
                / (layer_payload.expert_capacity.max(1) as u128);
            Ok(largest_page_bytes.max(u64::try_from(routed_page_bytes).unwrap_or(u64::MAX)))
        })
}
