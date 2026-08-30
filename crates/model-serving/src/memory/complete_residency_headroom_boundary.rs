//! Ceiling that fits static expert weights but not activation headroom.
//!
//! Callers supply artifact geometry and the startup headroom policy already
//! decided. This module does not know model ids or gigabyte souvenirs.

use super::MlxRamBudgetModelGeometry;

/// Static complete-residency projection plus the extra bytes a forward still needs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompleteResidencyHeadroomBoundary {
    /// Model core plus complete expert payload, with no activation reserve.
    pub static_complete_residency_bytes: u64,
    /// Exact complete expert payload; zero means this is not a sparse MoE boundary.
    pub complete_expert_payload_bytes: u64,
    /// Startup activation headroom required before complete residency may promote.
    pub required_headroom_bytes: u64,
}

impl CompleteResidencyHeadroomBoundary {
    /// Composes the boundary from validated payload geometry and policy headroom.
    #[must_use]
    pub const fn from_model_geometry(
        geometry: MlxRamBudgetModelGeometry,
        required_headroom_bytes: u64,
    ) -> Self {
        Self {
            static_complete_residency_bytes: geometry
                .model_core_payload_bytes
                .saturating_add(geometry.complete_expert_payload_bytes),
            complete_expert_payload_bytes: geometry.complete_expert_payload_bytes,
            required_headroom_bytes,
        }
    }

    /// Ceiling that still covers the expert payload but not expert payload plus headroom.
    ///
    /// Startup admission projects `idle_active + complete_experts + headroom`. Idle active
    /// is measured after load and is often smaller than disk core bytes, so this ceiling is
    /// anchored on the expert payload, not core+experts. Returns `None` when the gap does not exist.
    #[must_use]
    pub const fn paging_ceiling_bytes(self) -> Option<u64> {
        if self.complete_expert_payload_bytes == 0 || self.required_headroom_bytes == 0 {
            None
        } else {
            Some(
                self.complete_expert_payload_bytes
                    .saturating_add(self.required_headroom_bytes)
                    .saturating_sub(1),
            )
        }
    }
}
