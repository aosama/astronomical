//! Budget policy: "how many bytes does this phase own or need?"
//!
//! `ram.rs` is the single owner that composes the production split of the
//! active-memory ceiling (context reserve, activation headroom,
//! complete-layer stream slot, retained-expert budget) and learns from
//! completed forwards. `adaptive_growth.rs` guards transient growth windows;
//! `ram_geometry.rs` turns measured artifact layer facts into model geometry;
//! `live_allocation.rs` is the runtime-backed per-allocation admission owner.

mod adaptive_growth;
#[cfg(feature = "direct-mlx")]
mod live_allocation;
mod ram;
mod ram_geometry;

pub use adaptive_growth::{
    AdaptiveRamGrowthContext, AdaptiveRamGrowthGuard, AdaptiveRamGrowthGuardError,
    AdaptiveRamGrowthProjection,
};
#[cfg(feature = "direct-mlx")]
pub use live_allocation::{MlxAllocationAdmission, MlxAllocationAdmissionError};
#[cfg(feature = "direct-mlx")]
pub(crate) use ram::context_token_bucket;
pub use ram::{
    BOOTSTRAP_CONTEXT_WINDOW_RESERVE_BYTES, MlxRamBudget, MlxRamBudgetError,
    MlxRamBudgetMeasurement, MlxRamBudgetModelGeometry, MlxRamBudgetSnapshot,
    measured_non_expert_forward_growth_bytes,
};
pub use ram_geometry::{
    MeasuredExpertLayerPayload, RamBudgetGeometryError,
    mlx_ram_budget_model_geometry_from_measured_layer_facts,
};
