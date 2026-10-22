//! Budget policy: "how many bytes does this phase own or need?"
//!
//! `ram.rs` is the single owner that composes the production split of the
//! active-memory ceiling (context reserve, activation headroom,
//! complete-layer stream slot, retained-expert budget) and learns from
//! completed forwards. `adaptive_growth.rs` guards transient growth windows;
//! `ram_geometry.rs` turns measured artifact layer facts into model geometry.
//! The runtime-backed per-allocation admission owner (`live_allocation`)
//! samples the MLX runtime and stays beside the bindings in
//! `astronomical-model-serving`.

pub mod adaptive_growth;
pub mod adaptive_growth_projection;
pub mod ram;
pub mod ram_activation;
pub mod ram_geometry;
pub mod ram_values;

#[cfg(test)]
mod geometry_memory_floor_tests;
#[cfg(test)]
mod mlx_ram_budget_ceiling_adjustment_tests;
#[cfg(test)]
mod mlx_ram_budget_tests;

pub use adaptive_growth::{
    AdaptiveRamGrowthContext, AdaptiveRamGrowthExecutionProfile, AdaptiveRamGrowthGuard,
    AdaptiveRamGrowthGuardError,
};
pub use adaptive_growth_projection::{
    AdaptiveRamGrowthProjection, AdaptiveRamGrowthTransientReserveSource,
};
pub use ram::{
    BOOTSTRAP_CONTEXT_WINDOW_RESERVE_BYTES, MlxRamBudget, measured_non_expert_forward_growth_bytes,
    measured_non_expert_forward_growth_bytes_excluding_expert_page_streaming,
};
pub use ram_geometry::{
    MeasuredExpertLayerPayload, RamBudgetGeometryError,
    mlx_ram_budget_model_geometry_from_measured_layer_facts,
};
pub use ram_values::{
    MlxRamBudgetError, MlxRamBudgetMeasurement, MlxRamBudgetModelGeometry, MlxRamBudgetSnapshot,
};
