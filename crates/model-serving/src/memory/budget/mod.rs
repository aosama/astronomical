//! Budget policy owner in the serving crate: the runtime-backed
//! per-allocation admission (`live_allocation`) samples the MLX runtime and
//! stays beside the bindings. The pure budget arithmetic (composition,
//! activation evidence, adaptive growth, measured geometry) lives in
//! `astronomical-memory-policy::budget` and is re-exported here so consumer
//! paths stay stable.

#[cfg(feature = "direct-mlx")]
mod live_allocation;

#[cfg(feature = "direct-mlx")]
pub use live_allocation::{MlxAllocationAdmission, MlxAllocationAdmissionError};
pub use astronomical_memory_policy::budget::{
    AdaptiveRamGrowthContext, AdaptiveRamGrowthExecutionProfile, AdaptiveRamGrowthGuard,
    AdaptiveRamGrowthGuardError, AdaptiveRamGrowthProjection,
    AdaptiveRamGrowthTransientReserveSource, BOOTSTRAP_CONTEXT_WINDOW_RESERVE_BYTES, MlxRamBudget,
    MlxRamBudgetError, MlxRamBudgetMeasurement, MlxRamBudgetModelGeometry, MlxRamBudgetSnapshot,
    MeasuredExpertLayerPayload, RamBudgetGeometryError, measured_non_expert_forward_growth_bytes,
    measured_non_expert_forward_growth_bytes_excluding_expert_page_streaming,
    mlx_ram_budget_model_geometry_from_measured_layer_facts,
};
