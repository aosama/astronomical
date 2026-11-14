//! Budget policy owner in the serving crate: the runtime-backed
//! per-allocation admission (`live_allocation`) samples the MLX runtime and
//! stays beside the bindings. The pure budget arithmetic (composition,
//! activation evidence, adaptive growth, measured geometry) lives in
//! `astronomical-memory-policy::budget` and is re-exported here so consumer
//! paths stay stable.

#[cfg(feature = "direct-mlx")]
mod live_allocation;

pub use astronomical_memory_policy::budget::{
    MeasuredExpertLayerPayload, MlxRamBudget, MlxRamBudgetError, MlxRamBudgetModelGeometry,
    MlxRamBudgetSnapshot, RamBudgetGeometryError,
    mlx_ram_budget_model_geometry_from_measured_layer_facts,
};
#[cfg(feature = "direct-mlx")]
pub use live_allocation::{MlxAllocationAdmission, MlxAllocationAdmissionError};
