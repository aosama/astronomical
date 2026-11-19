//! The runtime-backed check made before a page of experts is read
//! (`live_allocation`), which samples the MLX runtime and so stays beside
//! the bindings. The facts a memory governor is opened with live in
//! `memory-policy::model_memory` and are re-exported here so
//! consumer paths stay stable.

#[cfg(feature = "direct-mlx")]
mod live_allocation;

#[cfg(feature = "direct-mlx")]
pub use live_allocation::{MlxAllocationAdmission, MlxAllocationAdmissionError};
pub use memory_policy::model_memory::{
    MeasuredExpertLayerPayload, MemoryReserveSnapshot, ModelMemoryFacts, ModelMemoryFactsError,
    ModelMemoryGeometry, ModelMemoryGeometryError, model_memory_geometry_from_measured_layer_facts,
};
