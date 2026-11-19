//! The facts about a model the memory governor is opened with.
//!
//! `model_memory_facts.rs` carries the limit a model runs under and its
//! geometry; `model_memory_geometry.rs` turns measured artifact layer facts
//! into that geometry; `memory_values.rs` holds the value types, including
//! the snapshot the status breakdown is composed against. How the limit is
//! used is the memory governor's business (`crate::ledger`).

pub mod memory_values;
pub mod model_memory_facts;
pub mod model_memory_geometry;

#[cfg(test)]
mod geometry_memory_floor_tests;
#[cfg(test)]
mod model_memory_geometry_boundary_tests;

pub use memory_values::{MemoryReserveSnapshot, ModelMemoryFactsError, ModelMemoryGeometry};
pub use model_memory_facts::ModelMemoryFacts;
pub use model_memory_geometry::{
    MeasuredExpertLayerPayload, ModelMemoryGeometryError,
    model_memory_geometry_from_measured_layer_facts,
};
