//! The facts about a model the memory governor is opened with.
//!
//! `ram.rs` carries the limit a model runs under and its geometry;
//! `ram_geometry.rs` turns measured artifact layer facts into that
//! geometry; `ram_values.rs` holds the value types, including the snapshot
//! the status breakdown is composed against. How the limit is used is the
//! memory governor's business (`crate::ledger`).

pub mod ram;
pub mod ram_geometry;
pub mod ram_values;

#[cfg(test)]
mod geometry_memory_floor_tests;
#[cfg(test)]
mod ram_geometry_boundary_tests;

pub use ram::MlxRamBudget;
pub use ram_geometry::{
    MeasuredExpertLayerPayload, RamBudgetGeometryError,
    mlx_ram_budget_model_geometry_from_measured_layer_facts,
};
pub use ram_values::{MlxRamBudgetError, MlxRamBudgetModelGeometry, MlxRamBudgetSnapshot};
