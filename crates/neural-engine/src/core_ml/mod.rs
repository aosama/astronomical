//! The Core ML runtime: a model document or compiled program loaded for chosen
//! compute units, Apple's report of where it was placed, its predictions, and
//! its IOSurfaces.

mod compiled_model_files;
pub mod compute_device_report;
mod compute_units;
mod error;
mod prediction;
mod projection;
mod surfaces;

pub use compute_units::{ComputePlacement, CoreMlComputeUnits, LayerPlacement};
pub use error::CoreMlProjectionError;
pub use prediction::PendingPrediction;
pub use projection::CoreMlProjection;
pub use surfaces::LockedOutputSurface;
