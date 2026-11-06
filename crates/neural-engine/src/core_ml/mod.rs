//! The Core ML runtime: a model document or compiled program loaded for chosen
//! compute units, Apple's report of where it was placed, its predictions, and
//! its IOSurfaces. Nothing here knows how a model document is encoded.

mod compiled_model_files;
pub mod compute_device_report;
mod compute_units;
mod error;
pub(crate) mod ffi;
mod prediction;
mod program;
mod surfaces;
mod wired_memory;

pub use compiled_model_files::compile_model_to;
pub use compute_units::{ComputePlacement, CoreMlComputeUnits, LayerPlacement};
pub use error::CoreMlProgramError;
pub use prediction::PendingPrediction;
pub use program::CoreMlProgram;
pub use surfaces::LockedOutputSurface;
pub use wired_memory::{system_wired_bytes, wired_growth_bytes};
