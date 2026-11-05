//! Works with the vision towers of open model artifacts: whether a model has one,
//! whether a given tower shares a text model's base, and adding that tower to the
//! text model as a new artifact.
//!
//! The questions read safetensors headers and a bounded sample of small tensors,
//! so a many-gigabyte repository costs a few megabytes to examine. Adding a tower
//! fetches only the tower's own tensors, never the rest of the repository. The
//! crate deliberately has no MLX or Metal dependency: it never runs a model, so it
//! stays cheap to build and the CLI's default binary stays MLX-free.

#![forbid(unsafe_code)]

pub mod artifact_reference;
pub mod artifact_source;
pub mod assembled_documents;
pub mod base_matching;
pub mod build_record;
pub mod error;
pub mod hugging_face_source;
pub mod local_directory_source;
pub mod model_config;
pub mod safetensors_header;
pub mod tensor_agreement;
pub mod tensor_names;
pub mod tensor_values;
pub mod tower_addition;
pub mod tower_file;
pub mod tower_inspection;
pub mod tower_layout;
pub mod weight_inventory;

#[cfg(test)]
mod test_fixtures;

pub use artifact_reference::ArtifactReference;
pub use artifact_source::{ArtifactFile, ArtifactListing, ArtifactSource};
pub use base_matching::{
    BaseMatchProgress, BaseMatchReport, BaseMatchVerdict, ComparedTensor, match_tower_to_text,
};
pub use error::VisionTowerError;
pub use model_config::{ModelConfigSummary, VisionConfigSummary};
pub use tensor_agreement::TensorAgreement;
pub use tower_addition::{
    TowerAdditionProgress, TowerAdditionReport, TowerAdditionRequest, TowerAdditionStage, add_tower,
};
pub use tower_inspection::{TowerInspection, TowerStatus, TowerSummary, inspect_artifact};
