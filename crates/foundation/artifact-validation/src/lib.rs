//! Checks that a model folder holds what a model family needs before any
//! weight is loaded (issue #2340): required files read with bounded sizes,
//! safetensors headers parsed and checked against each family's tensor
//! profiles, and the tensor inventory the loaders then read from.
//!
//! Every model family validates its artifact through this crate, so it sits
//! below them all and depends on no family and on no MLX binding; the
//! `direct-mlx` feature only keeps the absolute file ranges the MLX loaders
//! read.

mod bounded_safetensors;
mod error;
mod raw_safetensors_inventory;
mod required_files;
mod safetensors_dtype;
mod shared_blob_resolution;
pub mod tensor_inventory;
pub mod types;
pub mod validated_artifact;
mod validated_safetensors_source;

pub use bounded_safetensors::{
    PartialProfileMetadata, validate_bounded_safetensors_with_partial_profiles,
};
pub use error::ArtifactValidationError;
pub use raw_safetensors_inventory::{
    RawSafetensorsInventory, RawSafetensorsTensorDescriptor, read_raw_safetensors_inventory,
};
#[doc(hidden)]
pub use required_files::validate_required_file_for_tests;
pub use required_files::{
    hugging_face_snapshot_model_id, read_bounded_required_file_bytes, validate_required_file,
    validate_required_files,
};
pub use tensor_inventory::{
    TensorDeclarationOrigin, TensorInventory, TensorLocation, TensorSemanticRole, TensorSourceId,
};
pub use types::{RequiredFileProfile, TensorDtype, TensorProfile};
pub use validated_artifact::ValidatedRequiredFile;
pub use validated_artifact::{
    RawSafetensorsInventoryForTests, RawSafetensorsTensorDescriptorForTests, ValidatedWeightsFile,
};
pub use validated_safetensors_source::ValidatedSafetensorsSource;
#[cfg(feature = "direct-mlx")]
pub use validated_safetensors_source::ValidatedStoredTensorRange;
#[doc(hidden)]
pub use validated_safetensors_source::validate_safetensors_required_profiles_for_tests;
