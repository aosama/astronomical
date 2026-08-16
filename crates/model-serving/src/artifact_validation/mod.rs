mod bounded_safetensors;
mod error;
mod raw_safetensors_inventory;
mod required_files;
mod safetensors_dtype;
mod types;
mod validated_artifact;

pub(crate) use bounded_safetensors::{
    PartialProfileMetadata, validate_bounded_safetensors_with_indexed_profiles,
    validate_bounded_safetensors_with_partial_profiles,
};
pub use error::ArtifactValidationError;
pub(crate) use raw_safetensors_inventory::RawSafetensorsInventory;
#[doc(hidden)]
pub use required_files::validate_required_file_for_tests;
pub(crate) use required_files::{
    hugging_face_snapshot_model_id, read_bounded_required_file_bytes, validate_required_file,
    validate_required_files,
};
pub use types::{RequiredFileProfile, TensorDtype, TensorProfile};
pub(crate) use validated_artifact::ValidatedRequiredFile;
pub use validated_artifact::{
    RawSafetensorsInventoryForTests, RawSafetensorsTensorDescriptorForTests, ValidatedWeightsFile,
};
