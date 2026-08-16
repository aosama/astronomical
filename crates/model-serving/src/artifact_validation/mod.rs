mod bounded_safetensors;
mod error;
mod raw_safetensors_inventory;
mod required_files;
mod safetensors_dtype;
mod tensor_inventory;
mod types;
mod validated_artifact;
mod validated_safetensors_source;

pub(crate) use bounded_safetensors::{
    PartialProfileMetadata, validate_bounded_safetensors_with_partial_profiles,
};
pub use error::ArtifactValidationError;
pub(crate) use raw_safetensors_inventory::RawSafetensorsInventory;
#[doc(hidden)]
pub use required_files::validate_required_file_for_tests;
pub(crate) use required_files::{
    hugging_face_snapshot_model_id, read_bounded_required_file_bytes, validate_required_file,
    validate_required_files,
};
pub use tensor_inventory::{
    TensorDeclarationOrigin, TensorFeature, TensorInventory, TensorInventoryError, TensorLocation,
    TensorSemanticRole, TensorSourceId,
};
pub use types::{RequiredFileProfile, TensorDtype, TensorProfile};
pub(crate) use validated_artifact::ValidatedRequiredFile;
pub use validated_artifact::{
    RawSafetensorsInventoryForTests, RawSafetensorsTensorDescriptorForTests, ValidatedWeightsFile,
};
pub(crate) use validated_safetensors_source::ValidatedSafetensorsSource;
#[doc(hidden)]
pub use validated_safetensors_source::validate_safetensors_profile_partitions_for_tests;
