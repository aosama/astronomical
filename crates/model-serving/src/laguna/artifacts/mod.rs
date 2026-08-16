mod artifact_error;
mod artifact_validator;
mod canonical_tensor_contract;
mod direct_storage_validation;
mod exact_storage_binding;
mod exact_storage_validation;
mod expected_tensors;
mod kv_cache_metadata;
mod raw_tensor_name_parser;
mod retained_artifact;
mod shard_index;
mod storage_fingerprint;
mod template_source_validator;
mod tensor_assembly;
mod tensor_id;
mod tensor_name_contract;
mod tensor_name_error;
mod tensor_name_normalizer;
mod tensor_storage;

pub use artifact_error::{LagunaArtifactValidationError, LagunaShardIndexError};
pub use artifact_validator::LagunaArtifactValidator;
pub use canonical_tensor_contract::{
    LagunaCanonicalSourceLayout, LagunaCanonicalTensorAssemblyKind,
    LagunaCanonicalTensorDescriptor, LagunaNonExecutableMetadataDescriptor, LagunaTensorContract,
    LagunaTensorSourceDescriptor, LagunaTensorSourceRole,
};
#[cfg(feature = "direct-mlx")]
pub(crate) use expected_tensors::laguna_canonical_module_name;
pub use retained_artifact::{
    LagunaIndexTotalSizeSemantics, LagunaRetainedArtifactFiles, ValidatedLagunaArtifact,
};
pub use shard_index::LagunaShardIndex;
pub use tensor_assembly::{LagunaRawTensorNameRecord, LagunaTensorAssembly, LagunaTensorSource};
pub use tensor_id::{
    LagunaAttentionProjection, LagunaExpertProjection, LagunaGlobalTensorRole,
    LagunaLayerTensorRole, LagunaTensorComponent, LagunaTensorId,
};
pub use tensor_name_contract::{LagunaExpertGateUpLayout, LagunaTensorNameContract};
pub use tensor_name_error::LagunaTensorNameNormalizationError;
pub use tensor_name_normalizer::LagunaTensorNameNormalizer;
pub use tensor_storage::LagunaTensorStorageEncoding;
