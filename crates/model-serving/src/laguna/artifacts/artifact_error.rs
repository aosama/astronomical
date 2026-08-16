use ::safetensors::Dtype;
use thiserror::Error;

use super::tensor_id::LagunaTensorId;
use crate::artifact_validation::ArtifactValidationError;
use crate::laguna::{LagunaNormalizationError, LagunaTextArtifactError};

use super::tensor_name_error::LagunaTensorNameNormalizationError;

/// A bounded structural failure in the Laguna-owned shard index.
#[derive(Debug, Error)]
pub enum LagunaShardIndexError {
    #[error(
        "Laguna shard index has {actual_bytes} bytes, exceeding the {maximum_bytes}-byte limit"
    )]
    IndexTooLarge {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    #[error("failed to decode the Laguna shard index")]
    MalformedIndex(#[source] serde_json::Error),
    #[error("Laguna shard index contains duplicate tensor name '{tensor_name}'")]
    DuplicateTensorName { tensor_name: String },
    #[error(
        "Laguna shard index has {actual_count} tensors, exceeding the {maximum_count}-tensor limit"
    )]
    TensorCountTooLarge {
        actual_count: usize,
        maximum_count: usize,
    },
    #[error(
        "Laguna shard index tensor name has {actual_bytes} bytes, expected 1..={maximum_bytes}"
    )]
    InvalidTensorNameLength {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    #[error("Laguna shard index contains unsafe shard file name '{shard_file_name}'")]
    UnsafeShardFileName { shard_file_name: String },
}

/// A cause-preserving failure before any Laguna model construction.
#[derive(Debug, Error)]
pub enum LagunaArtifactValidationError {
    #[error("Laguna model artifact directory is unavailable")]
    ModelDirectoryUnavailable,
    #[error("Laguna retained-file or SafeTensors validation failed")]
    Artifact(#[from] ArtifactValidationError),
    #[error("Laguna configuration normalization failed")]
    Configuration(#[from] LagunaNormalizationError),
    #[error("Laguna text artifact normalization failed")]
    TextArtifact(#[from] LagunaTextArtifactError),
    #[error("Laguna shard-index validation failed")]
    ShardIndex(#[from] LagunaShardIndexError),
    #[error("Laguna tensor-name normalization failed")]
    TensorNames(#[from] LagunaTensorNameNormalizationError),
    #[error("indexed Laguna shard '{shard_file_name}' is missing")]
    MissingShard { shard_file_name: String },
    #[error("Laguna tensor '{tensor_name}' is indexed but absent from shard '{shard_file_name}'")]
    IndexedTensorMissing {
        tensor_name: String,
        shard_file_name: String,
    },
    #[error("physical Laguna tensor '{tensor_name}' is absent from the shard index")]
    PhysicalTensorNotIndexed { tensor_name: String },
    #[error(
        "physical Laguna tensor '{tensor_name}' is in shard '{actual_shard_file_name}', but the index assigns '{expected_shard_file_name}'"
    )]
    PhysicalTensorInWrongShard {
        tensor_name: String,
        expected_shard_file_name: String,
        actual_shard_file_name: String,
    },
    #[error(
        "physical Laguna tensor '{tensor_name}' occurs in shards '{first_shard_file_name}' and '{second_shard_file_name}'"
    )]
    DuplicatePhysicalTensor {
        tensor_name: String,
        first_shard_file_name: String,
        second_shard_file_name: String,
    },
    #[error(
        "Laguna index total_size {declared_total_size_bytes} matches neither {actual_shard_file_bytes} serialized shard bytes nor {actual_tensor_payload_bytes} tensor payload bytes"
    )]
    IndexTotalSizeMismatch {
        declared_total_size_bytes: u64,
        actual_shard_file_bytes: u64,
        actual_tensor_payload_bytes: u64,
    },
    #[error(
        "Laguna target requires canonical tensor {tensor_id:?}, but the inventory does not provide it"
    )]
    ExpectedTensorMissing { tensor_id: LagunaTensorId },
    #[error(
        "Laguna inventory provides canonical tensor {tensor_id:?}, but the target does not execute it"
    )]
    UnexpectedCanonicalTensor { tensor_id: LagunaTensorId },
    #[error(
        "Laguna tensor {tensor_id:?} shape mismatch: expected {expected_shape:?}, got {actual_shape:?}"
    )]
    TensorShapeMismatch {
        tensor_id: LagunaTensorId,
        expected_shape: Vec<usize>,
        actual_shape: Vec<usize>,
    },
    #[error(
        "Laguna affine tensor {tensor_id:?} cannot represent logical input width {logical_input_width} exactly with {bit_width} bits and group size {group_size}"
    )]
    InvalidAffineDimension {
        tensor_id: LagunaTensorId,
        logical_input_width: usize,
        bit_width: u32,
        group_size: u32,
    },
    #[error(
        "Laguna tensor {tensor_id:?} physical dtype mismatch: expected {expected_dtype:?}, got {actual_dtype:?}"
    )]
    TensorDtypeMismatch {
        tensor_id: LagunaTensorId,
        expected_dtype: Dtype,
        actual_dtype: Dtype,
    },
    #[error("Laguna tensor {tensor_id:?} assembly has inconsistent source dtypes")]
    MixedAssemblyDtypes { tensor_id: LagunaTensorId },
    #[error("Laguna tensor {tensor_id:?} has no physical source descriptors")]
    EmptyTensorAssembly { tensor_id: LagunaTensorId },
    #[error("Laguna canonical tensor {tensor_id:?} lost physical source '{tensor_name}'")]
    CanonicalSourceMissing {
        tensor_id: LagunaTensorId,
        tensor_name: String,
    },
    #[error("Laguna canonical tensor geometry overflowed this platform")]
    TensorGeometryOverflow,
    #[error("Laguna aggregate retained shard-file bytes overflowed u64")]
    ShardFileSizeAccountingOverflow,
    #[error("Laguna aggregate tensor payload bytes overflowed u64")]
    TensorPayloadAccountingOverflow,
    #[error(
        "Laguna affine override '{module_name}' resolves to {resolved_module_count} executable canonical modules, expected exactly one"
    )]
    AffineOverrideResolution {
        module_name: String,
        resolved_module_count: usize,
    },
    #[error("Laguna tensor {tensor_id:?} declares an unsupported zero point for symmetric storage")]
    UnsupportedAsymmetricStorage { tensor_id: LagunaTensorId },
    #[error(
        "Laguna block-FP8 tensor {tensor_id:?} cannot cover logical shape {logical_shape:?} with scale shape {scale_shape:?}"
    )]
    InvalidBlockFp8Coverage {
        tensor_id: LagunaTensorId,
        logical_shape: Vec<usize>,
        scale_shape: Vec<usize>,
    },
    #[error(
        "Laguna block-FP8 tensor {tensor_id:?} derives {actual_block_row_extent}x{actual_block_column_extent} blocks, but config declares {declared_block_row_extent}x{declared_block_column_extent}"
    )]
    BlockFp8GeometryMismatch {
        tensor_id: LagunaTensorId,
        declared_block_row_extent: usize,
        declared_block_column_extent: usize,
        actual_block_row_extent: usize,
        actual_block_column_extent: usize,
    },
}
