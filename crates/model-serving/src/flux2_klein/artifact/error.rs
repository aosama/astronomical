//! Why a FLUX.2 Klein artifact was refused. Displays never include a local path.

use thiserror::Error;

use super::super::Flux2KleinConfigError;

/// A bounded artifact rejection whose display never includes a local path.
#[derive(Debug, Error)]
pub enum Flux2KleinArtifactError {
    #[error("FLUX.2 Klein artifact directory is unavailable")]
    ModelDirectoryUnavailable,
    #[error("unsupported FLUX.2 Klein model, revision, or license provenance")]
    UnsupportedProvenance {
        model_id: String,
        revision: String,
        license_identifier: String,
    },
    #[error("required FLUX.2 Klein artifact file '{file_name}' is unavailable or invalid")]
    ArtifactFile { file_name: String },
    #[error("FLUX.2 Klein configuration is incompatible")]
    Configuration(#[from] Flux2KleinConfigError),
    #[error("FLUX.2 Klein LICENSE.md is not the Apache License 2.0 text")]
    InvalidLicense,
    #[error("malformed FLUX.2 Klein text shard index")]
    MalformedTextShardIndex(#[source] serde_json::Error),
    #[error("FLUX.2 Klein shard index contains an unsafe or unsupported shard name")]
    UnsupportedShardName { shard_file_name: String },
    #[error("FLUX.2 Klein text shard index and physical inventory disagree for '{tensor_name}'")]
    TextShardIndexDisagreement { tensor_name: String },
    #[error("FLUX.2 Klein text shard index total_size disagrees with physical tensor payload")]
    TextShardIndexTotalSizeMismatch {
        declared_bytes: u64,
        actual_bytes: u64,
    },
    #[error(
        "FLUX.2 Klein text shard index total_parameters disagrees with physical tensor geometry"
    )]
    TextShardIndexTotalParameterMismatch {
        declared_parameters: u64,
        actual_parameters: u64,
    },
    #[error("FLUX.2 Klein {component} tensor '{tensor_name}' must use BF16 storage")]
    TensorDtype {
        component: &'static str,
        tensor_name: String,
    },
    #[error(
        "FLUX.2 Klein {component} tensor '{tensor_name}' shape does not match model configuration"
    )]
    TensorShape {
        component: &'static str,
        tensor_name: String,
    },
    #[error("FLUX.2 Klein {component} tensor inventory is missing '{tensor_name}'")]
    MissingTensor {
        component: &'static str,
        tensor_name: String,
    },
    #[error("FLUX.2 Klein {component} tensor inventory contains unsupported '{tensor_name}'")]
    UnsupportedTensor {
        component: &'static str,
        tensor_name: String,
    },
    #[error(
        "FLUX.2 Klein {component} shard index and physical inventory disagree for '{tensor_name}'"
    )]
    ShardIndexDisagreement {
        component: &'static str,
        tensor_name: String,
    },
    #[error("FLUX.2 Klein shard index '{file_name}' declares an unsupported quantization")]
    UnsupportedQuantization { file_name: String },
    #[error(
        "FLUX.2 Klein {component} projection '{tensor_name}' is not packed as 4-bit affine with group size 64"
    )]
    QuantizedTensorPacking {
        component: &'static str,
        tensor_name: String,
    },
    #[error("FLUX.2 Klein aggregate tensor payload accounting overflowed")]
    PayloadAccountingOverflow,
}
