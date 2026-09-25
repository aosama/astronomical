//! Typed, path-free rejections for Qwen-Image-2.1 artifact validation.
//!
//! Every message names the file or tensor that failed using artifact-relative names only, so
//! a rejection never leaks a local directory into a log or a REST response.

use thiserror::Error;

use crate::qwen_image_21::configuration::QwenImage21ConfigError;

#[derive(Debug, Error)]
pub enum QwenImage21ArtifactError {
    #[error("Qwen-Image-2.1 artifact directory is unavailable")]
    ModelDirectoryUnavailable,
    #[error("unsupported Qwen-Image-2.1 model, revision, or license provenance")]
    UnsupportedProvenance {
        model_id: String,
        revision: String,
        license_identifier: String,
    },
    #[error("required Qwen-Image-2.1 artifact file '{file_name}' is unavailable or invalid")]
    ArtifactFile { file_name: String },
    #[error("Qwen-Image-2.1 configuration is incompatible")]
    Configuration(#[from] QwenImage21ConfigError),
    #[error("malformed Qwen-Image-2.1 {component} shard index")]
    MalformedShardIndex {
        component: &'static str,
        #[source]
        source: serde_json::Error,
    },
    #[error(
        "Qwen-Image-2.1 {component} shard index references an unexpected shard name for '{tensor_name}'"
    )]
    UnexpectedShardName {
        component: &'static str,
        tensor_name: String,
        shard_file_name: String,
    },
    #[error("Qwen-Image-2.1 {component} shard index total_size disagrees with physical payload")]
    ShardIndexTotalSizeMismatch {
        component: &'static str,
        declared_bytes: u64,
        actual_bytes: u64,
    },
    #[error("Qwen-Image-2.1 {component} tensor '{tensor_name}' must use the reviewed dtype")]
    TensorDtype {
        component: &'static str,
        tensor_name: String,
    },
    #[error(
        "Qwen-Image-2.1 {component} tensor '{tensor_name}' shape does not match model configuration"
    )]
    TensorShape {
        component: &'static str,
        tensor_name: String,
    },
    #[error(
        "Qwen-Image-2.1 {component} profile for tensor '{tensor_name}' declares unsupported dtype '{declared_dtype}'"
    )]
    UnsupportedProfileDtype {
        component: &'static str,
        tensor_name: String,
        declared_dtype: String,
    },
    #[error("Qwen-Image-2.1 {component} tensor inventory is missing '{tensor_name}'")]
    MissingTensor {
        component: &'static str,
        tensor_name: String,
    },
    #[error("Qwen-Image-2.1 {component} tensor inventory contains unsupported '{tensor_name}'")]
    UnsupportedTensor {
        component: &'static str,
        tensor_name: String,
    },
}
