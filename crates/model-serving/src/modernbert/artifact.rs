//! Loads and validates one ModernBERT embedding artifact before GPU allocation.
//!
//! Startup rereads the exact on-disk evidence: bounded config geometry, a
//! loadable tokenizer, and one safetensors payload. Tensor inventory shape
//! checks happen when the engine binds lazy arrays by name.

use std::path::Path;

use astronomical_ipc_protocol::EmbeddingsFailureReason;
use thiserror::Error;

use crate::modernbert::configuration::ModernBertConfiguration;

const MAXIMUM_CONFIG_JSON_BYTES: u64 = 1_048_576;
const MAXIMUM_TOKENIZER_JSON_BYTES: u64 = 16_777_216;

/// Everything the engine needs to run one embedding forward pass.
pub struct ModernBertArtifact {
    pub configuration: ModernBertConfiguration,
    pub tokenizer_bytes: Vec<u8>,
    pub weights_file: std::fs::File,
}

impl ModernBertArtifact {
    /// Validates shallow artifact completeness without reading model bytes.
    pub fn load(model_directory: &Path) -> Result<Self, ModernBertArtifactError> {
        let config_json_bytes = bounded_read(
            model_directory.join("config.json"),
            MAXIMUM_CONFIG_JSON_BYTES,
        )?;
        let config_document: serde_json::Value = serde_json::from_slice(&config_json_bytes)?;
        let configuration = ModernBertConfiguration::from_config_value(&config_document)?;
        let tokenizer_bytes = bounded_read(
            model_directory.join("tokenizer.json"),
            MAXIMUM_TOKENIZER_JSON_BYTES,
        )?;
        let weights_file = std::fs::File::open(model_directory.join("model.safetensors"))
            .map_err(|source| ModernBertArtifactError::MissingWeights { source })?;
        Ok(Self {
            configuration,
            tokenizer_bytes,
            weights_file,
        })
    }
}

fn bounded_read(
    file_path: impl AsRef<std::path::Path>,
    maximum_bytes: u64,
) -> Result<Vec<u8>, ModernBertArtifactError> {
    let opened_file = std::fs::File::open(&file_path)
        .map_err(|source| ModernBertArtifactError::MissingConfig { source })?;
    let actual_file_bytes = opened_file
        .metadata()
        .map_err(|source| ModernBertArtifactError::MissingConfig { source })?
        .len();
    if actual_file_bytes > maximum_bytes {
        return Err(ModernBertArtifactError::ConfigTooLarge);
    }
    std::fs::read(&file_path).map_err(|source| ModernBertArtifactError::MissingConfig { source })
}

/// Maps a bounded configuration validation failure into the artifact error set.
impl From<crate::modernbert::configuration::ModernBertConfigurationError>
    for ModernBertArtifactError
{
    fn from(
        configuration_error: crate::modernbert::configuration::ModernBertConfigurationError,
    ) -> Self {
        Self::MalformedConfig(serde_json::Error::io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            configuration_error.to_string(),
        )))
    }
}

#[derive(Debug, Error)]
pub enum ModernBertArtifactError {
    #[error("ModernBERT embedding artifact config.json is missing or unreadable: {source}")]
    MissingConfig { source: std::io::Error },
    #[error("ModernBERT embedding artifact config.json exceeds the bounded classification limit")]
    ConfigTooLarge,
    #[error("ModernBERT embedding artifact config.json is malformed: {0}")]
    MalformedConfig(#[from] serde_json::Error),
    #[error("ModernBERT embedding artifact model.safetensors is missing: {source}")]
    MissingWeights { source: std::io::Error },
}

/// Maps a bounded artifact validation failure into a worker-admissible failure reason.
impl From<ModernBertArtifactError> for EmbeddingsFailureReason {
    fn from(artifact_error: ModernBertArtifactError) -> Self {
        EmbeddingsFailureReason::FatalExecution {
            reason: format!("embedding artifact initialization failed: {artifact_error}")
                .chars()
                .take(256)
                .collect(),
        }
    }
}
