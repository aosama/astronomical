use std::path::PathBuf;

use runtime_integration::MlxRuntimeError;

use super::block_format_error::AstroSsdPromptCacheBlockError;
use super::model_contract_error::AstroSsdPromptCacheModelContractError;

/// One failure while opening, scanning, saving, or loading a persistent astro SSD prompt cache block.
///
/// Required prompt-state initialization, restoration, and publication surface these failures at
/// the model or request boundary. The error retains its lower-level cause for bounded diagnostics.
#[derive(Debug, thiserror::Error)]
pub enum AstroSsdPromptCacheDiskStoreError {
    #[error(
        "persistent astro-ssd-prompt-cache parent state was not published before child block {block_index}"
    )]
    ParentStateNotPublished { block_index: u32 },
    #[error(
        "persistent astro-ssd-prompt-cache block {block_hash:02x?} has conflicting stored topology"
    )]
    ExistingBlockTopologyMismatch { block_hash: [u8; 32] },
    #[error("persistent astro SSD prompt cache block {block_index} has invalid requested ancestry")]
    InvalidRequestedBlockAncestry { block_index: u32 },
    #[error(
        "failed to create persistent astro-ssd-prompt-cache directory at {astro_ssd_prompt_cache_directory:?}"
    )]
    CreateAstroSsdPromptCacheDirectory {
        astro_ssd_prompt_cache_directory: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error(
        "active astro-ssd-prompt-cache directory {active_model_astro_ssd_prompt_cache_directory:?} is outside global root {global_astro_ssd_prompt_cache_root_directory:?}"
    )]
    ActiveAstroSsdPromptCacheDirectoryOutsideGlobalRoot {
        active_model_astro_ssd_prompt_cache_directory: PathBuf,
        global_astro_ssd_prompt_cache_root_directory: PathBuf,
    },
    #[error(
        "astro-ssd-prompt-cache directory must be a real directory without symlink components: {astro_ssd_prompt_cache_directory:?}"
    )]
    UnsafeAstroSsdPromptCacheDirectory {
        astro_ssd_prompt_cache_directory: PathBuf,
    },
    #[error(
        "failed to read persistent astro-ssd-prompt-cache directory at {astro_ssd_prompt_cache_directory:?}"
    )]
    ReadAstroSsdPromptCacheDirectory {
        astro_ssd_prompt_cache_directory: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to open temp file at {temp_file_path:?}")]
    OpenTempFile {
        temp_file_path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to write persistent astro SSD prompt cache temp file at {temp_file_path:?}")]
    WriteTempFile {
        temp_file_path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error(
        "failed to synchronize persistent astro-ssd-prompt-cache temp file at {temp_file_path:?}"
    )]
    SynchronizeTempFile {
        temp_file_path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to save safetensors block")]
    SaveSafetensors {
        #[source]
        source: MlxRuntimeError,
    },
    #[error("failed to write safetensors through its retained file descriptor")]
    WriteSafetensorsDescriptor {
        #[source]
        source: std::io::Error,
    },
    #[error(
        "decoder-state tensor {tensor_name} has dtype {dtype_description}, which no state file stores"
    )]
    UnsupportedHostTensorDtype {
        tensor_name: String,
        dtype_description: String,
    },
    #[error("failed to serialize the safetensors header of a decoder-state file")]
    SerializeSafetensorsHeader {
        #[source]
        source: serde_json::Error,
    },
    #[error("the background astro_ssd_prompt_cache writer has stopped")]
    PublicationWriterStopped,
    #[error("failed to read MLX memory for persistent model-state admission")]
    ReadMlxMemorySnapshot {
        #[source]
        source: MlxRuntimeError,
    },
    #[error("failed to rename temp file {temp_file_path:?} to {block_file_path:?}")]
    RenameTempFile {
        temp_file_path: PathBuf,
        block_file_path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to read block metadata at {block_file_path:?}")]
    ReadBlockMetadata {
        block_file_path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error(
        "persistent astro-ssd-prompt-cache file {file_path:?} reported {reported_size_bytes} bytes but contains {actual_size_bytes} bytes"
    )]
    WrittenFileSizeMismatch {
        file_path: PathBuf,
        reported_size_bytes: u64,
        actual_size_bytes: u64,
    },
    #[error("unsupported persistent astro SSD prompt cache state file name {file_name}")]
    InvalidStateFileName { file_name: String },
    #[error("failed to open block file at {block_file_path:?}")]
    OpenBlockFile {
        block_file_path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to validate block at {block_file_path:?}: {source}")]
    ValidateBlock {
        block_file_path: PathBuf,
        #[source]
        source: AstroSsdPromptCacheBlockError,
    },
    #[error("failed to read astro SSD prompt cache block manifest at {manifest_file_path:?}")]
    ReadBlockManifest {
        manifest_file_path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse astro SSD prompt cache block manifest at {manifest_file_path:?}")]
    ParseBlockManifest {
        manifest_file_path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("failed to serialize astro SSD prompt cache block manifest")]
    SerializeBlockManifest {
        #[source]
        source: serde_json::Error,
    },
    #[error(
        "invalid astro-ssd-prompt-cache block manifest at {manifest_file_path:?}: {description}"
    )]
    InvalidBlockManifest {
        manifest_file_path: PathBuf,
        description: String,
    },
    #[error(
        "failed to synchronize astro-ssd-prompt-cache directory at {astro_ssd_prompt_cache_directory:?}"
    )]
    SynchronizeAstroSsdPromptCacheDirectory {
        astro_ssd_prompt_cache_directory: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to validate model-specific persistent artifact at {artifact_file_path:?}")]
    ValidateModelSpecificArtifact {
        artifact_file_path: PathBuf,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("failed to load safetensors block")]
    LoadSafetensors {
        #[source]
        source: MlxRuntimeError,
    },
    #[error(
        "persistent astro-ssd-prompt-cache size bound {maximum_size_bytes} bytes cannot fit a new block of {estimated_block_bytes} bytes"
    )]
    SizeBoundExceeded {
        maximum_size_bytes: u64,
        estimated_block_bytes: u64,
    },
    #[error(
        "persistent model-state {state_kind} tensors do not match the storage contract: expected_present={expected_present} actual_tensor_count={actual_tensor_count}"
    )]
    StateKindTensorPresenceMismatch {
        state_kind: &'static str,
        expected_present: bool,
        actual_tensor_count: usize,
    },
    #[error(
        "failed to remove persistent astro-ssd-prompt-cache file at {astro_ssd_prompt_cache_file_path:?}"
    )]
    RemoveAstroSsdPromptCacheFile {
        astro_ssd_prompt_cache_file_path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error(
        "global astro-ssd-prompt-cache size overflowed while scanning {global_astro_ssd_prompt_cache_root_directory:?}"
    )]
    GlobalAstroSsdPromptCacheSizeOverflow {
        global_astro_ssd_prompt_cache_root_directory: PathBuf,
    },
    #[error(
        "global astro-ssd-prompt-cache quota {maximum_size_bytes} bytes remains exceeded by {remaining_size_bytes} bytes after eviction"
    )]
    GlobalAstroSsdPromptCacheQuotaNotSatisfied {
        maximum_size_bytes: u64,
        remaining_size_bytes: u64,
    },
    #[error("failed to compute persistent astro SSD prompt cache state file geometry: {source}")]
    ModelContractGeometry {
        #[source]
        source: AstroSsdPromptCacheModelContractError,
    },
}

impl From<AstroSsdPromptCacheModelContractError> for AstroSsdPromptCacheDiskStoreError {
    fn from(source: AstroSsdPromptCacheModelContractError) -> Self {
        Self::ModelContractGeometry { source }
    }
}

impl AstroSsdPromptCacheDiskStoreError {
    // An astro SSD prompt cache file can disappear or its model-bound header can fail validation after lookup.
    // Treat those pre-materialization failures as misses; payload corruption remains fatal
    // because array restoration may already have started.
    #[must_use]
    pub fn is_treatable_as_astro_ssd_prompt_cache_miss(&self) -> bool {
        match self {
            Self::OpenBlockFile { source, .. }
                if matches!(source.kind(), std::io::ErrorKind::NotFound) =>
            {
                true
            }
            Self::ValidateBlock { .. } | Self::ValidateModelSpecificArtifact { .. } => true,
            _ => false,
        }
    }

    #[must_use]
    pub fn active_memory_deficit_bytes(&self) -> Option<usize> {
        let Self::SaveSafetensors {
            source:
                MlxRuntimeError::ActiveMemoryLimitExceeded {
                    active_memory_bytes,
                    attempted_allocation_bytes,
                    allowed_active_memory_bytes,
                },
        } = self
        else {
            return None;
        };
        Some(
            active_memory_bytes
                .saturating_add(*attempted_allocation_bytes)
                .saturating_sub(*allowed_active_memory_bytes),
        )
    }
}
