use std::path::PathBuf;

use astronomical_runtime_integration::MlxRuntimeError;

use super::block_format_error::PersistentPromptCacheBlockError;

/// One failure while opening, scanning, saving, or loading a persistent prompt-cache block.
///
/// Required prompt-state initialization, restoration, and publication surface these failures at
/// the model or request boundary. The error retains its lower-level cause for bounded diagnostics.
#[derive(Debug, thiserror::Error)]
pub enum PersistentPromptCacheDiskStoreError {
    #[error(
        "persistent prompt-cache parent state was not published before child block {block_index}"
    )]
    ParentStateNotPublished { block_index: u32 },
    #[error("persistent prompt-cache block {block_hash:02x?} has conflicting stored topology")]
    ExistingBlockTopologyMismatch { block_hash: [u8; 32] },
    #[error("persistent prompt-cache block {block_index} has invalid requested ancestry")]
    InvalidRequestedBlockAncestry { block_index: u32 },
    #[error(
        "failed to create persistent prompt-cache directory at {persistent_prompt_cache_directory:?}"
    )]
    CreatePromptCacheDirectory {
        persistent_prompt_cache_directory: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error(
        "active prompt-cache directory {active_model_prompt_cache_directory:?} is outside global root {global_prompt_cache_root_directory:?}"
    )]
    ActivePromptCacheDirectoryOutsideGlobalRoot {
        active_model_prompt_cache_directory: PathBuf,
        global_prompt_cache_root_directory: PathBuf,
    },
    #[error(
        "prompt-cache directory must be a real directory without symlink components: {persistent_prompt_cache_directory:?}"
    )]
    UnsafePromptCacheDirectory {
        persistent_prompt_cache_directory: PathBuf,
    },
    #[error(
        "failed to read persistent prompt-cache directory at {persistent_prompt_cache_directory:?}"
    )]
    ReadPromptCacheDirectory {
        persistent_prompt_cache_directory: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to open temp file at {temp_file_path:?}")]
    OpenTempFile {
        temp_file_path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to write persistent prompt-cache temp file at {temp_file_path:?}")]
    WriteTempFile {
        temp_file_path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to synchronize persistent prompt-cache temp file at {temp_file_path:?}")]
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
        "persistent prompt-cache file {file_path:?} reported {reported_size_bytes} bytes but contains {actual_size_bytes} bytes"
    )]
    WrittenFileSizeMismatch {
        file_path: PathBuf,
        reported_size_bytes: u64,
        actual_size_bytes: u64,
    },
    #[error("unsupported persistent prompt-cache state file name {file_name}")]
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
        source: PersistentPromptCacheBlockError,
    },
    #[error("failed to read prompt-cache block manifest at {manifest_file_path:?}")]
    ReadBlockManifest {
        manifest_file_path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse prompt-cache block manifest at {manifest_file_path:?}")]
    ParseBlockManifest {
        manifest_file_path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("failed to serialize prompt-cache block manifest")]
    SerializeBlockManifest {
        #[source]
        source: serde_json::Error,
    },
    #[error("invalid prompt-cache block manifest at {manifest_file_path:?}: {description}")]
    InvalidBlockManifest {
        manifest_file_path: PathBuf,
        description: String,
    },
    #[error(
        "failed to synchronize prompt-cache directory at {persistent_prompt_cache_directory:?}"
    )]
    SynchronizePromptCacheDirectory {
        persistent_prompt_cache_directory: PathBuf,
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
        "persistent prompt-cache size bound {maximum_size_bytes} bytes cannot fit a new block of {estimated_block_bytes} bytes"
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
        "failed to remove persistent prompt-cache file at {persistent_prompt_cache_file_path:?}"
    )]
    RemovePromptCacheFile {
        persistent_prompt_cache_file_path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error(
        "global prompt-cache size overflowed while scanning {global_prompt_cache_root_directory:?}"
    )]
    GlobalPromptCacheSizeOverflow {
        global_prompt_cache_root_directory: PathBuf,
    },
    #[error(
        "global prompt-cache quota {maximum_size_bytes} bytes remains exceeded by {remaining_size_bytes} bytes after eviction"
    )]
    GlobalPromptCacheQuotaNotSatisfied {
        maximum_size_bytes: u64,
        remaining_size_bytes: u64,
    },
}

impl PersistentPromptCacheDiskStoreError {
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
