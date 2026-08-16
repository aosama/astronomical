use std::net::SocketAddr;
use std::path::PathBuf;

/// Failure while loading or resolving Astronomical runtime configuration.
#[derive(Debug, thiserror::Error)]
pub enum AstronomicalConfigError {
    #[error("HOME is required to derive the Astronomical instance directory")]
    HomeDirectoryRequired,
    #[error("failed to resolve HOME at {home_directory:?}")]
    ResolveHomeDirectory {
        home_directory: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("HOME must not resolve to the filesystem root")]
    HomeDirectoryMustNotBeRoot,
    #[error("runtime instance must be 'stable' or 'development', got '{raw_instance}'")]
    InvalidRuntimeInstance { raw_instance: String },
    #[error("invalid {field_name}: {description}")]
    InvalidChunkingValue {
        field_name: &'static str,
        description: &'static str,
    },
    #[error("failed to read Astronomical config file at {config_file_path:?}")]
    ReadConfigFile {
        config_file_path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse Astronomical config file at {config_file_path:?}")]
    ParseConfigFile {
        config_file_path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("{field_name} must be an absolute path, got {configured_path:?}")]
    PathMustBeAbsolute {
        field_name: String,
        configured_path: PathBuf,
    },
    #[error(
        "chunking.fixed_prompt_processing_chunk_size_tokens is required when chunking.prompt_processing_chunk_size_optimizer_enabled is false"
    )]
    FixedPromptProcessingChunkSizeTokensRequiredWhenOptimizerDisabled,
    #[error("chunking.fixed_prompt_processing_chunk_size_tokens must be positive")]
    InvalidFixedPromptProcessingChunkSizeTokens,
    #[error("chunking.fixed_ssd_streaming_prompt_processing_chunk_size_tokens must be positive")]
    InvalidFixedSsdStreamingPromptProcessingChunkSizeTokens,
    #[error(
        "chunking.prompt_processing_chunk_size_optimizer_candidate_token_counts must not be empty"
    )]
    OptimizerCandidateTokenCountsMustNotBeEmpty,
    #[error(
        "chunking.prompt_processing_chunk_size_optimizer_candidate_token_counts must contain only positive values"
    )]
    OptimizerCandidateTokenCountsMustBePositive,
    #[error(
        "chunking.prompt_processing_chunk_size_optimizer_candidate_token_counts must be strictly increasing"
    )]
    OptimizerCandidateTokenCountsMustBeStrictlyIncreasing,
    #[error("failed to parse supervisor bind address '{raw_bind_address}'")]
    ParseBindAddress {
        raw_bind_address: String,
        #[source]
        source: std::net::AddrParseError,
    },
    #[error("refusing to bind supervisor to non-loopback address {supervisor_bind_address}")]
    NonLoopbackBindAddress { supervisor_bind_address: SocketAddr },
    #[error(
        "standard runtime instance must bind to {expected_bind_address}, not {configured_bind_address}"
    )]
    StandardInstanceBindAddressMismatch {
        configured_bind_address: SocketAddr,
        expected_bind_address: SocketAddr,
    },
    #[error("invalid prompt cache max_size_gb: {description}")]
    InvalidPromptCacheMaxSizeGb { description: &'static str },
    #[error("invalid maximum_mlx_memory_gb: {description}")]
    InvalidMaximumMlxMemoryGb { description: &'static str },
    #[error("mtp_draft_depth must be between 1 and 3")]
    InvalidMtpDraftDepth,
    #[error("Astronomical config file at {config_file_path:?} must contain a JSON object")]
    ConfigFileMustBeJsonObject { config_file_path: PathBuf },
    #[error("failed to serialize Astronomical config file at {config_file_path:?}")]
    SerializeConfigFile {
        config_file_path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("failed to write Astronomical config file at {config_file_path:?}")]
    WriteConfigFile {
        config_file_path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("logging.retained_files must be positive")]
    InvalidRetainedLogFileCount,
    #[error("speculative_prefill.draft_model_id is required when speculative prefill is enabled")]
    SpeculativePrefillDraftModelRequired,
    #[error("speculative_prefill.draft_model_id must not be empty")]
    SpeculativePrefillDraftModelIdMustNotBeEmpty,
    #[error("speculative_prefill.target_model_id is required when speculative prefill is enabled")]
    SpeculativePrefillTargetModelRequired,
    #[error("speculative_prefill.target_model_id must not be empty")]
    SpeculativePrefillTargetModelIdMustNotBeEmpty,
    #[error("speculative_prefill.minimum_prompt_tokens must be positive")]
    SpeculativePrefillMinimumPromptTokensMustBePositive,
    #[error("speculative_prefill.keep_percentage is required when speculative prefill is enabled")]
    SpeculativePrefillKeepPercentageRequired,
    #[error("speculative_prefill.keep_percentage must be between 1 and 100")]
    SpeculativePrefillKeepPercentageOutOfRange,
    #[error("speculative_prefill.selection_chunck_token_count must be positive")]
    SpeculativePrefillSelectionChunckTokenCountMustBePositive,
    #[error("speculative_prefill.mandatory_trailing_token_count must be positive")]
    SpeculativePrefillMandatoryTrailingTokenCountMustBePositive,
    #[error("speculative_prefill.lookahead_token_count must be positive")]
    SpeculativePrefillLookaheadTokenCountMustBePositive,
    #[error("speculative_prefill.importance_pooling_kernel_token_count must be positive")]
    SpeculativePrefillImportancePoolingKernelTokenCountMustBePositive,
}
