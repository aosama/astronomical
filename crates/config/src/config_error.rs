use std::net::SocketAddr;
use std::path::PathBuf;

/// Failure while loading or resolving Astronomical runtime configuration.
#[derive(Debug, thiserror::Error)]
pub enum AstronomicalConfigError {
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
        "chunking.fixed_prefill_tokens is required when chunking.prefill_size_optimizer_enabled is false"
    )]
    FixedPrefillChunckTokensRequiredWhenOptimizerDisabled,
    #[error("chunking.fixed_prefill_tokens must be positive")]
    InvalidFixedPrefillChunckTokens,
    #[error("chunking.optimizer_prefill_token_candidates must not be empty")]
    OptimizerPrefillChunckTokenCandidatesMustNotBeEmpty,
    #[error("chunking.optimizer_prefill_token_candidates must contain only positive values")]
    OptimizerPrefillChunckTokenCandidatesMustBePositive,
    #[error("chunking.optimizer_prefill_token_candidates must be strictly increasing")]
    OptimizerPrefillChunckTokenCandidatesMustBeStrictlyIncreasing,
    #[error("failed to parse supervisor bind address '{raw_bind_address}'")]
    ParseBindAddress {
        raw_bind_address: String,
        #[source]
        source: std::net::AddrParseError,
    },
    #[error("refusing to bind supervisor to non-loopback address {supervisor_bind_address}")]
    NonLoopbackBindAddress { supervisor_bind_address: SocketAddr },
    #[error("invalid prompt cache max_size_gb: {description}")]
    InvalidPromptCacheMaxSizeGb { description: &'static str },
    #[error("invalid maximum_mlx_memory_gb: {description}")]
    InvalidMaximumMlxMemoryGb { description: &'static str },
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
    #[error("HOME is required to derive the default prompt cache directory")]
    DefaultPromptCacheDirectoryRequiresHome,
    #[error("optimizer state directory requires HOME to derive the default path")]
    DefaultOptimizerDirectoryRequiresHome,
    #[error("logging requires HOME to derive the default log directory")]
    DefaultLogDirectoryRequiresHome,
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
