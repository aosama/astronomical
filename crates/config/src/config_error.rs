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
    #[error("failed to parse Astronomical config file at {config_file_path:?}: {source}")]
    ParseConfigFile {
        config_file_path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error(
        "Astronomical config file at {config_file_path:?} contains duplicate object key {duplicate_key:?}"
    )]
    DuplicateConfigKey {
        config_file_path: PathBuf,
        duplicate_key: String,
    },
    #[error(
        "Astronomical config file at {config_file_path:?} exceeds the {maximum_bytes}-byte limit"
    )]
    ConfigFileTooLarge {
        config_file_path: PathBuf,
        maximum_bytes: usize,
    },
    #[error("Astronomical config changed while a live setting update was being prepared")]
    ConfigChangedDuringUpdate,
    #[error("unsupported Astronomical configuration schema_version {schema_version}; expected 1")]
    UnsupportedSchemaVersion { schema_version: u32 },
    #[error("$schema must be './astronomical-config.schema.json'")]
    InvalidSchemaReference,
    #[error("{field_name} must be an absolute path, got {configured_path:?}")]
    PathMustBeAbsolute {
        field_name: String,
        configured_path: PathBuf,
    },
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
    #[error("MTP draft_depth must be between 1 and 3")]
    InvalidMtpDraftDepth,
    #[error("invalid models[{model_id:?}].{field_name}: {description}")]
    InvalidModelConfig {
        model_id: String,
        field_name: &'static str,
        description: &'static str,
    },
    #[error(
        "models[{model_id:?}].limits.maximum_context_tokens ({configured_maximum_context_tokens}) exceeds the discovered artifact maximum ({artifact_maximum_context_tokens})"
    )]
    ConfiguredContextExceedsArtifact {
        model_id: String,
        configured_maximum_context_tokens: u32,
        artifact_maximum_context_tokens: u32,
    },
    #[error(
        "models[{model_id:?}].generation_defaults.maximum_output_tokens ({configured_maximum_output_tokens}) must be smaller than the effective context ({effective_maximum_context_tokens})"
    )]
    ConfiguredOutputNotSmallerThanContext {
        model_id: String,
        configured_maximum_output_tokens: u32,
        effective_maximum_context_tokens: u32,
    },
    #[error("legacy Astronomical configuration cannot be migrated safely: {description}")]
    LegacyMigration { description: String },
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
    #[error("diagnostics.retained_log_files must be positive")]
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
