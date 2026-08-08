use std::path::PathBuf;

use astronomical_config::{
    AstronomicalConfig, AstronomicalConfigError, LogLevel, LoggingConfig,
    PrefillChunckSizingPolicy, PromptCacheConfig, SpeculativePrefillConfig, restore_config_file,
    write_maximum_mlx_memory_gb,
};

use super::write_config;

mod logging;
mod maximum_mlx_memory;
mod optimizer_directory;
mod prefill_chunck_sizing;
mod prompt_cache;
mod runtime;
mod speculative_prefill;
