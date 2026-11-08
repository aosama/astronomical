use std::path::PathBuf;

use astronomical_config::{
    AstronomicalConfig, AstronomicalConfigError, LogLevel, LoggingConfig, PromptCacheConfig,
    write_default_model, write_maximum_mlx_memory_gb,
};

use super::write_config;

mod chunking;
mod default_model;
mod idle_model_unload;
mod local_api_base_urls;
mod logging;
mod maximum_mlx_memory;
mod metal_residency_refresh;
mod migration;
mod model_config;
mod prompt_cache;
mod prompt_processing_chunk_sizing;
mod runtime;
mod runtime_instance;
