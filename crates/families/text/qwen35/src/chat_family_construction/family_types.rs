//! The chat construction contract and the two chat families that implement
//! it.

use std::path::PathBuf;

use config::AstroSsdPromptCacheConfig;
use ipc_protocol::{WorkerAutoregressiveModelConfiguration, WorkerDecodeAstroExpertCachePolicy};

use crate::{EnginePlacement, Qwen3_5GenerationProcessor};

use super::ModelFamilyInferenceEngine;
use super::qwen3_5_model_startup::initialize_qwen3_5_model;

/// The worker's settings every chat family's startup reads.
#[derive(Clone)]
pub struct ChatFamilySettings {
    pub effective_mlx_memory_ceiling_bytes: usize,
    pub mlx_buffer_cache_limit_bytes: usize,
    pub astro_ssd_prompt_cache_config: AstroSsdPromptCacheConfig,
    /// `None` keeps programs out of the `astro_neural_engine_program_cache`,
    /// which costs a slower load and nothing else.
    pub neural_engine_program_directory: Option<PathBuf>,
    pub performance_attribution_enabled: bool,
    pub decode_astro_expert_cache_policy: WorkerDecodeAstroExpertCachePolicy,
    pub performance_attribution_log_path: PathBuf,
    pub astro_ssd_prompt_cache_enabled: bool,
}

/// A chat family's unloaded runtime: its processor, its engine, and the
/// startup placement, which says whether the MLX memory limit holds the
/// whole model.
pub struct ChatRuntimeParts {
    pub generation_processor: Qwen3_5GenerationProcessor,
    pub inference_engine: ModelFamilyInferenceEngine,
    pub engine_placement: EnginePlacement,
}

/// What a chat family supplies so the worker can build it by kind.
pub trait ChatFamilyConstruction {
    /// The reason the worker reports when the blocking startup task itself
    /// fails.
    const INITIALIZATION_TASK_FAILURE: &'static str;

    /// Builds the family's runtime. It reads the model folder and starts the
    /// engine's owner thread, so the worker runs it on a blocking task. The
    /// failure reason names no local path; the worker bounds its length.
    fn build_chat_runtime(
        chat_family_settings: ChatFamilySettings,
        model_directory_path: PathBuf,
        model_configuration: WorkerAutoregressiveModelConfiguration,
    ) -> Result<ChatRuntimeParts, String>;
}

/// The Qwen 3.5 chat family, served by the one text engine, which
/// keeps in memory what the MLX memory limit holds and streams the rest.
pub struct Qwen3_5ChatFamily;

impl ChatFamilyConstruction for Qwen3_5ChatFamily {
    const INITIALIZATION_TASK_FAILURE: &'static str = "Qwen3.5 initialization task failed";

    fn build_chat_runtime(
        chat_family_settings: ChatFamilySettings,
        model_directory_path: PathBuf,
        model_configuration: WorkerAutoregressiveModelConfiguration,
    ) -> Result<ChatRuntimeParts, String> {
        initialize_qwen3_5_model(
            model_directory_path,
            chat_family_settings,
            model_configuration,
        )
        .map_err(|startup_error| startup_error.public_model_load_failure_reason())
    }
}
