//! The chat construction contract and the two chat families that implement
//! it.

use std::path::PathBuf;

use config::AstroSsdPromptCacheConfig;
use ipc_protocol::{WorkerAutoregressiveModelConfiguration, WorkerDecodeAstroExpertCachePolicy};

use crate::{
    EnginePlacement, PrismHadamardQwen35ServingSettings, Qwen3_5GenerationProcessor,
    initialize_prism_hadamard_qwen35_model_with_serving_settings,
};

use super::qwen3_5_model_startup::initialize_qwen3_5_model;
use super::{ModelFamilyInferenceEngine, boxed_chat_engine};

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

/// The folded-ternary Prism Hadamard chat family. The folded pack is dense,
/// so complete residency is not a sparse-expert concern and the placement
/// records that the whole model fits by construction.
pub struct PrismHadamardQwen35ChatFamily;

impl ChatFamilyConstruction for PrismHadamardQwen35ChatFamily {
    const INITIALIZATION_TASK_FAILURE: &'static str = "folded ternary initialization task failed";

    fn build_chat_runtime(
        chat_family_settings: ChatFamilySettings,
        model_directory_path: PathBuf,
        model_configuration: WorkerAutoregressiveModelConfiguration,
    ) -> Result<ChatRuntimeParts, String> {
        let astro_ssd_prompt_cache_enabled = chat_family_settings.astro_ssd_prompt_cache_enabled;
        let (generation_processor, prism_engine) =
            initialize_prism_hadamard_qwen35_model_with_serving_settings(
                &model_directory_path,
                chat_family_settings.effective_mlx_memory_ceiling_bytes,
                chat_family_settings.mlx_buffer_cache_limit_bytes,
                chat_family_settings.performance_attribution_enabled,
                PrismHadamardQwen35ServingSettings {
                    maximum_context_tokens: Some(model_configuration.maximum_context_tokens),
                    maximum_output_tokens: Some(model_configuration.maximum_output_tokens),
                    prompt_processing_chunk_tokens: model_configuration
                        .chunking
                        .fixed_prompt_processing_chunk_size_tokens,
                    // The folded pack ships the Qwen thinking template and
                    // the shared processor honors per-request thinking
                    // budgets.
                    enable_thinking: true,
                    chunking: model_configuration.chunking,
                    astro_ssd_prompt_cache_enabled,
                    astro_ssd_prompt_cache_config: astro_ssd_prompt_cache_enabled
                        .then_some(chat_family_settings.astro_ssd_prompt_cache_config),
                    performance_attribution_log_path: Some(
                        chat_family_settings.performance_attribution_log_path,
                    ),
                },
            )
            .map_err(|startup_error| startup_error.to_string())?;
        Ok(ChatRuntimeParts {
            generation_processor,
            inference_engine: boxed_chat_engine(prism_engine),
            engine_placement: EnginePlacement::CompleteResidencyFits,
        })
    }
}
