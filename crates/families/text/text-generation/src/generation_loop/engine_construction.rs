//! Starting a text engine: the constructors every engine offers,
//! written once for any engine that builds itself from a
//! [`TextExecutionCore`] plus its own settings.

use std::path::PathBuf;

use ipc_protocol::WorkerChunkingConfiguration;

use crate::{
    AstroSsdPromptCacheDiskStoreConfig, InferenceEngineError, MlxInferenceEngine,
    MlxInferenceExecution, PerformanceAttribution, PerformanceAttributionLog,
};

use super::execution_core::{TextExecutionCore, TextExecutionCoreOwner, TextExecutionCoreSettings};
use super::prompt_processing_chunk_sizer::PromptProcessingChunkSizer;
use super::serving_artifact::TextServingArtifact;

/// An engine built from the shared core plus settings of its own.
pub trait TextEngineConstruction: MlxInferenceExecution + TextExecutionCoreOwner {
    /// What only this engine is started with. `Default` is what the shorter
    /// constructors use: no Neural Engine program folder for resident, the
    /// baseline `astro_expert_cache` decode policy for streaming.
    type EngineSettings: Default + Send + 'static;

    /// Builds the engine on the MLX owner thread, with no model loaded yet.
    fn new_on_owner_thread(
        core: TextExecutionCore<Self::Model, Self::RequestExtension, Self::Artifact>,
        engine_settings: Self::EngineSettings,
    ) -> Self;
}

/// The constructors every text engine offers. They are a trait because
/// the owner-thread engine lives in the engine core, and only the crate that
/// defines a type may add inherent methods to it.
pub trait TextMlxEngineConstructors: Sized {
    /// What only this engine is started with.
    type EngineSettings;
    /// The validated artifact the engine loads its model from.
    type Artifact;

    /// Starts the owner thread from the shared settings and the engine's own.
    fn new_with_execution_core_settings(
        core_settings: TextExecutionCoreSettings<Self::Artifact>,
        engine_settings: Self::EngineSettings,
    ) -> Result<Self, InferenceEngineError>;

    /// Starts the owner thread with an explicit `prefill_chunk_tokens` sizer,
    /// the artifact's whole context, and attribution off.
    #[allow(clippy::too_many_arguments)]
    fn new_with_prompt_processing_chunk_sizer(
        validated_artifact: Self::Artifact,
        active_memory_limit_bytes: usize,
        mlx_buffer_cache_limit_bytes: usize,
        astro_ssd_prompt_cache_disk_store_config: Option<AstroSsdPromptCacheDiskStoreConfig>,
        prompt_processing_chunk_sizer: PromptProcessingChunkSizer,
        think_end_token_id: u32,
        model_directory: PathBuf,
        chunking: WorkerChunkingConfiguration,
    ) -> Result<Self, InferenceEngineError>;

    /// Starts the owner thread with every model-serving work boundary
    /// resolved and the artifact's whole context.
    #[allow(clippy::too_many_arguments)]
    fn new_with_runtime_chunking_and_performance_attribution(
        validated_artifact: Self::Artifact,
        active_memory_limit_bytes: usize,
        mlx_buffer_cache_limit_bytes: usize,
        astro_ssd_prompt_cache_disk_store_config: Option<AstroSsdPromptCacheDiskStoreConfig>,
        prompt_processing_chunk_sizer: PromptProcessingChunkSizer,
        think_end_token_id: u32,
        model_directory: PathBuf,
        chunking: WorkerChunkingConfiguration,
        model_loading_performance_attribution: PerformanceAttribution,
        performance_attribution_log: PerformanceAttributionLog,
    ) -> Result<Self, InferenceEngineError>;
}

impl<Execution> TextMlxEngineConstructors for MlxInferenceEngine<Execution>
where
    Execution: TextEngineConstruction<Artifact: TextServingArtifact + Send + 'static>,
{
    type EngineSettings = Execution::EngineSettings;
    type Artifact = Execution::Artifact;

    /// Starts the owner thread from the shared settings and the engine's own.
    /// The settings are checked first, so a bad one fails here and not at the
    /// first request.
    fn new_with_execution_core_settings(
        core_settings: TextExecutionCoreSettings<Self::Artifact>,
        engine_settings: Execution::EngineSettings,
    ) -> Result<Self, InferenceEngineError> {
        let validated_core_settings = core_settings.validate()?;
        MlxInferenceEngine::new(move || {
            Execution::new_on_owner_thread(
                TextExecutionCore::new(validated_core_settings),
                engine_settings,
            )
        })
    }

    /// Starts the owner thread with an explicit `prefill_chunk_tokens` sizer,
    /// the artifact's whole context, and attribution off.
    // Construction dependencies remain explicit to avoid another configuration facade.
    #[allow(clippy::too_many_arguments)]
    fn new_with_prompt_processing_chunk_sizer(
        validated_artifact: Self::Artifact,
        active_memory_limit_bytes: usize,
        mlx_buffer_cache_limit_bytes: usize,
        astro_ssd_prompt_cache_disk_store_config: Option<AstroSsdPromptCacheDiskStoreConfig>,
        prompt_processing_chunk_sizer: PromptProcessingChunkSizer,
        think_end_token_id: u32,
        model_directory: PathBuf,
        chunking: WorkerChunkingConfiguration,
    ) -> Result<Self, InferenceEngineError> {
        Self::new_with_runtime_chunking_and_performance_attribution(
            validated_artifact,
            active_memory_limit_bytes,
            mlx_buffer_cache_limit_bytes,
            astro_ssd_prompt_cache_disk_store_config,
            prompt_processing_chunk_sizer,
            think_end_token_id,
            model_directory,
            chunking,
            PerformanceAttribution::disabled(),
            PerformanceAttributionLog::disabled(),
        )
    }

    /// Starts the owner thread with every model-serving work boundary
    /// resolved and the artifact's whole context.
    #[allow(clippy::too_many_arguments)]
    fn new_with_runtime_chunking_and_performance_attribution(
        validated_artifact: Self::Artifact,
        active_memory_limit_bytes: usize,
        mlx_buffer_cache_limit_bytes: usize,
        astro_ssd_prompt_cache_disk_store_config: Option<AstroSsdPromptCacheDiskStoreConfig>,
        prompt_processing_chunk_sizer: PromptProcessingChunkSizer,
        think_end_token_id: u32,
        model_directory: PathBuf,
        chunking: WorkerChunkingConfiguration,
        model_loading_performance_attribution: PerformanceAttribution,
        performance_attribution_log: PerformanceAttributionLog,
    ) -> Result<Self, InferenceEngineError> {
        let maximum_context_tokens = validated_artifact.maximum_position_count();
        Self::new_with_execution_core_settings(
            TextExecutionCoreSettings {
                validated_artifact,
                active_memory_limit_bytes,
                mlx_buffer_cache_limit_bytes,
                astro_ssd_prompt_cache_disk_store_config,
                prompt_processing_chunk_sizer,
                think_end_token_id,
                model_directory,
                maximum_context_tokens,
                chunking,
                model_loading_performance_attribution,
                performance_attribution_log,
            },
            Execution::EngineSettings::default(),
        )
    }
}
