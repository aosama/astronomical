//! Model-family dispatch: build the concrete image or autoregressive engine for
//! a verified model directory, routing each supported family (FLUX.2-klein,
//! Qwen-Image-21, the Qwen3.5 text engine) through its own startup pipeline.
//!
//! The Qwen3.5 startup pipeline lives alongside this orchestrator in the
//! `model_family_factory/` folder: the startup routine (`qwen3_5_model_startup`),
//! the dense placement it consults (`qwen3_5_dense_engine_placement`), and its
//! failure type (`qwen3_5_model_startup_error`).

use std::path::PathBuf;

use config::AstroSsdPromptCacheConfig;
use ipc_protocol::{
    WorkerAutoregressiveModelConfiguration, WorkerDecodeAstroExpertCachePolicy,
    WorkerDiagnosticQwen3_5EngineOverride, WorkerEmbeddingModelConfiguration,
    WorkerImageGenerationModelConfiguration, WorkerModelConfiguration,
};
use model_serving::{
    DecisionEngine, EmbeddingEngine, EmbeddingFamilyConstruction, EngineBackedWorker,
    FamilyEngineSettings, ImageArtifactRefusal, ImageFamilyConstruction, ImageGenerationEngine,
    ModelFactory, ModelFactoryRuntime, ModelFamilyInferenceEngine,
    PrismHadamardQwen35ServingSettings, Qwen3_5GenerationProcessor, RecordedTrainingEngine,
    SpeechEngine, boxed_chat_engine, image_engine_from_evidence,
    initialize_prism_hadamard_qwen35_model_with_serving_settings, verified_image_artifact,
};

mod configured_family_check;
mod family_runtime_builders;
#[cfg(test)]
mod family_test_support;
mod qwen3_5_dense_engine_placement;
#[cfg(test)]
mod qwen3_5_dense_engine_placement_tests;
mod qwen3_5_model_startup;
mod qwen3_5_model_startup_error;

#[cfg(test)]
mod qwen3_5_model_startup_error_tests;

use self::configured_family_check::classified_model_family;
use self::family_runtime_builders::family_runtime_builder;
use self::qwen3_5_model_startup::Qwen3_5EngineSelection;

/// Creates the concrete family processor and engine for a selected model directory.
#[doc(hidden)]
pub struct ModelFamilyFactory {
    effective_mlx_memory_ceiling_bytes: usize,
    mlx_buffer_cache_limit_bytes: usize,
    astro_ssd_prompt_cache_config: AstroSsdPromptCacheConfig,
    /// `None` keeps programs out of the `astro_neural_engine_program_cache`, which costs a slower load and nothing else.
    neural_engine_program_directory: Option<PathBuf>,
    performance_attribution_enabled: bool,
    decode_astro_expert_cache_policy: WorkerDecodeAstroExpertCachePolicy,
    performance_attribution_log_path: PathBuf,
    astro_ssd_prompt_cache_enabled: bool,
    diagnostic_qwen3_5_engine_override: WorkerDiagnosticQwen3_5EngineOverride,
    /// The highest ceiling an image engine may plan its load at to reach its
    /// floor (#1902): the user's own limit, or unbounded when they set none.
    /// Startup sets it and live limit changes update it; zero until then,
    /// which keeps loads refusing below their floor.
    floor_raise_limit_bytes: u64,
}

/// The one unloaded runtime the factory hands the worker for any model
/// family: a chat engine, an image engine, the embedding engine, or a
/// decision engine of whichever family the model belongs to.
type ModelFamilyRuntime = ModelFactoryRuntime<
    Qwen3_5GenerationProcessor,
    ModelFamilyInferenceEngine,
    Box<dyn ImageGenerationEngine>,
    Box<dyn EmbeddingEngine>,
    Box<dyn DecisionEngine>,
    Box<dyn SpeechEngine>,
>;

pub(crate) type InferenceWorker = EngineBackedWorker<
    Qwen3_5GenerationProcessor,
    ModelFamilyInferenceEngine,
    ModelFamilyFactory,
    Box<dyn ImageGenerationEngine>,
    Box<dyn EmbeddingEngine>,
    Box<dyn DecisionEngine>,
    Box<dyn SpeechEngine>,
>;

impl ModelFamilyFactory {
    #[must_use]
    pub(crate) fn with_neural_engine_program_directory(
        mut self,
        neural_engine_program_directory: PathBuf,
    ) -> Self {
        self.neural_engine_program_directory = Some(neural_engine_program_directory);
        self
    }

    #[doc(hidden)]
    #[must_use]
    pub fn new(
        effective_mlx_memory_ceiling_bytes: usize,
        mlx_buffer_cache_limit_bytes: usize,
        astro_ssd_prompt_cache_config: AstroSsdPromptCacheConfig,
        performance_attribution_enabled: bool,
        decode_astro_expert_cache_policy: WorkerDecodeAstroExpertCachePolicy,
        performance_attribution_log_path: PathBuf,
        astro_ssd_prompt_cache_enabled: bool,
    ) -> Self {
        Self {
            effective_mlx_memory_ceiling_bytes,
            mlx_buffer_cache_limit_bytes,
            astro_ssd_prompt_cache_config,
            neural_engine_program_directory: None,
            performance_attribution_enabled,
            decode_astro_expert_cache_policy,
            performance_attribution_log_path,
            astro_ssd_prompt_cache_enabled,
            diagnostic_qwen3_5_engine_override: WorkerDiagnosticQwen3_5EngineOverride::Automatic,
            floor_raise_limit_bytes: 0,
        }
    }

    /// Records how far an image load may plan above its ceiling; see the field.
    #[must_use]
    pub(crate) fn with_floor_raise_limit_bytes(mut self, floor_raise_limit_bytes: u64) -> Self {
        self.floor_raise_limit_bytes = floor_raise_limit_bytes;
        self
    }

    /// Forces the Qwen3.5 engine family for controlled diagnostic journeys.
    #[doc(hidden)]
    #[must_use]
    pub fn with_diagnostic_qwen3_5_engine_override(
        mut self,
        diagnostic_qwen3_5_engine_override: WorkerDiagnosticQwen3_5EngineOverride,
    ) -> Self {
        self.diagnostic_qwen3_5_engine_override = diagnostic_qwen3_5_engine_override;
        self
    }

    async fn create_qwen3_5_runtime(
        &self,
        model_directory_path: PathBuf,
        model_configuration: WorkerAutoregressiveModelConfiguration,
        engine_selection: Qwen3_5EngineSelection,
    ) -> Result<
        (
            model_serving::Qwen3_5GenerationProcessor,
            ModelFamilyInferenceEngine,
            Option<model_serving::EnginePlacement>,
        ),
        String,
    > {
        let effective_mlx_memory_ceiling_bytes = self.effective_mlx_memory_ceiling_bytes;
        let mlx_buffer_cache_limit_bytes = self.mlx_buffer_cache_limit_bytes;
        let astro_ssd_prompt_cache_config = self.astro_ssd_prompt_cache_config.clone();
        let neural_engine_program_directory = self.neural_engine_program_directory.clone();
        let performance_attribution_enabled = self.performance_attribution_enabled;
        let decode_astro_expert_cache_policy = self.decode_astro_expert_cache_policy;
        let performance_attribution_log_path = self.performance_attribution_log_path.clone();
        let astro_ssd_prompt_cache_enabled = self.astro_ssd_prompt_cache_enabled;
        tokio::task::spawn_blocking(move || {
            let chunking = model_configuration.chunking.clone();
            qwen3_5_model_startup::initialize_qwen3_5_model(
                model_directory_path,
                effective_mlx_memory_ceiling_bytes,
                mlx_buffer_cache_limit_bytes,
                astro_ssd_prompt_cache_config,
                neural_engine_program_directory,
                model_configuration.model_id,
                model_configuration.maximum_context_tokens,
                model_configuration.maximum_output_tokens,
                astro_ssd_prompt_cache_enabled,
                performance_attribution_enabled,
                decode_astro_expert_cache_policy,
                performance_attribution_log_path,
                chunking,
                engine_selection,
            )
            .map_err(|startup_error| startup_error.public_model_load_failure_reason())
        })
        .await
        .map_err(|_| "Qwen3.5 initialization task failed".to_owned())?
    }
}

impl
    ModelFactory<
        Qwen3_5GenerationProcessor,
        ModelFamilyInferenceEngine,
        Box<dyn ImageGenerationEngine>,
        Box<dyn EmbeddingEngine>,
        Box<dyn DecisionEngine>,
        Box<dyn SpeechEngine>,
    > for ModelFamilyFactory
{
    async fn create(
        &self,
        model_directory: &str,
        model_configuration: WorkerModelConfiguration,
    ) -> Result<ModelFamilyRuntime, String> {
        let model_directory_path = PathBuf::from(model_directory);
        let model_family =
            classified_model_family(model_directory_path.clone(), &model_configuration).await?;
        let Some(runtime_builder) = model_family.and_then(family_runtime_builder) else {
            return Err("selected model has an unsupported model family".to_owned());
        };
        runtime_builder
            .build(self, model_directory_path, model_configuration)
            .await
    }

    fn supports_streaming_retry(&self) -> bool {
        true
    }

    async fn create_streaming_retry(
        &self,
        model_directory: &str,
        model_configuration: WorkerModelConfiguration,
    ) -> Result<ModelFamilyRuntime, String> {
        let WorkerModelConfiguration::Autoregressive(model_configuration) = model_configuration
        else {
            return Err("streaming retry requires an autoregressive model".to_owned());
        };
        let (generation_processor, inference_engine, _) = self
            .create_qwen3_5_runtime(
                PathBuf::from(model_directory),
                model_configuration,
                Qwen3_5EngineSelection::StreamingRetry,
            )
            .await?;
        Ok(ModelFactoryRuntime::autoregressive_for_streaming_retry(
            generation_processor,
            inference_engine,
        ))
    }

    fn supports_resident_promotion(&self) -> bool {
        true
    }

    async fn create_resident_promotion(
        &self,
        model_directory: &str,
        model_configuration: WorkerModelConfiguration,
    ) -> Result<ModelFamilyRuntime, String> {
        let WorkerModelConfiguration::Autoregressive(model_configuration) = model_configuration
        else {
            return Err("resident promotion requires an autoregressive model".to_owned());
        };
        let (generation_processor, inference_engine, _) = self
            .create_qwen3_5_runtime(
                PathBuf::from(model_directory),
                model_configuration,
                Qwen3_5EngineSelection::ResidentPromotion,
            )
            .await?;
        Ok(ModelFactoryRuntime::autoregressive_for_resident_promotion(
            generation_processor,
            inference_engine,
        ))
    }

    fn update_mlx_memory_limits(
        &mut self,
        effective_mlx_memory_ceiling_bytes: u64,
        mlx_buffer_cache_limit_bytes: u64,
    ) {
        self.effective_mlx_memory_ceiling_bytes =
            usize::try_from(effective_mlx_memory_ceiling_bytes).unwrap_or(usize::MAX);
        // MLX limits are process-global, so every replacement runtime must reuse the exact
        // policy acknowledged by the loaded engine rather than derive a fresh MLX buffer cache limit.
        self.mlx_buffer_cache_limit_bytes =
            usize::try_from(mlx_buffer_cache_limit_bytes).unwrap_or(usize::MAX);
    }

    fn update_floor_raise_limit(&mut self, floor_raise_limit_bytes: u64) {
        self.floor_raise_limit_bytes = floor_raise_limit_bytes;
    }

    fn global_astro_ssd_prompt_cache_root_directory(&self) -> Option<&std::path::Path> {
        Some(
            self.astro_ssd_prompt_cache_config
                .global_astro_ssd_prompt_cache_root_directory()
                .as_path(),
        )
    }

    fn performance_attribution_enabled(&self) -> bool {
        self.performance_attribution_enabled
    }
}

impl ModelFamilyFactory {
    /// Creates the Qwen3.5 autoregressive runtime through the automatic
    /// startup selection. Automatic selection always yields a startup
    /// placement; the worker's model swap rejects a runtime without one.
    async fn create_qwen3_5_autoregressive_runtime(
        &self,
        model_directory_path: PathBuf,
        model_configuration: WorkerAutoregressiveModelConfiguration,
    ) -> Result<ModelFamilyRuntime, String> {
        let (generation_processor, qwen3_5_engine, engine_placement) = self
            .create_qwen3_5_runtime(
                model_directory_path,
                model_configuration,
                Qwen3_5EngineSelection::Automatic(self.diagnostic_qwen3_5_engine_override),
            )
            .await?;
        Ok(ModelFactoryRuntime::Autoregressive {
            processor: generation_processor,
            engine: qwen3_5_engine,
            engine_placement,
        })
    }

    /// The settings every family's engine builder reads.
    fn family_engine_settings(&self) -> FamilyEngineSettings {
        FamilyEngineSettings {
            effective_mlx_memory_ceiling_bytes: self.effective_mlx_memory_ceiling_bytes,
            mlx_buffer_cache_limit_bytes: self.mlx_buffer_cache_limit_bytes,
            floor_raise_limit_bytes: self.floor_raise_limit_bytes,
            performance_attribution_enabled: self.performance_attribution_enabled,
            performance_attribution_log_path: self.performance_attribution_log_path.clone(),
        }
    }

    /// Checks an image family's folder against the selected configuration on
    /// a blocking task, then builds its unloaded engine.
    async fn create_image_engine<Family: ImageFamilyConstruction>(
        &self,
        model_directory_path: PathBuf,
        model_configuration: WorkerImageGenerationModelConfiguration,
    ) -> Result<ModelFamilyRuntime, String> {
        let verification_directory_path = model_directory_path.clone();
        let verification_task_result = tokio::task::spawn_blocking(move || {
            verified_image_artifact::<Family>(
                &verification_directory_path,
                &model_configuration.model_id,
                &model_configuration.artifact_revision,
            )
        })
        .await
        .map_err(|_| format!("{} verification task failed", Family::FAMILY_NAME))?;
        let verified_evidence = verification_task_result
            .map_err(|refusal| image_artifact_refusal_reason(Family::FAMILY_NAME, refusal))?;
        Ok(ModelFactoryRuntime::Image(Box::new(
            image_engine_from_evidence::<Family>(
                &self.family_engine_settings(),
                model_directory_path,
                verified_evidence,
            ),
        )))
    }

    /// Initializes the folded-ternary Prism-Hadamard runtime on a blocking
    /// task. The folded pack is dense, so complete residency is not a
    /// sparse-expert concern and the placement records that the resident
    /// engine fits by construction.
    async fn create_prism_hadamard_autoregressive_runtime(
        &self,
        model_directory_path: PathBuf,
        model_configuration: WorkerAutoregressiveModelConfiguration,
    ) -> Result<ModelFamilyRuntime, String> {
        let effective_mlx_memory_ceiling_bytes = self.effective_mlx_memory_ceiling_bytes;
        let mlx_buffer_cache_limit_bytes = self.mlx_buffer_cache_limit_bytes;
        let performance_attribution_enabled = self.performance_attribution_enabled;
        let performance_attribution_log_path = self.performance_attribution_log_path.clone();
        let astro_ssd_prompt_cache_enabled = self.astro_ssd_prompt_cache_enabled;
        let astro_ssd_prompt_cache_config = self.astro_ssd_prompt_cache_config.clone();
        let initialization_task_result = tokio::task::spawn_blocking(move || {
            let (generation_processor, prism_engine) =
                initialize_prism_hadamard_qwen35_model_with_serving_settings(
                    &model_directory_path,
                    effective_mlx_memory_ceiling_bytes,
                    mlx_buffer_cache_limit_bytes,
                    performance_attribution_enabled,
                    PrismHadamardQwen35ServingSettings {
                        maximum_context_tokens: Some(model_configuration.maximum_context_tokens),
                        maximum_output_tokens: Some(model_configuration.maximum_output_tokens),
                        prompt_processing_chunk_tokens: model_configuration
                            .chunking
                            .fixed_prompt_processing_chunk_size_tokens,
                        // The folded pack ships the Qwen thinking
                        // template and the shared processor honors
                        // per-request thinking budgets.
                        enable_thinking: true,
                        chunking: model_configuration.chunking.clone(),
                        astro_ssd_prompt_cache_enabled,
                        astro_ssd_prompt_cache_config: astro_ssd_prompt_cache_enabled
                            .then_some(astro_ssd_prompt_cache_config),
                        performance_attribution_log_path: Some(performance_attribution_log_path),
                    },
                )
                .map_err(|startup_error| startup_error.to_string())?;
            Ok::<_, String>((generation_processor, prism_engine))
        })
        .await
        .map_err(|_| "folded ternary initialization task failed".to_owned())?;
        let (generation_processor, prism_engine) = initialization_task_result?;
        Ok(ModelFactoryRuntime::autoregressive(
            generation_processor,
            boxed_chat_engine(prism_engine),
            model_serving::EnginePlacement::CompleteResidencyFits,
        ))
    }

    /// Wraps a decision family's unloaded engine for serving. The unloaded
    /// engine only reads the config and tokenizer bytes; MLX runtime
    /// initialization stays inside load() on the worker runtime thread.
    fn decision_runtime<Engine, InitializationFailure>(
        &self,
        model_directory_path: &std::path::Path,
        engine_outcome: Result<Engine, InitializationFailure>,
    ) -> Result<ModelFamilyRuntime, String>
    where
        Engine: DecisionEngine + 'static,
        InitializationFailure: std::fmt::Display,
    {
        recorded_decision_engine(model_directory_path, engine_outcome)
            .map(ModelFactoryRuntime::Decisions)
            // The failure text is the engine's own bounded, path-free reason.
            .map_err(|initialization_failure| {
                format!(
                    "selected decision artifact failed initialization: {initialization_failure}"
                )
            })
    }

    /// Builds an embedding family's unloaded engine. The engine only reads
    /// its configuration and tokenizer here; MLX work waits for `load()` on
    /// the worker runtime thread, because MLX handles are not `Send` across
    /// task boundaries.
    fn create_embedding_engine<Family: EmbeddingFamilyConstruction>(
        &self,
        model_directory_path: PathBuf,
        model_configuration: WorkerEmbeddingModelConfiguration,
    ) -> Result<ModelFamilyRuntime, String> {
        let embedding_engine = (Family::BUILD_ENGINE)(
            &self.family_engine_settings(),
            model_configuration.model_id,
            &model_directory_path,
        )
        .map_err(|failure_reason| {
            format!("selected embedding artifact failed initialization: {failure_reason:?}")
        })?;
        Ok(ModelFactoryRuntime::Embeddings(Box::new(embedding_engine)))
    }
}

/// The reason the worker gives for an image folder it will not serve. It
/// names the family and not which detail of the folder changed, because
/// discovery keeps the typed diagnostics and the selection may be stale.
fn image_artifact_refusal_reason(family_name: &str, refusal: ImageArtifactRefusal) -> String {
    match refusal {
        ImageArtifactRefusal::FailedVerification => {
            format!("selected {family_name} artifact failed exact-directory verification")
        }
        ImageArtifactRefusal::UnsupportedIdentity => {
            format!("selected {family_name} model identity or revision is unsupported")
        }
    }
}

/// A decision family's engine behind the trait object the worker serves
/// from. Every answer from a folder `astronomical decision` wrote carries
/// what judging that folder showed.
fn recorded_decision_engine<Engine, InitializationFailure>(
    model_directory_path: &std::path::Path,
    engine_outcome: Result<Engine, InitializationFailure>,
) -> Result<Box<dyn DecisionEngine>, String>
where
    Engine: DecisionEngine + 'static,
    InitializationFailure: std::fmt::Display,
{
    let family_engine: Box<dyn DecisionEngine> = Box::new(
        engine_outcome.map_err(|initialization_failure| initialization_failure.to_string())?,
    );
    RecordedTrainingEngine::wrapping(family_engine, model_directory_path)
        .map_err(|record_failure| record_failure.to_string())
}
