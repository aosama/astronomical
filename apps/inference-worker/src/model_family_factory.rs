//! Model-family dispatch: build the unloaded runtime of whichever family a
//! verified model directory classifies as, through that family's builder in
//! the family list. Each family's own startup lives with the family; the
//! factory holds only the settings every builder reads.

use std::path::PathBuf;

use config::AstroSsdPromptCacheConfig;
use ipc_protocol::{
    WorkerAutoregressiveModelConfiguration, WorkerDecodeAstroExpertCachePolicy,
    WorkerEmbeddingModelConfiguration, WorkerImageGenerationModelConfiguration,
    WorkerModelConfiguration,
};
use model_serving::{ChatFamilyConstruction, ChatFamilySettings, ChatRuntimeParts};
use model_serving::{
    DecisionEngine, EmbeddingEngine, EmbeddingFamilyConstruction, EngineBackedWorker,
    FamilyEngineSettings, ImageArtifactRefusal, ImageFamilyConstruction, ImageGenerationEngine,
    ModelFactory, ModelFactoryRuntime, ModelFamilyInferenceEngine, Qwen3_5GenerationProcessor,
    RecordedTrainingEngine, SpeechEngine, image_engine_from_evidence, verified_image_artifact,
};

use crate::worker_startup::error::bound_public_model_load_failure_reason;

mod configured_family_check;
mod family_runtime_builders;
#[cfg(test)]
mod family_test_support;

use self::configured_family_check::classified_model_family;
use self::family_runtime_builders::family_runtime_builder;

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
            floor_raise_limit_bytes: 0,
        }
    }

    /// Records how far an image load may plan above its ceiling; see the field.
    #[must_use]
    pub(crate) fn with_floor_raise_limit_bytes(mut self, floor_raise_limit_bytes: u64) -> Self {
        self.floor_raise_limit_bytes = floor_raise_limit_bytes;
        self
    }

    /// The settings every chat family's startup reads.
    fn chat_family_settings(&self) -> ChatFamilySettings {
        ChatFamilySettings {
            effective_mlx_memory_ceiling_bytes: self.effective_mlx_memory_ceiling_bytes,
            mlx_buffer_cache_limit_bytes: self.mlx_buffer_cache_limit_bytes,
            astro_ssd_prompt_cache_config: self.astro_ssd_prompt_cache_config.clone(),
            neural_engine_program_directory: self.neural_engine_program_directory.clone(),
            performance_attribution_enabled: self.performance_attribution_enabled,
            decode_astro_expert_cache_policy: self.decode_astro_expert_cache_policy,
            performance_attribution_log_path: self.performance_attribution_log_path.clone(),
            astro_ssd_prompt_cache_enabled: self.astro_ssd_prompt_cache_enabled,
        }
    }

    /// Builds a chat family's runtime on a blocking task, since its startup
    /// reads the model folder and starts the engine's owner thread. Every
    /// chat family's failure reason is bounded here, at the worker boundary.
    async fn create_chat_runtime_parts<Family: ChatFamilyConstruction + 'static>(
        &self,
        model_directory_path: PathBuf,
        model_configuration: WorkerAutoregressiveModelConfiguration,
    ) -> Result<ChatRuntimeParts, String> {
        let chat_family_settings = self.chat_family_settings();
        tokio::task::spawn_blocking(move || {
            Family::build_chat_runtime(
                chat_family_settings,
                model_directory_path,
                model_configuration,
            )
        })
        .await
        .map_err(|_| Family::INITIALIZATION_TASK_FAILURE.to_owned())?
        .map_err(bound_public_model_load_failure_reason)
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
