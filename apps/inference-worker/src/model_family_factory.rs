use std::path::PathBuf;

use astronomical_config::{
    ModelFamily, PromptCacheConfig, classify_model_directory, verify_flux2_klein_model_directory,
};
use astronomical_ipc_protocol::{
    WorkerImageGenerationModelFamily, WorkerModelConfiguration,
    WorkerSpeculativePrefillConfiguration,
};
use astronomical_model_serving::{
    EngineBackedWorker, Flux2KleinArtifactProvenance, Flux2KleinImageEngine, LagunaServingSettings,
    ModelFactory, ModelFactoryRuntime, ModelFamilyGenerationProcessor, ModelFamilyInferenceEngine,
    deepseek_v4_unavailable_reason, initialize_laguna_model_with_serving_settings,
};

use crate::qwen3_5_model_startup::initialize_qwen3_5_model;

/// Creates the concrete family processor and engine for a selected model directory.
#[doc(hidden)]
pub struct ModelFamilyFactory {
    effective_mlx_memory_ceiling_bytes: usize,
    allocator_cache_memory_limit_bytes: usize,
    prompt_cache_config: PromptCacheConfig,
    performance_attribution_enabled: bool,
    performance_attribution_log_path: PathBuf,
    persistent_prompt_cache_enabled: bool,
}

pub(crate) type InferenceWorker = EngineBackedWorker<
    ModelFamilyGenerationProcessor,
    ModelFamilyInferenceEngine,
    ModelFamilyFactory,
    Flux2KleinImageEngine,
>;

impl ModelFamilyFactory {
    #[doc(hidden)]
    #[must_use]
    pub fn new(
        effective_mlx_memory_ceiling_bytes: usize,
        allocator_cache_memory_limit_bytes: usize,
        prompt_cache_config: PromptCacheConfig,
        performance_attribution_enabled: bool,
        performance_attribution_log_path: PathBuf,
        persistent_prompt_cache_enabled: bool,
    ) -> Self {
        Self {
            effective_mlx_memory_ceiling_bytes,
            allocator_cache_memory_limit_bytes,
            prompt_cache_config,
            performance_attribution_enabled,
            performance_attribution_log_path,
            persistent_prompt_cache_enabled,
        }
    }
}

fn disabled_speculative_prefill() -> WorkerSpeculativePrefillConfiguration {
    WorkerSpeculativePrefillConfiguration {
        enabled: false,
        target_model_id: None,
        draft_model_id: None,
        draft_model_directory: None,
        minimum_prompt_tokens: 1,
        keep_percentage: 1,
        selection_chunck_token_count: 1,
        mandatory_trailing_token_count: 1,
        lookahead_token_count: 1,
        importance_pooling_kernel_token_count: 1,
    }
}

impl ModelFactory<ModelFamilyGenerationProcessor, ModelFamilyInferenceEngine, Flux2KleinImageEngine>
    for ModelFamilyFactory
{
    async fn create(
        &self,
        model_directory: &str,
        model_configuration: WorkerModelConfiguration,
    ) -> Result<
        ModelFactoryRuntime<
            ModelFamilyGenerationProcessor,
            ModelFamilyInferenceEngine,
            Flux2KleinImageEngine,
        >,
        String,
    > {
        let model_directory_path = PathBuf::from(model_directory);
        let effective_mlx_memory_ceiling_bytes = self.effective_mlx_memory_ceiling_bytes;
        let allocator_cache_memory_limit_bytes = self.allocator_cache_memory_limit_bytes;
        let prompt_cache_config = self.prompt_cache_config.clone();
        let performance_attribution_enabled = self.performance_attribution_enabled;
        let performance_attribution_log_path = self.performance_attribution_log_path.clone();
        let persistent_prompt_cache_enabled = self.persistent_prompt_cache_enabled;
        let classification_directory_path = model_directory_path.clone();
        let model_family = tokio::task::spawn_blocking(move || {
            classify_model_directory(&classification_directory_path)
                .map_err(|_| "selected model family could not be classified".to_owned())
        })
        .await
        .map_err(|_| "model-family classification task failed".to_owned())??;
        match (model_family, model_configuration) {
            (
                Some(ModelFamily::Qwen3_5),
                WorkerModelConfiguration::Autoregressive(model_configuration),
            ) => {
                let (generation_processor, qwen3_5_engine) =
                    tokio::task::spawn_blocking(move || {
                        let chunking = model_configuration.chunking.clone();
                        let (generation_processor, qwen3_5_engine) = initialize_qwen3_5_model(
                            model_directory_path,
                            effective_mlx_memory_ceiling_bytes,
                            allocator_cache_memory_limit_bytes,
                            prompt_cache_config,
                            model_configuration.model_id,
                            model_configuration.maximum_context_tokens,
                            model_configuration.maximum_output_tokens,
                            model_configuration.mtp_enabled,
                            model_configuration.mtp_draft_depth,
                            model_configuration
                                .speculative_prefill
                                .unwrap_or_else(disabled_speculative_prefill),
                            persistent_prompt_cache_enabled,
                            performance_attribution_enabled,
                            performance_attribution_log_path,
                            chunking,
                        )
                        .map_err(|startup_error| {
                            startup_error.public_model_load_failure_reason()
                        })?;
                        Ok::<_, String>((generation_processor, qwen3_5_engine))
                    })
                    .await
                    .map_err(|_| "Qwen3.5 initialization task failed".to_owned())??;
                Ok(ModelFactoryRuntime::autoregressive(
                    ModelFamilyGenerationProcessor::Qwen3_5(generation_processor),
                    ModelFamilyInferenceEngine::Qwen3_5(qwen3_5_engine),
                ))
            }
            (
                Some(ModelFamily::Laguna),
                WorkerModelConfiguration::Autoregressive(model_configuration),
            ) => {
                let (generation_processor, laguna_engine) =
                    tokio::task::spawn_blocking(move || {
                        let chunking = model_configuration.chunking.clone();
                        let (generation_processor, laguna_engine) =
                            initialize_laguna_model_with_serving_settings(
                                &model_directory_path,
                                effective_mlx_memory_ceiling_bytes,
                                allocator_cache_memory_limit_bytes,
                                performance_attribution_enabled,
                                LagunaServingSettings {
                                    maximum_context_tokens: Some(
                                        model_configuration.maximum_context_tokens,
                                    ),
                                    maximum_output_tokens: Some(
                                        model_configuration.maximum_output_tokens,
                                    ),
                                    chunking: Some(chunking),
                                    persistent_prompt_cache_enabled,
                                    prompt_cache_config: persistent_prompt_cache_enabled
                                        .then_some(prompt_cache_config),
                                    performance_attribution_log_path: Some(
                                        performance_attribution_log_path,
                                    ),
                                },
                            )
                            .map_err(|startup_error| {
                                startup_error.public_model_load_failure_reason()
                            })?;
                        Ok::<_, String>((generation_processor, laguna_engine))
                    })
                    .await
                    .map_err(|_| "Laguna initialization task failed".to_owned())??;
                Ok(ModelFactoryRuntime::autoregressive(
                    ModelFamilyGenerationProcessor::Laguna(generation_processor),
                    ModelFamilyInferenceEngine::Laguna(laguna_engine),
                ))
            }
            (
                Some(ModelFamily::Flux2Klein),
                WorkerModelConfiguration::Flux2Klein(model_configuration),
            ) => {
                let verification_directory_path = model_directory_path.clone();
                let verified_evidence = tokio::task::spawn_blocking(move || {
                    verify_flux2_klein_model_directory(&verification_directory_path)
                        // Discovery retains typed diagnostics, while this worker boundary must not
                        // reveal which mutable local artifact detail changed after selection.
                        .map_err(|_| {
                            "selected FLUX.2 Klein artifact failed exact-directory verification"
                                .to_owned()
                        })
                })
                .await
                .map_err(|_| "FLUX.2 Klein verification task failed".to_owned())??;
                if model_configuration.model_family != WorkerImageGenerationModelFamily::Flux2Klein
                    || model_configuration.model_id != verified_evidence.canonical_model_id
                    || model_configuration.artifact_revision != verified_evidence.revision
                {
                    return Err(
                        "selected FLUX.2 Klein model identity or revision is unsupported"
                            .to_owned(),
                    );
                }
                let provenance = Flux2KleinArtifactProvenance::new(
                    verified_evidence.provider_model_id,
                    verified_evidence.revision,
                    verified_evidence.license.spdx_identifier(),
                );
                Ok(ModelFactoryRuntime::Image(
                    Flux2KleinImageEngine::from_model_family_factory(
                        model_directory_path,
                        provenance,
                        effective_mlx_memory_ceiling_bytes,
                        allocator_cache_memory_limit_bytes,
                        performance_attribution_enabled,
                        performance_attribution_log_path,
                    ),
                ))
            }
            (Some(ModelFamily::DeepSeekV4), WorkerModelConfiguration::Autoregressive(_)) => {
                Err(deepseek_v4_unavailable_reason().to_owned())
            }
            (Some(_), _) => Err(
                "selected model configuration does not match its classified model family"
                    .to_owned(),
            ),
            (None, _) => Err("selected model has an unsupported model family".to_owned()),
        }
    }

    fn update_mlx_memory_limits(
        &mut self,
        effective_mlx_memory_ceiling_bytes: u64,
        allocator_cache_memory_limit_bytes: u64,
    ) {
        self.effective_mlx_memory_ceiling_bytes =
            usize::try_from(effective_mlx_memory_ceiling_bytes).unwrap_or(usize::MAX);
        // MLX limits are process-global, so every replacement runtime must reuse the exact
        // policy acknowledged by the loaded engine rather than derive a fresh cache limit.
        self.allocator_cache_memory_limit_bytes =
            usize::try_from(allocator_cache_memory_limit_bytes).unwrap_or(usize::MAX);
    }

    fn global_prompt_cache_root_directory(&self) -> Option<&std::path::Path> {
        Some(
            self.prompt_cache_config
                .global_prompt_cache_root_directory()
                .as_path(),
        )
    }

    fn performance_attribution_enabled(&self) -> bool {
        self.performance_attribution_enabled
    }
}
