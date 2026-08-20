use std::path::PathBuf;

use astronomical_config::{ModelFamily, PromptCacheConfig, classify_model_directory};
use astronomical_ipc_protocol::{WorkerModelConfiguration, WorkerSpeculativePrefillConfiguration};
use astronomical_model_serving::{
    LagunaServingSettings, ModelFactory, ModelFamilyGenerationProcessor,
    ModelFamilyInferenceEngine, deepseek_v4_unavailable_reason,
    initialize_laguna_model_with_serving_settings,
};

use crate::qwen3_5_model_startup::initialize_qwen3_5_model;

/// Creates the concrete family processor and engine for a selected model directory.
pub(crate) struct ModelFamilyFactory {
    pub(crate) effective_mlx_memory_ceiling_bytes: usize,
    pub(crate) allocator_cache_memory_limit_bytes: usize,
    pub(crate) prompt_cache_config: PromptCacheConfig,
    pub(crate) performance_attribution_enabled: bool,
    pub(crate) performance_attribution_log_path: PathBuf,
    pub(crate) persistent_prompt_cache_enabled: bool,
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

impl ModelFactory<ModelFamilyGenerationProcessor, ModelFamilyInferenceEngine>
    for ModelFamilyFactory
{
    async fn create(
        &self,
        model_directory: &str,
        model_configuration: WorkerModelConfiguration,
    ) -> Result<(ModelFamilyGenerationProcessor, ModelFamilyInferenceEngine), String> {
        let model_directory_path = PathBuf::from(model_directory);
        let effective_mlx_memory_ceiling_bytes = self.effective_mlx_memory_ceiling_bytes;
        let allocator_cache_memory_limit_bytes = self.allocator_cache_memory_limit_bytes;
        let prompt_cache_config = self.prompt_cache_config.clone();
        let performance_attribution_enabled = self.performance_attribution_enabled;
        let performance_attribution_log_path = self.performance_attribution_log_path.clone();
        let persistent_prompt_cache_enabled = self.persistent_prompt_cache_enabled;
        let chunking = model_configuration.chunking.clone();

        tokio::task::spawn_blocking(move || {
            let model_family = classify_model_directory(&model_directory_path)
                .map_err(|_| "selected model family could not be classified".to_owned())?;
            match model_family {
                Some(ModelFamily::Qwen3_5) => {
                    let (generation_processor, qwen3_5_engine) = initialize_qwen3_5_model(
                        model_directory_path,
                        effective_mlx_memory_ceiling_bytes,
                        allocator_cache_memory_limit_bytes,
                        prompt_cache_config,
                        model_configuration.model_id,
                        model_configuration.maximum_context_tokens,
                        model_configuration.maximum_output_tokens,
                        model_configuration.mtp_draft_depth,
                        model_configuration
                            .speculative_prefill
                            .unwrap_or_else(disabled_speculative_prefill),
                        persistent_prompt_cache_enabled,
                        performance_attribution_enabled,
                        performance_attribution_log_path,
                        chunking,
                    )
                    .map_err(|startup_error| startup_error.public_model_load_failure_reason())?;
                    Ok((
                        ModelFamilyGenerationProcessor::Qwen3_5(generation_processor),
                        ModelFamilyInferenceEngine::Qwen3_5(qwen3_5_engine),
                    ))
                }
                Some(ModelFamily::Laguna) => {
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
                    Ok((
                        ModelFamilyGenerationProcessor::Laguna(generation_processor),
                        ModelFamilyInferenceEngine::Laguna(laguna_engine),
                    ))
                }
                Some(ModelFamily::DeepSeekV4) => Err(deepseek_v4_unavailable_reason().to_owned()),
                None => Err("selected model has an unsupported model family".to_owned()),
            }
        })
        .await
        .map_err(|_| "model-family initialization task failed".to_owned())?
    }

    fn update_mlx_memory_limits(
        &mut self,
        effective_mlx_memory_ceiling_bytes: u64,
        allocator_cache_memory_limit_bytes: u64,
    ) {
        self.effective_mlx_memory_ceiling_bytes =
            usize::try_from(effective_mlx_memory_ceiling_bytes).unwrap_or(usize::MAX);
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
