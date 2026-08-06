use std::path::PathBuf;

use astronomical_config::{
    ModelFamily, PrefillChunckSizingPolicy, PromptCacheConfig, classify_model_directory,
};
use astronomical_model_serving::{
    ModelFactory, ModelFamilyGenerationProcessor, ModelFamilyInferenceEngine,
    Qwen3_5PrefillChunckSizer, deepseek_v4_unavailable_reason,
};

use crate::qwen3_5_model_startup::initialize_qwen3_5_model;

/// Creates the concrete family processor and engine for a selected model directory.
pub(crate) struct ModelFamilyFactory {
    pub(crate) effective_mlx_memory_ceiling_bytes: usize,
    pub(crate) prompt_cache_config: PromptCacheConfig,
    pub(crate) prefill_chunck_sizing_policy: PrefillChunckSizingPolicy,
    pub(crate) optimizer_state_directory: Option<PathBuf>,
    pub(crate) performance_attribution_enabled: bool,
    pub(crate) performance_attribution_log_path: PathBuf,
    pub(crate) prefill_chunck_sizer_override: Option<Qwen3_5PrefillChunckSizer>,
    pub(crate) mtp_enabled: bool,
    pub(crate) persistent_prompt_cache_enabled: bool,
}

impl ModelFactory<ModelFamilyGenerationProcessor, ModelFamilyInferenceEngine>
    for ModelFamilyFactory
{
    async fn create(
        &self,
        model_directory: &str,
        max_output_tokens: u32,
    ) -> Result<(ModelFamilyGenerationProcessor, ModelFamilyInferenceEngine), String> {
        let model_directory_path = PathBuf::from(model_directory);
        let effective_mlx_memory_ceiling_bytes = self.effective_mlx_memory_ceiling_bytes;
        let prompt_cache_config = self.prompt_cache_config.clone();
        let prefill_chunck_sizing_policy = self.prefill_chunck_sizing_policy;
        let optimizer_state_directory = self.optimizer_state_directory.clone();
        let performance_attribution_enabled = self.performance_attribution_enabled;
        let performance_attribution_log_path = self.performance_attribution_log_path.clone();
        let prefill_chunck_sizer_override = self.prefill_chunck_sizer_override.clone();
        let mtp_enabled = self.mtp_enabled;
        let persistent_prompt_cache_enabled = self.persistent_prompt_cache_enabled;

        tokio::task::spawn_blocking(move || {
            let model_family = classify_model_directory(&model_directory_path)
                .map_err(|_| "selected model family could not be classified".to_owned())?;
            match model_family {
                Some(ModelFamily::Qwen3_5) => {
                    let (generation_processor, qwen3_5_engine) = initialize_qwen3_5_model(
                        model_directory_path,
                        effective_mlx_memory_ceiling_bytes,
                        prompt_cache_config,
                        prefill_chunck_sizing_policy,
                        prefill_chunck_sizer_override,
                        optimizer_state_directory,
                        max_output_tokens,
                        mtp_enabled,
                        persistent_prompt_cache_enabled,
                        performance_attribution_enabled,
                        performance_attribution_log_path,
                    )
                    .map_err(|startup_error| startup_error.public_model_load_failure_reason())?;
                    Ok((
                        ModelFamilyGenerationProcessor::Qwen3_5(generation_processor),
                        ModelFamilyInferenceEngine::Qwen3_5(qwen3_5_engine),
                    ))
                }
                Some(ModelFamily::DeepSeekV4) => Err(deepseek_v4_unavailable_reason().to_owned()),
                None => Err("selected model has an unsupported model family".to_owned()),
            }
        })
        .await
        .map_err(|join_error| join_error.to_string())?
    }

    fn update_mlx_memory_ceiling_bytes(&mut self, effective_mlx_memory_ceiling_bytes: u64) {
        self.effective_mlx_memory_ceiling_bytes =
            usize::try_from(effective_mlx_memory_ceiling_bytes).unwrap_or(usize::MAX);
    }
}
