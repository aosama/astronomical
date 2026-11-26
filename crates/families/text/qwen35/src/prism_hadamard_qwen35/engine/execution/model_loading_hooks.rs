//! The folded-ternary engine's answers to the shared Qwen 3.5 model load
//! (issue #2473, Prism step 10).
//!
//! The shared load starts the MLX runtime, opens the
//! `astro_ssd_prompt_cache`, installs the model and records the loading
//! attribution. What stays here is how a folded pack becomes a model, the
//! quota-bounded contract its store is opened with, and what the worker
//! learns from the load. The minimum ceiling comes from the model's own
//! dense-resident floor (`TextModel::minimum_mlx_memory_ceiling_bytes`).

use ipc_protocol::ExpertMemoryMode;
use runtime_integration::MlxRuntime;

use super::PrismHadamardQwen35InferenceExecution;
use super::astro_ssd_prompt_cache_hooks::prism_quota_bounded_contract;
use super::engine_trait_impl::gpu_error;
use crate::inference_engine::load_result::EngineLoadResult;
use crate::prism_hadamard_qwen35::artifacts::ValidatedFoldedPack;
use crate::prism_hadamard_qwen35::model::PrismHadamardTextModel;
use crate::{InferenceEngineError, PerformanceAttribution, PerformanceOperation};
use text_generation::generation_loop::model_loading_lifecycle::TextModelLoadingHooks;
use text_generation::generation_loop::model_loading_steps::AstroSsdPromptCacheContractResolver;
use text_generation::model::model_chunking_configuration::TextModelChunkingConfiguration;

impl TextModelLoadingHooks for PrismHadamardQwen35InferenceExecution {
    /// A pack with a long position range keeps a store instead of failing to
    /// load: the contract is fitted to the disk quota.
    const ASTRO_SSD_PROMPT_CACHE_CONTRACT_RESOLVER: AstroSsdPromptCacheContractResolver =
        prism_quota_bounded_contract;

    /// Maps the pack's one weights file and builds the dense model from it,
    /// measured as the safetensors mapping the folded-ternary load has
    /// always reported. The model reads its graph sizes from the pack, so
    /// the chunking settings have nothing to change here.
    fn load_and_materialize_model(
        &mut self,
        runtime: MlxRuntime,
        validated_pack: ValidatedFoldedPack,
        _model_chunking: TextModelChunkingConfiguration,
        model_loading_performance_attribution: &mut PerformanceAttribution,
    ) -> Result<PrismHadamardTextModel, InferenceEngineError> {
        model_loading_performance_attribution
            .measure_operation(
                PerformanceOperation::ModelSafetensorsMapping,
                |_performance_attribution| PrismHadamardTextModel::new(runtime, validated_pack),
            )
            .map_err(gpu_error)
    }

    /// The dense model holds every weight once built; there is no expert,
    /// Neural Engine program or drafter to prepare.
    fn prepare_loaded_model(
        &mut self,
        _model: &mut PrismHadamardTextModel,
        _model_loading_performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), InferenceEngineError> {
        Ok(())
    }

    fn adopt_loaded_model(&mut self, _model: &PrismHadamardTextModel) {}

    /// The whole pack is resident, and a Neural Engine prefill the settings
    /// asked for is reported as one this engine cannot serve.
    fn engine_load_result(&self, minimum_mlx_memory_ceiling_bytes: u64) -> EngineLoadResult {
        EngineLoadResult::new()
            .with_expert_memory_mode(Some(ExpertMemoryMode::Resident))
            .with_minimum_mlx_memory_ceiling_bytes(minimum_mlx_memory_ceiling_bytes)
            .with_neural_engine_prefill_status(
                EngineLoadResult::neural_engine_prefill_unsupported_when_enabled(
                    self.neural_engine_mlp_prefill_requested,
                ),
            )
    }
}

#[cfg(test)]
#[path = "model_loading_hooks_tests.rs"]
mod model_loading_hooks_tests;
