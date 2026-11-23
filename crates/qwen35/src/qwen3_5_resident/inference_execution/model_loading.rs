use runtime_integration::MlxRuntime;

use crate::inference_engine::load_result::EngineLoadResult;
use crate::{InferenceEngineError, PerformanceAttribution, PerformanceOperation};

use super::speculative_decode_lifecycle::{
    Qwen3_5SpeculativeDraftRuntime, draft_model_status_report,
};
use super::{Qwen3_5EngineState, qwen3_5_runtime_error};
use crate::qwen3_5_core::artifacts::ValidatedQwen3_5Artifact;
use crate::qwen3_5_core::model::model_chunking_configuration::Qwen3_5ModelChunkingConfiguration;
use crate::qwen3_5_core::shared_execution::model_loading_lifecycle::Qwen3_5ModelLoadingHooks;
use crate::qwen3_5_resident::model::Qwen3_5ResidentModel;
use crate::qwen3_5_resident::neural_engine_prefill_loading::{
    NeuralEnginePreparationStep, begin_neural_engine_prefill_preparation,
};
use crate::qwen3_5_resident::neural_engine_program_settlement::settle_neural_engine_programs;

impl Qwen3_5EngineState {
    /// The model's last parts, once its core weights are in place. Programs come
    /// last so their admission sees the model's own memory, and are only started
    /// here: the model serves on the GPU while they are built beside it.
    fn load_experts_and_begin_neural_engine_programs(
        &mut self,
        model: &mut Qwen3_5ResidentModel,
        model_loading_performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), InferenceEngineError> {
        synchronize_gpu_stream_then_load_experts(model, model_loading_performance_attribution)?;
        (
            self.neural_engine_prefill_status,
            self.neural_engine_prefill_preparation,
        ) = begin_neural_engine_prefill_preparation(
            model,
            &self.core.chunking,
            self.astro_neural_engine_program_cache_directory.as_deref(),
            None,
        );
        Ok(())
    }

    /// A request's start and every forward call this on the owner thread, which together
    /// cover the GPU work a waiting measurement must not read through.
    pub(super) fn tell_neural_engine_preparation_the_gpu_is_in_use(&mut self) {
        if let Some(preparation) = self.neural_engine_prefill_preparation.as_mut() {
            preparation.note_gpu_used();
        }
    }

    /// Moves the programs one step along, or settles what the last request learned about
    /// programs already running, and returns a status a reader has not yet seen. Only
    /// ever called while no request is active, which is when programs can be released.
    pub(super) fn advance_neural_engine_preparation(
        &mut self,
    ) -> Option<ipc_protocol::NeuralEnginePrefillStatus> {
        let model = self.core.model.as_mut()?;
        let Some(preparation) = self.neural_engine_prefill_preparation.as_mut() else {
            let settled_status =
                settle_neural_engine_programs(model, &self.neural_engine_prefill_status)?;
            self.neural_engine_prefill_status = settled_status.clone();
            return Some(settled_status);
        };
        let reported_status = match preparation.advance(model) {
            NeuralEnginePreparationStep::Unchanged => return None,
            NeuralEnginePreparationStep::Progressed(status) => status,
            NeuralEnginePreparationStep::ReadyToFinish => {
                let finished_preparation = self.neural_engine_prefill_preparation.take()?;
                finished_preparation.finish()
            }
        };
        self.neural_engine_prefill_status = reported_status.clone();
        Some(reported_status)
    }

    /// A DFlash drafter of either generation installed beside the target
    /// turns on draft trees; with none (or on a GPU without the kernels they
    /// need) every session decodes one token per forward. A drafter that
    /// fails to load costs the speedup, never the model. Status learns which
    /// of these happened.
    fn install_speculative_draft_runtime(
        &mut self,
        model: &mut Qwen3_5ResidentModel,
        model_loading_performance_attribution: &mut PerformanceAttribution,
    ) {
        let load_outcome = Qwen3_5SpeculativeDraftRuntime::load_for_target(
            model,
            self.core.memory_limits.active_memory_limit_bytes(),
            &self.core.model_directory,
            model_loading_performance_attribution,
        );
        (self.speculative_draft_runtime, self.draft_model_status) =
            draft_model_status_report::draft_runtime_and_status(load_outcome);
    }
}

impl Qwen3_5ModelLoadingHooks for Qwen3_5EngineState {
    /// The resident engine reads nothing from the artifact ahead of the model.
    fn observe_artifact_before_load(
        &mut self,
        _validated_artifact: &ValidatedQwen3_5Artifact,
        _model_loading_performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), InferenceEngineError> {
        Ok(())
    }

    fn load_and_materialize_model(
        &mut self,
        runtime: MlxRuntime,
        validated_artifact: ValidatedQwen3_5Artifact,
        model_chunking: Qwen3_5ModelChunkingConfiguration,
        model_loading_performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Qwen3_5ResidentModel, InferenceEngineError> {
        load_and_materialize_resident_model(
            runtime,
            validated_artifact,
            &self.core.model_directory,
            model_chunking,
            model_loading_performance_attribution,
        )
    }

    /// The experts, then the Neural Engine programs, then the drafter.
    fn prepare_loaded_model(
        &mut self,
        model: &mut Qwen3_5ResidentModel,
        model_loading_performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), InferenceEngineError> {
        self.load_experts_and_begin_neural_engine_programs(
            model,
            model_loading_performance_attribution,
        )?;
        self.install_speculative_draft_runtime(model, model_loading_performance_attribution);
        Ok(())
    }

    /// The memory governor opens on the model the engine will hold.
    fn adopt_loaded_model(&mut self, model: &Qwen3_5ResidentModel) {
        self.open_memory_governor(model);
    }

    fn engine_load_result(&self, minimum_mlx_memory_ceiling_bytes: u64) -> EngineLoadResult {
        Qwen3_5EngineState::engine_load_result(self, minimum_mlx_memory_ceiling_bytes)
    }
}

/// Synchronizes the graphics stream and clears the `mlx_buffer_cache` after the
/// weights land, then loads the resident expert weights when the model plans
/// expert layers.
fn synchronize_gpu_stream_then_load_experts(
    model: &mut Qwen3_5ResidentModel,
    model_loading_performance_attribution: &mut PerformanceAttribution,
) -> Result<(), InferenceEngineError> {
    model
        .runtime()
        .synchronize_gpu_stream_and_clear_mlx_buffer_cache()
        .map_err(qwen3_5_runtime_error)?;
    load_resident_expert_weights_when_planned(model, model_loading_performance_attribution)
}

/// Loads the resident model from the validated artifact and materializes
/// its target weights, both under attribution.
fn load_and_materialize_resident_model(
    runtime: MlxRuntime,
    validated_artifact: ValidatedQwen3_5Artifact,
    model_directory: &std::path::Path,
    model_chunking: Qwen3_5ModelChunkingConfiguration,
    model_loading_performance_attribution: &mut PerformanceAttribution,
) -> Result<Qwen3_5ResidentModel, InferenceEngineError> {
    let model = Qwen3_5ResidentModel::load_with_performance_attribution(
        runtime,
        validated_artifact,
        model_directory,
        true,
        model_chunking,
        model_loading_performance_attribution,
    )?;
    model_loading_performance_attribution
        .measure_operation(
            PerformanceOperation::ResidentWeightMaterializationSynchronizationWait,
            |_performance_attribution| model.materialize_target_weights(),
        )
        .map_err(qwen3_5_runtime_error)?;
    Ok(model)
}

/// Loads the resident expert weights when the model plans expert layers;
/// models without expert planning skip the storage read entirely.
fn load_resident_expert_weights_when_planned(
    model: &mut Qwen3_5ResidentModel,
    model_loading_performance_attribution: &mut PerformanceAttribution,
) -> Result<(), InferenceEngineError> {
    if model.expert_layer_plans.is_empty() {
        return Ok(());
    }
    let positional_file_read_metrics =
        model_loading_performance_attribution.positional_file_read_metrics();
    let resident_expert_weights = model_loading_performance_attribution
        .measure_operation(
            PerformanceOperation::ResidentWeightMaterializationSynchronizationWait,
            |_performance_attribution| {
                crate::qwen3_5_resident::Qwen3_5ResidentExpertWeights::load(
                    model.runtime(),
                    &model.expert_layer_plans,
                    positional_file_read_metrics,
                )
            },
        )
        .map_err(qwen3_5_runtime_error)?;
    model.resident_expert_weights = Some(resident_expert_weights);
    Ok(())
}
