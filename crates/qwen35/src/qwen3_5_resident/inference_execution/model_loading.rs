use runtime_integration::MlxRuntime;
use std::sync::Arc;

use crate::inference_engine::load_result::EngineLoadResult;
use crate::{
    AstroSsdPromptCacheDiskStore, AstroSsdPromptCacheModelContract, InferenceEngineError,
    PerformanceAttribution, PerformanceAttributionOutcome, PerformanceOperation,
    PersistentVisualEmbeddingModelContract,
};

use super::speculative_decode_lifecycle::{
    Qwen3_5SpeculativeDraftRuntime, draft_model_status_report,
};
use super::{Qwen3_5EngineState, fatal_engine_error, qwen3_5_runtime_error};
use crate::qwen3_5_core::artifacts::ValidatedQwen3_5Artifact;
use crate::qwen3_5_core::model::model_chunking_configuration::Qwen3_5ModelChunkingConfiguration;
use crate::qwen3_5_core::shared_execution::model_loading_steps::{
    initialize_mlx_runtime_with_cleanup, open_astro_ssd_prompt_cache,
    resolve_model_chunking_configuration, resolve_validated_model_identity,
    resolve_visual_embedding_model_contract,
};
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

    pub(super) fn load(&mut self) -> Result<EngineLoadResult, InferenceEngineError> {
        if let Some(model) = self.core.model.as_ref() {
            return Ok(self.engine_load_result(model.minimum_mlx_memory_ceiling_bytes()?));
        }
        let mut model_loading_performance_attribution =
            self.core.take_model_loading_performance_attribution();
        let mut model_id = None;
        let mut model_revision = None;
        let mut total_artifact_payload_bytes = None;
        let mut model_shard_count = None;
        let model_loading_result: Result<_, InferenceEngineError> = (|| {
            let validated_artifact = self.core.validated_artifact.take().ok_or_else(|| {
                fatal_engine_error("validated Qwen3.5 artifact is unavailable during MLX load")
            })?;
            let qwen3_5_vision_config = validated_artifact.vision_config().cloned();
            model_id = Some(validated_artifact.model_id().to_owned());
            model_revision = Some(validated_artifact.revision().to_owned());
            total_artifact_payload_bytes = Some(validated_artifact.total_payload_bytes());
            model_shard_count = Some(validated_artifact.shard_count());
            let runtime = initialize_mlx_runtime_with_cleanup(
                self.core.memory_limits,
                &mut model_loading_performance_attribution,
            )
            .map_err(qwen3_5_runtime_error)?;
            let model_chunking = resolve_model_chunking_configuration(&self.core.chunking)?;
            let mut model = load_and_materialize_resident_model(
                runtime,
                validated_artifact,
                &self.core.model_directory,
                model_chunking,
                &mut model_loading_performance_attribution,
            )
            .map_err(qwen3_5_runtime_error)?;
            let (resolved_model_id, resolved_model_revision) =
                resolve_validated_model_identity(&model_id, &model_revision)?;
            let persistent_visual_embedding_model_contract =
                resolve_visual_embedding_model_contract(
                    qwen3_5_vision_config.as_ref(),
                    &resolved_model_id,
                    &resolved_model_revision,
                );
            let (astro_ssd_prompt_cache_model_contract, astro_ssd_prompt_cache) =
                open_astro_ssd_prompt_cache(
                    self.core.astro_ssd_prompt_cache_disk_store_config.clone(),
                    &model,
                    &self.core.chunking,
                    &resolved_model_id,
                    &resolved_model_revision,
                    persistent_visual_embedding_model_contract.as_ref(),
                    &mut model_loading_performance_attribution,
                )?;
            self.load_experts_and_begin_neural_engine_programs(
                &mut model,
                &mut model_loading_performance_attribution,
            )?;
            self.install_speculative_draft_runtime(
                &mut model,
                &mut model_loading_performance_attribution,
            );
            Ok((
                model,
                astro_ssd_prompt_cache_model_contract,
                persistent_visual_embedding_model_contract,
                astro_ssd_prompt_cache,
            ))
        })();

        match model_loading_result {
            Ok((
                model,
                astro_ssd_prompt_cache_model_contract,
                persistent_visual_embedding_model_contract,
                astro_ssd_prompt_cache,
            )) => self.install_loaded_resident_model(
                model,
                astro_ssd_prompt_cache_model_contract,
                persistent_visual_embedding_model_contract,
                astro_ssd_prompt_cache,
                model_loading_performance_attribution,
                model_id,
                model_revision,
                total_artifact_payload_bytes,
                model_shard_count,
            ),
            Err(model_loading_error) => {
                record_failed_resident_model_loading(
                    self,
                    model_loading_performance_attribution,
                    model_id,
                    model_revision,
                    total_artifact_payload_bytes,
                    model_shard_count,
                    &model_loading_error,
                );
                Err(model_loading_error)
            }
        }
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

    /// Moves the successfully loaded model and its `astro_ssd_prompt_cache` ownership into the
    /// engine state, applies the diagnostic prefill activation floor when one
    /// is configured, records the successful loading attribution, and
    /// reports the engine load result.
    #[allow(clippy::too_many_arguments)]
    fn install_loaded_resident_model(
        &mut self,
        model: Qwen3_5ResidentModel,
        astro_ssd_prompt_cache_model_contract: Option<AstroSsdPromptCacheModelContract>,
        persistent_visual_embedding_model_contract: Option<PersistentVisualEmbeddingModelContract>,
        astro_ssd_prompt_cache: Option<Arc<AstroSsdPromptCacheDiskStore>>,
        model_loading_performance_attribution: PerformanceAttribution,
        model_id: Option<String>,
        model_revision: Option<String>,
        total_artifact_payload_bytes: Option<u64>,
        model_shard_count: Option<usize>,
    ) -> Result<EngineLoadResult, InferenceEngineError> {
        self.core.model_id = model_id.clone();
        self.core.model_revision = model_revision.clone();
        self.core.astro_ssd_prompt_cache_model_contract = astro_ssd_prompt_cache_model_contract;
        self.core.persistent_visual_embedding_model_contract =
            persistent_visual_embedding_model_contract;
        self.core.astro_ssd_prompt_cache = astro_ssd_prompt_cache;
        let mlx_memory_snapshot = model.runtime().memory_snapshot().ok();
        let resident_model_payload_bytes = Some(model.resident_model_payload_byte_count());
        let minimum_mlx_memory_ceiling_bytes = model.minimum_mlx_memory_ceiling_bytes()?;
        self.open_memory_governor(&model);
        self.core.model = Some(model);
        if let Err(performance_attribution_error) = self
            .record_model_loading_performance_attribution(
                model_loading_performance_attribution,
                PerformanceAttributionOutcome::Success,
                model_id,
                model_revision,
                total_artifact_payload_bytes,
                resident_model_payload_bytes,
                model_shard_count,
                mlx_memory_snapshot,
                None,
            )
        {
            tracing::warn!(
                error = %performance_attribution_error,
                "failed to persist model-loading performance attribution after successful load"
            );
        }
        Ok(self.engine_load_result(minimum_mlx_memory_ceiling_bytes))
    }
}

/// Synchronizes the graphics stream and clears the MLX buffer cache after the
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

/// Records the failed loading attribution for the artifact metadata
/// captured before the failure, warning without masking the original error.
fn record_failed_resident_model_loading(
    engine_state: &mut Qwen3_5EngineState,
    model_loading_performance_attribution: PerformanceAttribution,
    model_id: Option<String>,
    model_revision: Option<String>,
    total_artifact_payload_bytes: Option<u64>,
    model_shard_count: Option<usize>,
    model_loading_error: &InferenceEngineError,
) {
    if let Err(performance_attribution_error) = engine_state
        .record_model_loading_performance_attribution(
            model_loading_performance_attribution,
            PerformanceAttributionOutcome::Failed,
            model_id,
            model_revision,
            total_artifact_payload_bytes,
            None,
            model_shard_count,
            None,
            Some(model_loading_error.to_string()),
        )
    {
        tracing::warn!(
            error = %performance_attribution_error,
            "failed to persist model-loading performance attribution after failure"
        );
    }
}
