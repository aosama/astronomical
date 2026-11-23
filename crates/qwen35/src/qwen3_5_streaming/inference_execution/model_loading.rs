use ipc_protocol::WorkerDecodeAstroExpertCachePolicy;
use runtime_integration::MlxRuntime;
use std::sync::Arc;

use crate::inference_engine::load_result::EngineLoadResult;
use crate::{
    AstroSsdPromptCacheDiskStore, AstroSsdPromptCacheModelContract, InferenceEngineError,
    PerformanceAttribution, PerformanceAttributionOutcome, PerformanceOperation,
    PersistentVisualEmbeddingModelContract, StreamingStartupResidencyFacts,
};

use super::{Qwen3_5EngineState, fatal_engine_error, qwen3_5_runtime_error};
use crate::qwen3_5_core::artifacts::ValidatedQwen3_5Artifact;
use crate::qwen3_5_core::model::model_chunking_configuration::Qwen3_5ModelChunkingConfiguration;
use crate::qwen3_5_core::shared_execution::model_loading_steps::{
    initialize_mlx_runtime_with_cleanup, open_astro_ssd_prompt_cache,
    resolve_model_chunking_configuration, resolve_validated_model_identity,
    resolve_visual_embedding_model_contract,
};
use crate::qwen3_5_streaming::model::Qwen3_5Model;

impl Qwen3_5EngineState {
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
            // Capture the startup engine-selection facts before the artifact
            // moves into the model: a later live raise consults them to
            // decide whether the raised ceiling would select the resident
            // engine in a replacement worker. Derived from the same artifact
            // and directory the selection itself reads, so the two answers
            // cannot drift.
            self.startup_residency_facts = capture_streaming_startup_residency_facts(
                self,
                &validated_artifact,
                &mut model_loading_performance_attribution,
            )?
            .or(self.startup_residency_facts);
            let runtime = initialize_mlx_runtime_with_cleanup(
                self.core.memory_limits,
                &mut model_loading_performance_attribution,
            )?;
            let model_chunking = resolve_model_chunking_configuration(&self.core.chunking)?;
            let model = load_and_materialize_streaming_model(
                runtime,
                validated_artifact,
                &self.core.model_directory,
                self.streaming_forward_shape(model_chunking),
                self.decode_astro_expert_cache_policy,
                &mut model_loading_performance_attribution,
            )?;
            self.startup_residency_facts =
                with_loaded_sequence_state_rate(self.startup_residency_facts, &model);
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
            model
                .runtime()
                .synchronize_gpu_stream_and_clear_mlx_buffer_cache()
                .map_err(qwen3_5_runtime_error)?;
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
            )) => self.install_loaded_streaming_model(
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
                record_failed_streaming_model_loading(
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

    /// Moves the successfully loaded model and its `astro_ssd_prompt_cache` ownership into the
    /// engine state, records the successful loading attribution, and reports
    /// the engine load result.
    #[allow(clippy::too_many_arguments)]
    fn install_loaded_streaming_model(
        &mut self,
        model: Qwen3_5Model,
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

impl Qwen3_5EngineState {
    /// The forward shape this engine serves, which a dense model's
    /// MLX working-buffer probe reproduces: the chunk sizer's largest prefill
    /// chunk and the attention slab granularity requests use.
    fn streaming_forward_shape(
        &self,
        model_chunking: Qwen3_5ModelChunkingConfiguration,
    ) -> StreamingForwardShape {
        StreamingForwardShape {
            model_chunking,
            layer_probe_shape: self.layer_probe_shape(),
        }
    }

    fn layer_probe_shape(&self) -> crate::qwen3_5_streaming::model::LayerProbeShape {
        crate::qwen3_5_streaming::model::LayerProbeShape {
            prefill_chunk_token_count: self
                .core
                .prompt_processing_chunk_sizer
                .maximum_prompt_processing_chunk_size_tokens(),
            full_attention_kv_state_growth_tokens: self.core.full_attention_kv_state_growth_tokens,
        }
    }
}

/// Captures the startup residency facts from the validated artifact under
/// attribution: a sparse model's expert geometry, or a dense model's text
/// weights, which a live raise compares the same way startup placement did.
/// A sparse artifact without expert bytes leaves the previous facts in place.
fn capture_streaming_startup_residency_facts(
    engine_state: &Qwen3_5EngineState,
    validated_artifact: &ValidatedQwen3_5Artifact,
    model_loading_performance_attribution: &mut PerformanceAttribution,
) -> Result<Option<StreamingStartupResidencyFacts>, InferenceEngineError> {
    let startup_geometry_result = model_loading_performance_attribution.measure_operation(
        PerformanceOperation::ArtifactMemoryGeometry,
        |_performance_attribution| {
            crate::model_memory_geometry_from_validated_artifact(
                validated_artifact,
                &engine_state.core.model_directory,
            )
        },
    );
    startup_residency_facts_from_geometry(startup_geometry_result, validated_artifact)
}

/// Builds startup facts from the artifact geometry: a sparse model's expert
/// geometry, or a dense model's text weights when the geometry answers that
/// the model has no experts. A sparse artifact without expert bytes records
/// none.
fn startup_residency_facts_from_geometry(
    startup_geometry_result: Result<
        (crate::ModelMemoryGeometry, u64),
        crate::Qwen3_5MemoryGeometryError,
    >,
    validated_artifact: &ValidatedQwen3_5Artifact,
) -> Result<Option<StreamingStartupResidencyFacts>, InferenceEngineError> {
    let (startup_geometry, startup_complete_residency_headroom_bytes) =
        match startup_geometry_result {
            Ok(startup_geometry) => startup_geometry,
            Err(crate::Qwen3_5MemoryGeometryError::NotSparseMixtureOfExperts) => {
                return dense_startup_residency_facts(validated_artifact).map(Some);
            }
            Err(geometry_error) => {
                return Err(fatal_engine_error(format!(
                    "startup residency selection facts are unreadable for this artifact: {geometry_error}"
                )));
            }
        };
    if startup_geometry.complete_expert_payload_bytes == 0 {
        return Ok(None);
    }
    Ok(Some(crate::StreamingStartupResidencyFacts {
        model_core_payload_bytes: startup_geometry.model_core_payload_bytes,
        complete_expert_payload_bytes: startup_geometry.complete_expert_payload_bytes,
        startup_complete_residency_headroom_bytes,
        sequence_state_bytes_per_token: startup_geometry.sequence_state_bytes_per_token,
    }))
}

/// A dense model's startup facts. Its decoder-state rate is read from the
/// loaded model afterwards (`with_loaded_sequence_state_rate`), because
/// the artifact alone does not know the `astro_decoder_state_cache` layout.
fn dense_startup_residency_facts(
    validated_artifact: &ValidatedQwen3_5Artifact,
) -> Result<StreamingStartupResidencyFacts, InferenceEngineError> {
    let dense_text_weight_payload_bytes = validated_artifact
        .text_weight_payload_bytes()
        .ok_or_else(|| fatal_engine_error("the dense model's text weight sizes are unreadable"))?;
    Ok(StreamingStartupResidencyFacts::for_dense_text_weights(
        dense_text_weight_payload_bytes,
        0,
    ))
}

/// Fills a dense model's decoder-state rate from the loaded model's
/// `astro_decoder_state_cache` layout; sparse facts already carry theirs from the artifact.
fn with_loaded_sequence_state_rate(
    startup_residency_facts: Option<StreamingStartupResidencyFacts>,
    model: &Qwen3_5Model,
) -> Option<StreamingStartupResidencyFacts> {
    let mut startup_residency_facts = startup_residency_facts?;
    if startup_residency_facts.complete_expert_payload_bytes == 0 {
        startup_residency_facts.sequence_state_bytes_per_token = model
            .base
            .memory_facts()
            .model_geometry()
            .sequence_state_bytes_per_token;
    }
    Some(startup_residency_facts)
}

/// How the engine will run forwards: the chunking the model loads with, and
/// the prefill shape a dense model's MLX working buffers are measured at.
struct StreamingForwardShape {
    model_chunking: Qwen3_5ModelChunkingConfiguration,
    layer_probe_shape: crate::qwen3_5_streaming::model::LayerProbeShape,
}

/// Loads the streaming model from the validated artifact under the engine's
/// decode astro expert cache policy and materializes its target weights.
fn load_and_materialize_streaming_model(
    runtime: MlxRuntime,
    validated_artifact: ValidatedQwen3_5Artifact,
    model_directory: &std::path::Path,
    streaming_forward_shape: StreamingForwardShape,
    decode_astro_expert_cache_policy: WorkerDecodeAstroExpertCachePolicy,
    model_loading_performance_attribution: &mut PerformanceAttribution,
) -> Result<Qwen3_5Model, InferenceEngineError> {
    let model =
        Qwen3_5Model::load_with_performance_attribution_and_decode_astro_expert_cache_policy(
            runtime,
            validated_artifact,
            model_directory,
            true,
            streaming_forward_shape.model_chunking,
            decode_astro_expert_cache_policy,
            model_loading_performance_attribution,
        )?;
    model_loading_performance_attribution
        .measure_operation(
            PerformanceOperation::ResidentWeightMaterializationSynchronizationWait,
            |_performance_attribution| {
                // A dense model then measures one streamed layer's working
                // memory at the prefill chunk size it will serve; its layers
                // are kept when the memory governor opens below.
                model.materialize_target_weights_and_probe_dense_layers(
                    streaming_forward_shape.layer_probe_shape,
                )
            },
        )
        .map_err(InferenceEngineError::from)?;
    // A sparse model's turn: one expert layer step, measured, so the memory
    // ledger knows a prompt layer's MLX working buffers before the first request.
    model.probe_expert_layer_mlx_working_buffers(streaming_forward_shape.layer_probe_shape);
    model.begin_serving_session_observation();
    model.open_memory_governor();
    Ok(model)
}

/// Records the failed loading attribution for the artifact metadata
/// captured before the failure, warning without masking the original error.
fn record_failed_streaming_model_loading(
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
