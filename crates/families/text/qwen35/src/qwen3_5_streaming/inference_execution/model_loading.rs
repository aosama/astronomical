use ipc_protocol::WorkerDecodeAstroExpertCachePolicy;
use runtime_integration::MlxRuntime;

use crate::inference_engine::load_result::EngineLoadResult;
use crate::{
    InferenceEngineError, PerformanceAttribution, PerformanceOperation,
    StreamingStartupResidencyFacts,
};

use super::{Qwen3_5EngineState, fatal_engine_error, fatal_runtime_error};
use crate::qwen3_5_core::artifacts::ValidatedQwen3_5Artifact;
use crate::qwen3_5_streaming::model::Qwen3_5Model;
use text_generation::generation_loop::model_loading_lifecycle::TextModelLoadingHooks;
use text_generation::model::model_chunking_configuration::TextModelChunkingConfiguration;

impl TextModelLoadingHooks for Qwen3_5EngineState {
    /// Captures the startup engine-selection facts before the artifact moves
    /// into the model: a later live raise consults them to decide whether the
    /// raised ceiling would select the resident engine in a replacement
    /// worker. They come from the same artifact and directory the selection
    /// itself reads, so the two answers cannot drift.
    fn observe_artifact_before_load(
        &mut self,
        validated_artifact: &ValidatedQwen3_5Artifact,
        model_loading_performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), InferenceEngineError> {
        self.startup_residency_facts = capture_streaming_startup_residency_facts(
            self,
            validated_artifact,
            model_loading_performance_attribution,
        )?
        .or(self.startup_residency_facts);
        Ok(())
    }

    fn load_and_materialize_model(
        &mut self,
        runtime: MlxRuntime,
        validated_artifact: ValidatedQwen3_5Artifact,
        model_chunking: TextModelChunkingConfiguration,
        model_loading_performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Qwen3_5Model, InferenceEngineError> {
        let model = load_and_materialize_streaming_model(
            runtime,
            validated_artifact,
            &self.core.model_directory,
            self.streaming_forward_shape(model_chunking),
            self.decode_astro_expert_cache_policy,
            model_loading_performance_attribution,
        )?;
        self.startup_residency_facts =
            with_loaded_sequence_state_rate(self.startup_residency_facts, &model);
        Ok(model)
    }

    /// Clears what loading left in the `mlx_buffer_cache` before serving.
    fn prepare_loaded_model(
        &mut self,
        model: &mut Qwen3_5Model,
        _model_loading_performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), InferenceEngineError> {
        model
            .runtime()
            .synchronize_gpu_stream_and_clear_mlx_buffer_cache()
            .map_err(fatal_runtime_error)
    }

    /// The memory governor already opened as the model loaded.
    fn adopt_loaded_model(&mut self, _model: &Qwen3_5Model) {}

    fn engine_load_result(&self, minimum_mlx_memory_ceiling_bytes: u64) -> EngineLoadResult {
        Qwen3_5EngineState::engine_load_result(self, minimum_mlx_memory_ceiling_bytes)
    }
}

impl Qwen3_5EngineState {
    /// The forward shape this engine serves, which a dense model's
    /// MLX working-buffer probe reproduces: the chunk sizer's largest prefill
    /// chunk and the attention slab granularity requests use.
    fn streaming_forward_shape(
        &self,
        model_chunking: TextModelChunkingConfiguration,
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
    model_chunking: TextModelChunkingConfiguration,
    layer_probe_shape: crate::qwen3_5_streaming::model::LayerProbeShape,
}

/// Loads, materializes and measures the streaming model, then bounds the
/// `mlx_buffer_cache` to the working buffers it measured.
fn load_and_materialize_streaming_model(
    runtime: MlxRuntime,
    mut validated_artifact: ValidatedQwen3_5Artifact,
    model_directory: &std::path::Path,
    streaming_forward_shape: StreamingForwardShape,
    decode_astro_expert_cache_policy: WorkerDecodeAstroExpertCachePolicy,
    model_loading_performance_attribution: &mut PerformanceAttribution,
) -> Result<Qwen3_5Model, InferenceEngineError> {
    // The head has a shard of its own, taken before the load consumes the
    // artifact; a failure to load it surfaces once the model itself loaded.
    let mtp_head_load = crate::qwen4_exp::mtp_head_weights::load_mtp_head_weights(
        &runtime,
        &mut validated_artifact,
    );
    let model = load_materialize_and_probe_streaming_model(
        runtime,
        validated_artifact,
        model_directory,
        streaming_forward_shape,
        decode_astro_expert_cache_policy,
        model_loading_performance_attribution,
    )?;
    let mut model = finish_streaming_model_load(model, mtp_head_load)?;
    host_paired_drafter(
        &mut model,
        model_directory,
        model_loading_performance_attribution,
    );
    Ok(model)
}

/// Hosts the DFlash drafter that pairs with `model`, when one is installed
/// and this GPU runs the tree kernels, and counts what it holds with the
/// model's core weights so the `astro_expert_cache` is planned around it.
/// A drafter that cannot load is logged and replies are written one token
/// a forward, as without one.
fn host_paired_drafter(
    model: &mut Qwen3_5Model,
    model_directory: &std::path::Path,
    model_loading_performance_attribution: &mut PerformanceAttribution,
) {
    use crate::qwen3_5_core::draft_tree_decode::engine_draft_runtime::{
        Qwen3_5DraftRuntimeLoadOutcome, Qwen3_5SpeculativeDraftRuntime,
    };
    let active_memory_bytes_before = active_memory_bytes(model);
    let active_memory_limit_bytes = model.runtime().memory_limits().active_memory_limit_bytes();
    let load_outcome = Qwen3_5SpeculativeDraftRuntime::load_for_target(
        model,
        active_memory_limit_bytes,
        model_directory,
        model_loading_performance_attribution,
    );
    let mut draft_runtime = match load_outcome {
        Ok(Qwen3_5DraftRuntimeLoadOutcome::Loaded(draft_runtime)) => draft_runtime,
        Ok(_) => return,
        Err(draft_loading_error) => {
            tracing::warn!(
                %draft_loading_error,
                "the installed DFlash drafter could not be loaded; replies are written one token a forward"
            );
            return;
        }
    };
    draft_runtime.hosted_weight_bytes =
        active_memory_bytes(model).saturating_sub(active_memory_bytes_before);
    tracing::info!(
        hosted_drafter_weight_bytes = draft_runtime.hosted_weight_bytes,
        "the streaming engine hosts a DFlash drafter"
    );
    model.base.optional_accelerators.draft_runtime = Some(draft_runtime);
    // The drafter's weights joined the core: the governor plans again.
    model.open_memory_governor();
}

/// MLX's active memory, or zero when it cannot be read: the difference
/// across the drafter's load is then zero and nothing is counted.
fn active_memory_bytes(model: &Qwen3_5Model) -> u64 {
    model
        .runtime()
        .memory_snapshot()
        .map_or(0, |memory_snapshot| {
            u64::try_from(memory_snapshot.active_memory_bytes()).unwrap_or(0)
        })
}

/// Bounds the `mlx_buffer_cache` to what the loaded model measured and
/// hands it the multi-token-prediction head its checkpoint ships, if any.
fn finish_streaming_model_load(
    mut model: Qwen3_5Model,
    mtp_head_load: Result<
        Option<crate::qwen4_exp::mtp_head_weights::MtpHeadWeights>,
        crate::TextExecutionError,
    >,
) -> Result<Qwen3_5Model, InferenceEngineError> {
    model
        .bound_mlx_buffer_cache_to_measured_working_buffers()
        .map_err(text_generation::engine_errors::fatal_runtime_error)?;
    model.base.optional_accelerators.mtp_head = mtp_head_load?.map(Box::new);
    // Reply tokens of a model that streams experts look their routes up
    // inside the graph, which needs MLX's fast synch to pay. A dense model
    // has no routes to look up and leaves the setting alone: it is
    // process-wide, and turning it off here would slow a streaming model
    // already loaded beside this one.
    if model.expert_pager.is_some() {
        model.set_in_graph_table_serving(true);
    }
    Ok(model)
}

/// Loads the streaming model from the validated artifact under the engine's
/// decode astro expert cache policy and materializes its target weights.
fn load_materialize_and_probe_streaming_model(
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
