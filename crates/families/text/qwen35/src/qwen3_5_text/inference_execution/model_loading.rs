use ipc_protocol::WorkerDecodeAstroExpertCachePolicy;
use runtime_integration::MlxRuntime;

use crate::inference_engine::load_result::EngineLoadResult;
use crate::{InferenceEngineError, PerformanceAttribution, PerformanceOperation};

use super::{Qwen3_5EngineState, fatal_runtime_error};
use crate::qwen3_5_core::artifacts::ValidatedQwen3_5Artifact;
use crate::qwen3_5_text::model::Qwen3_5Model;
use text_generation::generation_loop::model_loading_lifecycle::TextModelLoadingHooks;
use text_generation::model::model_chunking_configuration::TextModelChunkingConfiguration;

impl TextModelLoadingHooks for Qwen3_5EngineState {
    fn load_and_materialize_model(
        &mut self,
        runtime: MlxRuntime,
        validated_artifact: ValidatedQwen3_5Artifact,
        model_chunking: TextModelChunkingConfiguration,
        model_loading_performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Qwen3_5Model, InferenceEngineError> {
        load_and_materialize_streaming_model(
            runtime,
            validated_artifact,
            &self.core.model_directory,
            self.streaming_forward_shape(model_chunking),
            self.decode_astro_expert_cache_policy,
            model_loading_performance_attribution,
        )
    }

    /// Clears what loading left in the `mlx_buffer_cache` before serving.
    fn prepare_loaded_model(
        &mut self,
        model: &mut Qwen3_5Model,
        _model_loading_performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), InferenceEngineError> {
        let cleared_after_loading = model
            .runtime()
            .synchronize_gpu_stream_and_clear_mlx_buffer_cache()
            .map_err(fatal_runtime_error);
        // Last, so the programs' admission sees the model's own memory and
        // the layers the governor kept. They are only started here: the
        // model serves on the GPU while they build.
        self.neural_engine_prefill
            .begin(&*model, &self.core.chunking);
        cleared_after_loading
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

    fn layer_probe_shape(&self) -> crate::qwen3_5_text::model::LayerProbeShape {
        crate::qwen3_5_text::model::LayerProbeShape {
            prefill_chunk_token_count: self
                .core
                .prompt_processing_chunk_sizer
                .maximum_prompt_processing_chunk_size_tokens(),
            full_attention_kv_state_growth_tokens: self.core.full_attention_kv_state_growth_tokens,
        }
    }
}

/// How the engine will run forwards: the chunking the model loads with, and
/// the prefill shape a dense model's MLX working buffers are measured at.
struct StreamingForwardShape {
    model_chunking: TextModelChunkingConfiguration,
    layer_probe_shape: crate::qwen3_5_text::model::LayerProbeShape,
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
    let mtp_head_load = crate::qwen4_exp::mtp_head_weights::load_flash_next_mtp_head_weights(
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
    host_bundled_dense_mtp_head(&mut model, model_directory);
    Ok(model)
}

/// Hosts the multi-token-prediction head a dense pack ships in its `mtp/`
/// folder, when the model has no head of its own and hosts no DFlash
/// drafter: a hosted drafter checks a tree of guesses a round and is the one
/// that drafts wherever it loads. A head that cannot be used is logged, and
/// replies are written one token a forward, as without one.
fn host_bundled_dense_mtp_head(model: &mut Qwen3_5Model, model_directory: &std::path::Path) {
    use crate::qwen3_5_core::dense::mtp_head_weights::load_dense_mtp_head_weights;
    use crate::qwen3_5_core::model_math::mtp_head_kind::MtpHead;
    if model.mtp_head().is_some() || model.base.optional_accelerators.draft_runtime.is_some() {
        return;
    }
    match load_dense_mtp_head_weights(model.runtime(), &model.config, model_directory) {
        Ok(Some(head)) => {
            tracing::info!(
                head_weight_bytes = head.payload_byte_count,
                "the text engine hosts the pack's multi-token-prediction head"
            );
            model.base.optional_accelerators.mtp_head = Some(MtpHead::DenseQwen3_5(Box::new(head)));
            // The governor planned the kept weights before the head joined
            // them; a head that drafts is core weights, so it plans again.
            // One that does not draft stays lazy and takes no memory.
            if model.mtp_draft_token_count() > 0 {
                model.open_memory_governor();
            }
        }
        Ok(None) => {}
        Err(load_error) => tracing::warn!(
            %load_error,
            "the pack's multi-token-prediction head cannot be used; replies are written one token a forward"
        ),
    }
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
    use crate::qwen3_5_core::draft_tree_decode::draft_model_status_report::draft_runtime_and_status;
    use crate::qwen3_5_core::draft_tree_decode::engine_draft_runtime::Qwen3_5SpeculativeDraftRuntime;
    let memory_before_the_load = memory_around_a_load(model);
    let active_memory_limit_bytes = model.runtime().memory_limits().active_memory_limit_bytes();
    let (draft_runtime, draft_model_status) =
        draft_runtime_and_status(Qwen3_5SpeculativeDraftRuntime::load_for_target(
            model,
            active_memory_limit_bytes,
            model_directory,
            model_loading_performance_attribution,
        ));
    model.base.optional_accelerators.draft_model_status = Some(draft_model_status);
    if let Some(mut draft_runtime) = draft_runtime {
        draft_runtime.hosted_weight_bytes = super::hosted_drafter_bytes::bytes_a_load_added(
            memory_before_the_load,
            memory_around_a_load(model),
        );
        tracing::info!(
            hosted_drafter_weight_bytes = draft_runtime.hosted_weight_bytes,
            "the text engine hosts a DFlash drafter"
        );
        model.base.optional_accelerators.draft_runtime = Some(Box::new(draft_runtime));
    }
    // The drafter's measured weights joined the core in place of the room
    // set aside for its load, or nothing joined it: either way the governor
    // plans again without that room.
    model.reserve_room_for_a_loading_drafter(0);
}

/// MLX's active memory and the kept weights' share of it. An unreadable
/// snapshot reads as zero, and nothing is then counted across the load.
fn memory_around_a_load(model: &Qwen3_5Model) -> super::hosted_drafter_bytes::MemoryAroundALoad {
    let active_bytes = model
        .runtime()
        .memory_snapshot()
        .map_or(0, |memory_snapshot| {
            u64::try_from(memory_snapshot.active_memory_bytes()).unwrap_or(0)
        });
    super::hosted_drafter_bytes::MemoryAroundALoad {
        active_bytes,
        kept_weight_bytes: model.reclaimable_retained_payload_bytes(),
    }
}

/// Bounds the `mlx_buffer_cache` to what the loaded model measured and
/// hands it the multi-token-prediction head its checkpoint ships, if any.
fn finish_streaming_model_load(
    mut model: Qwen3_5Model,
    mtp_head_load: Result<
        Option<crate::qwen4_exp::mtp_head_weights::FlashNextMtpHeadWeights>,
        crate::TextExecutionError,
    >,
) -> Result<Qwen3_5Model, InferenceEngineError> {
    model
        .bound_mlx_buffer_cache_to_measured_working_buffers()
        .map_err(text_generation::engine_errors::fatal_runtime_error)?;
    model.base.optional_accelerators.mtp_head = mtp_head_load?
        .map(Box::new)
        .map(crate::qwen3_5_core::model_math::mtp_head_kind::MtpHead::FlashNext);
    if model.base.enable_fused_lane_mixing() {
        tracing::info!("decode path: fused lane-mixing kernels");
    }
    // The memory governor opened while the model loaded, before it had its
    // head. A head that drafts is 1.6 GB of core weights on Flash-Next;
    // left out, the `astro_expert_cache` was given that room too, and MLX
    // refused an ordinary expert read mid-reply (a multi-turn chat at a
    // 34 GB limit, four runs in eight). Opened again, the governor counts it.
    if model.mtp_head().is_some() {
        model.open_memory_governor();
    }
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
