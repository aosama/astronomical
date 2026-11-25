//! The SSD-streaming Qwen3.5 engine's loaded model: the shared base plus every
//! residency-forked field (expert pager, retained experts, residency plans,
//! route observation) and the paged execution entry points.

use crate::qwen3_5_core::model::prefill_forward::Qwen3_5PrefillForward;
use crate::qwen3_5_core::model::target_forward_graph::Qwen3_5TargetForwardGraph;
use mlx_bridge::MlxArray;

use std::cell::{Cell, RefCell};

use crate::qwen3_5_core::model::Qwen3_5ModelBase;
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5DecoderLayerWeights;
use crate::qwen3_5_core::model_math::forward_contract;
use crate::qwen3_5_streaming::model::route_observation::RouteObservationCollector;
use crate::qwen3_5_streaming::{
    AstroExpertCache, PagedForwardMissingRouteCollector, Qwen3_5ExpertPager,
    Qwen3_5MoEPagedExecutionMode,
};
use crate::{AstroDecoderStateCacheState, ExpertResidencyPlan, PerformanceAttribution};
use engine_core::expert_paging::AstroExpertCacheStatistics;
use text_generation::RequestDecoderStateStack;
use text_generation::execution_error::TextExecutionError;

use text_generation::AstroSsdPromptCacheBoundaryCheckpointCollector;

/// The streaming engine's model: shared base plus residency-forked state.
#[derive(Debug)]
pub struct Qwen3_5StreamingModel {
    pub base: Qwen3_5ModelBase,
    /// Sparse models own a pager; dense models have no sparse-expert weights.
    pub expert_pager: Option<Qwen3_5ExpertPager>,
    /// A dense model's decoder layers, kept or read from the SSD per forward
    /// (issue #1985); `None` for a sparse model.
    pub dense_layer_streamer:
        Option<crate::qwen3_5_streaming::model::dense_layer_streamer::Qwen3_5DenseLayerStreamer>,
    /// Individual routed experts retained within the paged-mode RAM ceiling.
    pub astro_expert_cache: Option<RefCell<AstroExpertCache>>,
    /// Reads a reply token's missing experts from storage straight into the
    /// tables of the `astro_expert_cache`, and counts how often it could.
    pub direct_expert_fetching:
        crate::qwen3_5_streaming::model::direct_expert_fetch::DirectExpertFetching,
    /// On only while a measurement prices a route stop: every layer then
    /// copies its route to the host even when its table holds the layer whole.
    pub brings_every_route_to_the_host: std::cell::Cell<bool>,
    /// Where each expert layer should end up for the reply being written;
    /// `None` while a prompt is read, when every layer may be kept whole.
    pub active_expert_residency_plan: RefCell<Option<ExpertResidencyPlan>>,
    /// Deferred GPU missing-route roots collected during one paged forward.
    pub paged_forward_missing_route_collector: PagedForwardMissingRouteCollector,
    /// What recent reply tokens say about serving the next one optimistically.
    pub(super) optimistic_serving_history:
        super::optimistic_table_serving::OptimisticServingHistory,
    /// Warm-table slot capacity for decode-time hot-expert warming; zero
    /// disables warming. The decode handoff derives it from the adaptive
    /// growth guard's headroom so warming only claims memory the guard can
    /// spare (issue #372).
    pub hot_expert_warm_slot_count: Cell<usize>,
    /// Expert payload bytes streamed from storage for the active request,
    /// written only by `stream_expert_page_from_storage` and read by
    /// `observed_expert_memory_mode` to report the residency actually
    /// experienced (issue #1533): a Paged/Hybrid ownership claim that never
    /// streams reads is observed as Resident.
    pub(super) request_expert_storage_read_tally:
        super::request_expert_storage_read_tally::RequestExpertStorageReadTally,
    /// One layer's prompt MLX working buffers, measured at load, for the memory
    /// governor: an expert layer's for a sparse model, a streamed decoder
    /// layer's for a dense one. `None` when the probe could not run.
    pub probed_prompt_layer_mlx_working_buffers: Cell<
        Option<
            crate::qwen3_5_streaming::model::expert_layer_mlx_working_buffer_probe::ProbedPromptLayerMlxWorkingBuffers,
        >,
    >,
    /// The memory governor and what this engine holds from it (issue
    /// #2221).
    pub governed_memory: crate::qwen3_5_streaming::model::memory_governance::GovernedMemory,
    /// Decode route capture for the expert-predictor program: lazy pending
    /// route arrays plus the bounded observation history (issue #536).
    pub route_observation: RefCell<RouteObservationCollector>,
    /// Routes of seated complete layers multiplied on-device, recorded as
    /// expert demand after the forward's completion evaluation.
    pub deferred_expert_demand:
        RefCell<crate::qwen3_5_streaming::model::deferred_expert_demand::DeferredExpertDemandCollector>,
    /// The complete expert pages a prompt forward has in hand.
    pub prompt_expert_pages_in_hand:
        crate::qwen3_5_streaming::model::held_prompt_expert_layer::PromptExpertPagesInHand,
    /// Layers whose prefill completion read the ceiling already refused.
    /// Remembering the refusal stops every later chunk from re-reading the
    /// unseated remainder and then the experts already in RAM.
    pub prefill_layer_completion_refused: RefCell<
        crate::qwen3_5_streaming::model::prefill_partial_layer_read::PrefillLayerCompletionRefusals,
    >,
    /// The fused jangtq2 decode-token projection kernel; `None` for models
    /// without codebook experts or when the capability probe or compilation failed.
    /// Without it decode tokens decode a transient bfloat16 page like prefill.
    pub fused_jangtq2_gathered_projection_kernel: Option<
        jangtq2_qwen35::fused_gathered_projection::FusedJangtq2GatheredProjectionKernel,
    >,
}

crate::qwen3_5_core::model::serving_model::serve_text_from_wrapped_model_base!(
    Qwen3_5StreamingModel
);

impl std::ops::Deref for Qwen3_5StreamingModel {
    type Target = Qwen3_5ModelBase;

    fn deref(&self) -> &Qwen3_5ModelBase {
        &self.base
    }
}

impl text_generation::model::memory_breakdown::TextModelMemoryOwnership for Qwen3_5StreamingModel {
    fn mlx_active_memory_ceiling_bytes(&self) -> u64 {
        self.base.mlx_active_memory_ceiling_bytes()
    }

    fn held_expert_payload_bytes(&self) -> u64 {
        self.astro_expert_cache_statistics()
            .resident_payload_byte_count
    }

    /// Excludes dense layers read from the SSD on each forward, which the
    /// base's bound-weight count would otherwise charge to the model core.
    fn held_model_core_payload_bytes(&self) -> u64 {
        // Named explicitly: method syntax would fall through to the base's
        // count, which includes the streamed layers, if this were renamed.
        Qwen3_5StreamingModel::resident_model_payload_byte_count(self)
    }

    fn governed_memory_reserve_snapshot(&self) -> Option<crate::memory::MemoryReserveSnapshot> {
        self.memory_reserve_snapshot_from_governor(self.held_expert_payload_bytes())
    }
}

// Named calls: method syntax inside these bodies would resolve to the trait
// itself if an inherent method were ever renamed, recursing forever.
impl text_generation::model::text_model::TextModel for Qwen3_5StreamingModel {
    fn observed_expert_memory_mode(&self) -> ipc_protocol::ExpertMemoryMode {
        Qwen3_5StreamingModel::observed_expert_memory_mode(self)
    }

    fn expert_memory_mode(&self) -> ipc_protocol::ExpertMemoryMode {
        Qwen3_5StreamingModel::expert_memory_mode(self)
    }

    fn memory_snapshot_for_telemetry(
        &self,
    ) -> Result<runtime_integration::MlxMemorySnapshot, crate::InferenceEngineError> {
        self.memory_snapshot_with_forward_peak()
            .map_err(text_generation::engine_errors::fatal_runtime_error)
    }

    fn resident_model_payload_byte_count(&self) -> u64 {
        Qwen3_5StreamingModel::resident_model_payload_byte_count(self)
    }

    crate::qwen3_5_core::model::shared_model_methods::qwen3_5_shared_text_model_forwarding!(
        Qwen3_5StreamingModel
    );
}

impl std::ops::DerefMut for Qwen3_5StreamingModel {
    fn deref_mut(&mut self) -> &mut Qwen3_5ModelBase {
        &mut self.base
    }
}

impl Qwen3_5StreamingModel {
    /// Returns the retained `astro_expert_cache`'s expert-memory snapshot.
    #[must_use]
    pub fn astro_expert_cache_statistics(&self) -> AstroExpertCacheStatistics {
        if self.expert_pager.is_none() {
            return AstroExpertCacheStatistics::default();
        }
        self.astro_expert_cache
            .as_ref()
            .map_or_else(AstroExpertCacheStatistics::default, |astro_expert_cache| {
                astro_expert_cache.borrow().statistics()
            })
    }

    /// Sets the warm-table slot capacity for decode-time hot-expert warming.
    pub fn set_hot_expert_warm_slot_count(&self, warm_slot_count: usize) {
        self.hot_expert_warm_slot_count.set(warm_slot_count);
    }

    /// Returns the warm-table slot capacity (zero disables warming).
    pub fn hot_expert_warm_slot_count(&self) -> usize {
        self.hot_expert_warm_slot_count.get()
    }

    /// Removes one retained expert layer, returning whether it existed.
    pub fn remove_retained_expert_layer(&self, layer_index: usize) -> bool {
        self.astro_expert_cache
            .as_ref()
            .is_some_and(|astro_expert_cache| {
                astro_expert_cache.borrow_mut().remove_layer(layer_index)
            })
    }

    /// Executes one prompt chunk, injecting visual embeddings at image_pad positions.
    ///
    /// `chunk_token_ids` are the token IDs for this prefill chunk.
    /// `visual_embeddings` carries the full visual embedding tensor for all images in the request.
    /// `starting_visual_embedding_index` tracks how many visual embeddings earlier chunks consumed.
    /// Returns the count of visual embeddings consumed by this chunk.
    pub fn prefill_chunk_with_visual_embeddings(
        &self,
        chunk_token_ids: &[u32],
        starting_position_tokens: u32,
        visual_embeddings: &MlxArray,
        starting_visual_embedding_index: usize,
        request_decoder_state: &mut RequestDecoderStateStack,
        image_pad_token_id: u32,
    ) -> Result<usize, TextExecutionError> {
        let mut disabled_performance_attribution = PerformanceAttribution::disabled();
        self.prefill_chunk_with_visual_embeddings_and_performance_attribution(
            chunk_token_ids,
            starting_position_tokens,
            visual_embeddings,
            starting_visual_embedding_index,
            request_decoder_state,
            image_pad_token_id,
            &mut disabled_performance_attribution,
        )
    }

    /// Executes one intermediate prompt chunk and materializes only reusable decoder state.
    pub fn prefill_chunk(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
    ) -> Result<(), TextExecutionError> {
        let mut disabled_performance_attribution = PerformanceAttribution::disabled();
        self.prefill_chunk_with_performance_attribution(
            token_ids,
            starting_position_tokens,
            request_decoder_state,
            &mut disabled_performance_attribution,
        )
    }

    /// Executes one attributed prompt chunk for direct-MLX performance probes.
    ///
    /// The caller supplies its own attribution so a probe can isolate one
    /// chunk's expert-streaming operations and positional reads from every
    /// other chunk of the same request.
    #[doc(hidden)]
    pub fn prefill_chunk_with_attribution_for_tests(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), TextExecutionError> {
        self.prefill_chunk_with_performance_attribution(
            token_ids,
            starting_position_tokens,
            request_decoder_state,
            performance_attribution,
        )
    }

    /// Executes one prompt or decode chunk and materializes final logits plus all layer state.
    pub fn forward_chunk(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
    ) -> Result<MlxArray, TextExecutionError> {
        let mut disabled_performance_attribution = PerformanceAttribution::disabled();
        self.forward_chunk_with_performance_attribution(
            token_ids,
            starting_position_tokens,
            request_decoder_state,
            &mut disabled_performance_attribution,
        )
    }

    /// Executes one prompt chunk with a test-only paged MoE execution selector.
    #[doc(hidden)]
    pub fn forward_chunk_with_paged_execution_mode_for_tests(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        paged_execution_mode: Qwen3_5MoEPagedExecutionMode,
    ) -> Result<MlxArray, TextExecutionError> {
        let mut disabled_performance_attribution = PerformanceAttribution::disabled();
        let final_logits = self
            .build_forward_chunk_with_paged_execution_mode_and_performance_attribution(
                token_ids,
                starting_position_tokens,
                request_decoder_state,
                paged_execution_mode,
                &mut disabled_performance_attribution,
            )?;
        self.evaluate_forward_state(&final_logits, request_decoder_state)?;
        Ok(final_logits)
    }

    pub fn build_forward_chunk_with_paged_execution_mode_and_performance_attribution(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        paged_execution_mode: Qwen3_5MoEPagedExecutionMode,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, TextExecutionError> {
        let token_count = forward_contract::validate_forward_input(
            token_ids,
            starting_position_tokens,
            None,
            request_decoder_state.layer_count(),
            self.config.layer_count() as usize,
            self.config.vocabulary_size(),
            self.config.maximum_position_count(),
        )?;
        let signed_token_ids = token_ids
            .iter()
            .map(|token_id| {
                i32::try_from(*token_id).map_err(|_| TextExecutionError::InvalidInput {
                    description: "token ID exceeds the MLX int32 range",
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let token_indices = self
            .runtime
            .array_from_i32(&signed_token_ids, &[1, token_count])?;
        self.build_forward_graph(
            &token_indices,
            token_count,
            starting_position_tokens,
            request_decoder_state,
            paged_execution_mode,
            performance_attribution,
        )
    }

    // Decoder-layer inputs stay explicit instead of introducing another per-layer facade.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_decoder_layer(
        &self,
        hidden_states: &MlxArray,
        token_count: i32,
        rope_offset_tokens: i32,
        layer_index: usize,
        decoder_layer_weights: &Qwen3_5DecoderLayerWeights,
        layer_model_state: &mut AstroDecoderStateCacheState,
        token_position_offsets: Option<&MlxArray>,
        boundary_checkpoint_collector: Option<&mut AstroSsdPromptCacheBoundaryCheckpointCollector>,
        paged_execution_mode: Qwen3_5MoEPagedExecutionMode,
        qwen4_exp_state: Option<&mut crate::qwen4_exp::request_state::Qwen4ExpRequestState>,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, TextExecutionError> {
        let Some(flash_next) = decoder_layer_weights.flash_next() else {
            return self.forward_qwen3_5_decoder_layer(Qwen3_5LayerCall {
                hidden_states,
                token_count,
                rope_offset_tokens,
                layer_index,
                paged_execution_mode,
                decoder_layer_weights,
                layer_model_state,
                token_position_offsets,
                boundary_checkpoint_collector,
                performance_attribution,
            });
        };
        self.forward_flash_next_decoder_layer(
            hidden_states,
            decoder_layer_weights,
            flash_next,
            layer_model_state,
            super::lane_mixing_decoder_layer::LaneMixingLayerInputs {
                token_count,
                rope_offset_tokens,
                layer_index,
                paged_execution_mode,
                request_state: qwen4_exp_state,
                boundary_checkpoint_collector,
            },
            performance_attribution,
        )
    }

    /// A Qwen 3.5 layer: the normalized attention and feed-forward halves.
    fn forward_qwen3_5_decoder_layer(
        &self,
        layer_call: Qwen3_5LayerCall<'_>,
    ) -> Result<MlxArray, TextExecutionError> {
        let attention_output = self.forward_decoder_layer_attention(
            layer_call.hidden_states,
            layer_call.token_count,
            layer_call.rope_offset_tokens,
            layer_call.layer_index,
            layer_call.decoder_layer_weights,
            layer_call.layer_model_state,
            layer_call.token_position_offsets,
            layer_call.boundary_checkpoint_collector,
            layer_call.performance_attribution,
        )?;
        self.forward_decoder_layer_feed_forward(
            &attention_output,
            layer_call.token_count,
            layer_call.layer_index,
            layer_call.decoder_layer_weights,
            layer_call.paged_execution_mode,
            layer_call.performance_attribution,
        )
    }
}

/// The inputs of one Qwen 3.5 layer forward, gathered so the Flash-Next
/// dispatch in `forward_decoder_layer` hands them on in one piece.
struct Qwen3_5LayerCall<'call> {
    hidden_states: &'call MlxArray,
    token_count: i32,
    rope_offset_tokens: i32,
    layer_index: usize,
    paged_execution_mode: Qwen3_5MoEPagedExecutionMode,
    decoder_layer_weights: &'call Qwen3_5DecoderLayerWeights,
    layer_model_state: &'call mut AstroDecoderStateCacheState,
    token_position_offsets: Option<&'call MlxArray>,
    boundary_checkpoint_collector:
        Option<&'call mut AstroSsdPromptCacheBoundaryCheckpointCollector>,
    performance_attribution: &'call mut PerformanceAttribution,
}
