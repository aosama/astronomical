//! Resident Qwen3.5 model: shared state and complete sparse-expert arrays.

use mlx_c_rust::MlxArray;

use std::cell::Cell;

use crate::qwen3_5_core::artifacts::QuantizedExpertLayerPlan;
use crate::qwen3_5_core::decoder::RequestDecoderStateStack;
use crate::qwen3_5_core::model::Qwen3_5ModelBase;
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5DecoderLayerWeights;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_core::model_math::forward_contract;
use crate::qwen3_5_resident::Qwen3_5ResidentExpertWeights;
use crate::{DecoderCacheState, ExpertWeightMemoryCacheStatistics, PerformanceAttribution};

use crate::qwen3_5_core::decoder::Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector;

/// The resident engine's model: shared base plus complete expert arrays.
#[derive(Debug)]
pub struct Qwen3_5ResidentModel {
    pub(crate) base: Qwen3_5ModelBase,
    /// Shared artifact-derived metadata used for loading and cache dtype flow.
    pub(crate) expert_layer_plans: Vec<QuantizedExpertLayerPlan>,
    /// Complete contiguous expert arrays when the whole sparse payload fits.
    pub(crate) resident_expert_weights: Option<Qwen3_5ResidentExpertWeights>,
    /// Opt-in Neural Engine programs for dense prefill; `None` keeps every chunk
    /// on the GPU.
    pub(crate) neural_engine_mlp_prefill:
        Option<crate::qwen3_5_resident::neural_engine_mlp_prefill::NeuralEngineMlpPrefill>,
}

impl std::ops::Deref for Qwen3_5ResidentModel {
    type Target = Qwen3_5ModelBase;

    fn deref(&self) -> &Qwen3_5ModelBase {
        &self.base
    }
}

impl std::ops::DerefMut for Qwen3_5ResidentModel {
    fn deref_mut(&mut self) -> &mut Qwen3_5ModelBase {
        &mut self.base
    }
}

/// Library size and uniform expert count from the resident layer plans.
///
/// One failed layer measurement publishes a zero library so Observatory does
/// not subtract the model file from the resident expert bytes. The capacity
/// is the first plan's count; resident layers share it.
fn resident_expert_library_facts(expert_layer_plans: &[QuantizedExpertLayerPlan]) -> (u64, u32) {
    let mut complete_expert_payload_byte_count = 0_u64;
    let mut expert_capacity = 0_u32;
    for expert_layer_plan in expert_layer_plans {
        if expert_capacity == 0 {
            expert_capacity = u32::try_from(expert_layer_plan.expert_capacity).unwrap_or(u32::MAX);
        }
        match expert_layer_plan.complete_expert_payload_byte_count() {
            Ok(layer_payload_byte_count) => {
                complete_expert_payload_byte_count =
                    complete_expert_payload_byte_count.saturating_add(layer_payload_byte_count);
            }
            Err(_) => return (0, expert_capacity),
        }
    }
    (complete_expert_payload_byte_count, expert_capacity)
}

/// Every resident layer is fully seated. A zero capacity means the plans did
/// not name an expert count, and a fabricated zero-count map would paint the
/// library as still on the SSD.
fn resident_layer_seating(
    expert_layer_plans: &[QuantizedExpertLayerPlan],
    expert_capacity: u32,
) -> astronomical_ipc_protocol::ExpertLayerSeatingCounts {
    if expert_capacity == 0 || expert_layer_plans.is_empty() {
        return astronomical_ipc_protocol::ExpertLayerSeatingCounts::default();
    }
    let seated_layer_count = expert_layer_plans
        .len()
        .min(astronomical_ipc_protocol::EXPERT_LAYER_SEATING_LIMIT);
    let mut retained_expert_counts = Vec::with_capacity(seated_layer_count);
    let mut layer_index = 0;
    while layer_index < seated_layer_count {
        retained_expert_counts.push(expert_capacity);
        layer_index += 1;
    }
    astronomical_ipc_protocol::ExpertLayerSeatingCounts::from_retained_counts(
        &retained_expert_counts,
    )
}

impl Qwen3_5ResidentModel {
    /// Materializes every bound target weight: the resident engine keeps the
    /// whole model in memory.
    pub fn materialize_target_weights(&self) -> Result<(), Qwen3_5ExecutionError> {
        self.base.materialize_target_weights_with(
            crate::qwen3_5_core::model_math::weights::DecoderLayerResidency::BoundWeights,
        )
    }
}

impl Qwen3_5ResidentModel {
    #[must_use]
    pub fn expert_memory_mode(&self) -> astronomical_ipc_protocol::ExpertMemoryMode {
        astronomical_ipc_protocol::ExpertMemoryMode::Resident
    }

    /// Creates an empty per-request decoder state whose per-layer cache
    /// dtypes match this model's loaded weights, exactly as production
    /// generation does.
    ///
    /// Direct-MLX probes drive raw forwards without the engine's request
    /// plumbing; this constructor keeps their decoder state identical to
    /// production. The public config-only constructor instead assumes
    /// BFloat16 caches, which can diverge from a loaded artifact's dtypes
    /// and would quietly measure a different graph than the one serving.
    #[cfg(feature = "direct-mlx")]
    pub fn empty_request_decoder_state_matching_loaded_weights(
        &self,
        full_attention_kv_state_growth_tokens: i32,
    ) -> Result<RequestDecoderStateStack, astronomical_runtime_integration::MlxRuntimeError> {
        let decoder_cache_layout = self.base.decoder_cache_layout().clone();
        RequestDecoderStateStack::
            empty_from_decoder_cache_layout_with_full_attention_kv_state_growth_tokens(
                &decoder_cache_layout,
                full_attention_kv_state_growth_tokens,
            )
    }

    /// Builds the opt-in Neural Engine programs through the same steps serving does,
    /// and waits for them, so a journey can measure one loaded model with and without.
    #[cfg(feature = "direct-mlx")]
    pub fn prepare_neural_engine_prefill_matching_production(
        &mut self,
        chunking: &astronomical_ipc_protocol::WorkerChunkingConfiguration,
        program_cache_directory: Option<&std::path::Path>,
    ) -> astronomical_ipc_protocol::NeuralEnginePrefillStatus {
        crate::qwen3_5_resident::neural_engine_prefill_loading::prepare_neural_engine_prefill(
            self,
            chunking,
            program_cache_directory,
            None,
        )
    }

    /// The same steps with a chosen token split, so a journey can sweep it.
    #[cfg(feature = "direct-mlx")]
    pub fn prepare_neural_engine_prefill_with_token_count(
        &mut self,
        chunking: &astronomical_ipc_protocol::WorkerChunkingConfiguration,
        neural_engine_token_count: u32,
    ) -> astronomical_ipc_protocol::NeuralEnginePrefillStatus {
        crate::qwen3_5_resident::neural_engine_prefill_loading::prepare_neural_engine_prefill(
            self,
            chunking,
            None,
            Some(neural_engine_token_count),
        )
    }

    /// Releases the programs so the next load in a measurement starts from none.
    #[cfg(feature = "direct-mlx")]
    pub fn release_neural_engine_mlp_prefill(&mut self) {
        self.neural_engine_mlp_prefill = None;
        self.base.output_gate_engine = None;
    }

    /// Sends full chunks to the GPU alone while paused, so one loaded model can
    /// alternate between the arms of a measurement.
    #[cfg(feature = "direct-mlx")]
    pub fn set_neural_engine_mlp_prefill_paused(&self, is_paused: bool) {
        if let Some(prefill) = self.neural_engine_mlp_prefill.as_ref() {
            prefill.set_paused(is_paused);
        }
    }

    /// As [`Self::set_neural_engine_mlp_prefill_paused`] for the output gate programs.
    #[cfg(feature = "direct-mlx")]
    pub fn set_neural_engine_output_gate_paused(&self, is_paused: bool) {
        if let Some(output_gate_engine) = self.base.output_gate_engine.as_ref() {
            output_gate_engine.set_paused(is_paused);
        }
    }

    /// Materializes dense target weights and the complete resident expert
    /// payload in the exact order production model loading uses.
    ///
    /// Direct-MLX probes need the same weight residency the daemon serves
    /// with: `load` alone binds shared metadata without evaluating the
    /// payload, so the first forward would fail with a missing resident
    /// expert tensor. The allocator-cache drain between the two stages
    /// mirrors production so transient load buffers cannot outlive the
    /// materialization the same way they must not in serving.
    #[cfg(feature = "direct-mlx")]
    pub fn materialize_complete_weights_matching_production(
        &mut self,
    ) -> Result<(), Qwen3_5ExecutionError> {
        self.materialize_target_weights()?;
        self.runtime()
            .synchronize_gpu_stream_and_clear_allocator_cache()?;
        if !self.expert_layer_plans.is_empty() {
            let resident_expert_weights =
                Qwen3_5ResidentExpertWeights::load(self.runtime(), &self.expert_layer_plans, None)?;
            self.resident_expert_weights = Some(resident_expert_weights);
        }
        Ok(())
    }

    pub(crate) fn expert_residency_telemetry_for_breakdown(
        &self,
        active_memory_breakdown: &crate::memory::telemetry::MlxActiveMemoryBreakdown,
    ) -> crate::inference_engine::contract::ExpertResidencyTelemetry {
        let resident_expert_statistics = self.expert_weight_memory_cache_statistics();
        let (complete_expert_payload_byte_count, expert_capacity) =
            resident_expert_library_facts(&self.expert_layer_plans);
        crate::inference_engine::contract::ExpertResidencyTelemetry {
            total_layer_count: u32::try_from(resident_expert_statistics.complete_layer_count)
                .unwrap_or(u32::MAX),
            resident_expert_count: u32::try_from(resident_expert_statistics.entry_count)
                .unwrap_or(u32::MAX),
            resident_expert_payload_bytes: active_memory_breakdown.expert_payload_bytes,
            complete_expert_payload_byte_count,
            expert_capacity,
            layer_seating: resident_layer_seating(&self.expert_layer_plans, expert_capacity),
        }
    }

    /// Returns complete-owner expert-memory telemetry without page counters.
    #[must_use]
    pub fn expert_weight_memory_cache_statistics(&self) -> ExpertWeightMemoryCacheStatistics {
        if let Some(resident_expert_weights) = self.resident_expert_weights.as_ref() {
            return ExpertWeightMemoryCacheStatistics {
                entry_count: resident_expert_weights.expert_entry_count(),
                resident_payload_byte_count: resident_expert_weights.payload_byte_count(),
                maximum_resident_payload_byte_count: resident_expert_weights.payload_byte_count(),
                eviction_count: 0,
                disk_page_load_count: 0,
                disk_batch_load_count: 0,
                complete_layer_count: resident_expert_weights.layer_count(),
                complete_layer_payload_byte_count: resident_expert_weights.payload_byte_count(),
                partial_layer_count: 0,
                partial_layer_payload_byte_count: 0,
                mandatory_read_promotion_count: 0,
                complete_layer_eviction_count: 0,
                partial_layer_eviction_count: 0,
            };
        }
        ExpertWeightMemoryCacheStatistics::default()
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
    ) -> Result<usize, Qwen3_5ExecutionError> {
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
    ) -> Result<(), Qwen3_5ExecutionError> {
        let mut disabled_performance_attribution = PerformanceAttribution::disabled();
        self.prefill_chunk_with_performance_attribution(
            token_ids,
            starting_position_tokens,
            request_decoder_state,
            &mut disabled_performance_attribution,
        )
    }

    /// Executes one prompt or decode chunk and materializes final logits plus all layer state.
    pub fn forward_chunk(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let mut disabled_performance_attribution = PerformanceAttribution::disabled();
        self.forward_chunk_with_performance_attribution(
            token_ids,
            starting_position_tokens,
            request_decoder_state,
            &mut disabled_performance_attribution,
        )
    }

    // Decoder-layer inputs stay explicit instead of introducing another per-layer facade.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_decoder_layer(
        &self,
        hidden_states: &MlxArray,
        token_count: i32,
        rope_offset_tokens: i32,
        layer_index: usize,
        decoder_layer_weights: &Qwen3_5DecoderLayerWeights,
        layer_model_state: &mut DecoderCacheState,
        token_position_offsets: Option<&MlxArray>,
        boundary_checkpoint_collector: Option<
            &mut Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector,
        >,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let attention_output = self.forward_decoder_layer_attention(
            hidden_states,
            token_count,
            rope_offset_tokens,
            layer_index,
            decoder_layer_weights,
            layer_model_state,
            token_position_offsets,
            boundary_checkpoint_collector,
            performance_attribution,
        )?;
        self.forward_decoder_layer_feed_forward(
            &attention_output,
            token_count,
            layer_index,
            decoder_layer_weights,
            performance_attribution,
        )
    }
}

#[cfg(test)]
mod resident_expert_library_facts_tests;
