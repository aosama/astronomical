//! Resident Qwen3.5 model: shared state and complete sparse-expert arrays.

use astronomical_mlx_c_rust::MlxArray;

use std::cell::Cell;

use crate::expert_paging::{ExpertWeightMemoryCacheStatistics, QuantizedExpertLayerPlan};
use crate::qwen3_5_core::decoder::RequestDecoderStateStack;
use crate::qwen3_5_core::model::Qwen3_5ModelBase;
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5DecoderLayerWeights;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_core::model_math::forward_contract;
use crate::qwen3_5_resident::Qwen3_5ResidentExpertWeights;
use crate::{DecoderCacheState, PerformanceAttribution};

use crate::qwen3_5_core::decoder::Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector;

/// The resident engine's model: shared base plus complete expert arrays.
#[derive(Debug)]
pub struct Qwen3_5ResidentModel {
    pub(crate) base: Qwen3_5ModelBase,
    /// Shared artifact-derived metadata used for loading and cache dtype flow.
    pub(crate) expert_layer_plans: Vec<QuantizedExpertLayerPlan>,
    /// Complete contiguous expert arrays when the whole sparse payload fits.
    pub(crate) resident_expert_weights: Option<Qwen3_5ResidentExpertWeights>,
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

impl Qwen3_5ResidentModel {
    #[must_use]
    pub fn expert_memory_mode(&self) -> astronomical_ipc_protocol::ExpertMemoryMode {
        astronomical_ipc_protocol::ExpertMemoryMode::Resident
    }

    pub(crate) fn expert_residency_telemetry_for_breakdown(
        &self,
        active_memory_breakdown: &crate::MlxActiveMemoryBreakdown,
    ) -> crate::ExpertResidencyTelemetry {
        let resident_expert_statistics = self.expert_weight_memory_cache_statistics();
        crate::ExpertResidencyTelemetry {
            total_layer_count: u32::try_from(resident_expert_statistics.complete_layer_count)
                .unwrap_or(u32::MAX),
            resident_expert_count: u32::try_from(resident_expert_statistics.entry_count)
                .unwrap_or(u32::MAX),
            resident_expert_payload_bytes: active_memory_breakdown.expert_payload_bytes,
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
