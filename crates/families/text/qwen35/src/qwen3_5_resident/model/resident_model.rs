//! Resident Qwen3.5 model: shared state and complete sparse-expert arrays.

use crate::qwen3_5_core::model::prefill_forward::Qwen3_5PrefillForward;
use mlx_bridge::MlxArray;

use crate::qwen3_5_core::artifacts::QuantizedExpertLayerPlan;
use crate::qwen3_5_core::model::Qwen3_5ModelBase;
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5DecoderLayerWeights;
use crate::qwen3_5_resident::Qwen3_5ResidentExpertWeights;
use crate::{AstroDecoderStateCacheState, AstroExpertCacheStatistics, PerformanceAttribution};
use text_generation::RequestDecoderStateStack;
use text_generation::execution_error::TextExecutionError;

use text_generation::AstroSsdPromptCacheBoundaryCheckpointCollector;

/// The resident engine's model: shared base plus complete expert arrays.
#[derive(Debug)]
pub struct Qwen3_5ResidentModel {
    pub base: Qwen3_5ModelBase,
    /// Shared artifact-derived metadata used for loading and `astro_decoder_state_cache` dtype flow.
    pub expert_layer_plans: Vec<QuantizedExpertLayerPlan>,
    /// Complete contiguous expert arrays when the whole sparse payload fits.
    pub resident_expert_weights: Option<Qwen3_5ResidentExpertWeights>,
    /// The owner of this model's memory ledger, opened once the model has
    /// loaded and its MLX working buffers have been measured.
    pub memory_governor: std::cell::RefCell<Option<memory_policy::ledger::MemoryGovernor>>,
}

crate::qwen3_5_core::model::serving_model::serve_text_from_wrapped_model_base!(
    Qwen3_5ResidentModel
);

impl std::ops::Deref for Qwen3_5ResidentModel {
    type Target = Qwen3_5ModelBase;

    fn deref(&self) -> &Qwen3_5ModelBase {
        &self.base
    }
}

impl text_generation::model::memory_breakdown::TextModelMemoryOwnership for Qwen3_5ResidentModel {
    fn mlx_active_memory_ceiling_bytes(&self) -> u64 {
        self.base.mlx_active_memory_ceiling_bytes()
    }

    fn held_expert_payload_bytes(&self) -> u64 {
        self.astro_expert_cache_statistics()
            .resident_payload_byte_count
    }

    fn held_model_core_payload_bytes(&self) -> u64 {
        self.base.resident_model_payload_byte_count()
    }

    fn governed_memory_reserve_snapshot(&self) -> Option<crate::memory::MemoryReserveSnapshot> {
        let memory_governor = self.memory_governor.borrow();
        let accounts = memory_governor.as_ref()?.accounts();
        // The ledger counts every weight as fixed; the breakdown shows the
        // experts apart from the rest of the model.
        let held_expert_payload_bytes = self.held_expert_payload_bytes();
        let mut budget_snapshot =
            crate::memory::MemoryReserveSnapshot::from_ledger_accounts(&accounts);
        budget_snapshot.model_core_payload_bytes = budget_snapshot
            .model_core_payload_bytes
            .saturating_sub(held_expert_payload_bytes);
        budget_snapshot.retained_expert_budget_bytes = held_expert_payload_bytes;
        Some(budget_snapshot)
    }
}

impl Qwen3_5ResidentModel {
    /// The memory governor's view after the last measured forward, with its
    /// accounts compared against what MLX reported; `None` before the first.
    pub fn latest_memory_ledger_reading(&self) -> Option<memory_policy::ledger::LedgerReading> {
        self.memory_governor
            .borrow()
            .as_ref()
            .and_then(memory_policy::ledger::MemoryGovernor::latest_reading)
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
) -> ipc_protocol::ExpertLayerSeatingCounts {
    if expert_capacity == 0 || expert_layer_plans.is_empty() {
        return ipc_protocol::ExpertLayerSeatingCounts::default();
    }
    let seated_layer_count = expert_layer_plans
        .len()
        .min(ipc_protocol::EXPERT_LAYER_SEATING_LIMIT);
    let mut retained_expert_counts = Vec::with_capacity(seated_layer_count);
    let mut layer_index = 0;
    while layer_index < seated_layer_count {
        retained_expert_counts.push(expert_capacity);
        layer_index += 1;
    }
    ipc_protocol::ExpertLayerSeatingCounts::from_retained_counts(&retained_expert_counts)
}

impl Qwen3_5ResidentModel {
    /// Materializes every bound target weight: the resident engine keeps the
    /// whole model in memory.
    pub fn materialize_target_weights(&self) -> Result<(), TextExecutionError> {
        self.base.materialize_target_weights_with(
            crate::qwen3_5_core::model_math::weights::DecoderLayerResidency::BoundWeights,
        )
    }
}

impl Qwen3_5ResidentModel {
    #[must_use]
    pub fn expert_memory_mode(&self) -> ipc_protocol::ExpertMemoryMode {
        ipc_protocol::ExpertMemoryMode::Resident
    }

    /// Materializes dense target weights and the complete resident expert
    /// payload in the exact order production model loading uses.
    ///
    /// Direct-MLX probes need the same weight residency the daemon serves
    /// with: `load` alone binds shared metadata without evaluating the
    /// payload, so the first forward would fail with a missing resident
    /// expert tensor. The `mlx_buffer_cache` drain between the two stages
    /// mirrors production so transient load buffers cannot outlive the
    /// materialization the same way they must not in serving.
    #[cfg(feature = "direct-mlx")]
    pub fn materialize_complete_weights_matching_production(
        &mut self,
    ) -> Result<(), TextExecutionError> {
        self.materialize_target_weights()?;
        self.runtime()
            .synchronize_gpu_stream_and_clear_mlx_buffer_cache()?;
        if !self.expert_layer_plans.is_empty() {
            let resident_expert_weights =
                Qwen3_5ResidentExpertWeights::load(self.runtime(), &self.expert_layer_plans, None)?;
            self.resident_expert_weights = Some(resident_expert_weights);
        }
        Ok(())
    }

    pub fn expert_residency_telemetry_for_breakdown(
        &self,
        active_memory_breakdown: &crate::memory::telemetry::MlxActiveMemoryBreakdown,
    ) -> crate::inference_engine::contract::ExpertResidencyTelemetry {
        let resident_expert_statistics = self.astro_expert_cache_statistics();
        let (complete_expert_payload_byte_count, expert_capacity) =
            resident_expert_library_facts(&self.expert_layer_plans);
        crate::inference_engine::contract::ExpertResidencyTelemetry {
            dense_layers: None,
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
    pub fn astro_expert_cache_statistics(&self) -> AstroExpertCacheStatistics {
        if let Some(resident_expert_weights) = self.resident_expert_weights.as_ref() {
            return AstroExpertCacheStatistics {
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
        AstroExpertCacheStatistics::default()
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
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, TextExecutionError> {
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
