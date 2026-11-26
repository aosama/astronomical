//! The probe model itself: the shared base plus every routed expert held
//! whole (see the `probe_model` module for why the probe keeps its own).

use mlx_bridge::MlxArray;
use qwen35::qwen3_5_core::model::prefill_forward::Qwen3_5PrefillForward;

use crate::probe_model::experts::Qwen3_5ProbeExpertWeights;
use crate::{AstroDecoderStateCacheState, PerformanceAttribution};
use qwen35::qwen3_5_core::artifacts::QuantizedExpertLayerPlan;
use qwen35::qwen3_5_core::model::Qwen3_5ModelBase;
use qwen35::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5DecoderLayerWeights;
use text_generation::RequestDecoderStateStack;
use text_generation::execution_error::TextExecutionError;

use text_generation::AstroSsdPromptCacheBoundaryCheckpointCollector;

/// The probe's model: shared base plus complete expert arrays.
#[derive(Debug)]
pub struct Qwen3_5ProbeModel {
    pub base: Qwen3_5ModelBase,
    /// Shared artifact-derived metadata used for loading and `astro_decoder_state_cache` dtype flow.
    pub expert_layer_plans: Vec<QuantizedExpertLayerPlan>,
    /// Complete contiguous expert arrays once the weights are materialized.
    pub held_expert_weights: Option<Qwen3_5ProbeExpertWeights>,
}

impl std::ops::Deref for Qwen3_5ProbeModel {
    type Target = Qwen3_5ModelBase;

    fn deref(&self) -> &Qwen3_5ModelBase {
        &self.base
    }
}

impl std::ops::DerefMut for Qwen3_5ProbeModel {
    fn deref_mut(&mut self) -> &mut Qwen3_5ModelBase {
        &mut self.base
    }
}

impl Qwen3_5ProbeModel {
    /// Materializes every bound target weight: this model keeps the whole
    /// model in memory.
    pub fn materialize_target_weights(&self) -> Result<(), TextExecutionError> {
        self.base.materialize_target_weights_with(
            qwen35::qwen3_5_core::model_math::weights::DecoderLayerResidency::BoundWeights,
        )
    }
}

impl Qwen3_5ProbeModel {
    /// Materializes dense target weights and the complete held expert
    /// payload in the exact order production model loading uses.
    ///
    /// `load` alone binds shared metadata without evaluating the payload, so
    /// the first forward would fail with a missing held expert tensor.
    /// The `mlx_buffer_cache` drain between the two stages keeps transient
    /// load buffers from outliving the materialization.
    pub fn materialize_every_weight(&mut self) -> Result<(), TextExecutionError> {
        self.materialize_target_weights()?;
        self.runtime()
            .synchronize_gpu_stream_and_clear_mlx_buffer_cache()?;
        if !self.expert_layer_plans.is_empty() {
            let held_expert_weights =
                Qwen3_5ProbeExpertWeights::load(self.runtime(), &self.expert_layer_plans, None)?;
            self.held_expert_weights = Some(held_expert_weights);
        }
        Ok(())
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
