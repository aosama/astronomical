//! The in-memory Qwen3.5 model the quantization sensitivity probe swaps
//! weights in: the shared base plus every routed expert held whole, so one
//! projection can be re-encoded in place and put back. Serving runs the
//! streaming model; this one exists for the probe alone, because the
//! streaming model keeps its experts in the `astro_expert_cache` and reads
//! the rest from the SSD, where a swapped copy cannot be put.

use crate::qwen3_5_core::model::prefill_forward::Qwen3_5PrefillForward;
use mlx_bridge::MlxArray;

use crate::qwen3_5_core::artifacts::QuantizedExpertLayerPlan;
use crate::qwen3_5_core::model::Qwen3_5ModelBase;
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5DecoderLayerWeights;
use crate::qwen3_5_resident::Qwen3_5ResidentExpertWeights;
use crate::{AstroDecoderStateCacheState, PerformanceAttribution};
use text_generation::RequestDecoderStateStack;
use text_generation::execution_error::TextExecutionError;

use text_generation::AstroSsdPromptCacheBoundaryCheckpointCollector;

/// The probe's model: shared base plus complete expert arrays.
#[derive(Debug)]
pub struct Qwen3_5ResidentModel {
    pub base: Qwen3_5ModelBase,
    /// Shared artifact-derived metadata used for loading and `astro_decoder_state_cache` dtype flow.
    pub expert_layer_plans: Vec<QuantizedExpertLayerPlan>,
    /// Complete contiguous expert arrays once the weights are materialized.
    pub resident_expert_weights: Option<Qwen3_5ResidentExpertWeights>,
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

impl std::ops::DerefMut for Qwen3_5ResidentModel {
    fn deref_mut(&mut self) -> &mut Qwen3_5ModelBase {
        &mut self.base
    }
}

impl Qwen3_5ResidentModel {
    /// Materializes every bound target weight: this model keeps the whole
    /// model in memory.
    pub fn materialize_target_weights(&self) -> Result<(), TextExecutionError> {
        self.base.materialize_target_weights_with(
            crate::qwen3_5_core::model_math::weights::DecoderLayerResidency::BoundWeights,
        )
    }
}

impl Qwen3_5ResidentModel {
    /// Materializes dense target weights and the complete resident expert
    /// payload in the exact order production model loading uses.
    ///
    /// `load` alone binds shared metadata without evaluating the payload, so
    /// the first forward would fail with a missing resident expert tensor.
    /// The `mlx_buffer_cache` drain between the two stages keeps transient
    /// load buffers from outliving the materialization.
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
