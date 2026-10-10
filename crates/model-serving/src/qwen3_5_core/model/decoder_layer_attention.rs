//! Decoder-layer attention execution: the invariant half of the residual
//! sandwich that both engines share (`input RMS norm -> attention -> residual
//! add -> post-attention RMS norm`). Attention state mutation is delegated to
//! the linear-attention and full-attention owners. Keeping the residual
//! boundaries explicit provides one restart-safe attention output for chunk
//! recovery.

use astronomical_mlx_c_rust::MlxArray;

use crate::qwen3_5_core::decoder::Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector;
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5AttentionWeights;
use crate::qwen3_5_core::model_math::error::{
    Qwen3_5ExecutionError, invalid_request_decoder_state,
};
use crate::{DecoderCacheState, PerformanceAttribution, PerformanceOperation};

use super::base::Qwen3_5ModelBase;
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5DecoderLayerWeights;

/// Correct attention result retained as one decoder layer's restart boundary.
pub(crate) struct Qwen3_5DecoderLayerAttentionOutput {
    /// Hidden state after adding attention output; final MLP output adds to this.
    pub(crate) attention_residual: MlxArray,
    /// Normalized view consumed by either dense or mixture-of-experts feed-forward.
    pub(crate) normalized_attention: MlxArray,
}

impl Qwen3_5ModelBase {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_decoder_layer_attention(
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
    ) -> Result<Qwen3_5DecoderLayerAttentionOutput, Qwen3_5ExecutionError> {
        let normalized_input = self.runtime.rms_norm(
            hidden_states,
            &decoder_layer_weights.input_normalization_weight,
            f32::from_bits(self.config.rms_norm_epsilon_bits()),
        )?;
        let attention_forward_span_started_at = performance_attribution.begin_operation_span();
        // Decoder-cache enum shape is part of model correctness. A linear
        // attention layer must own convolution/recurrent state; a full-attention
        // layer must own append-only key/value state. Mismatch is never recoverable
        // by choosing the other branch.
        let attention_output = match (&decoder_layer_weights.attention_weights, layer_model_state) {
            (
                Qwen3_5AttentionWeights::Linear(linear_attention_weights),
                DecoderCacheState::Composite {
                    convolution,
                    recurrent,
                },
            ) => performance_attribution.measure_operation(
                PerformanceOperation::LinearAttentionGraphConstruction,
                |performance_attribution| {
                    self.forward_linear_attention(
                        &normalized_input,
                        token_count,
                        layer_index,
                        linear_attention_weights,
                        convolution,
                        recurrent,
                        boundary_checkpoint_collector,
                        performance_attribution,
                    )
                },
            ),
            (
                Qwen3_5AttentionWeights::Full(full_attention_weights),
                DecoderCacheState::AppendOnlyAttention { attention },
            ) => performance_attribution.measure_operation(
                PerformanceOperation::FullAttentionGraphConstruction,
                |_performance_attribution| {
                    self.forward_full_attention(
                        &normalized_input,
                        token_count,
                        rope_offset_tokens,
                        full_attention_weights,
                        attention,
                        token_position_offsets,
                    )
                },
            ),
            _ => Err(invalid_request_decoder_state(
                layer_index,
                "decoder state attention family does not match the bound layer weights",
            )),
        };
        performance_attribution.complete_operation_span(
            PerformanceOperation::AttentionForwardSpan,
            attention_forward_span_started_at,
        );
        // Multi-token prefill only: force one evaluation boundary per attention
        // family so the per-family graphics-processor time is isolated from the
        // chunk-terminal wait. The terminal wait then owns only the residual
        // work (route observations, final logits), keeping the attributed sum
        // honest. One-token decode skips the boundary so stage attribution never
        // serializes the latency-sensitive decode step.
        let attention_family_gpu_wait_operation = match &decoder_layer_weights.attention_weights {
            Qwen3_5AttentionWeights::Linear(_) => {
                PerformanceOperation::PrefillLinearAttentionGraphicsProcessorCompletionWait
            }
            Qwen3_5AttentionWeights::Full(_) => {
                PerformanceOperation::PrefillFullAttentionGraphicsProcessorCompletionWait
            }
        };
        let attention_output = match attention_output {
            Ok(attention_output) if performance_attribution.is_enabled() && token_count > 1 => {
                performance_attribution.measure_operation(
                    attention_family_gpu_wait_operation,
                    |_performance_attribution| self.runtime.evaluate_arrays(&[&attention_output]),
                )?;
                Ok(attention_output)
            }
            attention_output => attention_output,
        };
        // Delay `?` until after closing the attribution span so failed graph
        // construction is measured rather than silently leaving an open interval.
        let attention_residual = self.runtime.add(hidden_states, &attention_output?)?;
        let normalized_attention = self.runtime.rms_norm(
            &attention_residual,
            &decoder_layer_weights.post_attention_normalization_weight,
            f32::from_bits(self.config.rms_norm_epsilon_bits()),
        )?;
        Ok(Qwen3_5DecoderLayerAttentionOutput {
            attention_residual,
            normalized_attention,
        })
    }
}
