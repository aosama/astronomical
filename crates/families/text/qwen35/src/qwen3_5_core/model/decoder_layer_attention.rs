//! Decoder-layer attention execution: the invariant half of the residual
//! sandwich (`input RMS norm -> attention -> residual
//! add -> post-attention RMS norm`). Attention state mutation is delegated to
//! the linear-attention and full-attention owners. Keeping the residual
//! boundaries explicit provides one restart-safe attention output for chunk
//! recovery.

use mlx_bridge::MlxArray;

use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5AttentionWeights;
use crate::{AstroDecoderStateCacheState, PerformanceAttribution, PerformanceOperation};
use text_generation::AstroSsdPromptCacheBoundaryCheckpointCollector;
use text_generation::execution_error::{TextExecutionError, invalid_request_decoder_state};

use super::base::Qwen3_5ModelBase;
use crate::qwen3_5_core::model_math::decoder_layer_weights::Qwen3_5DecoderLayerWeights;

/// Correct attention result retained as one decoder layer's restart boundary.
pub struct Qwen3_5DecoderLayerAttentionOutput {
    /// Hidden state after adding attention output; final MLP output adds to this.
    pub attention_residual: MlxArray,
    /// Normalized view consumed by either dense or mixture-of-experts feed-forward.
    pub normalized_attention: MlxArray,
}

impl Qwen3_5ModelBase {
    #[allow(clippy::too_many_arguments)]
    pub fn forward_decoder_layer_attention(
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
    ) -> Result<Qwen3_5DecoderLayerAttentionOutput, TextExecutionError> {
        let normalized_input = self.runtime.rms_norm(
            hidden_states,
            &decoder_layer_weights.input_normalization_weight,
            f32::from_bits(self.config.rms_norm_epsilon_bits()),
        )?;
        let attention_forward_span_started_at = performance_attribution.begin_operation_span();
        // The `astro_decoder_state_cache` enum shape is part of model correctness. A linear
        // attention layer must own convolution/recurrent state; a full-attention
        // layer must own append-only key/value state. Mismatch is never recoverable
        // by choosing the other branch.
        let attention_output = match (&decoder_layer_weights.attention_weights, layer_model_state) {
            (
                Qwen3_5AttentionWeights::Linear(linear_attention_weights),
                AstroDecoderStateCacheState::Composite {
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
                AstroDecoderStateCacheState::AppendOnlyAttention { attention },
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
        let attention_output = self.isolate_prefill_attention_wait(
            attention_output,
            &decoder_layer_weights.attention_weights,
            token_count,
            performance_attribution,
        );
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

    /// Multi-token prefill only: force one evaluation boundary per attention
    /// family so the per-family graphics-processor time is isolated from the
    /// chunk-terminal wait. The terminal wait then owns only the residual
    /// work (route observations, final logits), keeping the attributed sum
    /// honest. One-token decode skips the boundary so stage attribution never
    /// serializes the latency-sensitive decode step. A graph-construction
    /// error passes through untouched; a failed evaluation is returned in
    /// its place.
    fn isolate_prefill_attention_wait(
        &self,
        attention_output: Result<MlxArray, TextExecutionError>,
        attention_weights: &Qwen3_5AttentionWeights,
        token_count: i32,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, TextExecutionError> {
        let attention_family_gpu_wait_operation =
            prefill_attention_wait_operation(attention_weights);
        match attention_output {
            Ok(attention_output)
                if should_isolate_prefill_attention_wait(
                    performance_attribution.is_enabled(),
                    token_count,
                ) =>
            {
                performance_attribution.measure_operation(
                    attention_family_gpu_wait_operation,
                    |_performance_attribution| self.runtime.evaluate_arrays(&[&attention_output]),
                )?;
                Ok(attention_output)
            }
            attention_output => attention_output,
        }
    }
}

/// Which prefill completion wait an attention family is attributed to.
fn prefill_attention_wait_operation(
    attention_weights: &Qwen3_5AttentionWeights,
) -> PerformanceOperation {
    match attention_weights {
        Qwen3_5AttentionWeights::Linear(_) => {
            PerformanceOperation::PrefillLinearAttentionGraphicsProcessorCompletionWait
        }
        Qwen3_5AttentionWeights::Full(_) => {
            PerformanceOperation::PrefillFullAttentionGraphicsProcessorCompletionWait
        }
    }
}

/// Whether to wait on the attention output separately: only while performance
/// attribution is enabled and the step reads more than one token.
fn should_isolate_prefill_attention_wait(
    performance_attribution_enabled: bool,
    token_count: i32,
) -> bool {
    performance_attribution_enabled && token_count > 1
}

#[cfg(test)]
#[path = "decoder_layer_attention_tests.rs"]
mod tests;
