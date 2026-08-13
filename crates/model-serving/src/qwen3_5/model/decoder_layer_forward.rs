//! Decoder-layer attention and feed-forward execution.
//!
//! This file owns the invariant shared by dense and sparse decoder layers:
//!
//! `hidden -> input RMS norm -> attention -> residual add -> post-attention RMS
//! norm -> feed-forward -> residual add`.
//!
//! Attention state mutation and sparse expert paging are delegated to their
//! specialized owners. Keeping the two halves here makes the residual boundaries
//! explicit and provides one restart-safe attention output for chunk recovery.

use astronomical_runtime_integration::MlxArray;

use crate::qwen3_5::decoder::Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector;
use crate::qwen3_5_moe::Qwen3_5MoEPagedPrefillExecutionMode;
use crate::{DecoderCacheState, PerformanceAttribution, PerformanceOperation};

use super::decoder_layer_weights::{
    Qwen3_5AttentionWeights, Qwen3_5DecoderFeedForwardWeights, Qwen3_5DecoderLayerWeights,
};
use super::error::invalid_request_decoder_state;
use super::{Qwen3_5AttentionCapture, Qwen3_5ExecutionError, Qwen3_5Model};

/// Correct attention result retained as one decoder layer's restart boundary.
pub(crate) struct Qwen3_5DecoderLayerAttentionOutput {
    /// Hidden state after adding attention output; final MLP output adds to this.
    pub(crate) attention_residual: MlxArray,
    /// Normalized view consumed by either dense or mixture-of-experts feed-forward.
    pub(crate) normalized_attention: MlxArray,
}

impl Qwen3_5Model {
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
        attention_capture: Option<&mut Qwen3_5AttentionCapture>,
        boundary_checkpoint_collector: Option<
            &mut Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector,
        >,
        paged_prefill_execution_mode: Qwen3_5MoEPagedPrefillExecutionMode,
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
                |_performance_attribution| {
                    self.forward_linear_attention(
                        &normalized_input,
                        token_count,
                        layer_index,
                        linear_attention_weights,
                        convolution,
                        recurrent,
                        boundary_checkpoint_collector,
                        paged_prefill_execution_mode,
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
                        layer_index,
                        token_position_offsets,
                        attention_capture,
                        paged_prefill_execution_mode,
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

    pub(crate) fn forward_decoder_layer_feed_forward(
        &self,
        attention_output: &Qwen3_5DecoderLayerAttentionOutput,
        token_count: i32,
        layer_index: usize,
        decoder_layer_weights: &Qwen3_5DecoderLayerWeights,
        paged_prefill_execution_mode: Qwen3_5MoEPagedPrefillExecutionMode,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let should_use_compiled_elementwise_graphs = token_count != 1
            && paged_prefill_execution_mode
                != Qwen3_5MoEPagedPrefillExecutionMode::TargetVerificationWindow;
        // One-token decode keeps the low-overhead existing graph. Verification
        // windows use shape/ownership semantics incompatible with the ordinary
        // compiled multi-token elementwise closure.
        let mlp_forward_span_started_at = performance_attribution.begin_operation_span();
        let mlp_output = match &decoder_layer_weights.mlp_weights {
            Qwen3_5DecoderFeedForwardWeights::Dense(dense_mlp_weights) => self
                .forward_qwen3_5_dense_mlp(
                    &attention_output.normalized_attention,
                    dense_mlp_weights,
                    paged_prefill_execution_mode,
                ),
            Qwen3_5DecoderFeedForwardWeights::MixtureOfExperts(mixture_of_experts_weights) => {
                // Sparse layer metadata and router weights live in the decoder
                // layer, while source geometry and bounded loading live in the
                // pager. Both are required to execute exact routed experts.
                let expert_pager = self.expert_pager.as_ref().ok_or_else(|| {
                    Qwen3_5ExecutionError::MissingTensor {
                        tensor_name: "sparse model expert pager".to_owned(),
                    }
                })?;
                self.forward_qwen3_5_moe(
                    &attention_output.normalized_attention,
                    mixture_of_experts_weights,
                    expert_pager,
                    layer_index,
                    should_use_compiled_elementwise_graphs,
                    paged_prefill_execution_mode,
                    performance_attribution,
                )
            }
        };
        performance_attribution.complete_operation_span(
            PerformanceOperation::MlpForwardSpan,
            mlp_forward_span_started_at,
        );
        Ok(self
            .runtime
            .add(&attention_output.attention_residual, &mlp_output?)?)
    }
}
