//! Decoder-layer feed-forward execution for the streaming engine: the half of
//! the residual sandwich that forks by feed-forward architecture (dense MLP
//! versus routed mixture-of-experts with expert paging).

use astronomical_mlx_c_rust::MlxArray;

use crate::qwen3_5_core::model::Qwen3_5DecoderLayerAttentionOutput;
use crate::qwen3_5_core::model_math::decoder_layer_weights::{
    Qwen3_5DecoderFeedForwardWeights, Qwen3_5DecoderLayerWeights,
};
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_streaming::Qwen3_5MoEPagedPrefillExecutionMode;
use crate::{PerformanceAttribution, PerformanceOperation};

use crate::qwen3_5_streaming::model::streaming_model::Qwen3_5StreamingModel;

impl Qwen3_5StreamingModel {
    pub(crate) fn forward_decoder_layer_feed_forward(
        &self,
        attention_output: &Qwen3_5DecoderLayerAttentionOutput,
        token_count: i32,
        layer_index: usize,
        decoder_layer_weights: &Qwen3_5DecoderLayerWeights,
        paged_prefill_execution_mode: Qwen3_5MoEPagedPrefillExecutionMode,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let should_use_compiled_elementwise_graphs = token_count != 1;
        // One-token decode keeps the low-overhead existing graph.
        let mlp_forward_span_started_at = performance_attribution.begin_operation_span();
        let mlp_output = match &decoder_layer_weights.mlp_weights {
            Qwen3_5DecoderFeedForwardWeights::Dense(dense_mlp_weights) => self
                .forward_qwen3_5_dense_mlp(
                    &attention_output.normalized_attention,
                    dense_mlp_weights,
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
        // Multi-token prefill only: the feed-forward family (routed experts,
        // weighted reduction, shared expert) owns its own evaluation boundary so
        // its graphics-processor time is attributed separately from the chunk
        // terminal wait. See the attention boundary above for the rationale.
        let mlp_output = match mlp_output {
            Ok(mlp_output) if performance_attribution.is_enabled() && token_count > 1 => {
                performance_attribution.measure_operation(
                    PerformanceOperation::PrefillFeedForwardGraphicsProcessorCompletionWait,
                    |_performance_attribution| self.runtime.evaluate_arrays(&[&mlp_output]),
                )?;
                Ok(mlp_output)
            }
            mlp_output => mlp_output,
        };
        Ok(self
            .runtime
            .add(&attention_output.attention_residual, &mlp_output?)?)
    }
}
