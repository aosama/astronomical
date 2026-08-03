//! Direct Qwen3.5-MoE text execution for the pinned model artifact.

use astronomical_runtime_integration::{
    MlxArray, MlxCompiledElementwiseGraphs, MlxCompiledSwiGlu, MlxMetalKernel, MlxRuntime,
};

use std::cell::RefCell;

use crate::{DecoderCacheLayout, DecoderCacheState, PerformanceAttribution, PerformanceOperation};

use super::decoder_layer_weights::{
    Qwen3_5DecoderLayerMlpWeights, Qwen3_5MoEAffineWeights, Qwen3_5MoEAttentionWeights,
    Qwen3_5MoEDecoderLayerWeights,
};
use super::error::invalid_request_decoder_state;
use super::expert_paging::{
    ExpertWeightMemoryCache, ExpertWeightMemoryCacheStatistics, expert_pager::ExpertPager,
};
use super::forward_contract::validate_forward_input;
use super::mtp::Qwen3_5MoEMtpWeights;
use super::{
    Qwen3_5MoEConfig, Qwen3_5MoEExecutionError, Qwen3_5MoEPagedPrefillExecutionMode,
    Qwen3_5MoEVisionModel, Qwen3_5MoEWeights, RequestDecoderStateStack,
};

/// One resident native Qwen3.5-MoE text model, optional vision tower, and its direct MLX runtime.
#[derive(Debug)]
pub struct Qwen3_5MoEModel {
    pub(in crate::qwen3_5_moe) runtime: MlxRuntime,
    pub(in crate::qwen3_5_moe) config: Qwen3_5MoEConfig,
    pub(in crate::qwen3_5_moe) decoder_cache_layout: DecoderCacheLayout,
    pub(in crate::qwen3_5_moe) weights: Qwen3_5MoEWeights,
    pub(super) mtp_weights: Option<Qwen3_5MoEMtpWeights>,
    pub(super) vision_model: Option<Qwen3_5MoEVisionModel>,
    /// Sparse models own a pager; dense models have no sparse-expert weights.
    pub(super) expert_pager: Option<ExpertPager>,
    pub(super) expert_weight_memory_cache: RefCell<ExpertWeightMemoryCache>,
    pub(super) gated_delta_kernel: MlxMetalKernel,
    pub(super) sorted_expert_weighted_sum_kernel: MlxMetalKernel,
    pub(super) compiled_swiglu: MlxCompiledSwiGlu,
    pub(super) compiled_elementwise_graphs: MlxCompiledElementwiseGraphs,
    /// Model-owned BF16 scalar for the query normalization scale in every
    /// linear-attention forward pass.
    pub(super) inverse_linear_head_dimension_scale: MlxArray,
    /// Model-owned BF16 scalar for the key normalization scale in every
    /// linear-attention forward pass.
    pub(super) inverse_square_root_linear_head_dimension_scale: MlxArray,
}

impl Qwen3_5MoEModel {
    /// Returns the MLX runtime used by this model.
    #[must_use]
    pub fn runtime(&self) -> &MlxRuntime {
        &self.runtime
    }

    /// Returns cumulative expert memory-cache counters for low-level performance tests.
    #[must_use]
    pub fn expert_weight_memory_cache_statistics(&self) -> ExpertWeightMemoryCacheStatistics {
        self.expert_weight_memory_cache.borrow().statistics()
    }

    /// Returns the validated text configuration bound to this loaded model.
    #[must_use]
    pub(in crate::qwen3_5_moe) const fn config(&self) -> &Qwen3_5MoEConfig {
        &self.config
    }

    /// Returns the validated decoder-cache layout bound to this model artifact.
    #[must_use]
    pub(in crate::qwen3_5_moe) const fn decoder_cache_layout(&self) -> &DecoderCacheLayout {
        &self.decoder_cache_layout
    }

    /// Returns the optional vision tower loaded beside the language model.
    #[must_use]
    pub fn vision_model(&self) -> Option<&Qwen3_5MoEVisionModel> {
        self.vision_model.as_ref()
    }

    /// Returns whether a compatible resident MTP head is available.
    #[must_use]
    pub fn mtp_weights(&self) -> bool {
        self.mtp_weights.is_some()
    }

    /// Executes one prompt chunk, injecting visual embeddings at image_pad positions.
    ///
    /// `chunk_token_ids` are the token IDs for this prefill chunk.
    /// `visual_embeddings` carries the full visual embedding tensor for all images in the request.
    /// `starting_visual_embedding_index` tracks how many visual embeddings earlier chunks consumed.
    /// Returns the count of visual embeddings consumed by this chunk.
    pub fn prefill_chunck_with_visual_embeddings(
        &self,
        chunk_token_ids: &[u32],
        starting_position_tokens: u32,
        visual_embeddings: &MlxArray,
        starting_visual_embedding_index: usize,
        request_decoder_state: &mut RequestDecoderStateStack,
        image_pad_token_id: u32,
    ) -> Result<usize, Qwen3_5MoEExecutionError> {
        let mut disabled_performance_attribution = PerformanceAttribution::disabled();
        self.prefill_chunck_with_visual_embeddings_and_performance_attribution(
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
    pub fn prefill_chunck(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
    ) -> Result<(), Qwen3_5MoEExecutionError> {
        let mut disabled_performance_attribution = PerformanceAttribution::disabled();
        self.prefill_chunck_with_performance_attribution(
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
    ) -> Result<MlxArray, Qwen3_5MoEExecutionError> {
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
    pub fn forward_chunk_with_paged_prefill_execution_mode_for_tests(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        paged_prefill_execution_mode: Qwen3_5MoEPagedPrefillExecutionMode,
    ) -> Result<MlxArray, Qwen3_5MoEExecutionError> {
        let mut disabled_performance_attribution = PerformanceAttribution::disabled();
        let final_logits = self
            .build_forward_chunk_with_paged_prefill_execution_mode_and_performance_attribution(
                token_ids,
                starting_position_tokens,
                request_decoder_state,
                paged_prefill_execution_mode,
                &mut disabled_performance_attribution,
            )?;
        self.evaluate_forward_state(&final_logits, request_decoder_state)?;
        Ok(final_logits)
    }

    pub(super) fn build_forward_chunk_with_paged_prefill_execution_mode_and_performance_attribution(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        paged_prefill_execution_mode: Qwen3_5MoEPagedPrefillExecutionMode,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5MoEExecutionError> {
        let token_count = validate_forward_input(
            token_ids,
            starting_position_tokens,
            request_decoder_state.layer_count(),
            self.config.layer_count() as usize,
            self.config.vocabulary_size(),
            self.config.maximum_position_count(),
        )?;
        let signed_token_ids = token_ids
            .iter()
            .map(|token_id| {
                i32::try_from(*token_id).map_err(|_| Qwen3_5MoEExecutionError::InvalidInput {
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
            paged_prefill_execution_mode,
            performance_attribution,
        )
    }

    pub(in crate::qwen3_5_moe) fn materialize_target_weights(
        &self,
    ) -> Result<(), Qwen3_5MoEExecutionError> {
        self.weights.materialize(&self.runtime)?;
        self.runtime.evaluate_arrays(&[
            &self.inverse_linear_head_dimension_scale,
            &self.inverse_square_root_linear_head_dimension_scale,
        ])?;
        Ok(())
    }

    pub(in crate::qwen3_5_moe) fn materialize_mtp_weights(
        &mut self,
    ) -> Result<(), Qwen3_5MoEExecutionError> {
        let Some(mtp_weights) = self.mtp_weights.as_ref() else {
            return Ok(());
        };
        if let Err(mtp_materialization_error) = mtp_weights.materialize(&self.runtime) {
            self.mtp_weights = None;
            return Err(mtp_materialization_error);
        }
        Ok(())
    }

    pub(super) fn embedding_lookup(
        &self,
        token_indices: &MlxArray,
    ) -> Result<MlxArray, Qwen3_5MoEExecutionError> {
        match &self.weights.embedding_weights {
            Qwen3_5MoEAffineWeights::NativeBfloat16 { weight } => {
                Ok(self.runtime.take_axis(weight, token_indices, 0)?)
            }
            Qwen3_5MoEAffineWeights::Quantized {
                packed_weight,
                quantization_scales,
                quantization_biases,
                quantization_group_size,
                quantization_bits,
            } => {
                let selected_weights = self.runtime.take_axis(packed_weight, token_indices, 0)?;
                let selected_scales =
                    self.runtime
                        .take_axis(quantization_scales, token_indices, 0)?;
                let selected_biases =
                    self.runtime
                        .take_axis(quantization_biases, token_indices, 0)?;
                Ok(self.runtime.dequantize_affine(
                    &selected_weights,
                    &selected_scales,
                    &selected_biases,
                    *quantization_group_size,
                    *quantization_bits,
                )?)
            }
        }
    }

    // Decoder-layer inputs stay explicit instead of introducing another per-layer facade.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::qwen3_5_moe::model) fn forward_decoder_layer(
        &self,
        hidden_states: &MlxArray,
        token_count: i32,
        rope_offset_tokens: i32,
        layer_index: usize,
        decoder_layer_weights: &Qwen3_5MoEDecoderLayerWeights,
        layer_model_state: &mut DecoderCacheState,
        paged_prefill_execution_mode: Qwen3_5MoEPagedPrefillExecutionMode,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5MoEExecutionError> {
        let normalized_input = self.runtime.rms_norm(
            hidden_states,
            &decoder_layer_weights.input_normalization_weight,
            f32::from_bits(self.config.rms_norm_epsilon_bits()),
        )?;
        let attention_forward_span_started_at = performance_attribution.begin_operation_span();
        let attention_output = match (&decoder_layer_weights.attention_weights, layer_model_state) {
            (
                Qwen3_5MoEAttentionWeights::Linear(linear_attention_weights),
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
                        linear_attention_weights,
                        convolution,
                        recurrent,
                    )
                },
            ),
            (
                Qwen3_5MoEAttentionWeights::Full(full_attention_weights),
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
        let attention_output = attention_output?;
        let attention_residual = self.runtime.add(hidden_states, &attention_output)?;
        let normalized_attention = self.runtime.rms_norm(
            &attention_residual,
            &decoder_layer_weights.post_attention_normalization_weight,
            f32::from_bits(self.config.rms_norm_epsilon_bits()),
        )?;
        let should_use_compiled_elementwise_graphs = token_count != 1;
        let mlp_forward_span_started_at = performance_attribution.begin_operation_span();
        let mlp_output = match &decoder_layer_weights.mlp_weights {
            Qwen3_5DecoderLayerMlpWeights::Dense(dense_mlp_weights) => {
                self.forward_dense_mlp(&normalized_attention, dense_mlp_weights)
            }
            Qwen3_5DecoderLayerMlpWeights::Sparse(mixture_of_experts_weights) => {
                let expert_pager = self.expert_pager.as_ref().ok_or_else(|| {
                    Qwen3_5MoEExecutionError::MissingTensor {
                        tensor_name: "sparse model expert pager".to_owned(),
                    }
                })?;
                self.forward_moe_with_paging(
                    &normalized_attention,
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
        let mlp_output = mlp_output?;
        Ok(self.runtime.add(&attention_residual, &mlp_output)?)
    }

    pub(super) fn quantized_linear(
        &self,
        activations: &MlxArray,
        affine_weights: &Qwen3_5MoEAffineWeights,
    ) -> Result<MlxArray, Qwen3_5MoEExecutionError> {
        match affine_weights {
            Qwen3_5MoEAffineWeights::NativeBfloat16 { weight } => {
                let transposed_weight = self.runtime.transpose_axes(weight, &[1, 0])?;
                Ok(self.runtime.matmul(activations, &transposed_weight)?)
            }
            Qwen3_5MoEAffineWeights::Quantized {
                packed_weight,
                quantization_scales,
                quantization_biases,
                quantization_group_size,
                quantization_bits,
            } => Ok(self.runtime.quantized_matmul_affine(
                activations,
                packed_weight,
                quantization_scales,
                quantization_biases,
                true,
                *quantization_group_size,
                *quantization_bits,
            )?),
        }
    }

    pub(super) fn quantized_expert_linear(
        &self,
        activations: &MlxArray,
        affine_weights: &Qwen3_5MoEAffineWeights,
        selected_indices: &MlxArray,
        sorted_indices: bool,
    ) -> Result<MlxArray, Qwen3_5MoEExecutionError> {
        match affine_weights {
            Qwen3_5MoEAffineWeights::NativeBfloat16 { weight } => {
                let transposed_expert_weights = self.runtime.transpose_axes(weight, &[0, 2, 1])?;
                Ok(self.runtime.gather_dense_matmul(
                    activations,
                    &transposed_expert_weights,
                    None,
                    Some(selected_indices),
                    sorted_indices,
                )?)
            }
            Qwen3_5MoEAffineWeights::Quantized {
                packed_weight,
                quantization_scales,
                quantization_biases,
                quantization_group_size,
                quantization_bits,
            } => Ok(self.runtime.gather_quantized_matmul_affine(
                activations,
                packed_weight,
                quantization_scales,
                quantization_biases,
                None,
                Some(selected_indices),
                true,
                *quantization_group_size,
                *quantization_bits,
                sorted_indices,
            )?),
        }
    }
}
