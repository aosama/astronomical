//! Direct Qwen3.5 text execution for the pinned model artifact.

use astronomical_runtime_integration::{
    MlxArray, MlxCompiledElementwiseGraphs, MlxCompiledSwiGlu, MlxMetalKernel, MlxRuntime,
};

use std::cell::RefCell;

use crate::expert_paging::{ExpertWeightMemoryCache, ExpertWeightMemoryCacheStatistics};
use crate::qwen3_5_moe::{
    Qwen3_5ExpertPager, Qwen3_5MoEPagedPrefillExecutionMode, Qwen3_5PagedExpertWeights,
};
use crate::{DecoderCacheLayout, DecoderCacheState, PerformanceAttribution, PerformanceOperation};

use super::decoder_layer_weights::{
    Qwen3_5AffineWeights, Qwen3_5AttentionWeights, Qwen3_5DecoderFeedForwardWeights,
    Qwen3_5DecoderLayerWeights,
};
use super::error::invalid_request_decoder_state;
use super::forward_contract::validate_forward_input;
use super::model_chunking_configuration::Qwen3_5ModelChunkingConfiguration;
use super::{
    Qwen3_5AttentionCapture, Qwen3_5Config, Qwen3_5ExecutionError, Qwen3_5VisionModel,
    Qwen3_5Weights, RequestDecoderStateStack,
};
use crate::qwen3_5::decoder::Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector;
use crate::qwen3_5::multi_token_prediction::Qwen3_5MtpWeights;

/// One resident native Qwen3.5 text model, optional vision tower, and its direct MLX runtime.
#[derive(Debug)]
pub struct Qwen3_5Model {
    pub(crate) runtime: MlxRuntime,
    pub(crate) config: Qwen3_5Config,
    pub(crate) decoder_cache_layout: DecoderCacheLayout,
    pub(crate) weights: Qwen3_5Weights,
    pub(crate) mtp_weights: Option<Qwen3_5MtpWeights>,
    pub(crate) vision_model: Option<Qwen3_5VisionModel>,
    /// Sparse models own a pager; dense models have no sparse-expert weights.
    pub(crate) expert_pager: Option<Qwen3_5ExpertPager>,
    pub(crate) expert_weight_memory_cache:
        Option<RefCell<ExpertWeightMemoryCache<Qwen3_5PagedExpertWeights>>>,
    pub(crate) gated_delta_kernel: MlxMetalKernel,
    pub(crate) gated_delta_checkpoint_kernel: MlxMetalKernel,
    pub(crate) sorted_expert_weighted_sum_kernel: Option<MlxMetalKernel>,
    pub(crate) target_verification_quantized_linear_kernel: MlxMetalKernel,
    pub(crate) compiled_swiglu: MlxCompiledSwiGlu,
    pub(crate) compiled_elementwise_graphs: MlxCompiledElementwiseGraphs,
    pub(crate) chunking: Qwen3_5ModelChunkingConfiguration,
    /// Model-owned BF16 scalar for the query normalization scale in every
    /// linear-attention forward pass.
    pub(crate) inverse_linear_head_dimension_scale: MlxArray,
    /// Model-owned BF16 scalar for the key normalization scale in every
    /// linear-attention forward pass.
    pub(crate) inverse_square_root_linear_head_dimension_scale: MlxArray,
}

impl Qwen3_5Model {
    /// Returns the MLX runtime used by this model.
    #[must_use]
    pub fn runtime(&self) -> &MlxRuntime {
        &self.runtime
    }

    /// Returns cumulative expert memory-cache counters for low-level performance tests.
    #[must_use]
    pub fn expert_weight_memory_cache_statistics(&self) -> ExpertWeightMemoryCacheStatistics {
        self.expert_weight_memory_cache.as_ref().map_or_else(
            ExpertWeightMemoryCacheStatistics::default,
            |expert_weight_memory_cache| expert_weight_memory_cache.borrow().statistics(),
        )
    }

    pub(crate) fn sparse_expert_weight_memory_cache(
        &self,
    ) -> Result<&RefCell<ExpertWeightMemoryCache<Qwen3_5PagedExpertWeights>>, Qwen3_5ExecutionError>
    {
        self.expert_weight_memory_cache
            .as_ref()
            .ok_or(Qwen3_5ExecutionError::InvalidInput {
                description: "sparse Qwen3.5 execution requires an expert weight memory cache",
            })
    }

    pub(crate) fn sorted_expert_weighted_sum_kernel(
        &self,
    ) -> Result<&MlxMetalKernel, Qwen3_5ExecutionError> {
        self.sorted_expert_weighted_sum_kernel
            .as_ref()
            .ok_or(Qwen3_5ExecutionError::InvalidInput {
                description: "sparse Qwen3.5 execution requires a sorted expert output kernel",
            })
    }

    /// Returns the validated text configuration bound to this loaded model.
    #[must_use]
    pub(crate) const fn config(&self) -> &Qwen3_5Config {
        &self.config
    }

    /// Returns the validated decoder-cache layout bound to this model artifact.
    #[must_use]
    pub(crate) const fn decoder_cache_layout(&self) -> &DecoderCacheLayout {
        &self.decoder_cache_layout
    }

    /// Returns the optional vision tower loaded beside the language model.
    #[must_use]
    pub fn vision_model(&self) -> Option<&Qwen3_5VisionModel> {
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
    ) -> Result<usize, Qwen3_5ExecutionError> {
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
    ) -> Result<(), Qwen3_5ExecutionError> {
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
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
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
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
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
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let token_count = validate_forward_input(
            token_ids,
            starting_position_tokens,
            None,
            request_decoder_state.layer_count(),
            self.config.layer_count() as usize,
            self.config.vocabulary_size(),
            self.config.maximum_position_count(),
        )?;
        let signed_token_ids = token_ids
            .iter()
            .map(|token_id| {
                i32::try_from(*token_id).map_err(|_| Qwen3_5ExecutionError::InvalidInput {
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

    pub(crate) fn materialize_target_weights(&self) -> Result<(), Qwen3_5ExecutionError> {
        self.weights.materialize(&self.runtime)?;
        self.runtime.evaluate_arrays(&[
            &self.inverse_linear_head_dimension_scale,
            &self.inverse_square_root_linear_head_dimension_scale,
        ])?;
        Ok(())
    }

    pub(crate) fn embedding_lookup(
        &self,
        token_indices: &MlxArray,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        match &self.weights.embedding_weights {
            Qwen3_5AffineWeights::NativeBfloat16 { weight } => {
                Ok(self.runtime.take_axis(weight, token_indices, 0)?)
            }
            Qwen3_5AffineWeights::Quantized {
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
    pub(crate) fn forward_decoder_layer(
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
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let normalized_input = self.runtime.rms_norm(
            hidden_states,
            &decoder_layer_weights.input_normalization_weight,
            f32::from_bits(self.config.rms_norm_epsilon_bits()),
        )?;
        let attention_forward_span_started_at = performance_attribution.begin_operation_span();
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
        let attention_output = attention_output?;
        let attention_residual = self.runtime.add(hidden_states, &attention_output)?;
        let normalized_attention = self.runtime.rms_norm(
            &attention_residual,
            &decoder_layer_weights.post_attention_normalization_weight,
            f32::from_bits(self.config.rms_norm_epsilon_bits()),
        )?;
        let should_use_compiled_elementwise_graphs = token_count != 1
            && paged_prefill_execution_mode
                != Qwen3_5MoEPagedPrefillExecutionMode::TargetVerificationWindow;
        let mlp_forward_span_started_at = performance_attribution.begin_operation_span();
        let mlp_output = match &decoder_layer_weights.mlp_weights {
            Qwen3_5DecoderFeedForwardWeights::Dense(dense_mlp_weights) => self
                .forward_qwen3_5_dense_mlp(
                    &normalized_attention,
                    dense_mlp_weights,
                    paged_prefill_execution_mode,
                ),
            Qwen3_5DecoderFeedForwardWeights::MixtureOfExperts(mixture_of_experts_weights) => {
                let expert_pager = self.expert_pager.as_ref().ok_or_else(|| {
                    Qwen3_5ExecutionError::MissingTensor {
                        tensor_name: "sparse model expert pager".to_owned(),
                    }
                })?;
                self.forward_qwen3_5_moe_with_paging(
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

    pub(crate) fn quantized_linear(
        &self,
        activations: &MlxArray,
        affine_weights: &Qwen3_5AffineWeights,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        match affine_weights {
            Qwen3_5AffineWeights::NativeBfloat16 { weight } => {
                let transposed_weight = self.runtime.transpose_axes(weight, &[1, 0])?;
                Ok(self.runtime.matmul(activations, &transposed_weight)?)
            }
            Qwen3_5AffineWeights::Quantized {
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

    pub(crate) fn quantized_expert_linear(
        &self,
        activations: &MlxArray,
        affine_weights: &Qwen3_5AffineWeights,
        selected_indices: &MlxArray,
        sorted_indices: bool,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        match affine_weights {
            Qwen3_5AffineWeights::NativeBfloat16 { weight } => {
                let transposed_expert_weights = self.runtime.transpose_axes(weight, &[0, 2, 1])?;
                Ok(self.runtime.gather_dense_matmul(
                    activations,
                    &transposed_expert_weights,
                    None,
                    Some(selected_indices),
                    sorted_indices,
                )?)
            }
            Qwen3_5AffineWeights::Quantized {
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
