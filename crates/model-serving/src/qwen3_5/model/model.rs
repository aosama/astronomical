//! Direct Qwen3.5 text execution for the pinned model artifact.

use astronomical_runtime_integration::{
    MlxArray, MlxCompiledElementwiseGraphs, MlxCompiledSwiGlu, MlxMetalKernel, MlxRuntime,
};

use std::cell::RefCell;

use crate::expert_paging::{ExpertWeightMemoryCacheStatistics, RetainedExpertLayerCache};
use crate::qwen3_5_moe::{
    PagedForwardMissingRouteCollector, Qwen3_5ExpertPager, Qwen3_5MoEPagedPrefillExecutionMode,
    Qwen3_5ResidentExpertWeights, Qwen3_5RetainedExpertLayer,
};
use crate::{DecoderCacheLayout, DecoderCacheState, MlxRamBudget, PerformanceAttribution};

use super::decoder_layer_weights::{Qwen3_5AffineWeights, Qwen3_5DecoderLayerWeights};
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
    /// Complete contiguous expert arrays when the whole sparse payload fits.
    pub(crate) resident_expert_weights: Option<Qwen3_5ResidentExpertWeights>,
    /// Complete Rust-loaded layers retained within the paged-mode RAM ceiling.
    pub(crate) retained_expert_layers:
        Option<RefCell<RetainedExpertLayerCache<Qwen3_5RetainedExpertLayer>>>,
    /// Single-source MLX RAM split for context, activations, streaming, and experts.
    pub(crate) mlx_ram_budget: RefCell<MlxRamBudget>,
    /// True after request pressure forced the complete expert owner out.
    /// Finalization consumes this one-shot flag and stays paged instead of
    /// immediately reading the same complete payload back into memory.
    pub(crate) should_defer_next_request_finalization_resident_promotion: bool,
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
    /// Deferred GPU missing-route roots collected during one paged forward.
    pub(crate) paged_forward_missing_route_collector: PagedForwardMissingRouteCollector,
}

impl Qwen3_5Model {
    /// Returns the MLX runtime used by this model.
    #[must_use]
    pub fn runtime(&self) -> &MlxRuntime {
        &self.runtime
    }

    /// Returns the single-source MLX RAM budget owner.
    #[must_use]
    pub fn mlx_ram_budget(&self) -> std::cell::Ref<'_, MlxRamBudget> {
        self.mlx_ram_budget.borrow()
    }

    /// Returns the mutable single-source MLX RAM budget owner.
    pub fn mlx_ram_budget_mut(&self) -> std::cell::RefMut<'_, MlxRamBudget> {
        self.mlx_ram_budget.borrow_mut()
    }

    /// Returns one mode-neutral expert-memory snapshot.
    ///
    /// Resident mode reports complete-owner entries and payload while retaining
    /// native cumulative page counters at their prior values. Paged mode reports
    /// the native cache directly. This lets telemetry change ownership without
    /// resetting process-lifetime paging evidence.
    #[must_use]
    pub fn expert_weight_memory_cache_statistics(&self) -> ExpertWeightMemoryCacheStatistics {
        if self.expert_pager.is_none() {
            return ExpertWeightMemoryCacheStatistics::default();
        }
        if let Some(resident_expert_weights) = self.resident_expert_weights.as_ref() {
            return ExpertWeightMemoryCacheStatistics {
                entry_count: resident_expert_weights.expert_entry_count(),
                resident_payload_byte_count: resident_expert_weights.payload_byte_count(),
                maximum_resident_payload_byte_count: resident_expert_weights.payload_byte_count(),
                eviction_count: 0,
                disk_page_load_count: 0,
                disk_batch_load_count: 0,
            };
        }
        self.retained_expert_layers.as_ref().map_or_else(
            ExpertWeightMemoryCacheStatistics::default,
            |retained_expert_layers| retained_expert_layers.borrow().statistics(),
        )
    }

    /// Updates the expert pager's memory budget with the observed transient
    /// high-water mark from completed forward-pass evidence. The retention
    /// ceiling uses this reservation to ensure enough headroom for computation
    /// buffers (attention KV growth, intermediate tensors) between expert-page
    /// loads.
    pub(crate) fn update_expert_pager_transient_high_water_bytes(
        &self,
        observed_transient_high_water_bytes: u64,
    ) {
        if let Some(expert_pager) = self.expert_pager.as_ref() {
            expert_pager
                .update_observed_transient_high_water_bytes(observed_transient_high_water_bytes);
        }
    }

    /// Retains the admission handoff while Rust-streamed pages remain operation-local.
    pub(crate) fn set_expert_pager_admitted_forward_reserve_bytes(
        &self,
        admitted_forward_reserve_bytes: u64,
    ) {
        if let Some(expert_pager) = self.expert_pager.as_ref() {
            expert_pager.set_pending_admitted_forward_reserve_bytes(admitted_forward_reserve_bytes);
        }
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
        let attention_output = self.forward_decoder_layer_attention(
            hidden_states,
            token_count,
            rope_offset_tokens,
            layer_index,
            decoder_layer_weights,
            layer_model_state,
            token_position_offsets,
            attention_capture,
            boundary_checkpoint_collector,
            paged_prefill_execution_mode,
            performance_attribution,
        )?;
        self.forward_decoder_layer_feed_forward(
            &attention_output,
            token_count,
            layer_index,
            decoder_layer_weights,
            paged_prefill_execution_mode,
            performance_attribution,
        )
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
}
