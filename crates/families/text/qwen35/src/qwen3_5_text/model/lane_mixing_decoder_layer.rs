//! One Flash-Next decoder layer on the streaming engine.
//!
//! The layer carries several residual lanes. The n-gram layer adds its
//! table rows first; then each block (attention, then the routed experts)
//! reads the lanes through its hyper-connection and writes its output back
//! to every lane. The attention and expert computations themselves are the
//! Qwen 3.5 ones, so experts stream through the same pager and
//! `astro_expert_cache` as any other model.

use mlx_bridge::MlxArray;
use text_generation::execution_error::{TextExecutionError, invalid_request_decoder_state};

use crate::qwen3_5_core::configuration::Qwen4ExpArchitecture;
use crate::qwen3_5_core::model_math::decoder_layer_weights::{
    Qwen3_5AttentionWeights, Qwen3_5DecoderFeedForwardWeights, Qwen3_5DecoderLayerWeights,
};
use crate::qwen3_5_text::Qwen3_5MoEPagedExecutionMode;
use crate::qwen3_5_text::model::streaming_model::Qwen3_5TextModel;
use crate::qwen4_exp::layer_weights::FlashNextLayerWeights;
use crate::qwen4_exp::request_state::Qwen4ExpRequestState;
use crate::{AstroDecoderStateCacheState, PerformanceAttribution};
use text_generation::AstroSsdPromptCacheBoundaryCheckpointCollector;

/// Where one layer forward runs and what it reads beyond its lanes.
pub struct LaneMixingLayerInputs<'forward> {
    pub token_count: i32,
    pub rope_offset_tokens: i32,
    pub layer_index: usize,
    pub paged_execution_mode: Qwen3_5MoEPagedExecutionMode,
    pub request_state: Option<&'forward mut Qwen4ExpRequestState>,
    /// Captures a linear-attention layer's state at the row boundaries it
    /// names, so a drafted block can be rewound to its kept rows.
    pub boundary_checkpoint_collector:
        Option<&'forward mut AstroSsdPromptCacheBoundaryCheckpointCollector>,
}

impl Qwen3_5TextModel {
    /// `lanes` after one Flash-Next decoder layer.
    pub fn forward_flash_next_decoder_layer(
        &self,
        lanes: &MlxArray,
        decoder_layer_weights: &Qwen3_5DecoderLayerWeights,
        flash_next: &FlashNextLayerWeights,
        layer_model_state: &mut AstroDecoderStateCacheState,
        mut inputs: LaneMixingLayerInputs<'_>,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, TextExecutionError> {
        let architecture =
            self.config
                .qwen4_exp_architecture()
                .ok_or(TextExecutionError::InvalidInput {
                    description: "a lane-mixing layer needs the Flash-Next architecture",
                })?;
        let request_state =
            inputs
                .request_state
                .take()
                .ok_or(TextExecutionError::InvalidInput {
                    description: "a Flash-Next layer needs the request's Flash-Next state",
                })?;
        let lane_connections = &flash_next.lane_connections;
        let lanes = match &flash_next.ngram_layer {
            Some(ngram_layer) => {
                self.add_ngram_contribution(lanes, ngram_layer, request_state, &architecture)?
            }
            None => lanes.retain()?,
        };
        let attention_read = self.read_lanes_for_block(
            &lanes,
            &decoder_layer_weights.input_normalization_weight,
            &lane_connections.attention,
            &architecture,
        )?;
        let attention_output = self.lane_mixing_attention(
            &attention_read.block_input,
            decoder_layer_weights,
            flash_next,
            layer_model_state,
            &mut inputs,
            request_state,
            &architecture,
            performance_attribution,
        )?;
        let lanes = self.write_back_to_lanes(&attention_output, &attention_read)?;
        let feed_forward_read = self.read_lanes_for_block(
            &lanes,
            &decoder_layer_weights.post_attention_normalization_weight,
            &lane_connections.feed_forward,
            &architecture,
        )?;
        let Qwen3_5DecoderFeedForwardWeights::MixtureOfExperts(expert_weights) =
            &decoder_layer_weights.mlp_weights
        else {
            return Err(TextExecutionError::InvalidInput {
                description: "every Flash-Next layer routes to experts",
            });
        };
        let expert_pager =
            self.expert_pager
                .as_ref()
                .ok_or_else(|| TextExecutionError::MissingTensor {
                    tensor_name: "sparse model expert pager".to_owned(),
                })?;
        let feed_forward_output = self.forward_qwen3_5_moe(
            &feed_forward_read.block_input,
            expert_weights,
            expert_pager,
            inputs.layer_index,
            inputs.token_count != 1,
            inputs.paged_execution_mode,
            performance_attribution,
        )?;
        self.write_back_to_lanes(&feed_forward_output, &feed_forward_read)
    }

    /// The layer's attention over the block input: Gated DeltaNet, or full
    /// attention with the sparse-attention indexer.
    #[allow(clippy::too_many_arguments)]
    fn lane_mixing_attention(
        &self,
        block_input: &MlxArray,
        decoder_layer_weights: &Qwen3_5DecoderLayerWeights,
        flash_next: &FlashNextLayerWeights,
        layer_model_state: &mut AstroDecoderStateCacheState,
        inputs: &mut LaneMixingLayerInputs<'_>,
        request_state: &mut Qwen4ExpRequestState,
        architecture: &Qwen4ExpArchitecture,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, TextExecutionError> {
        match (&decoder_layer_weights.attention_weights, layer_model_state) {
            (
                Qwen3_5AttentionWeights::Linear(linear_attention_weights),
                AstroDecoderStateCacheState::Composite {
                    convolution,
                    recurrent,
                },
            ) => self.forward_linear_attention(
                block_input,
                inputs.token_count,
                inputs.layer_index,
                linear_attention_weights,
                convolution,
                recurrent,
                inputs.boundary_checkpoint_collector.as_deref_mut(),
                performance_attribution,
            ),
            (
                Qwen3_5AttentionWeights::Full(full_attention_weights),
                AstroDecoderStateCacheState::AppendOnlyAttention { attention },
            ) => {
                let indexer_state = request_state
                    .indexer_by_layer
                    .get_mut(inputs.layer_index)
                    .ok_or(invalid_request_decoder_state(
                        inputs.layer_index,
                        "the Flash-Next request state has no slot for this layer",
                    ))?;
                let indexer = flash_next.sparse_attention_indexer.as_ref().ok_or(
                    TextExecutionError::MissingTensor {
                        tensor_name: "self_attn.indexer.index_qk_proj".to_owned(),
                    },
                )?;
                self.forward_sparse_full_attention(
                    block_input,
                    inputs.token_count,
                    inputs.rope_offset_tokens,
                    full_attention_weights,
                    indexer,
                    attention,
                    indexer_state,
                    architecture,
                )
            }
            _ => Err(invalid_request_decoder_state(
                inputs.layer_index,
                "decoder state attention family does not match the bound layer weights",
            )),
        }
    }
}
