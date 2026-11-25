//! Flash-Next's multi-token-prediction head on the streaming engine.
//!
//! The head drafts the tokens after the one the target just chose. Given a
//! token and the target's residual lanes at the position before it, it
//! returns logits for the token after it and its own lanes, which stand in
//! for the target's when it drafts one token further. It keeps its own
//! attention keys and values and its own indexer keys, in positions that
//! count the head's own rows.
//!
//! Its experts stay in memory whole, so a draft step waits for no storage
//! read and no route readback: the routed arithmetic is the streaming
//! engine's own, over one page that holds every expert.

use mlx_bridge::MlxArray;
use runtime_integration::MlxRuntimeError;
use text_generation::execution_error::TextExecutionError;

use jangtq2_qwen35::configuration::Jangtq2CodebookRotation;

use crate::qwen3_5_core::configuration::Qwen4ExpArchitecture;
use crate::qwen3_5_core::model_math::decoder_layer_weights::{
    Qwen3_5AttentionWeights, Qwen3_5DecoderFeedForwardWeights,
};
use crate::qwen3_5_streaming::expert_streaming::expert_pager::Qwen3_5PagedExpertWeights;
use crate::qwen3_5_streaming::model::Qwen3_5Model;
use crate::qwen4_exp::centered_norm::grouped_rms_norm;
use crate::qwen4_exp::mtp_head_weights::MtpHeadWeights;
use crate::qwen4_exp::request_state::IndexerLayerState;
use crate::{FullAttentionKeyValueState, PerformanceAttribution};

/// Every expert of the head's layer as one page, in expert order, so an
/// expert's id is its slot.
fn head_expert_page(head: &MtpHeadWeights) -> Result<Qwen3_5PagedExpertWeights, MlxRuntimeError> {
    Ok(Qwen3_5PagedExpertWeights {
        gate_projection: head.expert_gate_projection.retained_reference()?,
        up_projection: head.expert_up_projection.retained_reference()?,
        down_projection: head.expert_down_projection.retained_reference()?,
        gate_activation_rotation: Jangtq2CodebookRotation::NoRotation,
        up_activation_rotation: Jangtq2CodebookRotation::NoRotation,
        down_activation_rotation: Jangtq2CodebookRotation::NoRotation,
    })
}

/// One request's memory of the head: what its attention has seen.
pub struct MtpHeadState {
    key_value_state: FullAttentionKeyValueState,
    indexer_state: Option<IndexerLayerState>,
}

impl MtpHeadState {
    pub fn empty(full_attention_kv_state_growth_tokens: i32) -> Result<Self, MlxRuntimeError> {
        Ok(Self {
            key_value_state: FullAttentionKeyValueState::empty_with_growth_tokens(
                full_attention_kv_state_growth_tokens,
            )?,
            indexer_state: None,
        })
    }

    /// Rows the head has forwarded and kept.
    #[must_use]
    pub fn row_count(&self) -> i32 {
        self.key_value_state.offset_tokens()
    }

    /// Forgets every row after the first `kept_row_count`: the rows a draft
    /// added on the head's own guesses, before the target's lanes for the
    /// kept tokens replace them.
    pub fn truncate_rows(
        &mut self,
        kept_row_count: i32,
        tokens_per_indexer_block: i32,
    ) -> Result<(), MlxRuntimeError> {
        self.key_value_state.truncate_to_offset(kept_row_count)?;
        if let Some(indexer_state) = self.indexer_state.as_mut() {
            indexer_state.keys.truncate_rows(kept_row_count);
            if let Some(block_keys) = indexer_state.block_keys.as_mut() {
                block_keys.truncate_rows(kept_row_count / tokens_per_indexer_block.max(1));
            }
        }
        Ok(())
    }

    /// The arrays a forward leaves to evaluate with its outputs.
    pub fn append_evaluation_arrays<'state>(&'state self, arrays: &mut Vec<&'state MlxArray>) {
        arrays.extend(self.key_value_state.keys_state());
        arrays.extend(self.key_value_state.values_state());
        if let Some(indexer_state) = &self.indexer_state {
            arrays.push(indexer_state.keys.storage());
            if let Some(block_keys) = &indexer_state.block_keys {
                arrays.push(block_keys.storage());
            }
        }
    }
}

/// What one head forward returns, both lazy.
pub struct MtpHeadOutput {
    /// `[1, tokens, vocabulary]` float32 logits for the token after each
    /// input token.
    pub logits: MlxArray,
    /// `[1, tokens, lanes * hidden]`: the head's lanes before its final
    /// mixer, the stand-in for the target's lanes one draft further.
    pub lanes: MlxArray,
}

impl Qwen3_5Model {
    /// The checkpoint's head, when it ships one.
    #[must_use]
    pub fn mtp_head(&self) -> Option<&MtpHeadWeights> {
        self.optional_accelerators.mtp_head.as_deref()
    }

    /// Sets `reserve_bytes` aside for a drafter that is loading and plans
    /// the kept weights around it, giving back what no longer fits; zero
    /// ends the reservation.
    pub fn reserve_room_for_a_loading_drafter(&self, reserve_bytes: u64) {
        self.governed_memory
            .drafter_load_reserve_bytes
            .set(reserve_bytes);
        self.open_memory_governor();
    }

    /// The model's geometry as the memory ledger should see it: while
    /// replies draft, the head's weights are core weights, always in memory,
    /// and the `astro_expert_cache` gets what is left after them.
    #[must_use]
    pub fn geometry_with_drafting_head(
        &self,
        mut model_geometry: crate::ModelMemoryGeometry,
    ) -> crate::ModelMemoryGeometry {
        if self.chunking.multi_token_prediction_draft_tokens > 0
            && let Some(head) = self.mtp_head()
        {
            model_geometry.model_core_payload_bytes = model_geometry
                .model_core_payload_bytes
                .saturating_add(head.payload_byte_count);
        }
        // A hosted DFlash drafter never pages either; one that is loading
        // is counted by the room set aside for it.
        let drafter_bytes = match self.optional_accelerators.draft_runtime.as_deref() {
            Some(draft_runtime) => draft_runtime.hosted_weight_bytes,
            None => self.governed_memory.drafter_load_reserve_bytes.get(),
        };
        model_geometry.model_core_payload_bytes = model_geometry
            .model_core_payload_bytes
            .saturating_add(drafter_bytes);
        model_geometry
    }

    /// Runs the head over `token_indices` (`[1, tokens]` int32), each paired
    /// with the lanes at the position before it (`preceding_lanes`,
    /// `[1, tokens, lanes * hidden]`).
    pub fn forward_mtp_head(
        &self,
        head: &MtpHeadWeights,
        token_indices: &MlxArray,
        preceding_lanes: &MlxArray,
        head_state: &mut MtpHeadState,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MtpHeadOutput, TextExecutionError> {
        let architecture =
            self.config
                .qwen4_exp_architecture()
                .ok_or(TextExecutionError::InvalidInput {
                    description: "only a Flash-Next model has a multi-token-prediction head",
                })?;
        let token_count = token_indices.shape().get(1).copied().unwrap_or(0);
        if token_count == 0 || preceding_lanes.shape().get(1).copied() != Some(token_count) {
            return Err(TextExecutionError::InvalidInput {
                description: "the head needs one row of lanes for each of its tokens",
            });
        }
        let fused_lanes =
            self.fuse_mtp_head_inputs(head, token_indices, preceding_lanes, &architecture)?;
        let lanes = self.forward_mtp_head_layer(
            head,
            &fused_lanes,
            token_count,
            head_state,
            &architecture,
            performance_attribution,
        )?;
        let (mixed, _) = self.read_lanes(
            &lanes,
            &head.final_norm_scale,
            &head.final_mixer,
            &architecture,
        )?;
        let logits = self.quantized_linear(&mixed, &self.weights.language_model_head_weights)?;
        Ok(MtpHeadOutput {
            logits: self
                .runtime
                .astype(&logits, mlx_bridge::MlxDtype::Float32)?,
            lanes,
        })
    }

    /// The head's first lanes: the projected embedding of each token added
    /// to every projected lane. The lanes are normalized together, as one
    /// vector, and projected one by one.
    fn fuse_mtp_head_inputs(
        &self,
        head: &MtpHeadWeights,
        token_indices: &MlxArray,
        preceding_lanes: &MlxArray,
        architecture: &Qwen4ExpArchitecture,
    ) -> Result<MlxArray, TextExecutionError> {
        let runtime = &self.runtime;
        let hidden_size = self.config.hidden_size() as i32;
        let lane_count = architecture.residual_lane_count as i32;
        let epsilon = f32::from_bits(self.config.rms_norm_epsilon_bits());
        let token_count = token_indices.shape()[1];
        let embeddings = self.embedding_lookup(token_indices)?;
        let projected_embeddings = self.quantized_linear(
            &grouped_rms_norm(
                runtime,
                &embeddings,
                &head.embedding_norm_scale,
                hidden_size,
                epsilon,
            )?,
            &head.embedding_projection,
        )?;
        let normalized_lanes = grouped_rms_norm(
            runtime,
            preceding_lanes,
            &head.lane_norm_scale,
            hidden_size * lane_count,
            epsilon,
        )?;
        let projected_lanes = self.quantized_linear(
            &runtime.reshape(
                &normalized_lanes,
                &[1, token_count, lane_count, hidden_size],
            )?,
            &head.lane_projection,
        )?;
        let fused = runtime.add(
            &runtime.expand_dims(&projected_embeddings, -2)?,
            &runtime.astype(&projected_lanes, projected_embeddings.dtype())?,
        )?;
        Ok(runtime.reshape(&fused, &[1, token_count, lane_count * hidden_size])?)
    }

    /// The head's decoder layer: attention, then routed experts, each read
    /// from and written back to the lanes.
    fn forward_mtp_head_layer(
        &self,
        head: &MtpHeadWeights,
        lanes: &MlxArray,
        token_count: i32,
        head_state: &mut MtpHeadState,
        architecture: &Qwen4ExpArchitecture,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, TextExecutionError> {
        let layer = &head.layer;
        let (
            Some(flash_next),
            Qwen3_5AttentionWeights::Full(attention_weights),
            Qwen3_5DecoderFeedForwardWeights::MixtureOfExperts(expert_weights),
        ) = (
            layer.flash_next(),
            &layer.attention_weights,
            &layer.mlp_weights,
        )
        else {
            return Err(TextExecutionError::InvalidInput {
                description: "the head's layer is a Flash-Next full-attention layer with experts",
            });
        };
        let indexer = flash_next.sparse_attention_indexer.as_ref().ok_or(
            TextExecutionError::MissingTensor {
                tensor_name: "mtp.layers.0.self_attn.indexer.index_qk_proj".to_owned(),
            },
        )?;
        let attention_read = self.read_lanes_for_block(
            lanes,
            &layer.input_normalization_weight,
            &flash_next.lane_connections.attention,
            architecture,
        )?;
        let rope_offset_tokens = head_state.row_count();
        let attention_output = self.forward_sparse_full_attention(
            &attention_read.block_input,
            token_count,
            rope_offset_tokens,
            attention_weights,
            indexer,
            &mut head_state.key_value_state,
            &mut head_state.indexer_state,
            architecture,
        )?;
        let lanes = self.write_back_to_lanes(&attention_output, &attention_read)?;
        let expert_read = self.read_lanes_for_block(
            &lanes,
            &layer.post_attention_normalization_weight,
            &flash_next.lane_connections.feed_forward,
            architecture,
        )?;
        let (selected_expert_ids, selected_scores) =
            self.routed_expert_selection(&expert_read.block_input, expert_weights)?;
        let routed_output = self.forward_moe_with_streamed_weights(
            &expert_read.block_input,
            &head_expert_page(head)?,
            &selected_expert_ids,
            &selected_scores,
            performance_attribution,
        )?;
        let expert_output = self.combine_paged_sparse_and_shared_outputs(
            &expert_read.block_input,
            expert_weights,
            &routed_output,
            false,
        )?;
        self.write_back_to_lanes(&expert_output, &expert_read)
    }
}
