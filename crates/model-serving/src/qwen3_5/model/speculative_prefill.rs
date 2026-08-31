use std::collections::HashMap;

use crate::qwen3_5::decoder::RequestDecoderStateStackAllocationCheckpoint;
use astronomical_runtime_integration::{MlxArray, MlxDtype};

use crate::qwen3_5_moe::Qwen3_5MoEPagedPrefillExecutionMode;
use crate::{PerformanceAttribution, PerformanceOperation, Qwen3_5TargetForwardOutput};

use super::speculative_prefill_draft_forward::qwen3_5_speculative_prefill_draft_forward_end;
use super::{
    Qwen3_5AttentionCapture, Qwen3_5ExecutionError, Qwen3_5Model, RequestDecoderStateStack,
    qwen3_5_aggregate_speculative_prefill_attention_weights,
};

/// Draft scoring output with a reusable checkpoint taken before lookahead.
pub(crate) struct Qwen3_5SpeculativePrefillDraftScoringOutcome {
    pub(crate) importance_scores: MlxArray,
    pub(crate) draft_prompt_prefix_allocation_checkpoint:
        RequestDecoderStateStackAllocationCheckpoint,
    pub(crate) draft_prompt_prefix_payload_bytes: u64,
}

/// One dense draft decoder-state block captured at a complete prompt boundary.
pub(crate) struct Qwen3_5SpeculativePrefillDraftPersistentPromptCacheBlock {
    pub(crate) block_start_tokens: usize,
    pub(crate) block_end_tokens: usize,
    pub(crate) kv_block_tensors: HashMap<String, MlxArray>,
    pub(crate) recurrent_snapshot_tensors: HashMap<String, MlxArray>,
}

pub(crate) type Qwen3_5SpeculativePrefillDraftPersistentPromptCacheBlockConsumer<'a> =
    dyn FnMut(
            Qwen3_5SpeculativePrefillDraftPersistentPromptCacheBlock,
            &mut PerformanceAttribution,
        ) -> Result<(), Qwen3_5ExecutionError>
        + 'a;

impl Qwen3_5Model {
    /// Scores prompt-token importance with a local draft-model decoder state.
    ///
    /// The prompt forward retains full-attention keys. Bounded highest-logit lookahead
    /// then retains post-RoPE queries, and the resulting attention distributions
    /// are reduced to one float32 score per prompt position. The decoder state is
    /// deliberately local because only the scores are needed after this method.
    pub(crate) fn score_speculative_prefill_importance_with_performance_attribution(
        &self,
        draft_suffix_token_ids: &[u32],
        starting_position_tokens: u32,
        prompt_key_start_token_index: usize,
        scored_prompt_token_count: usize,
        lookahead_token_count: usize,
        importance_pooling_kernel_token_count: usize,
        persistent_prompt_cache_block_token_count: usize,
        persistent_prompt_cache_block_consumer: Option<
            &mut Qwen3_5SpeculativePrefillDraftPersistentPromptCacheBlockConsumer<'_>,
        >,
        mut request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Qwen3_5SpeculativePrefillDraftScoringOutcome, Qwen3_5ExecutionError> {
        self.score_speculative_prefill_importance_with_optional_visual_embeddings_and_performance_attribution(
            draft_suffix_token_ids,
            starting_position_tokens,
            prompt_key_start_token_index,
            scored_prompt_token_count,
            lookahead_token_count,
            importance_pooling_kernel_token_count,
            persistent_prompt_cache_block_token_count,
            persistent_prompt_cache_block_consumer,
            None,
            &mut request_decoder_state,
            performance_attribution,
        )
    }

    /// Scores a prompt whose image-pad rows use draft-owned visual embeddings.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn score_speculative_prefill_importance_with_visual_embeddings_and_performance_attribution(
        &self,
        draft_suffix_token_ids: &[u32],
        starting_position_tokens: u32,
        prompt_key_start_token_index: usize,
        scored_prompt_token_count: usize,
        lookahead_token_count: usize,
        importance_pooling_kernel_token_count: usize,
        persistent_prompt_cache_block_token_count: usize,
        persistent_prompt_cache_block_consumer: Option<
            &mut Qwen3_5SpeculativePrefillDraftPersistentPromptCacheBlockConsumer<'_>,
        >,
        visual_embeddings: &MlxArray,
        image_pad_token_id: u32,
        mut request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Qwen3_5SpeculativePrefillDraftScoringOutcome, Qwen3_5ExecutionError> {
        self.score_speculative_prefill_importance_with_optional_visual_embeddings_and_performance_attribution(
            draft_suffix_token_ids,
            starting_position_tokens,
            prompt_key_start_token_index,
            scored_prompt_token_count,
            lookahead_token_count,
            importance_pooling_kernel_token_count,
            persistent_prompt_cache_block_token_count,
            persistent_prompt_cache_block_consumer,
            Some((visual_embeddings, image_pad_token_id)),
            &mut request_decoder_state,
            performance_attribution,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn score_speculative_prefill_importance_with_optional_visual_embeddings_and_performance_attribution(
        &self,
        draft_suffix_token_ids: &[u32],
        starting_position_tokens: u32,
        prompt_key_start_token_index: usize,
        scored_prompt_token_count: usize,
        lookahead_token_count: usize,
        importance_pooling_kernel_token_count: usize,
        persistent_prompt_cache_block_token_count: usize,
        mut persistent_prompt_cache_block_consumer: Option<
            &mut Qwen3_5SpeculativePrefillDraftPersistentPromptCacheBlockConsumer<'_>,
        >,
        visual_embedding_input: Option<(&MlxArray, u32)>,
        mut request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Qwen3_5SpeculativePrefillDraftScoringOutcome, Qwen3_5ExecutionError> {
        if draft_suffix_token_ids.is_empty() {
            return Err(Qwen3_5ExecutionError::InvalidInput {
                description: "speculative-prefill scoring requires prompt tokens",
            });
        }
        if scored_prompt_token_count < draft_suffix_token_ids.len()
            || prompt_key_start_token_index
                .checked_add(scored_prompt_token_count)
                .is_none()
        {
            return Err(Qwen3_5ExecutionError::InvalidInput {
                description: "speculative-prefill scoring prompt range is invalid",
            });
        }
        if lookahead_token_count == 0 {
            return Err(Qwen3_5ExecutionError::InvalidInput {
                description: "speculative-prefill lookahead token count must be positive",
            });
        }

        let mut attention_capture =
            Qwen3_5AttentionCapture::new(self.config.layer_count() as usize);
        let mut draft_forward_position_tokens = starting_position_tokens;
        let mut final_draft_logits = None;
        let mut consumed_visual_embedding_count = 0_usize;

        let mut draft_suffix_start_index = 0_usize;
        while draft_suffix_start_index < draft_suffix_token_ids.len() {
            let draft_forward_start_token_count = usize::try_from(draft_forward_position_tokens)
                .map_err(|_| Qwen3_5ExecutionError::InvalidInput {
                    description: "speculative-prefill draft position exceeds usize",
                })?;
            let remaining_draft_suffix_token_count =
                draft_suffix_token_ids.len() - draft_suffix_start_index;
            let capture_cache_block_token_count = persistent_prompt_cache_block_consumer
                .as_ref()
                .map(|_| persistent_prompt_cache_block_token_count);
            let draft_forward_end_token_count = qwen3_5_speculative_prefill_draft_forward_end(
                draft_forward_start_token_count,
                remaining_draft_suffix_token_count,
                self.chunking.speculative_prefill_draft_forward_tokens,
                capture_cache_block_token_count,
            )
            .ok_or(Qwen3_5ExecutionError::InvalidInput {
                description: "speculative-prefill draft forward range is invalid",
            })?;
            let draft_forward_token_count = draft_forward_end_token_count
                .checked_sub(draft_forward_start_token_count)
                .ok_or(Qwen3_5ExecutionError::InvalidInput {
                    description: "speculative-prefill draft forward range moved backwards",
                })?;
            let draft_suffix_end_index = draft_suffix_start_index
                .checked_add(draft_forward_token_count)
                .ok_or(Qwen3_5ExecutionError::InvalidInput {
                    description: "speculative-prefill draft suffix range overflowed",
                })?;
            let draft_prompt_token_ids = draft_suffix_token_ids
                .get(draft_suffix_start_index..draft_suffix_end_index)
                .ok_or(Qwen3_5ExecutionError::InvalidInput {
                    description: "speculative-prefill draft suffix range exceeds the prompt",
                })?;
            let draft_forward_output = if let Some((visual_embeddings, image_pad_token_id)) =
                visual_embedding_input
            {
                let (draft_forward_output, consumed_visual_embeddings_in_chunk) = self
                    .forward_visual_chunk_with_speculative_prefill_attention_capture_and_performance_attribution(
                        draft_prompt_token_ids,
                        draft_forward_position_tokens,
                        visual_embeddings,
                        consumed_visual_embedding_count,
                        image_pad_token_id,
                        &mut request_decoder_state,
                        &mut attention_capture,
                        performance_attribution,
                    )?;
                consumed_visual_embedding_count = consumed_visual_embedding_count
                    .checked_add(consumed_visual_embeddings_in_chunk)
                    .ok_or(Qwen3_5ExecutionError::InvalidInput {
                        description: "speculative-prefill draft visual embedding cursor overflowed",
                    })?;
                draft_forward_output
            } else {
                self.forward_chunk_with_speculative_prefill_attention_capture_and_performance_attribution(
                    draft_prompt_token_ids,
                    draft_forward_position_tokens,
                    &mut request_decoder_state,
                    &mut attention_capture,
                    performance_attribution,
                )?
            };
            final_draft_logits = Some(draft_forward_output.final_logits().retain()?);
            draft_forward_position_tokens =
                u32::try_from(draft_forward_end_token_count).map_err(|_| {
                    Qwen3_5ExecutionError::InvalidInput {
                        description: "speculative-prefill draft prompt exceeds the u32 range",
                    }
                })?;
            draft_suffix_start_index = draft_suffix_end_index;
            if persistent_prompt_cache_block_consumer.is_some()
                && usize::try_from(draft_forward_position_tokens).is_ok_and(
                    |completed_prompt_token_count| {
                        completed_prompt_token_count
                            .is_multiple_of(persistent_prompt_cache_block_token_count)
                    },
                )
            {
                let block_end_tokens =
                    usize::try_from(draft_forward_position_tokens).map_err(|_| {
                        Qwen3_5ExecutionError::InvalidInput {
                            description: "speculative-prefill draft block end exceeds usize",
                        }
                    })?;
                let block_start_tokens = block_end_tokens
                    .checked_sub(persistent_prompt_cache_block_token_count)
                    .ok_or(Qwen3_5ExecutionError::InvalidInput {
                        description: "speculative-prefill draft block start underflowed",
                    })?;
                let persistent_prompt_cache_block =
                    Qwen3_5SpeculativePrefillDraftPersistentPromptCacheBlock {
                        block_start_tokens,
                        block_end_tokens,
                        kv_block_tensors: performance_attribution.measure_operation(
                            PerformanceOperation::PersistentPromptCacheStateExtraction,
                            |_performance_attribution| {
                                request_decoder_state
                                    .extract_persistent_prompt_cache_kv_block_tensors(
                                        &self.runtime,
                                        block_start_tokens,
                                        block_end_tokens,
                                        persistent_prompt_cache_block_token_count,
                                    )
                            },
                        )?,
                        recurrent_snapshot_tensors: performance_attribution.measure_operation(
                            PerformanceOperation::PersistentPromptCacheStateExtraction,
                            |_performance_attribution| {
                                request_decoder_state
                                    .extract_persistent_prompt_cache_recurrent_snapshot_tensors()
                            },
                        )?,
                    };
                if let Some(persistent_prompt_cache_block_consumer) =
                    persistent_prompt_cache_block_consumer.as_deref_mut()
                {
                    persistent_prompt_cache_block_consumer(
                        persistent_prompt_cache_block,
                        performance_attribution,
                    )?;
                }
            }
        }

        if let Some((visual_embeddings, _image_pad_token_id)) = visual_embedding_input
            && consumed_visual_embedding_count != visual_embeddings.shape()[0] as usize
        {
            return Err(Qwen3_5ExecutionError::InvalidInput {
                description: "draft visual embeddings do not exactly match image-pad tokens",
            });
        }

        let draft_prompt_prefix_allocation_checkpoint =
            request_decoder_state.allocation_checkpoint()?;
        let draft_prompt_prefix_payload_bytes = request_decoder_state.payload_byte_count();
        attention_capture.begin_lookahead_capture();
        let mut draft_logits = final_draft_logits.ok_or(Qwen3_5ExecutionError::InvalidInput {
            description: "speculative-prefill draft prompt produced no logits",
        })?;
        for _lookahead_step in 0..lookahead_token_count {
            let lookahead_token_id = self.highest_logit_token_id(&draft_logits)?;
            let lookahead_forward_output = self
                .forward_chunk_with_speculative_prefill_attention_capture_and_performance_attribution(
                    &[lookahead_token_id],
                    draft_forward_position_tokens,
                    &mut request_decoder_state,
                    &mut attention_capture,
                    performance_attribution,
                )?;
            draft_logits = lookahead_forward_output.final_logits().retain()?;
            draft_forward_position_tokens = draft_forward_position_tokens.checked_add(1).ok_or(
                Qwen3_5ExecutionError::InvalidInput {
                    description: "speculative-prefill lookahead position counter overflowed",
                },
            )?;
        }

        let importance_scores = self.compute_speculative_prefill_importance_scores(
            &attention_capture,
            prompt_key_start_token_index,
            scored_prompt_token_count,
            lookahead_token_count,
            importance_pooling_kernel_token_count,
        )?;
        Ok(Qwen3_5SpeculativePrefillDraftScoringOutcome {
            importance_scores,
            draft_prompt_prefix_allocation_checkpoint,
            draft_prompt_prefix_payload_bytes,
        })
    }

    fn compute_speculative_prefill_importance_scores(
        &self,
        attention_capture: &Qwen3_5AttentionCapture,
        prompt_key_start_token_index: usize,
        prompt_token_count: usize,
        lookahead_token_count: usize,
        importance_pooling_kernel_token_count: usize,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let prompt_token_count_i32 =
            i32::try_from(prompt_token_count).map_err(|_| Qwen3_5ExecutionError::InvalidInput {
                description: "speculative-prefill prompt token count exceeds the MLX range",
            })?;
        let prompt_key_start_token_index_i32 = i32::try_from(prompt_key_start_token_index)
            .map_err(|_| Qwen3_5ExecutionError::InvalidInput {
                description: "speculative-prefill prompt key start exceeds the MLX range",
            })?;
        let prompt_key_end_token_index_i32 = prompt_key_start_token_index_i32
            .checked_add(prompt_token_count_i32)
            .ok_or(Qwen3_5ExecutionError::InvalidInput {
                description: "speculative-prefill prompt key end exceeds the MLX range",
            })?;
        let lookahead_token_count_i32 = i32::try_from(lookahead_token_count).map_err(|_| {
            Qwen3_5ExecutionError::InvalidInput {
                description: "speculative-prefill lookahead token count exceeds the MLX range",
            }
        })?;
        let mut per_layer_scores = Vec::new();

        for decoder_layer_index in 0..self.config.layer_count() as usize {
            let Some(prompt_keys) = attention_capture.prompt_keys_for_layer(decoder_layer_index)
            else {
                continue;
            };
            let Some(lookahead_queries) =
                attention_capture.lookahead_queries_for_layer(decoder_layer_index)
            else {
                continue;
            };
            if lookahead_queries.is_empty() {
                continue;
            }
            let prompt_key_shape = prompt_keys.shape();
            let first_query_shape = lookahead_queries[0].shape();
            if prompt_key_shape.len() != 4
                || first_query_shape.len() != 4
                || prompt_key_shape[0] != 1
                || first_query_shape[0] != 1
                || prompt_key_shape[2] < prompt_key_end_token_index_i32
                || first_query_shape[2] != 1
                || prompt_key_shape[3] != first_query_shape[3]
            {
                return Err(Qwen3_5ExecutionError::InvalidInput {
                    description: "captured speculative-prefill attention tensors have invalid shapes",
                });
            }
            let key_value_head_count = prompt_key_shape[1];
            let query_head_count = first_query_shape[1];
            if key_value_head_count <= 0
                || query_head_count <= 0
                || query_head_count % key_value_head_count != 0
            {
                return Err(Qwen3_5ExecutionError::InvalidInput {
                    description: "captured speculative-prefill attention heads are incompatible",
                });
            }
            let prompt_keys = self.runtime.slice(
                prompt_keys,
                &[0, 0, prompt_key_start_token_index_i32, 0],
                &[
                    1,
                    key_value_head_count,
                    prompt_key_end_token_index_i32,
                    prompt_key_shape[3],
                ],
                &[1, 1, 1, 1],
            )?;
            let lookahead_query_references = lookahead_queries.iter().collect::<Vec<_>>();
            let lookahead_queries = self
                .runtime
                .concatenate_axis(&lookahead_query_references, 2)?;
            if lookahead_queries.shape()[2] != lookahead_token_count_i32 {
                return Err(Qwen3_5ExecutionError::InvalidInput {
                    description: "captured speculative-prefill lookahead count does not match configuration",
                });
            }
            let expanded_prompt_keys = self.runtime.repeat_axis(
                &prompt_keys,
                query_head_count / key_value_head_count,
                1,
            )?;
            let transposed_prompt_keys = self
                .runtime
                .transpose_axes(&expanded_prompt_keys, &[0, 1, 3, 2])?;
            let attention_scores = self
                .runtime
                .matmul(&lookahead_queries, &transposed_prompt_keys)?;
            let attention_scores = self.runtime.multiply_scalar(
                &attention_scores,
                (first_query_shape[3] as f32).sqrt().recip(),
            )?;
            let attention_scores = self.runtime.astype(&attention_scores, MlxDtype::Float32)?;
            let attention_weights = self.runtime.softmax_axis(&attention_scores, -1)?;
            per_layer_scores.push(self.runtime.reshape(
                &attention_weights,
                &[
                    query_head_count,
                    lookahead_token_count_i32,
                    prompt_token_count_i32,
                ],
            )?);
        }

        if per_layer_scores.is_empty() {
            return Err(Qwen3_5ExecutionError::InvalidInput {
                description: "draft model produced no full-attention tensors for speculative prefill scoring",
            });
        }
        let per_layer_score_references = per_layer_scores.iter().collect::<Vec<_>>();
        let combined_head_scores = self
            .runtime
            .concatenate_axis(&per_layer_score_references, 0)?;
        qwen3_5_aggregate_speculative_prefill_attention_weights(
            &self.runtime,
            &combined_head_scores,
            importance_pooling_kernel_token_count,
        )
    }

    /// Runs one draft or target forward while retaining optional full-attention capture tensors.
    pub(crate) fn forward_chunk_with_speculative_prefill_attention_capture_and_performance_attribution(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        attention_capture: &mut Qwen3_5AttentionCapture,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Qwen3_5TargetForwardOutput, Qwen3_5ExecutionError> {
        let target_forward_output = self.build_target_forward_graph_with_attention_capture(
            token_ids,
            starting_position_tokens,
            None,
            request_decoder_state,
            Some(attention_capture),
            None,
            Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault,
            performance_attribution,
        )?;
        self.evaluate_forward_state(target_forward_output.final_logits(), request_decoder_state)?;
        Ok(target_forward_output)
    }
}
