//! Both ways of prefilling a prompt that holds image-pad tokens, for the
//! test that holds them bit-identical (issue #2221).
//!
//! A request with images reads its prompt one layer at a time like any
//! other. The image embeddings replace the image-pad tokens' embeddings
//! when the spans are embedded, and a pad run may be split across two
//! spans. These entry points run the same spans as chunks and one layer at
//! a time with the same stand-in image embeddings and return what each
//! would publish.

use mlx_bridge::MlxArray;

use super::Qwen3_5Model;
use super::layer_major_checkpoint_parity::{
    PublishedBoundarySnapshot, planned_spans, push_final_state_boundary,
};
use super::layer_major_prefill::{
    LayerMajorCheckpointContract, LayerMajorPrefillRequest, LayerMajorSpan, LayerMajorVisualInputs,
};
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::{PerformanceAttribution, RequestDecoderStateStack};

/// What both paths published for one prompt with images.
pub struct ImagePromptBoundaries {
    pub chunk_path: Vec<PublishedBoundarySnapshot>,
    pub layer_at_a_time: Vec<PublishedBoundarySnapshot>,
    /// Image embedding rows each path consumed, chunk path first.
    pub consumed_image_embedding_counts: (usize, usize),
}

impl Qwen3_5Model {
    /// Prefills `token_ids` as chunks and one layer at a time, with image
    /// embeddings in place of every token equal to `image_pad_token_id`.
    ///
    /// The stand-in image embeddings are the text embeddings of the first
    /// vocabulary entries: the right width and dtype, and different from
    /// what the pad token itself embeds to, so a row put in the wrong place
    /// changes the published state.
    #[doc(hidden)]
    pub fn prefill_image_prompt_both_ways(
        &self,
        token_ids: &[u32],
        chunk_token_count: usize,
        block_token_count: usize,
        image_pad_token_id: u32,
        chunk_path_decoder_state: &mut RequestDecoderStateStack,
        layer_at_a_time_decoder_state: &mut RequestDecoderStateStack,
    ) -> Result<ImagePromptBoundaries, Qwen3_5ExecutionError> {
        let (group_token_ids, spans) =
            planned_spans(token_ids, chunk_token_count, block_token_count)?;
        let image_pad_token_count = group_token_ids
            .iter()
            .filter(|token_id| **token_id == image_pad_token_id)
            .count();
        let image_embeddings = self.stand_in_image_embeddings(image_pad_token_count)?;
        let image_inputs = LayerMajorVisualInputs {
            visual_embeddings: &image_embeddings,
            consumed_visual_embedding_count: 0,
            image_pad_token_id,
        };
        let (chunk_path, chunk_path_consumed_count) = self.prefill_image_prompt_as_chunks(
            group_token_ids,
            &spans,
            block_token_count,
            image_inputs,
            chunk_path_decoder_state,
        )?;
        let (layer_at_a_time, layer_at_a_time_consumed_count) = self
            .prefill_image_prompt_one_layer_at_a_time(
                group_token_ids,
                &spans,
                block_token_count,
                image_inputs,
                layer_at_a_time_decoder_state,
            )?;
        Ok(ImagePromptBoundaries {
            chunk_path,
            layer_at_a_time,
            consumed_image_embedding_counts: (
                chunk_path_consumed_count,
                layer_at_a_time_consumed_count,
            ),
        })
    }

    /// `row_count` rows of the embedding table, as a `[rows, width]` array.
    fn stand_in_image_embeddings(
        &self,
        row_count: usize,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let signed_row_count =
            i32::try_from(row_count).map_err(|_| Qwen3_5ExecutionError::InvalidInput {
                description: "the image-pad token count exceeds the MLX int32 range",
            })?;
        let stand_in_token_ids: Vec<i32> = (0..signed_row_count).collect();
        let token_indices = self
            .runtime
            .array_from_i32(&stand_in_token_ids, &[1, signed_row_count])?;
        let stand_in_embeddings = self.embedding_lookup(&token_indices)?;
        let embedding_width = self.config.hidden_size() as i32;
        Ok(self
            .runtime
            .reshape(&stand_in_embeddings, &[signed_row_count, embedding_width])?)
    }

    /// The planned spans one chunk forward each, the way the engine read
    /// image prompts before they went one layer at a time.
    fn prefill_image_prompt_as_chunks(
        &self,
        group_token_ids: &[u32],
        spans: &[LayerMajorSpan],
        block_token_count: usize,
        image_inputs: LayerMajorVisualInputs<'_>,
        request_decoder_state: &mut RequestDecoderStateStack,
    ) -> Result<(Vec<PublishedBoundarySnapshot>, usize), Qwen3_5ExecutionError> {
        let mut performance_attribution = PerformanceAttribution::disabled();
        let mut consumed_image_embedding_count = 0_usize;
        let mut boundary_checkpoints = Vec::new();
        for span in spans {
            let starting_position_tokens = u32::try_from(span.start_token).map_err(|_| {
                Qwen3_5ExecutionError::InvalidInput {
                    description: "a planned span starts beyond the u32 range",
                }
            })?;
            let span_token_ids = &group_token_ids[span.start_token..span.end_token];
            if span.interior_boundary_tokens.is_empty() {
                consumed_image_embedding_count += self
                    .prefill_chunk_with_visual_embeddings_and_performance_attribution(
                        span_token_ids,
                        starting_position_tokens,
                        image_inputs.visual_embeddings,
                        consumed_image_embedding_count,
                        request_decoder_state,
                        image_inputs.image_pad_token_id,
                        &mut performance_attribution,
                    )?;
            } else {
                let checkpoint_outcome = self
                    .prefill_chunk_with_visual_embeddings_and_boundary_checkpoints_with_performance_attribution(
                        span_token_ids,
                        starting_position_tokens,
                        image_inputs.visual_embeddings,
                        consumed_image_embedding_count,
                        request_decoder_state,
                        image_inputs.image_pad_token_id,
                        span.interior_boundary_tokens.clone(),
                        block_token_count,
                        &mut performance_attribution,
                    )?;
                consumed_image_embedding_count +=
                    checkpoint_outcome.consumed_visual_embedding_count;
                for mut boundary_checkpoint in checkpoint_outcome.boundary_checkpoints {
                    boundary_checkpoint.completed_prefill_chunk_tokens += span.start_token;
                    boundary_checkpoints.push(boundary_checkpoint);
                }
            }
            push_final_state_boundary(
                &mut boundary_checkpoints,
                span.end_token,
                block_token_count,
                request_decoder_state,
            )?;
        }
        Ok((
            self.host_snapshots(boundary_checkpoints)?,
            consumed_image_embedding_count,
        ))
    }

    /// The same spans one layer at a time, as the engine reads them now.
    fn prefill_image_prompt_one_layer_at_a_time(
        &self,
        group_token_ids: &[u32],
        spans: &[LayerMajorSpan],
        block_token_count: usize,
        image_inputs: LayerMajorVisualInputs<'_>,
        request_decoder_state: &mut RequestDecoderStateStack,
    ) -> Result<(Vec<PublishedBoundarySnapshot>, usize), Qwen3_5ExecutionError> {
        let layer_major_outcome = self.prefill_layer_major(
            group_token_ids,
            0,
            LayerMajorPrefillRequest {
                spans,
                checkpoint_contract: Some(LayerMajorCheckpointContract {
                    boundary_tensor_count: self.decoder_cache_layout.boundary_tensor_count(),
                    block_token_count,
                }),
                should_materialize_logits: false,
                visual_inputs: Some(image_inputs),
            },
            request_decoder_state,
            &mut PerformanceAttribution::disabled(),
        )?;
        let mut boundary_checkpoints = layer_major_outcome.boundary_checkpoints;
        push_final_state_boundary(
            &mut boundary_checkpoints,
            group_token_ids.len(),
            block_token_count,
            request_decoder_state,
        )?;
        Ok((
            self.host_snapshots(boundary_checkpoints)?,
            layer_major_outcome.consumed_visual_embedding_count,
        ))
    }
}
