//! Embedding a prompt's spans before its first layer runs, with a request's
//! image embeddings put in place of image-pad tokens (issues #1985 and
//! #2221).

use mlx_bridge::MlxArray;

use super::Qwen3_5Model;
use super::layer_major_prefill::{
    LayerMajorCheckpointContract, LayerMajorSpan, LayerMajorVisualInputs,
};
use super::layer_major_span_state::{SpanState, span_collector};
use std::collections::HashMap;
use text_generation::execution_error::TextExecutionError;

impl Qwen3_5Model {
    /// Looks up each span's embeddings, with image embeddings put in place
    /// of image-pad tokens, and opens its checkpoint collector. Returns the
    /// span states and how many image embedding rows the spans consumed.
    pub(super) fn embed_layer_major_spans(
        &self,
        token_ids: &[u32],
        spans: &[LayerMajorSpan],
        checkpoint_contract: Option<LayerMajorCheckpointContract>,
        visual_inputs: Option<LayerMajorVisualInputs<'_>>,
    ) -> Result<(Vec<SpanState>, usize), TextExecutionError> {
        let mut consumed_visual_embedding_count = 0_usize;
        let mut span_states = Vec::with_capacity(spans.len());
        for span in spans {
            let span_token_ids = &token_ids[span.start_token..span.end_token];
            let signed_token_ids = span_token_ids
                .iter()
                .map(|token_id| {
                    i32::try_from(*token_id).map_err(|_| TextExecutionError::InvalidInput {
                        description: "token ID exceeds the MLX int32 range",
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let span_token_count = i32::try_from(span.token_count()).map_err(|_| {
                TextExecutionError::InvalidInput {
                    description: "a layer-major span exceeds the MLX int32 range",
                }
            })?;
            let token_indices = self
                .runtime
                .array_from_i32(&signed_token_ids, &[1, span_token_count])?;
            span_states.push(SpanState {
                hidden_states: Some(self.embed_span(
                    &token_indices,
                    span_token_ids,
                    visual_inputs,
                    &mut consumed_visual_embedding_count,
                )?),
                collector: span_collector(span, checkpoint_contract)?,
                end_state_snapshot: (span.ends_at_inner_boundary && checkpoint_contract.is_some())
                    .then(HashMap::new),
            });
        }
        Ok((span_states, consumed_visual_embedding_count))
    }

    /// One span's embeddings. A span that holds image-pad tokens has the
    /// request's image embeddings put in their place, continuing from the
    /// rows earlier spans and earlier forwards consumed; a pad run split
    /// across two spans stays in order.
    fn embed_span(
        &self,
        token_indices: &MlxArray,
        span_token_ids: &[u32],
        visual_inputs: Option<LayerMajorVisualInputs<'_>>,
        consumed_visual_embedding_count: &mut usize,
    ) -> Result<MlxArray, TextExecutionError> {
        let text_embeddings = self.embedding_lookup(token_indices)?;
        let Some(visual_inputs) = visual_inputs else {
            return Ok(text_embeddings);
        };
        if !span_token_ids.contains(&visual_inputs.image_pad_token_id) {
            return Ok(text_embeddings);
        }
        let (embeddings_with_images, span_visual_embedding_count) =
            super::visual_embedding_injection::qwen3_5_inject_visual_embeddings(
                &self.runtime,
                &text_embeddings,
                span_token_ids,
                visual_inputs.visual_embeddings,
                visual_inputs.consumed_visual_embedding_count + *consumed_visual_embedding_count,
                visual_inputs.image_pad_token_id,
            )?;
        *consumed_visual_embedding_count += span_visual_embedding_count;
        Ok(embeddings_with_images)
    }
}
