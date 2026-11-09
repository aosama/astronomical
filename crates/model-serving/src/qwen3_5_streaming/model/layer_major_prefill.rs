//! Prefill that runs each dense decoder layer over every span of a prompt
//! before the next layer (issue #1985).
//!
//! Chunk by chunk, a dense model that streams its layers reads every
//! streamed layer once per chunk; with the prompt cache's 1,792-token blocks
//! a 10,000-token prompt read each one six times. Here the spans are the
//! ones the chunk-by-chunk engine would run, with the same prompt-cache
//! checkpoints, and only the loop order changes: one layer runs over every
//! span, then the next is read. Within a layer each span still sees the
//! previous span's state, and across layers each span sees its own input, so
//! every kernel runs on the same tokens with the same inputs and the result
//! is bit-identical to running the spans as chunks.

use std::collections::HashMap;

use mlx_c_rust::MlxArray;

use super::Qwen3_5Model;
use super::layer_submission::{CompletedLayerRoots, LayerGroupSubmission};
use crate::qwen3_5_core::decoder::{
    Qwen3_5PersistentPromptCacheBoundaryCheckpoint,
    Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector,
};
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_streaming::Qwen3_5MoEPagedExecutionMode;
use crate::{DecoderCacheState, PerformanceAttribution, RequestDecoderStateStack};

/// One span of a layer-major prefill: the tokens one chunk would have run.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct LayerMajorSpan {
    /// Start, relative to the first span.
    pub(crate) start_token: usize,
    /// Exclusive end, relative to the first span.
    pub(crate) end_token: usize,
    /// Prompt-cache boundaries strictly inside the span, relative to its
    /// start, captured the way a chunk captures them.
    pub(crate) interior_boundary_tokens: Vec<usize>,
    /// Whether the span ends on a prompt-cache boundary short of the last
    /// span; a chunk publishes that boundary from its final state.
    pub(crate) ends_at_inner_boundary: bool,
}

impl LayerMajorSpan {
    const fn token_count(&self) -> usize {
        self.end_token - self.start_token
    }
}

/// Where a layer-major prefill's prompt-cache checkpoints come from.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LayerMajorCheckpointContract {
    pub(crate) boundary_tensor_count: usize,
    pub(crate) block_token_count: usize,
}

/// The spans one layer-major prefill runs and what it must leave behind.
#[derive(Clone, Copy)]
pub(crate) struct LayerMajorPrefillRequest<'spans> {
    pub(crate) spans: &'spans [LayerMajorSpan],
    /// The prompt-cache contract when this request captures checkpoints.
    pub(crate) checkpoint_contract: Option<LayerMajorCheckpointContract>,
    /// Whether the last span ends the prompt, so its logits seed the first
    /// generated token.
    pub(crate) should_materialize_logits: bool,
}

/// What a layer-major prefill leaves for the engine.
pub(crate) struct LayerMajorPrefillOutcome {
    /// Float32 logits of the last token when the caller asked for them.
    pub(crate) final_logits: Option<MlxArray>,
    /// Checkpoints at boundaries before the last span's end, in order, with
    /// positions relative to the first span.
    pub(crate) boundary_checkpoints: Vec<Qwen3_5PersistentPromptCacheBoundaryCheckpoint>,
}

/// The per-span bookkeeping that lives across the layer loop.
struct SpanProgress {
    hidden_states: Option<MlxArray>,
    collector: Option<Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector>,
    end_state_snapshot: Option<HashMap<String, MlxArray>>,
}

/// Bytes per element of the widest floating activation MLX produces, used
/// only if the model's own hidden-state dtype cannot be read.
const WIDEST_ACTIVATION_ELEMENT_BYTES: usize = 4;

impl Qwen3_5Model {
    /// Whether a forward reads any decoder layer from the SSD: a dense model
    /// with fewer layers kept than it has.
    pub(crate) fn streams_dense_layers(&self) -> bool {
        self.dense_streaming_slot_reservation_bytes() > 0
    }

    /// Bytes one token's hidden state occupies between decoder layers, read
    /// from the shape and dtype of a one-token array, which MLX sizes
    /// without allocating it.
    pub(crate) fn hidden_state_bytes_per_token(&self) -> usize {
        let hidden_size = self.config.hidden_size() as usize;
        super::decoder_cache_dtype_flow::embedded_hidden_state_dtype(&self.weights)
            .ok()
            .and_then(|hidden_state_dtype| {
                self.runtime
                    .zeros(&[1, 1, hidden_size as i32], hidden_state_dtype)
                    .ok()
            })
            .map_or(
                hidden_size.saturating_mul(WIDEST_ACTIVATION_ELEMENT_BYTES),
                |one_token_hidden_state| one_token_hidden_state.byte_count(),
            )
    }

    /// Runs `token_ids` from `starting_position_tokens` through every decoder
    /// layer, one layer over all `spans` at a time.
    pub(crate) fn prefill_layer_major(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        layer_major_prefill_request: LayerMajorPrefillRequest<'_>,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<LayerMajorPrefillOutcome, Qwen3_5ExecutionError> {
        let LayerMajorPrefillRequest {
            spans,
            checkpoint_contract,
            should_materialize_logits,
        } = layer_major_prefill_request;
        validate_layer_major_spans(token_ids.len(), spans)?;
        let mut span_progress =
            self.embed_layer_major_spans(token_ids, spans, checkpoint_contract)?;
        let mut previous_dense_layer_output = None;
        for layer_index in 0..self.config.layer_count() as usize {
            self.run_layer_over_spans(
                layer_index,
                starting_position_tokens,
                spans,
                &mut span_progress,
                request_decoder_state,
                &mut previous_dense_layer_output,
                performance_attribution,
            )?;
        }
        let final_logits = match (should_materialize_logits, span_progress.last()) {
            (
                true,
                Some(SpanProgress {
                    hidden_states: Some(hidden_states),
                    ..
                }),
            ) => {
                let last_span_token_count = spans.last().map_or(0, LayerMajorSpan::token_count);
                Some(self.final_vocabulary_logits(
                    hidden_states,
                    i32::try_from(last_span_token_count).unwrap_or(i32::MAX),
                )?)
            }
            _ => None,
        };
        let mut evaluation_roots =
            super::forward_contract::decoder_state_arrays(request_decoder_state)?;
        evaluation_roots.extend(final_logits.as_ref());
        self.runtime.evaluate_arrays(&evaluation_roots)?;
        Ok(LayerMajorPrefillOutcome {
            final_logits,
            boundary_checkpoints: completed_checkpoints(spans, span_progress)?,
        })
    }

    /// Runs a whole prompt layer by layer over spans of `span_token_count`
    /// tokens, without prompt-cache checkpoints, and returns the last token's
    /// Float32 logits: the prefill the engine runs, for parity and speed
    /// measurements.
    #[doc(hidden)]
    pub fn prefill_prompt_layer_major(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        span_token_count: usize,
        request_decoder_state: &mut RequestDecoderStateStack,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let spans = even_layer_major_spans(token_ids.len(), span_token_count);
        let layer_major_outcome = self.prefill_layer_major(
            token_ids,
            starting_position_tokens,
            LayerMajorPrefillRequest {
                spans: &spans,
                checkpoint_contract: None,
                should_materialize_logits: true,
            },
            request_decoder_state,
            &mut PerformanceAttribution::disabled(),
        )?;
        layer_major_outcome
            .final_logits
            .ok_or(Qwen3_5ExecutionError::InvalidInput {
                description: "a layer-major prefill of a prompt produced no logits",
            })
    }

    /// Looks up each span's embeddings and opens its checkpoint collector.
    fn embed_layer_major_spans(
        &self,
        token_ids: &[u32],
        spans: &[LayerMajorSpan],
        checkpoint_contract: Option<LayerMajorCheckpointContract>,
    ) -> Result<Vec<SpanProgress>, Qwen3_5ExecutionError> {
        let mut span_progress = Vec::with_capacity(spans.len());
        for span in spans {
            let span_token_ids = &token_ids[span.start_token..span.end_token];
            let signed_token_ids = span_token_ids
                .iter()
                .map(|token_id| {
                    i32::try_from(*token_id).map_err(|_| Qwen3_5ExecutionError::InvalidInput {
                        description: "token ID exceeds the MLX int32 range",
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let span_token_count = i32::try_from(span.token_count()).map_err(|_| {
                Qwen3_5ExecutionError::InvalidInput {
                    description: "a layer-major span exceeds the MLX int32 range",
                }
            })?;
            let token_indices = self
                .runtime
                .array_from_i32(&signed_token_ids, &[1, span_token_count])?;
            span_progress.push(SpanProgress {
                hidden_states: Some(self.embedding_lookup(&token_indices)?),
                collector: span_collector(span, checkpoint_contract)?,
                end_state_snapshot: (span.ends_at_inner_boundary && checkpoint_contract.is_some())
                    .then(HashMap::new),
            });
        }
        Ok(span_progress)
    }

    /// Runs one decoder layer over every span in order, reading a streamed
    /// layer once for all of them.
    #[allow(clippy::too_many_arguments)]
    fn run_layer_over_spans(
        &self,
        layer_index: usize,
        starting_position_tokens: u32,
        spans: &[LayerMajorSpan],
        span_progress: &mut [SpanProgress],
        request_decoder_state: &mut RequestDecoderStateStack,
        previous_dense_layer_output: &mut Option<MlxArray>,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), Qwen3_5ExecutionError> {
        let forward_layer_weights = self.forward_layer_weights(layer_index)?;
        for (span, progress) in spans.iter().zip(span_progress.iter_mut()) {
            let span_input =
                progress
                    .hidden_states
                    .take()
                    .ok_or(Qwen3_5ExecutionError::InvalidInput {
                        description: "a layer-major span lost its hidden states",
                    })?;
            let span_token_count = i32::try_from(span.token_count()).unwrap_or(i32::MAX);
            let rope_offset_tokens =
                i32::try_from(u64::from(starting_position_tokens) + span.start_token as u64)
                    .map_err(|_| Qwen3_5ExecutionError::InvalidInput {
                        description: "a layer-major span starts beyond the MLX int32 range",
                    })?;
            let layer_model_state = request_decoder_state.layer_mut(layer_index).ok_or(
                Qwen3_5ExecutionError::InvalidRequestDecoderState {
                    layer_index,
                    description: "decoder layer entry is missing",
                },
            )?;
            let span_output = self.forward_decoder_layer(
                &span_input,
                span_token_count,
                rope_offset_tokens,
                layer_index,
                forward_layer_weights.decoder_layer_weights(),
                layer_model_state,
                None,
                progress.collector.as_mut(),
                Qwen3_5MoEPagedExecutionMode::ProductionDefault,
                performance_attribution,
            )?;
            drop(span_input);
            let layer_state = request_decoder_state.layer(layer_index);
            self.submit_completed_layer(
                &forward_layer_weights,
                CompletedLayerRoots {
                    layer_output: &span_output,
                    convolution_state: layer_state
                        .and_then(DecoderCacheState::convolution_state_array),
                    boundary_checkpoint_states: progress
                        .collector
                        .as_ref()
                        .map(|collector| collector.layer_evaluation_arrays(layer_index))
                        .unwrap_or_default(),
                },
                previous_dense_layer_output,
                LayerGroupSubmission {
                    completed_layer_count: layer_index + 1,
                    decoder_layer_count: self.config.layer_count() as usize,
                    graph_submission_layer_interval: 0,
                    forward_token_count: span_token_count,
                },
                performance_attribution,
            )?;
            if let (Some(end_state_snapshot), Some(layer_state)) =
                (progress.end_state_snapshot.as_mut(), layer_state)
            {
                record_end_state_snapshot(end_state_snapshot, layer_index, layer_state)?;
            }
            progress.hidden_states = Some(span_output);
        }
        Ok(())
    }
}

/// A span's checkpoint collector, when it has boundaries strictly inside it.
fn span_collector(
    span: &LayerMajorSpan,
    checkpoint_contract: Option<LayerMajorCheckpointContract>,
) -> Result<Option<Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector>, Qwen3_5ExecutionError>
{
    let Some(checkpoint_contract) = checkpoint_contract else {
        return Ok(None);
    };
    if span.interior_boundary_tokens.is_empty() {
        return Ok(None);
    }
    Ok(Some(
        Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector::new(
            span.interior_boundary_tokens.clone(),
            checkpoint_contract.boundary_tensor_count,
            checkpoint_contract.block_token_count,
        )?,
    ))
}

/// Keeps one linear-attention layer's state as a span leaves it: the same
/// arrays a chunk ending there publishes from its final state.
fn record_end_state_snapshot(
    end_state_snapshot: &mut HashMap<String, MlxArray>,
    layer_index: usize,
    layer_state: &DecoderCacheState,
) -> Result<(), Qwen3_5ExecutionError> {
    let DecoderCacheState::Composite {
        convolution,
        recurrent,
    } = layer_state
    else {
        return Ok(());
    };
    let (Some(convolution_state), Some(recurrent_state)) = (convolution.state(), recurrent.state())
    else {
        return Err(Qwen3_5ExecutionError::InvalidRequestDecoderState {
            layer_index,
            description: "a linear-attention layer left a span without its state",
        });
    };
    end_state_snapshot.insert(
        format!("layer_{layer_index}_linear.convolution"),
        convolution_state.retain()?,
    );
    end_state_snapshot.insert(
        format!("layer_{layer_index}_linear.gated_delta_recurrent"),
        recurrent_state.retain()?,
    );
    Ok(())
}

/// Every span's checkpoints, moved to positions relative to the first span.
fn completed_checkpoints(
    spans: &[LayerMajorSpan],
    span_progress: Vec<SpanProgress>,
) -> Result<Vec<Qwen3_5PersistentPromptCacheBoundaryCheckpoint>, Qwen3_5ExecutionError> {
    let mut boundary_checkpoints = Vec::new();
    for (span, progress) in spans.iter().zip(span_progress) {
        if let Some(collector) = progress.collector {
            for mut boundary_checkpoint in collector.complete()? {
                boundary_checkpoint.completed_prefill_chunk_tokens += span.start_token;
                boundary_checkpoints.push(boundary_checkpoint);
            }
        }
        if let Some(recurrent_snapshot_tensors) = progress.end_state_snapshot {
            boundary_checkpoints.push(Qwen3_5PersistentPromptCacheBoundaryCheckpoint {
                completed_prefill_chunk_tokens: span.end_token,
                recurrent_snapshot_tensors,
            });
        }
    }
    Ok(boundary_checkpoints)
}

/// Spans of `span_token_count` tokens, the last one shorter.
fn even_layer_major_spans(token_count: usize, span_token_count: usize) -> Vec<LayerMajorSpan> {
    let span_token_count = span_token_count.max(1);
    let mut spans = Vec::with_capacity(token_count.div_ceil(span_token_count));
    let mut start_token = 0;
    while start_token < token_count {
        let end_token = (start_token + span_token_count).min(token_count);
        spans.push(LayerMajorSpan {
            start_token,
            end_token,
            interior_boundary_tokens: Vec::new(),
            ends_at_inner_boundary: false,
        });
        start_token = end_token;
    }
    spans
}

/// Spans must tile the tokens from zero, in order, with no empty span.
fn validate_layer_major_spans(
    token_count: usize,
    spans: &[LayerMajorSpan],
) -> Result<(), Qwen3_5ExecutionError> {
    let mut expected_start_token = 0;
    for span in spans {
        if span.start_token != expected_start_token || span.end_token <= span.start_token {
            return Err(Qwen3_5ExecutionError::InvalidInput {
                description: "layer-major spans must tile the prompt in order",
            });
        }
        expected_start_token = span.end_token;
    }
    if expected_start_token != token_count || spans.is_empty() {
        return Err(Qwen3_5ExecutionError::InvalidInput {
            description: "layer-major spans must cover every prompt token",
        });
    }
    Ok(())
}

#[cfg(test)]
#[path = "layer_major_prefill_tests.rs"]
mod tests;
