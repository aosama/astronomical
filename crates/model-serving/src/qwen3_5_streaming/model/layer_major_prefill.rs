//! Reading a prompt one layer at a time: each decoder layer runs over
//! every span of the prompt before the next layer (issues #1985 and #2221).
//!
//! A layer read from storage is then read once for the whole prompt, not
//! once per span. The spans are cut at the configured size and at the
//! prompt cache's boundaries, with the same checkpoints a span-by-span
//! forward would leave, and only the loop order differs: one layer runs
//! over every span, then the next is read. Within a layer each span still
//! sees the previous span's state, and across layers each span sees its
//! own input, so every kernel runs on the same tokens with the same inputs
//! and the result is bit-identical to running each span through all the
//! layers in turn, which the parity contracts hold it to.

use mlx_bridge::MlxArray;

use super::Qwen3_5Model;
use super::forward_layer_weights::ForwardLayerWeights;
pub(crate) use super::layer_major_prefill_run::LayerMajorPrefillRun;
use super::layer_major_prefill_run::rewind_to_failed_span;
use super::layer_major_span_redo::UnconfirmedSpans;
use super::layer_major_span_state::{
    SpanState, completed_checkpoints, layer_snapshot_arrays, record_end_state_snapshot,
};
use super::layer_submission::{CompletedLayerRoots, LayerGroupSubmission};
use crate::qwen3_5_core::decoder::Qwen3_5PersistentPromptCacheBoundaryCheckpoint;
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
    pub(super) const fn token_count(&self) -> usize {
        self.end_token - self.start_token
    }
}

/// The shape of a layer-major prefill's prompt-cache checkpoints: how many
/// tensors each holds, and the block size boundaries fall on.
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
    /// The request's images, when it has any.
    pub(crate) visual_inputs: Option<LayerMajorVisualInputs<'spans>>,
}

/// The projected image embeddings a prompt's image-pad tokens stand for.
#[derive(Clone, Copy)]
pub(crate) struct LayerMajorVisualInputs<'request> {
    pub(crate) visual_embeddings: &'request MlxArray,
    /// Embedding rows earlier forwards of this request already consumed.
    pub(crate) consumed_visual_embedding_count: usize,
    pub(crate) image_pad_token_id: u32,
}

/// What a layer-major prefill leaves for the engine.
pub(crate) struct LayerMajorPrefillOutcome {
    /// Image embedding rows the prompt's spans consumed.
    pub(crate) consumed_visual_embedding_count: usize,
    /// Float32 logits of the last token when the caller asked for them.
    pub(crate) final_logits: Option<MlxArray>,
    /// Checkpoints at boundaries before the last span's end, in order, with
    /// positions relative to the first span.
    pub(crate) boundary_checkpoints: Vec<Qwen3_5PersistentPromptCacheBoundaryCheckpoint>,
}

/// Bytes per element of the widest floating activation MLX produces, used
/// only if the model's own hidden-state dtype cannot be read.
const WIDEST_ACTIVATION_ELEMENT_BYTES: usize = 4;

impl Qwen3_5Model {
    /// Bytes one token's hidden state occupies between decoder layers, read
    /// from the shape and dtype of a one-token array, which MLX sizes
    /// without allocating it.
    #[doc(hidden)]
    pub fn hidden_state_bytes_per_token(&self) -> usize {
        let hidden_size = self.config.hidden_size() as usize;
        crate::qwen3_5_core::model::bound_weight_dtype_flow::embedded_hidden_state_dtype(
            &self.weights,
        )
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
        let mut layer_major_prefill_run = self.begin_layer_major_prefill(
            token_ids,
            starting_position_tokens,
            layer_major_prefill_request,
        )?;
        loop {
            if let Some(layer_major_prefill_outcome) = self.advance_layer_major_prefill(
                &mut layer_major_prefill_run,
                request_decoder_state,
                performance_attribution,
            )? {
                return Ok(layer_major_prefill_outcome);
            }
        }
    }

    /// Embeds the spans and leaves the prefill ready to run its first layer.
    ///
    /// The prefill then advances one layer per call, so whoever drives it
    /// can report progress and stop between layers.
    pub(crate) fn begin_layer_major_prefill(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        layer_major_prefill_request: LayerMajorPrefillRequest<'_>,
    ) -> Result<LayerMajorPrefillRun, Qwen3_5ExecutionError> {
        let LayerMajorPrefillRequest {
            spans,
            checkpoint_contract,
            should_materialize_logits,
            visual_inputs,
        } = layer_major_prefill_request;
        validate_layer_major_spans(token_ids.len(), spans)?;
        self.discard_expert_state_of_an_unfinished_prompt();
        let (span_states, consumed_visual_embedding_count) =
            self.embed_layer_major_spans(token_ids, spans, checkpoint_contract, visual_inputs)?;
        self.begin_prompt_layer_measurement();
        Ok(LayerMajorPrefillRun {
            spans: spans.to_vec(),
            checkpoint_contract,
            should_materialize_logits,
            starting_position_tokens,
            span_states,
            consumed_visual_embedding_count,
            previous_layer_output: None,
            completed_layer_count: 0,
            next_span_index: 0,
            unconfirmed_spans: UnconfirmedSpans::default(),
            is_rewound_to_failed_span: false,
            span_to_fail_once: None,
        })
    }

    /// Runs the next layer over every span. `Ok(None)` while layers remain;
    /// the outcome once the last layer has run and the prompt's state has
    /// been evaluated. A failure leaves no expert page in hand and no route
    /// pending: a failed prompt's routes are not demand. An expert model's
    /// failed layer is also put back to where its oldest unconfirmed span
    /// began, so advancing again runs that span again and nothing before it
    /// (`LayerMajorPrefillRun::can_run_failed_span_again`).
    pub(crate) fn advance_layer_major_prefill(
        &self,
        layer_major_prefill_run: &mut LayerMajorPrefillRun,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Option<LayerMajorPrefillOutcome>, Qwen3_5ExecutionError> {
        let advance_outcome = self.advance_layer_major_prefill_unguarded(
            layer_major_prefill_run,
            request_decoder_state,
            performance_attribution,
        );
        if advance_outcome.is_err() {
            self.discard_expert_state_of_an_unfinished_prompt();
            rewind_to_failed_span(layer_major_prefill_run, request_decoder_state);
        }
        advance_outcome
    }

    fn advance_layer_major_prefill_unguarded(
        &self,
        layer_major_prefill_run: &mut LayerMajorPrefillRun,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Option<LayerMajorPrefillOutcome>, Qwen3_5ExecutionError> {
        let layer_count = self.config.layer_count() as usize;
        if std::mem::take(&mut layer_major_prefill_run.is_rewound_to_failed_span) {
            // The failure ended the measurement of the layer in flight.
            self.begin_prompt_layer_measurement();
        }
        if layer_major_prefill_run.completed_layer_count < layer_count {
            self.run_layer_over_remaining_spans(
                LayerPass {
                    layer_index: layer_major_prefill_run.completed_layer_count,
                    starting_position_tokens: layer_major_prefill_run.starting_position_tokens,
                    last_span_ends_the_prompt: layer_major_prefill_run.should_materialize_logits,
                },
                layer_major_prefill_run,
                request_decoder_state,
                performance_attribution,
            )?;
            layer_major_prefill_run.completed_layer_count += 1;
            if layer_major_prefill_run.completed_layer_count < layer_count {
                return Ok(None);
            }
        }
        self.finish_layer_major_prefill(
            layer_major_prefill_run,
            request_decoder_state,
            performance_attribution,
        )
        .map(Some)
    }

    /// After the last layer: lets the expert pages go, builds the logits
    /// the caller asked for, and evaluates the prompt's state.
    fn finish_layer_major_prefill(
        &self,
        layer_major_prefill_run: &mut LayerMajorPrefillRun,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<LayerMajorPrefillOutcome, Qwen3_5ExecutionError> {
        self.end_prompt_layer_measurement();
        self.release_held_prompt_expert_layer();
        self.discard_prefill_complete_layer_prefetch();
        let spans = &layer_major_prefill_run.spans;
        let final_logits = match (
            layer_major_prefill_run.should_materialize_logits,
            layer_major_prefill_run.span_states.last(),
        ) {
            (
                true,
                Some(SpanState {
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
        // An expert model's layers recorded their routes as they finished;
        // this wait also writes the experts they queued for the cache and
        // records any route still pending, so none is left for the first
        // reply token to record as reply demand.
        self.evaluate_arrays_resolving_paged_routes(&evaluation_roots, performance_attribution)?;
        Ok(LayerMajorPrefillOutcome {
            consumed_visual_embedding_count: layer_major_prefill_run
                .consumed_visual_embedding_count,
            final_logits,
            boundary_checkpoints: completed_checkpoints(
                spans,
                std::mem::take(&mut layer_major_prefill_run.span_states),
                layer_major_prefill_run.checkpoint_contract,
            )?,
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
                visual_inputs: None,
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

    /// Drops the expert pages in hand and the routes not yet recorded: what
    /// a failed or cancelled attempt leaves behind. Recording such a route
    /// would count a discarded forward as demand, and reading it would
    /// evaluate a graph that already failed.
    pub(crate) fn discard_expert_state_of_an_unfinished_prompt(&self) {
        self.end_prompt_layer_measurement();
        self.release_held_prompt_expert_layer();
        self.discard_prefill_complete_layer_prefetch();
        self.deferred_expert_demand
            .borrow_mut()
            .discard_pending_layer_routes();
    }

    /// Runs one decoder layer over its remaining spans in order, reading a
    /// streamed layer once for all of them. Every span remains, except
    /// after a failure put the layer back to the span that failed.
    fn run_layer_over_remaining_spans(
        &self,
        layer_pass: LayerPass,
        layer_major_prefill_run: &mut LayerMajorPrefillRun,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), Qwen3_5ExecutionError> {
        let forward_layer_weights = self.forward_layer_weights(layer_pass.layer_index)?;
        let span_count = layer_major_prefill_run.spans.len();
        let spans_token_count = i32::try_from(
            layer_major_prefill_run
                .spans
                .last()
                .map_or(0, |last_span| last_span.end_token),
        )
        .unwrap_or(i32::MAX);
        self.hold_prompt_expert_layer(
            layer_pass.layer_index,
            spans_token_count,
            performance_attribution,
        );
        let first_span_index = layer_major_prefill_run.next_span_index;
        for span_index in first_span_index..span_count {
            self.set_prefill_demand_assignment_weight_for_residency(span_demand_weight(
                layer_pass,
                &layer_major_prefill_run.spans[span_index],
                span_index + 1 == span_count,
            ));
            self.remember_span_start(
                layer_pass.layer_index,
                span_index,
                layer_major_prefill_run,
                request_decoder_state,
            )?;
            self.run_and_submit_span(
                layer_pass,
                &forward_layer_weights,
                &layer_major_prefill_run.spans[span_index],
                &mut layer_major_prefill_run.span_states[span_index],
                request_decoder_state,
                &mut layer_major_prefill_run.previous_layer_output,
                performance_attribution,
            )?;
            layer_major_prefill_run.fail_marked_span_once(layer_pass.layer_index, span_index)?;
            // Submitting this span waited for the one before it.
            layer_major_prefill_run
                .unconfirmed_spans
                .confirm_all_but_newest();
            layer_major_prefill_run.next_span_index = span_index + 1;
            self.read_next_expert_layer_ahead_after_first_span(
                span_index - first_span_index,
                layer_pass.layer_index,
                spans_token_count,
                performance_attribution,
            );
        }
        let pages_in_hand = self.expert_pages_in_hand_bytes();
        self.release_held_prompt_expert_layer();
        self.record_finished_expert_layer_routes(
            &mut layer_major_prefill_run.previous_layer_output,
            performance_attribution,
        )?;
        layer_major_prefill_run.unconfirmed_spans.confirm_all();
        layer_major_prefill_run.next_span_index = 0;
        self.govern_completed_prompt_layer(
            pages_in_hand,
            request_decoder_state.payload_byte_count(),
        );
        Ok(())
    }

    /// Keeps what a span finds as it begins, so the span can be run again
    /// if MLX cannot fit it. Only for an expert model: it waits for every
    /// layer, so a failure is reported inside the layer that caused it and
    /// what the span found is still evaluated state.
    fn remember_span_start(
        &self,
        layer_index: usize,
        span_index: usize,
        layer_major_prefill_run: &mut LayerMajorPrefillRun,
        request_decoder_state: &RequestDecoderStateStack,
    ) -> Result<(), Qwen3_5ExecutionError> {
        if self.retained_experts.is_none() {
            return Ok(());
        }
        layer_major_prefill_run
            .unconfirmed_spans
            .remember_span_start(
                layer_index,
                span_index,
                &layer_major_prefill_run.span_states,
                request_decoder_state,
                layer_major_prefill_run.previous_layer_output.as_ref(),
            )
    }

    /// Runs one span through one layer and submits it with the state and
    /// prompt-cache snapshots it leaves.
    #[allow(clippy::too_many_arguments)]
    fn run_and_submit_span(
        &self,
        layer_pass: LayerPass,
        forward_layer_weights: &ForwardLayerWeights<'_>,
        span: &LayerMajorSpan,
        span_state: &mut SpanState,
        request_decoder_state: &mut RequestDecoderStateStack,
        previous_layer_output: &mut Option<MlxArray>,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), Qwen3_5ExecutionError> {
        let span_output = self.run_span_through_layer(
            layer_pass,
            forward_layer_weights,
            span,
            span_state,
            request_decoder_state,
            performance_attribution,
        )?;
        let layer_state = request_decoder_state.layer(layer_pass.layer_index);
        if let (Some(end_state_snapshot), Some(layer_state)) =
            (span_state.end_state_snapshot.as_mut(), layer_state)
        {
            record_end_state_snapshot(end_state_snapshot, layer_pass.layer_index, layer_state)?;
        }
        let mut boundary_checkpoint_states = span_state
            .collector
            .as_ref()
            .map(|collector| collector.layer_evaluation_arrays(layer_pass.layer_index))
            .unwrap_or_default();
        // The end-state snapshot holds the layer's own state arrays, which
        // this submission evaluates anyway; naming them keeps that true if
        // the state ever stops being reachable from the output.
        boundary_checkpoint_states.extend(
            span_state
                .end_state_snapshot
                .as_ref()
                .map(|snapshot| layer_snapshot_arrays(snapshot, layer_pass.layer_index))
                .unwrap_or_default(),
        );
        self.submit_completed_layer(
            forward_layer_weights,
            CompletedLayerRoots {
                layer_output: &span_output,
                convolution_state: layer_state.and_then(DecoderCacheState::convolution_state_array),
                boundary_checkpoint_states,
            },
            previous_layer_output,
            LayerGroupSubmission {
                completed_layer_count: layer_pass.layer_index + 1,
                decoder_layer_count: self.config.layer_count() as usize,
                graph_submission_layer_interval: 0,
                forward_token_count: i32::try_from(span.token_count()).unwrap_or(i32::MAX),
            },
            performance_attribution,
        )?;
        span_state.hidden_states = Some(span_output);
        Ok(())
    }

    /// Builds one span's pass through one layer from that span's hidden
    /// states, at the span's own position.
    fn run_span_through_layer(
        &self,
        layer_pass: LayerPass,
        forward_layer_weights: &ForwardLayerWeights<'_>,
        span: &LayerMajorSpan,
        span_state: &mut SpanState,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let layer_index = layer_pass.layer_index;
        let span_input =
            span_state
                .hidden_states
                .take()
                .ok_or(Qwen3_5ExecutionError::InvalidInput {
                    description: "a layer-major span lost its hidden states",
                })?;
        let rope_offset_tokens =
            i32::try_from(u64::from(layer_pass.starting_position_tokens) + span.start_token as u64)
                .map_err(|_| Qwen3_5ExecutionError::InvalidInput {
                    description: "a layer-major span starts beyond the MLX int32 range",
                })?;
        let layer_model_state = request_decoder_state.layer_mut(layer_index).ok_or(
            Qwen3_5ExecutionError::InvalidRequestDecoderState {
                layer_index,
                description: "decoder layer entry is missing",
            },
        )?;
        if !layer_model_state.tensors_are_allocated_consistently() {
            return Err(super::error::invalid_request_decoder_state(
                layer_index,
                "decoder layer tensors must be allocated together or absent together",
            ));
        }
        self.forward_decoder_layer(
            &span_input,
            i32::try_from(span.token_count()).unwrap_or(i32::MAX),
            rope_offset_tokens,
            layer_index,
            forward_layer_weights.decoder_layer_weights(),
            layer_model_state,
            None,
            span_state.collector.as_mut(),
            Qwen3_5MoEPagedExecutionMode::ProductionDefault,
            performance_attribution,
        )
    }
}

/// Which layer one pass over the spans runs, and where the prompt starts.
#[derive(Clone, Copy)]
struct LayerPass {
    layer_index: usize,
    starting_position_tokens: u32,
    /// Whether the last span ends the prompt, which makes it the span whose
    /// expert routes best predict the reply's.
    last_span_ends_the_prompt: bool,
}

/// The weight one span's expert routes are recorded under as prompt demand.
///
/// The span that ends the prompt records under the final-chunk weight, as
/// the chunk that ends a prompt does; every other span records plainly.
fn span_demand_weight(layer_pass: LayerPass, span: &LayerMajorSpan, is_last_span: bool) -> u64 {
    if !(is_last_span && layer_pass.last_span_ends_the_prompt) {
        return 1;
    }
    crate::last_prefill_chunk_demand_weight(
        u64::from(layer_pass.starting_position_tokens).saturating_add(span.start_token as u64),
        span.token_count() as u64,
    )
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
