//! The public measurement surface for the folded ternary family.
//!
//! The acceptance performance tiers climb the call stack from the lowest
//! GPU operation to full generation. Each tier needs one seam the private
//! forward pass does not expose on its own: the Hadamard transform alone, one
//! rotated ternary projection, one decoder layer, one chunked prefill, one
//! decode step, and a greedy session whose prefill and decode phases a caller
//! can time separately. Every probe runs exactly the arithmetic the production
//! forward runs, on the same retained weights and compiled graphs, so a
//! measurement here is a measurement of the serving path.

use mlx_c_rust::MlxArray;

use super::forward::StraightGenerationState;
use super::weights::DecoderLayerAttentionWeights;
use super::{PrismHadamardModelError, PrismHadamardTextModel, validate_greedy_generation_request};

/// The per-sequence generation state (gated-delta convolution and recurrent
/// state plus full-attention key-value slabs), owned by the caller across
/// prefill chunks and decode steps.
pub struct PrismHadamardGenerationState(pub(crate) StraightGenerationState);

impl PrismHadamardGenerationState {
    /// A fresh state covering no tokens.
    #[must_use]
    pub fn empty() -> Self {
        Self(StraightGenerationState::empty())
    }

    /// How many tokens the state already covers.
    #[must_use]
    pub const fn covered_token_count(&self) -> i32 {
        self.0.sequence_position_offset()
    }
}

impl Default for PrismHadamardGenerationState {
    fn default() -> Self {
        Self::empty()
    }
}

/// The pack dimensions the operation tier sizes its reports with.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrismHadamardProbeDimensions {
    pub hidden_width: usize,
    pub intermediate_width: usize,
    /// The stacked output rows of the first gated-delta layer's fused
    /// query/key/value-and-gate launch (`2 * key + 2 * value` widths).
    pub gated_delta_input_projection_rows: usize,
    /// The stacked output rows of the fused MLP gate-and-up launch
    /// (`2 * intermediate`).
    pub dense_mlp_gate_and_up_rows: usize,
    pub vocabulary_size: usize,
}

/// Which retained rotated projection launch a projection probe exercises.
/// Each is one launch exactly as the forward pass issues it, so the fused
/// sibling projections are timed as the single launch serving runs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrismHadamardProbeProjection {
    /// The first gated-delta layer's fused query/key/value-and-gate launch
    /// (hidden width in, `2 * key + 2 * value` rows out).
    GatedDeltaQueryKeyValueAndGate,
    /// The first decoder layer's fused MLP gate-and-up launch (hidden width
    /// in, `2 * intermediate` rows out): the tallest per-layer launch.
    DenseMlpGateAndUp,
    /// The first decoder layer's MLP down projection (intermediate width in,
    /// hidden width out): the widest activation the transform sees.
    DenseMlpDown,
    /// The vocabulary projection (hidden width in, vocabulary out): the
    /// largest packed matrix in the pack.
    LanguageModelHead,
}

impl PrismHadamardTextModel {
    /// The Hadamard block size every rotated projection applies.
    #[must_use]
    pub fn hadamard_block_size(&self) -> u32 {
        self.pack.hadamard_contract().block_size()
    }

    /// The widths the operation tier measures against.
    #[must_use]
    pub fn probe_dimensions(&self) -> PrismHadamardProbeDimensions {
        let text = self.pack.config().text();
        PrismHadamardProbeDimensions {
            hidden_width: text.hidden_size as usize,
            intermediate_width: text.intermediate_size as usize,
            gated_delta_input_projection_rows: (2
                * text.linear_num_key_heads
                * text.linear_key_head_dim
                + 2 * text.linear_num_value_heads * text.linear_value_head_dim)
                as usize,
            dense_mlp_gate_and_up_rows: 2 * text.intermediate_size as usize,
            vocabulary_size: text.vocab_size as usize,
        }
    }

    /// Materializes lazy outputs on the model's runtime.
    pub fn evaluate_outputs(&self, outputs: &[&MlxArray]) -> Result<(), PrismHadamardModelError> {
        self.runtime
            .evaluate_arrays(outputs)
            .map_err(PrismHadamardModelError::Forward)
    }

    /// Embeds `token_ids` into `[1, token_count, hidden]` activations through
    /// the packed embedding and its inverse transform.
    pub fn probe_embedding(&self, token_ids: &[i32]) -> Result<MlxArray, PrismHadamardModelError> {
        let token_indices = self
            .runtime
            .array_from_i32(token_ids, &[1, token_ids.len() as i32])
            .map_err(PrismHadamardModelError::Forward)?;
        self.embed_tokens(&token_indices)
            .map_err(PrismHadamardModelError::Forward)
    }

    /// Applies the forward Hadamard transform exactly as every rotated
    /// projection's input receives it, for an activation whose last axis is
    /// a width the pack's sign table declares.
    pub fn probe_forward_transform(
        &self,
        activations: &MlxArray,
    ) -> Result<MlxArray, PrismHadamardModelError> {
        self.rotate(activations)
            .map(super::fwht::RotatedActivations::into_activations)
            .map_err(PrismHadamardModelError::Forward)
    }

    /// Runs one rotated ternary projection launch (transform plus packed
    /// matmul) on `activations`, exactly as the forward pass runs it; a fused
    /// launch returns its stacked output unsplit.
    pub fn probe_packed_projection(
        &self,
        projection: PrismHadamardProbeProjection,
        activations: &MlxArray,
    ) -> Result<MlxArray, PrismHadamardModelError> {
        let rotated_activations = self
            .rotate(activations)
            .map_err(PrismHadamardModelError::Forward)?;
        match projection {
            PrismHadamardProbeProjection::GatedDeltaQueryKeyValueAndGate => {
                match &self.first_decoder_layer()?.attention {
                    DecoderLayerAttentionWeights::Linear(linear_layer) => linear_layer
                        .query_key_value_and_gate
                        .forward_fused(&self.runtime, &rotated_activations),
                    DecoderLayerAttentionWeights::Full(_) => {
                        return Err(PrismHadamardModelError::UnsupportedLayerType {
                            layer_type: "the first decoder layer is not a gated-delta layer"
                                .to_owned(),
                        });
                    }
                }
            }
            PrismHadamardProbeProjection::DenseMlpGateAndUp => self
                .first_decoder_layer()?
                .mlp
                .gate_and_up
                .forward_fused(&self.runtime, &rotated_activations),
            PrismHadamardProbeProjection::DenseMlpDown => self
                .first_decoder_layer()?
                .mlp
                .down
                .forward(&self.runtime, &rotated_activations),
            PrismHadamardProbeProjection::LanguageModelHead => self
                .weights
                .language_model_head
                .forward(&self.runtime, &rotated_activations),
        }
        .map_err(PrismHadamardModelError::Forward)
    }

    /// Runs the first decoder layer's dense MLP up to (and including) the
    /// SwiGLU, returning the intermediate-width activations the down
    /// projection consumes.
    pub fn probe_dense_mlp_intermediate(
        &self,
        normalized_hidden_states: &MlxArray,
    ) -> Result<MlxArray, PrismHadamardModelError> {
        let dense_mlp = &self.first_decoder_layer()?.mlp;
        let rotated_input = self
            .rotate(normalized_hidden_states)
            .map_err(PrismHadamardModelError::Forward)?;
        let [gate_activations, up_activations] = dense_mlp
            .gate_and_up
            .forward(&self.runtime, &rotated_input)
            .map_err(PrismHadamardModelError::Forward)?;
        self.runtime
            .apply_compiled_swiglu(&self.compiled_swiglu, &gate_activations, &up_activations)
            .map_err(PrismHadamardModelError::Forward)
    }

    /// The index of the first decoder layer of each attention kind, if the
    /// pack has one: `(gated_delta_layer_index, full_attention_layer_index)`.
    #[must_use]
    pub fn first_layer_indices_by_kind(&self) -> (Option<usize>, Option<usize>) {
        let mut first_linear = None;
        let mut first_full = None;
        for (layer_index, decoder_layer) in self.weights.decoder_layers.iter().enumerate() {
            match decoder_layer.attention {
                DecoderLayerAttentionWeights::Linear(_) => {
                    first_linear.get_or_insert(layer_index);
                }
                DecoderLayerAttentionWeights::Full(_) => {
                    first_full.get_or_insert(layer_index);
                }
            }
        }
        (first_linear, first_full)
    }

    /// Runs one decoder layer over `[1, token_count, hidden]` activations,
    /// advancing that layer's slot in `state`. The caller is responsible for
    /// keeping `state` consistent with the positions already covered: a
    /// decode-shaped probe (`token_count == 1`) should follow a prefill probe
    /// on the same state so the layer attends over a realistic prefix.
    pub fn probe_decoder_layer(
        &self,
        layer_index: usize,
        hidden_states: &MlxArray,
        state: &mut PrismHadamardGenerationState,
    ) -> Result<MlxArray, PrismHadamardModelError> {
        if layer_index >= self.weights.decoder_layers.len() {
            return Err(PrismHadamardModelError::UnsupportedLayerType {
                layer_type: format!("decoder layer {layer_index} does not exist"),
            });
        }
        self.forward_decoder_layer(layer_index, hidden_states, &mut state.0)
    }

    /// Advances `state` by the token count of the last layer probe so a
    /// following probe sees the right RoPE offset and key-value window.
    pub fn advance_probe_state(&self, state: &mut PrismHadamardGenerationState, token_count: i32) {
        state.0.advance_position(token_count);
    }

    /// Runs one prefill pass over `token_ids` (one chunk; the caller slices
    /// chunks) and returns the final position's vocabulary logits, lazily.
    pub fn prefill_chunk(
        &self,
        token_ids: &[i32],
        state: &mut PrismHadamardGenerationState,
    ) -> Result<MlxArray, PrismHadamardModelError> {
        let token_indices = self
            .runtime
            .array_from_i32(token_ids, &[1, token_ids.len() as i32])
            .map_err(PrismHadamardModelError::Forward)?;
        self.forward_pass(&token_indices, &mut state.0)
    }

    /// Runs one decode step for `token_id` and returns its logits, lazily.
    pub fn decode_step(
        &self,
        token_id: u32,
        state: &mut PrismHadamardGenerationState,
    ) -> Result<MlxArray, PrismHadamardModelError> {
        let token_indices = self
            .runtime
            .array_from_i32(&[token_id as i32], &[1, 1])
            .map_err(PrismHadamardModelError::Forward)?;
        self.forward_pass(&token_indices, &mut state.0)
    }

    /// Materializes `output` together with every retained state array, so
    /// the step's whole graph runs now instead of being dragged into the
    /// next step.
    pub fn evaluate_step(
        &self,
        output: &MlxArray,
        state: &PrismHadamardGenerationState,
    ) -> Result<(), PrismHadamardModelError> {
        self.runtime
            .evaluate_arrays(&state.0.evaluation_arrays(output))
            .map_err(PrismHadamardModelError::Forward)
    }

    /// The highest-probability vocabulary id at the final position of
    /// `logits`, synchronizing with the graphics processor.
    pub fn greedy_token_id(&self, logits: &MlxArray) -> Result<u32, PrismHadamardModelError> {
        self.argmax_last_token(logits)
    }

    /// The highest-probability vocabulary id as a lazy `[1, 1]` device
    /// array that the next forward can consume directly as its token index.
    pub(crate) fn greedy_token_lazily(
        &self,
        logits: &MlxArray,
    ) -> Result<MlxArray, PrismHadamardModelError> {
        self.runtime
            .argmax_axis(logits, 2)
            .map_err(PrismHadamardModelError::Forward)
    }

    /// Prefills the prompt in chunks of `prompt_processing_chunk_tokens` and
    /// returns a session that decodes greedily one token per call. The prompt
    /// is fully processed (every chunk evaluated) before this returns, so a
    /// caller timing the call measures prompt processing alone.
    pub fn begin_greedy_session(
        &self,
        prompt_token_ids: &[i32],
        prompt_processing_chunk_tokens: u32,
        maximum_new_token_count: usize,
        stop_token_ids: &[u32],
    ) -> Result<PrismHadamardGreedySession<'_>, PrismHadamardModelError> {
        let text_dimensions = self.pack.config().text();
        validate_greedy_generation_request(
            prompt_token_ids,
            maximum_new_token_count,
            text_dimensions.vocab_size,
            text_dimensions.max_position_embeddings,
        )?;
        let chunk_tokens = (prompt_processing_chunk_tokens.max(1)) as usize;
        let mut state = PrismHadamardGenerationState::empty();
        let mut final_logits = None;
        for chunk in prompt_token_ids.chunks(chunk_tokens) {
            let chunk_logits = self.prefill_chunk(chunk, &mut state)?;
            self.evaluate_step(&chunk_logits, &state)?;
            final_logits = Some(chunk_logits);
        }
        let final_logits = final_logits.ok_or(PrismHadamardModelError::EmptyGeneration)?;
        let first_token = self.greedy_token_lazily(&final_logits)?;
        self.runtime
            .async_eval_arrays(&[&first_token])
            .map_err(PrismHadamardModelError::Forward)?;
        Ok(PrismHadamardGreedySession {
            model: self,
            state,
            pending_token: Some(first_token),
            remaining_token_count: maximum_new_token_count,
            stop_token_ids: stop_token_ids.to_vec(),
        })
    }

    fn first_decoder_layer(
        &self,
    ) -> Result<&super::weights::DecoderLayerWeights, PrismHadamardModelError> {
        self.weights
            .decoder_layers
            .first()
            .ok_or(PrismHadamardModelError::UnsupportedLayerType {
                layer_type: "the pack declares no decoder layers".to_owned(),
            })
    }
}

/// One greedy generation in progress: the prompt is already processed, and
/// each `next_token` call commits one more generated token.
///
/// The session keeps one token in flight. The token a call returns was
/// sampled on the graphics processor by the previous call and its successor
/// forward is already submitted before the call synchronizes, so host graph
/// construction for step `n + 1` overlaps the device's execution of step
/// `n` instead of serializing behind it.
pub struct PrismHadamardGreedySession<'model> {
    model: &'model PrismHadamardTextModel,
    state: PrismHadamardGenerationState,
    /// The lazily sampled next token, `[1, 1]` uint32, whose producing graph
    /// is already submitted.
    pending_token: Option<MlxArray>,
    remaining_token_count: usize,
    stop_token_ids: Vec<u32>,
}

impl PrismHadamardGreedySession<'_> {
    /// Commits the next greedy token, or `None` once a stop id appeared or
    /// the budget is spent. A stop id is never returned.
    pub fn next_token(&mut self) -> Result<Option<u32>, PrismHadamardModelError> {
        if self.remaining_token_count == 0 {
            self.pending_token = None;
            return Ok(None);
        }
        let Some(token) = self.pending_token.take() else {
            return Ok(None);
        };
        // Build and submit the successor before synchronizing: the device
        // keeps working while the host waits for this token's value. The
        // successor is wasted only when this token turns out to be a stop id.
        let successor = if self.remaining_token_count > 1 {
            let successor_logits = self.model.forward_pass(&token, &mut self.state.0)?;
            let successor = self.model.greedy_token_lazily(&successor_logits)?;
            self.model
                .runtime
                .async_eval_arrays(&self.state.0.evaluation_arrays(&successor))
                .map_err(PrismHadamardModelError::Forward)?;
            Some(successor)
        } else {
            None
        };
        let token_id = token.item_u32().map_err(PrismHadamardModelError::Forward)?;
        if self.stop_token_ids.contains(&token_id) {
            self.pending_token = None;
            return Ok(None);
        }
        self.remaining_token_count -= 1;
        self.pending_token = successor;
        Ok(Some(token_id))
    }

    /// How many tokens the session's state covers (prompt plus committed
    /// generated tokens already fed back).
    #[must_use]
    pub const fn covered_token_count(&self) -> i32 {
        self.state.covered_token_count()
    }
}
