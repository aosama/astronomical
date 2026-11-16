//! The speculative-verification forward for the resident target model.
//!
//! Speculative decoding asks one thing of the target that neither prefill nor
//! decode provides today: a multi-token forward whose outputs are logits at
//! the positions the caller needs (to verify drafted tokens against the
//! target distribution) plus hidden states captured at the DFlash draft's
//! conditioning taps (to grow the draft's context key-value state). This
//! entry reuses the exact per-layer forward, normalization, and
//! language-model head of the ordinary graph; the differences are the tap
//! captures and the caller-selected head scope.

use crate::performance_attribution::PerformanceAttribution;
use crate::qwen3_5_core::decoder::request_state::RequestDecoderStateStack;
use crate::qwen3_5_core::decoder::{
    Qwen3_5AstroSsdPromptCacheBoundaryCheckpoint,
    Qwen3_5AstroSsdPromptCacheBoundaryCheckpointCollector,
};
use mlx_bridge::{MlxArray, MlxDtype};

use super::{Qwen3_5ExecutionError, Qwen3_5ResidentModel};

/// What the verification forward does with the language-model head.
///
/// Intermediate prefill chunks only need the taps: materializing the head
/// for thousands of positions would spend gigabytes on distributions nobody
/// reads. The terminal prefill chunk needs one row (the first bonus token);
/// speculative rounds need every row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationLogitsMode {
    /// No head pass at all; `per_position_logits` comes back `None`.
    NoLogits,
    /// Head on the final position only: `[1, 1, vocabulary]`.
    LastRowOnly,
    /// Head on every position: `[1, token count, vocabulary]`.
    AllPositions,
}

/// One drafted block's verification: logits for every row, the taps, and the
/// DeltaNet state after each row but the last.
pub struct Qwen3_5DraftedRowsVerification {
    /// `[1, rows, vocabulary]` float32.
    pub per_position_logits: MlxArray,
    pub hidden_state_taps: Vec<MlxArray>,
    /// Entry `n - 1` holds the DeltaNet state after the first `n` rows.
    pub row_boundaries: Vec<Qwen3_5AstroSsdPromptCacheBoundaryCheckpoint>,
}

/// One verification forward's outputs.
pub struct Qwen3_5SpeculativeVerificationOutput {
    /// Target logits according to the requested mode (see
    /// [`VerificationLogitsMode`]), in float32.
    pub per_position_logits: Option<MlxArray>,
    /// One `[1, token count, hidden size]` capture per requested tap, in the
    /// caller's tap order: the output hidden states of those decoder layers.
    pub hidden_state_taps: Vec<MlxArray>,
}

/// Range-checked inputs for one speculative verification forward: the RoPE
/// offset, the token count, and the materialized token-id tensor.
struct SpeculativeVerificationInputs {
    rope_offset_tokens: i32,
    token_count: i32,
    token_indices: MlxArray,
}

impl Qwen3_5ResidentModel {
    /// Runs one multi-token verification forward at `starting_position_tokens`.
    ///
    /// `target_hidden_state_tap_layer_ids` names target decoder layers whose
    /// output hidden states the caller conditions the draft on; the draft's
    /// configuration pins them and the caller keeps the order stable.
    pub fn build_speculative_verification_forward(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        target_hidden_state_tap_layer_ids: &[i32],
        logits_mode: VerificationLogitsMode,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Qwen3_5SpeculativeVerificationOutput, Qwen3_5ExecutionError> {
        let SpeculativeVerificationInputs {
            rope_offset_tokens,
            token_count,
            token_indices,
        } = self.validated_speculative_verification_inputs(
            token_ids,
            starting_position_tokens,
            target_hidden_state_tap_layer_ids,
        )?;

        let hidden_states = self.embedding_lookup(&token_indices)?;
        let (hidden_states, hidden_state_taps) = self.run_verification_layers(
            hidden_states,
            token_count,
            rope_offset_tokens,
            request_decoder_state,
            target_hidden_state_tap_layer_ids,
            None,
            performance_attribution,
        )?;
        let per_position_logits =
            self.verification_per_position_logits(&hidden_states, logits_mode, token_count)?;
        Ok(Qwen3_5SpeculativeVerificationOutput {
            per_position_logits,
            hidden_state_taps,
        })
    }

    /// Runs every decoder layer over the verification rows, capturing the
    /// requested taps, and the DeltaNet state at each boundary the collector
    /// names when one is passed. Returns the final hidden rows and the taps.
    #[allow(clippy::too_many_arguments)]
    fn run_verification_layers(
        &self,
        mut hidden_states: MlxArray,
        token_count: i32,
        rope_offset_tokens: i32,
        request_decoder_state: &mut RequestDecoderStateStack,
        target_hidden_state_tap_layer_ids: &[i32],
        mut boundary_checkpoint_collector: Option<
            &mut Qwen3_5AstroSsdPromptCacheBoundaryCheckpointCollector,
        >,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(MlxArray, Vec<MlxArray>), Qwen3_5ExecutionError> {
        let mut hidden_state_tap_slots: Vec<Option<MlxArray>> = (0
            ..target_hidden_state_tap_layer_ids.len())
            .map(|_| None)
            .collect();
        for layer_index in 0..self.config.layer_count() as usize {
            let decoder_layer_weights = self
                .weights
                .decoder_layer_weights
                .get(layer_index)
                .ok_or(Qwen3_5ExecutionError::MissingDecoderLayerWeights { layer_index })?;
            let layer_model_state = request_decoder_state.layer_mut(layer_index).ok_or(
                Qwen3_5ExecutionError::InvalidRequestDecoderState {
                    layer_index,
                    description: "decoder layer entry is missing",
                },
            )?;
            if !layer_model_state.tensors_are_allocated_consistently() {
                return Err(Qwen3_5ExecutionError::InvalidRequestDecoderState {
                    layer_index,
                    description: "decoder layer tensors must be allocated together or absent together",
                });
            }
            hidden_states = self.forward_decoder_layer(
                &hidden_states,
                token_count,
                rope_offset_tokens,
                layer_index,
                decoder_layer_weights,
                layer_model_state,
                None,
                boundary_checkpoint_collector.as_deref_mut(),
                performance_attribution,
            )?;
            let tap_order = target_hidden_state_tap_layer_ids
                .iter()
                .position(|&tap_layer_id| tap_layer_id == layer_index as i32);
            if let Some(tap_order) = tap_order {
                // The tap must survive the graph that produced it: retain a
                // handle so the layer's lazy graph can be freed once the
                // caller finishes the round.
                hidden_state_tap_slots[tap_order] = Some(hidden_states.retain()?);
            }
        }
        let mut hidden_state_taps = Vec::with_capacity(hidden_state_tap_slots.len());
        for tap_slot in hidden_state_tap_slots {
            hidden_state_taps.push(tap_slot.ok_or(Qwen3_5ExecutionError::InvalidInput {
                description: "a requested DFlash target hidden-state tap was never captured",
            })?);
        }
        Ok((hidden_states, hidden_state_taps))
    }

    /// Verifies one drafted block: the target's logits for every row, its
    /// taps, and the DeltaNet state after each row but the last, so a round
    /// that keeps only some rows rewinds without running the target again
    /// (`RequestDecoderStateStack::rewind_to_kept_row_boundary`). Keeping
    /// every row needs no boundary: the live state is already there.
    pub fn verify_drafted_rows(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        target_hidden_state_tap_layer_ids: &[i32],
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Qwen3_5DraftedRowsVerification, Qwen3_5ExecutionError> {
        let SpeculativeVerificationInputs {
            rope_offset_tokens,
            token_count,
            token_indices,
        } = self.validated_speculative_verification_inputs(
            token_ids,
            starting_position_tokens,
            target_hidden_state_tap_layer_ids,
        )?;
        if token_count < 2 {
            return Err(Qwen3_5ExecutionError::InvalidInput {
                description: "a drafted block verifies the anchor and at least one draft",
            });
        }
        let mut row_boundary_collector =
            Qwen3_5AstroSsdPromptCacheBoundaryCheckpointCollector::new(
                (1..token_ids.len()).collect(),
                self.astro_decoder_state_cache_layout
                    .boundary_tensor_count(),
                1,
            )?;
        let (hidden_states, hidden_state_taps) = self.run_verification_layers(
            self.embedding_lookup(&token_indices)?,
            token_count,
            rope_offset_tokens,
            request_decoder_state,
            target_hidden_state_tap_layer_ids,
            Some(&mut row_boundary_collector),
            performance_attribution,
        )?;
        let per_position_logits = self
            .verification_per_position_logits(
                &hidden_states,
                VerificationLogitsMode::AllPositions,
                token_count,
            )?
            .ok_or(Qwen3_5ExecutionError::InvalidInput {
                description: "a drafted block must produce logits",
            })?;
        Ok(Qwen3_5DraftedRowsVerification {
            per_position_logits,
            hidden_state_taps,
            row_boundaries: row_boundary_collector.complete()?,
        })
    }

    /// The range-checked inputs one verification forward needs: the RoPE
    /// offset as MLX's int32, the token count in the same range, and the
    /// token-id tensor materialized from signed ids. Also proves every
    /// requested hidden-state tap names a real decoder layer of this model —
    /// a tap outside the target model can never be captured later.
    fn validated_speculative_verification_inputs(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        target_hidden_state_tap_layer_ids: &[i32],
    ) -> Result<SpeculativeVerificationInputs, Qwen3_5ExecutionError> {
        let rope_offset_tokens = i32::try_from(starting_position_tokens).map_err(|_| {
            Qwen3_5ExecutionError::InvalidInput {
                description: "starting position exceeds the MLX int32 range",
            }
        })?;
        let token_count =
            i32::try_from(token_ids.len()).map_err(|_| Qwen3_5ExecutionError::InvalidInput {
                description: "verification token count exceeds the MLX int32 range",
            })?;
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
        for &tap_layer_id in target_hidden_state_tap_layer_ids {
            let tap_layer_index = usize::try_from(tap_layer_id)
                .ok()
                .filter(|tap_layer_index| *tap_layer_index < self.config.layer_count() as usize);
            if tap_layer_index.is_none() {
                return Err(Qwen3_5ExecutionError::InvalidInput {
                    description: "a DFlash target hidden-state tap names a layer outside the target model",
                });
            }
        }
        Ok(SpeculativeVerificationInputs {
            rope_offset_tokens,
            token_count,
            token_indices,
        })
    }

    /// Produces the requested logits from the final hidden states. NoLogits
    /// skips the head entirely. LastRowOnly reads just the terminal prefill
    /// position — the only row whose distribution the caller consumes — while
    /// AllPositions reads every row because each position's logits are the
    /// target's distribution over its next token. The head output is widened
    /// to float32 for deterministic verification comparisons.
    pub(super) fn verification_per_position_logits(
        &self,
        hidden_states: &MlxArray,
        logits_mode: VerificationLogitsMode,
        token_count: i32,
    ) -> Result<Option<MlxArray>, Qwen3_5ExecutionError> {
        let per_position_logits = match logits_mode {
            VerificationLogitsMode::NoLogits => None,
            VerificationLogitsMode::LastRowOnly | VerificationLogitsMode::AllPositions => {
                let normalized_states = self.runtime.rms_norm(
                    hidden_states,
                    &self.weights.final_normalization_weight,
                    f32::from_bits(self.config.rms_norm_epsilon_bits()),
                )?;
                let head_input = if logits_mode == VerificationLogitsMode::LastRowOnly {
                    self.runtime.slice(
                        &normalized_states,
                        &[0, token_count - 1, 0],
                        &[1, token_count, self.config.hidden_size() as i32],
                        &[1, 1, 1],
                    )?
                } else {
                    normalized_states
                };
                let head_logits =
                    self.quantized_linear(&head_input, &self.weights.language_model_head_weights)?;
                Some(self.runtime.astype(&head_logits, MlxDtype::Float32)?)
            }
        };
        Ok(per_position_logits)
    }
}
