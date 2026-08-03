use astronomical_runtime_integration::{MlxArray, MlxDtype, MlxRuntime, MlxRuntimeError};

use crate::qwen3_5_moe::Qwen3_5MoEPagedPrefillExecutionMode;
use crate::{PerformanceAttribution, PerformanceOperation};

use super::model::Qwen3_5Model;
use super::{Qwen3_5ExecutionError, RequestDecoderStateStack};

const PREFILL_ASYNC_SUBMISSION_LAYER_INTERVAL: usize = 8;

/// Target-model graph outputs needed to seed MTP drafting from a verified target forward.
pub struct Qwen3_5TargetForwardOutput {
    final_logits: MlxArray,
    all_position_logits: Option<MlxArray>,
    pre_final_normalization_hidden_states: MlxArray,
}

impl Qwen3_5TargetForwardOutput {
    /// Returns float32 logits for the final target position.
    #[must_use]
    pub fn final_logits(&self) -> &MlxArray {
        &self.final_logits
    }

    /// Returns float32 logits for every target input position when requested.
    #[must_use]
    pub fn all_position_logits(&self) -> Option<&MlxArray> {
        self.all_position_logits.as_ref()
    }

    /// Returns every final decoder row before the trunk's final RMS normalization.
    #[must_use]
    pub fn pre_final_normalization_hidden_states(&self) -> &MlxArray {
        &self.pre_final_normalization_hidden_states
    }

    pub(crate) fn into_pre_final_normalization_hidden_states(self) -> MlxArray {
        self.pre_final_normalization_hidden_states
    }

    /// Returns one pre-final-normalization hidden row from this target forward.
    pub fn pre_final_normalization_hidden_state_at(
        &self,
        runtime: &MlxRuntime,
        token_position_index: i32,
    ) -> Result<MlxArray, MlxRuntimeError> {
        pre_final_normalization_hidden_state_at(
            runtime,
            &self.pre_final_normalization_hidden_states,
            token_position_index,
        )
    }
}

impl Qwen3_5Model {
    /// Executes a target forward and retains the pre-final-normalization rows needed by MTP.
    pub fn forward_chunk_with_pre_final_normalization_hidden_states(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
    ) -> Result<Qwen3_5TargetForwardOutput, Qwen3_5ExecutionError> {
        let mut disabled_performance_attribution = PerformanceAttribution::disabled();
        let target_forward_output = self.build_target_forward_graph(
            token_ids,
            starting_position_tokens,
            request_decoder_state,
            Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault,
            &mut disabled_performance_attribution,
        )?;
        self.evaluate_forward_state(target_forward_output.final_logits(), request_decoder_state)?;
        self.runtime
            .evaluate_arrays(&[target_forward_output.pre_final_normalization_hidden_states()])?;
        Ok(target_forward_output)
    }

    /// Executes a target forward for MTP verification and retains logits for each input row.
    pub fn forward_chunk_with_all_position_logits_and_pre_final_normalization_hidden_states(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
    ) -> Result<Qwen3_5TargetForwardOutput, Qwen3_5ExecutionError> {
        let mut disabled_performance_attribution = PerformanceAttribution::disabled();
        self.forward_chunk_with_all_position_logits_and_pre_final_normalization_hidden_states_and_performance_attribution(
            token_ids,
            starting_position_tokens,
            request_decoder_state,
            &mut disabled_performance_attribution,
        )
        .map(|(target_forward_output, _target_verify_token_ids)| target_forward_output)
    }

    pub(super) fn build_forward_graph(
        &self,
        token_indices: &MlxArray,
        token_count: i32,
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        paged_prefill_execution_mode: Qwen3_5MoEPagedPrefillExecutionMode,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        Ok(self
            .build_target_forward_graph_from_token_indices(
                token_indices,
                token_count,
                starting_position_tokens,
                request_decoder_state,
                paged_prefill_execution_mode,
                performance_attribution,
                false,
            )?
            .final_logits)
    }

    pub(super) fn build_target_forward_graph(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        paged_prefill_execution_mode: Qwen3_5MoEPagedPrefillExecutionMode,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Qwen3_5TargetForwardOutput, Qwen3_5ExecutionError> {
        let token_count = super::forward_contract::validate_forward_input(
            token_ids,
            starting_position_tokens,
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
        self.build_target_forward_graph_from_token_indices(
            &token_indices,
            token_count,
            starting_position_tokens,
            request_decoder_state,
            paged_prefill_execution_mode,
            performance_attribution,
            false,
        )
    }

    pub(super) fn build_target_forward_graph_from_token_indices(
        &self,
        token_indices: &MlxArray,
        token_count: i32,
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        paged_prefill_execution_mode: Qwen3_5MoEPagedPrefillExecutionMode,
        performance_attribution: &mut PerformanceAttribution,
        should_retain_all_position_logits: bool,
    ) -> Result<Qwen3_5TargetForwardOutput, Qwen3_5ExecutionError> {
        let hidden_states = self.embedding_lookup(token_indices)?;
        self.build_target_forward_graph_from_embeddings(
            hidden_states,
            token_count,
            starting_position_tokens,
            request_decoder_state,
            paged_prefill_execution_mode,
            performance_attribution,
            should_retain_all_position_logits,
        )
    }

    pub(super) fn build_forward_graph_from_embeddings(
        &self,
        hidden_states: MlxArray,
        token_count: i32,
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        paged_prefill_execution_mode: Qwen3_5MoEPagedPrefillExecutionMode,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        Ok(self
            .build_target_forward_graph_from_embeddings(
                hidden_states,
                token_count,
                starting_position_tokens,
                request_decoder_state,
                paged_prefill_execution_mode,
                performance_attribution,
                false,
            )?
            .final_logits)
    }

    pub(super) fn build_target_forward_graph_from_embeddings(
        &self,
        mut hidden_states: MlxArray,
        token_count: i32,
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        paged_prefill_execution_mode: Qwen3_5MoEPagedPrefillExecutionMode,
        performance_attribution: &mut PerformanceAttribution,
        should_retain_all_position_logits: bool,
    ) -> Result<Qwen3_5TargetForwardOutput, Qwen3_5ExecutionError> {
        let rope_offset_tokens = i32::try_from(starting_position_tokens).map_err(|_| {
            Qwen3_5ExecutionError::InvalidInput {
                description: "starting position exceeds the MLX int32 range",
            }
        })?;
        let decoder_layer_count = self.config.layer_count() as usize;
        for layer_index in 0..decoder_layer_count {
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
                return Err(super::error::invalid_request_decoder_state(
                    layer_index,
                    "decoder layer tensors must be allocated together or absent together",
                ));
            }
            hidden_states = self.forward_decoder_layer(
                &hidden_states,
                token_count,
                rope_offset_tokens,
                layer_index,
                decoder_layer_weights,
                layer_model_state,
                paged_prefill_execution_mode,
                performance_attribution,
            )?;
            if token_count > 1
                && (layer_index + 1).is_multiple_of(PREFILL_ASYNC_SUBMISSION_LAYER_INTERVAL)
                && layer_index + 1 < decoder_layer_count
            {
                self.runtime.async_eval_arrays(&[&hidden_states])?;
            }
        }

        let (final_logits, all_position_logits) = performance_attribution.measure_operation(
            PerformanceOperation::FinalLogitsGraphConstruction,
            |_performance_attribution| {
                let normalized_states = self.runtime.rms_norm(
                    &hidden_states,
                    &self.weights.final_normalization_weight,
                    f32::from_bits(self.config.rms_norm_epsilon_bits()),
                )?;
                if should_retain_all_position_logits {
                    let target_all_position_logits = self.quantized_linear(
                        &normalized_states,
                        &self.weights.language_model_head_weights,
                    )?;
                    let target_all_position_logits = self
                        .runtime
                        .astype(&target_all_position_logits, MlxDtype::Float32)?;
                    let vocabulary_size = self.config.vocabulary_size() as i32;
                    let final_logits = self.runtime.slice(
                        &target_all_position_logits,
                        &[0, token_count - 1, 0],
                        &[1, token_count, vocabulary_size],
                        &[1, 1, 1],
                    )?;
                    return Ok::<(MlxArray, Option<MlxArray>), Qwen3_5ExecutionError>((
                        final_logits,
                        Some(target_all_position_logits),
                    ));
                }
                let hidden_size = self.config.hidden_size() as i32;
                let final_hidden_state = self.runtime.slice(
                    &normalized_states,
                    &[0, token_count - 1, 0],
                    &[1, token_count, hidden_size],
                    &[1, 1, 1],
                )?;
                let final_logits = self.quantized_linear(
                    &final_hidden_state,
                    &self.weights.language_model_head_weights,
                )?;
                Ok::<(MlxArray, Option<MlxArray>), Qwen3_5ExecutionError>((
                    self.runtime.astype(&final_logits, MlxDtype::Float32)?,
                    None,
                ))
            },
        )?;
        Ok(Qwen3_5TargetForwardOutput {
            final_logits,
            all_position_logits,
            pre_final_normalization_hidden_states: hidden_states,
        })
    }
}

pub(super) fn pre_final_normalization_hidden_state_at(
    runtime: &MlxRuntime,
    pre_final_normalization_hidden_states: &MlxArray,
    token_position_index: i32,
) -> Result<MlxArray, MlxRuntimeError> {
    let hidden_state_shape = pre_final_normalization_hidden_states.shape();
    if hidden_state_shape.len() != 3
        || hidden_state_shape[0] != 1
        || token_position_index < 0
        || token_position_index >= hidden_state_shape[1]
    {
        return Err(MlxRuntimeError::RuntimeOperation {
            operation: "slice Qwen3.5 pre-final hidden state",
            description: "hidden-state row index is outside the target forward output".to_owned(),
        });
    }
    runtime.slice(
        pre_final_normalization_hidden_states,
        &[0, token_position_index, 0],
        &[1, token_position_index + 1, hidden_state_shape[2]],
        &[1, 1, 1],
    )
}
