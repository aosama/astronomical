use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};

use crate::qwen3_5_streaming::Qwen3_5MoEPagedPrefillExecutionMode;
use crate::{PerformanceAttribution, PerformanceOperation};

use super::Qwen3_5Model;
use super::{Qwen3_5ExecutionError, RequestDecoderStateStack};
use crate::qwen3_5::decoder::Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector;
use astronomical_mlx_c_rust::{MlxArray, MlxDtype};

/// Target-model graph outputs retained for specialized consumers.
pub struct Qwen3_5TargetForwardOutput {
    final_logits: MlxArray,
    pre_final_normalization_hidden_states: MlxArray,
}

impl Qwen3_5TargetForwardOutput {
    /// Returns float32 logits for the final target position.
    #[must_use]
    pub fn final_logits(&self) -> &MlxArray {
        &self.final_logits
    }

    /// Returns every final decoder row before the trunk's final RMS normalization.
    #[must_use]
    pub fn pre_final_normalization_hidden_states(&self) -> &MlxArray {
        &self.pre_final_normalization_hidden_states
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
    /// Executes a target forward and retains pre-final-normalization rows for specialized consumers.
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
            None,
            Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault,
            &mut disabled_performance_attribution,
        )?;
        self.evaluate_forward_state(target_forward_output.final_logits(), request_decoder_state)?;
        self.runtime
            .evaluate_arrays(&[target_forward_output.pre_final_normalization_hidden_states()])?;
        Ok(target_forward_output)
    }

    pub(crate) fn build_forward_graph(
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
                None,
                paged_prefill_execution_mode,
                performance_attribution,
            )?
            .final_logits)
    }

    /// Builds decoder-state prefill without the vocabulary head.
    ///
    /// Intermediate chunks only need cache tensors; no `lm_head` runs until
    /// sampling. Building then dropping logits was host-side waste on every
    /// 2,048-token chunk.
    pub(super) fn build_prefill_decoder_state_graph(
        &self,
        token_indices: &MlxArray,
        token_count: i32,
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        boundary_checkpoint_collector: Option<
            &mut Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector,
        >,
        paged_prefill_execution_mode: Qwen3_5MoEPagedPrefillExecutionMode,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), Qwen3_5ExecutionError> {
        drop(
            self.build_target_forward_graph_from_token_indices_with_position_offsets(
                token_indices,
                token_count,
                starting_position_tokens,
                None,
                request_decoder_state,
                boundary_checkpoint_collector,
                paged_prefill_execution_mode,
                performance_attribution,
                false,
            )?,
        );
        Ok(())
    }

    pub(super) fn build_target_forward_graph(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        boundary_checkpoint_collector: Option<
            &mut Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector,
        >,
        paged_prefill_execution_mode: Qwen3_5MoEPagedPrefillExecutionMode,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Qwen3_5TargetForwardOutput, Qwen3_5ExecutionError> {
        self.build_target_forward_graph_with_position_offsets(
            token_ids,
            starting_position_tokens,
            None,
            request_decoder_state,
            boundary_checkpoint_collector,
            paged_prefill_execution_mode,
            performance_attribution,
        )
    }

    pub(super) fn build_target_forward_graph_with_position_offsets(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        token_position_offsets: Option<&MlxArray>,
        request_decoder_state: &mut RequestDecoderStateStack,
        boundary_checkpoint_collector: Option<
            &mut Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector,
        >,
        paged_prefill_execution_mode: Qwen3_5MoEPagedPrefillExecutionMode,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Qwen3_5TargetForwardOutput, Qwen3_5ExecutionError> {
        let token_count = super::forward_contract::validate_forward_input(
            token_ids,
            starting_position_tokens,
            token_position_offsets,
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
        self.build_target_forward_graph_from_token_indices_with_position_offsets(
            &token_indices,
            token_count,
            starting_position_tokens,
            token_position_offsets,
            request_decoder_state,
            boundary_checkpoint_collector,
            paged_prefill_execution_mode,
            performance_attribution,
            true,
        )
    }

    pub(crate) fn build_target_forward_graph_from_token_indices(
        &self,
        token_indices: &MlxArray,
        token_count: i32,
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        boundary_checkpoint_collector: Option<
            &mut Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector,
        >,
        paged_prefill_execution_mode: Qwen3_5MoEPagedPrefillExecutionMode,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Qwen3_5TargetForwardOutput, Qwen3_5ExecutionError> {
        self.build_target_forward_graph_from_token_indices_with_position_offsets(
            token_indices,
            token_count,
            starting_position_tokens,
            None,
            request_decoder_state,
            boundary_checkpoint_collector,
            paged_prefill_execution_mode,
            performance_attribution,
            true,
        )
    }

    pub(super) fn build_target_forward_graph_from_token_indices_with_position_offsets(
        &self,
        token_indices: &MlxArray,
        token_count: i32,
        starting_position_tokens: u32,
        token_position_offsets: Option<&MlxArray>,
        request_decoder_state: &mut RequestDecoderStateStack,
        boundary_checkpoint_collector: Option<
            &mut Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector,
        >,
        paged_prefill_execution_mode: Qwen3_5MoEPagedPrefillExecutionMode,
        performance_attribution: &mut PerformanceAttribution,
        should_materialize_vocabulary_logits: bool,
    ) -> Result<Qwen3_5TargetForwardOutput, Qwen3_5ExecutionError> {
        let hidden_states = self.embedding_lookup(token_indices)?;
        self.build_target_forward_graph_from_embeddings_with_position_offsets(
            hidden_states,
            token_count,
            starting_position_tokens,
            token_position_offsets,
            request_decoder_state,
            boundary_checkpoint_collector,
            paged_prefill_execution_mode,
            performance_attribution,
            should_materialize_vocabulary_logits,
        )
    }

    pub(super) fn build_forward_graph_from_embeddings(
        &self,
        hidden_states: MlxArray,
        token_count: i32,
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        boundary_checkpoint_collector: Option<
            &mut Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector,
        >,
        paged_prefill_execution_mode: Qwen3_5MoEPagedPrefillExecutionMode,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        Ok(self
            .build_target_forward_graph_from_embeddings(
                hidden_states,
                token_count,
                starting_position_tokens,
                request_decoder_state,
                boundary_checkpoint_collector,
                paged_prefill_execution_mode,
                performance_attribution,
            )?
            .final_logits)
    }

    pub(super) fn build_target_forward_graph_from_embeddings(
        &self,
        hidden_states: MlxArray,
        token_count: i32,
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        boundary_checkpoint_collector: Option<
            &mut Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector,
        >,
        paged_prefill_execution_mode: Qwen3_5MoEPagedPrefillExecutionMode,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Qwen3_5TargetForwardOutput, Qwen3_5ExecutionError> {
        self.build_target_forward_graph_from_embeddings_with_position_offsets(
            hidden_states,
            token_count,
            starting_position_tokens,
            None,
            request_decoder_state,
            boundary_checkpoint_collector,
            paged_prefill_execution_mode,
            performance_attribution,
            true,
        )
    }

    pub(super) fn build_target_forward_graph_from_embeddings_with_position_offsets(
        &self,
        mut hidden_states: MlxArray,
        token_count: i32,
        starting_position_tokens: u32,
        token_position_offsets: Option<&MlxArray>,
        request_decoder_state: &mut RequestDecoderStateStack,
        mut boundary_checkpoint_collector: Option<
            &mut Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector,
        >,
        paged_prefill_execution_mode: Qwen3_5MoEPagedPrefillExecutionMode,
        performance_attribution: &mut PerformanceAttribution,
        should_materialize_vocabulary_logits: bool,
    ) -> Result<Qwen3_5TargetForwardOutput, Qwen3_5ExecutionError> {
        // Each forward owns one deferred missing-route collection. Clear any
        // leftover roots from a cancelled or failed previous attempt.
        self.clear_paged_forward_missing_route_roots();
        self.route_observation
            .borrow_mut()
            .discard_pending_layer_route_arrays();
        let rope_offset_tokens = i32::try_from(starting_position_tokens).map_err(|_| {
            Qwen3_5ExecutionError::InvalidInput {
                description: "starting position exceeds the MLX int32 range",
            }
        })?;
        let decoder_layer_count = self.config.layer_count() as usize;
        // Multi-token prefill may submit completed layer groups so macOS can
        // schedule other graphics work between command buffers. Paged experts
        // additionally benefit because completed streamed pages can detach.
        let graph_submission_layer_interval =
            usize::try_from(astronomical_ipc_protocol::graph_submission_layer_interval(
                token_count,
                self.sparse_experts_are_paged(),
                self.chunking.prefill_graph_submission_layer_interval,
                self.chunking
                    .experimental_ssd_paging_prefill_graph_submission_layer_interval,
                self.chunking
                    .experimental_ssd_paging_generation_graph_submission_layer_interval,
            ))
            .unwrap_or(0);
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
                token_position_offsets,
                boundary_checkpoint_collector.as_deref_mut(),
                paged_prefill_execution_mode,
                performance_attribution,
            )?;
            if graph_submission_layer_interval > 0
                && (layer_index + 1).is_multiple_of(graph_submission_layer_interval)
                && layer_index + 1 < decoder_layer_count
            {
                // Intermediate mlx_async_eval commits the completed layer group.
                // Do not submit after the final decoder layer: final
                // normalization and logits extend that same graph and the
                // caller owns the terminal evaluation.
                performance_attribution.measure_operation(
                    PerformanceOperation::PrefillStateAsyncEvaluationSubmission,
                    |_performance_attribution| self.runtime.async_eval_arrays(&[&hidden_states]),
                )?;
            }
            // Do not start a second complete-layer host read here. The positional
            // reader is synchronous on this thread, so a prefetch stalls GPU
            // submission and holds an unbudgeted extra complete layer.
        }

        if !should_materialize_vocabulary_logits {
            return Ok(Qwen3_5TargetForwardOutput {
                final_logits: hidden_states
                    .retain()
                    .map_err(Qwen3_5ExecutionError::from)?,
                pre_final_normalization_hidden_states: hidden_states,
            });
        }
        let final_logits = performance_attribution.measure_operation(
            PerformanceOperation::FinalLogitsGraphConstruction,
            |_performance_attribution| {
                let normalized_states = self.runtime.rms_norm(
                    &hidden_states,
                    &self.weights.final_normalization_weight,
                    f32::from_bits(self.config.rms_norm_epsilon_bits()),
                )?;
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
                Ok::<MlxArray, Qwen3_5ExecutionError>(
                    self.runtime.astype(&final_logits, MlxDtype::Float32)?,
                )
            },
        )?;
        Ok(Qwen3_5TargetForwardOutput {
            final_logits,
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
    Ok(runtime.slice(
        pre_final_normalization_hidden_states,
        &[0, token_position_index, 0],
        &[1, token_position_index + 1, hidden_state_shape[2]],
        &[1, 1, 1],
    )?)
}
