//! Request-owned performance-attribution forwarding for Qwen3.5 MLX graphs.

use astronomical_runtime_integration::{MlxArray, MlxDtype};

use crate::qwen3_5_moe::{PagedRouteValidationOutcome, Qwen3_5MoEPagedPrefillExecutionMode};
use crate::{PerformanceAttribution, PerformanceOperation};

use super::forward_contract::{validate_forward_input, validate_generated_token_forward};
use super::model::Qwen3_5Model;
use super::visual_embedding_injection::qwen3_5_inject_visual_embeddings;
use super::{Qwen3_5ExecutionError, Qwen3_5TargetForwardOutput, RequestDecoderStateStack};
use crate::qwen3_5::decoder::{
    Qwen3_5PersistentPromptCacheBoundaryCheckpoint,
    Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector,
};

pub(crate) struct Qwen3_5BoundaryCheckpointPrefillOutcome {
    pub(crate) consumed_visual_embedding_count: usize,
    pub(crate) boundary_checkpoints: Vec<Qwen3_5PersistentPromptCacheBoundaryCheckpoint>,
}

/// Outcome of a terminal prefill that produces both boundary checkpoints and logits.
///
/// Unlike intermediate checkpointed prefill, the terminal chunk needs the vocabulary
/// head to produce first-token logits. This single-forward outcome carries both the
/// cache checkpoints and the logits needed for first-token seeding.
pub(crate) struct Qwen3_5TerminalCheckpointPrefillOutcome {
    pub(crate) target_forward_output: Qwen3_5TargetForwardOutput,
    pub(crate) boundary_checkpoints: Vec<Qwen3_5PersistentPromptCacheBoundaryCheckpoint>,
}

impl Qwen3_5Model {
    // Visual-prefill inputs stay explicit rather than introducing a parameter facade.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefill_chunck_with_visual_embeddings_and_performance_attribution(
        &self,
        chunk_token_ids: &[u32],
        starting_position_tokens: u32,
        visual_embeddings: &MlxArray,
        starting_visual_embedding_index: usize,
        request_decoder_state: &mut RequestDecoderStateStack,
        image_pad_token_id: u32,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<usize, Qwen3_5ExecutionError> {
        let maximum_paged_route_replay_attempts = 1;
        for _paged_route_replay_attempt in 0..maximum_paged_route_replay_attempts {
            let consumed_visual_embedding_count = self.build_visual_prefill_graph(
                chunk_token_ids,
                starting_position_tokens,
                visual_embeddings,
                starting_visual_embedding_index,
                request_decoder_state,
                image_pad_token_id,
                None,
                performance_attribution,
            )?;
            match self.evaluate_decoder_state_for_paged_route_resolution(
                request_decoder_state,
                None,
                performance_attribution,
            )? {
                PagedRouteValidationOutcome::CompleteHit => {
                    return Ok(consumed_visual_embedding_count);
                }
            }
        }
        Err(Qwen3_5ExecutionError::InvalidInput {
            description: "paged route replay exceeded the sparse-layer safety bound",
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn build_visual_prefill_graph(
        &self,
        chunk_token_ids: &[u32],
        starting_position_tokens: u32,
        visual_embeddings: &MlxArray,
        starting_visual_embedding_index: usize,
        request_decoder_state: &mut RequestDecoderStateStack,
        image_pad_token_id: u32,
        boundary_checkpoint_collector: Option<
            &mut Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector,
        >,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<usize, Qwen3_5ExecutionError> {
        let token_count = validate_forward_input(
            chunk_token_ids,
            starting_position_tokens,
            None,
            request_decoder_state.layer_count(),
            self.config.layer_count() as usize,
            self.config.vocabulary_size(),
            self.config.maximum_position_count(),
        )?;
        let signed_token_ids = chunk_token_ids
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
        let text_embeddings = self.embedding_lookup(&token_indices)?;
        let (injected_embeddings, consumed_visual_embedding_count) =
            qwen3_5_inject_visual_embeddings(
                &self.runtime,
                &text_embeddings,
                chunk_token_ids,
                visual_embeddings,
                starting_visual_embedding_index,
                image_pad_token_id,
            )?;
        drop(self.build_forward_graph_from_embeddings(
            injected_embeddings,
            token_count,
            starting_position_tokens,
            request_decoder_state,
            boundary_checkpoint_collector,
            Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault,
            performance_attribution,
        )?);
        Ok(consumed_visual_embedding_count)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefill_chunck_with_visual_embeddings_and_boundary_checkpoints_with_performance_attribution(
        &self,
        chunk_token_ids: &[u32],
        starting_position_tokens: u32,
        visual_embeddings: &MlxArray,
        starting_visual_embedding_index: usize,
        request_decoder_state: &mut RequestDecoderStateStack,
        image_pad_token_id: u32,
        completed_prefill_chunck_tokens: Vec<usize>,
        checkpoint_interval_token_count: usize,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Qwen3_5BoundaryCheckpointPrefillOutcome, Qwen3_5ExecutionError> {
        let maximum_paged_route_replay_attempts = 1;
        for _paged_route_replay_attempt in 0..maximum_paged_route_replay_attempts {
            let mut boundary_checkpoint_collector =
                Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector::new(
                    completed_prefill_chunck_tokens.clone(),
                    self.decoder_cache_layout.boundary_tensor_count(),
                    checkpoint_interval_token_count,
                )?;
            let consumed_visual_embedding_count = self.build_visual_prefill_graph(
                chunk_token_ids,
                starting_position_tokens,
                visual_embeddings,
                starting_visual_embedding_index,
                request_decoder_state,
                image_pad_token_id,
                Some(&mut boundary_checkpoint_collector),
                performance_attribution,
            )?;
            match self.evaluate_decoder_state_for_paged_route_resolution(
                request_decoder_state,
                Some(&boundary_checkpoint_collector),
                performance_attribution,
            )? {
                PagedRouteValidationOutcome::CompleteHit => {
                    return Ok(Qwen3_5BoundaryCheckpointPrefillOutcome {
                        consumed_visual_embedding_count,
                        boundary_checkpoints: boundary_checkpoint_collector.complete()?,
                    });
                }
            }
        }
        Err(Qwen3_5ExecutionError::InvalidInput {
            description: "paged route replay exceeded the sparse-layer safety bound",
        })
    }

    pub(crate) fn prefill_chunck_with_performance_attribution(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), Qwen3_5ExecutionError> {
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
            .array_from_i32(&signed_token_ids, &[1, token_ids.len() as i32])?;
        let maximum_paged_route_replay_attempts = 1;
        for _paged_route_replay_attempt in 0..maximum_paged_route_replay_attempts {
            let graph_started_at = std::time::Instant::now();
            self.build_prefill_decoder_state_graph(
                &token_indices,
                token_ids.len() as i32,
                starting_position_tokens,
                request_decoder_state,
                None,
                Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault,
                performance_attribution,
            )?;
            let graph_elapsed = graph_started_at.elapsed();
            let eval_started_at = std::time::Instant::now();
            match self.evaluate_decoder_state_for_paged_route_resolution(
                request_decoder_state,
                None,
                performance_attribution,
            )? {
                PagedRouteValidationOutcome::CompleteHit => {
                    let eval_elapsed = eval_started_at.elapsed();
                    if graph_elapsed + eval_elapsed > std::time::Duration::from_secs(5) {
                        tracing::info!(
                            token_count = token_ids.len(),
                            graph_elapsed_millis = graph_elapsed.as_millis(),
                            eval_elapsed_millis = eval_elapsed.as_millis(),
                            "slow prefill chunk"
                        );
                    }
                    return Ok(());
                }
            }
        }
        Err(Qwen3_5ExecutionError::InvalidInput {
            description: "paged route replay exceeded the sparse-layer safety bound",
        })
    }

    pub(crate) fn prefill_chunck_with_boundary_checkpoints_and_performance_attribution(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        completed_prefill_chunck_tokens: Vec<usize>,
        checkpoint_interval_token_count: usize,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Qwen3_5BoundaryCheckpointPrefillOutcome, Qwen3_5ExecutionError> {
        let maximum_paged_route_replay_attempts = 1;
        for _paged_route_replay_attempt in 0..maximum_paged_route_replay_attempts {
            let mut boundary_checkpoint_collector =
                Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector::new(
                    completed_prefill_chunck_tokens.clone(),
                    self.decoder_cache_layout.boundary_tensor_count(),
                    checkpoint_interval_token_count,
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
                .array_from_i32(&signed_token_ids, &[1, token_ids.len() as i32])?;
            self.build_prefill_decoder_state_graph(
                &token_indices,
                token_ids.len() as i32,
                starting_position_tokens,
                request_decoder_state,
                Some(&mut boundary_checkpoint_collector),
                Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault,
                performance_attribution,
            )?;
            match self.evaluate_decoder_state_for_paged_route_resolution(
                request_decoder_state,
                Some(&boundary_checkpoint_collector),
                performance_attribution,
            )? {
                PagedRouteValidationOutcome::CompleteHit => {
                    return Ok(Qwen3_5BoundaryCheckpointPrefillOutcome {
                        consumed_visual_embedding_count: 0,
                        boundary_checkpoints: boundary_checkpoint_collector.complete()?,
                    });
                }
            }
        }
        Err(Qwen3_5ExecutionError::InvalidInput {
            description: "paged route replay exceeded the sparse-layer safety bound",
        })
    }

    /// Terminal prefill with boundary checkpoints and vocabulary logits.
    ///
    /// Unlike intermediate checkpointed prefill, the terminal chunk needs the
    /// vocabulary head to produce first-token logits. This builds the full forward
    /// graph (with lm_head) while also collecting boundary checkpoints, so the
    /// terminal chunk is a single forward pass instead of a prefix+tail split.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn terminal_prefill_chunck_with_boundary_checkpoints_and_performance_attribution(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        completed_prefill_chunck_tokens: Vec<usize>,
        checkpoint_interval_token_count: usize,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Qwen3_5TerminalCheckpointPrefillOutcome, Qwen3_5ExecutionError> {
        let maximum_paged_route_replay_attempts = 1;
        for _paged_route_replay_attempt in 0..maximum_paged_route_replay_attempts {
            let mut boundary_checkpoint_collector =
                Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector::new(
                    completed_prefill_chunck_tokens.clone(),
                    self.decoder_cache_layout.boundary_tensor_count(),
                    checkpoint_interval_token_count,
                )?;
            let target_forward_output = self.build_target_forward_graph(
                token_ids,
                starting_position_tokens,
                request_decoder_state,
                Some(&mut boundary_checkpoint_collector),
                Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault,
                performance_attribution,
            )?;
            match self.evaluate_terminal_prefill_state_with_performance_attribution(
                target_forward_output.final_logits(),
                request_decoder_state,
                Some(&boundary_checkpoint_collector),
                performance_attribution,
            )? {
                PagedRouteValidationOutcome::CompleteHit => {
                    return Ok(Qwen3_5TerminalCheckpointPrefillOutcome {
                        target_forward_output,
                        boundary_checkpoints: boundary_checkpoint_collector.complete()?,
                    });
                }
            }
        }
        Err(Qwen3_5ExecutionError::InvalidInput {
            description: "paged route replay exceeded the sparse-layer safety bound",
        })
    }

    pub(crate) fn forward_chunk_with_performance_attribution(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
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
            .array_from_i32(&signed_token_ids, &[1, token_ids.len() as i32])?;
        let maximum_paged_route_replay_attempts = 1;
        for _paged_route_replay_attempt in 0..maximum_paged_route_replay_attempts {
            let final_logits = self.build_forward_graph(
                &token_indices,
                token_ids.len() as i32,
                starting_position_tokens,
                request_decoder_state,
                Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault,
                performance_attribution,
            )?;
            match self.evaluate_forward_state_with_performance_attribution(
                &final_logits,
                request_decoder_state,
                performance_attribution,
            )? {
                PagedRouteValidationOutcome::CompleteHit => return Ok(final_logits),
            }
        }
        Err(Qwen3_5ExecutionError::InvalidInput {
            description: "paged route replay exceeded the sparse-layer safety bound",
        })
    }

    pub(crate) fn forward_chunk_with_pre_final_normalization_hidden_states_and_performance_attribution(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Qwen3_5TargetForwardOutput, Qwen3_5ExecutionError> {
        self.forward_chunk_with_pre_final_normalization_hidden_states_and_synchronization_attribution(
            token_ids,
            starting_position_tokens,
            request_decoder_state,
            performance_attribution,
            None,
        )
    }

    fn forward_chunk_with_pre_final_normalization_hidden_states_and_synchronization_attribution(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
        synchronization_operation: Option<PerformanceOperation>,
    ) -> Result<Qwen3_5TargetForwardOutput, Qwen3_5ExecutionError> {
        let maximum_paged_route_replay_attempts = 1;
        for _paged_route_replay_attempt in 0..maximum_paged_route_replay_attempts {
            let target_forward_output = self.build_target_forward_graph(
                token_ids,
                starting_position_tokens,
                request_decoder_state,
                None,
                Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault,
                performance_attribution,
            )?;
            let synchronize_target_forward_output =
                |performance_attribution: &mut PerformanceAttribution| -> Result<
                    PagedRouteValidationOutcome,
                    Qwen3_5ExecutionError,
                > {
                    let mut evaluation_arrays = super::forward_contract::forward_state_arrays(
                        target_forward_output.final_logits(),
                        request_decoder_state,
                    )?;
                    evaluation_arrays
                        .push(target_forward_output.pre_final_normalization_hidden_states());
                    self.evaluate_arrays_resolving_paged_routes(
                        &evaluation_arrays,
                        performance_attribution,
                    )
                };
            let paged_route_validation_outcome = match synchronization_operation {
                Some(synchronization_operation) => performance_attribution.measure_operation(
                    synchronization_operation,
                    synchronize_target_forward_output,
                )?,
                None => synchronize_target_forward_output(performance_attribution)?,
            };
            match paged_route_validation_outcome {
                PagedRouteValidationOutcome::CompleteHit => return Ok(target_forward_output),
            }
        }
        Err(Qwen3_5ExecutionError::InvalidInput {
            description: "paged route replay exceeded the sparse-layer safety bound",
        })
    }

    pub(crate) fn build_forward_chunk_with_performance_attribution(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
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
            .array_from_i32(&signed_token_ids, &[1, token_ids.len() as i32])?;
        self.build_forward_graph(
            &token_indices,
            token_ids.len() as i32,
            starting_position_tokens,
            request_decoder_state,
            Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault,
            performance_attribution,
        )
    }

    pub(crate) fn build_generated_token_forward_with_performance_attribution(
        &self,
        generated_token: &MlxArray,
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        validate_generated_token_forward(
            generated_token,
            starting_position_tokens,
            request_decoder_state.layer_count(),
            self.config.layer_count() as usize,
            self.config.maximum_position_count(),
        )?;
        let token_indices = self.runtime.astype(generated_token, MlxDtype::Int32)?;
        // Paged decode must resolve deferred missing-route bitmaps before the
        // generated token becomes request-visible. Use a synchronous completion
        // root with exact replay instead of decode-ahead async evaluation alone.
        if self.sparse_experts_are_paged() {
            let maximum_paged_route_replay_attempts = 1;
            for _paged_route_replay_attempt in 0..maximum_paged_route_replay_attempts {
                let next_logits = self.build_forward_graph(
                    &token_indices,
                    1,
                    starting_position_tokens,
                    request_decoder_state,
                    Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault,
                    performance_attribution,
                )?;
                match self.evaluate_forward_state_with_performance_attribution(
                    &next_logits,
                    request_decoder_state,
                    performance_attribution,
                )? {
                    PagedRouteValidationOutcome::CompleteHit => return Ok(next_logits),
                }
            }
            return Err(Qwen3_5ExecutionError::InvalidInput {
                description: "paged route replay exceeded the sparse-layer safety bound",
            });
        }
        self.build_forward_graph(
            &token_indices,
            1,
            starting_position_tokens,
            request_decoder_state,
            Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault,
            performance_attribution,
        )
    }

    pub(crate) fn generated_token_forward_with_pre_final_normalization_hidden_states_and_performance_attribution(
        &self,
        generated_token: &MlxArray,
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Qwen3_5TargetForwardOutput, Qwen3_5ExecutionError> {
        validate_generated_token_forward(
            generated_token,
            starting_position_tokens,
            request_decoder_state.layer_count(),
            self.config.layer_count() as usize,
            self.config.maximum_position_count(),
        )?;
        let token_indices = self.runtime.astype(generated_token, MlxDtype::Int32)?;
        let target_forward_output = self.build_target_forward_graph_from_token_indices(
            &token_indices,
            1,
            starting_position_tokens,
            request_decoder_state,
            None,
            Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault,
            performance_attribution,
            false,
        )?;
        self.evaluate_forward_state(target_forward_output.final_logits(), request_decoder_state)?;
        self.runtime
            .evaluate_arrays(&[target_forward_output.pre_final_normalization_hidden_states()])?;
        Ok(target_forward_output)
    }
}
