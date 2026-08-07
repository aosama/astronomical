//! Request-owned performance-attribution forwarding for Qwen3.5 MLX graphs.

use astronomical_runtime_integration::{MlxArray, MlxDtype};

use crate::qwen3_5_moe::Qwen3_5MoEPagedPrefillExecutionMode;
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
        self.evaluate_decoder_state_with_performance_attribution(
            request_decoder_state,
            performance_attribution,
        )?;
        Ok(consumed_visual_embedding_count)
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
        let mut boundary_checkpoint_collector =
            Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector::new(
                completed_prefill_chunck_tokens,
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
        self.evaluate_decoder_state_and_boundary_checkpoints_with_performance_attribution(
            request_decoder_state,
            &boundary_checkpoint_collector,
            performance_attribution,
        )?;
        Ok(Qwen3_5BoundaryCheckpointPrefillOutcome {
            consumed_visual_embedding_count,
            boundary_checkpoints: boundary_checkpoint_collector.complete()?,
        })
    }

    pub(crate) fn prefill_chunck_with_performance_attribution(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<(), Qwen3_5ExecutionError> {
        drop(self.build_forward_chunk_with_performance_attribution(
            token_ids,
            starting_position_tokens,
            request_decoder_state,
            performance_attribution,
        )?);
        self.evaluate_decoder_state_with_performance_attribution(
            request_decoder_state,
            performance_attribution,
        )
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
        let mut boundary_checkpoint_collector =
            Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector::new(
                completed_prefill_chunck_tokens,
                self.decoder_cache_layout.boundary_tensor_count(),
                checkpoint_interval_token_count,
            )?;
        drop(self.build_target_forward_graph(
            token_ids,
            starting_position_tokens,
            request_decoder_state,
            Some(&mut boundary_checkpoint_collector),
            Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault,
            performance_attribution,
        )?);
        self.evaluate_decoder_state_and_boundary_checkpoints_with_performance_attribution(
            request_decoder_state,
            &boundary_checkpoint_collector,
            performance_attribution,
        )?;
        Ok(Qwen3_5BoundaryCheckpointPrefillOutcome {
            consumed_visual_embedding_count: 0,
            boundary_checkpoints: boundary_checkpoint_collector.complete()?,
        })
    }

    pub(crate) fn forward_chunk_with_performance_attribution(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let final_logits = self.build_forward_chunk_with_performance_attribution(
            token_ids,
            starting_position_tokens,
            request_decoder_state,
            performance_attribution,
        )?;
        self.evaluate_forward_state(&final_logits, request_decoder_state)?;
        Ok(final_logits)
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
        let target_forward_output = self.build_target_forward_graph(
            token_ids,
            starting_position_tokens,
            request_decoder_state,
            None,
            Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault,
            performance_attribution,
        )?;
        let synchronize_target_forward_output =
            |_performance_attribution: &mut PerformanceAttribution| -> Result<
                (),
                Qwen3_5ExecutionError,
            > {
                self.evaluate_forward_state(
                    target_forward_output.final_logits(),
                    request_decoder_state,
                )?;
                self.runtime.evaluate_arrays(&[
                    target_forward_output.pre_final_normalization_hidden_states(),
                ])?;
                Ok(())
            };
        match synchronization_operation {
            Some(synchronization_operation) => performance_attribution
                .measure_operation(synchronization_operation, synchronize_target_forward_output)?,
            None => synchronize_target_forward_output(performance_attribution)?,
        }
        Ok(target_forward_output)
    }

    pub(crate) fn forward_depth_one_mtp_verification_with_performance_attribution(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<
        (
            Qwen3_5TargetForwardOutput,
            Vec<u32>,
            Qwen3_5PersistentPromptCacheBoundaryCheckpoint,
        ),
        Qwen3_5ExecutionError,
    > {
        if token_ids.len() != 2 {
            return Err(Qwen3_5ExecutionError::InvalidInput {
                description: "depth-one MTP verification requires exactly two target tokens",
            });
        }
        let token_count = validate_forward_input(
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
        let recurrent_boundary_tensor_count = self.decoder_cache_layout.boundary_tensor_count();
        let mut verified_prefix_boundary_checkpoint_collector =
            if recurrent_boundary_tensor_count == 0 {
                None
            } else {
                Some(
                    Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector::new(
                        vec![1],
                        recurrent_boundary_tensor_count,
                        1,
                    )?,
                )
            };
        let target_forward_output = self.build_target_forward_graph_from_token_indices(
            &token_indices,
            token_count,
            starting_position_tokens,
            request_decoder_state,
            verified_prefix_boundary_checkpoint_collector.as_mut(),
            Qwen3_5MoEPagedPrefillExecutionMode::ProductionDecodeVerification,
            performance_attribution,
            true,
        )?;
        let all_position_logits = target_forward_output.all_position_logits().ok_or(
            Qwen3_5ExecutionError::InvalidInput {
                description: "target verification forward did not retain all-position logits",
            },
        )?;
        let target_verify_token_indices = self.build_greedy_token(all_position_logits)?;
        let target_verify_token_ids = performance_attribution.measure_operation(
            PerformanceOperation::MtpTargetVerificationSynchronizationWait,
            |_performance_attribution| -> Result<Vec<u32>, Qwen3_5ExecutionError> {
                let mut target_verification_evaluation_arrays =
                    super::forward_contract::forward_state_arrays(
                        &target_verify_token_indices,
                        request_decoder_state,
                    )?;
                target_verification_evaluation_arrays
                    .push(target_forward_output.pre_final_normalization_hidden_states());
                if let Some(verified_prefix_boundary_checkpoint_collector) =
                    verified_prefix_boundary_checkpoint_collector.as_ref()
                {
                    target_verification_evaluation_arrays
                        .extend(verified_prefix_boundary_checkpoint_collector.evaluation_arrays());
                }
                self.runtime
                    .evaluate_arrays(&target_verification_evaluation_arrays)?;
                Ok(target_verify_token_indices.to_vec_u32()?)
            },
        )?;
        let verified_prefix_boundary_checkpoint =
            match verified_prefix_boundary_checkpoint_collector {
                Some(verified_prefix_boundary_checkpoint_collector) => {
                    let mut verified_prefix_boundary_checkpoints =
                        verified_prefix_boundary_checkpoint_collector.complete()?;
                    let verified_prefix_boundary_checkpoint =
                        verified_prefix_boundary_checkpoints.pop().ok_or(
                            Qwen3_5ExecutionError::InvalidInput {
                                description:
                                    "MTP target verification did not retain its first-row boundary",
                            },
                        )?;
                    if !verified_prefix_boundary_checkpoints.is_empty() {
                        return Err(Qwen3_5ExecutionError::InvalidInput {
                            description: "MTP target verification retained unexpected extra boundaries",
                        });
                    }
                    verified_prefix_boundary_checkpoint
                }
                None => Qwen3_5PersistentPromptCacheBoundaryCheckpoint {
                    completed_prefill_chunck_tokens: 1,
                    recurrent_snapshot_tensors: std::collections::HashMap::new(),
                },
            };
        Ok((
            target_forward_output,
            target_verify_token_ids,
            verified_prefix_boundary_checkpoint,
        ))
    }

    pub(crate) fn build_forward_chunk_with_performance_attribution(
        &self,
        token_ids: &[u32],
        starting_position_tokens: u32,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        self.build_forward_chunk_with_paged_prefill_execution_mode_and_performance_attribution(
            token_ids,
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
