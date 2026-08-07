use std::time::Instant;

use astronomical_ipc_protocol::RequestId;
use astronomical_runtime_integration::MlxRuntimeError;

use crate::{
    AdaptiveRamGrowthContext, InferenceEngineError, PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT,
    PerformanceCounter, PerformanceOperation, Qwen3_5ExecutionError,
    Qwen3_5PersistentPromptCacheBoundaryCheckpoint,
    persistent_prompt_cache_boundary_completed_prefill_chunck_tokens,
};

use super::engine_request::{Qwen3_5EngineRequest, Qwen3_5PrefillRequestCheckpoint};
use super::memory_admission::AdaptiveRamGrowthMemoryAdmissionError;
use super::prefill_execution_context::Qwen3_5PrefillExecutionContext;
use super::{
    Qwen3_5EngineState, Qwen3_5SpeculativePrefillChunckMode, fatal_engine_error,
    qwen3_5_runtime_error, qwen3_5_speculative_prefill_chunck_mode,
};

pub(super) enum PromptPrefillChunckAttemptError {
    AdaptiveMemoryLimitExceeded {
        reason: String,
    },
    ActiveMemoryLimitExceeded {
        active_memory_bytes: usize,
        attempted_allocation_bytes: usize,
        allowed_active_memory_bytes: usize,
        prefill_request_checkpoint: Qwen3_5PrefillRequestCheckpoint,
    },
    GraphicsProcessorMemoryExhausted {
        reason: String,
        prefill_request_checkpoint: Qwen3_5PrefillRequestCheckpoint,
    },
    Engine(InferenceEngineError),
}

pub(super) struct PromptPrefillChunckOutcome {
    pub(super) active_memory_bytes_before_growth: usize,
    pub(super) forward_chunk_elapsed_millis: u64,
    pub(super) adaptive_ram_growth_context: AdaptiveRamGrowthContext,
    pub(super) exact_temporary_workspace_bytes: usize,
    pub(super) boundary_checkpoints: Vec<Qwen3_5PersistentPromptCacheBoundaryCheckpoint>,
    pub(super) speculative_prefill_chunck_mode: Qwen3_5SpeculativePrefillChunckMode,
}

impl From<InferenceEngineError> for PromptPrefillChunckAttemptError {
    fn from(inference_engine_error: InferenceEngineError) -> Self {
        Self::Engine(inference_engine_error)
    }
}

impl From<AdaptiveRamGrowthMemoryAdmissionError> for PromptPrefillChunckAttemptError {
    fn from(admission_error: AdaptiveRamGrowthMemoryAdmissionError) -> Self {
        match admission_error {
            AdaptiveRamGrowthMemoryAdmissionError::InsufficientCapacity { reason } => {
                Self::AdaptiveMemoryLimitExceeded { reason }
            }
            AdaptiveRamGrowthMemoryAdmissionError::Engine(inference_engine_error) => {
                Self::Engine(inference_engine_error)
            }
        }
    }
}

impl Qwen3_5EngineState {
    pub(super) fn execute_prompt_prefill_chunck(
        &self,
        request_id: RequestId,
        active_request: &mut Qwen3_5EngineRequest,
        prefill_start: usize,
        prefill_end: usize,
    ) -> Result<PromptPrefillChunckOutcome, PromptPrefillChunckAttemptError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let prefill_token_count = prefill_end - prefill_start;
        let final_prompt_index = active_request
            .input_token_ids
            .len()
            .checked_sub(1)
            .ok_or_else(|| fatal_engine_error("generation prompt must not be empty"))?;
        let speculative_prefill_chunck_mode = qwen3_5_speculative_prefill_chunck_mode(
            active_request.mtp_request_state.is_some(),
            prefill_end,
            final_prompt_index,
        );
        let capture_is_eligible = self.persistent_prompt_cache.is_some()
            && active_request.can_use_persistent_prompt_cache
            && !active_request.persistent_prompt_cache_capture_has_stopped
            && active_request.mtp_request_state.is_none();
        let planned_completed_prefill_chunck_tokens = if capture_is_eligible {
            persistent_prompt_cache_boundary_completed_prefill_chunck_tokens(
                prefill_start,
                prefill_end,
            )
        } else {
            Vec::new()
        };
        let projected_single_capture_tensor_payload_bytes = model
            .decoder_cache_layout()
            .persistent_prompt_cache_block_payload_byte_count(
                PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT,
            )
            .map_err(|error| {
                fatal_engine_error(format!(
                    "failed to project prompt-cache capture bytes: {error}"
                ))
            })?;
        let writer_can_accept_projected_captures = self
            .persistent_prompt_cache_write_queue
            .as_ref()
            .is_some_and(|persistent_prompt_cache_write_queue| {
                persistent_prompt_cache_write_queue.can_accept_projected_captures(
                    projected_single_capture_tensor_payload_bytes,
                    planned_completed_prefill_chunck_tokens.len(),
                )
            });
        let capture_is_active = capture_is_eligible && writer_can_accept_projected_captures;
        if capture_is_eligible
            && !planned_completed_prefill_chunck_tokens.is_empty()
            && !writer_can_accept_projected_captures
        {
            active_request.persistent_prompt_cache_capture_has_stopped = true;
            tracing::info!(
                "persistent prompt-cache capture stopped before forward because projected boundaries could not be accepted"
            );
        }
        let all_completed_prefill_chunck_tokens = if capture_is_active {
            planned_completed_prefill_chunck_tokens
        } else {
            Vec::new()
        };
        let mut intermediate_completed_prefill_chunck_tokens =
            all_completed_prefill_chunck_tokens.clone();
        if intermediate_completed_prefill_chunck_tokens.last().copied() == Some(prefill_token_count)
        {
            intermediate_completed_prefill_chunck_tokens.pop();
        }
        let exact_temporary_workspace_bytes = model
            .decoder_cache_layout()
            .boundary_snapshot_payload_byte_count()
            .map_err(|error| {
                fatal_engine_error(format!(
                    "failed to project boundary checkpoint workspace: {error}"
                ))
            })?
            .checked_mul(intermediate_completed_prefill_chunck_tokens.len())
            .ok_or_else(|| fatal_engine_error("boundary checkpoint workspace bytes overflowed"))?;
        active_request
            .performance_attribution
            .record_counter(PerformanceCounter::PrefillChunckCount, 1);
        let prefill_token_ids = &active_request.input_token_ids[prefill_start..prefill_end];
        let mtp_full_attention_growth_bytes = match (
            speculative_prefill_chunck_mode,
            active_request.mtp_request_state.as_ref(),
        ) {
            (Qwen3_5SpeculativePrefillChunckMode::TerminalMtpCapture, Some(mtp_request_state))
                if active_request.visual_embeddings.is_none() =>
            {
                let mtp_full_attention_bytes_per_layer_token = model
                    .config()
                    .full_attention_key_value_state_bytes_per_layer_token()
                    .ok_or_else(|| {
                        fatal_engine_error("MTP full-attention bytes per layer token overflowed")
                    })?;
                mtp_request_state
                    .projected_capacity_growth_bytes(
                        mtp_full_attention_bytes_per_layer_token,
                        prefill_token_count,
                    )
                    .map_err(qwen3_5_runtime_error)?
            }
            _ => 0,
        };
        let adaptive_ram_growth_context = AdaptiveRamGrowthContext::prefill(
            prefill_token_count,
            self.prefill_chunck_sizer
                .prompt_processing_context_identifier(
                    prefill_start,
                    Qwen3_5PrefillExecutionContext::new(
                        active_request.visual_embeddings.is_some(),
                        active_request.mtp_request_state.is_some(),
                        model.sparse_experts_are_paged(),
                        self.persistent_prompt_cache.is_some()
                            && active_request.can_use_persistent_prompt_cache
                            && !active_request.persistent_prompt_cache_capture_has_stopped
                            && active_request.mtp_request_state.is_none(),
                    )
                    .with_target_only_mtp_prefix(matches!(
                        speculative_prefill_chunck_mode,
                        Qwen3_5SpeculativePrefillChunckMode::TargetOnlyMtpPrefix
                    )),
                ),
            active_request.visual_embeddings.is_some(),
            active_request.mtp_request_state.is_some(),
            model.sparse_experts_are_paged(),
        );
        let active_memory_bytes_before_growth = self.measure_adaptive_ram_growth_memory_admission(
            adaptive_ram_growth_context,
            &mut active_request.performance_attribution,
            &active_request.request_decoder_state,
            mtp_full_attention_growth_bytes,
            exact_temporary_workspace_bytes,
        )?;
        let prefill_request_checkpoint = active_request
            .prefill_request_checkpoint()
            .map_err(qwen3_5_runtime_error)?;
        let forward_chunck_started_at = Instant::now();
        let mut boundary_checkpoints = Vec::new();
        let mut terminal_mtp_history_token_count = 0;
        if let Some(visual_embeddings) = active_request.visual_embeddings.as_ref() {
            let visual_prefill_outcome = if intermediate_completed_prefill_chunck_tokens.is_empty()
            {
                model
                    .prefill_chunck_with_visual_embeddings_and_performance_attribution(
                        prefill_token_ids,
                        active_request.next_position_tokens,
                        visual_embeddings,
                        active_request.consumed_visual_embedding_count,
                        &mut active_request.request_decoder_state,
                        active_request.image_pad_token_id,
                        &mut active_request.performance_attribution,
                    )
                    .map(|consumed_visual_embedding_count| {
                        (consumed_visual_embedding_count, Vec::new())
                    })
            } else {
                model
                    .prefill_chunck_with_visual_embeddings_and_boundary_checkpoints_with_performance_attribution(
                    prefill_token_ids,
                    active_request.next_position_tokens,
                    visual_embeddings,
                    active_request.consumed_visual_embedding_count,
                    &mut active_request.request_decoder_state,
                    active_request.image_pad_token_id,
                    intermediate_completed_prefill_chunck_tokens.clone(),
                    PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT,
                    &mut active_request.performance_attribution,
                )
                    .map(|checkpoint_outcome| {
                        (
                            checkpoint_outcome.consumed_visual_embedding_count,
                            checkpoint_outcome.boundary_checkpoints,
                        )
                    })
            };
            let (consumed_visual_embedding_count, visual_boundary_checkpoints) =
                match visual_prefill_outcome {
                    Ok(visual_prefill_outcome) => visual_prefill_outcome,
                    Err(qwen3_5_execution_error) => {
                        return Err(prefill_execution_error(
                            qwen3_5_execution_error,
                            prefill_request_checkpoint,
                        ));
                    }
                };
            boundary_checkpoints = visual_boundary_checkpoints;
            active_request.consumed_visual_embedding_count += consumed_visual_embedding_count;
        } else if matches!(
            speculative_prefill_chunck_mode,
            Qwen3_5SpeculativePrefillChunckMode::TerminalMtpCapture
        ) {
            let target_prefill_output = match model
                .forward_chunk_with_pre_final_normalization_hidden_states_and_performance_attribution(
                    prefill_token_ids,
                    active_request.next_position_tokens,
                    &mut active_request.request_decoder_state,
                    &mut active_request.performance_attribution,
                )
            {
                Ok(target_prefill_output) => target_prefill_output,
                Err(qwen3_5_execution_error) => {
                    return Err(prefill_execution_error(
                        qwen3_5_execution_error,
                        prefill_request_checkpoint,
                    ));
                }
            };
            let shifted_prompt_start = prefill_start
                .checked_add(1)
                .ok_or_else(|| fatal_engine_error("shifted MTP prompt start overflowed"))?;
            let shifted_prompt_end = prefill_end
                .checked_add(1)
                .ok_or_else(|| fatal_engine_error("shifted MTP prompt end overflowed"))?;
            let shifted_prompt_token_ids = active_request
                .input_token_ids
                .get(shifted_prompt_start..shifted_prompt_end)
                .ok_or_else(|| fatal_engine_error("shifted MTP prompt range was invalid"))?;
            let mtp_prefill_started_at = active_request
                .performance_attribution
                .begin_operation_span();
            let mtp_prefill_result = model
                .prefill_mtp_history_from_token_ids_with_performance_attribution(
                    target_prefill_output.pre_final_normalization_hidden_states(),
                    shifted_prompt_token_ids,
                    active_request
                        .mtp_request_state
                        .as_mut()
                        .ok_or_else(|| fatal_engine_error("MTP request state disappeared"))?,
                    &mut active_request.performance_attribution,
                );
            active_request
                .performance_attribution
                .complete_operation_span(
                    PerformanceOperation::MtpPromptHistoryInitializationSpan,
                    mtp_prefill_started_at,
                );
            match mtp_prefill_result {
                Ok(()) => {
                    terminal_mtp_history_token_count = shifted_prompt_token_ids.len();
                }
                Err(mtp_prefill_error) => {
                    if !terminal_mtp_prefill_error_is_optional_fallback(&mtp_prefill_error) {
                        return Err(prefill_execution_error(
                            mtp_prefill_error,
                            prefill_request_checkpoint,
                        ));
                    }
                    tracing::warn!(
                        request_id = request_id.value(),
                        error = %mtp_prefill_error,
                        "optional terminal MTP prompt-history initialization failed; continuing target-only"
                    );
                    active_request.mtp_request_state = None;
                    active_request.mtp_target_hidden_states = None;
                    active_request.performance_attribution.record_counter(
                        PerformanceCounter::MtpPromptHistoryInitializationFallbackCount,
                        1,
                    );
                }
            }
        } else {
            let text_prefill_outcome = if intermediate_completed_prefill_chunck_tokens.is_empty() {
                model
                    .prefill_chunck_with_performance_attribution(
                        prefill_token_ids,
                        active_request.next_position_tokens,
                        &mut active_request.request_decoder_state,
                        &mut active_request.performance_attribution,
                    )
                    .map(|()| Vec::new())
            } else {
                model
                    .prefill_chunck_with_boundary_checkpoints_and_performance_attribution(
                        prefill_token_ids,
                        active_request.next_position_tokens,
                        &mut active_request.request_decoder_state,
                        intermediate_completed_prefill_chunck_tokens,
                        PERSISTENT_PROMPT_CACHE_BLOCK_TOKEN_COUNT,
                        &mut active_request.performance_attribution,
                    )
                    .map(|checkpoint_outcome| checkpoint_outcome.boundary_checkpoints)
            };
            boundary_checkpoints = match text_prefill_outcome {
                Ok(boundary_checkpoints) => boundary_checkpoints,
                Err(qwen3_5_execution_error) => {
                    return Err(prefill_execution_error(
                        qwen3_5_execution_error,
                        prefill_request_checkpoint,
                    ));
                }
            };
        }
        if std::mem::take(&mut active_request.force_next_prefill_capacity_rejection_for_tests) {
            return Err(PromptPrefillChunckAttemptError::ActiveMemoryLimitExceeded {
                active_memory_bytes: 1,
                attempted_allocation_bytes: 1,
                allowed_active_memory_bytes: 1,
                prefill_request_checkpoint,
            });
        }
        if all_completed_prefill_chunck_tokens.last().copied() == Some(prefill_token_count) {
            let recurrent_snapshot_tensors = active_request
                .request_decoder_state
                .extract_persistent_prompt_cache_recurrent_snapshot_tensors();
            match recurrent_snapshot_tensors {
                Ok(recurrent_snapshot_tensors) => {
                    boundary_checkpoints.push(Qwen3_5PersistentPromptCacheBoundaryCheckpoint {
                        completed_prefill_chunck_tokens: prefill_token_count,
                        recurrent_snapshot_tensors,
                    });
                }
                Err(error) => {
                    tracing::warn!(%error, "final prompt-cache boundary extraction failed");
                    boundary_checkpoints.clear();
                    active_request.persistent_prompt_cache_capture_has_stopped = true;
                }
            }
        }
        match speculative_prefill_chunck_mode {
            Qwen3_5SpeculativePrefillChunckMode::TargetOnlyMtpPrefix => {
                active_request.performance_attribution.record_counter(
                    PerformanceCounter::SpeculativePrefillTargetOnlyPrefixChunckCount,
                    1,
                );
                active_request.performance_attribution.record_counter(
                    PerformanceCounter::SpeculativePrefillTargetOnlyPrefixTokenCount,
                    u64::try_from(prefill_token_count).unwrap_or(u64::MAX),
                );
            }
            Qwen3_5SpeculativePrefillChunckMode::TerminalMtpCapture => {
                active_request.performance_attribution.record_counter(
                    PerformanceCounter::SpeculativePrefillTerminalCaptureChunckCount,
                    1,
                );
                active_request.performance_attribution.record_counter(
                    PerformanceCounter::SpeculativePrefillTerminalMtpHistoryTokenCount,
                    u64::try_from(terminal_mtp_history_token_count).unwrap_or(u64::MAX),
                );
            }
            Qwen3_5SpeculativePrefillChunckMode::OrdinaryTarget => {}
        }
        Ok(PromptPrefillChunckOutcome {
            active_memory_bytes_before_growth,
            forward_chunk_elapsed_millis: forward_chunck_started_at.elapsed().as_millis() as u64,
            adaptive_ram_growth_context: adaptive_ram_growth_context
                .with_sparse_experts_are_paged(model.sparse_experts_are_paged()),
            exact_temporary_workspace_bytes,
            boundary_checkpoints,
            speculative_prefill_chunck_mode,
        })
    }
}

fn terminal_mtp_prefill_error_is_optional_fallback(
    qwen3_5_execution_error: &Qwen3_5ExecutionError,
) -> bool {
    match qwen3_5_execution_error {
        Qwen3_5ExecutionError::Runtime(mlx_runtime_error) => {
            matches!(mlx_runtime_error, MlxRuntimeError::RuntimeOperation { .. })
                && !mlx_runtime_error.is_recoverable_graphics_processor_out_of_memory()
        }
        Qwen3_5ExecutionError::ExpertPaging(_) => true,
        Qwen3_5ExecutionError::Artifact(_)
        | Qwen3_5ExecutionError::MissingTensor { .. }
        | Qwen3_5ExecutionError::InvalidTensor { .. }
        | Qwen3_5ExecutionError::MissingQuantization { .. }
        | Qwen3_5ExecutionError::UnassignedTensor { .. }
        | Qwen3_5ExecutionError::TypedTensorCountMismatch { .. }
        | Qwen3_5ExecutionError::MissingDecoderLayerWeights { .. }
        | Qwen3_5ExecutionError::TensorPayloadMismatch { .. }
        | Qwen3_5ExecutionError::InvalidInput { .. }
        | Qwen3_5ExecutionError::InvalidDecoderCacheLayout { .. }
        | Qwen3_5ExecutionError::DecoderLayerCountMismatch { .. }
        | Qwen3_5ExecutionError::InvalidRequestDecoderState { .. } => false,
    }
}

fn prefill_execution_error(
    qwen3_5_execution_error: Qwen3_5ExecutionError,
    prefill_request_checkpoint: Qwen3_5PrefillRequestCheckpoint,
) -> PromptPrefillChunckAttemptError {
    match qwen3_5_execution_error {
        Qwen3_5ExecutionError::Runtime(MlxRuntimeError::ActiveMemoryLimitExceeded {
            active_memory_bytes,
            attempted_allocation_bytes,
            allowed_active_memory_bytes,
        }) => PromptPrefillChunckAttemptError::ActiveMemoryLimitExceeded {
            active_memory_bytes,
            attempted_allocation_bytes,
            allowed_active_memory_bytes,
            prefill_request_checkpoint,
        },
        Qwen3_5ExecutionError::Runtime(mlx_runtime_error)
            if mlx_runtime_error.is_recoverable_graphics_processor_out_of_memory() =>
        {
            PromptPrefillChunckAttemptError::GraphicsProcessorMemoryExhausted {
                reason: mlx_runtime_error.to_string(),
                prefill_request_checkpoint,
            }
        }
        other_qwen3_5_execution_error => {
            PromptPrefillChunckAttemptError::Engine(other_qwen3_5_execution_error.into())
        }
    }
}
