use std::time::Instant;

use crate::{
    AdaptiveRamGrowthContext, PerformanceCounter, PerformanceOperation,
    Qwen3_5PersistentPromptCacheBoundaryCheckpoint,
    persistent_prompt_cache_boundary_completed_prefill_chunck_tokens,
};
use astronomical_ipc_protocol::RequestId;

use super::engine_request::{Qwen3_5EngineRequest, Qwen3_5SpeculativePrefillFailureStageForTests};
use super::prefill_execution_context::Qwen3_5PrefillExecutionContext;
use super::prompt_prefill_errors::{
    PromptPrefillChunckAttemptError, configured_speculative_prefill_execution_error,
    prefill_execution_error, terminal_optional_prefill_error_is_fallback,
};
use super::{
    Qwen3_5EngineState, Qwen3_5SpeculativePrefillChunckMode, fatal_engine_error,
    prompt_prefill_counters::{
        prepare_sparse_target_gpu_inputs, record_sparse_target_and_mode_counters,
    },
    qwen3_5_runtime_error, qwen3_5_selected_speculative_prefill_positions_for_range,
    qwen3_5_speculative_prefill_chunck_mode, qwen3_5_speculative_prefill_sparse_target_is_active,
    speculative_prefill::configured_speculative_prefill_failure,
};
use crate::qwen3_5::multi_token_prediction::{
    execute_terminal_optional_history_capture_with_performance_attribution,
    record_prompt_history_initialization_fallback,
};

pub(super) struct PromptPrefillChunckOutcome {
    pub(super) active_memory_bytes_before_growth: usize,
    pub(super) forward_chunk_elapsed_millis: u64,
    pub(super) adaptive_ram_growth_context: AdaptiveRamGrowthContext,
    pub(super) exact_temporary_workspace_bytes: usize,
    pub(super) boundary_checkpoints: Vec<Qwen3_5PersistentPromptCacheBoundaryCheckpoint>,
    pub(super) speculative_prefill_chunck_mode: Qwen3_5SpeculativePrefillChunckMode,
}

impl Qwen3_5EngineState {
    pub(super) fn execute_prompt_prefill_chunck(
        &mut self,
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
            active_request.has_optional_prediction_session(),
            prefill_end,
            final_prompt_index,
        );
        let speculative_prefill_sparse_conversation_range_is_active =
            qwen3_5_speculative_prefill_sparse_target_is_active(
                active_request.should_use_speculative_prefill,
                prefill_start,
                active_request.ordinary_target_prefill_control_span_token_count,
            );
        // Dense persistent-cache capture and sparse target execution represent
        // different decoder-state contracts and may never share one checkpoint.
        let capture_is_eligible = self.persistent_prompt_cache.is_some()
            && active_request.can_use_persistent_prompt_cache
            && !active_request.has_optional_prediction_session()
            && !speculative_prefill_sparse_conversation_range_is_active;
        // Cache-disabled and request-ineligible paths stop here: they do not
        // plan checkpoint boundaries, derive a synthetic block length, or
        // reserve checkpoint/publication memory. Cache-enabled execution uses
        // the one contract owned by the open store.
        let (all_completed_prefill_chunck_tokens, persistent_prompt_cache_block_token_count) =
            if capture_is_eligible {
                let persistent_prompt_cache_block_token_count = self
                    .persistent_prompt_cache
                    .as_ref()
                    .ok_or_else(|| {
                        fatal_engine_error(
                            "eligible prompt-cache capture has no persistent cache owner",
                        )
                    })?
                    .model_contract_ref()
                    .block_token_count();
                (
                    persistent_prompt_cache_boundary_completed_prefill_chunck_tokens(
                        prefill_start,
                        prefill_end,
                        persistent_prompt_cache_block_token_count,
                    ),
                    Some(persistent_prompt_cache_block_token_count),
                )
            } else {
                (Vec::new(), None)
            };
        let mut intermediate_completed_prefill_chunck_tokens =
            all_completed_prefill_chunck_tokens.clone();
        if intermediate_completed_prefill_chunck_tokens.last().copied() == Some(prefill_token_count)
        {
            intermediate_completed_prefill_chunck_tokens.pop();
        }
        let boundary_checkpoint_workspace_bytes =
            if intermediate_completed_prefill_chunck_tokens.is_empty() {
                0
            } else {
                model
                    .decoder_cache_layout()
                    .boundary_snapshot_payload_byte_count()
                    .map_err(|error| {
                        fatal_engine_error(format!(
                            "failed to project boundary checkpoint workspace: {error}"
                        ))
                    })?
                    .checked_mul(intermediate_completed_prefill_chunck_tokens.len())
                    .ok_or_else(|| {
                        fatal_engine_error("boundary checkpoint workspace bytes overflowed")
                    })?
            };
        let direct_publication_workspace_bytes = if !all_completed_prefill_chunck_tokens.is_empty()
        {
            self.persistent_prompt_cache
                .as_ref()
                .map(|persistent_prompt_cache| {
                    persistent_prompt_cache
                        .model_contract_ref()
                        .direct_publication_workspace_bytes()
                })
                .unwrap_or(0)
        } else {
            0
        };
        let exact_temporary_workspace_bytes = boundary_checkpoint_workspace_bytes
            .checked_add(direct_publication_workspace_bytes)
            .ok_or_else(|| {
                fatal_engine_error("prompt-cache publication workspace bytes overflowed")
            })?;
        active_request
            .performance_attribution
            .record_counter(PerformanceCounter::PrefillChunckCount, 1);
        let selected_speculative_prefill_positions_for_current_chunck =
            if active_request.should_use_speculative_prefill {
                active_request
                    .speculative_prefill_selected_token_positions
                    .as_deref()
                    .map_or_else(Vec::new, |selected_token_positions| {
                        qwen3_5_selected_speculative_prefill_positions_for_range(
                            selected_token_positions,
                            prefill_start,
                            prefill_end,
                        )
                    })
            } else {
                Vec::new()
            };
        // Global selection positions are ascending absolute offsets. Slice them
        // to this logical chunk without changing their original prompt positions.
        let speculative_prefill_target_token_count =
            selected_speculative_prefill_positions_for_current_chunck.len();
        let speculative_prefill_target_is_active =
            speculative_prefill_sparse_conversation_range_is_active
                && !matches!(
                    speculative_prefill_chunck_mode,
                    Qwen3_5SpeculativePrefillChunckMode::TerminalAdditionalHistoryCapture
                );
        // Terminal optional-history capture needs dense target hidden rows, so it
        // deliberately overrides sparse execution for that one final prefill chunk.
        let additional_persistent_state_growth_bytes = match (
            speculative_prefill_chunck_mode,
            active_request.optional_prediction_session(),
        ) {
            (
                Qwen3_5SpeculativePrefillChunckMode::TerminalAdditionalHistoryCapture,
                Some(optional_prediction_session),
            ) if active_request.visual_embeddings.is_none() => {
                let additional_full_attention_bytes_per_layer_token = model
                    .config()
                    .full_attention_key_value_state_bytes_per_layer_token()
                    .ok_or_else(|| {
                        fatal_engine_error(
                            "additional full-attention bytes per layer token overflowed",
                        )
                    })?;
                optional_prediction_session
                    .projected_full_attention_growth_bytes(
                        additional_full_attention_bytes_per_layer_token,
                        prefill_token_count,
                    )
                    .map_err(qwen3_5_runtime_error)?
            }
            _ => 0,
        };
        let adaptive_ram_growth_context = AdaptiveRamGrowthContext::prefill(
            speculative_prefill_target_token_count,
            self.prefill_chunck_sizer
                .prompt_processing_context_identifier(
                    prefill_start,
                    Qwen3_5PrefillExecutionContext::new(
                        active_request.visual_embeddings.is_some(),
                        active_request.has_optional_prediction_session(),
                        model.sparse_experts_are_paged(),
                        self.persistent_prompt_cache.is_some()
                            && active_request.can_use_persistent_prompt_cache
                            && !active_request.has_optional_prediction_session(),
                    )
                    .with_target_only_prefix(matches!(
                        speculative_prefill_chunck_mode,
                        Qwen3_5SpeculativePrefillChunckMode::TargetOnlyPrefix
                    ))
                    .with_speculative_prefill_sparse_target(speculative_prefill_target_is_active),
                ),
            active_request.visual_embeddings.is_some(),
            active_request.has_optional_prediction_session(),
            model.sparse_experts_are_paged(),
        );
        let target_expert_payload_bytes_before_context_admission = model
            .expert_weight_memory_cache_statistics()
            .resident_payload_byte_count;
        // Admission has mutable access because this exact chunk can trigger a
        // Resident -> Paged transition. Reborrow the model afterward so graph
        // construction cannot accidentally retain the pre-transition owner.
        let active_memory_bytes_before_growth = self.measure_adaptive_ram_growth_memory_admission(
            adaptive_ram_growth_context,
            &mut active_request.performance_attribution,
            &active_request.request_decoder_state,
            additional_persistent_state_growth_bytes,
            exact_temporary_workspace_bytes,
        )?;
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let target_expert_payload_bytes_after_context_admission = model
            .expert_weight_memory_cache_statistics()
            .resident_payload_byte_count;
        let target_expert_payload_bytes_reclaimed_during_context_admission =
            target_expert_payload_bytes_before_context_admission
                .saturating_sub(target_expert_payload_bytes_after_context_admission);
        if direct_publication_workspace_bytes > 0
            && let Some(persistent_prompt_cache_diagnostics) =
                active_request.persistent_prompt_cache_diagnostics.as_mut()
        {
            persistent_prompt_cache_diagnostics.expert_bytes_reclaimed_for_publication =
                persistent_prompt_cache_diagnostics
                    .expert_bytes_reclaimed_for_publication
                    .saturating_add(target_expert_payload_bytes_reclaimed_during_context_admission);
        }
        if speculative_prefill_target_is_active {
            active_request.performance_attribution.record_counter(
                PerformanceCounter::SpeculativePrefillContextTargetExpertReclaimedPayloadBytes,
                target_expert_payload_bytes_reclaimed_during_context_admission,
            );
        }
        let prefill_request_checkpoint = active_request
            .prefill_request_checkpoint()
            .map_err(qwen3_5_runtime_error)?;
        let forward_chunck_started_at = Instant::now();
        let mut boundary_checkpoints = Vec::new();
        let mut terminal_history_token_count = 0;
        if speculative_prefill_target_is_active {
            if !selected_speculative_prefill_positions_for_current_chunck.is_empty() {
                // A sparse logical chunk may contain zero selected rows. In that
                // case advancing the logical cursor is correct and no empty MLX
                // forward should be submitted.
                let sparse_target_gpu_inputs = if active_request
                    .take_forced_speculative_prefill_failure_for_tests(
                        Qwen3_5SpeculativePrefillFailureStageForTests::SparseTargetInputAssembly,
                    ) {
                    Err(configured_speculative_prefill_failure(
                        active_request.request_id,
                        "sparse target input assembly",
                        "forced speculative-prefill sparse input assembly failure",
                    ))
                } else {
                    prepare_sparse_target_gpu_inputs(
                        active_request,
                        model,
                        &selected_speculative_prefill_positions_for_current_chunck,
                        speculative_prefill_target_token_count,
                    )
                }
                .map_err(|sparse_input_assembly_error| {
                    configured_speculative_prefill_failure(
                        active_request.request_id,
                        "sparse target input assembly",
                        sparse_input_assembly_error,
                    )
                })?;
                let should_force_sparse_target_active_memory_limit_rejection = active_request
                    .take_forced_speculative_prefill_failure_for_tests(
                        Qwen3_5SpeculativePrefillFailureStageForTests::SparseTargetActiveMemoryLimitRejection,
                    );
                let should_force_sparse_target_execution_failure = active_request
                    .take_forced_speculative_prefill_failure_for_tests(
                        Qwen3_5SpeculativePrefillFailureStageForTests::SparseTargetExecution,
                    );
                let sparse_target_forward_result = (|| {
                    if should_force_sparse_target_active_memory_limit_rejection {
                        return Err(crate::Qwen3_5ExecutionError::Runtime(
                            astronomical_runtime_integration::MlxRuntimeError::ActiveMemoryLimitExceeded {
                                active_memory_bytes: 2,
                                attempted_allocation_bytes: 2,
                                allowed_active_memory_bytes: 3,
                            },
                        ));
                    }
                    if should_force_sparse_target_execution_failure {
                        return Err(crate::Qwen3_5ExecutionError::InvalidInput {
                            description: "forced speculative-prefill sparse target execution failure",
                        });
                    }
                    active_request.performance_attribution.measure_operation(
                        PerformanceOperation::SpeculativePrefillSparseTargetForward,
                        |performance_attribution| {
                            if let Some(visual_embeddings) = active_request.visual_embeddings.as_ref() {
                                // Selected token IDs are compact, but explicit
                                // offsets preserve original rotary positions and
                                // image rows preserve visual consumption order.
                                model
                                    .prefill_chunck_with_speculative_prefill_gpu_token_indices_and_visual_embeddings_and_position_offsets_and_performance_attribution(
                                        &sparse_target_gpu_inputs.selected_token_indices_on_gpu,
                                        &sparse_target_gpu_inputs.selected_prompt_token_ids,
                                        active_request.next_position_tokens,
                                        &sparse_target_gpu_inputs.selected_prompt_position_offsets,
                                        visual_embeddings,
                                        active_request.consumed_visual_embedding_count,
                                        active_request.image_pad_token_id,
                                        &mut active_request.request_decoder_state,
                                        performance_attribution,
                                    )
                                    .map(|consumed_visual_embedding_count| {
                                        active_request.consumed_visual_embedding_count = active_request
                                            .consumed_visual_embedding_count
                                            .saturating_add(consumed_visual_embedding_count);
                                    })
                            } else {
                                 model
                                     .prefill_chunck_with_speculative_prefill_gpu_token_indices_and_position_offsets_and_performance_attribution(
                                         &sparse_target_gpu_inputs.selected_token_indices_on_gpu,
                                         sparse_target_gpu_inputs.selected_token_count_i32,
                                         active_request.next_position_tokens,
                                         &sparse_target_gpu_inputs.selected_prompt_position_offsets,
                                         &mut active_request.request_decoder_state,
                                         performance_attribution,
                                     )
                                    .map(|_| ())
                            }
                        },
                    )
                })();
                if let Err(qwen3_5_execution_error) = sparse_target_forward_result {
                    return Err(configured_speculative_prefill_execution_error(
                        active_request.request_id,
                        "sparse target execution",
                        qwen3_5_execution_error,
                        prefill_request_checkpoint,
                    ));
                }
            }
        }
        if !speculative_prefill_target_is_active {
            if let Some(visual_embeddings) = active_request.visual_embeddings.as_ref() {
                let visual_prefill_outcome = if intermediate_completed_prefill_chunck_tokens
                    .is_empty()
                {
                    model
                        .prefill_chunck_with_visual_embeddings_and_performance_attribution(
                            &active_request.input_token_ids[prefill_start..prefill_end],
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
                    let persistent_prompt_cache_block_token_count =
                        persistent_prompt_cache_block_token_count.ok_or_else(|| {
                            fatal_engine_error(
                                "planned visual prompt-cache checkpoints have no block contract",
                            )
                        })?;
                    model
                    .prefill_chunck_with_visual_embeddings_and_boundary_checkpoints_with_performance_attribution(
                    &active_request.input_token_ids[prefill_start..prefill_end],
                    active_request.next_position_tokens,
                    visual_embeddings,
                    active_request.consumed_visual_embedding_count,
                    &mut active_request.request_decoder_state,
                    active_request.image_pad_token_id,
                    intermediate_completed_prefill_chunck_tokens.clone(),
                    persistent_prompt_cache_block_token_count,
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
                Qwen3_5SpeculativePrefillChunckMode::TerminalAdditionalHistoryCapture
            ) {
                let optional_history_capture_result =
                    execute_terminal_optional_history_capture_with_performance_attribution(
                        model,
                        prefill_start,
                        prefill_end,
                        active_request,
                    );
                match optional_history_capture_result {
                    Ok(history_token_count) => terminal_history_token_count = history_token_count,
                    Err(optional_history_capture_error) => {
                        if !terminal_optional_prefill_error_is_fallback(
                            &optional_history_capture_error,
                        ) {
                            return Err(prefill_execution_error(
                                optional_history_capture_error,
                                prefill_request_checkpoint,
                            ));
                        }
                        tracing::warn!(
                            request_id = request_id.value(),
                            error = %optional_history_capture_error,
                            "optional terminal history initialization failed; continuing target-only"
                        );
                        active_request.clear_optional_prediction_session();
                        record_prompt_history_initialization_fallback(active_request);
                    }
                }
            } else {
                let text_prefill_outcome =
                    if intermediate_completed_prefill_chunck_tokens.is_empty() {
                        model
                            .prefill_chunck_with_performance_attribution(
                                &active_request.input_token_ids[prefill_start..prefill_end],
                                active_request.next_position_tokens,
                                &mut active_request.request_decoder_state,
                                &mut active_request.performance_attribution,
                            )
                            .map(|()| Vec::new())
                    } else {
                        let persistent_prompt_cache_block_token_count =
                            persistent_prompt_cache_block_token_count.ok_or_else(|| {
                                fatal_engine_error(
                                    "planned text prompt-cache checkpoints have no block contract",
                                )
                            })?;
                        model
                            .prefill_chunck_with_boundary_checkpoints_and_performance_attribution(
                                &active_request.input_token_ids[prefill_start..prefill_end],
                                active_request.next_position_tokens,
                                &mut active_request.request_decoder_state,
                                intermediate_completed_prefill_chunck_tokens,
                                persistent_prompt_cache_block_token_count,
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
        }
        if std::mem::take(&mut active_request.force_next_prefill_capacity_rejection_for_tests) {
            return Err(PromptPrefillChunckAttemptError::ActiveMemoryLimitExceeded {
                active_memory_bytes: 1,
                attempted_allocation_bytes: 1,
                allowed_active_memory_bytes: 1,
                prefill_request_checkpoint,
            });
        }
        record_sparse_target_and_mode_counters(
            active_request,
            model,
            speculative_prefill_target_is_active,
            speculative_prefill_target_token_count,
            speculative_prefill_chunck_mode,
            prefill_token_count,
            &all_completed_prefill_chunck_tokens,
            terminal_history_token_count,
            &mut boundary_checkpoints,
        )?;
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
