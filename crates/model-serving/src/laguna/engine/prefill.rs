//! Incremental Laguna prompt processing so cancel and telemetry can run between chunks.

use std::time::Instant;

use astronomical_ipc_protocol::{ExpertMemoryMode, RequestId};
use astronomical_runtime_integration::MlxRuntime;

use super::active_generation::LagunaPrefillRequestCheckpoint;
use super::execution::LagunaInferenceExecution;
use super::memory::complete_laguna_forward_memory_observation;
use crate::laguna::{LagunaModel, laguna_decoder_cache_layout};
use crate::persistent_prompt_cache_boundary_clamped_prefill_chunk_end;
use crate::{
    GeneratedToken, InferenceEngineError, MemoryPhase, MlxRamBudget, PerformanceOperation,
    PersistentPromptCacheBlockKey, PersistentPromptCacheDiskStore,
};

impl LagunaInferenceExecution {
    /// Forwards one remaining prompt chunk and returns visible prefill progress.
    pub(super) fn advance_pending_prompt_prefill(
        &mut self,
        request_id: RequestId,
    ) -> Result<Option<GeneratedToken>, InferenceEngineError> {
        let Some(runtime) = self.runtime.as_ref() else {
            return Err(InferenceEngineError::Fatal {
                reason: "the Laguna runtime is not loaded".to_owned(),
            });
        };
        let Some(model) = self.model.as_mut() else {
            return Err(InferenceEngineError::Fatal {
                reason: "the Laguna model is not loaded".to_owned(),
            });
        };
        let mlx_ram_budget = self
            .mlx_ram_budget
            .as_mut()
            .ok_or(InferenceEngineError::Fatal {
                reason: "the Laguna RAM budget is not loaded".to_owned(),
            })?;
        let adaptive_ram_growth_guard =
            self.adaptive_ram_growth_guard
                .as_mut()
                .ok_or(InferenceEngineError::Fatal {
                    reason: "the Laguna adaptive RAM growth guard is not loaded".to_owned(),
                })?;
        let prefill_failure_injection = &mut self.prefill_failure_injection;
        let active_request =
            self.active_request
                .as_mut()
                .ok_or(InferenceEngineError::InvalidRequest {
                    reason: "no Laguna generation is active".to_owned(),
                })?;
        if active_request.request_id != request_id {
            return Err(InferenceEngineError::InvalidRequest {
                reason: "Laguna generation request identifiers do not match".to_owned(),
            });
        }
        if active_request.next_prompt_token_position >= active_request.prompt_token_ids.len() {
            model.resume_expert_retention_after_request_pressure();
            model.prepare_generation_expert_residency();
            return Ok(None);
        }
        let prompt_processing_chunk_sizer =
            self.prompt_processing_chunk_sizer
                .as_ref()
                .ok_or(InferenceEngineError::Fatal {
                    reason: "Laguna prompt-processing chunk sizer is missing".to_owned(),
                })?;
        let persistent_prompt_cache = self.persistent_prompt_cache.clone();
        let chunk_start_token_position = active_request.next_prompt_token_position;
        let final_prompt_end_token_position_exclusive = active_request.prompt_token_ids.len();
        let requested_chunk_end_token_position_exclusive = prompt_processing_chunk_sizer
            .next_prompt_processing_chunk_end(
                chunk_start_token_position,
                final_prompt_end_token_position_exclusive,
                !matches!(model.expert_memory_mode(), ExpertMemoryMode::Resident),
            );
        let initial_chunk_end_token_position_exclusive = persistent_prompt_cache
            .as_ref()
            .map(|store| {
                persistent_prompt_cache_boundary_clamped_prefill_chunk_end(
                    chunk_start_token_position,
                    requested_chunk_end_token_position_exclusive,
                    store.model_contract.block_token_count(),
                )
            })
            .unwrap_or(requested_chunk_end_token_position_exclusive)
            .max(chunk_start_token_position + 1)
            .min(final_prompt_end_token_position_exclusive);
        let prompt_chunk_started_at = Instant::now();
        let mut attempted_chunk_token_count = initial_chunk_end_token_position_exclusive
            .saturating_sub(chunk_start_token_position)
            .max(1);
        let mut has_retried_same_chunk_after_reclamation = false;
        let (chunk_end_token_position_exclusive, terminal_chunk_logits, forward_elapsed_millis) = loop {
            let attempted_chunk_end_token_position_exclusive = chunk_start_token_position
                .saturating_add(attempted_chunk_token_count)
                .min(final_prompt_end_token_position_exclusive);
            let chunk_token_ids = &active_request.prompt_token_ids
                [chunk_start_token_position..attempted_chunk_end_token_position_exclusive];
            let is_terminal_prompt_chunk = attempted_chunk_end_token_position_exclusive
                == final_prompt_end_token_position_exclusive;
            let inject_prefill_capacity_failure = prefill_failure_injection.take_next_failure();
            let request_checkpoint =
                active_request
                    .prefill_request_checkpoint()
                    .map_err(|checkpoint_error| InferenceEngineError::Fatal {
                        reason: format!(
                            "Laguna could not checkpoint prefill state: {checkpoint_error}"
                        ),
                    })?;
            let attempt = active_request.performance_attribution.measure_operation(
                PerformanceOperation::PromptPrefillAdvanceSpan,
                |performance_attribution| {
                    forward_one_prompt_chunk(
                        runtime,
                        model,
                        adaptive_ram_growth_guard,
                        mlx_ram_budget,
                        chunk_token_ids,
                        chunk_start_token_position,
                        attempted_chunk_end_token_position_exclusive,
                        is_terminal_prompt_chunk,
                        inject_prefill_capacity_failure,
                        persistent_prompt_cache.as_deref(),
                        &active_request.prompt_token_ids,
                        &mut active_request.last_published_block_key,
                        &mut active_request.decoder_state,
                        performance_attribution,
                    )
                },
            );
            match attempt {
                Ok((terminal_chunk_logits, forward_elapsed_millis)) => {
                    break (
                        attempted_chunk_end_token_position_exclusive,
                        terminal_chunk_logits,
                        forward_elapsed_millis,
                    );
                }
                Err(LagunaPrefillAttemptError::Engine(engine_error)) => {
                    active_request
                        .restore_prefill_request_checkpoint(request_checkpoint)
                        .map_err(|restore_error| InferenceEngineError::Fatal {
                            reason: format!(
                                "Laguna could not restore a failed prefill attempt: {restore_error}"
                            ),
                        })?;
                    return Err(engine_error);
                }
                Err(LagunaPrefillAttemptError::Capacity(capacity_error)) => {
                    let LagunaPrefillRequestCheckpoint {
                        decoder_allocation,
                        prompt_cursor,
                        cache_publication_cursor,
                    } = request_checkpoint;
                    active_request.next_prompt_token_position = prompt_cursor;
                    active_request.last_published_block_key = cache_publication_cursor;
                    let should_retry_same_chunk =
                        super::prefill_capacity_recovery::recover_laguna_prefill_capacity(
                            runtime,
                            model,
                            adaptive_ram_growth_guard,
                            &mut active_request.decoder_state,
                            decoder_allocation,
                            capacity_error,
                            has_retried_same_chunk_after_reclamation,
                            &mut active_request.performance_attribution,
                        )?;
                    if should_retry_same_chunk {
                        has_retried_same_chunk_after_reclamation = true;
                        continue;
                    }
                    let Some(smaller_chunk_token_count) =
                            crate::laguna::LagunaPromptProcessingChunkSizer::next_smaller_executable_chunk_size_tokens(
                                attempted_chunk_token_count,
                            )
                        else {
                            return Err(InferenceEngineError::InvalidRequest {
                                reason: "a one-token Laguna prompt chunk cannot fit after expert reclamation"
                                    .to_owned(),
                            });
                        };
                    attempted_chunk_token_count = smaller_chunk_token_count;
                    has_retried_same_chunk_after_reclamation = false;
                }
            }
        };
        if let Some(terminal_chunk_logits) = terminal_chunk_logits {
            active_request.terminal_prompt_logits = Some(terminal_chunk_logits);
        }
        active_request.next_prompt_token_position = chunk_end_token_position_exclusive;
        let prompt_chunk_elapsed_millis =
            u64::try_from(prompt_chunk_started_at.elapsed().as_millis())
                .unwrap_or(u64::MAX)
                .max(1);
        let processed_token_count =
            u32::try_from(chunk_end_token_position_exclusive - chunk_start_token_position)
                .unwrap_or(u32::MAX);
        let prompt_work_reuse = active_request.prompt_work_reuse;
        Ok(Some(GeneratedToken::PrefillProgress {
            processed_token_count,
            elapsed_millis: prompt_chunk_elapsed_millis,
            forward_prefill_chunk_elapsed_millis: forward_elapsed_millis,
            completed_prefill_chunk_tokens: processed_token_count,
            mlx_memory_telemetry: self.collect_current_mlx_memory_telemetry(),
            expert_residency_telemetry: self
                .model
                .as_ref()
                .map(LagunaModel::expert_residency_telemetry),
            speculative_prefill_draft_memory_telemetry: None,
            expert_memory_mode: self.model.as_ref().map(LagunaModel::expert_memory_mode),
            prompt_work_reuse,
            persistent_prompt_cache_diagnostics: None,
        }))
    }
}

fn forward_one_prompt_chunk(
    runtime: &MlxRuntime,
    model: &mut LagunaModel,
    adaptive_ram_growth_guard: &mut crate::AdaptiveRamGrowthGuard,
    mlx_ram_budget: &mut MlxRamBudget,
    chunk_token_ids: &[u32],
    chunk_start_token_position: usize,
    chunk_end_token_position_exclusive: usize,
    is_terminal_prompt_chunk: bool,
    inject_prefill_capacity_failure: bool,
    persistent_prompt_cache: Option<&PersistentPromptCacheDiskStore>,
    prompt_token_ids: &[u32],
    last_published_block_key: &mut Option<PersistentPromptCacheBlockKey>,
    decoder_state: &mut crate::laguna::LagunaDecoderState,
    performance_attribution: &mut crate::PerformanceAttribution,
) -> Result<(Option<astronomical_runtime_integration::MlxArray>, u64), LagunaPrefillAttemptError> {
    let prompt_cache_publication_workspace_bytes = prompt_cache_publication_workspace_bytes(
        model,
        persistent_prompt_cache,
        chunk_start_token_position,
        chunk_end_token_position_exclusive,
    )?;
    let chunk_token_array = runtime
        .array_from_u32(
            chunk_token_ids,
            &[i32::try_from(chunk_token_ids.len()).unwrap_or(i32::MAX)],
        )
        .map_err(|_| InferenceEngineError::Fatal {
            reason: "Laguna prompt tokens could not be placed on the runtime".to_owned(),
        })?;
    let (adaptive_ram_growth_context, memory_baseline) =
        super::memory::admit_laguna_forward_memory(
            runtime,
            model,
            adaptive_ram_growth_guard,
            decoder_state,
            chunk_token_ids.len(),
            prompt_cache_publication_workspace_bytes,
            MemoryPhase::Prefill,
            u64::try_from(chunk_end_token_position_exclusive).unwrap_or(u64::MAX),
            performance_attribution,
        )?;
    let chunk_started_at = Instant::now();
    let chunk_evaluation_root = if is_terminal_prompt_chunk {
        model.forward_prefill(
            runtime,
            &chunk_token_array,
            decoder_state,
            performance_attribution,
        )
    } else {
        model.forward_prompt_chunk_without_logits(
            runtime,
            &chunk_token_array,
            decoder_state,
            performance_attribution,
        )
    }
    .map_err(|forward_error| {
        let expert_cache_statistics = model.expert_weight_memory_cache_statistics();
        tracing::error!(
            ?forward_error,
            chunk_start_token_position,
            chunk_end_token_position_exclusive,
            complete_expert_layer_count = expert_cache_statistics.complete_layer_count,
            complete_expert_payload_bytes =
                expert_cache_statistics.complete_layer_payload_byte_count,
            partial_expert_layer_count = expert_cache_statistics.partial_layer_count,
            partial_expert_payload_bytes = expert_cache_statistics.partial_layer_payload_byte_count,
            retained_expert_ceiling_bytes = model.retained_expert_ceiling_bytes(),
            "Laguna prompt-processing chunk failed"
        );
        if forward_error.is_recoverable_memory_pressure() {
            LagunaPrefillAttemptError::Capacity(forward_error)
        } else {
            LagunaPrefillAttemptError::Engine(InferenceEngineError::Fatal {
                reason: format!("Laguna prompt processing failed: {forward_error:?}"),
            })
        }
    })?;
    if inject_prefill_capacity_failure {
        let memory_snapshot =
            runtime
                .memory_snapshot()
                .map_err(|memory_error| InferenceEngineError::Fatal {
                    reason: format!(
                        "Laguna prefill failure injection could not sample memory: {memory_error}"
                    ),
                })?;
        return Err(LagunaPrefillAttemptError::Capacity(
            crate::laguna::LagunaExecutionError::Runtime(
                astronomical_runtime_integration::MlxRuntimeError::ActiveMemoryLimitExceeded {
                    active_memory_bytes: memory_snapshot.active_memory_bytes(),
                    attempted_allocation_bytes: usize::try_from(model.maximum_expert_page_bytes())
                        .unwrap_or(usize::MAX)
                        .max(1),
                    allowed_active_memory_bytes: memory_snapshot.active_memory_bytes(),
                },
            ),
        ));
    }
    // A progress event is a real execution boundary, not merely evidence that a
    // lazy MLX graph was constructed. Materializing terminal logits here keeps
    // each blocking interval bounded by one selected prompt chunk.
    performance_attribution
        .measure_operation(
            PerformanceOperation::PrefillStateGraphicsProcessorCompletionWait,
            |_performance_attribution| runtime.evaluate_arrays(&[&chunk_evaluation_root]),
        )
        .map_err(|evaluation_error| {
            let execution_error = crate::laguna::LagunaExecutionError::from(evaluation_error);
            if execution_error.is_recoverable_memory_pressure() {
                LagunaPrefillAttemptError::Capacity(execution_error)
            } else {
                LagunaPrefillAttemptError::Engine(InferenceEngineError::Fatal {
                    reason: format!(
                        "Laguna prompt-processing chunk could not be materialized: {execution_error}"
                    ),
                })
            }
        })?;
    if let Some(persistent_prompt_cache) = persistent_prompt_cache {
        super::prompt_cache::capture_completed_cache_blocks(
            runtime,
            persistent_prompt_cache,
            prompt_token_ids,
            decoder_state,
            chunk_start_token_position,
            chunk_end_token_position_exclusive,
            last_published_block_key,
            performance_attribution,
        )
        .map_err(|capture_error| match capture_error {
            super::prompt_cache::LagunaPromptCacheCaptureError::Capacity(capacity_error) => {
                LagunaPrefillAttemptError::Capacity(capacity_error)
            }
            super::prompt_cache::LagunaPromptCacheCaptureError::Engine(engine_error) => {
                LagunaPrefillAttemptError::Engine(engine_error)
            }
        })?;
    }
    complete_laguna_forward_memory_observation(
        runtime,
        model,
        adaptive_ram_growth_guard,
        adaptive_ram_growth_context,
        mlx_ram_budget,
        memory_baseline,
        u64::try_from(chunk_end_token_position_exclusive).unwrap_or(u64::MAX),
        performance_attribution,
    )?;
    let forward_elapsed_millis = u64::try_from(chunk_started_at.elapsed().as_millis())
        .unwrap_or(u64::MAX)
        .max(1);
    let terminal_chunk_logits = is_terminal_prompt_chunk.then_some(chunk_evaluation_root);
    Ok((terminal_chunk_logits, forward_elapsed_millis))
}

fn prompt_cache_publication_workspace_bytes(
    model: &LagunaModel,
    persistent_prompt_cache: Option<&PersistentPromptCacheDiskStore>,
    chunk_start_token_position: usize,
    chunk_end_token_position_exclusive: usize,
) -> Result<usize, InferenceEngineError> {
    let Some(persistent_prompt_cache) = persistent_prompt_cache else {
        return Ok(0);
    };
    let block_token_count = persistent_prompt_cache.model_contract.block_token_count();
    if block_token_count == 0
        || !chunk_end_token_position_exclusive.is_multiple_of(block_token_count)
        || chunk_end_token_position_exclusive.saturating_sub(block_token_count)
            < chunk_start_token_position
    {
        return Ok(0);
    }
    let decoder_cache_layout =
        laguna_decoder_cache_layout(model.contract()).map_err(|layout_error| {
            InferenceEngineError::InvalidRequest {
                reason: format!(
                    "Laguna prompt-cache publication geometry is invalid: {layout_error}"
                ),
            }
        })?;
    let largest_sequence_tensor_bytes = decoder_cache_layout
        .maximum_sequence_tensor_payload_byte_count(block_token_count)
        .map_err(|layout_error| InferenceEngineError::InvalidRequest {
            reason: format!(
                "Laguna prompt-cache sequence publication geometry is invalid: {layout_error}"
            ),
        })?;
    let largest_boundary_tensor_bytes = decoder_cache_layout
        .boundary_tensor_layouts()
        .iter()
        .try_fold(0_usize, |largest_tensor_bytes, persisted_tensor_layout| {
            persisted_tensor_layout
                .tensor_layout()
                .fixed_payload_byte_count()
                .map(|tensor_bytes| largest_tensor_bytes.max(tensor_bytes))
        })
        .map_err(|layout_error| InferenceEngineError::InvalidRequest {
            reason: format!(
                "Laguna prompt-cache boundary publication geometry is invalid: {layout_error}"
            ),
        })?;
    Ok(largest_sequence_tensor_bytes.max(largest_boundary_tensor_bytes))
}

enum LagunaPrefillAttemptError {
    Engine(InferenceEngineError),
    Capacity(crate::laguna::LagunaExecutionError),
}

impl From<InferenceEngineError> for LagunaPrefillAttemptError {
    fn from(error: InferenceEngineError) -> Self {
        Self::Engine(error)
    }
}
