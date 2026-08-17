//! Incremental Laguna prompt processing so cancel and telemetry can run between chunks.

use std::time::Instant;

use astronomical_ipc_protocol::{ExpertMemoryMode, RequestId};
use astronomical_runtime_integration::MlxRuntime;

use super::execution::LagunaInferenceExecution;
use super::memory::{
    begin_laguna_forward_memory_observation, complete_laguna_forward_memory_observation,
};
use crate::laguna::LagunaModel;
use crate::persistent_prompt_cache_boundary_clamped_prefill_chunck_end;
use crate::{GeneratedToken, InferenceEngineError, MlxRamBudget, PerformanceOperation};

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
        let Some(model) = self.model.as_ref() else {
            return Err(InferenceEngineError::Fatal {
                reason: "the Laguna model is not loaded".to_owned(),
            });
        };
        let mlx_ram_budget = self
            .mlx_ram_budget
            .as_ref()
            .ok_or(InferenceEngineError::Fatal {
                reason: "the Laguna RAM budget is not loaded".to_owned(),
            })?;
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
        let chunk_end_token_position_exclusive = persistent_prompt_cache
            .as_ref()
            .map(|store| {
                persistent_prompt_cache_boundary_clamped_prefill_chunck_end(
                    chunk_start_token_position,
                    requested_chunk_end_token_position_exclusive,
                    store.model_contract.block_token_count(),
                )
            })
            .unwrap_or(requested_chunk_end_token_position_exclusive)
            .max(chunk_start_token_position + 1)
            .min(final_prompt_end_token_position_exclusive);
        let chunk_token_ids = &active_request.prompt_token_ids
            [chunk_start_token_position..chunk_end_token_position_exclusive];
        let prompt_chunk_started_at = Instant::now();
        let is_terminal_prompt_chunk =
            chunk_end_token_position_exclusive == final_prompt_end_token_position_exclusive;
        let (terminal_chunk_logits, forward_elapsed_millis) =
            active_request.performance_attribution.measure_operation(
                PerformanceOperation::PromptPrefillAdvanceSpan,
                |performance_attribution| {
                    forward_one_prompt_chunk(
                        runtime,
                        model,
                        mlx_ram_budget,
                        chunk_token_ids,
                        chunk_start_token_position,
                        chunk_end_token_position_exclusive,
                        is_terminal_prompt_chunk,
                        &mut active_request.decoder_state,
                        performance_attribution,
                    )
                },
            )?;
        if let Some(persistent_prompt_cache) = persistent_prompt_cache.as_deref() {
            super::prompt_cache::capture_completed_cache_blocks(
                runtime,
                persistent_prompt_cache,
                &active_request.prompt_token_ids,
                &active_request.decoder_state,
                chunk_start_token_position,
                chunk_end_token_position_exclusive,
                &mut active_request.last_published_block_key,
                &mut active_request.performance_attribution,
            )?;
        }
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
    model: &LagunaModel,
    mlx_ram_budget: &MlxRamBudget,
    chunk_token_ids: &[u32],
    chunk_start_token_position: usize,
    chunk_end_token_position_exclusive: usize,
    is_terminal_prompt_chunk: bool,
    decoder_state: &mut crate::laguna::LagunaDecoderState,
    performance_attribution: &mut crate::PerformanceAttribution,
) -> Result<(Option<astronomical_runtime_integration::MlxArray>, u64), InferenceEngineError> {
    let chunk_token_array = runtime
        .array_from_u32(
            chunk_token_ids,
            &[i32::try_from(chunk_token_ids.len()).unwrap_or(i32::MAX)],
        )
        .map_err(|_| InferenceEngineError::Fatal {
            reason: "Laguna prompt tokens could not be placed on the runtime".to_owned(),
        })?;
    let memory_baseline =
        begin_laguna_forward_memory_observation(runtime, model, performance_attribution)?;
    let chunk_started_at = Instant::now();
    let chunk_evaluation_root = if is_terminal_prompt_chunk {
        model.forward(
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
        InferenceEngineError::Fatal {
            reason: format!("Laguna prompt processing failed: {forward_error:?}"),
        }
    })?;
    // A progress event is a real execution boundary, not merely evidence that a
    // lazy MLX graph was constructed. Materializing terminal logits here keeps
    // each blocking interval bounded by one selected prompt chunk.
    performance_attribution
        .measure_operation(
            PerformanceOperation::PrefillStateGraphicsProcessorCompletionWait,
            |_performance_attribution| runtime.evaluate_arrays(&[&chunk_evaluation_root]),
        )
        .map_err(|evaluation_error| InferenceEngineError::Fatal {
            reason: format!(
                "Laguna prompt-processing chunk could not be materialized: {evaluation_error}"
            ),
        })?;
    complete_laguna_forward_memory_observation(
        runtime,
        model,
        mlx_ram_budget,
        memory_baseline,
        performance_attribution,
    )?;
    let forward_elapsed_millis = u64::try_from(chunk_started_at.elapsed().as_millis())
        .unwrap_or(u64::MAX)
        .max(1);
    let terminal_chunk_logits = is_terminal_prompt_chunk.then_some(chunk_evaluation_root);
    Ok((terminal_chunk_logits, forward_elapsed_millis))
}
