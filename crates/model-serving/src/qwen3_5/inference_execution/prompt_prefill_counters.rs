use crate::{PerformanceCounter, Qwen3_5PersistentPromptCacheBoundaryCheckpoint};

use super::engine_request::Qwen3_5EngineRequest;
use super::speculative_prefill_failure::configured_speculative_prefill_failure;
use super::{Qwen3_5SpeculativePrefillChunckMode, fatal_engine_error, qwen3_5_runtime_error};
use crate::qwen3_5::Qwen3_5Model;
use crate::qwen3_5::multi_token_prediction::record_terminal_history_token_count;

pub(super) struct SparseTargetGpuInputs {
    pub(super) selected_token_indices_on_gpu: astronomical_runtime_integration::MlxArray,
    pub(super) selected_prompt_token_ids: Vec<u32>,
    pub(super) selected_prompt_position_offsets: astronomical_runtime_integration::MlxArray,
    pub(super) selected_token_count_i32: i32,
}

pub(super) fn prepare_sparse_target_gpu_inputs(
    active_request: &mut Qwen3_5EngineRequest,
    model: &Qwen3_5Model,
    selected_speculative_prefill_positions_for_current_chunck: &[usize],
    speculative_prefill_target_token_count: usize,
) -> Result<SparseTargetGpuInputs, crate::InferenceEngineError> {
    let full_prompt_token_indices_on_gpu = active_request
        .speculative_prefill_prompt_token_indices
        .as_ref()
        .ok_or_else(|| {
            fatal_engine_error("speculative-prefill GPU prompt token indices are unavailable")
        })?;
    let selected_prompt_position_offsets: Vec<i32> =
        selected_speculative_prefill_positions_for_current_chunck
            .iter()
            .map(|selected_prompt_position| {
                i32::try_from(*selected_prompt_position).map_err(|_| {
                    fatal_engine_error("speculative-prefill prompt position exceeds the MLX range")
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
    let selected_prompt_position_offsets = model
        .runtime()
        .array_from_i32(
            &selected_prompt_position_offsets,
            &[
                i32::try_from(speculative_prefill_target_token_count).map_err(|_| {
                    fatal_engine_error(
                        "speculative-prefill selected token count exceeds the MLX range",
                    )
                })?,
            ],
        )
        .map_err(qwen3_5_runtime_error)?;
    let selected_token_count_i32 =
        i32::try_from(speculative_prefill_target_token_count).map_err(|_| {
            fatal_engine_error("speculative-prefill selected token count exceeds the MLX range")
        })?;
    let selected_prompt_token_ids = selected_speculative_prefill_positions_for_current_chunck
        .iter()
        .map(|selected_prompt_position| {
            active_request
                .input_token_ids
                .get(*selected_prompt_position)
                .copied()
                .ok_or_else(|| {
                    fatal_engine_error(
                        "speculative-prefill selected prompt position exceeds the prompt",
                    )
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let selected_token_indices_on_gpu = active_request
        .performance_attribution
        .measure_operation(
            crate::PerformanceOperation::SpeculativePrefillSparseInputAssembly,
            |_performance_attribution| {
                model.runtime().take_axis(
                    full_prompt_token_indices_on_gpu,
                    &selected_prompt_position_offsets,
                    1,
                )
            },
        )
        .map_err(qwen3_5_runtime_error)?;
    Ok(SparseTargetGpuInputs {
        selected_token_indices_on_gpu,
        selected_prompt_token_ids,
        selected_prompt_position_offsets,
        selected_token_count_i32,
    })
}

pub(super) fn record_sparse_target_and_mode_counters(
    active_request: &mut Qwen3_5EngineRequest,
    model: &Qwen3_5Model,
    speculative_prefill_target_is_active: bool,
    speculative_prefill_target_token_count: usize,
    speculative_prefill_chunck_mode: Qwen3_5SpeculativePrefillChunckMode,
    prefill_token_count: usize,
    all_completed_prefill_chunck_tokens: &[usize],
    terminal_history_token_count: usize,
    boundary_checkpoints: &mut Vec<Qwen3_5PersistentPromptCacheBoundaryCheckpoint>,
) -> Result<(), crate::InferenceEngineError> {
    if speculative_prefill_target_is_active {
        if let Some(previous_target_expert_payload_bytes) =
            active_request.speculative_prefill_target_expert_payload_bytes_after_draft_release
        {
            let current_target_expert_payload_bytes = model
                .expert_weight_memory_cache_statistics()
                .resident_payload_byte_count;
            active_request.performance_attribution.record_counter(
                PerformanceCounter::SpeculativePrefillTargetExpertRepopulatedPayloadBytes,
                current_target_expert_payload_bytes
                    .saturating_sub(previous_target_expert_payload_bytes),
            );
            active_request.speculative_prefill_target_expert_payload_bytes_after_draft_release =
                Some(current_target_expert_payload_bytes);
        }
        active_request.performance_attribution.record_counter(
            PerformanceCounter::SpeculativePrefillSparseTargetChunckCount,
            1,
        );
        active_request.performance_attribution.record_counter(
            PerformanceCounter::SpeculativePrefillSelectedTokenCount,
            u64::try_from(speculative_prefill_target_token_count).unwrap_or(u64::MAX),
        );
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
                if active_request.should_use_speculative_prefill {
                    return Err(configured_speculative_prefill_failure(
                        active_request.request_id,
                        "exact target prompt-state extraction",
                        error,
                    ));
                }
                tracing::warn!(%error, "final prompt-cache boundary extraction failed");
                boundary_checkpoints.clear();
                active_request.persistent_prompt_cache_capture_has_stopped = true;
            }
        }
    }
    match speculative_prefill_chunck_mode {
        Qwen3_5SpeculativePrefillChunckMode::TargetOnlyPrefix => {
            active_request.performance_attribution.record_counter(
                PerformanceCounter::SpeculativePrefillTargetOnlyPrefixChunckCount,
                1,
            );
            active_request.performance_attribution.record_counter(
                PerformanceCounter::SpeculativePrefillTargetOnlyPrefixTokenCount,
                u64::try_from(prefill_token_count).unwrap_or(u64::MAX),
            );
        }
        Qwen3_5SpeculativePrefillChunckMode::TerminalAdditionalHistoryCapture => {
            active_request.performance_attribution.record_counter(
                PerformanceCounter::SpeculativePrefillTerminalCaptureChunckCount,
                1,
            );
            record_terminal_history_token_count(active_request, terminal_history_token_count);
        }
        Qwen3_5SpeculativePrefillChunckMode::OrdinaryTarget => {}
    }
    Ok(())
}
