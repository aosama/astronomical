#[cfg(feature = "direct-mlx")]
use astronomical_runtime_integration::MlxRuntimeError;

#[cfg(feature = "direct-mlx")]
use crate::{PerformanceCounter, PerformanceOperation, Qwen3_5ExecutionError};

#[cfg(feature = "direct-mlx")]
use super::Qwen3_5EngineState;
#[cfg(feature = "direct-mlx")]
use super::engine_request::Qwen3_5EngineRequest;
#[cfg(feature = "direct-mlx")]
use super::speculative_prefill_store::Qwen3_5SpeculativePrefillStoreKey;

#[cfg(feature = "direct-mlx")]
impl Qwen3_5EngineState {
    pub(super) fn reuse_speculative_prefill_selection_if_available(
        &self,
        active_request: &mut Qwen3_5EngineRequest,
        selection_store_token_key: &Qwen3_5SpeculativePrefillStoreKey,
        scoring_start_position_tokens: usize,
        scoring_start_position_tokens_u32: u32,
        is_visual_speculative_prefill_request: bool,
    ) -> Result<bool, crate::InferenceEngineError> {
        if !is_visual_speculative_prefill_request
            && let Some(selected_token_positions) = self
                .speculative_prefill_selection_store
                .borrow()
                .get(selection_store_token_key)
                .cloned()
        {
            if let Err(prompt_token_indices_error) =
                self.prepare_speculative_prefill_prompt_token_indices_on_gpu(active_request)
            {
                if matches!(
                    &prompt_token_indices_error,
                    Qwen3_5ExecutionError::Runtime(
                        MlxRuntimeError::ActiveMemoryLimitExceeded { .. }
                    )
                ) {
                    return Err(prompt_token_indices_error.into());
                }
                self.record_speculative_prefill_input_assembly_fallback(
                    active_request,
                    prompt_token_indices_error,
                );
                return Ok(true);
            }
            active_request.speculative_prefill_selected_token_positions =
                Some(selected_token_positions);
            active_request.performance_attribution.record_counter(
                PerformanceCounter::SpeculativePrefillSelectionStoreHitCount,
                1,
            );
            tracing::info!(
                request_id = active_request.request_id.value(),
                selected_token_count = active_request
                    .speculative_prefill_selected_token_positions
                    .as_ref()
                    .map_or(0, Vec::len),
                selection_source = "worker_memory",
                "reused speculative-prefill selection"
            );
            return Ok(true);
        }
        if !is_visual_speculative_prefill_request
            && let (
                Some(selection_contract),
                Some(target_model),
                Some(draft_persistent_prompt_cache),
            ) = (
                self.speculative_prefill_selection_contract(
                    scoring_start_position_tokens_u32,
                    active_request.input_token_ids.len(),
                ),
                self.model.as_ref(),
                self.speculative_prefill_draft_persistent_prompt_cache
                    .as_ref(),
            )
        {
            let persistent_selection_load_result =
                active_request.performance_attribution.measure_operation(
                    PerformanceOperation::SpeculativePrefillSelectionDiskRead,
                    |performance_attribution| {
                        draft_persistent_prompt_cache.load_speculative_prefill_selection(
                            target_model.runtime(),
                            &selection_contract,
                            &active_request.input_token_ids,
                            performance_attribution.positional_file_read_metrics(),
                        )
                    },
                );
            if let Err(selection_load_error) = &persistent_selection_load_result {
                tracing::debug!(
                    error = %selection_load_error,
                    "persisted speculative-prefill selection was unavailable; continuing with draft scoring"
                );
            }
            if let Ok(Some(selected_token_positions_on_gpu)) = persistent_selection_load_result {
                let selected_token_positions = target_model
                    .runtime()
                    .copy_u32_values(&selected_token_positions_on_gpu)
                    .ok()
                    .and_then(|selected_token_positions| {
                        selected_token_positions
                            .into_iter()
                            .map(|selected_token_position| {
                                usize::try_from(selected_token_position).ok()
                            })
                            .collect::<Option<Vec<_>>>()
                    })
                    .filter(|selected_token_positions| {
                        selected_token_positions
                            .windows(2)
                            .all(|position_pair| position_pair[0] < position_pair[1])
                            && selected_token_positions
                                .iter()
                                .all(|selected_token_position| {
                                    *selected_token_position >= scoring_start_position_tokens
                                        && *selected_token_position
                                            < active_request.input_token_ids.len().saturating_sub(1)
                                })
                    });
                if let Some(selected_token_positions) = selected_token_positions {
                    if let Err(prompt_token_indices_error) =
                        self.prepare_speculative_prefill_prompt_token_indices_on_gpu(active_request)
                    {
                        if matches!(
                            &prompt_token_indices_error,
                            Qwen3_5ExecutionError::Runtime(
                                MlxRuntimeError::ActiveMemoryLimitExceeded { .. }
                            )
                        ) {
                            return Err(prompt_token_indices_error.into());
                        }
                        self.record_speculative_prefill_input_assembly_fallback(
                            active_request,
                            prompt_token_indices_error,
                        );
                        return Ok(true);
                    }
                    self.store_speculative_prefill_selection(
                        selection_store_token_key.clone(),
                        selected_token_positions.clone(),
                    );
                    active_request.speculative_prefill_selected_token_positions =
                        Some(selected_token_positions);
                    active_request.performance_attribution.record_counter(
                        PerformanceCounter::SpeculativePrefillSelectionStoreHitCount,
                        1,
                    );
                    active_request.performance_attribution.record_counter(
                        PerformanceCounter::SpeculativePrefillSelectionPersistentHitCount,
                        1,
                    );
                    tracing::info!(
                        request_id = active_request.request_id.value(),
                        selected_token_count = active_request
                            .speculative_prefill_selected_token_positions
                            .as_ref()
                            .map_or(0, Vec::len),
                        selection_source = "persistent_storage",
                        "reused speculative-prefill selection"
                    );
                    return Ok(true);
                }
                tracing::warn!(
                    "persisted speculative-prefill selection was outside the current prompt; continuing with draft scoring"
                );
            }
        }
        Ok(false)
    }
}
