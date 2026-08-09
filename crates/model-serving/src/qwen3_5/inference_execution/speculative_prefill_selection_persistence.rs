#[cfg(feature = "direct-mlx")]
use crate::{PerformanceCounter, PerformanceOperation, Qwen3_5ExecutionError};

#[cfg(feature = "direct-mlx")]
use super::super::model::Qwen3_5SpeculativePrefillDraftScoringOutcome;
#[cfg(feature = "direct-mlx")]
use super::Qwen3_5EngineState;
#[cfg(feature = "direct-mlx")]
use super::engine_request::Qwen3_5EngineRequest;
#[cfg(feature = "direct-mlx")]
use super::engine_request::Qwen3_5SpeculativePrefillFailureStageForTests;
#[cfg(feature = "direct-mlx")]
use super::speculative_prefill_failure::configured_speculative_prefill_failure;
#[cfg(feature = "direct-mlx")]
use super::speculative_prefill_selection::qwen3_5_merge_speculative_prefill_selection_with_image_pad_positions;
#[cfg(feature = "direct-mlx")]
use super::speculative_prefill_store::Qwen3_5SpeculativePrefillStoreKey;

#[cfg(feature = "direct-mlx")]
impl Qwen3_5EngineState {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn persist_speculative_prefill_selection(
        &self,
        active_request: &mut Qwen3_5EngineRequest,
        draft_model: &crate::Qwen3_5Model,
        draft_scoring_outcome: Result<
            (Vec<usize>, Qwen3_5SpeculativePrefillDraftScoringOutcome),
            Qwen3_5ExecutionError,
        >,
        is_visual_speculative_prefill_request: bool,
        scoring_start_position_tokens: usize,
        final_prompt_index: usize,
        selection_store_token_key: Qwen3_5SpeculativePrefillStoreKey,
        scoring_start_position_tokens_u32: u32,
        restored_draft_prefix_token_count: usize,
        draft_prompt_state_persistence_failed: bool,
    ) -> Result<(), crate::InferenceEngineError> {
        match draft_scoring_outcome {
            Ok((absolute_selected_token_positions, draft_scoring_outcome)) => {
                let absolute_selected_token_positions = if is_visual_speculative_prefill_request {
                    let mandatory_visual_token_count = active_request.input_token_ids
                        [scoring_start_position_tokens..final_prompt_index]
                        .iter()
                        .filter(|prompt_token_id| {
                            **prompt_token_id == active_request.image_pad_token_id
                        })
                        .count();
                    match qwen3_5_merge_speculative_prefill_selection_with_image_pad_positions(
                        absolute_selected_token_positions,
                        &active_request.input_token_ids,
                        scoring_start_position_tokens,
                        final_prompt_index,
                        active_request.image_pad_token_id,
                    ) {
                        Ok(absolute_selected_token_positions) => {
                            active_request.performance_attribution.record_counter(
                                PerformanceCounter::SpeculativePrefillMandatoryVisualTokenCount,
                                u64::try_from(mandatory_visual_token_count).unwrap_or(u64::MAX),
                            );
                            tracing::info!(
                                request_id = active_request.request_id.value(),
                                mandatory_visual_token_count,
                                selected_token_count = absolute_selected_token_positions.len(),
                                "retained every visual token for sparse target prefill"
                            );
                            absolute_selected_token_positions
                        }
                        Err(selection_merge_error) => {
                            return Err(configured_speculative_prefill_failure(
                                active_request.request_id,
                                "mandatory visual-position selection",
                                selection_merge_error,
                            ));
                        }
                    }
                } else {
                    absolute_selected_token_positions
                };
                if let Err(prompt_token_indices_error) =
                    self.prepare_speculative_prefill_prompt_token_indices_on_gpu(active_request)
                {
                    return Err(configured_speculative_prefill_failure(
                        active_request.request_id,
                        "sparse target input assembly",
                        prompt_token_indices_error,
                    ));
                }
                active_request.speculative_prefill_selected_token_positions =
                    Some(absolute_selected_token_positions.clone());
                let target_prefix_work_token_count = active_request
                    .speculative_prefill_restored_target_token_positions
                    .as_ref()
                    .map_or(
                        active_request.speculative_prefill_dense_target_prefix_token_count,
                        |restored_target_token_positions| {
                            restored_target_token_positions.shape()[0].max(0) as usize
                        },
                    );
                active_request.prompt_work_reuse.target_eligible_token_count = u64::try_from(
                    target_prefix_work_token_count
                        .saturating_add(absolute_selected_token_positions.len())
                        .saturating_add(1),
                )
                .unwrap_or(u64::MAX);
                if !is_visual_speculative_prefill_request
                    && self
                        .speculative_prefill_draft_persistent_prompt_cache
                        .is_some()
                    && let Some(selection_contract) = self.speculative_prefill_selection_contract(
                        scoring_start_position_tokens_u32,
                        active_request.input_token_ids.len(),
                    )
                {
                    if active_request.take_forced_speculative_prefill_failure_for_tests(
                        Qwen3_5SpeculativePrefillFailureStageForTests::SelectionPersistence,
                    ) {
                        return Err(configured_speculative_prefill_failure(
                            active_request.request_id,
                            "selection persistence",
                            "forced selection persistence failure",
                        ));
                    }
                    let selected_token_position_values = absolute_selected_token_positions
                        .iter()
                        .map(|selected_token_position| {
                            u32::try_from(*selected_token_position).map_err(|_| {
                                configured_speculative_prefill_failure(
                                    active_request.request_id,
                                    "selection persistence",
                                    "a selected prompt position exceeds the supported range",
                                )
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let selected_token_position_count_i32 =
                        i32::try_from(selected_token_position_values.len()).map_err(|_| {
                            configured_speculative_prefill_failure(
                                active_request.request_id,
                                "selection persistence",
                                "the selected position count exceeds the MLX range",
                            )
                        })?;
                    let selected_token_positions_on_gpu = draft_model
                        .runtime()
                        .array_from_u32(
                            &selected_token_position_values,
                            &[selected_token_position_count_i32],
                        )
                        .map_err(|selection_array_error| {
                            configured_speculative_prefill_failure(
                                active_request.request_id,
                                "selection persistence",
                                selection_array_error,
                            )
                        })?;
                    if let Err(selection_save_error) = active_request
                        .performance_attribution
                        .measure_operation(
                                PerformanceOperation::SpeculativePrefillSelectionDiskWrite,
                                |_performance_attribution| {
                                    self.speculative_prefill_draft_persistent_prompt_cache
                                        .as_ref()
                                        .ok_or(Qwen3_5ExecutionError::InvalidInput {
                                            description:
                                                "speculative-prefill drafter cache disappeared before selection save",
                                        })?
                                        .save_speculative_prefill_selection(
                                            draft_model.runtime(),
                                            &selection_contract,
                                            &active_request.input_token_ids,
                                            &selected_token_positions_on_gpu,
                                        )
                                        .map_err(Qwen3_5ExecutionError::from)
                                },
                            )
                    {
                        return Err(configured_speculative_prefill_failure(
                            active_request.request_id,
                            "selection persistence",
                            selection_save_error,
                        ));
                    }
                }
                if !is_visual_speculative_prefill_request {
                    self.store_speculative_prefill_selection(
                        selection_store_token_key,
                        absolute_selected_token_positions.clone(),
                    );
                }
                if !is_visual_speculative_prefill_request
                    && (scoring_start_position_tokens == 0 || restored_draft_prefix_token_count > 0)
                {
                    let scored_prompt_token_count =
                        u32::try_from(active_request.input_token_ids.len()).unwrap_or(u32::MAX);
                    let scored_prompt_prefix_store_key = self.speculative_prefill_store_key(
                        scored_prompt_token_count,
                        active_request.input_token_ids.clone(),
                    );
                    self.store_speculative_prefill_draft_prefix_checkpoint(
                        scored_prompt_prefix_store_key,
                        draft_scoring_outcome.draft_prompt_prefix_allocation_checkpoint,
                        draft_scoring_outcome.draft_prompt_prefix_payload_bytes,
                    );
                    active_request.performance_attribution.record_counter(
                        PerformanceCounter::SpeculativePrefillDraftPrefixStoreWriteCount,
                        1,
                    );
                }
                active_request
                    .performance_attribution
                    .record_counter(PerformanceCounter::SpeculativePrefillDraftScoringCount, 1);
                tracing::info!(
                    request_id = active_request.request_id.value(),
                    selected_token_count = active_request
                        .speculative_prefill_selected_token_positions
                        .as_ref()
                        .map_or(0, Vec::len),
                    "completed speculative-prefill drafter scoring"
                );
            }
            Err(speculative_prefill_error) => {
                let failure_stage = if draft_prompt_state_persistence_failed {
                    "drafter prompt-state persistence"
                } else {
                    "draft scoring or selection"
                };
                return Err(configured_speculative_prefill_failure(
                    active_request.request_id,
                    failure_stage,
                    speculative_prefill_error,
                ));
            }
        }
        Ok(())
    }
}
