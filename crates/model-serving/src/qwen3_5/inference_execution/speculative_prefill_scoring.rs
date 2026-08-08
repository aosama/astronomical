#[cfg(feature = "direct-mlx")]
use super::engine_request::Qwen3_5EngineRequest;
#[cfg(feature = "direct-mlx")]
use super::speculative_prefill_selection::{
    qwen3_5_speculative_prefill_scoring_plan,
    qwen3_5_speculative_prefill_selectable_importance_score_range,
};
#[cfg(feature = "direct-mlx")]
use super::{Qwen3_5EngineState, qwen3_5_runtime_error};
#[cfg(feature = "direct-mlx")]
use crate::RequestDecoderStateStack;
#[cfg(feature = "direct-mlx")]
use crate::{
    PerformanceCounter, PerformanceOperation, Qwen3_5ExecutionError,
    qwen3_5_select_speculative_prefill_token_positions_on_gpu,
};
#[cfg(feature = "direct-mlx")]
use astronomical_runtime_integration::MlxRuntimeError;
#[cfg(feature = "direct-mlx")]
impl Qwen3_5EngineState {
    pub(super) fn prepare_speculative_prefill_selection(
        &self,
        active_request: &mut Qwen3_5EngineRequest,
        scoring_start_position_tokens: usize,
        final_prompt_index: usize,
    ) -> Result<(), crate::InferenceEngineError> {
        if !active_request.should_use_speculative_prefill
            || active_request.speculative_prefill_scoring_attempted
        {
            return Ok(());
        }
        active_request.speculative_prefill_scoring_attempted = true;
        let Some((draft_scoring_token_range, selectable_importance_score_count)) =
            qwen3_5_speculative_prefill_scoring_plan(
                scoring_start_position_tokens,
                final_prompt_index,
                active_request.input_token_ids.len(),
            )
        else {
            active_request.should_use_speculative_prefill = false;
            active_request
                .performance_attribution
                .record_counter(PerformanceCounter::SpeculativePrefillFallbackCount, 1);
            tracing::warn!(
                request_id = active_request.request_id.value(),
                scoring_start_position_tokens,
                final_prompt_index,
                prompt_token_count = active_request.input_token_ids.len(),
                "speculative-prefill scoring plan was invalid; continuing target-only"
            );
            return Ok(());
        };
        tracing::info!(
            request_id = active_request.request_id.value(),
            scoring_start_position_tokens,
            final_prompt_index,
            selectable_importance_score_count,
            "starting speculative-prefill drafter scoring"
        );
        let draft_prompt_token_ids = &active_request.input_token_ids[draft_scoring_token_range];
        let scoring_start_position_tokens_u32 = match u32::try_from(scoring_start_position_tokens) {
            Ok(scoring_start_position_tokens_u32) => scoring_start_position_tokens_u32,
            Err(_) => {
                active_request.should_use_speculative_prefill = false;
                active_request
                    .performance_attribution
                    .record_counter(PerformanceCounter::SpeculativePrefillFallbackCount, 1);
                tracing::warn!(
                    request_id = active_request.request_id.value(),
                    "optional speculative-prefill suffix position exceeds the u32 range; continuing target-only"
                );
                return Ok(());
            }
        };
        let selection_store_token_key = self.speculative_prefill_store_key(
            scoring_start_position_tokens_u32,
            draft_prompt_token_ids.to_vec(),
        );
        let is_visual_speculative_prefill_request = !active_request
            .speculative_prefill_processed_visual_images
            .is_empty();
        if is_visual_speculative_prefill_request {
            tracing::info!(
                request_id = active_request.request_id.value(),
                processed_image_count = active_request
                    .speculative_prefill_processed_visual_images
                    .len(),
                "using image-digest-bound reusable draft state for visual speculative-prefill scoring"
            );
        }
        if self.reuse_speculative_prefill_selection_if_available(
            active_request,
            &selection_store_token_key,
            scoring_start_position_tokens,
            scoring_start_position_tokens_u32,
            is_visual_speculative_prefill_request,
        )? {
            return Ok(());
        }
        let draft_model = self.load_request_scoped_speculative_prefill_draft_model(
            active_request.request_id.value(),
            u32::from(active_request.maximum_output_tokens),
            &mut active_request.performance_attribution,
        )?;
        let mut draft_request_decoder_state =
            match RequestDecoderStateStack::empty_from_decoder_cache_layout(
                draft_model.decoder_cache_layout(),
            ) {
                Ok(draft_request_decoder_state) => draft_request_decoder_state,
                Err(draft_state_creation_error) => {
                    self.record_speculative_prefill_scoring_fallback(
                        active_request,
                        &draft_model,
                        draft_state_creation_error,
                    );
                    return Ok(());
                }
            };
        let should_force_draft_prefix_restore_failure = std::mem::take(
            &mut active_request
                .force_next_speculative_prefill_draft_prefix_restore_failure_for_tests,
        );
        let draft_prefix_restore_attempt = if should_force_draft_prefix_restore_failure {
            Err(Qwen3_5ExecutionError::InvalidInput {
                description: "forced speculative-prefill draft-prefix restore failure",
            })
        } else if is_visual_speculative_prefill_request {
            self.restore_speculative_prefill_draft_persistent_prefix(
                &draft_model,
                &active_request.input_token_ids,
                &active_request.ordered_image_sha256_digests,
                &mut draft_request_decoder_state,
                &mut active_request.performance_attribution,
            )
            .map(|draft_persistent_prefix_restore_outcome| {
                draft_persistent_prefix_restore_outcome.map_or(
                    (0, None, false),
                    |draft_persistent_prefix_restore_outcome| {
                        (
                            draft_persistent_prefix_restore_outcome.restored_token_count,
                            Some(
                                draft_persistent_prefix_restore_outcome
                                    .last_restored_persistent_prompt_cache_block_key,
                            ),
                            false,
                        )
                    },
                )
            })
        } else {
            match self.restore_longest_speculative_prefill_draft_prefix_checkpoint(
                &active_request.input_token_ids,
                &mut draft_request_decoder_state,
            ) {
                Ok(Some(restored_draft_prefix_token_count)) => {
                    Ok((restored_draft_prefix_token_count, None, true))
                }
                Ok(None) => self
                    .restore_speculative_prefill_draft_persistent_prefix(
                        &draft_model,
                        &active_request.input_token_ids,
                        &active_request.ordered_image_sha256_digests,
                        &mut draft_request_decoder_state,
                        &mut active_request.performance_attribution,
                    )
                    .map(|draft_persistent_prefix_restore_outcome| {
                        draft_persistent_prefix_restore_outcome.map_or(
                            (0, None, false),
                            |draft_persistent_prefix_restore_outcome| {
                                (
                                    draft_persistent_prefix_restore_outcome.restored_token_count,
                                    Some(
                                        draft_persistent_prefix_restore_outcome
                                            .last_restored_persistent_prompt_cache_block_key,
                                    ),
                                    false,
                                )
                            },
                        )
                    }),
                Err(draft_prefix_restore_error) => Err(draft_prefix_restore_error),
            }
        };
        let (
            restored_draft_prefix_token_count,
            draft_persistent_prefix_block_key,
            draft_prefix_store_hit,
        ) = match draft_prefix_restore_attempt {
            Ok(draft_prefix_restore_outcome) => draft_prefix_restore_outcome,
            Err(draft_prefix_restore_error) => {
                tracing::warn!(
                    request_id = active_request.request_id.value(),
                    error = %draft_prefix_restore_error,
                    "speculative-prefill draft prefix cache restore failed; retrying uncached scoring"
                );
                drop(draft_request_decoder_state);
                if let Err(draft_allocator_cleanup_error) =
                    active_request.performance_attribution.measure_operation(
                        PerformanceOperation::MlxAllocatorCacheCleanup,
                        |_performance_attribution| {
                            draft_model
                                .runtime()
                                .synchronize_gpu_stream_and_clear_allocator_cache()
                        },
                    )
                {
                    self.record_speculative_prefill_scoring_fallback(
                        active_request,
                        &draft_model,
                        draft_allocator_cleanup_error,
                    );
                    return Ok(());
                }
                draft_request_decoder_state =
                    match RequestDecoderStateStack::empty_from_decoder_cache_layout(
                        draft_model.decoder_cache_layout(),
                    ) {
                        Ok(draft_request_decoder_state) => draft_request_decoder_state,
                        Err(draft_state_creation_error) => {
                            self.record_speculative_prefill_scoring_fallback(
                                active_request,
                                &draft_model,
                                draft_state_creation_error,
                            );
                            return Ok(());
                        }
                    };
                (0, None, false)
            }
        };
        if draft_prefix_store_hit {
            active_request.performance_attribution.record_counter(
                PerformanceCounter::SpeculativePrefillDraftPrefixStoreHitCount,
                1,
            );
        }
        if draft_persistent_prefix_block_key.is_some() {
            active_request.performance_attribution.record_counter(
                PerformanceCounter::SpeculativePrefillDraftPersistentPrefixHitCount,
                1,
            );
            active_request.performance_attribution.record_counter(
                PerformanceCounter::SpeculativePrefillDraftPersistentPrefixRestoredTokenCount,
                restored_draft_prefix_token_count as u64,
            );
        }
        let draft_scored_suffix_token_count =
            active_request.input_token_ids[restored_draft_prefix_token_count..].len();
        active_request.performance_attribution.record_counter(
            PerformanceCounter::SpeculativePrefillDraftScoredSuffixTokenCount,
            u64::try_from(draft_scored_suffix_token_count).unwrap_or(u64::MAX),
        );
        let draft_forward_start_position_tokens = match u32::try_from(
            restored_draft_prefix_token_count,
        ) {
            Ok(draft_forward_start_position_tokens) => draft_forward_start_position_tokens,
            Err(_) => {
                self.record_speculative_prefill_scoring_fallback(
                        active_request,
                        &draft_model,
                        Qwen3_5ExecutionError::InvalidInput {
                            description: "restored speculative-prefill drafter prefix exceeds the u32 range",
                        },
                    );
                return Ok(());
            }
        };
        let draft_importance_score_start_position_tokens = 0_usize;
        let scored_draft_prompt_token_count = active_request.input_token_ids.len();
        let Some(selectable_importance_score_range) =
            qwen3_5_speculative_prefill_selectable_importance_score_range(
                draft_importance_score_start_position_tokens,
                scored_draft_prompt_token_count,
                scoring_start_position_tokens,
                selectable_importance_score_count,
            )
        else {
            self.record_speculative_prefill_scoring_fallback(
                active_request,
                &draft_model,
                "speculative-prefill draft scores do not cover the target selection range",
            );
            return Ok(());
        };
        let should_capture_persistent_prompt_cache_blocks = self
            .speculative_prefill_draft_persistent_prompt_cache_write_queue
            .is_some()
            && (restored_draft_prefix_token_count == 0
                || draft_persistent_prefix_block_key.is_some());
        let draft_memory_admission_result =
            active_request.performance_attribution.measure_operation(
                PerformanceOperation::SpeculativePrefillDraftMemoryAdmission,
                |performance_attribution| {
                    self.admit_speculative_prefill_draft_scoring_memory(
                        active_request.request_id.value(),
                        &draft_model,
                        &draft_request_decoder_state,
                        draft_scored_suffix_token_count,
                        is_visual_speculative_prefill_request,
                        performance_attribution,
                    )
                },
            );
        if let Err(draft_memory_admission_error) = draft_memory_admission_result {
            return Err(draft_memory_admission_error);
        }
        tracing::info!(
            request_id = active_request.request_id.value(),
            draft_suffix_token_count = draft_scored_suffix_token_count,
            restored_draft_prefix_token_count,
            "admitted speculative-prefill drafter scoring memory"
        );
        let draft_visual_embeddings = self.prepare_speculative_prefill_draft_visual_embeddings(
            active_request,
            &draft_model,
            restored_draft_prefix_token_count,
            is_visual_speculative_prefill_request,
        )?;
        if !active_request.should_use_speculative_prefill {
            return Ok(());
        }
        let draft_suffix_token_ids =
            &active_request.input_token_ids[restored_draft_prefix_token_count..];
        let draft_scoring_outcome = loop {
            let draft_scoring_allocation_checkpoint =
                match draft_request_decoder_state.allocation_checkpoint() {
                    Ok(draft_scoring_allocation_checkpoint) => draft_scoring_allocation_checkpoint,
                    Err(draft_scoring_checkpoint_error) => {
                        self.record_speculative_prefill_scoring_fallback(
                            active_request,
                            &draft_model,
                            draft_scoring_checkpoint_error,
                        );
                        return Ok(());
                    }
                };
            let draft_scoring_attempt_outcome = active_request.performance_attribution.measure_operation(
                PerformanceOperation::SpeculativePrefillDraftScoring,
                |performance_attribution| {
                    if let Some(draft_visual_embeddings) = draft_visual_embeddings.as_ref() {
                        draft_model.score_speculative_prefill_importance_with_visual_embeddings_and_performance_attribution(
                            draft_suffix_token_ids,
                            draft_forward_start_position_tokens,
                            draft_importance_score_start_position_tokens,
                            scored_draft_prompt_token_count,
                            usize::try_from(self.speculative_prefill.lookahead_token_count)
                                .unwrap_or(usize::MAX),
                            usize::try_from(
                                self.speculative_prefill.importance_pooling_kernel_token_count,
                            )
                            .unwrap_or(usize::MAX),
                            should_capture_persistent_prompt_cache_blocks,
                            draft_visual_embeddings,
                            active_request.image_pad_token_id,
                            &mut draft_request_decoder_state,
                            performance_attribution,
                        )
                    } else {
                        draft_model.score_speculative_prefill_importance_with_performance_attribution(
                            draft_suffix_token_ids,
                            draft_forward_start_position_tokens,
                            draft_importance_score_start_position_tokens,
                            scored_draft_prompt_token_count,
                            usize::try_from(self.speculative_prefill.lookahead_token_count)
                                .unwrap_or(usize::MAX),
                            usize::try_from(
                                self.speculative_prefill.importance_pooling_kernel_token_count,
                            )
                            .unwrap_or(usize::MAX),
                            should_capture_persistent_prompt_cache_blocks,
                            &mut draft_request_decoder_state,
                            performance_attribution,
                        )
                    }
                },
            );
            let Err(Qwen3_5ExecutionError::Runtime(MlxRuntimeError::ActiveMemoryLimitExceeded {
                active_memory_bytes,
                attempted_allocation_bytes,
                allowed_active_memory_bytes,
            })) = &draft_scoring_attempt_outcome
            else {
                break draft_scoring_attempt_outcome;
            };
            let rejected_draft_scoring_active_memory_bytes = *active_memory_bytes;
            let rejected_draft_scoring_allocation_bytes = *attempted_allocation_bytes;
            let rejected_draft_scoring_allowed_memory_bytes = *allowed_active_memory_bytes;
            if let Err(draft_scoring_restore_error) = draft_request_decoder_state
                .restore_allocation_checkpoint(draft_scoring_allocation_checkpoint)
            {
                self.record_speculative_prefill_scoring_fallback(
                    active_request,
                    &draft_model,
                    draft_scoring_restore_error,
                );
                return Ok(());
            }
            let target_expert_reclamation_result =
                active_request.performance_attribution.measure_operation(
                    PerformanceOperation::SpeculativePrefillDraftMemoryAdmission,
                    |performance_attribution| {
                        draft_model
                            .runtime()
                            .synchronize_gpu_stream_and_clear_allocator_cache()
                            .map_err(qwen3_5_runtime_error)?;
                        self.reclaim_target_experts_after_draft_scoring_allocation_rejection(
                            active_request.request_id.value(),
                            rejected_draft_scoring_active_memory_bytes,
                            rejected_draft_scoring_allocation_bytes,
                            rejected_draft_scoring_allowed_memory_bytes,
                            performance_attribution,
                        )
                    },
                );
            if let Err(target_expert_reclamation_error) = target_expert_reclamation_result {
                return Err(target_expert_reclamation_error);
            }
        };
        let draft_scoring_outcome = active_request.performance_attribution.measure_operation(
            PerformanceOperation::SpeculativePrefillSelection,
            |_performance_attribution| draft_scoring_outcome.and_then(|draft_scoring_outcome| {
                let importance_score_shape = draft_scoring_outcome.importance_scores.shape();
                let selectable_importance_score_range_start_i32 =
                    i32::try_from(selectable_importance_score_range.start).map_err(|_| {
                        Qwen3_5ExecutionError::InvalidInput {
                            description: "speculative-prefill importance score range start exceeds the MLX range",
                        }
                    })?;
                let selectable_importance_score_range_end_i32 =
                    i32::try_from(selectable_importance_score_range.end).map_err(|_| {
                        Qwen3_5ExecutionError::InvalidInput {
                            description: "speculative-prefill importance score range end exceeds the MLX range",
                        }
                    })?;
                if importance_score_shape.len() != 1
                    || importance_score_shape[0] < selectable_importance_score_range_end_i32
                {
                    return Err(Qwen3_5ExecutionError::InvalidInput {
                        description: "speculative-prefill draft produced fewer importance scores than expected",
                    });
                }
                let selectable_importance_scores = draft_model.runtime().slice(
                    &draft_scoring_outcome.importance_scores,
                    &[selectable_importance_score_range_start_i32],
                    &[selectable_importance_score_range_end_i32],
                    &[1],
                )?;
                let selected_token_positions =
                    qwen3_5_select_speculative_prefill_token_positions_on_gpu(
                        draft_model.runtime(),
                        &selectable_importance_scores,
                        self.speculative_prefill.keep_percentage,
                        usize::try_from(self.speculative_prefill.selection_chunck_token_count)
                            .map_err(|_| Qwen3_5ExecutionError::InvalidInput {
                                description: "speculative-prefill selection chunk count exceeds the usize range",
                            })?,
                        usize::try_from(self.speculative_prefill.mandatory_trailing_token_count)
                            .map_err(|_| Qwen3_5ExecutionError::InvalidInput {
                                description: "speculative-prefill trailing token count exceeds the usize range",
                            })?,
                    )?;
                let scoring_start_position_scalar = draft_model.runtime().array_from_i32(
                    &[i32::try_from(scoring_start_position_tokens).map_err(|_| {
                        Qwen3_5ExecutionError::InvalidInput {
                            description: "speculative-prefill scoring start exceeds the MLX range",
                        }
                    })?],
                    &[],
                )?;
                let absolute_selected_token_positions = draft_model
                    .runtime()
                    .add(&selected_token_positions, &scoring_start_position_scalar)?;
                let absolute_selected_token_positions = draft_model.runtime().astype(
                    &absolute_selected_token_positions,
                    astronomical_runtime_integration::MlxDtype::UInt32,
                )?;
                let absolute_selected_token_positions = draft_model
                    .runtime()
                    .copy_u32_values(&absolute_selected_token_positions)?
                    .into_iter()
                    .map(|selected_token_position| {
                        usize::try_from(selected_token_position).map_err(|_| {
                            Qwen3_5ExecutionError::InvalidInput {
                                description: "speculative-prefill selected position exceeds usize",
                            }
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((absolute_selected_token_positions, draft_scoring_outcome))
            }),
        );
        drop(draft_request_decoder_state);
        if let Err(draft_allocator_cleanup_error) =
            active_request.performance_attribution.measure_operation(
                PerformanceOperation::MlxAllocatorCacheCleanup,
                |_performance_attribution| {
                    draft_model
                        .runtime()
                        .synchronize_gpu_stream_and_clear_allocator_cache()
                },
            )
        {
            self.record_speculative_prefill_scoring_fallback(
                active_request,
                &draft_model,
                draft_allocator_cleanup_error,
            );
            return Ok(());
        }
        self.persist_speculative_prefill_selection(
            active_request,
            &draft_model,
            draft_scoring_outcome,
            is_visual_speculative_prefill_request,
            scoring_start_position_tokens,
            final_prompt_index,
            selection_store_token_key,
            scoring_start_position_tokens_u32,
            restored_draft_prefix_token_count,
            draft_persistent_prefix_block_key,
        )?;
        let target_model = self
            .model
            .as_ref()
            .ok_or_else(|| qwen3_5_runtime_error("Qwen3.5 engine lost its loaded target model"))?;
        active_request
            .performance_attribution
            .measure_operation(
                PerformanceOperation::SpeculativePrefillRequestScopedDraftRelease,
                |_performance_attribution| {
                    drop(draft_visual_embeddings);
                    drop(draft_model);
                    target_model
                        .runtime()
                        .synchronize_gpu_stream_and_clear_allocator_cache()
                },
            )
            .map_err(qwen3_5_runtime_error)?;
        let resumed_target_expert_retention =
            target_model.resume_expert_retention_after_request_memory_pressure();
        tracing::info!(
            request_id = active_request.request_id.value(),
            resumed_target_expert_retention,
            "released request-scoped speculative-prefill draft before target expert paging"
        );
        Ok(())
    }
}
