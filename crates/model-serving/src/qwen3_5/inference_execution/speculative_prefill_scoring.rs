#[cfg(feature = "direct-mlx")]
use super::super::model::Qwen3_5SpeculativePrefillDraftPersistentPromptCacheBlockConsumer;
#[cfg(feature = "direct-mlx")]
use super::engine_request::{Qwen3_5EngineRequest, Qwen3_5SpeculativePrefillFailureStageForTests};
#[cfg(feature = "direct-mlx")]
use super::speculative_prefill_selection::{
    qwen3_5_speculative_prefill_scoring_plan,
    qwen3_5_speculative_prefill_selectable_importance_score_range,
};
#[cfg(feature = "direct-mlx")]
use super::{
    Qwen3_5EngineState, qwen3_5_runtime_error,
    speculative_prefill_failure::configured_speculative_prefill_failure,
};
#[cfg(feature = "direct-mlx")]
use crate::RequestDecoderStateStack;
#[cfg(feature = "direct-mlx")]
use crate::{
    PerformanceCounter, PerformanceOperation, Qwen3_5ExecutionError,
    qwen3_5_select_speculative_prefill_token_positions_on_gpu,
};
#[cfg(feature = "direct-mlx")]
pub(super) enum SpeculativePrefillSelectionPreparation {
    Ready,
    DrafterPhaseStarted,
}
#[cfg(feature = "direct-mlx")]
impl Qwen3_5EngineState {
    pub(super) fn prepare_speculative_prefill_selection(
        &self,
        active_request: &mut Qwen3_5EngineRequest,
        scoring_start_position_tokens: usize,
        final_prompt_index: usize,
    ) -> Result<SpeculativePrefillSelectionPreparation, crate::InferenceEngineError> {
        if !active_request.should_use_speculative_prefill
            || active_request.speculative_prefill_scoring_attempted
        {
            return Ok(SpeculativePrefillSelectionPreparation::Ready);
        }
        active_request.speculative_prefill_dense_target_prefix_token_count = if active_request
            .speculative_prefill_restored_target_token_positions
            .is_some()
        {
            0
        } else {
            scoring_start_position_tokens
        };
        let Some((draft_scoring_token_range, selectable_importance_score_count)) =
            qwen3_5_speculative_prefill_scoring_plan(
                scoring_start_position_tokens,
                final_prompt_index,
                active_request.input_token_ids.len(),
            )
        else {
            return Err(configured_speculative_prefill_failure(
                active_request.request_id,
                "draft scoring planning",
                "the selectable conversation range is invalid",
            ));
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
                return Err(configured_speculative_prefill_failure(
                    active_request.request_id,
                    "draft scoring planning",
                    "the selectable conversation position exceeds the supported range",
                ));
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
            active_request.speculative_prefill_scoring_attempted = true;
            return Ok(SpeculativePrefillSelectionPreparation::Ready);
        }
        if !active_request.speculative_prefill_draft_phase_announced {
            active_request.speculative_prefill_draft_phase_announced = true;
            return Ok(SpeculativePrefillSelectionPreparation::DrafterPhaseStarted);
        }
        active_request.speculative_prefill_scoring_attempted = true;
        if active_request.take_forced_speculative_prefill_failure_for_tests(
            Qwen3_5SpeculativePrefillFailureStageForTests::DrafterLoading,
        ) {
            return Err(configured_speculative_prefill_failure(
                active_request.request_id,
                "drafter loading",
                "forced drafter loading failure",
            ));
        }
        let draft_model = self
            .load_request_scoped_speculative_prefill_draft_model(
                active_request.request_id.value(),
                u32::from(active_request.maximum_output_tokens),
                &mut active_request.performance_attribution,
            )
            .map_err(|draft_loading_error| {
                configured_speculative_prefill_failure(
                    active_request.request_id,
                    "drafter loading",
                    draft_loading_error,
                )
            })?;
        let mut draft_request_decoder_state =
            match RequestDecoderStateStack::empty_from_decoder_cache_layout(
                draft_model.decoder_cache_layout(),
            ) {
                Ok(draft_request_decoder_state) => draft_request_decoder_state,
                Err(draft_state_creation_error) => {
                    return Err(configured_speculative_prefill_failure(
                        active_request.request_id,
                        "drafter state initialization",
                        draft_state_creation_error,
                    ));
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
                return Err(configured_speculative_prefill_failure(
                    active_request.request_id,
                    "drafter persistent-state restoration",
                    draft_prefix_restore_error,
                ));
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
        active_request
            .prompt_work_reuse
            .drafter_eligible_token_count =
            u64::try_from(active_request.input_token_ids.len()).unwrap_or(u64::MAX);
        active_request
            .prompt_work_reuse
            .drafter_restored_token_count =
            u64::try_from(restored_draft_prefix_token_count).unwrap_or(u64::MAX);
        let draft_scored_suffix_token_count =
            active_request.input_token_ids[restored_draft_prefix_token_count..].len();
        active_request.performance_attribution.record_counter(
            PerformanceCounter::SpeculativePrefillDraftScoredSuffixTokenCount,
            u64::try_from(draft_scored_suffix_token_count).unwrap_or(u64::MAX),
        );
        let draft_forward_start_position_tokens =
            match u32::try_from(restored_draft_prefix_token_count) {
                Ok(draft_forward_start_position_tokens) => draft_forward_start_position_tokens,
                Err(_) => {
                    return Err(configured_speculative_prefill_failure(
                        active_request.request_id,
                        "drafter persistent-state restoration",
                        "the restored drafter prefix exceeds the supported range",
                    ));
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
            return Err(configured_speculative_prefill_failure(
                active_request.request_id,
                "draft score mapping",
                "speculative-prefill draft scores do not cover the target selection range",
            ));
        };
        let should_capture_persistent_prompt_cache_blocks = self
            .prepare_speculative_prefill_draft_cache_capture(
                active_request,
                restored_draft_prefix_token_count,
                draft_persistent_prefix_block_key.as_ref(),
            )?;
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
            return Err(configured_speculative_prefill_failure(
                active_request.request_id,
                "drafter memory admission",
                draft_memory_admission_error,
            ));
        }
        let draft_visual_embeddings = self
            .prepare_speculative_prefill_draft_visual_embeddings(
                active_request,
                &draft_model,
                restored_draft_prefix_token_count,
                is_visual_speculative_prefill_request,
            )
            .map_err(|drafter_visual_input_error| {
                configured_speculative_prefill_failure(
                    active_request.request_id,
                    "drafter visual input assembly",
                    drafter_visual_input_error,
                )
            })?;
        if !active_request.should_use_speculative_prefill {
            return Ok(SpeculativePrefillSelectionPreparation::Ready);
        }
        let should_force_draft_scoring_failure = active_request
            .take_forced_speculative_prefill_failure_for_tests(
                Qwen3_5SpeculativePrefillFailureStageForTests::DraftScoring,
            );
        let draft_suffix_token_ids =
            &active_request.input_token_ids[restored_draft_prefix_token_count..];
        let mut latest_persisted_draft_block_key = draft_persistent_prefix_block_key.clone();
        let mut draft_prompt_state_persistence_failed = false;
        let mut persist_completed_draft_block = self
            .speculative_prefill_draft_block_persistence_consumer(
                &draft_model,
                &active_request.input_token_ids,
                &active_request.ordered_image_sha256_digests,
                &mut latest_persisted_draft_block_key,
                &mut draft_prompt_state_persistence_failed,
            );
        let mut persistent_prompt_cache_block_consumer: Option<
            &mut Qwen3_5SpeculativePrefillDraftPersistentPromptCacheBlockConsumer<'_>,
        > = should_capture_persistent_prompt_cache_blocks
            .then_some(&mut persist_completed_draft_block);
        let persistent_prompt_cache_block_token_count = self
            .speculative_prefill_draft_persistent_prompt_cache
            .as_ref()
            .map(|persistent_prompt_cache| {
                persistent_prompt_cache
                    .model_contract_ref()
                    .block_token_count()
            })
            .unwrap_or(1);
        let draft_scoring_outcome = active_request.performance_attribution.measure_operation(
                PerformanceOperation::SpeculativePrefillDraftScoring,
                |performance_attribution| {
                    if should_force_draft_scoring_failure {
                        return Err(Qwen3_5ExecutionError::InvalidInput {
                            description: "forced speculative-prefill draft scoring failure",
                        });
                    }
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
                            persistent_prompt_cache_block_token_count,
                            persistent_prompt_cache_block_consumer.as_deref_mut(),
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
                            persistent_prompt_cache_block_token_count,
                            persistent_prompt_cache_block_consumer.as_deref_mut(),
                            &mut draft_request_decoder_state,
                            performance_attribution,
                        )
                    }
                },
             );
        drop(persistent_prompt_cache_block_consumer);
        drop(persist_completed_draft_block);
        let should_force_selection_failure = active_request
            .take_forced_speculative_prefill_failure_for_tests(
                Qwen3_5SpeculativePrefillFailureStageForTests::Selection,
            );
        let draft_scoring_outcome = active_request.performance_attribution.measure_operation(
            PerformanceOperation::SpeculativePrefillSelection,
            |_performance_attribution| draft_scoring_outcome.and_then(|draft_scoring_outcome| {
                if should_force_selection_failure {
                    return Err(Qwen3_5ExecutionError::InvalidInput {
                        description: "forced speculative-prefill selection failure",
                    });
                }
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
        active_request.speculative_prefill_draft_memory_telemetry = self
            .speculative_prefill_draft_memory_telemetry(
                active_request,
                &draft_model,
                &draft_request_decoder_state,
                draft_visual_embeddings.as_ref(),
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
            return Err(configured_speculative_prefill_failure(
                active_request.request_id,
                "drafter cleanup",
                draft_allocator_cleanup_error,
            ));
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
            draft_prompt_state_persistence_failed,
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
            .map_err(|draft_release_error| {
                configured_speculative_prefill_failure(
                    active_request.request_id,
                    "drafter release",
                    draft_release_error,
                )
            })?;
        let resumed_target_expert_retention =
            target_model.resume_expert_retention_after_request_memory_pressure();
        tracing::info!(
            request_id = active_request.request_id.value(),
            resumed_target_expert_retention,
            "released request-scoped speculative-prefill draft before target expert paging"
        );
        Ok(SpeculativePrefillSelectionPreparation::Ready)
    }
}
