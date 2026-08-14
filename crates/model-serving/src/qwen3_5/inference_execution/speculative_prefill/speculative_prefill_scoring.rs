//! Coordinates one request's complete drafter phase.
//!
//! This is the central SpecPrefill state machine. It first attempts cheap exact
//! selection reuse. On a miss it yields once to announce the Drafter phase; the
//! next call performs draft loading, prefix restoration, memory admission,
//! optional vision projection, scoring, selection, persistence, and complete
//! drafter release. No drafter owner may survive the `Ready` outcome.

#[cfg(feature = "direct-mlx")]
use super::super::super::model::Qwen3_5SpeculativePrefillDraftPersistentPromptCacheBlockConsumer;
#[cfg(feature = "direct-mlx")]
use super::super::Qwen3_5EngineState;
#[cfg(feature = "direct-mlx")]
use super::super::engine_request::{
    Qwen3_5EngineRequest, Qwen3_5SpeculativePrefillFailureStageForTests,
};
#[cfg(feature = "direct-mlx")]
use super::speculative_prefill_failure::configured_speculative_prefill_failure;
#[cfg(feature = "direct-mlx")]
use super::speculative_prefill_selection::{
    qwen3_5_speculative_prefill_scoring_plan,
    qwen3_5_speculative_prefill_selectable_importance_score_range,
};
#[cfg(feature = "direct-mlx")]
use super::speculative_prefill_selection_gpu::select_absolute_speculative_prefill_positions_from_draft_scores;
#[cfg(feature = "direct-mlx")]
use crate::RequestDecoderStateStack;
#[cfg(feature = "direct-mlx")]
use crate::{PerformanceCounter, PerformanceOperation, Qwen3_5ExecutionError};
#[cfg(feature = "direct-mlx")]
pub(crate) enum SpeculativePrefillSelectionPreparation {
    /// Selection is installed (or SpecPrefill does not need one) and target work may continue.
    Ready,
    /// The caller must emit the phase transition and invoke preparation again.
    DrafterPhaseStarted,
}
#[cfg(feature = "direct-mlx")]
impl Qwen3_5EngineState {
    /// Installs the absolute target positions selected for this request.
    ///
    /// The method is idempotent at its call boundary: once scoring was attempted,
    /// subsequent prefill advances return `Ready` without loading another draft.
    /// Any failure after configured work starts is translated by the caller into
    /// a request failure; there is intentionally no target-only retry.
    pub(in crate::qwen3_5) fn prepare_speculative_prefill_selection(
        &mut self,
        active_request: &mut Qwen3_5EngineRequest,
        scoring_start_position_tokens: usize,
        final_prompt_index: usize,
    ) -> Result<SpeculativePrefillSelectionPreparation, crate::InferenceEngineError> {
        // Ordinary requests and already-prepared requests must never repeat
        // selection side effects.
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
        // The final prompt token is reserved for generation kickoff. Planning
        // scores the complete prompt but marks only the uncached conversation
        // interval before that token as selectable.
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
            // Reuse installs both host selection positions and the complete GPU
            // prompt-token array, exactly as fresh selection would.
            active_request.speculative_prefill_scoring_attempted = true;
            return Ok(SpeculativePrefillSelectionPreparation::Ready);
        }
        if !active_request.speculative_prefill_draft_phase_announced {
            // Yield before expensive drafter work so streaming clients can
            // observe an accurate phase transition rather than a late event.
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
        // Decoder state is request-local even though compatible draft weights
        // are reconstructed from the configured artifact each time.
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
            // In-memory checkpoints intentionally exclude visual requests because
            // their key lacks ordered image digests. Disk roots bind those digests.
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
            // Worker memory is fastest. On a miss, fall through to the durable
            // block chain; both restore the same dense decoder-state semantics.
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
        // Work-reuse counters describe logical prompt work, independent of
        // whether restored state came from worker memory or disk.
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
        // Scoring output is indexed from the complete logical draft prompt, while
        // target selection may start after a restored/dense target prefix. Map
        // the target interval explicitly before slicing GPU scores.
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
        // Admission occurs before visual projection and scoring so all long-lived
        // and boundary-overlap allocations are covered by one exact reservation.
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
            // Defensive guard for future orchestration that may disable the
            // policy during visual preparation. Current paths normally stay true.
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
        // A disabled store uses one as an inert boundary value because no
        // consumer exists; the model will never publish a block in that case.
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
                         // Visual and text-only scoring share position/policy
                         // parameters; only embedding injection differs.
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
        // End closure borrows before reading the side-band persistence flag or
        // moving scoring output into GPU selection.
        drop(persistent_prompt_cache_block_consumer);
        drop(persist_completed_draft_block);
        let should_force_selection_failure = active_request
            .take_forced_speculative_prefill_failure_for_tests(
                Qwen3_5SpeculativePrefillFailureStageForTests::Selection,
            );
        let draft_scoring_outcome = active_request.performance_attribution.measure_operation(
            PerformanceOperation::SpeculativePrefillSelection,
            |_performance_attribution| {
                draft_scoring_outcome.and_then(|draft_scoring_outcome| {
                    if should_force_selection_failure {
                        return Err(Qwen3_5ExecutionError::InvalidInput {
                            description: "forced speculative-prefill selection failure",
                        });
                    }
                    select_absolute_speculative_prefill_positions_from_draft_scores(
                        &draft_model,
                        draft_scoring_outcome,
                        selectable_importance_score_range.clone(),
                        scoring_start_position_tokens,
                        self.speculative_prefill.keep_percentage,
                        self.speculative_prefill.selection_chunck_token_count,
                        self.speculative_prefill.mandatory_trailing_token_count,
                    )
                })
            },
        );
        self.capture_speculative_prefill_draft_memory_and_release_decoder_state(
            active_request,
            &draft_model,
            draft_request_decoder_state,
            draft_visual_embeddings.as_ref(),
        )?;
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
        self.release_speculative_prefill_draft_and_resume_target_retention(
            active_request,
            draft_visual_embeddings,
            draft_model,
        )?;
        Ok(SpeculativePrefillSelectionPreparation::Ready)
    }
}
