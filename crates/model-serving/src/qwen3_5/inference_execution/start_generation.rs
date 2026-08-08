use crate::{
    EngineGenerationStart, InferenceEngineError, PerformanceAttributionOutcome, PerformanceCounter,
    PerformanceOperation, PersistentPromptCacheBlockKey, Qwen3_5InferenceRequest,
    Qwen3_5SamplingStrategy,
};

use super::super::model::memory_admission::{
    invalid_request_error, validate_context_memory_admission,
};
use super::super::resolve_sampling_seed;
use super::super::text::sampler::{random_state_for_seed, validate_sampled_strategy};
use super::super::text::sampling_seed::current_time_millis_since_unix_epoch;
use super::super::{RequestDecoderStateStack, plan_qwen3_5_visual_embedding_suffix};
use super::engine_request::Qwen3_5EngineRequest;
use super::{
    Qwen3_5EngineState, fatal_engine_error, qwen3_5_runtime_error,
    speculative_prefill_eligibility::qwen3_5_speculative_prefill_request_eligibility,
};
use crate::qwen3_5::multi_token_prediction::create_optional_prediction_session;

impl Qwen3_5EngineState {
    pub(super) fn start_generation(
        &mut self,
        mut inference_request: Qwen3_5InferenceRequest,
    ) -> Result<EngineGenerationStart, InferenceEngineError> {
        let request_id = inference_request.request_id();
        let configured_maximum_output_tokens = inference_request.max_output_tokens();
        let mut performance_attribution = inference_request.take_performance_attribution();
        if self.active_request.is_some() {
            self.record_generation_performance_attribution(
                performance_attribution,
                PerformanceAttributionOutcome::Rejected,
                request_id,
                configured_maximum_output_tokens,
                None,
                Some("generation engine is already serving a request"),
            );
            return Err(InferenceEngineError::EngineBusy);
        }
        if self.model.is_none() {
            return Err(fatal_engine_error("Qwen3.5 engine is not loaded"));
        }
        if inference_request.input_token_ids().is_empty() {
            return Err(fatal_engine_error("generation prompt must not be empty"));
        }
        if inference_request.max_output_tokens() == 0 {
            return Err(fatal_engine_error(
                "generation output-token budget must be positive",
            ));
        }
        if inference_request
            .input_token_ids()
            .iter()
            .any(|token_id| *token_id >= self.vocabulary_size)
        {
            return Err(fatal_engine_error(
                "generation prompt contains a token outside the certified vocabulary",
            ));
        }
        let total_context_tokens = inference_request
            .input_token_ids()
            .len()
            .checked_add(usize::from(inference_request.max_output_tokens()))
            .ok_or_else(|| invalid_request_error("generation context token count overflowed"))?;
        if total_context_tokens > self.maximum_position_count {
            return Err(invalid_request_error(
                "generation context exceeds the certified maximum position count",
            ));
        }
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let decoder_cache_layout = model.decoder_cache_layout().clone();
        let model_has_optional_prediction_head = model.mtp_weights();
        let expert_weight_memory_cache_statistics_at_request_start =
            if performance_attribution.is_enabled() {
                model.expert_weight_memory_cache_statistics()
            } else {
                Default::default()
            };
        let context_admission_outcome = self.validate_initial_generation_context_memory_admission(
            total_context_tokens,
            &mut performance_attribution,
        );
        if let Err(context_admission_error) = context_admission_outcome {
            self.record_generation_performance_attribution(
                performance_attribution,
                PerformanceAttributionOutcome::Rejected,
                request_id,
                configured_maximum_output_tokens,
                None,
                Some("generation context admission rejected"),
            );
            return Err(context_admission_error);
        }
        let admitted_generation_start = (|| {
            let sampling_strategy = inference_request.sampling_strategy();
            let random_state = match sampling_strategy {
                Qwen3_5SamplingStrategy::Greedy => None,
                Qwen3_5SamplingStrategy::TopKTopP {
                    temperature_thousandths,
                    top_k,
                    top_p_thousandths,
                    seed,
                } => {
                    validate_sampled_strategy(temperature_thousandths, top_k, top_p_thousandths)?;
                    let model = self.model.as_ref().ok_or_else(|| {
                        fatal_engine_error("Qwen3.5 engine lost its loaded model")
                    })?;
                    Some(random_state_for_seed(
                        model,
                        resolve_sampling_seed(seed, current_time_millis_since_unix_epoch),
                    )?)
                }
            };
            let prompt_token_ids = inference_request.input_token_ids().to_vec();
            let image_pad_token_id = inference_request.image_pad_token_id().ok_or_else(|| {
                invalid_request_error("generation request is missing the image-pad token ID")
            })?;
            let prompt_image_pad_token_count = prompt_token_ids
                .iter()
                .filter(|token_id| **token_id == image_pad_token_id)
                .count();
            let has_precomputed_visual_embeddings = inference_request.has_visual_embeddings();
            let has_processed_visual_images = inference_request.has_processed_visual_images();
            let persistent_prompt_cache_is_available = self.persistent_prompt_cache.is_some();
            // Persistent snapshots currently contain target decoder state only. Prefer target-only
            // execution whenever the cache is available so optional prediction artifacts retain
            // prompt reuse without restoring an incompatible shifted history.
            let mut can_use_persistent_prompt_cache = !has_precomputed_visual_embeddings;
            let ordered_image_sha256_digests = if has_processed_visual_images {
                inference_request
                    .processed_visual_images()
                    .iter()
                    .map(|processed_visual_image| processed_visual_image.encoded_image_sha256)
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };
            let ordered_image_visual_embedding_row_counts = if has_processed_visual_images {
                inference_request
                    .processed_visual_images()
                    .iter()
                    .map(|processed_visual_image| {
                        processed_visual_image.image_token_count_after_spatial_merge
                    })
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };
            let expected_processed_visual_embedding_row_count =
                ordered_image_visual_embedding_row_counts
                    .iter()
                    .try_fold(0usize, |accumulated_visual_token_count, image_row_count| {
                        accumulated_visual_token_count.checked_add(*image_row_count)
                    })
                    .ok_or_else(|| fatal_engine_error("processed image token count overflowed"))?;
            if has_processed_visual_images
                && prompt_image_pad_token_count != expected_processed_visual_embedding_row_count
            {
                return Err(invalid_request_error(
                    "image pad token count does not match processed image token count",
                ));
            }
            let precomputed_visual_embeddings =
                if let Some(visual_embedding_values) = inference_request.visual_embeddings() {
                    let visual_embedding_row_count = inference_request.visual_embedding_row_count();
                    if visual_embedding_values.is_empty() || visual_embedding_row_count == 0 {
                        return Err(fatal_engine_error(
                            "image request has empty visual embeddings",
                        ));
                    }
                    if prompt_image_pad_token_count != visual_embedding_row_count {
                        return Err(invalid_request_error(
                            "image pad token count does not match visual embedding row count",
                        ));
                    }
                    let model = self.model.as_ref().ok_or_else(|| {
                        fatal_engine_error("Qwen3.5 engine lost its loaded model")
                    })?;
                    let visual_embedding_hidden_size = self
                        .persistent_visual_embedding_model_contract
                        .as_ref()
                        .ok_or_else(|| {
                            fatal_engine_error(
                                "Qwen3.5 persistent visual embedding model contract is not loaded",
                            )
                        })?
                        .visual_embedding_hidden_size();
                    if visual_embedding_values.len()
                        != visual_embedding_row_count.saturating_mul(visual_embedding_hidden_size)
                    {
                        return Err(fatal_engine_error(
                            "visual embedding buffer does not match the expected hidden size",
                        ));
                    }
                    Some(
                        model
                            .runtime()
                            .array_from_f32(
                                visual_embedding_values,
                                &[
                                    i32::try_from(visual_embedding_row_count).map_err(|_| {
                                        fatal_engine_error(
                                            "visual embedding row count exceeds the i32 range",
                                        )
                                    })?,
                                    i32::try_from(visual_embedding_hidden_size).map_err(|_| {
                                        fatal_engine_error(
                                            "visual embedding hidden size exceeds the i32 range",
                                        )
                                    })?,
                                ],
                            )
                            .map_err(qwen3_5_runtime_error)?,
                    )
                } else {
                    None
                };
            let mut request_decoder_state =
                RequestDecoderStateStack::empty_from_decoder_cache_layout_with_full_attention_kv_state_growth_tokens(
                    &decoder_cache_layout,
                    self.full_attention_kv_state_growth_tokens,
                )
                .map_err(qwen3_5_runtime_error)?;
            let mut persistent_prompt_cache_token_count: u32 = 0;
            let mut prefill_cursor: usize = 0;
            let mut next_position_tokens: u32 = 0;
            let mut speculative_prefill_restored_target_token_positions = None;
            let mut restored_target_work_token_count = 0_u64;
            let mut restored_sparse_target_state = false;
            let mut last_restored_persistent_prompt_cache_block_key: Option<
                PersistentPromptCacheBlockKey,
            > = None;
            if self.persistent_prompt_cache.is_some() && can_use_persistent_prompt_cache {
                // Split the borrow: take the cache out temporarily so the engine
                // state (including counters) can be mutated as &mut self while the
                // disk store is used as a plain borrowed reference.
                let persistent_prompt_cache = self.persistent_prompt_cache.take();
                let mut persistent_prompt_cache_restore_failed = false;
                let restore_outcome =
                    if let Some(persistent_prompt_cache) = persistent_prompt_cache.as_ref() {
                        match self.restore_persistent_prompt_cache_prefix(
                            inference_request.request_id(),
                            persistent_prompt_cache,
                            &prompt_token_ids,
                            &ordered_image_sha256_digests,
                            total_context_tokens,
                            &mut request_decoder_state,
                            &mut performance_attribution,
                        ) {
                            Ok(restore_outcome) => Some(restore_outcome),
                            Err(persistent_prompt_cache_error) => {
                                tracing::warn!(
                                    "Qwen3.5 persistent prompt-cache restore failed; \
                             falling back to cold prefill: {persistent_prompt_cache_error}"
                                );
                                persistent_prompt_cache_restore_failed = true;
                                None
                            }
                        }
                    } else {
                        None
                    };
                self.persistent_prompt_cache = persistent_prompt_cache;
                if persistent_prompt_cache_restore_failed {
                    let model_after_restore_failure = self.model.as_ref().ok_or_else(|| {
                        fatal_engine_error("Qwen3.5 engine lost its loaded model")
                    })?;
                    request_decoder_state =
                        RequestDecoderStateStack::empty_from_decoder_cache_layout_with_full_attention_kv_state_growth_tokens(
                            model_after_restore_failure.decoder_cache_layout(),
                            self.full_attention_kv_state_growth_tokens,
                        )
                        .map_err(qwen3_5_runtime_error)?;
                    performance_attribution
                        .measure_operation(
                            PerformanceOperation::MlxAllocatorCacheCleanup,
                            |_performance_attribution| {
                                model_after_restore_failure
                                    .runtime()
                                    .synchronize_gpu_stream_and_clear_allocator_cache()
                            },
                        )
                        .map_err(qwen3_5_runtime_error)?;
                    performance_attribution.measure_operation(
                        PerformanceOperation::MemoryAdmissionSnapshot,
                        |_performance_attribution| {
                            validate_context_memory_admission(
                                model_after_restore_failure,
                                self.memory_limits,
                                self.context_memory_reservation_bytes_per_token,
                                total_context_tokens,
                                0,
                                self.speculative_prefill_draft_maximum_expert_page_reservation_bytes(),
                            )
                        },
                    )?;
                }
                if let Some(restore_outcome) = restore_outcome {
                    persistent_prompt_cache_token_count =
                        restore_outcome.persistent_prompt_cache_token_count;
                    prefill_cursor = restore_outcome.restored_token_count;
                    next_position_tokens = restore_outcome.persistent_prompt_cache_token_count;
                    last_restored_persistent_prompt_cache_block_key =
                        restore_outcome.last_restored_persistent_prompt_cache_block_key;
                    restored_target_work_token_count =
                        u64::from(restore_outcome.persistent_prompt_cache_token_count);
                } else {
                    persistent_prompt_cache_token_count = 0;
                    prefill_cursor = 0;
                    next_position_tokens = 0;
                    last_restored_persistent_prompt_cache_block_key = None;
                }
            }
            if self.speculative_prefill.enabled
                && self.speculative_prefill_draft_is_available
                && !has_precomputed_visual_embeddings
            {
                match self.restore_longest_speculative_prefill_target_prefix(
                    &prompt_token_ids,
                    &ordered_image_sha256_digests,
                    prefill_cursor,
                    &mut performance_attribution,
                ) {
                    Ok(Some((restored_request_decoder_state, restored_target_prefix))) => {
                        request_decoder_state = restored_request_decoder_state;
                        prefill_cursor = restored_target_prefix.prompt_prefix_token_count;
                        next_position_tokens = u32::try_from(prefill_cursor).map_err(|_| {
                            fatal_engine_error(
                                "restored speculative-prefill target prefix exceeds u32",
                            )
                        })?;
                        last_restored_persistent_prompt_cache_block_key = None;
                        restored_target_work_token_count = restored_target_prefix
                            .selected_target_token_positions
                            .shape()[0]
                            .max(0)
                            as u64;
                        speculative_prefill_restored_target_token_positions =
                            Some(restored_target_prefix.selected_target_token_positions);
                        restored_sparse_target_state = true;
                        can_use_persistent_prompt_cache = false;
                    }
                    Ok(None) => {}
                    Err(target_state_restore_error) => {
                        tracing::warn!(
                            request_id = inference_request.request_id().value(),
                            error = %target_state_restore_error,
                            "sparse target state restore failed; retaining the exact target prefix"
                        );
                        self.clear_failed_speculative_prefill_target_restore(
                            &mut performance_attribution,
                        )
                        .map_err(qwen3_5_runtime_error)?;
                    }
                }
            }
            let minimum_speculative_prefill_prompt_token_count =
                usize::try_from(self.speculative_prefill.minimum_prompt_tokens).map_err(|_| {
                    fatal_engine_error("speculative-prefill minimum prompt length overflowed")
                })?;
            let restored_target_prompt_prefix_token_count = prefill_cursor;
            let speculative_prefill_request_eligibility =
                qwen3_5_speculative_prefill_request_eligibility(
                    self.speculative_prefill.enabled,
                    self.speculative_prefill_draft_is_available,
                    self.speculative_prefill_draft_supports_processed_visual_images(),
                    prompt_token_ids.len(),
                    minimum_speculative_prefill_prompt_token_count,
                    restored_target_prompt_prefix_token_count,
                    has_precomputed_visual_embeddings,
                    has_processed_visual_images,
                );
            let should_use_speculative_prefill =
                speculative_prefill_request_eligibility.is_eligible();
            tracing::info!(
                request_id = inference_request.request_id().value(),
                speculative_prefill_runtime_state = ?self.speculative_prefill_runtime_state,
                speculative_prefill_enabled = self.speculative_prefill.enabled,
                draft_model_is_available = self.speculative_prefill_draft_is_available,
                prompt_token_count = prompt_token_ids.len(),
                minimum_prompt_token_count = minimum_speculative_prefill_prompt_token_count,
                restored_target_prompt_prefix_token_count,
                has_precomputed_visual_embeddings,
                has_processed_visual_images,
                draft_model_supports_processed_visual_images =
                    self.speculative_prefill_draft_supports_processed_visual_images(),
                eligibility = speculative_prefill_request_eligibility.identifier(),
                should_use_speculative_prefill,
                "evaluated speculative-prefill request eligibility"
            );
            if should_use_speculative_prefill {
                // Sparse decoder state cannot be serialized as an exact dense
                // prompt-cache block. Exact restores have already taken place;
                // disable capture for this request while retaining cache reads.
                can_use_persistent_prompt_cache = false;
            }
            let visual_embeddings = if let Some(precomputed_visual_embeddings) =
                precomputed_visual_embeddings
            {
                Some(precomputed_visual_embeddings)
            } else if has_processed_visual_images {
                let visual_embedding_suffix_plan = plan_qwen3_5_visual_embedding_suffix(
                        &prompt_token_ids,
                        prefill_cursor,
                        &ordered_image_visual_embedding_row_counts,
                        image_pad_token_id,
                    )
                    .map_err(|visual_embedding_suffix_plan_error| {
                        invalid_request_error(format!(
                            "visual embedding suffix planning failed: {visual_embedding_suffix_plan_error}"
                        ))
                    })?;
                self.resolve_visual_embeddings_for_processed_images(
                    inference_request.request_id(),
                    inference_request.processed_visual_images(),
                    &visual_embedding_suffix_plan,
                    &mut performance_attribution,
                )?
            } else {
                None
            };
            let speculative_prefill_processed_visual_images =
                if should_use_speculative_prefill && has_processed_visual_images {
                    inference_request.take_processed_visual_images()
                } else {
                    Vec::new()
                };
            self.prefill_chunck_sizer
                .start_prompt_processing_request(prefill_cursor);
            let optional_prediction_session = create_optional_prediction_session(
                self.mtp_enabled,
                self.mtp_runtime_state == super::Qwen3_5MtpRuntimeState::Active,
                model_has_optional_prediction_head,
                sampling_strategy,
                has_precomputed_visual_embeddings,
                has_processed_visual_images,
                persistent_prompt_cache_is_available,
                prompt_token_ids.len(),
                persistent_prompt_cache_token_count,
                self.full_attention_kv_state_growth_tokens,
            )
            .map_err(qwen3_5_runtime_error)?;
            performance_attribution.record_counter(
                PerformanceCounter::PromptTokenCount,
                u64::try_from(prompt_token_ids.len()).unwrap_or(u64::MAX),
            );
            performance_attribution.record_counter(
                PerformanceCounter::RestoredPersistentPromptCacheTokenCount,
                u64::from(persistent_prompt_cache_token_count),
            );
            let target_eligible_prompt_work_token_count = if restored_sparse_target_state {
                restored_target_work_token_count.saturating_add(
                    u64::try_from(prompt_token_ids.len().saturating_sub(prefill_cursor))
                        .unwrap_or(u64::MAX),
                )
            } else {
                u64::try_from(prompt_token_ids.len()).unwrap_or(u64::MAX)
            };
            self.active_request = Some(Qwen3_5EngineRequest {
                request_decoder_state,
                generated_token_count: 0,
                input_token_ids: prompt_token_ids,
                last_restored_persistent_prompt_cache_block_key,
                can_use_persistent_prompt_cache,
                maximum_output_tokens: inference_request.max_output_tokens(),
                ordered_image_sha256_digests,
                next_position_tokens,
                pending_generated_token: None,
                persistent_prompt_cache_capture_has_stopped: false,
                prefill_cursor,
                maximum_successful_prefill_chunck_tokens: None,
                random_state,
                request_id: inference_request.request_id(),
                sampling_strategy,
                visual_embeddings,
                consumed_visual_embedding_count: 0,
                image_pad_token_id,
                thinking_budget: inference_request.thinking_budget(),
                thinking_token_count: 0,
                is_inside_thinking: true,
                expert_weight_memory_cache_statistics_at_request_start,
                performance_attribution,
                optional_prediction_session,
                should_use_speculative_prefill,
                speculative_prefill_scoring_attempted: false,
                speculative_prefill_selected_token_positions: None,
                speculative_prefill_prompt_token_indices: None,
                speculative_prefill_processed_visual_images,
                speculative_prefill_restored_target_token_positions,
                speculative_prefill_target_expert_payload_bytes_after_draft_release: None,
                prompt_work_reuse: astronomical_ipc_protocol::WorkerPromptWorkReuse {
                    target_eligible_token_count: target_eligible_prompt_work_token_count,
                    target_restored_token_count: restored_target_work_token_count,
                    drafter_eligible_token_count: 0,
                    drafter_restored_token_count: 0,
                },
                force_next_speculative_prefill_draft_prefix_restore_failure_for_tests: false,
                force_next_prefill_capacity_rejection_for_tests: false,
            });
            Ok(EngineGenerationStart::with_expert_memory_mode(
                persistent_prompt_cache_token_count,
                self.model
                    .as_ref()
                    .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?
                    .expert_memory_mode(),
            ))
        })();
        if admitted_generation_start.is_err()
            && let Some(model) = self.model.as_ref()
        {
            model.resume_expert_retention_after_request_memory_pressure();
        }
        admitted_generation_start
    }
}
