use crate::{
    EngineGenerationStart, InferenceEngineError, PerformanceAttributionOutcome, PerformanceCounter,
    PersistentPromptCacheBlockKey, Qwen3_5InferenceRequest, Qwen3_5SamplingStrategy,
    Qwen3_5ThinkingBudgetState,
};

use super::super::model::memory_admission::invalid_request_error;
use super::super::resolve_sampling_seed;
use super::super::text::sampler::{random_state_for_seed, validate_sampled_strategy};
use super::super::{RequestDecoderStateStack, plan_qwen3_5_visual_embedding_suffix};
use super::engine_request::Qwen3_5EngineRequest;
use super::persistent_prompt_cache_visual_identity::{
    Qwen3_5PersistentPromptCacheVisualIdentity, Qwen3_5PersistentPromptCacheVisualIdentityInput,
};
use super::{
    Qwen3_5EngineState, fatal_engine_error, qwen3_5_runtime_error,
    speculative_prefill::configured_speculative_prefill_failure,
    speculative_prefill::{
        Qwen3_5SpeculativePrefillRequestEligibility,
        qwen3_5_speculative_prefill_request_eligibility,
    },
};
use crate::qwen3_5::multi_token_prediction::{MtpDraftDepth, create_optional_prediction_session};
use crate::sampling_seed::current_time_millis_since_unix_epoch;

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
        let total_context_tokens =
            self.validate_generation_request_and_resolve_total_context(&inference_request)?;
        let mut natural_reasoning_end_token_ids =
            inference_request.natural_reasoning_end_token_ids().to_vec();
        if !natural_reasoning_end_token_ids.contains(&self.think_end_token_id) {
            // The model thinking marker remains authoritative for direct engine callers that do not
            // pass tokenizer-derived implicit boundaries such as tool-call starts.
            natural_reasoning_end_token_ids.push(self.think_end_token_id);
        }
        let thinking_budget_state = Qwen3_5ThinkingBudgetState::new(
            inference_request.generation_starts_inside_thinking_block(),
            inference_request.thinking_budget(),
            inference_request
                .forced_thinking_transition_token_ids()
                .to_vec(),
            natural_reasoning_end_token_ids,
        )
        .map_err(|source| {
            invalid_request_error(format!(
                "invalid Qwen3.5 thinking-budget configuration: {source}"
            ))
        })?;
        let ordinary_target_prefill_control_span_token_count =
            inference_request.ordinary_target_prefill_control_span_token_count();
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        model.clear_phase_aware_expert_residency_plan();
        let decoder_cache_layout = model.decoder_cache_layout().clone();
        let model_has_optional_prediction_head = model.mtp_weights();
        let expert_weight_memory_cache_statistics_at_request_start =
            if performance_attribution.is_enabled() {
                model.expert_weight_memory_cache_statistics()
            } else {
                Default::default()
            };
        let initial_publication_expert_reclaimed_bytes = self
            .admit_initial_generation_context_or_record_rejection(
                request_id,
                configured_maximum_output_tokens,
                total_context_tokens,
                inference_request.input_token_ids().len(),
                self.persistent_prompt_cache.is_some()
                    && !inference_request.has_visual_embeddings(),
                &mut performance_attribution,
            )?;
        let admitted_generation_start = (|| {
            let sampling_strategy = inference_request.sampling_strategy();
            let random_state = match sampling_strategy {
                Qwen3_5SamplingStrategy::HighestLogit => None,
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
            let mut can_use_persistent_prompt_cache =
                persistent_prompt_cache_is_available && !has_precomputed_visual_embeddings;
            let visual_prompt_cache_identity = Qwen3_5PersistentPromptCacheVisualIdentity::prepare(
                &inference_request,
                Qwen3_5PersistentPromptCacheVisualIdentityInput {
                    prompt_token_ids: &prompt_token_ids,
                    prompt_image_pad_token_count,
                    image_pad_token_id,
                    persistent_prompt_cache: self.persistent_prompt_cache.as_deref(),
                    can_use_persistent_prompt_cache,
                    speculative_prefill_is_enabled: self.speculative_prefill.enabled,
                },
                &mut performance_attribution,
            )?;
            let ordered_image_sha256_digests =
                visual_prompt_cache_identity.ordered_image_sha256_digests;
            let ordered_image_visual_embedding_row_counts =
                visual_prompt_cache_identity.ordered_image_visual_embedding_row_counts;
            let persistent_prompt_cache_block_causal_inputs =
                visual_prompt_cache_identity.block_causal_inputs;
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
            let mut persistent_prompt_cache_diagnostics = None;
            if self.persistent_prompt_cache.is_some() && can_use_persistent_prompt_cache {
                // Split the borrow: take the cache out temporarily so the engine
                // state (including counters) can be mutated as &mut self while the
                // disk store is used as a plain borrowed reference.
                let persistent_prompt_cache = self.persistent_prompt_cache.take();
                let restore_result =
                    if let Some(persistent_prompt_cache) = persistent_prompt_cache.as_ref() {
                        self.restore_persistent_prompt_cache_prefix(
                            inference_request.request_id(),
                            persistent_prompt_cache,
                            &prompt_token_ids,
                            &persistent_prompt_cache_block_causal_inputs,
                            total_context_tokens,
                            &mut request_decoder_state,
                            &mut performance_attribution,
                        )
                        .map(Some)
                    } else {
                        Ok(None)
                    };
                // Restores can fail closed for data correctness, but the disk-store owner must
                // remain installed for the next independent request. Returning while it is
                // temporarily taken would silently turn later traffic into cache-disabled mode.
                self.persistent_prompt_cache = persistent_prompt_cache;
                let restore_outcome = restore_result?;
                if let Some(restore_outcome) = restore_outcome {
                    persistent_prompt_cache_token_count =
                        restore_outcome.persistent_prompt_cache_token_count;
                    prefill_cursor = restore_outcome.restored_token_count;
                    next_position_tokens = restore_outcome.persistent_prompt_cache_token_count;
                    last_restored_persistent_prompt_cache_block_key =
                        restore_outcome.last_restored_persistent_prompt_cache_block_key;
                    persistent_prompt_cache_diagnostics =
                        Some(restore_outcome.persistent_prompt_cache_diagnostics);
                    restored_target_work_token_count =
                        u64::from(restore_outcome.persistent_prompt_cache_token_count);
                } else {
                    persistent_prompt_cache_token_count = 0;
                    prefill_cursor = 0;
                    next_position_tokens = 0;
                    last_restored_persistent_prompt_cache_block_key = None;
                }
            }
            if let Some(persistent_prompt_cache_diagnostics) =
                persistent_prompt_cache_diagnostics.as_mut()
            {
                // Initial admission can evict experts before lookup diagnostics
                // exist. Merge those bytes now so the final per-request record
                // accounts for all publication-related reclamation.
                persistent_prompt_cache_diagnostics.expert_bytes_reclaimed_for_publication =
                    persistent_prompt_cache_diagnostics
                        .expert_bytes_reclaimed_for_publication
                        .saturating_add(initial_publication_expert_reclaimed_bytes);
            }
            if self.speculative_prefill.enabled
                && self.speculative_prefill_draft_is_available
                && !has_precomputed_visual_embeddings
            {
                // Selection-bound sparse target state can advance farther than an
                // ordinary dense cache hit. Restore it only after ordinary lookup
                // so the engine can require a strict improvement and avoid
                // replacing useful state with an equivalent/shorter prefix.
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
                        // Sparse state has its own contract and ancestry. It must
                        // not append ordinary dense prompt-cache blocks.
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
                        return Err(configured_speculative_prefill_failure(
                            inference_request.request_id(),
                            "sparse target-state restoration",
                            target_state_restore_error,
                        ));
                    }
                }
            }
            let minimum_speculative_prefill_prompt_token_count =
                usize::try_from(self.speculative_prefill.minimum_prompt_tokens).map_err(|_| {
                    fatal_engine_error("speculative-prefill minimum prompt length overflowed")
                })?;
            let restored_target_prompt_prefix_token_count = prefill_cursor;
            // Eligibility is intentionally evaluated after both restore systems.
            // The threshold applies to remaining prompt work, not original size.
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
            if speculative_prefill_request_eligibility
                == Qwen3_5SpeculativePrefillRequestEligibility::DraftModelUnavailable
            {
                return Err(configured_speculative_prefill_failure(
                    inference_request.request_id(),
                    "drafter availability validation",
                    "the configured drafter is unavailable",
                ));
            }
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
                    // Transfer processed pixels into the request so the temporary
                    // drafter can project embeddings at its own hidden width.
                    inference_request.take_processed_visual_images()
                } else {
                    Vec::new()
                };
            let sparse_experts_are_paged = self
                .model
                .as_ref()
                .is_some_and(|loaded_model| loaded_model.sparse_experts_are_paged());
            let optional_prediction_session = create_optional_prediction_session(
                self.mtp_enabled,
                self.mtp_runtime_state == super::Qwen3_5MtpRuntimeState::Active,
                model_has_optional_prediction_head,
                sampling_strategy,
                has_precomputed_visual_embeddings,
                has_processed_visual_images,
                persistent_prompt_cache_is_available,
                sparse_experts_are_paged,
                prompt_token_ids.len(),
                persistent_prompt_cache_token_count,
                self.full_attention_kv_state_growth_tokens,
                self.mtp_depth_status
                    .effective_execution_draft_depth
                    .map(MtpDraftDepth::new)
                    .transpose()
                    .map_err(|_| fatal_engine_error("loaded MTP depth is outside 1 through 3"))?,
            )
            .map_err(qwen3_5_runtime_error)?;
            let sampling_selects_highest_logit =
                matches!(sampling_strategy, Qwen3_5SamplingStrategy::HighestLogit);
            let effective_temperature_thousandths = match sampling_strategy {
                Qwen3_5SamplingStrategy::HighestLogit => 0,
                Qwen3_5SamplingStrategy::TopKTopP {
                    temperature_thousandths,
                    ..
                } => temperature_thousandths,
            };
            tracing::info!(
                request_id = inference_request.request_id().value(),
                mtp_runtime_state = ?self.mtp_runtime_state,
                mtp_enabled = self.mtp_enabled,
                sparse_experts_are_paged,
                persistent_prompt_cache_is_available,
                sampling_selects_highest_logit,
                effective_temperature_thousandths,
                optional_prediction_session_is_active = optional_prediction_session.is_some(),
                "resolved optional multi-token prediction request session"
            );
            performance_attribution.record_counter(
                PerformanceCounter::PromptTokenCount,
                u64::try_from(prompt_token_ids.len()).unwrap_or(u64::MAX),
            );
            performance_attribution.record_counter(
                PerformanceCounter::RestoredPersistentPromptCacheTokenCount,
                u64::from(persistent_prompt_cache_token_count),
            );
            if should_use_speculative_prefill {
                performance_attribution.record_counter(
                    PerformanceCounter::SpeculativePrefillOrdinaryControlSpanTokenCount,
                    u64::try_from(ordinary_target_prefill_control_span_token_count)
                        .unwrap_or(u64::MAX),
                );
            }
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
                ordinary_target_prefill_control_span_token_count,
                last_restored_persistent_prompt_cache_block_key,
                can_use_persistent_prompt_cache,
                maximum_output_tokens: inference_request.max_output_tokens(),
                ordered_image_sha256_digests,
                persistent_prompt_cache_block_causal_inputs,
                next_position_tokens,
                pending_generated_token: None,
                prefill_cursor,
                maximum_successful_prefill_chunk_tokens: None,
                random_state,
                request_id: inference_request.request_id(),
                sampling_strategy,
                visual_embeddings,
                consumed_visual_embedding_count: 0,
                has_visual_inputs: has_precomputed_visual_embeddings || has_processed_visual_images,
                image_pad_token_id,
                thinking_budget_state,
                expert_weight_memory_cache_statistics_at_request_start,
                performance_attribution,
                optional_prediction_session,
                should_use_speculative_prefill,
                speculative_prefill_scoring_attempted: false,
                speculative_prefill_draft_phase_announced: false,
                speculative_prefill_selected_token_positions: None,
                speculative_prefill_dense_target_prefix_token_count: 0,
                speculative_prefill_prompt_token_indices: None,
                speculative_prefill_processed_visual_images,
                speculative_prefill_restored_target_token_positions,
                speculative_prefill_target_expert_payload_bytes_after_draft_release: None,
                speculative_prefill_draft_memory_telemetry: None,
                prompt_work_reuse: astronomical_ipc_protocol::WorkerPromptWorkReuse {
                    target_eligible_token_count: target_eligible_prompt_work_token_count,
                    target_restored_token_count: restored_target_work_token_count,
                    drafter_eligible_token_count: 0,
                    drafter_restored_token_count: 0,
                },
                persistent_prompt_cache_diagnostics: persistent_prompt_cache_diagnostics.clone(),
                force_next_speculative_prefill_draft_prefix_restore_failure_for_tests: false,
                forced_speculative_prefill_failure_stage_for_tests: None,
                force_next_prefill_capacity_rejection_for_tests: false,
                generation_residency_preparation_attempted: false,
                first_decode_forward_elapsed_millis: None,
                generation_preparation_announced: false,
            });
            let restored_prompt_prefix_token_count = u32::try_from(prefill_cursor)
                .map_err(|_| fatal_engine_error("restored prompt prefix exceeds the u32 range"))?;
            let initial_prompt_processing_phase = (!should_use_speculative_prefill
                || restored_sparse_target_state
                || prefill_cursor < ordinary_target_prefill_control_span_token_count)
                .then_some(astronomical_ipc_protocol::WorkerPromptProcessingPhase::Target);
            // A fresh eligible request already at the sparse boundary announces
            // Drafter on its first advance instead; all other paths begin in Target.
            Ok(EngineGenerationStart::with_expert_memory_mode(
                persistent_prompt_cache_token_count,
                self.model
                    .as_ref()
                    .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?
                    .expert_memory_mode(),
            )
            .with_restored_prompt_prefix_token_count(restored_prompt_prefix_token_count)
            .with_prompt_processing_phase(initial_prompt_processing_phase)
            .with_persistent_prompt_cache_diagnostics(persistent_prompt_cache_diagnostics))
        })();
        if admitted_generation_start.is_err()
            && let Some(model) = self.model.as_ref()
        {
            model.resume_expert_retention_after_request_memory_pressure();
        }
        admitted_generation_start
    }
}
