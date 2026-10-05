use crate::{
    EngineGenerationStart, InferenceEngineError, PerformanceAttributionOutcome, PerformanceCounter,
    PersistentPromptCacheBlockKey, Qwen3_5InferenceRequest, Qwen3_5SamplingStrategy,
    Qwen3_5ThinkingBudgetState,
};

use super::super::RequestDecoderStateStack;
use super::super::model::memory_admission;
use super::super::text::sampler;
use super::engine_request::Qwen3_5EngineRequest;
use super::persistent_prompt_cache_visual_identity::{
    Qwen3_5PersistentPromptCacheVisualIdentity, Qwen3_5PersistentPromptCacheVisualIdentityInput,
};
use super::{Qwen3_5EngineState, qwen3_5_runtime_error};
use crate::memory::MtpDraftDepth;
use crate::qwen3_5::multi_token_prediction;
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
            memory_admission::invalid_request_error(format!(
                "invalid Qwen3.5 thinking-budget configuration: {source}"
            ))
        })?;
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| super::fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        model.clear_phase_aware_expert_residency_plan();
        let decoder_cache_layout = model.decoder_cache_layout().clone();
        let model_has_optional_prediction_head = model.mtp_weights();
        self.admit_initial_generation_context_or_record_rejection(
            request_id,
            configured_maximum_output_tokens,
            total_context_tokens,
            inference_request.input_token_ids().len(),
            self.persistent_prompt_cache.is_some() && !inference_request.has_visual_embeddings(),
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
                    sampler::validate_sampled_strategy(
                        temperature_thousandths,
                        top_k,
                        top_p_thousandths,
                    )?;
                    let model = self.model.as_ref().ok_or_else(|| {
                        super::fatal_engine_error("Qwen3.5 engine lost its loaded model")
                    })?;
                    Some(sampler::random_state_for_seed(
                        model,
                        super::super::resolve_sampling_seed(
                            seed,
                            current_time_millis_since_unix_epoch,
                        ),
                    )?)
                }
            };
            let prompt_token_ids = inference_request.input_token_ids().to_vec();
            let image_pad_token_id = inference_request.image_pad_token_id().ok_or_else(|| {
                memory_admission::invalid_request_error(
                    "generation request is missing the image-pad token ID",
                )
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
            let can_use_persistent_prompt_cache =
                persistent_prompt_cache_is_available && !has_precomputed_visual_embeddings;
            let visual_prompt_cache_identity = Qwen3_5PersistentPromptCacheVisualIdentity::prepare(
                &inference_request,
                Qwen3_5PersistentPromptCacheVisualIdentityInput {
                    prompt_token_ids: &prompt_token_ids,
                    prompt_image_pad_token_count,
                    image_pad_token_id,
                    persistent_prompt_cache: self.persistent_prompt_cache.as_deref(),
                    can_use_persistent_prompt_cache,
                },
                &mut performance_attribution,
            )?;
            let ordered_image_visual_embedding_row_counts =
                visual_prompt_cache_identity.ordered_image_visual_embedding_row_counts;
            let persistent_prompt_cache_block_causal_inputs =
                visual_prompt_cache_identity.block_causal_inputs;
            let precomputed_visual_embeddings =
                if let Some(visual_embedding_values) = inference_request.visual_embeddings() {
                    let visual_embedding_row_count = inference_request.visual_embedding_row_count();
                    if visual_embedding_values.is_empty() || visual_embedding_row_count == 0 {
                        return Err(super::fatal_engine_error(
                            "image request has empty visual embeddings",
                        ));
                    }
                    if prompt_image_pad_token_count != visual_embedding_row_count {
                        return Err(memory_admission::invalid_request_error(
                            "image pad token count does not match visual embedding row count",
                        ));
                    }
                    let model = self.model.as_ref().ok_or_else(|| {
                        super::fatal_engine_error("Qwen3.5 engine lost its loaded model")
                    })?;
                    let visual_embedding_hidden_size = self
                        .persistent_visual_embedding_model_contract
                        .as_ref()
                        .ok_or_else(|| {
                            super::fatal_engine_error(
                                "Qwen3.5 persistent visual embedding model contract is not loaded",
                            )
                        })?
                        .visual_embedding_hidden_size();
                    if visual_embedding_values.len()
                        != visual_embedding_row_count.saturating_mul(visual_embedding_hidden_size)
                    {
                        return Err(super::fatal_engine_error(
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
                                        super::fatal_engine_error(
                                            "visual embedding row count exceeds the i32 range",
                                        )
                                    })?,
                                    i32::try_from(visual_embedding_hidden_size).map_err(|_| {
                                        super::fatal_engine_error(
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
            let mut restored_target_work_token_count = 0_u64;
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
            let visual_embeddings = if let Some(precomputed_visual_embeddings) =
                precomputed_visual_embeddings
            {
                Some(precomputed_visual_embeddings)
            } else if has_processed_visual_images {
                let visual_embedding_suffix_plan = super::super::plan_qwen3_5_visual_embedding_suffix(
                        &prompt_token_ids,
                        prefill_cursor,
                        &ordered_image_visual_embedding_row_counts,
                        image_pad_token_id,
                    )
                    .map_err(|visual_embedding_suffix_plan_error| {
                        memory_admission::invalid_request_error(format!(
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
            let sparse_experts_are_paged = self
                .model
                .as_ref()
                .is_some_and(|loaded_model| loaded_model.sparse_experts_are_paged());
            let optional_prediction_session =
                multi_token_prediction::create_optional_prediction_session(
                    self.mtp_enabled,
                    self.mtp_runtime_state == super::Qwen3_5MtpRuntimeState::Active
                        && !inference_request.has_structured_generation(),
                    model_has_optional_prediction_head,
                    has_precomputed_visual_embeddings,
                    has_processed_visual_images,
                    sparse_experts_are_paged,
                    prompt_token_ids.len(),
                    persistent_prompt_cache_token_count,
                    self.full_attention_kv_state_growth_tokens,
                    self.mtp_depth_status
                        .effective_execution_draft_depth
                        .map(MtpDraftDepth::new)
                        .transpose()
                        .map_err(|_| {
                            super::fatal_engine_error("loaded MTP depth is outside 1 through 3")
                        })?,
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
            let target_eligible_prompt_work_token_count =
                u64::try_from(prompt_token_ids.len()).unwrap_or(u64::MAX);
            let prompt_prefill_end_exclusive = prompt_token_ids
                .len()
                .saturating_sub(usize::from(optional_prediction_session.is_some()));
            let initial_prompt_processing_phase = (prefill_cursor < prompt_prefill_end_exclusive)
                .then_some(astronomical_ipc_protocol::WorkerPromptProcessingPhase::Target);
            self.active_request = Some(Qwen3_5EngineRequest {
                request_decoder_state,
                generated_token_count: 0,
                input_token_ids: prompt_token_ids,
                last_restored_persistent_prompt_cache_block_key,
                can_use_persistent_prompt_cache,
                maximum_output_tokens: inference_request.max_output_tokens(),
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
                performance_attribution,
                optional_prediction_session,
                prompt_work_reuse: astronomical_ipc_protocol::WorkerPromptWorkReuse {
                    target_eligible_token_count: target_eligible_prompt_work_token_count,
                    target_restored_token_count: restored_target_work_token_count,
                },
                persistent_prompt_cache_diagnostics: persistent_prompt_cache_diagnostics.clone(),
                force_next_prefill_capacity_rejection_for_tests: false,
                generation_residency_preparation_attempted: false,
                first_decode_forward_elapsed_millis: None,
                generation_preparation_announced: false,
                structured_generation: inference_request.take_structured_generation(),
            });
            let restored_prompt_prefix_token_count =
                u32::try_from(prefill_cursor).map_err(|_| {
                    super::fatal_engine_error("restored prompt prefix exceeds the u32 range")
                })?;
            Ok(EngineGenerationStart::with_expert_memory_mode(
                restored_prompt_prefix_token_count,
                self.model
                    .as_ref()
                    .ok_or_else(|| {
                        super::fatal_engine_error("Qwen3.5 engine lost its loaded model")
                    })?
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
