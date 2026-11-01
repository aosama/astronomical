use crate::{
    EngineGenerationStart, InferenceEngineError, PerformanceAttribution,
    PerformanceAttributionOutcome, PerformanceCounter, PersistentPromptCacheBlockKey,
    Qwen3_5InferenceRequest, Qwen3_5SamplingStrategy, Qwen3_5ThinkingBudgetState,
};

use astronomical_ipc_protocol::WorkerPersistentPromptCacheRequestDiagnostics;
use mlx_c_rust::MlxArray;

use super::engine_request::Qwen3_5EngineRequest;
use super::persistent_prompt_cache_visual_identity::{
    Qwen3_5PersistentPromptCacheVisualIdentity, Qwen3_5PersistentPromptCacheVisualIdentityInput,
};
use super::{Qwen3_5EngineState, qwen3_5_runtime_error};
use crate::decoder_cache::DecoderCacheLayout;
use crate::persistent_cache::block_causal_input::PersistentPromptCacheBlockCausalInput;
use crate::qwen3_5_core::decoder::RequestDecoderStateStack;
use crate::qwen3_5_core::text::sampler;
use crate::qwen3_5_streaming::model::memory_admission;
use crate::sampling_seed::current_time_millis_since_unix_epoch;

/// The cache-derived starting cursor of a generation session: what the
/// persistent prompt cache restored, and where prefill therefore resumes.
struct RestoredPromptPrefixSession {
    persistent_prompt_cache_token_count: u32,
    prefill_cursor: usize,
    next_position_tokens: u32,
    last_restored_persistent_prompt_cache_block_key: Option<PersistentPromptCacheBlockKey>,
    persistent_prompt_cache_diagnostics: Option<WorkerPersistentPromptCacheRequestDiagnostics>,
    restored_target_work_token_count: u64,
}

/// Everything the active-request installation consumes, grouped so the
/// installation reads as one named phase instead of a positional argument
/// tower.
struct InstallActiveGenerationRequest<'a> {
    request_decoder_state: RequestDecoderStateStack,
    prompt_token_ids: Vec<u32>,
    sampling_strategy: Qwen3_5SamplingStrategy,
    random_state: Option<MlxArray>,
    inference_request: &'a mut Qwen3_5InferenceRequest,
    thinking_budget_state: Qwen3_5ThinkingBudgetState,
    reserved_context_tokens: usize,
    can_use_persistent_prompt_cache: bool,
    persistent_prompt_cache_block_causal_inputs: Vec<PersistentPromptCacheBlockCausalInput>,
    restored_prompt_prefix_session: RestoredPromptPrefixSession,
    visual_embeddings: Option<MlxArray>,
    has_precomputed_visual_embeddings: bool,
    has_processed_visual_images: bool,
    image_pad_token_id: u32,
    performance_attribution: PerformanceAttribution,
}

impl Qwen3_5EngineState {
    pub(super) fn start_generation(
        &mut self,
        mut inference_request: Qwen3_5InferenceRequest,
    ) -> Result<EngineGenerationStart, InferenceEngineError> {
        let request_id = inference_request.request_id();
        let configured_maximum_output_tokens = inference_request.max_output_tokens();
        let taken_performance_attribution = inference_request.take_performance_attribution();
        let mut performance_attribution =
            self.ensure_no_active_generation(&inference_request, taken_performance_attribution)?;
        let reserved_context_tokens =
            self.validate_generation_request_and_resolve_reserved_context(&inference_request)?;
        let thinking_budget_state = self.build_thinking_budget_state(&inference_request)?;
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| super::fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        model.clear_phase_aware_expert_residency_plan();
        // A fresh request observes a fresh SSD-read tally so the label
        // emitted across this request's lifecycle reflects only this
        // request's expert storage reads (issue #1533).
        model.begin_request_expert_storage_read_observation();
        let decoder_cache_layout = model.decoder_cache_layout().clone();
        self.admit_initial_generation_context_or_record_rejection(
            request_id,
            configured_maximum_output_tokens,
            reserved_context_tokens,
            inference_request.input_token_ids().len(),
            self.persistent_prompt_cache.is_some() && !inference_request.has_visual_embeddings(),
            &mut performance_attribution,
        )?;
        // Everything below can still fail; on failure the engine's
        // expert-retention state must be resumed before the error escapes.
        let admitted_generation_start = self.assemble_admitted_generation_session(
            &mut inference_request,
            &decoder_cache_layout,
            reserved_context_tokens,
            thinking_budget_state,
            performance_attribution,
        );
        if admitted_generation_start.is_err()
            && let Some(model) = self.model.as_ref()
        {
            model.resume_expert_retention_after_request_memory_pressure();
            model.lift_prefill_memory_recovery_freeze();
        }
        admitted_generation_start
    }

    /// Rejects a second concurrent request outright — this engine serves
    /// exactly one generation at a time — recording the rejection into the
    /// performance attribution before handing the attribution back on the
    /// free-engine path.
    fn ensure_no_active_generation(
        &mut self,
        inference_request: &Qwen3_5InferenceRequest,
        performance_attribution: PerformanceAttribution,
    ) -> Result<PerformanceAttribution, InferenceEngineError> {
        if self.active_request.is_none() {
            return Ok(performance_attribution);
        }
        self.record_generation_performance_attribution(
            performance_attribution,
            PerformanceAttributionOutcome::Rejected,
            inference_request.request_id(),
            inference_request.max_output_tokens(),
            None,
            Some("generation engine is already serving a request"),
        );
        Err(InferenceEngineError::EngineBusy)
    }

    /// Builds the request's thinking-budget state machine, appending the
    /// model's own think-end marker as a safety net when the caller supplied
    /// no tokenizer-derived boundaries such as tool-call starts.
    fn build_thinking_budget_state(
        &self,
        inference_request: &Qwen3_5InferenceRequest,
    ) -> Result<Qwen3_5ThinkingBudgetState, InferenceEngineError> {
        let mut natural_reasoning_end_token_ids =
            inference_request.natural_reasoning_end_token_ids().to_vec();
        if !natural_reasoning_end_token_ids.contains(&self.think_end_token_id) {
            // The model thinking marker remains authoritative for direct engine callers that do not
            // pass tokenizer-derived implicit boundaries such as tool-call starts.
            natural_reasoning_end_token_ids.push(self.think_end_token_id);
        }
        Qwen3_5ThinkingBudgetState::new(
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
        })
    }

    /// Assembles the admitted session: sampling state, prompt metadata, visual
    /// identity and embeddings, cached-prefix restore, and the active-request
    /// installation. Runs after admission; every failure here resumes expert
    /// retention in the caller.
    fn assemble_admitted_generation_session(
        &mut self,
        inference_request: &mut Qwen3_5InferenceRequest,
        decoder_cache_layout: &DecoderCacheLayout,
        reserved_context_tokens: usize,
        thinking_budget_state: Qwen3_5ThinkingBudgetState,
        mut performance_attribution: PerformanceAttribution,
    ) -> Result<EngineGenerationStart, InferenceEngineError> {
        let sampling_strategy = inference_request.sampling_strategy();
        let random_state = self.resolve_sampling_random_state(sampling_strategy)?;
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
            inference_request,
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
        let precomputed_visual_embeddings = self.prepare_precomputed_visual_embeddings(
            inference_request,
            prompt_image_pad_token_count,
        )?;
        let mut request_decoder_state = RequestDecoderStateStack::
            empty_from_decoder_cache_layout_with_full_attention_kv_state_growth_tokens(
                decoder_cache_layout,
                self.full_attention_kv_state_growth_tokens,
            )
            .map_err(qwen3_5_runtime_error)?;
        let restored_prompt_prefix_session = self.restore_cached_prompt_prefix_session(
            inference_request,
            &prompt_token_ids,
            &persistent_prompt_cache_block_causal_inputs,
            can_use_persistent_prompt_cache,
            reserved_context_tokens,
            &mut request_decoder_state,
            &mut performance_attribution,
        )?;
        let visual_embeddings = self.resolve_session_visual_embeddings(
            inference_request,
            precomputed_visual_embeddings,
            has_processed_visual_images,
            &prompt_token_ids,
            restored_prompt_prefix_session.prefill_cursor,
            &ordered_image_visual_embedding_row_counts,
            image_pad_token_id,
            &mut performance_attribution,
        )?;
        self.install_active_generation_request(InstallActiveGenerationRequest {
            request_decoder_state,
            prompt_token_ids,
            sampling_strategy,
            random_state,
            inference_request,
            thinking_budget_state,
            reserved_context_tokens,
            can_use_persistent_prompt_cache,
            persistent_prompt_cache_block_causal_inputs,
            restored_prompt_prefix_session,
            visual_embeddings,
            has_precomputed_visual_embeddings,
            has_processed_visual_images,
            image_pad_token_id,
            performance_attribution,
        })
    }

    /// Validates the sampling strategy and resolves its device-side random
    /// state; greedy sampling needs no randomness.
    fn resolve_sampling_random_state(
        &self,
        sampling_strategy: Qwen3_5SamplingStrategy,
    ) -> Result<Option<MlxArray>, InferenceEngineError> {
        match sampling_strategy {
            Qwen3_5SamplingStrategy::HighestLogit => Ok(None),
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
                Ok(Some(sampler::random_state_for_seed(
                    model,
                    crate::qwen3_5_core::text::resolve_sampling_seed(
                        seed,
                        current_time_millis_since_unix_epoch,
                    ),
                )?))
            }
        }
    }

    /// Validates and materializes the caller-supplied visual embeddings, when
    /// the request carries any, into one device array.
    fn prepare_precomputed_visual_embeddings(
        &self,
        inference_request: &Qwen3_5InferenceRequest,
        prompt_image_pad_token_count: usize,
    ) -> Result<Option<MlxArray>, InferenceEngineError> {
        let Some(visual_embedding_values) = inference_request.visual_embeddings() else {
            return Ok(None);
        };
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
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| super::fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
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
        Ok(Some(
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
        ))
    }

    /// Restores the cached prompt prefix, when the persistent prompt cache can
    /// serve this request, and derives the session's starting cursor from the
    /// outcome.
    #[allow(clippy::too_many_arguments)]
    fn restore_cached_prompt_prefix_session(
        &mut self,
        inference_request: &Qwen3_5InferenceRequest,
        prompt_token_ids: &[u32],
        persistent_prompt_cache_block_causal_inputs: &[PersistentPromptCacheBlockCausalInput],
        can_use_persistent_prompt_cache: bool,
        reserved_context_tokens: usize,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<RestoredPromptPrefixSession, InferenceEngineError> {
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
                        prompt_token_ids,
                        persistent_prompt_cache_block_causal_inputs,
                        reserved_context_tokens,
                        request_decoder_state,
                        performance_attribution,
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
                return Ok(RestoredPromptPrefixSession {
                    persistent_prompt_cache_token_count: restore_outcome
                        .persistent_prompt_cache_token_count,
                    prefill_cursor: restore_outcome.restored_token_count,
                    next_position_tokens: restore_outcome.persistent_prompt_cache_token_count,
                    last_restored_persistent_prompt_cache_block_key: restore_outcome
                        .last_restored_persistent_prompt_cache_block_key,
                    persistent_prompt_cache_diagnostics: Some(
                        restore_outcome.persistent_prompt_cache_diagnostics,
                    ),
                    restored_target_work_token_count: u64::from(
                        restore_outcome.persistent_prompt_cache_token_count,
                    ),
                });
            }
        }
        Ok(RestoredPromptPrefixSession {
            persistent_prompt_cache_token_count: 0,
            prefill_cursor: 0,
            next_position_tokens: 0,
            last_restored_persistent_prompt_cache_block_key: None,
            persistent_prompt_cache_diagnostics: None,
            restored_target_work_token_count: 0,
        })
    }

    /// Resolves the session's visual embeddings: caller-supplied arrays pass
    /// through, processed images run the vision encoder, and text-only
    /// requests carry none.
    #[allow(clippy::too_many_arguments)]
    fn resolve_session_visual_embeddings(
        &mut self,
        inference_request: &Qwen3_5InferenceRequest,
        precomputed_visual_embeddings: Option<MlxArray>,
        has_processed_visual_images: bool,
        prompt_token_ids: &[u32],
        prefill_cursor: usize,
        ordered_image_visual_embedding_row_counts: &[usize],
        image_pad_token_id: u32,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Option<MlxArray>, InferenceEngineError> {
        if let Some(precomputed_visual_embeddings) = precomputed_visual_embeddings {
            return Ok(Some(precomputed_visual_embeddings));
        }
        if !has_processed_visual_images {
            return Ok(None);
        }
        let visual_embedding_suffix_plan =
            crate::qwen3_5_core::vision::plan_qwen3_5_visual_embedding_suffix(
                prompt_token_ids,
                prefill_cursor,
                ordered_image_visual_embedding_row_counts,
                image_pad_token_id,
            )
            .map_err(|visual_embedding_suffix_plan_error| {
                memory_admission::invalid_request_error(format!(
                    "visual embedding suffix planning failed: {visual_embedding_suffix_plan_error}"
                ))
            })?;
        let visual_embedding_model = self
            .model
            .as_ref()
            .ok_or_else(|| super::fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
        let visual_embedding_model_contract = self
            .persistent_visual_embedding_model_contract
            .as_ref()
            .ok_or_else(|| {
                super::fatal_engine_error(
                    "Qwen3.5 persistent visual embedding model contract is not loaded",
                )
            })?;
        let visual_embedding_context = crate::qwen3_5_core::vision::VisualEmbeddingEngineContext {
            runtime: visual_embedding_model.runtime(),
            vision_model: visual_embedding_model.vision_model(),
            compiled_elementwise_graphs: &visual_embedding_model.compiled_elementwise_graphs,
            visual_embedding_model_contract,
            persistent_prompt_cache: self.persistent_prompt_cache.as_deref(),
        };
        crate::qwen3_5_core::vision::resolve_visual_embeddings_for_processed_images(
            visual_embedding_context,
            &mut self.persistent_prompt_cache_counters,
            inference_request.request_id(),
            inference_request.processed_visual_images(),
            &visual_embedding_suffix_plan,
            performance_attribution,
        )
    }

    /// Installs the assembled session as the engine's active request and
    /// builds the generation-start contract the caller observes. The
    /// performance attribution moves into the request here, exactly as the
    /// original single body did.
    fn install_active_generation_request(
        &mut self,
        install_inputs: InstallActiveGenerationRequest<'_>,
    ) -> Result<EngineGenerationStart, InferenceEngineError> {
        let InstallActiveGenerationRequest {
            request_decoder_state,
            prompt_token_ids,
            sampling_strategy,
            random_state,
            inference_request,
            thinking_budget_state,
            reserved_context_tokens,
            can_use_persistent_prompt_cache,
            persistent_prompt_cache_block_causal_inputs,
            restored_prompt_prefix_session,
            visual_embeddings,
            has_precomputed_visual_embeddings,
            has_processed_visual_images,
            image_pad_token_id,
            mut performance_attribution,
        } = install_inputs;
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
            persistent_prompt_cache_is_available = self.persistent_prompt_cache.is_some(),
            sampling_selects_highest_logit,
            effective_temperature_thousandths,
            "resolved target-only generation request session"
        );
        performance_attribution.record_counter(
            PerformanceCounter::PromptTokenCount,
            u64::try_from(prompt_token_ids.len()).unwrap_or(u64::MAX),
        );
        performance_attribution.record_counter(
            PerformanceCounter::RestoredPersistentPromptCacheTokenCount,
            u64::from(restored_prompt_prefix_session.persistent_prompt_cache_token_count),
        );
        let target_eligible_prompt_work_token_count =
            u64::try_from(prompt_token_ids.len()).unwrap_or(u64::MAX);
        let prompt_prefill_end_exclusive = prompt_token_ids.len();
        let initial_prompt_processing_phase = (restored_prompt_prefix_session.prefill_cursor
            < prompt_prefill_end_exclusive)
            .then_some(astronomical_ipc_protocol::WorkerPromptProcessingPhase::Target);
        self.active_request = Some(Qwen3_5EngineRequest {
            request_decoder_state,
            generated_token_count: 0,
            input_token_ids: prompt_token_ids,
            last_restored_persistent_prompt_cache_block_key: restored_prompt_prefix_session
                .last_restored_persistent_prompt_cache_block_key,
            can_use_persistent_prompt_cache,
            maximum_output_tokens: inference_request.max_output_tokens(),
            persistent_prompt_cache_block_causal_inputs,
            next_position_tokens: restored_prompt_prefix_session.next_position_tokens,
            pending_generated_token: None,
            prefill_cursor: restored_prompt_prefix_session.prefill_cursor,
            random_state,
            request_id: inference_request.request_id(),
            sampling_strategy,
            repetition_penalty_thousandths: inference_request.repetition_penalty_thousandths(),
            penalized_token_ids: inference_request.initial_penalized_token_ids(),
            visual_embeddings,
            consumed_visual_embedding_count: 0,
            has_visual_inputs: has_precomputed_visual_embeddings || has_processed_visual_images,
            image_pad_token_id,
            thinking_budget_state,
            performance_attribution,
            prompt_work_reuse: astronomical_ipc_protocol::WorkerPromptWorkReuse {
                target_eligible_token_count: target_eligible_prompt_work_token_count,
                target_restored_token_count: restored_prompt_prefix_session
                    .restored_target_work_token_count,
            },
            persistent_prompt_cache_diagnostics: restored_prompt_prefix_session
                .persistent_prompt_cache_diagnostics
                .clone(),
            generation_residency_preparation_attempted: false,
            has_opened_prefill_demand_window: false,
            reserved_context_tokens,
            completed_prefill_memory_recovery_attempt_count: 0,
            first_decode_forward_elapsed_millis: None,
            generation_preparation_announced: false,
            structured_generation: inference_request.take_structured_generation(),
        });
        let restored_prompt_prefix_token_count =
            u32::try_from(restored_prompt_prefix_session.prefill_cursor).map_err(|_| {
                super::fatal_engine_error("restored prompt prefix exceeds the u32 range")
            })?;
        self.build_generation_start_contract(
            restored_prompt_prefix_token_count,
            initial_prompt_processing_phase,
            restored_prompt_prefix_session.persistent_prompt_cache_diagnostics,
        )
    }

    /// Builds the caller-visible generation-start contract from the installed
    /// session's restored cursor.
    fn build_generation_start_contract(
        &self,
        restored_prompt_prefix_token_count: u32,
        initial_prompt_processing_phase: Option<
            astronomical_ipc_protocol::WorkerPromptProcessingPhase,
        >,
        persistent_prompt_cache_diagnostics: Option<WorkerPersistentPromptCacheRequestDiagnostics>,
    ) -> Result<EngineGenerationStart, InferenceEngineError> {
        // No forward has run yet, so the freshly reset read tally carries
        // no evidence; reporting it would claim "resident" for every
        // streaming request during prefill. The start label is the
        // ownership claim, and token emission refines it with evidence.
        Ok(EngineGenerationStart::with_expert_memory_mode(
            restored_prompt_prefix_token_count,
            self.model
                .as_ref()
                .ok_or_else(|| super::fatal_engine_error("Qwen3.5 engine lost its loaded model"))?
                .expert_memory_mode(),
        )
        .with_restored_prompt_prefix_token_count(restored_prompt_prefix_token_count)
        .with_prompt_processing_phase(initial_prompt_processing_phase)
        .with_persistent_prompt_cache_diagnostics(persistent_prompt_cache_diagnostics))
    }
}
