use crate::inference_engine::contract::ExpertResidencyTelemetry;
use crate::qwen3_5_core::text::thinking_budget::Qwen3_5ThinkingBudgetError;
use crate::{
    EngineGenerationStart, InferenceEngineError, PerformanceAttribution,
    PerformanceAttributionOutcome, PerformanceCounter, Qwen3_5InferenceRequest,
    Qwen3_5SamplingStrategy, Qwen3_5ThinkingBudgetState,
};

use mlx_bridge::MlxArray;

use super::engine_request::{Qwen3_5EngineRequest, ResidentRequestExtension};
use super::{Qwen3_5EngineState, qwen3_5_runtime_error};
use crate::astro_decoder_state_cache::AstroDecoderStateCacheLayout;
use crate::astro_ssd_prompt_cache::block_causal_input::AstroSsdPromptCacheBlockCausalInput;
use crate::qwen3_5_core::decoder::RequestDecoderStateStack;
use crate::qwen3_5_core::shared_execution::astro_ssd_prompt_cache_visual_identity::{
    Qwen3_5AstroSsdPromptCacheVisualIdentity, Qwen3_5AstroSsdPromptCacheVisualIdentityInput,
};
use crate::qwen3_5_core::shared_execution::generation_request_validation::{
    self, Qwen3_5GenerationRequestLimits,
};
use crate::qwen3_5_core::text::sampler;
use crate::qwen3_5_resident::model::request_errors;
use crate::sampling_seed::current_time_millis_since_unix_epoch;

/// The prompt-side identity of a generation session: token list, image-pad
/// accounting, `astro_ssd_prompt_cache` eligibility flags, speculative admission, and the
/// visual astro SSD prompt cache identity they feed.
struct GenerationPromptIdentity {
    prompt_token_ids: Vec<u32>,
    image_pad_token_id: u32,
    prompt_image_pad_token_count: usize,
    has_precomputed_visual_embeddings: bool,
    has_processed_visual_images: bool,
    can_use_astro_ssd_prompt_cache: bool,
    speculative_session_admitted: bool,
    ordered_image_visual_embedding_row_counts: Vec<usize>,
    astro_ssd_prompt_cache_block_causal_inputs: Vec<AstroSsdPromptCacheBlockCausalInput>,
}

/// Everything the active-request installation consumes, grouped so the
/// installation reads as one named phase instead of a positional argument
/// tower.
struct InstallActiveGenerationRequest<'a> {
    request_decoder_state: RequestDecoderStateStack,
    prompt_identity: GenerationPromptIdentity,
    visual_embeddings: Option<MlxArray>,
    sampling_strategy: Qwen3_5SamplingStrategy,
    random_state: Option<MlxArray>,
    inference_request: &'a mut Qwen3_5InferenceRequest,
    thinking_budget_state: Qwen3_5ThinkingBudgetState,
    generation_position: super::generation_restore_position_derivation::GenerationSessionPosition,
    performance_attribution: PerformanceAttribution,
}

impl Qwen3_5EngineState {
    /// Admits one chat request into the engine and assembles its generation
    /// session. This is the first step of inference for a fully resident
    /// Qwen3.5 mixture-of-experts model, and it runs no model math: every
    /// transformer layer and expert executes later, during prompt prefill
    /// (the pass that processes the input tokens) and decoding (the pass that
    /// emits answer tokens one at a time). This function validates the
    /// request, reserves memory, restores any reusable prompt state, and
    /// installs the per-request session that later steps drive.
    pub(in crate::qwen3_5_resident::inference_execution) fn start_generation(
        &mut self,
        mut inference_request: Qwen3_5InferenceRequest,
    ) -> Result<EngineGenerationStart, InferenceEngineError> {
        // An `astro_ssd_prompt_cache` restore or the vision encoder uses the GPU before any forward.
        self.tell_neural_engine_preparation_the_gpu_is_in_use();
        let request_id = inference_request.request_id();
        let configured_maximum_output_tokens = inference_request.max_output_tokens();
        // Per-request performance measurement (timings and counters) travels
        // inside the request itself; taking it out here means rejection paths
        // below can still record why a request never started.
        let taken_performance_attribution = inference_request.take_performance_attribution();
        let mut performance_attribution =
            self.ensure_no_active_generation(&inference_request, taken_performance_attribution)?;
        // Early validation: the model must be loaded, the prompt must contain
        // at least one token, the output budget must be positive, and a
        // thinking budget must reserve room for its forced transition. The
        // returned value is the total context reservation — prompt tokens plus
        // every token the model may still generate — which must fit inside the
        // model's context window (its maximum sequence length).
        let reserved_context_tokens =
            generation_request_validation::validate_generation_request_and_resolve_reserved_context(
                &self.core.generation_request_limits(),
                &inference_request,
            )?;
        // Qwen3.5 can "think" before answering: it emits a private reasoning
        // block bounded by model markers, then produces the visible answer.
        // The unit appends the model's own think-end marker as a safety net so
        // a thinking block can always be closed, even when the caller supplied
        // no tokenizer-derived boundary.
        let natural_reasoning_end_token_ids =
            crate::qwen3_5_core::shared_execution::generation_thinking_end_markers::natural_reasoning_end_token_ids_with_model_safety_net(
                inference_request.natural_reasoning_end_token_ids(),
                self.core.think_end_token_id,
            );
        // Turns the caller's thinking allowance into a small state machine
        // with three phases — thinking, forcing the transition out of
        // thinking, and visible answer — so the engine can cut reasoning off
        // once the allowance is spent while still leaving room for a complete
        // final answer inside the output budget.
        let thinking_budget_state = Qwen3_5ThinkingBudgetState::new(
            inference_request.generation_starts_inside_thinking_block(),
            inference_request.thinking_budget(),
            inference_request
                .forced_thinking_transition_token_ids()
                .to_vec(),
            natural_reasoning_end_token_ids,
        )
        .map_err(|source| {
            request_errors::invalid_request_error(format!(
                "invalid Qwen3.5 thinking-budget configuration: {source}"
            ))
        })?;
        // Borrow the loaded model weights. "Fully resident" in this engine
        // means every expert of this mixture-of-experts model already sits in
        // graphics memory; each token routes through only a few experts, and
        // nothing streams in from disk while generating.
        let model = self
            .core
            .model
            .as_ref()
            .ok_or_else(super::generation_error_constructors::engine_lost_loaded_model_error)?;
        // The `astro_decoder_state_cache` layout describes, per decoder layer, the shape of the
        // key-value state — the model's memory of tokens it has already seen.
        // Recording attention once per token is what lets each new token be
        // processed without re-reading the whole conversation.
        let astro_decoder_state_cache_layout = model.astro_decoder_state_cache_layout().clone();
        // Memory gate: before any graphics memory is touched, verify the full
        // reserved context (prompt plus potential output) fits the live memory
        // limit. A rejection here is recorded into performance attribution so
        // operators can later see why a request never started.
        self.admit_initial_generation_context_or_record_rejection(
            request_id,
            configured_maximum_output_tokens,
            reserved_context_tokens,
            &mut performance_attribution,
        )?;
        // Everything below can still fail: `?` now returns from this function
        // directly, and the engine's single active-request slot is only filled
        // by the final statement of this straight-line body.
        // This is deliberately NOT a transaction: on a mid-way failure the
        // active request stays empty, but engine-side effects already
        // performed persist (astro SSD prompt cache counters, an adopted
        // retained live prefix), and the moved-in performance attribution is
        // dropped rather than recorded as a rejection — unlike the two gates
        // above, which record every rejection they issue.
        let sampling_strategy = inference_request.sampling_strategy();
        let random_state = self.resolve_sampling_random_state(sampling_strategy)?;
        let prompt_identity = self.resolve_prompt_identity(
            &inference_request,
            sampling_strategy,
            &mut performance_attribution,
        )?;
        // Image requests can arrive with visual embeddings already
        // computed by an earlier pipeline stage; this validates and uploads
        // them into the array prefill will splice over the placeholders.
        let precomputed_visual_embeddings = self.prepare_precomputed_visual_embeddings(
            &inference_request,
            prompt_identity.prompt_image_pad_token_count,
        )?;
        // The per-request `astro_decoder_state_cache`: one state owner per decoder
        // layer, allocated empty here and grown as tokens are processed.
        // Full-attention layers grow in the configured slab granularity so
        // resize cost stays amortized across the request's lifetime.
        let mut request_decoder_state =
            RequestDecoderStateStack::empty_from_astro_decoder_state_cache_layout_with_full_attention_kv_state_growth_tokens(
                &astro_decoder_state_cache_layout,
                self.core.full_attention_kv_state_growth_tokens,
            )
            .map_err(qwen3_5_runtime_error)?;
        // Restore the longest prompt prefix already present in the
        // `astro_ssd_prompt_cache`. On success `prefill_cursor` moves past the
        // restored tokens so prefill computes only the new suffix, and
        // `next_position_tokens` keeps token positions continuous across
        // the restored boundary.
        let restored_astro_ssd_prompt_cache_outcome = self
            .restore_astro_ssd_prompt_cache_prefix_outcome(
                &inference_request,
                &prompt_identity.prompt_token_ids,
                &prompt_identity.astro_ssd_prompt_cache_block_causal_inputs,
                prompt_identity.can_use_astro_ssd_prompt_cache,
                reserved_context_tokens,
                &mut request_decoder_state,
                &mut performance_attribution,
            )?;
        let generation_position =
            super::generation_restore_position_derivation::derive_generation_session_position(
                restored_astro_ssd_prompt_cache_outcome,
            );
        // If the request instead carries raw processed images, the vision
        // encoder runs now: plan where each image's embedding rows land in
        // the not-yet-restored suffix, then encode pixels into rows.
        let visual_embeddings = self.resolve_session_visual_embeddings(
            &inference_request,
            precomputed_visual_embeddings,
            prompt_identity.has_processed_visual_images,
            &prompt_identity.prompt_token_ids,
            generation_position.prefill_cursor,
            &prompt_identity.ordered_image_visual_embedding_row_counts,
            prompt_identity.image_pad_token_id,
            &mut performance_attribution,
        )?;
        // Every check passed: install the session. This object owns all
        // mutable state for the generation that follows — the
        // `astro_decoder_state_cache` stack, the prompt, the prefill cursor, the sampler, the
        // thinking-budget machine, and the counters. From here on,
        // `advance_generation` drives prefill and then decoding.
        self.install_active_generation_request(InstallActiveGenerationRequest {
            request_decoder_state,
            prompt_identity,
            visual_embeddings,
            sampling_strategy,
            random_state,
            inference_request: &mut inference_request,
            thinking_budget_state,
            generation_position,
            performance_attribution,
        })
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
        // This engine serves exactly one request at a time: a model resident
        // in graphics memory plus one active session is all it can hold, so a
        // second concurrent request is rejected outright instead of queued.
        if self.core.active_request.is_none() {
            return Ok(performance_attribution);
        }
        crate::qwen3_5_core::shared_execution::generation_finalization_steps::record_generation_performance_attribution(
            self,
            crate::qwen3_5_core::shared_execution::generation_finalization_steps::GenerationAttributionRecord {
                performance_attribution,
                outcome: PerformanceAttributionOutcome::Rejected,
                request_id: inference_request.request_id(),
                configured_maximum_output_tokens: inference_request.max_output_tokens(),
                mlx_memory_snapshot: None,
                failure_description: Some("generation engine is already serving a request"),
            },
        );
        Err(InferenceEngineError::EngineBusy)
    }

    /// Sampling decides how the next token is picked from the model's
    /// probability distribution. HighestLogit is greedy — no randomness
    /// needed. TopKTopP samples randomly and gets validated here, with one
    /// reusable randomness buffer created up front on the device so
    /// per-token sampling later stays cheap.
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
                let model = self.core.model.as_ref().ok_or_else(
                    super::generation_error_constructors::engine_lost_loaded_model_error,
                )?;
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

    /// Resolves the prompt-side session identity: token list, image-pad
    /// accounting, `astro_ssd_prompt_cache` eligibility, speculative admission, and the visual
    /// astro SSD prompt cache identity text and image reuse share.
    fn resolve_prompt_identity(
        &mut self,
        inference_request: &Qwen3_5InferenceRequest,
        sampling_strategy: Qwen3_5SamplingStrategy,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<GenerationPromptIdentity, InferenceEngineError> {
        // The conversation as the model sees it: a list of integer token
        // IDs. Image content has no token form, so image positions carry a
        // placeholder ID whose attention rows the vision encoder's output
        // will later overwrite; counting the placeholders lets the engine
        // verify the encoder produced exactly one embedding row per
        // placeholder.
        let prompt_token_ids = inference_request.input_token_ids().to_vec();
        let image_pad_token_id = inference_request
            .image_pad_token_id()
            .ok_or_else(super::generation_error_constructors::missing_image_pad_token_id_error)?;
        let prompt_image_pad_token_count =
            super::generation_visual_embedding_validation::count_image_pad_placeholder_tokens(
                &prompt_token_ids,
                image_pad_token_id,
            );
        let has_precomputed_visual_embeddings = inference_request.has_visual_embeddings();
        let has_processed_visual_images = inference_request.has_processed_visual_images();
        // The `astro_ssd_prompt_cache` stores key-value blocks on
        // disk, keyed by prompt prefix; a repeated prefix restores instead
        // of recomputing. Persistent snapshots contain target decoder state
        // only, so target-only execution is preferred whenever the
        // `astro_ssd_prompt_cache` is available.
        let astro_ssd_prompt_cache_is_available = self.core.astro_ssd_prompt_cache.is_some();
        let can_use_astro_ssd_prompt_cache =
            super::generation_session_decisions::can_use_astro_ssd_prompt_cache(
                astro_ssd_prompt_cache_is_available,
                has_precomputed_visual_embeddings,
            );
        // Speculative admission, decided once from request properties plus
        // draft residency. An admitted session uses the
        // `astro_ssd_prompt_cache` like any other: the drafter's context then
        // starts at the restored boundary and grows from the taps of the
        // suffix that prefill computes.
        let speculative_session_admitted = self
            .speculative_draft_runtime
            .as_ref()
            .is_some_and(|_| {
                super::super::speculative_decode_lifecycle::speculation_session_admission::should_admit_speculation_for_session(
                    &super::super::speculative_decode_lifecycle::speculation_session_admission::SpeculationSessionRequestProperties {
                        sampling_strategy: super::super::speculative_decode_lifecycle::speculation_session_admission::verification_token_choice_for_strategy(&sampling_strategy),
                        structured_generation_is_active: inference_request
                            .structured_generation()
                            .is_some(),
                        forced_thinking_transition_is_pending: false,
                        visual_embeddings_are_active: has_precomputed_visual_embeddings
                            || has_processed_visual_images,
                        repetition_penalty_is_active: inference_request
                            .repetition_penalty_thousandths()
                            != 1_000,
                    },
                )
            });
        // Computes which prompt blocks are eligible for the `astro_ssd_prompt_cache` and how many
        // visual embedding rows each image contributes, so text and image
        // reuse decisions come from one consistent identity.
        let visual_astro_ssd_prompt_cache_identity =
            Qwen3_5AstroSsdPromptCacheVisualIdentity::prepare(
                inference_request,
                Qwen3_5AstroSsdPromptCacheVisualIdentityInput {
                    prompt_token_ids: &prompt_token_ids,
                    prompt_image_pad_token_count,
                    image_pad_token_id,
                    astro_ssd_prompt_cache: self.core.astro_ssd_prompt_cache.as_deref(),
                    can_use_astro_ssd_prompt_cache,
                },
                performance_attribution,
            )?;
        let ordered_image_visual_embedding_row_counts =
            visual_astro_ssd_prompt_cache_identity.ordered_image_visual_embedding_row_counts;
        let astro_ssd_prompt_cache_block_causal_inputs =
            visual_astro_ssd_prompt_cache_identity.block_causal_inputs;
        Ok(GenerationPromptIdentity {
            prompt_token_ids,
            image_pad_token_id,
            prompt_image_pad_token_count,
            has_precomputed_visual_embeddings,
            has_processed_visual_images,
            can_use_astro_ssd_prompt_cache,
            speculative_session_admitted,
            ordered_image_visual_embedding_row_counts,
            astro_ssd_prompt_cache_block_causal_inputs,
        })
    }

    /// Validates and uploads caller-supplied visual embeddings — row count
    /// must match the image-placeholder count and the buffer size must match
    /// the model's embedding width — into the graphics-processor array that
    /// prefill will splice over the image placeholders.
    fn prepare_precomputed_visual_embeddings(
        &self,
        inference_request: &Qwen3_5InferenceRequest,
        prompt_image_pad_token_count: usize,
    ) -> Result<Option<MlxArray>, InferenceEngineError> {
        let Some(visual_embedding_values) = inference_request.visual_embeddings() else {
            return Ok(None);
        };
        let visual_embedding_row_count = inference_request.visual_embedding_row_count();
        super::generation_visual_embedding_validation::validate_precomputed_visual_embedding_row_counts(
            prompt_image_pad_token_count,
            visual_embedding_row_count,
            visual_embedding_values.len(),
        )
        .map_err(
            super::generation_visual_embedding_validation::VisualEmbeddingValidationError::into_generation_failure,
        )?;
        let model = self
            .core
            .model
            .as_ref()
            .ok_or_else(super::generation_error_constructors::engine_lost_loaded_model_error)?;
        let visual_embedding_hidden_size = self
            .core
            .persistent_visual_embedding_model_contract
            .as_ref()
            .ok_or_else(
                super::generation_error_constructors::visual_embedding_model_contract_not_loaded_error,
            )?
            .visual_embedding_hidden_size();
        super::generation_visual_embedding_validation::validate_precomputed_visual_embedding_buffer_size(
            visual_embedding_row_count,
            visual_embedding_values.len(),
            visual_embedding_hidden_size,
        )
        .map_err(
            super::generation_visual_embedding_validation::VisualEmbeddingValidationError::into_generation_failure,
        )?;
        Ok(Some(
            model
                .runtime()
                .array_from_f32(
                    visual_embedding_values,
                    &[
                        i32::try_from(visual_embedding_row_count).map_err(
                            super::generation_error_constructors::visual_embedding_row_count_exceeds_i32_range_error,
                        )?,
                        i32::try_from(visual_embedding_hidden_size).map_err(
                            super::generation_error_constructors::visual_embedding_hidden_size_exceeds_i32_range_error,
                        )?,
                    ],
                )
                .map_err(qwen3_5_runtime_error)?,
        ))
    }

    /// Restores the longest prompt prefix already present in the
    /// `astro_ssd_prompt_cache`, returning the restore outcome for position derivation. The
    /// disk-store owner is always reinstalled before any path escapes.
    #[allow(clippy::too_many_arguments)]
    fn restore_astro_ssd_prompt_cache_prefix_outcome(
        &mut self,
        inference_request: &Qwen3_5InferenceRequest,
        prompt_token_ids: &[u32],
        astro_ssd_prompt_cache_block_causal_inputs: &[AstroSsdPromptCacheBlockCausalInput],
        can_use_astro_ssd_prompt_cache: bool,
        reserved_context_tokens: usize,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<
        Option<crate::qwen3_5_core::shared_execution::astro_ssd_prompt_cache_restore::AstroSsdPromptCacheRestoreOutcome>,
        InferenceEngineError,
    >{
        if self.core.astro_ssd_prompt_cache.is_some() && can_use_astro_ssd_prompt_cache {
            // Split the borrow: take the `astro_ssd_prompt_cache` out temporarily so the engine
            // state (including counters) can be mutated as &mut self while the
            // disk store is used as a plain borrowed reference.
            let astro_ssd_prompt_cache = self.core.astro_ssd_prompt_cache.take();
            let restore_result = if let Some(astro_ssd_prompt_cache) =
                astro_ssd_prompt_cache.as_ref()
            {
                crate::qwen3_5_core::shared_execution::astro_ssd_prompt_cache_restore::restore_astro_ssd_prompt_cache_prefix(
                        self,
                        &crate::qwen3_5_core::shared_execution::astro_ssd_prompt_cache_restore::AstroSsdPromptCacheRestoreRequest {
                            request_id: inference_request.request_id(),
                            astro_ssd_prompt_cache,
                            prompt_token_ids,
                            block_causal_inputs: astro_ssd_prompt_cache_block_causal_inputs,
                            total_context_tokens: reserved_context_tokens,
                        },
                        request_decoder_state,
                        performance_attribution,
                    )
                    .map(Some)
            } else {
                Ok(None)
            };
            // Restores can fail closed for data correctness, but the disk-store owner must
            // remain installed for the next independent request. Returning while it is
            // temporarily taken would silently turn later traffic into a mode with the `astro_ssd_prompt_cache` disabled.
            self.core.astro_ssd_prompt_cache = astro_ssd_prompt_cache;
            return restore_result;
        }
        Ok(None)
    }

    /// Resolves the session's visual embeddings: caller-supplied arrays pass
    /// through, processed images run the vision encoder against the
    /// not-yet-restored suffix plan, and text-only requests carry none.
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
                request_errors::invalid_request_error(format!(
                    "visual embedding suffix planning failed: {visual_embedding_suffix_plan_error}"
                ))
            })?;
        super::visual_embeddings_runtime::resolve_visual_embeddings_for_processed_images(
            self,
            inference_request.request_id(),
            inference_request.processed_visual_images(),
            &visual_embedding_suffix_plan,
            performance_attribution,
        )
    }

    /// Installs the assembled session as the engine's active request and
    /// builds the generation-start contract the caller observes. The session
    /// log and the prompt-size counters record before the attribution moves
    /// into the request, exactly as the original single body did.
    fn install_active_generation_request(
        &mut self,
        install_inputs: InstallActiveGenerationRequest<'_>,
    ) -> Result<EngineGenerationStart, InferenceEngineError> {
        let InstallActiveGenerationRequest {
            request_decoder_state,
            prompt_identity,
            visual_embeddings,
            sampling_strategy,
            random_state,
            inference_request,
            thinking_budget_state,
            generation_position,
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
        // Observability: one log line plus two counters (prompt size, and
        // how much of it came from the `astro_ssd_prompt_cache`) so each request's
        // behavior stays measurable after the fact.
        tracing::info!(
            request_id = inference_request.request_id().value(),
            astro_ssd_prompt_cache_is_available = self.core.astro_ssd_prompt_cache.is_some(),
            sampling_selects_highest_logit,
            effective_temperature_thousandths,
            "resolved target-only generation request session"
        );
        performance_attribution.record_counter(
            PerformanceCounter::PromptTokenCount,
            u64::try_from(prompt_identity.prompt_token_ids.len()).unwrap_or(u64::MAX),
        );
        performance_attribution.record_counter(
            PerformanceCounter::AstroSsdPromptCacheRestoredTokenCount,
            u64::from(generation_position.astro_ssd_prompt_cache_restored_token_count),
        );
        let target_eligible_prompt_work_token_count =
            u64::try_from(prompt_identity.prompt_token_ids.len()).unwrap_or(u64::MAX);
        // If any prompt tokens remain beyond the restored prefix, the
        // first work this request performs is prompt prefill (the "Target"
        // phase); a fully restored prompt skips prefill and starts
        // directly at decoding.
        let prompt_prefill_end_exclusive = prompt_identity.prompt_token_ids.len();
        let initial_prompt_processing_phase =
            super::generation_session_decisions::initial_prompt_processing_phase(
                generation_position.prefill_cursor,
                prompt_prefill_end_exclusive,
            );
        let engine_extension = ResidentRequestExtension::for_new_request(
            prompt_identity.speculative_session_admitted,
            generation_position.prefill_cursor,
            prompt_prefill_end_exclusive,
            sampling_strategy,
        );
        self.core.active_request = Some(Qwen3_5EngineRequest {
            request_decoder_state,
            generated_token_count: 0,
            input_token_ids: prompt_identity.prompt_token_ids,
            last_restored_astro_ssd_prompt_cache_block_key: generation_position
                .last_restored_astro_ssd_prompt_cache_block_key,
            can_use_astro_ssd_prompt_cache: prompt_identity.can_use_astro_ssd_prompt_cache,
            maximum_output_tokens: inference_request.max_output_tokens(),
            astro_ssd_prompt_cache_block_causal_inputs: prompt_identity
                .astro_ssd_prompt_cache_block_causal_inputs,
            next_position_tokens: generation_position.next_position_tokens,
            pending_generated_token: None,
            // How many prompt tokens are already considered processed —
            // restored tokens count without being recomputed.
            prefill_cursor: generation_position.prefill_cursor,
            random_state,
            request_id: inference_request.request_id(),
            sampling_strategy,
            repetition_penalty_thousandths: inference_request.repetition_penalty_thousandths(),
            penalized_token_ids: inference_request.initial_penalized_token_ids(),
            visual_embeddings,
            consumed_visual_embedding_count: 0,
            has_visual_inputs: prompt_identity.has_precomputed_visual_embeddings
                || prompt_identity.has_processed_visual_images,
            image_pad_token_id: prompt_identity.image_pad_token_id,
            thinking_budget_state,
            performance_attribution,
            prompt_work_reuse: ipc_protocol::WorkerPromptWorkReuse {
                target_eligible_token_count: target_eligible_prompt_work_token_count,
                target_restored_token_count: generation_position.restored_target_work_token_count,
            },
            astro_ssd_prompt_cache_diagnostics: generation_position
                .astro_ssd_prompt_cache_diagnostics
                .clone(),
            generation_residency_preparation_attempted: false,
            first_decode_forward_elapsed_millis: None,
            outgrown_attention_slab_reclaim: Default::default(),
            generation_preparation_announced: false,
            structured_generation: inference_request.take_structured_generation(),
            engine_extension,
        });
        let restored_prompt_prefix_token_count = u32::try_from(generation_position.prefill_cursor)
            .map_err(
                super::generation_error_constructors::restored_prompt_prefix_exceeds_u32_range_error,
            )?;
        // Tell the caller how generation will proceed: how many prompt
        // tokens were restored rather than computed, which expert memory
        // mode the loaded model uses (fully resident in this engine), and
        // whether the first advance is prefill or decoding.
        self.build_generation_start_contract(
            restored_prompt_prefix_token_count,
            initial_prompt_processing_phase,
            generation_position.astro_ssd_prompt_cache_diagnostics,
        )
    }

    /// Builds the caller-visible generation-start contract: the restored
    /// cursor, the loaded model's expert memory mode, the first processing
    /// phase, and the `astro_ssd_prompt_cache` diagnostics.
    fn build_generation_start_contract(
        &self,
        restored_prompt_prefix_token_count: u32,
        initial_prompt_processing_phase: Option<ipc_protocol::WorkerPromptProcessingPhase>,
        astro_ssd_prompt_cache_diagnostics: Option<
            ipc_protocol::WorkerAstroSsdPromptCacheRequestDiagnostics,
        >,
    ) -> Result<EngineGenerationStart, InferenceEngineError> {
        Ok(EngineGenerationStart::with_expert_memory_mode(
            restored_prompt_prefix_token_count,
            self.core
                .model
                .as_ref()
                .ok_or_else(super::generation_error_constructors::engine_lost_loaded_model_error)?
                .expert_memory_mode(),
        )
        .with_restored_prompt_prefix_token_count(restored_prompt_prefix_token_count)
        .with_prompt_processing_phase(initial_prompt_processing_phase)
        .with_astro_ssd_prompt_cache_diagnostics(astro_ssd_prompt_cache_diagnostics))
    }
}
