use crate::qwen3_5_core::shared_execution::session_visual_embeddings;
use crate::{
    EngineGenerationStart, InferenceEngineError, PerformanceAttribution, TextInferenceRequest,
    TextSamplingStrategy, ThinkingBudgetState,
};
use text_generation::generation_loop::astro_ssd_prompt_cache_restore;
use text_generation::generation_loop::generation_start_steps;

use mlx_bridge::MlxArray;

use super::engine_request::Qwen3_5RequestExtension;
use super::{Qwen3_5EngineState, fatal_runtime_error};
use crate::astro_decoder_state_cache::AstroDecoderStateCacheLayout;
use crate::qwen3_5_core::shared_execution::astro_ssd_prompt_cache_visual_identity::{
    Qwen3_5AstroSsdPromptCacheVisualIdentity, Qwen3_5AstroSsdPromptCacheVisualIdentityInput,
};
use engine_core::astro_ssd_prompt_cache::block_causal_input::AstroSsdPromptCacheBlockCausalInput;
use text_generation::RequestDecoderStateStack;
use text_generation::engine_errors::invalid_request_error;
use text_generation::generation_loop::engine_request_installation::{
    AdmittedGenerationSession, new_engine_request,
};
use text_generation::generation_loop::generation_request_validation;
use text_generation::generation_loop::generation_session_position::{
    GenerationSessionPosition, derive_generation_session_position,
};

/// Everything the active-request installation consumes, grouped so the
/// installation reads as one named phase instead of a positional argument
/// tower.
struct InstallActiveGenerationRequest<'a> {
    request_decoder_state: RequestDecoderStateStack,
    prompt_token_ids: Vec<u32>,
    sampling_strategy: TextSamplingStrategy,
    random_state: Option<MlxArray>,
    inference_request: &'a mut TextInferenceRequest,
    thinking_budget_state: ThinkingBudgetState,
    reserved_context_tokens: usize,
    can_use_astro_ssd_prompt_cache: bool,
    astro_ssd_prompt_cache_block_causal_inputs: Vec<AstroSsdPromptCacheBlockCausalInput>,
    restored_prompt_prefix_session: GenerationSessionPosition,
    visual_embeddings: Option<MlxArray>,
    has_precomputed_visual_embeddings: bool,
    has_processed_visual_images: bool,
    image_pad_token_id: u32,
    performance_attribution: PerformanceAttribution,
}

/// What a generation request says about its prompt, read once before the
/// session is assembled.
struct PromptFacts {
    prompt_token_ids: Vec<u32>,
    image_pad_token_id: u32,
    prompt_image_pad_token_count: usize,
    has_precomputed_visual_embeddings: bool,
    has_processed_visual_images: bool,
    can_use_astro_ssd_prompt_cache: bool,
}

impl Qwen3_5EngineState {
    pub(super) fn start_generation(
        &mut self,
        mut inference_request: TextInferenceRequest,
    ) -> Result<EngineGenerationStart, InferenceEngineError> {
        // An `astro_ssd_prompt_cache` restore or the vision encoder uses the
        // GPU before any forward, which a waiting program measurement must
        // not read through.
        self.neural_engine_prefill.note_gpu_used();
        let request_id = inference_request.request_id();
        let taken_performance_attribution = inference_request.take_performance_attribution();
        let performance_attribution = generation_start_steps::ensure_no_active_generation(
            self,
            &inference_request,
            taken_performance_attribution,
        )?;
        let reserved_context_tokens =
            generation_request_validation::validate_generation_request_and_resolve_reserved_context(
                &self.core.generation_request_limits(),
                &inference_request,
            )?;
        let thinking_budget_state = self.build_thinking_budget_state(&inference_request)?;
        let model = self.core.loaded_model()?;
        model.forget_expert_layer_states();
        // A fresh request observes a fresh SSD-read tally so the label
        // emitted across this request's lifecycle reflects only this
        // request's expert storage reads (issue #1533).
        model.begin_request_expert_storage_read_observation();
        let astro_decoder_state_cache_layout = model.astro_decoder_state_cache_layout().clone();
        // No request is refused for memory (issue #1902): the governor sets
        // the prompt's stored state aside and the kept weights give back
        // what that displaces.
        self.govern_request_start(&inference_request);
        // Everything below can still fail; a request that never starts must
        // not leave the memory governor holding its prompt's stored state.
        let admitted_generation_start = self.assemble_admitted_generation_session(
            &mut inference_request,
            &astro_decoder_state_cache_layout,
            reserved_context_tokens,
            thinking_budget_state,
            performance_attribution,
        );
        if admitted_generation_start.is_err()
            && let Some(model) = self.core.model.as_ref()
        {
            model.govern_request_end(request_id.value());
        }
        admitted_generation_start
    }

    /// Tells the memory governor a request starts reading its prompt, in
    /// spans of at most the configured chunk.
    fn govern_request_start(&self, inference_request: &TextInferenceRequest) {
        let Some(model) = self.core.model.as_ref() else {
            return;
        };
        model.govern_request_start(
            u64::try_from(inference_request.input_token_ids().len()).unwrap_or(u64::MAX),
            u32::try_from(
                self.core
                    .prompt_processing_chunk_sizer
                    .maximum_prompt_processing_chunk_size_tokens(),
            )
            .unwrap_or(u32::MAX),
        );
    }

    /// Builds the request's thinking-budget state machine through the shared
    /// start step.
    fn build_thinking_budget_state(
        &self,
        inference_request: &TextInferenceRequest,
    ) -> Result<ThinkingBudgetState, InferenceEngineError> {
        generation_start_steps::build_thinking_budget_state(
            self.core.think_end_token_id,
            inference_request,
        )
    }

    /// Assembles the admitted session: sampling state, prompt metadata, visual
    /// identity and embeddings, `astro_ssd_prompt_cache` prefix restore, and the active-request
    /// installation. Runs after admission; every failure here resumes expert
    /// retention in the caller.
    fn assemble_admitted_generation_session(
        &mut self,
        inference_request: &mut TextInferenceRequest,
        astro_decoder_state_cache_layout: &AstroDecoderStateCacheLayout,
        reserved_context_tokens: usize,
        thinking_budget_state: ThinkingBudgetState,
        mut performance_attribution: PerformanceAttribution,
    ) -> Result<EngineGenerationStart, InferenceEngineError> {
        let sampling_strategy = inference_request.sampling_strategy();
        let random_state =
            generation_start_steps::resolve_sampling_random_state(self, sampling_strategy)?;
        let prompt_facts = self.read_prompt_facts(inference_request)?;
        let visual_astro_ssd_prompt_cache_identity = self
            .prepare_visual_astro_ssd_prompt_cache_identity(
                inference_request,
                &prompt_facts,
                &mut performance_attribution,
            )?;
        let precomputed_visual_embeddings =
            generation_start_steps::prepare_precomputed_visual_embeddings(
                self,
                inference_request,
                prompt_facts.prompt_image_pad_token_count,
            )?;
        let mut request_decoder_state =
            self.empty_request_decoder_state(astro_decoder_state_cache_layout)?;
        let restored_prompt_prefix_session = self.restore_astro_ssd_prompt_cache_prefix_session(
            inference_request,
            &prompt_facts.prompt_token_ids,
            &visual_astro_ssd_prompt_cache_identity.block_causal_inputs,
            prompt_facts.can_use_astro_ssd_prompt_cache,
            reserved_context_tokens,
            &mut request_decoder_state,
            &mut performance_attribution,
        )?;
        let visual_embeddings = session_visual_embeddings::resolve_session_visual_embeddings(
            self,
            inference_request,
            precomputed_visual_embeddings,
            prompt_facts.has_processed_visual_images,
            session_visual_embeddings::SessionVisualEmbeddingPlacement {
                prompt_token_ids: &prompt_facts.prompt_token_ids,
                prefill_cursor: restored_prompt_prefix_session.prefill_cursor,
                ordered_image_visual_embedding_row_counts: &visual_astro_ssd_prompt_cache_identity
                    .ordered_image_visual_embedding_row_counts,
                image_pad_token_id: prompt_facts.image_pad_token_id,
            },
            &mut performance_attribution,
        )?;
        self.install_active_generation_request(InstallActiveGenerationRequest {
            request_decoder_state,
            prompt_token_ids: prompt_facts.prompt_token_ids,
            sampling_strategy,
            random_state,
            inference_request,
            thinking_budget_state,
            reserved_context_tokens,
            can_use_astro_ssd_prompt_cache: prompt_facts.can_use_astro_ssd_prompt_cache,
            astro_ssd_prompt_cache_block_causal_inputs: visual_astro_ssd_prompt_cache_identity
                .block_causal_inputs,
            restored_prompt_prefix_session,
            visual_embeddings,
            has_precomputed_visual_embeddings: prompt_facts.has_precomputed_visual_embeddings,
            has_processed_visual_images: prompt_facts.has_processed_visual_images,
            image_pad_token_id: prompt_facts.image_pad_token_id,
            performance_attribution,
        })
    }

    /// Reads what the request says about its prompt: the token ids, how many
    /// of them are image placeholders, which visual inputs it carries, and
    /// whether it may use the `astro_ssd_prompt_cache`.
    fn read_prompt_facts(
        &self,
        inference_request: &TextInferenceRequest,
    ) -> Result<PromptFacts, InferenceEngineError> {
        let prompt_token_ids = inference_request.input_token_ids().to_vec();
        let image_pad_token_id = inference_request.image_pad_token_id().ok_or_else(|| {
            invalid_request_error("generation request is missing the image-pad token ID")
        })?;
        let prompt_image_pad_token_count =
            count_image_pad_tokens(&prompt_token_ids, image_pad_token_id);
        let has_precomputed_visual_embeddings = inference_request.has_visual_embeddings();
        let has_processed_visual_images = inference_request.has_processed_visual_images();
        let can_use_astro_ssd_prompt_cache = request_may_use_astro_ssd_prompt_cache(
            self.core.astro_ssd_prompt_cache.is_some(),
            has_precomputed_visual_embeddings,
        );
        Ok(PromptFacts {
            prompt_token_ids,
            image_pad_token_id,
            prompt_image_pad_token_count,
            has_precomputed_visual_embeddings,
            has_processed_visual_images,
            can_use_astro_ssd_prompt_cache,
        })
    }

    /// Prepares the visual identity the `astro_ssd_prompt_cache` keys a
    /// prompt with images by.
    fn prepare_visual_astro_ssd_prompt_cache_identity(
        &self,
        inference_request: &TextInferenceRequest,
        prompt_facts: &PromptFacts,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<Qwen3_5AstroSsdPromptCacheVisualIdentity, InferenceEngineError> {
        Qwen3_5AstroSsdPromptCacheVisualIdentity::prepare(
            inference_request,
            Qwen3_5AstroSsdPromptCacheVisualIdentityInput {
                prompt_token_ids: &prompt_facts.prompt_token_ids,
                prompt_image_pad_token_count: prompt_facts.prompt_image_pad_token_count,
                image_pad_token_id: prompt_facts.image_pad_token_id,
                astro_ssd_prompt_cache: self.core.astro_ssd_prompt_cache.as_deref(),
                can_use_astro_ssd_prompt_cache: prompt_facts.can_use_astro_ssd_prompt_cache,
            },
            performance_attribution,
        )
    }

    /// A fresh decoder state for the request, sized for the model's layout and
    /// the engine's full-attention growth step.
    fn empty_request_decoder_state(
        &self,
        astro_decoder_state_cache_layout: &AstroDecoderStateCacheLayout,
    ) -> Result<RequestDecoderStateStack, InferenceEngineError> {
        RequestDecoderStateStack::
            empty_from_astro_decoder_state_cache_layout_with_full_attention_kv_state_growth_tokens(
                astro_decoder_state_cache_layout,
                self.core.full_attention_kv_state_growth_tokens,
            )
            .map_err(fatal_runtime_error)
    }

    /// Restores the prompt prefix held in the `astro_ssd_prompt_cache`, when it can
    /// serve this request, and derives the session's starting cursor from the
    /// outcome.
    #[allow(clippy::too_many_arguments)]
    fn restore_astro_ssd_prompt_cache_prefix_session(
        &mut self,
        inference_request: &TextInferenceRequest,
        prompt_token_ids: &[u32],
        astro_ssd_prompt_cache_block_causal_inputs: &[AstroSsdPromptCacheBlockCausalInput],
        can_use_astro_ssd_prompt_cache: bool,
        reserved_context_tokens: usize,
        request_decoder_state: &mut RequestDecoderStateStack,
        performance_attribution: &mut PerformanceAttribution,
    ) -> Result<GenerationSessionPosition, InferenceEngineError> {
        if !can_use_astro_ssd_prompt_cache {
            return Ok(derive_generation_session_position(None));
        }
        let restore_outcome =
            astro_ssd_prompt_cache_restore::restore_prefix_keeping_store_installed(
                self,
                astro_ssd_prompt_cache_restore::AstroSsdPromptCachePrefixRestore {
                    request_id: inference_request.request_id(),
                    prompt_token_ids,
                    block_causal_inputs: astro_ssd_prompt_cache_block_causal_inputs,
                    total_context_tokens: reserved_context_tokens,
                },
                request_decoder_state,
                performance_attribution,
            )?;
        Ok(derive_generation_session_position(restore_outcome))
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
            can_use_astro_ssd_prompt_cache,
            astro_ssd_prompt_cache_block_causal_inputs,
            restored_prompt_prefix_session,
            visual_embeddings,
            has_precomputed_visual_embeddings,
            has_processed_visual_images,
            image_pad_token_id,
            performance_attribution,
        } = install_inputs;
        let initial_prompt_processing_phase = (restored_prompt_prefix_session.prefill_cursor
            < prompt_token_ids.len())
        .then_some(ipc_protocol::WorkerPromptProcessingPhase::Target);
        let astro_ssd_prompt_cache_is_available = self.core.astro_ssd_prompt_cache.is_some();
        self.core.active_request = Some(new_engine_request(
            AdmittedGenerationSession {
                request_decoder_state,
                prompt_token_ids,
                can_use_astro_ssd_prompt_cache,
                astro_ssd_prompt_cache_block_causal_inputs,
                astro_ssd_prompt_cache_is_available,
                session_position: &restored_prompt_prefix_session,
                sampling_strategy,
                random_state,
                inference_request,
                visual_embeddings,
                has_visual_inputs: has_precomputed_visual_embeddings || has_processed_visual_images,
                image_pad_token_id,
                thinking_budget_state,
                performance_attribution,
            },
            Qwen3_5RequestExtension::for_new_request(reserved_context_tokens),
        ));
        let restored_prompt_prefix_token_count =
            u32::try_from(restored_prompt_prefix_session.prefill_cursor).map_err(|_| {
                super::fatal_engine_error("restored prompt prefix exceeds the u32 range")
            })?;
        generation_start_steps::build_generation_start_contract(
            self,
            restored_prompt_prefix_token_count,
            initial_prompt_processing_phase,
            restored_prompt_prefix_session.astro_ssd_prompt_cache_diagnostics,
        )
    }
}

/// How many of the prompt's tokens are image placeholders.
fn count_image_pad_tokens(prompt_token_ids: &[u32], image_pad_token_id: u32) -> usize {
    prompt_token_ids
        .iter()
        .filter(|token_id| **token_id == image_pad_token_id)
        .count()
}

/// Whether the request may use the `astro_ssd_prompt_cache`: it must be
/// available, and a request carrying precomputed visual embeddings never uses
/// it.
///
/// Persistent snapshots currently contain target decoder state only. Prefer target-only
/// execution whenever the `astro_ssd_prompt_cache` is available so optional prediction artifacts retain
/// prompt reuse without restoring an incompatible shifted history.
fn request_may_use_astro_ssd_prompt_cache(
    astro_ssd_prompt_cache_is_available: bool,
    has_precomputed_visual_embeddings: bool,
) -> bool {
    astro_ssd_prompt_cache_is_available && !has_precomputed_visual_embeddings
}

#[cfg(test)]
#[path = "start_generation_tests.rs"]
mod start_generation_tests;
