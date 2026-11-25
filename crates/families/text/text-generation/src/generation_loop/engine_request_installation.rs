//! How any text engine turns an admitted session into its active
//! request (issue #2473): one session log line, the prompt-size counters, and
//! the request itself. Each engine supplies only its own request extension.

use mlx_bridge::MlxArray;

use crate::decoder_state::RequestDecoderStateStack;
use crate::engine_request::TextEngineRequest;
use crate::{
    PerformanceAttribution, PerformanceCounter, TextInferenceRequest, TextSamplingStrategy,
    ThinkingBudgetState,
};
use engine_core::astro_ssd_prompt_cache::block_causal_input::AstroSsdPromptCacheBlockCausalInput;

use super::generation_session_position::GenerationSessionPosition;

/// Everything an admitted session hands to its new request.
pub struct AdmittedGenerationSession<'a> {
    pub request_decoder_state: RequestDecoderStateStack,
    pub prompt_token_ids: Vec<u32>,
    pub can_use_astro_ssd_prompt_cache: bool,
    pub astro_ssd_prompt_cache_block_causal_inputs: Vec<AstroSsdPromptCacheBlockCausalInput>,
    pub astro_ssd_prompt_cache_is_available: bool,
    pub session_position: &'a GenerationSessionPosition,
    pub sampling_strategy: TextSamplingStrategy,
    pub random_state: Option<MlxArray>,
    pub inference_request: &'a mut TextInferenceRequest,
    pub visual_embeddings: Option<MlxArray>,
    pub has_visual_inputs: bool,
    pub image_pad_token_id: u32,
    pub thinking_budget_state: ThinkingBudgetState,
    pub performance_attribution: PerformanceAttribution,
}

/// The active request of an admitted session. The session log and the
/// prompt-size counters record before the attribution moves into the
/// request, so each request's behaviour stays measurable after the fact.
pub fn new_engine_request<RequestExtension>(
    admitted_session: AdmittedGenerationSession<'_>,
    engine_extension: RequestExtension,
) -> TextEngineRequest<RequestExtension> {
    let AdmittedGenerationSession {
        request_decoder_state,
        prompt_token_ids,
        can_use_astro_ssd_prompt_cache,
        astro_ssd_prompt_cache_block_causal_inputs,
        astro_ssd_prompt_cache_is_available,
        session_position,
        sampling_strategy,
        random_state,
        inference_request,
        visual_embeddings,
        has_visual_inputs,
        image_pad_token_id,
        thinking_budget_state,
        mut performance_attribution,
    } = admitted_session;
    log_resolved_generation_session(
        inference_request,
        astro_ssd_prompt_cache_is_available,
        sampling_strategy,
    );
    let prompt_token_count = u64::try_from(prompt_token_ids.len()).unwrap_or(u64::MAX);
    performance_attribution
        .record_counter(PerformanceCounter::PromptTokenCount, prompt_token_count);
    performance_attribution.record_counter(
        PerformanceCounter::AstroSsdPromptCacheRestoredTokenCount,
        u64::from(session_position.astro_ssd_prompt_cache_restored_token_count),
    );
    TextEngineRequest {
        request_decoder_state,
        generated_token_count: 0,
        input_token_ids: prompt_token_ids,
        last_restored_astro_ssd_prompt_cache_block_key: session_position
            .last_restored_astro_ssd_prompt_cache_block_key
            .clone(),
        can_use_astro_ssd_prompt_cache,
        maximum_output_tokens: inference_request.max_output_tokens(),
        astro_ssd_prompt_cache_block_causal_inputs,
        next_position_tokens: session_position.next_position_tokens,
        pending_generated_token: None,
        // How many prompt tokens are already considered processed: restored
        // tokens count without being recomputed.
        prefill_cursor: session_position.prefill_cursor,
        random_state,
        request_id: inference_request.request_id(),
        sampling_strategy,
        repetition_penalty_thousandths: inference_request.repetition_penalty_thousandths(),
        penalized_token_ids: inference_request.initial_penalized_token_ids(),
        visual_embeddings,
        consumed_visual_embedding_count: 0,
        has_visual_inputs,
        image_pad_token_id,
        thinking_budget_state,
        performance_attribution,
        prompt_work_reuse: ipc_protocol::WorkerPromptWorkReuse {
            target_eligible_token_count: prompt_token_count,
            target_restored_token_count: session_position.restored_target_work_token_count,
        },
        astro_ssd_prompt_cache_diagnostics: session_position
            .astro_ssd_prompt_cache_diagnostics
            .clone(),
        generation_residency_preparation_attempted: false,
        first_decode_forward_elapsed_millis: None,
        outgrown_attention_slab_reclaim: Default::default(),
        generation_preparation_announced: false,
        structured_generation: inference_request.take_structured_generation(),
        engine_extension,
    }
}

fn log_resolved_generation_session(
    inference_request: &TextInferenceRequest,
    astro_ssd_prompt_cache_is_available: bool,
    sampling_strategy: TextSamplingStrategy,
) {
    let sampling_selects_highest_logit =
        matches!(sampling_strategy, TextSamplingStrategy::HighestLogit);
    let effective_temperature_thousandths = match sampling_strategy {
        TextSamplingStrategy::HighestLogit => 0,
        TextSamplingStrategy::TopKTopP {
            temperature_thousandths,
            ..
        } => temperature_thousandths,
    };
    tracing::info!(
        request_id = inference_request.request_id().value(),
        astro_ssd_prompt_cache_is_available,
        sampling_selects_highest_logit,
        effective_temperature_thousandths,
        "resolved target-only generation request session"
    );
}
