//! Steps every text engine take when a request starts: refusing a second
//! request, resolving the sampling seed, preparing the request's visual
//! embeddings, and building the start contract the worker observes.
//!
//! What a request start does around them (memory admission, drafting, expert
//! residency) stays with each engine.

use crate::model::TextServingModel;
use mlx_bridge::MlxArray;

use crate::engine_errors::{fatal_engine_error, fatal_runtime_error, invalid_request_error};
use crate::request::sampler;
use crate::sampling_seed::current_time_millis_since_unix_epoch;
use crate::sampling_seed::resolve_sampling_seed;
use crate::{
    EngineGenerationStart, InferenceEngineError, PerformanceAttribution,
    PerformanceAttributionOutcome, ThinkingBudgetState,
};
use crate::{TextInferenceRequest, TextSamplingStrategy};

use super::execution_core::TextExecutionCoreOwner;
use super::generation_finalization_steps::{
    GenerationAttributionRecord, TextFinalizationHooks, record_generation_performance_attribution,
};
use super::generation_visual_embedding_validation::{
    VisualEmbeddingValidationError, validate_precomputed_visual_embedding_buffer_size,
    validate_precomputed_visual_embedding_row_counts,
};
use crate::model::text_model::TextModel;

/// Refuses a request while another is active, recording the refusal; hands
/// the request's attribution back when the engine is free.
pub fn ensure_no_active_generation<Engine: TextFinalizationHooks>(
    engine: &mut Engine,
    inference_request: &TextInferenceRequest,
    performance_attribution: PerformanceAttribution,
) -> Result<PerformanceAttribution, InferenceEngineError> {
    // An engine serves exactly one request at a time: a second concurrent
    // request is rejected outright instead of queued.
    if engine.execution_core().active_request.is_none() {
        return Ok(performance_attribution);
    }
    record_generation_performance_attribution(
        engine,
        GenerationAttributionRecord {
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

/// Builds the request's thinking-budget state machine, appending the
/// model's own think-end marker as a safety net when the caller supplied no
/// tokenizer-derived boundaries such as tool-call starts. Its three phases
/// (thinking, forcing the transition out of thinking, visible answer) let the
/// engine cut reasoning off once the allowance is spent while still leaving
/// room for a complete final answer inside the output budget.
pub fn build_thinking_budget_state(
    think_end_token_id: u32,
    inference_request: &TextInferenceRequest,
) -> Result<ThinkingBudgetState, InferenceEngineError> {
    let natural_reasoning_end_token_ids =
        super::generation_thinking_end_markers::natural_reasoning_end_token_ids_with_model_safety_net(
            inference_request.natural_reasoning_end_token_ids(),
            think_end_token_id,
        );
    ThinkingBudgetState::new(
        inference_request.generation_starts_inside_thinking_block(),
        inference_request.thinking_budget(),
        inference_request
            .forced_thinking_transition_token_ids()
            .to_vec(),
        natural_reasoning_end_token_ids,
    )
    .map_err(|source| {
        invalid_request_error(format!("invalid thinking-budget configuration: {source}"))
    })
}

/// The sampler's random state: none for greedy decoding, and for sampling a
/// state seeded by the request, or by the clock when it names no seed.
pub fn resolve_sampling_random_state<Engine: TextExecutionCoreOwner>(
    engine: &Engine,
    sampling_strategy: TextSamplingStrategy,
) -> Result<Option<MlxArray>, InferenceEngineError> {
    match sampling_strategy {
        TextSamplingStrategy::HighestLogit => Ok(None),
        TextSamplingStrategy::TopKTopP {
            temperature_thousandths,
            top_k,
            top_p_thousandths,
            seed,
        } => {
            sampler::validate_sampled_strategy(temperature_thousandths, top_k, top_p_thousandths)?;
            Ok(Some(sampler::random_state_for_seed(
                engine.execution_core().loaded_model()?,
                resolve_sampling_seed(seed, current_time_millis_since_unix_epoch),
            )?))
        }
    }
}

/// Validates and materializes the caller-supplied visual embeddings, when
/// the request carries any, into one device array. The row-count rules run
/// before the model-owned hidden size is fetched, so the same malformed
/// request keeps producing the same failure text.
pub fn prepare_precomputed_visual_embeddings<Engine: TextExecutionCoreOwner>(
    engine: &Engine,
    inference_request: &TextInferenceRequest,
    prompt_image_pad_token_count: usize,
) -> Result<Option<MlxArray>, InferenceEngineError> {
    let Some(visual_embedding_values) = inference_request.visual_embeddings() else {
        return Ok(None);
    };
    let visual_embedding_row_count = inference_request.visual_embedding_row_count();
    validate_precomputed_visual_embedding_row_counts(
        prompt_image_pad_token_count,
        visual_embedding_row_count,
        visual_embedding_values.len(),
    )
    .map_err(VisualEmbeddingValidationError::into_generation_failure)?;
    let model = engine.execution_core().loaded_model()?;
    let visual_embedding_hidden_size = engine
        .execution_core()
        .loaded_persistent_visual_embedding_model_contract()?
        .visual_embedding_hidden_size();
    validate_precomputed_visual_embedding_buffer_size(
        visual_embedding_row_count,
        visual_embedding_values.len(),
        visual_embedding_hidden_size,
    )
    .map_err(VisualEmbeddingValidationError::into_generation_failure)?;
    let row_count = i32::try_from(visual_embedding_row_count)
        .map_err(|_| fatal_engine_error("visual embedding row count exceeds the i32 range"))?;
    let hidden_size = i32::try_from(visual_embedding_hidden_size)
        .map_err(|_| fatal_engine_error("visual embedding hidden size exceeds the i32 range"))?;
    Ok(Some(
        model
            .runtime()
            .array_from_f32(visual_embedding_values, &[row_count, hidden_size])
            .map_err(fatal_runtime_error)?,
    ))
}

/// The start contract the worker observes. No forward has run yet, so the
/// expert memory mode is the model's ownership claim: the engine's
/// freshly reset read tally carries no evidence, and reporting it would
/// claim "resident" for every streaming request during prefill. Token
/// emission refines the claim with evidence.
pub fn build_generation_start_contract<Engine>(
    engine: &Engine,
    restored_prompt_prefix_token_count: u32,
    initial_prompt_processing_phase: Option<ipc_protocol::WorkerPromptProcessingPhase>,
    astro_ssd_prompt_cache_diagnostics: Option<
        ipc_protocol::WorkerAstroSsdPromptCacheRequestDiagnostics,
    >,
) -> Result<EngineGenerationStart, InferenceEngineError>
where
    Engine: TextExecutionCoreOwner<Model: TextModel>,
{
    Ok(EngineGenerationStart::with_expert_memory_mode(
        restored_prompt_prefix_token_count,
        engine.execution_core().loaded_model()?.expert_memory_mode(),
    )
    .with_restored_prompt_prefix_token_count(restored_prompt_prefix_token_count)
    .with_prompt_processing_phase(initial_prompt_processing_phase)
    .with_astro_ssd_prompt_cache_diagnostics(astro_ssd_prompt_cache_diagnostics))
}
