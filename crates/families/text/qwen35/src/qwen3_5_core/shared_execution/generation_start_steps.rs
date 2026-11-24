//! Steps both Qwen3.5 engines take when a request starts: refusing a second
//! request, resolving the sampling seed, preparing the request's visual
//! embeddings, and building the start contract the worker observes.
//!
//! What a request start does around them (memory admission, drafting and
//! retained-prefix adoption for resident; expert residency for streaming)
//! stays with each engine.

use crate::qwen3_5_core::model::Qwen3_5ServingModel;
use mlx_bridge::MlxArray;

use crate::qwen3_5_core::engine_errors::{
    fatal_engine_error, invalid_request_error, qwen3_5_runtime_error,
};
use crate::qwen3_5_core::text::{
    Qwen3_5InferenceRequest, Qwen3_5SamplingStrategy, resolve_sampling_seed, sampler,
};
use crate::qwen3_5_core::vision::{
    VisualEmbeddingEngineContext, plan_qwen3_5_visual_embedding_suffix,
    resolve_visual_embeddings_for_processed_images,
};
use crate::sampling_seed::current_time_millis_since_unix_epoch;
use crate::{
    EngineGenerationStart, InferenceEngineError, PerformanceAttribution,
    PerformanceAttributionOutcome,
};

use super::execution_core::Qwen3_5ExecutionCoreOwner;
use super::generation_finalization_steps::{
    GenerationAttributionRecord, Qwen3_5FinalizationHooks,
    record_generation_performance_attribution,
};
use super::generation_visual_embedding_validation::{
    VisualEmbeddingValidationError, validate_precomputed_visual_embedding_buffer_size,
    validate_precomputed_visual_embedding_row_counts,
};
use super::text_model::Qwen3_5TextModel;

/// Refuses a request while another is active, recording the refusal; hands
/// the request's attribution back when the engine is free.
pub fn ensure_no_active_generation<Engine: Qwen3_5FinalizationHooks>(
    engine: &mut Engine,
    inference_request: &Qwen3_5InferenceRequest,
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

/// The sampler's random state: none for greedy decoding, and for sampling a
/// state seeded by the request, or by the clock when it names no seed.
pub fn resolve_sampling_random_state<Engine: Qwen3_5ExecutionCoreOwner>(
    engine: &Engine,
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
pub fn prepare_precomputed_visual_embeddings<Engine: Qwen3_5ExecutionCoreOwner>(
    engine: &Engine,
    inference_request: &Qwen3_5InferenceRequest,
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
            .map_err(qwen3_5_runtime_error)?,
    ))
}

/// Where in the prompt a session's visual embeddings go.
pub struct SessionVisualEmbeddingPlacement<'a> {
    pub prompt_token_ids: &'a [u32],
    pub prefill_cursor: usize,
    pub ordered_image_visual_embedding_row_counts: &'a [usize],
    pub image_pad_token_id: u32,
}

/// The session's visual embeddings: the caller's precomputed ones, or those
/// of the request's processed images for the part of the prompt still to be
/// read. The vision tower is released once they are resolved.
pub fn resolve_session_visual_embeddings<Engine>(
    engine: &mut Engine,
    inference_request: &Qwen3_5InferenceRequest,
    precomputed_visual_embeddings: Option<MlxArray>,
    has_processed_visual_images: bool,
    placement: SessionVisualEmbeddingPlacement<'_>,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<Option<MlxArray>, InferenceEngineError>
where
    Engine: Qwen3_5ExecutionCoreOwner<Model: std::ops::DerefMut>,
{
    if let Some(precomputed_visual_embeddings) = precomputed_visual_embeddings {
        return Ok(Some(precomputed_visual_embeddings));
    }
    if !has_processed_visual_images {
        return Ok(None);
    }
    let visual_embedding_suffix_plan = plan_qwen3_5_visual_embedding_suffix(
        placement.prompt_token_ids,
        placement.prefill_cursor,
        placement.ordered_image_visual_embedding_row_counts,
        placement.image_pad_token_id,
    )
    .map_err(|visual_embedding_suffix_plan_error| {
        invalid_request_error(format!(
            "visual embedding suffix planning failed: {visual_embedding_suffix_plan_error}"
        ))
    })?;
    // Separate fields of the core are borrowed at once: the model, its
    // contract and the store read-only, the counters mutably.
    let execution_core = engine.execution_core_mut();
    let visual_embedding_model = execution_core
        .model
        .as_ref()
        .ok_or_else(|| fatal_engine_error("Qwen3.5 engine lost its loaded model"))?;
    let visual_embedding_model_contract = execution_core
        .persistent_visual_embedding_model_contract
        .as_ref()
        .ok_or_else(|| {
            fatal_engine_error("Qwen3.5 persistent visual embedding model contract is not loaded")
        })?;
    let visual_embedding_context = VisualEmbeddingEngineContext {
        runtime: visual_embedding_model.runtime(),
        vision_model: visual_embedding_model.vision_model(),
        compiled_elementwise_graphs: visual_embedding_model.compiled_elementwise_graphs(),
        visual_embedding_model_contract,
        astro_ssd_prompt_cache: execution_core.astro_ssd_prompt_cache.as_deref(),
    };
    let visual_embedding_resolution = resolve_visual_embeddings_for_processed_images(
        visual_embedding_context,
        &mut execution_core.astro_ssd_prompt_cache_counters,
        inference_request.request_id(),
        inference_request.processed_visual_images(),
        &visual_embedding_suffix_plan,
        performance_attribution,
    );
    if let Some(model) = execution_core.model.as_mut() {
        model.release_vision_tower_after_image_request();
    }
    visual_embedding_resolution
}

/// The start contract the worker observes. No forward has run yet, so the
/// expert memory mode is the model's ownership claim: the streaming engine's
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
    Engine: Qwen3_5ExecutionCoreOwner<Model: Qwen3_5TextModel>,
{
    Ok(EngineGenerationStart::with_expert_memory_mode(
        restored_prompt_prefix_token_count,
        engine.execution_core().loaded_model()?.expert_memory_mode(),
    )
    .with_restored_prompt_prefix_token_count(restored_prompt_prefix_token_count)
    .with_prompt_processing_phase(initial_prompt_processing_phase)
    .with_astro_ssd_prompt_cache_diagnostics(astro_ssd_prompt_cache_diagnostics))
}
