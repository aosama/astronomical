//! Resolving the visual embeddings of a Qwen 3.5 request at its start: the
//! caller's precomputed ones, or those the vision tower computes for the
//! request's processed images. Both Qwen 3.5 engines take this step; it
//! stays with the family because it runs the family's own vision tower.

use mlx_bridge::MlxArray;
use text_generation::TextInferenceRequest;
use text_generation::engine_errors::{fatal_engine_error, invalid_request_error};
use text_generation::generation_loop::execution_core::TextExecutionCoreOwner;
use text_generation::model::TextServingModel;

use crate::qwen3_5_core::model::Qwen3_5VisionTowerOwner;
use crate::qwen3_5_core::vision::{
    VisualEmbeddingEngineContext, plan_qwen3_5_visual_embedding_suffix,
    resolve_visual_embeddings_for_processed_images,
};
use crate::{InferenceEngineError, PerformanceAttribution};

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
    inference_request: &TextInferenceRequest,
    precomputed_visual_embeddings: Option<MlxArray>,
    has_processed_visual_images: bool,
    placement: SessionVisualEmbeddingPlacement<'_>,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<Option<MlxArray>, InferenceEngineError>
where
    Engine: TextExecutionCoreOwner<Model: Qwen3_5VisionTowerOwner>,
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
