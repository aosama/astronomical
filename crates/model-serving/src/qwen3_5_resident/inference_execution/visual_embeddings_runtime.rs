//! Resident-engine glue for the shared visual-embedding resolution.

use crate::InferenceEngineError;
use crate::PerformanceAttribution;
use crate::qwen3_5_core::vision::VisualEmbeddingEngineContext;
use crate::qwen3_5_core::vision::resolve_visual_embeddings_for_processed_images as resolve_shared_visual_embeddings;
use astronomical_ipc_protocol::RequestId;
use astronomical_mlx_c_rust::MlxArray;

use super::Qwen3_5EngineState;
use crate::qwen3_5_core::vision::{Qwen3_5ProcessedImage, Qwen3_5VisualEmbeddingSuffixPlan};

pub(crate) fn resolve_visual_embeddings_for_processed_images(
    engine_state: &mut Qwen3_5EngineState,
    request_id: RequestId,
    processed_visual_images: &[Qwen3_5ProcessedImage],
    visual_embedding_suffix_plan: &Qwen3_5VisualEmbeddingSuffixPlan,
    performance_attribution: &mut PerformanceAttribution,
) -> Result<Option<MlxArray>, InferenceEngineError> {
    let model = engine_state
        .model
        .as_ref()
        .ok_or_else(|| InferenceEngineError::Fatal {
            reason: "Qwen3.5 engine lost its loaded model".to_owned(),
        })?;
    let visual_embedding_model_contract = engine_state
        .persistent_visual_embedding_model_contract
        .as_ref()
        .ok_or_else(|| InferenceEngineError::Fatal {
            reason: "Qwen3.5 persistent visual embedding model contract is not loaded".to_owned(),
        })?;
    let visual_embedding_context = VisualEmbeddingEngineContext {
        runtime: model.runtime(),
        vision_model: model.vision_model(),
        compiled_elementwise_graphs: &model.compiled_elementwise_graphs,
        visual_embedding_model_contract,
        persistent_prompt_cache: engine_state.persistent_prompt_cache.as_deref(),
    };
    resolve_shared_visual_embeddings(
        visual_embedding_context,
        &mut engine_state.persistent_prompt_cache_counters,
        request_id,
        processed_visual_images,
        visual_embedding_suffix_plan,
        performance_attribution,
    )
}
