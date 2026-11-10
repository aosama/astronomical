//! The resident engine's half of the shared prompt-cache capture.

use crate::InferenceEngineError;
use crate::qwen3_5_core::prompt_cache_capture::Qwen3_5PromptCacheCapture;
use crate::qwen3_5_resident::inference_execution::Qwen3_5EngineState;
use crate::qwen3_5_resident::inference_execution::engine_request::{
    Qwen3_5EngineRequest, ResidentRequestExtension,
};
use crate::qwen3_5_resident::model::Qwen3_5ResidentModel;

impl Qwen3_5PromptCacheCapture for Qwen3_5EngineState {
    type Model = Qwen3_5ResidentModel;
    type RequestExtension = ResidentRequestExtension;

    /// Reclaims nothing: the resident engine holds every expert by design,
    /// so there is no pageable payload to evict, and the retry relies on the
    /// allocator cleanup that already ran before publication.
    fn reclaim_before_publication_retry(
        &self,
        _model: &Qwen3_5ResidentModel,
        _active_request: &mut Qwen3_5EngineRequest,
        _active_memory_deficit_bytes: usize,
    ) -> Result<(), InferenceEngineError> {
        Ok(())
    }
}
