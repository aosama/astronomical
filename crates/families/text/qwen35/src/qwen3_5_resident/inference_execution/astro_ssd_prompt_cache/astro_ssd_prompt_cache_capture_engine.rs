//! The resident engine's half of the shared astro SSD prompt cache capture.

use crate::InferenceEngineError;
use crate::qwen3_5_resident::inference_execution::Qwen3_5EngineState;
use crate::qwen3_5_resident::inference_execution::engine_request::ResidentEngineRequest;
use crate::qwen3_5_resident::model::Qwen3_5ResidentModel;
use text_generation::astro_ssd_prompt_cache_capture::TextAstroSsdPromptCacheCapture;

impl TextAstroSsdPromptCacheCapture for Qwen3_5EngineState {
    /// Reclaims nothing: the resident engine holds every expert by design,
    /// so there is no pageable payload to evict, and the retry repeats the
    /// first attempt with nothing changed in between (issue #1637).
    fn reclaim_before_publication_retry(
        &self,
        _model: &Qwen3_5ResidentModel,
        _active_request: &mut ResidentEngineRequest,
        _active_memory_deficit_bytes: usize,
    ) -> Result<(), InferenceEngineError> {
        Ok(())
    }
}
