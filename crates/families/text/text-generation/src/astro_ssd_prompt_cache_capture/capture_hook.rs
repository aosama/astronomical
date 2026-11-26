//! What a text engine supplies to the shared astro SSD prompt cache capture.

use crate::InferenceEngineError;
use crate::engine_request::TextEngineRequest;
use crate::generation_loop::execution_core::TextExecutionCoreOwner;

/// The engine-specific half of astro SSD prompt cache capture.
///
/// Extraction and publication are identical for every engine; only the
/// response to a publication blocked by the MLX active-memory limit
/// differs. The Qwen3.5 text engine evicts retained expert pages before its
/// single retry (issue #1637).
pub trait TextAstroSsdPromptCacheCapture: TextExecutionCoreOwner {
    /// Frees what this engine can before the one publication retry, given
    /// the bytes the MLX active-memory limit was short by, and records what
    /// it freed in the request's astro SSD prompt cache diagnostics.
    fn reclaim_before_publication_retry(
        &self,
        model: &Self::Model,
        active_request: &mut TextEngineRequest<Self::RequestExtension>,
        active_memory_deficit_bytes: usize,
    ) -> Result<(), InferenceEngineError>;
}
