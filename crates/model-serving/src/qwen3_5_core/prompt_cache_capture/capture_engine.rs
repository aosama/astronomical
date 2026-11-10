//! What a Qwen3.5 engine supplies to the shared prompt-cache capture.

use crate::InferenceEngineError;
use crate::qwen3_5_core::engine_request::Qwen3_5EngineRequest;
use crate::qwen3_5_core::model::Qwen3_5ModelBase;

/// The engine-specific half of prompt-cache capture.
///
/// Extraction and publication are identical in both engines; only the
/// response to a publication blocked by the MLX active-memory limit
/// differs. The streaming engine evicts retained expert pages before its
/// single retry, while the resident engine holds every expert by design and
/// has nothing to evict, so the retry relies on allocator cleanup alone.
pub(crate) trait Qwen3_5PromptCacheCapture {
    /// The engine's model; capture itself needs only the shared base.
    type Model: std::ops::Deref<Target = Qwen3_5ModelBase>;
    /// The engine-only request state carried by its requests.
    type RequestExtension;

    /// Frees what this engine can before the one publication retry, given
    /// the bytes the MLX active-memory limit was short by, and records what
    /// it freed in the request's prompt-cache diagnostics.
    fn reclaim_before_publication_retry(
        &self,
        model: &Self::Model,
        active_request: &mut Qwen3_5EngineRequest<Self::RequestExtension>,
        active_memory_deficit_bytes: usize,
    ) -> Result<(), InferenceEngineError>;
}
