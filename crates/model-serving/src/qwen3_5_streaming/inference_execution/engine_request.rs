//! The streaming engine's request: the shared Qwen3.5 request plus the
//! prefill demand and memory-recovery state only this engine keeps.

pub(in crate::qwen3_5_streaming) type Qwen3_5EngineRequest =
    crate::qwen3_5_core::engine_request::Qwen3_5EngineRequest<StreamingRequestExtension>;

/// Request state only the streaming engine reads.
pub(crate) struct StreamingRequestExtension {
    /// Whether this request's prefill has opened its demand window: the
    /// per-request prefill evidence stream starts empty while lifetime
    /// decode evidence persists. Earlier chunks record at weight one; the
    /// final chunk records boosted. A failed attempt reopens the window so
    /// a retry never mixes discarded evidence.
    pub(in crate::qwen3_5_streaming) has_opened_prefill_demand_window: bool,
    /// Context tokens admitted at request start; prefill memory recovery
    /// re-seeds the persistent prompt cache against this same total.
    pub(in crate::qwen3_5_streaming) reserved_context_tokens: usize,
    /// The prompt being read one layer at a time, between its first layer
    /// and its last.
    pub(in crate::qwen3_5_streaming) layer_at_a_time_prefill:
        Option<super::layer_at_a_time_prefill::LayerAtATimePrefillInFlight>,
}

impl Qwen3_5EngineRequest {}
