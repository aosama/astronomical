//! The streaming engine's request: the shared Qwen3.5 request plus the
//! prefill demand and memory-recovery state only this engine keeps.

pub(in crate::qwen3_5_streaming) type Qwen3_5EngineRequest =
    crate::qwen3_5_core::engine_request::Qwen3_5EngineRequest<StreamingRequestExtension>;

/// Request state only the streaming engine reads.
pub struct StreamingRequestExtension {
    /// Whether this request's prefill has opened its demand window: the
    /// per-request prefill evidence stream starts empty while lifetime
    /// decode evidence persists. Earlier chunks record at weight one; the
    /// final chunk records boosted. A failed attempt reopens the window so
    /// a retry never mixes discarded evidence.
    pub(in crate::qwen3_5_streaming) has_opened_prefill_demand_window: bool,
    /// Context tokens admitted at request start; prefill memory recovery
    /// re-seeds the `astro_ssd_prompt_cache` against this same total.
    pub(in crate::qwen3_5_streaming) reserved_context_tokens: usize,
    /// The prompt being read one layer at a time, between its first layer
    /// and its last.
    pub(in crate::qwen3_5_streaming) layer_at_a_time_prefill:
        Option<super::layer_at_a_time_prefill::LayerAtATimePrefillInFlight>,
    /// How full the `mlx_buffer_cache` was after each reply token.
    pub(in crate::qwen3_5_streaming) mlx_buffer_cache_reply_record:
        crate::memory::mlx_buffer_cache_reply_record::MlxBufferCacheReplyRecord,
    /// Where each reply token found its routed experts.
    pub(in crate::qwen3_5_streaming) reply_expert_route_record:
        crate::memory::reply_expert_route_record::ReplyExpertRouteRecord,
}

impl Qwen3_5EngineRequest {}
