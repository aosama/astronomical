//! The streaming engine's request: the shared Qwen3.5 request plus the
//! prefill demand and memory-recovery state only this engine keeps.

pub(in crate::qwen3_5_streaming) type StreamingEngineRequest =
    text_generation::engine_request::TextEngineRequest<StreamingRequestExtension>;

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
    /// The reply's drafting with the model's multi-token-prediction head;
    /// `None` while the reply is written one token a forward.
    pub(in crate::qwen3_5_streaming) mtp_reply_drafting:
        Option<super::mtp_reply_drafting::MtpReplyDrafting>,
    /// The reply's drafting with a hosted DFlash drafter, opened while the
    /// prompt is read; `None` when no drafter is hosted or the reply is
    /// written one token a forward.
    pub(in crate::qwen3_5_streaming) draft_tree_drafting:
        Option<super::draft_tree_reply_drafting::DraftTreeReplyDrafting>,
    /// The rounds this reply drafted and the drafts it kept, whichever
    /// drafter proposed them; status and the reply's usage report it.
    pub(in crate::qwen3_5_streaming) drafting_statistics: ipc_protocol::RequestDraftingStatistics,
}

impl StreamingRequestExtension {
    /// A new request's streaming state: nothing read, recorded or opened yet,
    /// with the context it reserved at admission.
    pub(in crate::qwen3_5_streaming) fn for_new_request(reserved_context_tokens: usize) -> Self {
        Self {
            has_opened_prefill_demand_window: false,
            reserved_context_tokens,
            layer_at_a_time_prefill: None,
            mlx_buffer_cache_reply_record: Default::default(),
            reply_expert_route_record: Default::default(),
            mtp_reply_drafting: None,
            draft_tree_drafting: None,
            drafting_statistics: ipc_protocol::RequestDraftingStatistics::NONE,
        }
    }
}
