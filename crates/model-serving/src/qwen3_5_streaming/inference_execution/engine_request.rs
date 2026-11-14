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
    /// Prefill memory recoveries already spent; bounds recovery so a
    /// genuinely oversized prompt still fails instead of looping.
    pub(in crate::qwen3_5_streaming) completed_prefill_memory_recovery_attempt_count: u8,
    /// Whether expert layers were put back in memory after a recovery, which
    /// happens at most once and earns one further recovery if it overflows.
    pub(in crate::qwen3_5_streaming) post_recovery_expert_refill:
        super::prefill_expert_refill::PostRecoveryExpertRefill,
    /// The prefill workspace reserve in force when a chunk last ran out of
    /// memory. A refill needs a later measurement to have raised it.
    pub(in crate::qwen3_5_streaming) planned_prefill_workspace_bytes_at_recovery: u64,
    /// Set when a layer-major prefill did not fit, so this request's prompt
    /// runs chunk by chunk from then on (issue #1985).
    pub(in crate::qwen3_5_streaming) layer_major_prefill_declined: bool,
}

impl Qwen3_5EngineRequest {
    /// The context the post-forward budget refresh must reserve for.
    ///
    /// Prefill reserves the whole prompt, as the chunk-start residency refresh
    /// does; decode reserves through the next token's position.
    pub(in crate::qwen3_5_streaming) fn request_context_extent(
        &self,
    ) -> super::completed_forward_memory::RequestContextExtent {
        let prompt_token_count = u64::try_from(self.input_token_ids.len()).unwrap_or(u64::MAX);
        let next_forward_context_token_count =
            u64::from(self.next_position_tokens).saturating_add(1);
        super::completed_forward_memory::RequestContextExtent {
            context_token_count: prompt_token_count.max(next_forward_context_token_count),
            live_context_state_payload_bytes: self.request_decoder_state.payload_byte_count(),
        }
    }
}
