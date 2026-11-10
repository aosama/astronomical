//! The resident engine's request: the shared Qwen3.5 request plus the state
//! of draft-and-verify speculative decoding, which only this engine runs.

pub(in crate::qwen3_5_resident) type Qwen3_5EngineRequest =
    crate::qwen3_5_core::engine_request::Qwen3_5EngineRequest<ResidentRequestExtension>;

/// Request state only the resident engine reads.
pub(crate) struct ResidentRequestExtension {
    /// Draft key-value states for the speculative session; present only
    /// while the session was admitted for draft-and-verify decoding.
    pub(in crate::qwen3_5_resident) speculative_draft_key_value_states: Option<
        crate::qwen3_5_resident::dflash_draft::draft_key_value_state::DFlashDraftKeyValueStates,
    >,
    /// Tokens one accepted speculative round still owes the caller, in
    /// emission order, as host token ids. The tail entry seeds the next
    /// round and is the only queued token absent from the target's decoder
    /// state. Ids stay on the host: the round produced them there, the
    /// emission output consumes them there, and a device round trip would
    /// only add a pipeline-draining synchronization per token.
    pub(in crate::qwen3_5_resident) speculative_emission_queue: std::collections::VecDeque<u32>,
    /// Whether this session runs draft-and-verify; decided once at install
    /// and only ever cleared when a path the block contract cannot preserve
    /// (forced thinking transition, mid-decode feedback injection) suspends
    /// speculation for the remainder of the session.
    pub(in crate::qwen3_5_resident) speculative_session_admitted: bool,
}

impl Qwen3_5EngineRequest {
    /// Whether a speculative round has anything to advance: a pending
    /// successor to verify from, or accepted tokens still owed the caller.
    pub(in crate::qwen3_5_resident) fn has_speculative_tokens_to_advance(&self) -> bool {
        self.pending_generated_token.is_some()
            || !self.engine_extension.speculative_emission_queue.is_empty()
    }

    /// Suspends speculative decoding for this session's remainder.
    ///
    /// Called when a path the block contract cannot preserve takes over:
    /// a forced thinking transition replaces the conditioned successor, and
    /// feedback injection's prefill bypasses the tap-capturing forward that
    /// grows the draft's context cache. The queue (including its uncommitted
    /// tail) and the draft states drop; the classic path owns everything
    /// from here.
    pub(in crate::qwen3_5_resident) fn suspend_speculative_session(
        &mut self,
        suspension_reason: &'static str,
    ) {
        if !self.engine_extension.speculative_session_admitted {
            return;
        }
        self.engine_extension.speculative_session_admitted = false;
        self.engine_extension.speculative_emission_queue.clear();
        self.engine_extension.speculative_draft_key_value_states = None;
        tracing::info!(
            request_id = self.request_id.value(),
            suspension_reason,
            "suspending speculative decoding for the session's remainder"
        );
    }
}
