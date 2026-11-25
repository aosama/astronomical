//! The resident engine's request: the shared Qwen3.5 request plus the state
//! of draft-and-verify speculative decoding, which only this engine runs.

pub(in crate::qwen3_5_resident) type ResidentEngineRequest =
    text_generation::engine_request::TextEngineRequest<ResidentRequestExtension>;

/// Request state only the resident engine reads.
pub struct ResidentRequestExtension {
    /// The drafter's state for the speculative session; present only while
    /// the session was admitted for draft-and-verify decoding.
    pub(in crate::qwen3_5_resident) speculative_draft_session: Option<
        crate::qwen3_5_resident::inference_execution::speculative_decode_lifecycle::draft_tree_round::DraftSession,
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
    /// What drafting has done for this reply so far, reported when it ends.
    pub(in crate::qwen3_5_resident) drafting_statistics: ipc_protocol::RequestDraftingStatistics,
}

impl ResidentRequestExtension {
    /// The resident engine's part of a new request: whether the session
    /// drafts (admitted, and prefill still computes prompt rows for the
    /// drafter to condition on) and the drafting state it then starts from.
    /// A sampled session's key comes from the request's seed (or the clock,
    /// as its sampler's does), so a seeded request samples the same reply
    /// every time.
    pub(in crate::qwen3_5_resident) fn for_new_request(
        speculative_session_admitted: bool,
        restored_prompt_token_count: usize,
        prompt_token_count: usize,
        sampling_strategy: crate::TextSamplingStrategy,
    ) -> Self {
        use crate::qwen3_5_resident::inference_execution::speculative_decode_lifecycle::{
            draft_tree_round::DraftSession,
            speculation_session_admission::drafting_survives_prompt_restore,
        };
        let speculative_session_admitted = drafting_survives_prompt_restore(
            speculative_session_admitted,
            restored_prompt_token_count,
            prompt_token_count,
        );
        let keyed_seed = match sampling_strategy {
            crate::TextSamplingStrategy::HighestLogit => 0,
            crate::TextSamplingStrategy::TopKTopP { seed, .. } => {
                crate::qwen3_5_core::text::resolve_sampling_seed(
                    seed,
                    crate::sampling_seed::current_time_millis_since_unix_epoch,
                )
            }
        };
        Self {
            speculative_draft_session: speculative_session_admitted
                .then_some(DraftSession::new(keyed_seed)),
            speculative_emission_queue: std::collections::VecDeque::new(),
            speculative_session_admitted,
            drafting_statistics: ipc_protocol::RequestDraftingStatistics::default(),
        }
    }
}

/// What the resident engine asks of its request beyond the shell's request
/// state: the speculative decoding kept in its extension. A trait, because
/// the request type belongs to the shell and takes no inherent methods here.
pub(in crate::qwen3_5_resident) trait ResidentSpeculativeRequest {
    /// Whether a speculative round has anything to advance: a pending
    /// successor to verify from, or accepted tokens still owed the caller.
    fn has_speculative_tokens_to_advance(&self) -> bool;

    /// Suspends speculative decoding for this session's remainder.
    ///
    /// Called when a path the block contract cannot preserve takes over:
    /// a forced thinking transition replaces the conditioned successor, and
    /// feedback injection's prefill bypasses the tap-capturing forward that
    /// grows the draft's context key-value state. The queue (including its uncommitted
    /// tail) and the draft states drop; the classic path owns everything
    /// from here.
    fn suspend_speculative_session(&mut self, suspension_reason: &'static str);
}

impl ResidentSpeculativeRequest for ResidentEngineRequest {
    fn has_speculative_tokens_to_advance(&self) -> bool {
        self.pending_generated_token.is_some()
            || !self.engine_extension.speculative_emission_queue.is_empty()
    }

    fn suspend_speculative_session(&mut self, suspension_reason: &'static str) {
        if !self.engine_extension.speculative_session_admitted {
            return;
        }
        self.engine_extension.speculative_session_admitted = false;
        self.engine_extension.speculative_emission_queue.clear();
        self.engine_extension.speculative_draft_session = None;
        tracing::info!(
            request_id = self.request_id.value(),
            suspension_reason,
            "suspending speculative decoding for the session's remainder"
        );
    }
}

#[cfg(test)]
#[path = "engine_request_tests.rs"]
mod engine_request_tests;
