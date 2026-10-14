//! The live decoder state retained between requests for prefix adoption.
//!
//! In-memory prefix adoption (issue #1167) keeps the just-completed
//! generation's decoder state alive so the next turn of the same conversation
//! adopts it instead of re-reading the prefix KV from the disk store. The
//! retention decision itself lives in the memory package; this holder is the
//! typed state the engine retains and installs.

use crate::persistent_cache::PersistentPromptCacheBlockKey;
use crate::qwen3_5_core::decoder::RequestDecoderStateStack;

use super::Qwen3_5EngineState;

impl Qwen3_5EngineState {
    /// Payload bytes of the decoder state that stays live across requests.
    ///
    /// After a completed generation the retained live prefix (issue #1167)
    /// keeps its KV arrays resident so the next turn can adopt them. Those
    /// bytes are active in every MLX measurement taken between requests, and
    /// the finalized ceiling-utilization split must charge them to the
    /// context owner; leaving them unattributed charges them against the
    /// one-decode-step activation reserve, whose clamp manufactures a
    /// phantom owner overrun (issue #1236).
    pub(super) fn live_retained_context_state_payload_bytes(&self) -> u64 {
        self.retained_live_prefix
            .as_ref()
            .map_or(0, |prefix| prefix.decoder_state.payload_byte_count())
    }
}

/// The retained live state of the last completed generation.
pub(in crate::qwen3_5_resident) struct RetainedLiveDecoderPrefix {
    /// The live decoder state at generation end. Append-only layers hold KV
    /// beyond the adoptable boundary (generated tokens); adoption truncates
    /// them to the boundary, and the stale tail is overwritten by the next
    /// prefill's in-place slab writes.
    pub(super) decoder_state: RequestDecoderStateStack,
    /// The complete prompt-cache block keys the generation's capture chain
    /// covers, in chain order. Adoption matches the next prompt's chain
    /// against these hashes.
    pub(super) complete_block_keys: Vec<PersistentPromptCacheBlockKey>,
    /// The token boundary of the last complete block: the token count the
    /// retained state can adopt.
    pub(super) adoptable_token_count: usize,
}
