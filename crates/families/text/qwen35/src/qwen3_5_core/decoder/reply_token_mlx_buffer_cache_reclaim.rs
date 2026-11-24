//! Returns outgrown attention slabs to macOS while a reply is being written.

use runtime_integration::{MLX_BUFFER_CACHE_RECLAIM_THRESHOLD_BYTES, MlxRuntime, MlxRuntimeError};

/// Remembers the `astro_decoder_state_cache` size one request last saw, so
/// the release runs only on the token after a slab growth.
#[derive(Debug, Default)]
pub struct OutgrownAttentionSlabReclaim {
    decoder_state_payload_bytes_at_last_check: u64,
}

impl OutgrownAttentionSlabReclaim {
    /// Releases the `mlx_buffer_cache` between two reply tokens after the
    /// attention slabs grew, once it holds more than the reclaim threshold.
    ///
    /// Why a reply needs this at all: the attention key and value slabs of the
    /// `astro_decoder_state_cache` grow by concatenation every
    /// `full_attention_key_value_growth_tokens` tokens. Each growth allocates a
    /// slab slightly larger than the last and frees the old one into the
    /// `mlx_buffer_cache`, where it can never be handed out again because every
    /// later request is for a larger slab. The `mlx_buffer_cache` limit equals
    /// the whole MLX memory limit, so nothing else trims it before the reply
    /// ends.
    ///
    /// Measured on a 48 GB Mac with a 34,000-token conversation: the
    /// `mlx_buffer_cache` gained 0.72 GB every 256 reply tokens and wired
    /// memory rose with it, from 30 GB to the system wired ceiling within
    /// 3,000 tokens. From there macOS compressed other apps' memory in bursts,
    /// so video, audio and windows stalled while MLX's memory in use never
    /// moved.
    ///
    /// The release is tied to a growth rather than to the threshold alone
    /// because between two tokens the `mlx_buffer_cache` also holds the reply
    /// step's own working buffers, which the next token reuses. On a model
    /// whose working buffers exceed the threshold, a threshold-only rule would
    /// empty them every token; a 122B model measured 10 tokens per second
    /// without them against 28 with them.
    ///
    /// Call this only where the previous token has been synchronized: the
    /// reply path submits one token ahead, and the stream is drained here
    /// before the release for the same reason request finalization drains it.
    pub fn reclaim_after_slab_growth(
        &mut self,
        runtime: &MlxRuntime,
        decoder_state_payload_bytes: u64,
    ) -> Result<(), MlxRuntimeError> {
        if !self.observe_slab_growth(decoder_state_payload_bytes) {
            return Ok(());
        }
        runtime.synchronize_gpu_stream_and_reclaim_mlx_buffer_cache_above_threshold(
            MLX_BUFFER_CACHE_RECLAIM_THRESHOLD_BYTES,
        )
    }

    /// Records the current size and says whether it grew since the last
    /// check. A drafted rewind can shrink the state; that frees nothing
    /// into the `mlx_buffer_cache`, so only growth counts.
    pub(super) fn observe_slab_growth(&mut self, decoder_state_payload_bytes: u64) -> bool {
        let slabs_grew =
            decoder_state_payload_bytes > self.decoder_state_payload_bytes_at_last_check;
        self.decoder_state_payload_bytes_at_last_check = decoder_state_payload_bytes;
        slabs_grew
    }
}

/// `mlx_buffer_cache` bytes a draft-assisted reply may hold before a round
/// releases them.
///
/// Larger than the prompt path's threshold on purpose: a round's own
/// working buffers pass 256 MB at long conversations, so that threshold
/// emptied them on almost every round. Measured on a 35B model with a
/// 34,000-token conversation, same text and seed, in reply tokens per
/// second: no release 49.8 (the `mlx_buffer_cache` reached 7.6 GB), 256 MB
/// 52.8, 1 GB 51.1, 2 GB 58.6 to 60.5, 4 GB 58.0. On a Mac whose MLX
/// memory limit leaves less room than this, MLX's own collection line
/// releases the `mlx_buffer_cache` first, as it always has.
pub const DRAFT_ROUND_MLX_BUFFER_CACHE_RECLAIM_THRESHOLD_BYTES: usize = 2_000_000_000;

/// Releases the `mlx_buffer_cache` between two draft-and-verify rounds once
/// it holds more than [`DRAFT_ROUND_MLX_BUFFER_CACHE_RECLAIM_THRESHOLD_BYTES`],
/// whether or not a slab grew.
///
/// A round verifies a whole draft tree at once, so its attention buffers
/// are sized by the tree's rows times the conversation's length. The row
/// count changes from round to round and the length grows every round,
/// and MLX hands a freed buffer out again only for a request within two
/// memory pages of its size. Almost every round therefore allocates
/// buffers of sizes never seen before and leaves the previous round's
/// behind for good: about 4 GB every 100 reply tokens at 34,000 tokens of
/// conversation, which took wired memory to the system wired ceiling.
///
/// The growth rule of [`OutgrownAttentionSlabReclaim`] would be wrong
/// here: it fires once per slab growth, which let the tree buffers pile up
/// for 256 tokens at a time.
pub fn reclaim_mlx_buffer_cache_between_draft_rounds(
    runtime: &MlxRuntime,
) -> Result<(), MlxRuntimeError> {
    runtime.synchronize_gpu_stream_and_reclaim_mlx_buffer_cache_above_threshold(
        DRAFT_ROUND_MLX_BUFFER_CACHE_RECLAIM_THRESHOLD_BYTES,
    )
}
