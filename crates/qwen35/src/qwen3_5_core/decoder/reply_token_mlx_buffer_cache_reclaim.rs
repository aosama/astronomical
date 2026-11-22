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
