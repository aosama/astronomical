//! What one reply did to memory, token by token.

use serde::{Deserialize, Serialize};

/// How one finished reply used the `astro_expert_cache` and the
/// `mlx_buffer_cache`: the evidence Observatory draws its reply strip from.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ReplyMemoryRecord {
    /// For each token of the reply, in order, how many of its routed experts
    /// were read from storage because the `astro_expert_cache` did not hold
    /// them. Zero means the token was served wholly from memory.
    pub storage_read_expert_counts_by_token: Vec<u16>,
    /// How many routed experts one token uses (layers times experts per
    /// layer), so each token's storage reads read as a share; zero when no
    /// token was recorded.
    pub routed_expert_count_per_token: u16,
    /// Tokens after which the `mlx_buffer_cache` was read.
    pub mlx_buffer_cache_sampled_token_count: u64,
    /// Of those, tokens after which it held nothing, so the next token
    /// allocated every buffer afresh.
    pub mlx_buffer_cache_empty_token_count: u64,
    /// The fewest bytes the `mlx_buffer_cache` held after any token.
    pub smallest_mlx_buffer_cache_bytes: u64,
    /// The average it held over the sampled tokens.
    pub mean_mlx_buffer_cache_bytes: u64,
    /// The most it held after any token.
    pub largest_mlx_buffer_cache_bytes: u64,
}
