//! What the `mlx_buffer_cache` held over one reply, read after each token.
//!
//! MLX reuses a finished step's buffers only while they sit in the
//! `mlx_buffer_cache`. Once memory in use plus the cache reaches MLX's
//! collection line the cache is emptied, and every token allocates its
//! buffers afresh: on a 48 GB Mac a 122B model then replied at a third of its
//! speed. Nothing showed that happening, so each reply now records how full
//! the cache was after every token, and says so when the reply ends.

/// The `mlx_buffer_cache` sizes read after each token of one reply.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct MlxBufferCacheReplyRecord {
    sampled_token_count: u64,
    empty_token_count: u64,
    smallest_bytes: u64,
    largest_bytes: u64,
    total_bytes: u64,
}

impl MlxBufferCacheReplyRecord {
    /// Records the `mlx_buffer_cache` size read after one reply token.
    pub(crate) fn record_token(&mut self, mlx_buffer_cache_bytes: u64) {
        self.smallest_bytes = if self.sampled_token_count == 0 {
            mlx_buffer_cache_bytes
        } else {
            self.smallest_bytes.min(mlx_buffer_cache_bytes)
        };
        self.sampled_token_count += 1;
        if mlx_buffer_cache_bytes == 0 {
            self.empty_token_count += 1;
        }
        self.largest_bytes = self.largest_bytes.max(mlx_buffer_cache_bytes);
        self.total_bytes = self.total_bytes.saturating_add(mlx_buffer_cache_bytes);
    }

    #[must_use]
    pub(crate) const fn sampled_token_count(&self) -> u64 {
        self.sampled_token_count
    }

    /// Tokens after which the `mlx_buffer_cache` held nothing: each of them
    /// left the next token to allocate every buffer afresh.
    #[must_use]
    pub(crate) const fn empty_token_count(&self) -> u64 {
        self.empty_token_count
    }

    #[must_use]
    pub(crate) const fn smallest_bytes(&self) -> u64 {
        self.smallest_bytes
    }

    #[must_use]
    pub(crate) const fn largest_bytes(&self) -> u64 {
        self.largest_bytes
    }

    /// The average size over the sampled tokens, or zero before any token.
    #[must_use]
    pub(crate) const fn mean_bytes(&self) -> u64 {
        match self.total_bytes.checked_div(self.sampled_token_count) {
            Some(mean_bytes) => mean_bytes,
            None => 0,
        }
    }
}

#[cfg(test)]
#[path = "mlx_buffer_cache_reply_record_tests.rs"]
mod mlx_buffer_cache_reply_record_tests;
