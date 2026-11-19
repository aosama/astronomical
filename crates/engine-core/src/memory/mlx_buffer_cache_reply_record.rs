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
pub struct MlxBufferCacheReplyRecord {
    sampled_token_count: u64,
    empty_token_count: u64,
    smallest_bytes: u64,
    largest_bytes: u64,
    total_bytes: u64,
}

impl MlxBufferCacheReplyRecord {
    /// Records the `mlx_buffer_cache` size read after one reply token.
    pub fn record_token(&mut self, mlx_buffer_cache_bytes: u64) {
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
    pub const fn sampled_token_count(&self) -> u64 {
        self.sampled_token_count
    }

    /// Tokens after which the `mlx_buffer_cache` held nothing: each of them
    /// left the next token to allocate every buffer afresh.
    #[must_use]
    pub const fn empty_token_count(&self) -> u64 {
        self.empty_token_count
    }

    #[must_use]
    pub const fn smallest_bytes(&self) -> u64 {
        self.smallest_bytes
    }

    #[must_use]
    pub const fn largest_bytes(&self) -> u64 {
        self.largest_bytes
    }

    /// The average size over the sampled tokens, or zero before any token.
    #[must_use]
    pub const fn mean_bytes(&self) -> u64 {
        match self.total_bytes.checked_div(self.sampled_token_count) {
            Some(mean_bytes) => mean_bytes,
            None => 0,
        }
    }

    /// The reply's memory record as Observatory reads it: these
    /// `mlx_buffer_cache` readings beside where each token found its experts.
    #[must_use]
    pub fn reply_memory_record(
        &self,
        reply_expert_route_record: crate::memory::reply_expert_route_record::ReplyExpertRouteRecord,
    ) -> astronomical_ipc_protocol::ReplyMemoryRecord {
        let reply_totals = reply_expert_route_record.reply_totals();
        astronomical_ipc_protocol::ReplyMemoryRecord {
            routed_expert_count_per_token: reply_expert_route_record
                .routed_expert_count_per_token(),
            expert_hit_count: reply_totals.memory_served_expert_count,
            expert_miss_count: reply_totals.storage_read_expert_count,
            storage_read_bytes: reply_totals.decode_storage_read_bytes,
            storage_read_wait_nanoseconds: reply_totals.decode_storage_read_wait_nanoseconds,
            storage_read_expert_counts_by_token: reply_expert_route_record
                .into_storage_read_expert_counts_by_token(),
            mlx_buffer_cache_sampled_token_count: self.sampled_token_count,
            mlx_buffer_cache_empty_token_count: self.empty_token_count,
            smallest_mlx_buffer_cache_bytes: self.smallest_bytes,
            mean_mlx_buffer_cache_bytes: self.mean_bytes(),
            largest_mlx_buffer_cache_bytes: self.largest_bytes,
        }
    }
}

#[cfg(test)]
#[path = "mlx_buffer_cache_reply_record_tests.rs"]
mod mlx_buffer_cache_reply_record_tests;
