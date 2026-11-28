use serde::{Deserialize, Serialize};

/// Complete resolved model-serving work-partition contract supplied by the daemon.
///
/// The user-facing configuration is validated before this data transfer object
/// is created. Keeping these values together prevents the worker, model, astro SSD prompt cache,
/// and model families from independently restoring hidden defaults.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerChunkingConfiguration {
    /// Prompt work while the whole model is held in memory: the size the
    /// person set, or `None` for Automatic. A model's startup settles
    /// Automatic into a number (`Some`) before any engine reads it.
    #[serde(default)]
    pub prompt_processing_chunk_size_tokens: Option<u32>,
    /// Fixed prompt work while sparse experts stream from storage.
    pub fixed_ssd_streaming_prompt_processing_chunk_size_tokens: u32,
    /// Capacity added when append-only attention state outgrows its current slab.
    pub full_attention_key_value_growth_tokens: u32,
    /// Decoder-layer interval between multi-token prefill command buffers while
    /// sparse experts are fully resident. Zero keeps one lazy tape per chunk.
    pub prefill_graph_submission_layer_interval: u32,
    /// Experimental decoder-layer interval for one-token solid-state-drive paging.
    ///
    /// Zero disables intermediate generation submission. Ignored while experts
    /// are fully memory resident.
    pub experimental_ssd_paging_generation_graph_submission_layer_interval: u32,
    /// Exact astro SSD prompt cache block length, or `None` for model-derived sizing.
    pub ssd_prompt_cache_block_tokens: Option<u32>,
    /// Number of astro SSD prompt cache blocks between retained branch restart checkpoints.
    pub ssd_prompt_cache_common_prefix_stride_blocks: u32,
    /// Retains the last completed generation's live decoder state and adopts
    /// it for the next turn of the same conversation instead of re-reading
    /// the prefix KV from the disk store. Adopted state is byte-identical to
    /// the restored state; the trade is one conversation's KV held between
    /// requests.
    #[serde(default)]
    pub experimental_in_memory_prefix_adoption_enabled: bool,
    /// Dense MLP and linear-attention output gate programs on the Neural Engine
    /// during prefill; costs memory beyond the MLX ceiling's own accounting.
    #[serde(default)]
    pub experimental_neural_engine_prefill_enabled: bool,
    /// How many tokens a model with a multi-token-prediction head drafts
    /// after each reply token: zero drafts none, and `None` leaves the count
    /// to the kind of head the model ships (one for Flash-Next, none for a
    /// dense pack, whose head was measured slower than plain decoding).
    #[serde(default)]
    pub multi_token_prediction_draft_tokens: Option<u32>,
}

/// Returns the command-buffer submission interval for one forward.
///
/// With every weight in memory, a multi-token prefill uses
/// `prefill_graph_submission_layer_interval` (default zero: one lazy tape)
/// and one-token generation stays at zero. While weights stream from the
/// SSD, a multi-token forward finishes every layer before the next by rule,
/// so no interval applies to it, and one-token generation uses the
/// experimental_ssd_paging_generation interval.
#[must_use]
pub const fn graph_submission_layer_interval(
    token_count: i32,
    weights_stream_from_storage: bool,
    prefill_graph_submission_layer_interval: u32,
    experimental_ssd_paging_generation_graph_submission_layer_interval: u32,
) -> u32 {
    match (weights_stream_from_storage, token_count == 1) {
        (false, true) | (true, false) => 0,
        (false, false) => prefill_graph_submission_layer_interval,
        (true, true) => experimental_ssd_paging_generation_graph_submission_layer_interval,
    }
}

impl WorkerChunkingConfiguration {
    /// Returns the command-buffer submission interval for one forward.
    #[must_use]
    pub const fn graph_submission_layer_interval(
        &self,
        token_count: i32,
        weights_stream_from_storage: bool,
    ) -> u32 {
        graph_submission_layer_interval(
            token_count,
            weights_stream_from_storage,
            self.prefill_graph_submission_layer_interval,
            self.experimental_ssd_paging_generation_graph_submission_layer_interval,
        )
    }
}
