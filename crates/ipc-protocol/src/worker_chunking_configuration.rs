use serde::{Deserialize, Serialize};

/// Complete resolved model-serving work-partition contract supplied by the supervisor.
///
/// The user-facing configuration is validated before this data transfer object
/// is created. Keeping these values together prevents the worker, model, cache,
/// and model families from independently restoring hidden defaults.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerChunkingConfiguration {
    /// Fixed prompt work while sparse experts are fully resident, and the paging fallback.
    pub fixed_prompt_processing_chunk_size_tokens: u32,
    /// Optional smaller prompt work while sparse experts stream from storage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fixed_ssd_streaming_prompt_processing_chunk_size_tokens: Option<u32>,
    /// Capacity added when append-only attention state outgrows its current slab.
    pub full_attention_key_value_growth_tokens: u32,
    /// Maximum token rows evaluated by one speculative-prefill drafter forward.
    pub speculative_prefill_draft_forward_tokens: u32,
    /// Experimental decoder-layer interval for multi-token solid-state-drive paging.
    ///
    /// Zero keeps one complete prefill chunk as one lazy MLX graph. A positive
    /// value is ignored while experts are fully memory resident.
    pub experimental_ssd_paging_prefill_graph_submission_layer_interval: u32,
    /// Experimental decoder-layer interval for one-token solid-state-drive paging.
    ///
    /// Zero disables intermediate generation submission. A positive value is
    /// ignored while experts are fully memory resident.
    pub experimental_ssd_paging_generation_graph_submission_layer_interval: u32,
    /// Exact persistent-cache block length, or `None` for model-derived sizing.
    pub prompt_cache_block_tokens: Option<u32>,
    /// Number of cache blocks between retained branch restart checkpoints.
    pub prompt_cache_common_prefix_stride_blocks: u32,
}

/// Returns the experimental solid-state-drive paging layer interval for one forward.
///
/// Fully resident experts always return 0 so the decoder stays one lazy tape.
/// A positive configured interval applies only while sparse experts stream from disk.
#[must_use]
pub const fn experimental_ssd_paging_graph_submission_layer_interval(
    token_count: i32,
    sparse_experts_are_paged: bool,
    experimental_ssd_paging_prefill_graph_submission_layer_interval: u32,
    experimental_ssd_paging_generation_graph_submission_layer_interval: u32,
) -> u32 {
    if !sparse_experts_are_paged {
        return 0;
    }
    if token_count == 1 {
        experimental_ssd_paging_generation_graph_submission_layer_interval
    } else {
        experimental_ssd_paging_prefill_graph_submission_layer_interval
    }
}

impl WorkerChunkingConfiguration {
    /// Returns the experimental solid-state-drive paging interval for one forward.
    #[must_use]
    pub const fn experimental_ssd_paging_graph_submission_layer_interval(
        &self,
        token_count: i32,
        sparse_experts_are_paged: bool,
    ) -> u32 {
        experimental_ssd_paging_graph_submission_layer_interval(
            token_count,
            sparse_experts_are_paged,
            self.experimental_ssd_paging_prefill_graph_submission_layer_interval,
            self.experimental_ssd_paging_generation_graph_submission_layer_interval,
        )
    }
}
