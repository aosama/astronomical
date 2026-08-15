use serde::{Deserialize, Serialize};

/// Prompt-processing sizing policy supplied by the supervisor.
///
/// This policy belongs inside [`WorkerChunkingConfiguration`] so one startup
/// message cannot pair a sizing policy with a different set of chunking
/// boundaries. The worker receives one authoritative partitioning contract.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum WorkerPromptProcessingChunkSizingPolicy {
    /// Learns among the supplied candidate capacities for each execution profile.
    Optimized {
        /// Positive configured capacities considered by the online optimizer.
        prompt_processing_chunk_size_optimizer_candidate_token_counts: Vec<u32>,
    },
    /// Uses explicit capacities without collecting optimizer measurements.
    Fixed {
        /// Capacity used for complete-resident execution and as the fallback.
        fixed_prompt_processing_chunk_size_tokens: u32,
        /// Smaller fixed size while sparse experts stream from storage.
        ///
        /// `None` keeps the complete-resident fixed size for every residency mode.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        fixed_ssd_streaming_prompt_processing_chunk_size_tokens: Option<u32>,
    },
}

/// Complete resolved model-serving work-partition contract supplied by the supervisor.
///
/// The user-facing configuration is validated before this data transfer object
/// is created. Keeping these values together prevents the worker, model, cache,
/// and optimizer from independently restoring hidden defaults.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerChunkingConfiguration {
    /// Chooses fixed or measured prompt-processing chunk lengths.
    pub prompt_processing_chunk_sizing_policy: WorkerPromptProcessingChunkSizingPolicy,
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
    /// Completed measurements retained per optimizer candidate and context.
    pub prompt_processing_chunk_size_optimizer_maximum_retained_measurements_per_candidate_and_context:
        u32,
    /// Prompt-position width represented by one optimizer position range.
    pub prompt_processing_chunk_size_optimizer_position_range_size_tokens: u32,
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
