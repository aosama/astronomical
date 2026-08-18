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
    /// Decoder-layer interval between multi-token prefill command buffers.
    ///
    /// Zero keeps one complete prefill chunk as one lazy MLX graph. A positive
    /// value gives macOS a scheduling boundary without splitting layer kernels.
    pub prefill_graph_submission_layer_interval: u32,
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

/// Returns the command-buffer submission interval for decode-time forwards.
///
/// Resident decode, including 2-4 token multi-token-prediction verification,
/// keeps one lazy tape. Intermediate submission exists only so paged decode can
/// detach streamed expert pages.
#[must_use]
pub const fn decode_graph_submission_layer_interval(
    sparse_experts_are_paged: bool,
    experimental_ssd_paging_generation_graph_submission_layer_interval: u32,
) -> u32 {
    if sparse_experts_are_paged {
        experimental_ssd_paging_generation_graph_submission_layer_interval
    } else {
        0
    }
}

/// Longest decode-shaped forward, including a depth-three MTP verify window.
const DECODE_SHAPED_FORWARD_TOKEN_LIMIT: i32 = 4;

/// Returns the command-buffer submission interval for one forward.
///
/// Multi-thousand-token prefill uses its configured interval so macOS can
/// schedule between layer groups. Decode-shaped forwards of one through four
/// tokens, including MTP verification, keep intermediate submission specific
/// to SSD paging.
#[must_use]
pub const fn graph_submission_layer_interval(
    token_count: i32,
    sparse_experts_are_paged: bool,
    prefill_graph_submission_layer_interval: u32,
    experimental_ssd_paging_generation_graph_submission_layer_interval: u32,
) -> u32 {
    if token_count <= DECODE_SHAPED_FORWARD_TOKEN_LIMIT {
        decode_graph_submission_layer_interval(
            sparse_experts_are_paged,
            experimental_ssd_paging_generation_graph_submission_layer_interval,
        )
    } else {
        prefill_graph_submission_layer_interval
    }
}

impl WorkerChunkingConfiguration {
    /// Returns the command-buffer submission interval for one forward.
    #[must_use]
    pub const fn graph_submission_layer_interval(
        &self,
        token_count: i32,
        sparse_experts_are_paged: bool,
    ) -> u32 {
        graph_submission_layer_interval(
            token_count,
            sparse_experts_are_paged,
            self.prefill_graph_submission_layer_interval,
            self.experimental_ssd_paging_generation_graph_submission_layer_interval,
        )
    }

    /// Returns the command-buffer interval for decode-time forwards.
    #[must_use]
    pub const fn decode_graph_submission_layer_interval(
        &self,
        sparse_experts_are_paged: bool,
    ) -> u32 {
        decode_graph_submission_layer_interval(
            sparse_experts_are_paged,
            self.experimental_ssd_paging_generation_graph_submission_layer_interval,
        )
    }
}
