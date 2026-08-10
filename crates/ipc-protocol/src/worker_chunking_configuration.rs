use serde::{Deserialize, Serialize};

/// Prompt-processing sizing policy supplied by the supervisor.
///
/// This policy belongs inside [`WorkerChunkingConfiguration`] so one startup
/// message cannot pair a sizing policy with a different set of chunking
/// boundaries. The worker receives one authoritative partitioning contract.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum WorkerPrefillChunckSizingPolicy {
    Optimized {
        optimizer_prefill_chunck_token_candidates: Vec<u32>,
    },
    Fixed {
        fixed_prefill_chunck_tokens: u32,
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
    pub prefill_sizing_policy: WorkerPrefillChunckSizingPolicy,
    /// Capacity added when append-only attention state outgrows its current slab.
    pub full_attention_key_value_growth_tokens: u32,
    /// Maximum token rows evaluated by one speculative-prefill drafter forward.
    pub speculative_prefill_draft_forward_tokens: u32,
    /// Decoder-layer interval for intermediate multi-token graph submission.
    ///
    /// Zero is meaningful: it keeps one complete prefill chunk as one lazy MLX
    /// graph instead of inserting intermediate scheduler boundaries.
    pub prefill_graph_submission_layer_interval: u32,
    /// Decoder-layer interval for intermediate one-token graph submission.
    ///
    /// Zero disables intermediate generation submission; a positive value can
    /// overlap host graph construction with graphics-processor execution.
    pub generation_graph_submission_layer_interval: u32,
    /// Completed observations retained per optimizer candidate and context.
    pub prefill_optimizer_observation_window: u32,
    /// Prompt-position width represented by one optimizer context identifier.
    pub prefill_optimizer_position_bucket_tokens: u32,
    /// Exact persistent-cache block length, or `None` for model-derived sizing.
    pub prompt_cache_block_tokens: Option<u32>,
    /// Number of cache blocks between retained branch restart checkpoints.
    pub prompt_cache_common_prefix_stride_blocks: u32,
}
