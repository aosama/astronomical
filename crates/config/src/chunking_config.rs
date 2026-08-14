use super::{
    AstronomicalConfigError, DEFAULT_OPTIMIZER_PREFILL_CHUNCK_TOKEN_CANDIDATES,
    PrefillChunckSizingPolicy, resolve_prefill_chunck_sizing_policy,
};
use crate::config_file::ChunkingConfigFile;

pub const DEFAULT_FULL_ATTENTION_KEY_VALUE_GROWTH_TOKENS: u32 = 256;
pub const DEFAULT_SPECULATIVE_PREFILL_DRAFT_FORWARD_TOKENS: u32 = 2_048;
pub const DEFAULT_EXPERIMENTAL_SSD_PAGING_GENERATION_GRAPH_SUBMISSION_LAYER_INTERVAL: u32 = 3;
pub const DEFAULT_EXPERIMENTAL_SSD_PAGING_PREFILL_GRAPH_SUBMISSION_LAYER_INTERVAL: u32 = 0;
pub const DEFAULT_PREFILL_OPTIMIZER_OBSERVATION_WINDOW: u32 = 5;
pub const DEFAULT_PREFILL_OPTIMIZER_POSITION_BUCKET_TOKENS: u32 = 32_768;
pub const DEFAULT_PROMPT_CACHE_COMMON_PREFIX_STRIDE_BLOCKS: u32 = 4;

/// Resolved user-visible boundaries that partition model-serving work.
///
/// Resolution is deliberately centralized at the configuration boundary. The
/// supervisor passes this complete contract to the worker, and downstream
/// components convert only numeric representation—not policy or defaults.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkingConfig {
    prefill_sizing_policy: PrefillChunckSizingPolicy,
    full_attention_key_value_growth_tokens: u32,
    speculative_prefill_draft_forward_tokens: u32,
    experimental_ssd_paging_prefill_graph_submission_layer_interval: u32,
    experimental_ssd_paging_generation_graph_submission_layer_interval: u32,
    prefill_optimizer_observation_window: u32,
    prefill_optimizer_position_bucket_tokens: u32,
    prompt_cache_block_tokens: Option<u32>,
    prompt_cache_common_prefix_stride_blocks: u32,
}

impl ChunkingConfig {
    pub(crate) fn resolve(
        configured: &ChunkingConfigFile,
    ) -> Result<Self, AstronomicalConfigError> {
        let prefill_sizing_policy = resolve_prefill_chunck_sizing_policy(
            configured.prefill_size_optimizer_enabled,
            configured.fixed_prefill_tokens,
            configured.fixed_ssd_streaming_prefill_tokens,
            configured.optimizer_prefill_token_candidates.as_deref(),
        )?;
        let resolved = Self {
            prefill_sizing_policy,
            full_attention_key_value_growth_tokens: configured
                .full_attention_key_value_growth_tokens
                .unwrap_or(DEFAULT_FULL_ATTENTION_KEY_VALUE_GROWTH_TOKENS),
            speculative_prefill_draft_forward_tokens: configured
                .speculative_prefill_draft_forward_tokens
                .unwrap_or(DEFAULT_SPECULATIVE_PREFILL_DRAFT_FORWARD_TOKENS),
            experimental_ssd_paging_prefill_graph_submission_layer_interval: configured
                .experimental_ssd_paging_prefill_graph_submission_layer_interval
                .unwrap_or(DEFAULT_EXPERIMENTAL_SSD_PAGING_PREFILL_GRAPH_SUBMISSION_LAYER_INTERVAL),
            experimental_ssd_paging_generation_graph_submission_layer_interval: configured
                .experimental_ssd_paging_generation_graph_submission_layer_interval
                .unwrap_or(
                    DEFAULT_EXPERIMENTAL_SSD_PAGING_GENERATION_GRAPH_SUBMISSION_LAYER_INTERVAL,
                ),
            prefill_optimizer_observation_window: configured
                .prefill_optimizer_observation_window
                .unwrap_or(DEFAULT_PREFILL_OPTIMIZER_OBSERVATION_WINDOW),
            prefill_optimizer_position_bucket_tokens: configured
                .prefill_optimizer_position_bucket_tokens
                .unwrap_or(DEFAULT_PREFILL_OPTIMIZER_POSITION_BUCKET_TOKENS),
            prompt_cache_block_tokens: configured.prompt_cache_block_tokens,
            prompt_cache_common_prefix_stride_blocks: configured
                .prompt_cache_common_prefix_stride_blocks
                .unwrap_or(DEFAULT_PROMPT_CACHE_COMMON_PREFIX_STRIDE_BLOCKS),
        };
        resolved.validate()?;
        Ok(resolved)
    }

    fn validate(&self) -> Result<(), AstronomicalConfigError> {
        for (field_name, field_value) in [
            (
                "chunking.full_attention_key_value_growth_tokens",
                self.full_attention_key_value_growth_tokens,
            ),
            (
                "chunking.speculative_prefill_draft_forward_tokens",
                self.speculative_prefill_draft_forward_tokens,
            ),
            (
                "chunking.prefill_optimizer_observation_window",
                self.prefill_optimizer_observation_window,
            ),
            (
                "chunking.prefill_optimizer_position_bucket_tokens",
                self.prefill_optimizer_position_bucket_tokens,
            ),
            (
                "chunking.prompt_cache_common_prefix_stride_blocks",
                self.prompt_cache_common_prefix_stride_blocks,
            ),
        ] {
            if field_value == 0 {
                return Err(AstronomicalConfigError::InvalidChunkingValue {
                    field_name,
                    description: "must be positive",
                });
            }
        }
        if self.prompt_cache_block_tokens == Some(0) {
            return Err(AstronomicalConfigError::InvalidChunkingValue {
                field_name: "chunking.prompt_cache_block_tokens",
                description: "must be null for automatic sizing or a positive token count",
            });
        }
        // MLX array dimensions cross the native boundary as signed 32-bit
        // integers. Reject an unrepresentable slab increment while loading the
        // user's config rather than failing later during the first model load.
        if self.full_attention_key_value_growth_tokens > i32::MAX as u32 {
            return Err(AstronomicalConfigError::InvalidChunkingValue {
                field_name: "chunking.full_attention_key_value_growth_tokens",
                description: "must fit the signed 32-bit MLX dimension range",
            });
        }
        Ok(())
    }

    #[must_use]
    pub const fn prefill_sizing_policy(&self) -> &PrefillChunckSizingPolicy {
        &self.prefill_sizing_policy
    }

    #[must_use]
    pub const fn full_attention_key_value_growth_tokens(&self) -> u32 {
        self.full_attention_key_value_growth_tokens
    }

    #[must_use]
    pub const fn speculative_prefill_draft_forward_tokens(&self) -> u32 {
        self.speculative_prefill_draft_forward_tokens
    }

    #[must_use]
    pub const fn experimental_ssd_paging_prefill_graph_submission_layer_interval(&self) -> u32 {
        self.experimental_ssd_paging_prefill_graph_submission_layer_interval
    }

    #[must_use]
    pub const fn experimental_ssd_paging_generation_graph_submission_layer_interval(&self) -> u32 {
        self.experimental_ssd_paging_generation_graph_submission_layer_interval
    }

    #[must_use]
    pub const fn prefill_optimizer_observation_window(&self) -> u32 {
        self.prefill_optimizer_observation_window
    }

    #[must_use]
    pub const fn prefill_optimizer_position_bucket_tokens(&self) -> u32 {
        self.prefill_optimizer_position_bucket_tokens
    }

    #[must_use]
    pub const fn prompt_cache_block_tokens(&self) -> Option<u32> {
        self.prompt_cache_block_tokens
    }

    #[must_use]
    pub const fn prompt_cache_common_prefix_stride_blocks(&self) -> u32 {
        self.prompt_cache_common_prefix_stride_blocks
    }
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            prefill_sizing_policy: PrefillChunckSizingPolicy::Optimized {
                optimizer_prefill_chunck_token_candidates:
                    DEFAULT_OPTIMIZER_PREFILL_CHUNCK_TOKEN_CANDIDATES.to_vec(),
            },
            full_attention_key_value_growth_tokens: DEFAULT_FULL_ATTENTION_KEY_VALUE_GROWTH_TOKENS,
            speculative_prefill_draft_forward_tokens:
                DEFAULT_SPECULATIVE_PREFILL_DRAFT_FORWARD_TOKENS,
            experimental_ssd_paging_prefill_graph_submission_layer_interval:
                DEFAULT_EXPERIMENTAL_SSD_PAGING_PREFILL_GRAPH_SUBMISSION_LAYER_INTERVAL,
            experimental_ssd_paging_generation_graph_submission_layer_interval:
                DEFAULT_EXPERIMENTAL_SSD_PAGING_GENERATION_GRAPH_SUBMISSION_LAYER_INTERVAL,
            prefill_optimizer_observation_window: DEFAULT_PREFILL_OPTIMIZER_OBSERVATION_WINDOW,
            prefill_optimizer_position_bucket_tokens:
                DEFAULT_PREFILL_OPTIMIZER_POSITION_BUCKET_TOKENS,
            prompt_cache_block_tokens: None,
            prompt_cache_common_prefix_stride_blocks:
                DEFAULT_PROMPT_CACHE_COMMON_PREFIX_STRIDE_BLOCKS,
        }
    }
}
