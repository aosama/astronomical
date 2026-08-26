use super::AstronomicalConfigError;
use serde::{Deserialize, Serialize};

/// Optional persisted chunking values shared by global policy and model overrides.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ChunkingConfigFile {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) fixed_prompt_processing_chunk_size_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) fixed_ssd_streaming_prompt_processing_chunk_size_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) full_attention_key_value_growth_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) speculative_prefill_draft_forward_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) prefill_graph_submission_layer_interval: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) experimental_ssd_paging_prefill_graph_submission_layer_interval: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) experimental_ssd_paging_generation_graph_submission_layer_interval: Option<u32>,
    #[serde(
        default,
        deserialize_with = "deserialize_present_nullable_u32",
        skip_serializing_if = "Option::is_none"
    )]
    pub(crate) prompt_cache_block_tokens: Option<Option<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) prompt_cache_common_prefix_stride_blocks: Option<u32>,
}

/// Presence map for advanced chunking fields after global/model inheritance.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ConfiguredChunkingFields {
    pub fixed_prompt_processing_chunk_size_tokens: bool,
    pub fixed_ssd_streaming_prompt_processing_chunk_size_tokens: bool,
    pub full_attention_key_value_growth_tokens: bool,
    pub speculative_prefill_draft_forward_tokens: bool,
    pub prefill_graph_submission_layer_interval: bool,
    pub experimental_ssd_paging_prefill_graph_submission_layer_interval: bool,
    pub experimental_ssd_paging_generation_graph_submission_layer_interval: bool,
    pub prompt_cache_block_tokens: bool,
    pub prompt_cache_common_prefix_stride_blocks: bool,
}

fn deserialize_present_nullable_u32<'de, Deserializer>(
    deserializer: Deserializer,
) -> Result<Option<Option<u32>>, Deserializer::Error>
where
    Deserializer: serde::Deserializer<'de>,
{
    Option::<u32>::deserialize(deserializer).map(Some)
}

impl ChunkingConfigFile {
    pub(crate) fn configured_fields(&self) -> ConfiguredChunkingFields {
        ConfiguredChunkingFields {
            fixed_prompt_processing_chunk_size_tokens: self
                .fixed_prompt_processing_chunk_size_tokens
                .is_some(),
            fixed_ssd_streaming_prompt_processing_chunk_size_tokens: self
                .fixed_ssd_streaming_prompt_processing_chunk_size_tokens
                .is_some(),
            full_attention_key_value_growth_tokens: self
                .full_attention_key_value_growth_tokens
                .is_some(),
            speculative_prefill_draft_forward_tokens: self
                .speculative_prefill_draft_forward_tokens
                .is_some(),
            prefill_graph_submission_layer_interval: self
                .prefill_graph_submission_layer_interval
                .is_some(),
            experimental_ssd_paging_prefill_graph_submission_layer_interval: self
                .experimental_ssd_paging_prefill_graph_submission_layer_interval
                .is_some(),
            experimental_ssd_paging_generation_graph_submission_layer_interval: self
                .experimental_ssd_paging_generation_graph_submission_layer_interval
                .is_some(),
            prompt_cache_block_tokens: self.prompt_cache_block_tokens.is_some(),
            prompt_cache_common_prefix_stride_blocks: self
                .prompt_cache_common_prefix_stride_blocks
                .is_some(),
        }
    }

    pub(crate) fn merged(global: &Self, model: Option<&Self>) -> Self {
        let Some(model) = model else {
            return global.clone();
        };
        Self {
            fixed_prompt_processing_chunk_size_tokens: model
                .fixed_prompt_processing_chunk_size_tokens
                .or(global.fixed_prompt_processing_chunk_size_tokens),
            fixed_ssd_streaming_prompt_processing_chunk_size_tokens: model
                .fixed_ssd_streaming_prompt_processing_chunk_size_tokens
                .or(global.fixed_ssd_streaming_prompt_processing_chunk_size_tokens),
            full_attention_key_value_growth_tokens: model
                .full_attention_key_value_growth_tokens
                .or(global.full_attention_key_value_growth_tokens),
            speculative_prefill_draft_forward_tokens: model
                .speculative_prefill_draft_forward_tokens
                .or(global.speculative_prefill_draft_forward_tokens),
            prefill_graph_submission_layer_interval: model
                .prefill_graph_submission_layer_interval
                .or(global.prefill_graph_submission_layer_interval),
            experimental_ssd_paging_prefill_graph_submission_layer_interval: model
                .experimental_ssd_paging_prefill_graph_submission_layer_interval
                .or(global.experimental_ssd_paging_prefill_graph_submission_layer_interval),
            experimental_ssd_paging_generation_graph_submission_layer_interval: model
                .experimental_ssd_paging_generation_graph_submission_layer_interval
                .or(global.experimental_ssd_paging_generation_graph_submission_layer_interval),
            prompt_cache_block_tokens: model
                .prompt_cache_block_tokens
                .or(global.prompt_cache_block_tokens),
            prompt_cache_common_prefix_stride_blocks: model
                .prompt_cache_common_prefix_stride_blocks
                .or(global.prompt_cache_common_prefix_stride_blocks),
        }
    }
}

pub const DEFAULT_FULL_ATTENTION_KEY_VALUE_GROWTH_TOKENS: u32 = 256;
pub const DEFAULT_FIXED_PROMPT_PROCESSING_CHUNK_SIZE_TOKENS: u32 = 2_048;
/// Independent SSD-paged prompt chunk. Owned separately from the resident chunk.
pub const DEFAULT_FIXED_SSD_STREAMING_PROMPT_PROCESSING_CHUNK_SIZE_TOKENS: u32 = 2_048;
pub const DEFAULT_SPECULATIVE_PREFILL_DRAFT_FORWARD_TOKENS: u32 = 2_048;
pub const DEFAULT_EXPERIMENTAL_SSD_PAGING_PREFILL_GRAPH_SUBMISSION_LAYER_INTERVAL: u32 = 1;
pub const DEFAULT_EXPERIMENTAL_SSD_PAGING_GENERATION_GRAPH_SUBMISSION_LAYER_INTERVAL: u32 = 3;
/// Resident multi-token prefill keeps one lazy tape unless the user raises this.
pub const DEFAULT_PREFILL_GRAPH_SUBMISSION_LAYER_INTERVAL: u32 = 0;
pub const DEFAULT_PROMPT_CACHE_COMMON_PREFIX_STRIDE_BLOCKS: u32 = 4;

/// Resolved user-visible boundaries that partition model-serving work.
///
/// Resolution is deliberately centralized at the configuration boundary. The
/// supervisor passes this complete contract to the worker, and downstream
/// components convert only numeric representation—not policy or defaults.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkingConfig {
    fixed_prompt_processing_chunk_size_tokens: u32,
    fixed_ssd_streaming_prompt_processing_chunk_size_tokens: u32,
    full_attention_key_value_growth_tokens: u32,
    speculative_prefill_draft_forward_tokens: u32,
    prefill_graph_submission_layer_interval: u32,
    experimental_ssd_paging_prefill_graph_submission_layer_interval: u32,
    experimental_ssd_paging_generation_graph_submission_layer_interval: u32,
    prompt_cache_block_tokens: Option<u32>,
    prompt_cache_common_prefix_stride_blocks: u32,
}

impl ChunkingConfig {
    /// Resolves all chunking defaults and validates them before worker startup.
    ///
    /// Downstream layers receive this complete policy and must not reconstruct
    /// hidden defaults from optional user fields.
    pub(crate) fn resolve(
        configured: &ChunkingConfigFile,
    ) -> Result<Self, AstronomicalConfigError> {
        let resolved = Self {
            fixed_prompt_processing_chunk_size_tokens: configured
                .fixed_prompt_processing_chunk_size_tokens
                .unwrap_or(DEFAULT_FIXED_PROMPT_PROCESSING_CHUNK_SIZE_TOKENS),
            fixed_ssd_streaming_prompt_processing_chunk_size_tokens: configured
                .fixed_ssd_streaming_prompt_processing_chunk_size_tokens
                .unwrap_or(DEFAULT_FIXED_SSD_STREAMING_PROMPT_PROCESSING_CHUNK_SIZE_TOKENS),
            full_attention_key_value_growth_tokens: configured
                .full_attention_key_value_growth_tokens
                .unwrap_or(DEFAULT_FULL_ATTENTION_KEY_VALUE_GROWTH_TOKENS),
            speculative_prefill_draft_forward_tokens: configured
                .speculative_prefill_draft_forward_tokens
                .unwrap_or(DEFAULT_SPECULATIVE_PREFILL_DRAFT_FORWARD_TOKENS),
            prefill_graph_submission_layer_interval: configured
                .prefill_graph_submission_layer_interval
                .unwrap_or(DEFAULT_PREFILL_GRAPH_SUBMISSION_LAYER_INTERVAL),
            experimental_ssd_paging_prefill_graph_submission_layer_interval: configured
                .experimental_ssd_paging_prefill_graph_submission_layer_interval
                .unwrap_or(DEFAULT_EXPERIMENTAL_SSD_PAGING_PREFILL_GRAPH_SUBMISSION_LAYER_INTERVAL),
            experimental_ssd_paging_generation_graph_submission_layer_interval: configured
                .experimental_ssd_paging_generation_graph_submission_layer_interval
                .unwrap_or(
                    DEFAULT_EXPERIMENTAL_SSD_PAGING_GENERATION_GRAPH_SUBMISSION_LAYER_INTERVAL,
                ),
            prompt_cache_block_tokens: configured.prompt_cache_block_tokens.unwrap_or(None),
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
                "chunking.fixed_prompt_processing_chunk_size_tokens",
                self.fixed_prompt_processing_chunk_size_tokens,
            ),
            (
                "chunking.full_attention_key_value_growth_tokens",
                self.full_attention_key_value_growth_tokens,
            ),
            (
                "chunking.speculative_prefill_draft_forward_tokens",
                self.speculative_prefill_draft_forward_tokens,
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
        if self.fixed_ssd_streaming_prompt_processing_chunk_size_tokens == 0 {
            return Err(AstronomicalConfigError::InvalidChunkingValue {
                field_name: "chunking.fixed_ssd_streaming_prompt_processing_chunk_size_tokens",
                description: "must be positive",
            });
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
    pub const fn fixed_prompt_processing_chunk_size_tokens(&self) -> u32 {
        self.fixed_prompt_processing_chunk_size_tokens
    }

    #[must_use]
    pub const fn fixed_ssd_streaming_prompt_processing_chunk_size_tokens(&self) -> u32 {
        self.fixed_ssd_streaming_prompt_processing_chunk_size_tokens
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
    pub const fn prefill_graph_submission_layer_interval(&self) -> u32 {
        self.prefill_graph_submission_layer_interval
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
            fixed_prompt_processing_chunk_size_tokens:
                DEFAULT_FIXED_PROMPT_PROCESSING_CHUNK_SIZE_TOKENS,
            fixed_ssd_streaming_prompt_processing_chunk_size_tokens:
                DEFAULT_FIXED_SSD_STREAMING_PROMPT_PROCESSING_CHUNK_SIZE_TOKENS,
            full_attention_key_value_growth_tokens: DEFAULT_FULL_ATTENTION_KEY_VALUE_GROWTH_TOKENS,
            speculative_prefill_draft_forward_tokens:
                DEFAULT_SPECULATIVE_PREFILL_DRAFT_FORWARD_TOKENS,
            prefill_graph_submission_layer_interval:
                DEFAULT_PREFILL_GRAPH_SUBMISSION_LAYER_INTERVAL,
            experimental_ssd_paging_prefill_graph_submission_layer_interval:
                DEFAULT_EXPERIMENTAL_SSD_PAGING_PREFILL_GRAPH_SUBMISSION_LAYER_INTERVAL,
            experimental_ssd_paging_generation_graph_submission_layer_interval:
                DEFAULT_EXPERIMENTAL_SSD_PAGING_GENERATION_GRAPH_SUBMISSION_LAYER_INTERVAL,
            prompt_cache_block_tokens: None,
            prompt_cache_common_prefix_stride_blocks:
                DEFAULT_PROMPT_CACHE_COMMON_PREFIX_STRIDE_BLOCKS,
        }
    }
}
