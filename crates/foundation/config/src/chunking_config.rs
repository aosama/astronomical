use super::AstronomicalConfigError;
use serde::{Deserialize, Serialize};

/// Optional persisted chunking values shared by global policy and model overrides.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ChunkingConfigFile {
    /// The in-memory prompt chunk: a number the user chose, or `null` for
    /// Automatic, where the engine picks the size for the model and the Mac
    /// when the model loads.
    #[serde(
        default,
        deserialize_with = "deserialize_present_nullable_u32",
        skip_serializing_if = "Option::is_none"
    )]
    pub(crate) prompt_processing_chunk_size_tokens: Option<Option<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) fixed_ssd_streaming_prompt_processing_chunk_size_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) full_attention_key_value_growth_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) prefill_graph_submission_layer_interval: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) experimental_ssd_paging_generation_graph_submission_layer_interval: Option<u32>,
    #[serde(
        default,
        deserialize_with = "deserialize_present_nullable_u32",
        skip_serializing_if = "Option::is_none"
    )]
    pub(crate) ssd_prompt_cache_block_tokens: Option<Option<u32>>,
    #[serde(
        default,
        deserialize_with = "deserialize_present_nullable_draft_tokens",
        skip_serializing_if = "Option::is_none"
    )]
    pub(crate) multi_token_prediction_draft_tokens: Option<Option<u32>>,
    // DFlash speculative decoding is disabled by design, so this key is
    // accepted and deliberately ignored wherever it appears (global or
    // per-model). The field stays on the parse surface only because this
    // struct denies unknown fields: a configuration written while DFlash
    // was enabled must keep parsing. resolve() reads and discards it.
    #[serde(
        default,
        deserialize_with = "deserialize_present_nullable_usize",
        skip_serializing_if = "Option::is_none"
    )]
    pub(crate) speculative_decoding_block_length_tokens: Option<Option<usize>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) ssd_prompt_cache_common_prefix_stride_blocks: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) experimental_in_memory_prefix_adoption_enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) experimental_neural_engine_prefill_enabled: Option<bool>,
}

/// Presence map for advanced chunking fields after global/model inheritance.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ConfiguredChunkingFields {
    pub prompt_processing_chunk_size_tokens: bool,
    pub fixed_ssd_streaming_prompt_processing_chunk_size_tokens: bool,
    pub full_attention_key_value_growth_tokens: bool,
    pub prefill_graph_submission_layer_interval: bool,
    pub experimental_ssd_paging_generation_graph_submission_layer_interval: bool,
    pub ssd_prompt_cache_block_tokens: bool,
    pub multi_token_prediction_draft_tokens: bool,
    pub ssd_prompt_cache_common_prefix_stride_blocks: bool,
    pub experimental_in_memory_prefix_adoption_enabled: bool,
    pub experimental_neural_engine_prefill_enabled: bool,
}

fn deserialize_present_nullable_u32<'de, Deserializer>(
    deserializer: Deserializer,
) -> Result<Option<Option<u32>>, Deserializer::Error>
where
    Deserializer: serde::Deserializer<'de>,
{
    Option::<u32>::deserialize(deserializer).map(Some)
}

/// A draft count is refused where it is read, so no configuration holding
/// more drafts than a head can make ever resolves.
fn deserialize_present_nullable_draft_tokens<'de, Deserializer>(
    deserializer: Deserializer,
) -> Result<Option<Option<u32>>, Deserializer::Error>
where
    Deserializer: serde::Deserializer<'de>,
{
    let draft_tokens = Option::<u32>::deserialize(deserializer)?;
    if draft_tokens.is_some_and(exceeds_what_a_head_drafts) {
        return Err(serde::de::Error::custom(
            "chunking.multi_token_prediction_draft_tokens must be null for the engine's own count, 0 for none, or at most 3",
        ));
    }
    Ok(Some(draft_tokens))
}

const fn exceeds_what_a_head_drafts(draft_tokens: u32) -> bool {
    draft_tokens > MAXIMUM_MULTI_TOKEN_PREDICTION_DRAFT_TOKENS
}

fn deserialize_present_nullable_usize<'de, Deserializer>(
    deserializer: Deserializer,
) -> Result<Option<Option<usize>>, Deserializer::Error>
where
    Deserializer: serde::Deserializer<'de>,
{
    Option::<usize>::deserialize(deserializer).map(Some)
}

impl ChunkingConfigFile {
    pub(crate) fn configured_fields(&self) -> ConfiguredChunkingFields {
        ConfiguredChunkingFields {
            prompt_processing_chunk_size_tokens: self.prompt_processing_chunk_size_tokens.is_some(),
            fixed_ssd_streaming_prompt_processing_chunk_size_tokens: self
                .fixed_ssd_streaming_prompt_processing_chunk_size_tokens
                .is_some(),
            full_attention_key_value_growth_tokens: self
                .full_attention_key_value_growth_tokens
                .is_some(),
            prefill_graph_submission_layer_interval: self
                .prefill_graph_submission_layer_interval
                .is_some(),
            experimental_ssd_paging_generation_graph_submission_layer_interval: self
                .experimental_ssd_paging_generation_graph_submission_layer_interval
                .is_some(),
            ssd_prompt_cache_block_tokens: self.ssd_prompt_cache_block_tokens.is_some(),
            multi_token_prediction_draft_tokens: self.multi_token_prediction_draft_tokens.is_some(),
            ssd_prompt_cache_common_prefix_stride_blocks: self
                .ssd_prompt_cache_common_prefix_stride_blocks
                .is_some(),
            experimental_in_memory_prefix_adoption_enabled: self
                .experimental_in_memory_prefix_adoption_enabled
                .is_some(),
            experimental_neural_engine_prefill_enabled: self
                .experimental_neural_engine_prefill_enabled
                .is_some(),
        }
    }

    pub(crate) fn merged(global: &Self, model: Option<&Self>) -> Self {
        let Some(model) = model else {
            return global.clone();
        };
        // The speculative block length is deliberately not inherited: the
        // key is accepted-and-ignored by design (see the file struct's
        // field comment), so the merged document carries None.
        Self {
            speculative_decoding_block_length_tokens: None,
            prompt_processing_chunk_size_tokens: model
                .prompt_processing_chunk_size_tokens
                .or(global.prompt_processing_chunk_size_tokens),
            fixed_ssd_streaming_prompt_processing_chunk_size_tokens: model
                .fixed_ssd_streaming_prompt_processing_chunk_size_tokens
                .or(global.fixed_ssd_streaming_prompt_processing_chunk_size_tokens),
            full_attention_key_value_growth_tokens: model
                .full_attention_key_value_growth_tokens
                .or(global.full_attention_key_value_growth_tokens),
            prefill_graph_submission_layer_interval: model
                .prefill_graph_submission_layer_interval
                .or(global.prefill_graph_submission_layer_interval),
            experimental_ssd_paging_generation_graph_submission_layer_interval: model
                .experimental_ssd_paging_generation_graph_submission_layer_interval
                .or(global.experimental_ssd_paging_generation_graph_submission_layer_interval),
            ssd_prompt_cache_block_tokens: model
                .ssd_prompt_cache_block_tokens
                .or(global.ssd_prompt_cache_block_tokens),
            multi_token_prediction_draft_tokens: model
                .multi_token_prediction_draft_tokens
                .or(global.multi_token_prediction_draft_tokens),
            ssd_prompt_cache_common_prefix_stride_blocks: model
                .ssd_prompt_cache_common_prefix_stride_blocks
                .or(global.ssd_prompt_cache_common_prefix_stride_blocks),
            experimental_in_memory_prefix_adoption_enabled: model
                .experimental_in_memory_prefix_adoption_enabled
                .or(global.experimental_in_memory_prefix_adoption_enabled),
            experimental_neural_engine_prefill_enabled: model
                .experimental_neural_engine_prefill_enabled
                .or(global.experimental_neural_engine_prefill_enabled),
        }
    }
}

pub const DEFAULT_FULL_ATTENTION_KEY_VALUE_GROWTH_TOKENS: u32 = 256;
/// Independent SSD-paged prompt chunk. Owned separately from the resident chunk.
///
/// Each SSD-paged prefill chunk re-streams every nonresident expert layer, so
/// halving the chunk count halves that traffic. A controlled A/B on the large
/// sparse model (issue #1717: second-of-two warm requests, alternating order,
/// three rounds, resident chunk held at 2,048) measured 4,096 faster in 12 of
/// 12 pairings: 21% at a 24 GiB ceiling (about 40% fewer bytes read from the
/// SSD) and 13% at 37 GiB, with identical peak memory and 41k-token prompts
/// still admitted, with the macOS file cache both cold and warm. The in-memory
/// chunk is a separate setting whose Automatic size the engine picks per model
/// and Mac (`text_generation::prompt_processing_chunk_choice`).
#[cfg(target_os = "macos")]
pub const DEFAULT_FIXED_SSD_STREAMING_PROMPT_PROCESSING_CHUNK_SIZE_TOKENS: u32 = 4_096;
/// The SSD-paged probe also runs one chunk at load, so non-macOS builds use
/// the same small chunk the engine picks for in-memory prompts there.
#[cfg(not(target_os = "macos"))]
pub const DEFAULT_FIXED_SSD_STREAMING_PROMPT_PROCESSING_CHUNK_SIZE_TOKENS: u32 = 64;
/// The previously shipped SSD-paged chunk default. Load-time persistence moves
/// this exact value to the current default; any other persisted value is
/// deliberate user intent and is kept.
pub const LEGACY_DEFAULT_FIXED_SSD_STREAMING_PROMPT_PROCESSING_CHUNK_SIZE_TOKENS: u32 = 2_048;

/// SSD-paged sparse models are not served on non-macOS builds yet, so a
/// saved SSD-paged chunk is not retired there beyond what macOS retires.
const RETIRED_SSD_STREAMING_CHUNK_DEFAULTS: &[u32] =
    &[LEGACY_DEFAULT_FIXED_SSD_STREAMING_PROMPT_PROCESSING_CHUNK_SIZE_TOKENS];

/// Pure: whether the persisted SSD-paged chunk should be (re)written to the
/// current default — absent, or still a retired default.
#[must_use]
pub const fn should_materialize_ssd_streaming_chunk_default(
    persisted_chunk_tokens: Option<u32>,
) -> bool {
    is_absent_or_retired(persisted_chunk_tokens, RETIRED_SSD_STREAMING_CHUNK_DEFAULTS)
}

const fn is_absent_or_retired(
    persisted_chunk_tokens: Option<u32>,
    retired_defaults: &[u32],
) -> bool {
    let Some(chunk_tokens) = persisted_chunk_tokens else {
        return true;
    };
    let mut retired_default_index = 0;
    while retired_default_index < retired_defaults.len() {
        if retired_defaults[retired_default_index] == chunk_tokens {
            return true;
        }
        retired_default_index += 1;
    }
    false
}
/// One-token decode submits every third layer while experts stream from SSD.
/// The exact value is not sensitive on the 4-bit sparse model: 1, 2, 3, 4
/// and 6 decoded within two percent of each other at a 14 GiB ceiling (about
/// 33.6 tok/s), as did 1, 3 and 6 at 20 GiB (about 60). Submitting inside the
/// forward at all is what matters: with no intermediate submission decode
/// fell six percent at 20 GiB (56.8 against 60.3 tok/s). Three keeps that
/// gain with a third of the hand-offs of submitting every layer.
pub const DEFAULT_EXPERIMENTAL_SSD_PAGING_GENERATION_GRAPH_SUBMISSION_LAYER_INTERVAL: u32 = 3;
/// Resident multi-token prefill keeps one lazy tape unless the user raises this.
pub const DEFAULT_PREFILL_GRAPH_SUBMISSION_LAYER_INTERVAL: u32 = 0;
pub const DEFAULT_ASTRO_SSD_PROMPT_CACHE_COMMON_PREFIX_STRIDE_BLOCKS: u32 = 4;
/// The most tokens a multi-token-prediction head may draft a round. The
/// head was trained to draft three, and each further draft is verified by
/// one more row of the model whether it is kept or not.
pub const MAXIMUM_MULTI_TOKEN_PREDICTION_DRAFT_TOKENS: u32 = 3;

/// Resolved user-visible boundaries that partition model-serving work.
///
/// Resolution is deliberately centralized at the configuration boundary. The
/// daemon passes this complete contract to the worker, and downstream
/// components convert only numeric representation—not policy or defaults.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkingConfig {
    prompt_processing_chunk_size_tokens: Option<u32>,
    fixed_ssd_streaming_prompt_processing_chunk_size_tokens: u32,
    full_attention_key_value_growth_tokens: u32,
    prefill_graph_submission_layer_interval: u32,
    experimental_ssd_paging_generation_graph_submission_layer_interval: u32,
    ssd_prompt_cache_block_tokens: Option<u32>,
    multi_token_prediction_draft_tokens: Option<u32>,
    ssd_prompt_cache_common_prefix_stride_blocks: u32,
    experimental_in_memory_prefix_adoption_enabled: bool,
    experimental_neural_engine_prefill_enabled: bool,
}

impl ChunkingConfig {
    /// Resolves all chunking defaults and validates them before worker startup.
    ///
    /// Downstream layers receive this complete policy and must not reconstruct
    /// hidden defaults from optional user fields.
    pub(crate) fn resolve(
        configured: &ChunkingConfigFile,
    ) -> Result<Self, AstronomicalConfigError> {
        // DFlash speculative decoding is disabled by design: the
        // speculative block length is read here and discarded, so a
        // configuration that still carries the key parses cleanly and the
        // value is ignored everywhere (see the field's comment on the
        // file struct above). Re-enabling is a deliberate design change,
        // not a configuration flip.
        let _ignored_speculative_decoding_block_length_tokens =
            configured.speculative_decoding_block_length_tokens;
        let resolved = Self {
            prompt_processing_chunk_size_tokens: configured
                .prompt_processing_chunk_size_tokens
                .unwrap_or(None),
            fixed_ssd_streaming_prompt_processing_chunk_size_tokens: configured
                .fixed_ssd_streaming_prompt_processing_chunk_size_tokens
                .unwrap_or(DEFAULT_FIXED_SSD_STREAMING_PROMPT_PROCESSING_CHUNK_SIZE_TOKENS),
            full_attention_key_value_growth_tokens: configured
                .full_attention_key_value_growth_tokens
                .unwrap_or(DEFAULT_FULL_ATTENTION_KEY_VALUE_GROWTH_TOKENS),
            prefill_graph_submission_layer_interval: configured
                .prefill_graph_submission_layer_interval
                .unwrap_or(DEFAULT_PREFILL_GRAPH_SUBMISSION_LAYER_INTERVAL),
            experimental_ssd_paging_generation_graph_submission_layer_interval: configured
                .experimental_ssd_paging_generation_graph_submission_layer_interval
                .unwrap_or(
                    DEFAULT_EXPERIMENTAL_SSD_PAGING_GENERATION_GRAPH_SUBMISSION_LAYER_INTERVAL,
                ),
            ssd_prompt_cache_block_tokens: configured.ssd_prompt_cache_block_tokens.unwrap_or(None),
            multi_token_prediction_draft_tokens: configured
                .multi_token_prediction_draft_tokens
                .unwrap_or(None),
            ssd_prompt_cache_common_prefix_stride_blocks: configured
                .ssd_prompt_cache_common_prefix_stride_blocks
                .unwrap_or(DEFAULT_ASTRO_SSD_PROMPT_CACHE_COMMON_PREFIX_STRIDE_BLOCKS),
            experimental_in_memory_prefix_adoption_enabled: configured
                .experimental_in_memory_prefix_adoption_enabled
                .unwrap_or(true),
            experimental_neural_engine_prefill_enabled: configured
                .experimental_neural_engine_prefill_enabled
                .unwrap_or(false),
        };
        resolved.validate()?;
        Ok(resolved)
    }

    fn validate(&self) -> Result<(), AstronomicalConfigError> {
        self.validate_positive_counts()?;
        self.validate_automatic_or_positive_counts()?;
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

    /// Counts that must always hold a positive number.
    fn validate_positive_counts(&self) -> Result<(), AstronomicalConfigError> {
        for (field_name, field_value) in [
            (
                "chunking.full_attention_key_value_growth_tokens",
                self.full_attention_key_value_growth_tokens,
            ),
            (
                "chunking.ssd_prompt_cache_common_prefix_stride_blocks",
                self.ssd_prompt_cache_common_prefix_stride_blocks,
            ),
            (
                "chunking.fixed_ssd_streaming_prompt_processing_chunk_size_tokens",
                self.fixed_ssd_streaming_prompt_processing_chunk_size_tokens,
            ),
        ] {
            if field_value == 0 {
                return Err(AstronomicalConfigError::InvalidChunkingValue {
                    field_name,
                    description: "must be positive",
                });
            }
        }
        Ok(())
    }

    /// Sizes where `null` asks for automatic sizing and a number must be
    /// positive.
    fn validate_automatic_or_positive_counts(&self) -> Result<(), AstronomicalConfigError> {
        for (field_name, field_value) in [
            (
                "chunking.prompt_processing_chunk_size_tokens",
                self.prompt_processing_chunk_size_tokens,
            ),
            (
                "chunking.ssd_prompt_cache_block_tokens",
                self.ssd_prompt_cache_block_tokens,
            ),
        ] {
            if field_value == Some(0) {
                return Err(AstronomicalConfigError::InvalidChunkingValue {
                    field_name,
                    description: "must be null for automatic sizing or a positive token count",
                });
            }
        }
        Ok(())
    }

    /// The in-memory prompt chunk the user chose; `None` is Automatic, which
    /// the engine settles when the model loads.
    #[must_use]
    pub const fn prompt_processing_chunk_size_tokens(&self) -> Option<u32> {
        self.prompt_processing_chunk_size_tokens
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
    pub const fn prefill_graph_submission_layer_interval(&self) -> u32 {
        self.prefill_graph_submission_layer_interval
    }

    #[must_use]
    pub const fn experimental_ssd_paging_generation_graph_submission_layer_interval(&self) -> u32 {
        self.experimental_ssd_paging_generation_graph_submission_layer_interval
    }

    pub const fn ssd_prompt_cache_block_tokens(&self) -> Option<u32> {
        self.ssd_prompt_cache_block_tokens
    }

    /// Tokens drafted after each reply token by a model with a
    /// multi-token-prediction head; `None` leaves the count to the engine.
    #[must_use]
    pub const fn multi_token_prediction_draft_tokens(&self) -> Option<u32> {
        self.multi_token_prediction_draft_tokens
    }

    #[must_use]
    pub const fn ssd_prompt_cache_common_prefix_stride_blocks(&self) -> u32 {
        self.ssd_prompt_cache_common_prefix_stride_blocks
    }

    /// Returns whether the engine may retain the just-completed generation's
    /// live decoder state and adopt it for the next turn of the same
    /// conversation instead of re-reading the prefix KV from the disk store.
    /// Adopted state is byte-identical to the restored state, so no numerics
    /// change; the trade is one conversation's KV held live between requests.
    /// Default on: the A/B on the resident measurement journey showed the
    /// byte-moving KV-block reads eliminated (six restore spans to three,
    /// the remainder being the lookup and the small boundary snapshot), with
    /// unchanged rates at the journey's astro SSD prompt cache depth and the production win
    /// growing with astro SSD prompt cache depth and cold stores.
    #[must_use]
    pub const fn experimental_in_memory_prefix_adoption_enabled(&self) -> bool {
        self.experimental_in_memory_prefix_adoption_enabled
    }

    /// Returns whether dense prefill may run a slice of each chunk's tokens
    /// through whole MLP programs on the Neural Engine. Off by default because
    /// the programs hold more memory than the model's own fp16 MLP weights, by an amount
    /// measured for each model.
    #[must_use]
    pub const fn experimental_neural_engine_prefill_enabled(&self) -> bool {
        self.experimental_neural_engine_prefill_enabled
    }
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            experimental_in_memory_prefix_adoption_enabled: true,
            experimental_neural_engine_prefill_enabled: false,
            prompt_processing_chunk_size_tokens: None,
            fixed_ssd_streaming_prompt_processing_chunk_size_tokens:
                DEFAULT_FIXED_SSD_STREAMING_PROMPT_PROCESSING_CHUNK_SIZE_TOKENS,
            full_attention_key_value_growth_tokens: DEFAULT_FULL_ATTENTION_KEY_VALUE_GROWTH_TOKENS,
            prefill_graph_submission_layer_interval:
                DEFAULT_PREFILL_GRAPH_SUBMISSION_LAYER_INTERVAL,
            experimental_ssd_paging_generation_graph_submission_layer_interval:
                DEFAULT_EXPERIMENTAL_SSD_PAGING_GENERATION_GRAPH_SUBMISSION_LAYER_INTERVAL,
            ssd_prompt_cache_block_tokens: None,
            multi_token_prediction_draft_tokens: None,
            ssd_prompt_cache_common_prefix_stride_blocks:
                DEFAULT_ASTRO_SSD_PROMPT_CACHE_COMMON_PREFIX_STRIDE_BLOCKS,
        }
    }
}
