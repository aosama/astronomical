//! Resolves one canonical model's inherited limits, generation defaults, chunking, and acceleration.

use crate::chunking_config::{ChunkingConfig, ChunkingConfigFile, ConfiguredChunkingFields};
use crate::config_document::ModelConfigFile;
use crate::{AstronomicalConfigError, SpeculativePrefillConfig};

/// Internal output default used when a model has no configured preference.
pub const DEFAULT_MAXIMUM_OUTPUT_TOKENS: u32 = 20_480;

/// Complete config-owned policy after global and per-model inheritance.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedModelConfig {
    maximum_context_tokens: Option<u32>,
    maximum_output_tokens: u32,
    configured_maximum_output_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    chunking: ChunkingConfig,
    configured_chunking_fields: ConfiguredChunkingFields,
    speculative_prefill: Option<SpeculativePrefillConfig>,
    mtp_draft_depth: Option<u8>,
}

impl ResolvedModelConfig {
    pub(crate) fn resolve(
        model_id: &str,
        artifact_maximum_context_tokens: u32,
        global_chunking: &ChunkingConfigFile,
        configured_model: Option<&ModelConfigFile>,
    ) -> Result<Self, AstronomicalConfigError> {
        let maximum_context_tokens = configured_model
            .and_then(|model| model.limits.as_ref())
            .and_then(|limits| limits.maximum_context_tokens);
        if maximum_context_tokens
            .is_some_and(|configured| configured > artifact_maximum_context_tokens)
        {
            return Err(AstronomicalConfigError::ConfiguredContextExceedsArtifact {
                model_id: model_id.to_owned(),
                configured_maximum_context_tokens: maximum_context_tokens.unwrap_or_default(),
                artifact_maximum_context_tokens,
            });
        }
        let generation_defaults =
            configured_model.and_then(|model| model.generation_defaults.as_ref());
        let effective_maximum_context_tokens =
            maximum_context_tokens.unwrap_or(artifact_maximum_context_tokens);
        let configured_maximum_output_tokens =
            generation_defaults.and_then(|defaults| defaults.maximum_output_tokens);
        if configured_maximum_output_tokens
            .is_some_and(|configured| configured >= effective_maximum_context_tokens)
        {
            return Err(
                AstronomicalConfigError::ConfiguredOutputNotSmallerThanContext {
                    model_id: model_id.to_owned(),
                    configured_maximum_output_tokens: configured_maximum_output_tokens
                        .unwrap_or_default(),
                    effective_maximum_context_tokens,
                },
            );
        }
        let effective_chunking = ChunkingConfigFile::merged(
            global_chunking,
            configured_model.and_then(|model| model.chunking.as_ref()),
        );
        let acceleration = configured_model.and_then(|model| model.acceleration.as_ref());
        let speculative_prefill = acceleration
            .and_then(|acceleration| acceleration.speculative_prefill.as_ref())
            .map(|configured| {
                SpeculativePrefillConfig::for_target(
                    model_id,
                    &configured.draft_model_id,
                    configured.minimum_prompt_tokens,
                    configured.keep_percentage,
                )
            });
        Ok(Self {
            maximum_context_tokens,
            // The internal default is policy, not an explicit user demand, so tiny
            // artifacts retain one prompt token instead of becoming undiscoverable.
            maximum_output_tokens: configured_maximum_output_tokens.unwrap_or_else(|| {
                DEFAULT_MAXIMUM_OUTPUT_TOKENS
                    .min(effective_maximum_context_tokens.saturating_sub(1))
            }),
            configured_maximum_output_tokens,
            temperature: generation_defaults.and_then(|defaults| defaults.temperature),
            top_p: generation_defaults.and_then(|defaults| defaults.top_p),
            chunking: ChunkingConfig::resolve(&effective_chunking)?,
            configured_chunking_fields: effective_chunking.configured_fields(),
            speculative_prefill,
            mtp_draft_depth: acceleration
                .and_then(|acceleration| acceleration.mtp.as_ref())
                .and_then(|mtp| mtp.draft_depth),
        })
    }

    /// Returns the configured operational context ceiling, if one was supplied.
    #[must_use]
    pub const fn maximum_context_tokens(&self) -> Option<u32> {
        self.maximum_context_tokens
    }

    /// Returns the configured output default or Astronomical's internal default.
    #[must_use]
    pub const fn maximum_output_tokens(&self) -> u32 {
        self.maximum_output_tokens
    }

    /// Distinguishes a user-authored generation default from internal fallback policy.
    #[must_use]
    pub const fn has_explicit_maximum_output_tokens(&self) -> bool {
        self.configured_maximum_output_tokens.is_some()
    }

    #[must_use]
    pub const fn configured_maximum_output_tokens(&self) -> Option<u32> {
        self.configured_maximum_output_tokens
    }

    /// Returns the configured sampling temperature without inventing a default.
    #[must_use]
    pub const fn temperature(&self) -> Option<f32> {
        self.temperature
    }

    /// Returns the configured nucleus-sampling probability without inventing a default.
    #[must_use]
    pub const fn top_p(&self) -> Option<f32> {
        self.top_p
    }

    /// Returns the complete chunking policy after global and model inheritance.
    #[must_use]
    pub const fn chunking(&self) -> &ChunkingConfig {
        &self.chunking
    }

    /// Identifies each authored inherited field without labelling sibling defaults as configured.
    #[must_use]
    pub const fn configured_chunking_fields(&self) -> ConfiguredChunkingFields {
        self.configured_chunking_fields
    }

    /// Returns per-target speculative prefill only when its section is present.
    #[must_use]
    pub const fn speculative_prefill(&self) -> Option<&SpeculativePrefillConfig> {
        self.speculative_prefill.as_ref()
    }

    /// Returns the configured proposal depth, or `None` for artifact policy.
    #[must_use]
    pub const fn mtp_draft_depth(&self) -> Option<u8> {
        self.mtp_draft_depth
    }
}
