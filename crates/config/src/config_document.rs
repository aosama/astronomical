//! Defines the strict schema-version-1 persisted document and its typed validation.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::chunking_config::ChunkingConfigFile;
use crate::{AstronomicalConfigError, LogLevel, prompt_cache_size_gb_to_bytes};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct UserConfigFile {
    #[serde(rename = "$schema")]
    pub(crate) schema: String,
    pub(crate) schema_version: u32,
    pub(crate) runtime: RuntimeConfigFile,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) prompt_cache: Option<PromptCacheConfigFile>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) chunking: Option<ChunkingConfigFile>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub(crate) models: BTreeMap<String, ModelConfigFile>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) diagnostics: Option<DiagnosticsConfigFile>,
}

impl UserConfigFile {
    pub(crate) fn minimal() -> Self {
        Self {
            schema: "./astronomical-config.schema.json".to_owned(),
            schema_version: 1,
            runtime: RuntimeConfigFile {
                model_directories: Vec::new(),
                maximum_mlx_memory_gb: None,
                experimental_qwen_thinking_channel_seed_enabled: None,
            },
            prompt_cache: None,
            chunking: None,
            models: BTreeMap::new(),
            diagnostics: None,
        }
    }

    pub(crate) fn validate(&self) -> Result<(), AstronomicalConfigError> {
        if let Some(prompt_cache) = &self.prompt_cache {
            if let Some(maximum_size_gb) = prompt_cache.maximum_size_gb {
                prompt_cache_size_gb_to_bytes(maximum_size_gb)?;
            }
        }
        if self
            .diagnostics
            .as_ref()
            .and_then(|diagnostics| diagnostics.retained_log_files)
            == Some(0)
        {
            return Err(AstronomicalConfigError::InvalidRetainedLogFileCount);
        }
        let global_chunking = self.chunking.clone().unwrap_or_default();
        crate::ChunkingConfig::resolve(&global_chunking)?;
        for (model_id, model_config) in &self.models {
            model_config.validate(model_id, &global_chunking)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RuntimeConfigFile {
    pub(crate) model_directories: Vec<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) maximum_mlx_memory_gb: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) experimental_qwen_thinking_channel_seed_enabled: Option<bool>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PromptCacheConfigFile {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) maximum_size_gb: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DiagnosticsConfigFile {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) performance_attribution_enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) log_level: Option<LogLevel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) retained_log_files: Option<usize>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ModelConfigFile {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) limits: Option<ModelLimitsConfigFile>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) generation_defaults: Option<GenerationDefaultsConfigFile>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) chunking: Option<ChunkingConfigFile>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) acceleration: Option<AccelerationConfigFile>,
}

impl ModelConfigFile {
    fn validate(
        &self,
        model_id: &str,
        global_chunking: &ChunkingConfigFile,
    ) -> Result<(), AstronomicalConfigError> {
        if model_id.trim().is_empty() {
            return Err(AstronomicalConfigError::InvalidModelConfig {
                model_id: model_id.to_owned(),
                field_name: "model ID",
                description: "must not be empty",
            });
        }
        if !is_valid_model_identity(model_id) {
            return Err(invalid_model_value(
                model_id,
                "model ID",
                "must not contain surrounding whitespace or control characters",
            ));
        }
        if let Some(limits) = &self.limits
            && limits
                .maximum_context_tokens
                .is_some_and(|maximum_context_tokens| maximum_context_tokens < 2)
        {
            return Err(invalid_model_value(
                model_id,
                "limits.maximum_context_tokens",
                "must leave room for prompt and output tokens",
            ));
        }
        if let Some(generation_defaults) = &self.generation_defaults {
            generation_defaults.validate(model_id)?;
        }
        let effective_chunking =
            ChunkingConfigFile::merged(global_chunking, self.chunking.as_ref());
        crate::ChunkingConfig::resolve(&effective_chunking)?;
        if let Some(acceleration) = &self.acceleration {
            acceleration.validate(model_id)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ModelLimitsConfigFile {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) maximum_context_tokens: Option<u32>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GenerationDefaultsConfigFile {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) maximum_output_tokens: Option<u32>,
}

impl GenerationDefaultsConfigFile {
    fn validate(&self, model_id: &str) -> Result<(), AstronomicalConfigError> {
        if self
            .temperature
            .is_some_and(|temperature| !(0.0..=2.0).contains(&temperature))
        {
            return Err(invalid_model_value(
                model_id,
                "generation_defaults.temperature",
                "must be between 0 and 2",
            ));
        }
        if self
            .temperature
            .is_some_and(|temperature| !is_representable_in_thousandths(temperature))
        {
            return Err(invalid_model_value(
                model_id,
                "generation_defaults.temperature",
                "must be representable in thousandths",
            ));
        }
        if self
            .top_p
            .is_some_and(|top_p| !(0.0..=1.0).contains(&top_p))
        {
            return Err(invalid_model_value(
                model_id,
                "generation_defaults.top_p",
                "must be between 0 and 1",
            ));
        }
        if self
            .top_p
            .is_some_and(|top_p| !is_representable_in_thousandths(top_p))
        {
            return Err(invalid_model_value(
                model_id,
                "generation_defaults.top_p",
                "must be representable in thousandths",
            ));
        }
        if self.maximum_output_tokens == Some(0) {
            return Err(invalid_model_value(
                model_id,
                "generation_defaults.maximum_output_tokens",
                "must be positive",
            ));
        }
        if self
            .maximum_output_tokens
            .is_some_and(|maximum_output_tokens| maximum_output_tokens > u32::from(u16::MAX))
        {
            return Err(invalid_model_value(
                model_id,
                "generation_defaults.maximum_output_tokens",
                "must fit the worker protocol maximum of 65535 tokens",
            ));
        }
        Ok(())
    }
}

fn is_representable_in_thousandths(sampling_parameter: f32) -> bool {
    let scaled_sampling_parameter = sampling_parameter * 1_000.0;
    (scaled_sampling_parameter - scaled_sampling_parameter.round()).abs() <= 0.000_1
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AccelerationConfigFile {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) speculative_prefill: Option<SpeculativePrefillConfigFile>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) mtp: Option<MtpConfigFile>,
}

impl AccelerationConfigFile {
    fn validate(&self, model_id: &str) -> Result<(), AstronomicalConfigError> {
        if let Some(speculative_prefill) = &self.speculative_prefill {
            speculative_prefill.validate(model_id)?;
        }
        if self
            .mtp
            .as_ref()
            .and_then(|mtp| mtp.draft_depth)
            .is_some_and(|draft_depth| !(1..=3).contains(&draft_depth))
        {
            return Err(AstronomicalConfigError::InvalidMtpDraftDepth);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SpeculativePrefillConfigFile {
    pub(crate) draft_model_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) keep_percentage: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) minimum_prompt_tokens: Option<u32>,
}

impl SpeculativePrefillConfigFile {
    fn validate(&self, model_id: &str) -> Result<(), AstronomicalConfigError> {
        if !is_valid_model_identity(&self.draft_model_id) {
            return Err(invalid_model_value(
                model_id,
                "acceleration.speculative_prefill.draft_model_id",
                "must be nonempty and contain no surrounding whitespace or control characters",
            ));
        }
        if self
            .keep_percentage
            .is_some_and(|keep_percentage| !(1..=100).contains(&keep_percentage))
        {
            return Err(AstronomicalConfigError::SpeculativePrefillKeepPercentageOutOfRange);
        }
        if self.minimum_prompt_tokens == Some(0) {
            return Err(
                AstronomicalConfigError::SpeculativePrefillMinimumPromptTokensMustBePositive,
            );
        }
        Ok(())
    }
}

fn is_valid_model_identity(model_id: &str) -> bool {
    !model_id.is_empty() && model_id.trim() == model_id && !model_id.chars().any(char::is_control)
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct MtpConfigFile {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) draft_depth: Option<u8>,
}

fn invalid_model_value(
    model_id: &str,
    field_name: &'static str,
    description: &'static str,
) -> AstronomicalConfigError {
    AstronomicalConfigError::InvalidModelConfig {
        model_id: model_id.to_owned(),
        field_name,
        description,
    }
}
