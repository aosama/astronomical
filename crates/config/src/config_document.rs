//! Defines the strict schema-version-1 persisted document and its typed validation.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::chunking_config::ChunkingConfigFile;
use crate::{AstronomicalConfigError, LogLevel};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct UserConfigFile {
    #[serde(rename = "$schema")]
    pub(crate) schema: String,
    pub(crate) schema_version: u32,
    pub(crate) runtime: RuntimeConfigFile,
    /// The `astro_ssd_prompt_cache` settings. The file is Astronomical's own,
    /// so its keys leave out the `astro_` owner prefix the code carries.
    #[serde(
        default,
        rename = "ssd_prompt_cache",
        skip_serializing_if = "Option::is_none"
    )]
    pub(crate) astro_ssd_prompt_cache: Option<AstroSsdPromptCacheConfigFile>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) chunking: Option<ChunkingConfigFile>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub(crate) models: BTreeMap<String, ModelConfigFile>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) diagnostics: Option<DiagnosticsConfigFile>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) menu_bar: Option<MenuBarConfigFile>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) skill: Option<SkillConfigFile>,
}

impl UserConfigFile {
    pub(crate) fn minimal() -> Self {
        Self {
            schema: "./astronomical-config.schema.json".to_owned(),
            schema_version: 1,
            runtime: RuntimeConfigFile {
                model_directories: Vec::new(),
                maximum_mlx_memory_gb: None,
                default_model: None,
                model_download_directory: None,
                neural_engine_program_directory: None,
                metal_residency_refresh_enabled: None,
                idle_model_unload_seconds: None,
            },
            astro_ssd_prompt_cache: None,
            chunking: Some(ChunkingConfigFile {
                fixed_prompt_processing_chunk_size_tokens: Some(
                    crate::DEFAULT_FIXED_PROMPT_PROCESSING_CHUNK_SIZE_TOKENS,
                ),
                fixed_ssd_streaming_prompt_processing_chunk_size_tokens: Some(
                    crate::DEFAULT_FIXED_SSD_STREAMING_PROMPT_PROCESSING_CHUNK_SIZE_TOKENS,
                ),
                prefill_graph_submission_layer_interval: Some(
                    crate::DEFAULT_PREFILL_GRAPH_SUBMISSION_LAYER_INTERVAL,
                ),
                ..ChunkingConfigFile::default()
            }),
            models: BTreeMap::new(),
            diagnostics: None,
            menu_bar: None,
            skill: None,
        }
    }

    pub(crate) fn validate(&self) -> Result<(), AstronomicalConfigError> {
        if let Some(astro_ssd_prompt_cache) = &self.astro_ssd_prompt_cache
            && let Some(maximum_size_gb) = astro_ssd_prompt_cache.maximum_size_gb
        {
            crate::astro_ssd_prompt_cache_size_gb_to_bytes(maximum_size_gb)?;
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) default_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) model_download_directory: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) neural_engine_program_directory: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) metal_residency_refresh_enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) idle_model_unload_seconds: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AstroSsdPromptCacheConfigFile {
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
    pub(crate) completion_attribution_enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) log_level: Option<LogLevel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) retained_log_files: Option<usize>,
}

/// What the macOS menu bar item shows. It lives in this file, not in the
/// menu-bar agent's own preferences, so Observatory's Settings page and the
/// agent's popover edit one stored value and can never disagree.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct MenuBarConfigFile {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) show_progress_and_speed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) show_technical_details: Option<bool>,
}

/// How the Astronomical skill reaches the models this app serves and how it
/// stays current. The daemon applies both switches itself; no worker reads
/// them.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SkillConfigFile {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) declare_to_models: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) update_from_web: Option<bool>,
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
    pub(crate) repetition_penalty: Option<f32>,
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
        if self
            .repetition_penalty
            .is_some_and(|repetition_penalty| !(1.0..=2.0).contains(&repetition_penalty))
        {
            return Err(invalid_model_value(
                model_id,
                "generation_defaults.repetition_penalty",
                "must be between 1 and 2; values below 1 would reward repetition",
            ));
        }
        if self
            .repetition_penalty
            .is_some_and(|repetition_penalty| !is_representable_in_thousandths(repetition_penalty))
        {
            return Err(invalid_model_value(
                model_id,
                "generation_defaults.repetition_penalty",
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

fn is_valid_model_identity(model_id: &str) -> bool {
    !model_id.is_empty() && model_id.trim() == model_id && !model_id.chars().any(char::is_control)
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
