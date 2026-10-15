//! Fill-omissions merge tests for the repetition penalty generation default.
//!
//! These tests verify that the daemon resolves the configured repetition
//! penalty into omitted request settings without ever overriding an explicit
//! caller value, without starting a worker process or HTTP listener.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use crate::config::request_generation_defaults::{
    RequestGenerationSettingsPresence, apply_generation_defaults,
};
use crate::{ResolvedRuntimeConfig, RuntimeModelGenerationDefaults, RuntimeModelPolicy};
use astronomical_config::{LogLevel, LoggingConfig, PromptCacheConfig};
use astronomical_ipc_protocol::{
    ChatGenerationSettings, WorkerEmbeddingModelConfiguration, WorkerEmbeddingModelFamily,
    WorkerModelConfiguration,
};

#[test]
fn should_fill_an_omitted_penalty_from_the_model_policy() {
    let reloadable_runtime_config = resolved_runtime_config_with_penalty(Some(1.15));
    let mut generation_settings = settings_with_penalty(None);

    apply_generation_defaults(
        Some(&reloadable_runtime_config),
        "default",
        presence_with_penalty(false),
        &mut generation_settings,
    );

    assert_eq!(
        generation_settings.repetition_penalty_thousandths,
        Some(1_150),
        "an omitted request penalty must take the configured model default"
    );
}

#[test]
fn should_keep_an_explicit_penalty_over_the_model_policy() {
    let reloadable_runtime_config = resolved_runtime_config_with_penalty(Some(1.15));
    let mut generation_settings = settings_with_penalty(Some(1_000));

    apply_generation_defaults(
        Some(&reloadable_runtime_config),
        "default",
        presence_with_penalty(true),
        &mut generation_settings,
    );

    assert_eq!(
        generation_settings.repetition_penalty_thousandths,
        Some(1_000),
        "an explicit caller penalty must survive the merge unchanged"
    );
}

#[test]
fn should_resolve_an_unconfigured_model_to_no_penalty() {
    let reloadable_runtime_config = resolved_runtime_config_with_penalty(None);
    let mut generation_settings = settings_with_penalty(None);

    apply_generation_defaults(
        Some(&reloadable_runtime_config),
        "default",
        presence_with_penalty(false),
        &mut generation_settings,
    );

    assert_eq!(
        generation_settings.repetition_penalty_thousandths, None,
        "a model without a configured penalty must stay unset so the engine keeps its neutral identity"
    );
}

fn presence_with_penalty(repetition_penalty_present: bool) -> RequestGenerationSettingsPresence {
    RequestGenerationSettingsPresence {
        maximum_output_tokens: false,
        temperature: false,
        top_p: false,
        repetition_penalty: repetition_penalty_present,
    }
}

fn settings_with_penalty(repetition_penalty_thousandths: Option<u16>) -> ChatGenerationSettings {
    ChatGenerationSettings {
        max_output_tokens: 256,
        temperature_thousandths: None,
        top_p_thousandths: None,
        repetition_penalty_thousandths,
        seed: None,
        thinking_budget: None,
    }
}

fn resolved_runtime_config_with_penalty(
    configured_repetition_penalty: Option<f32>,
) -> Arc<std::sync::RwLock<ResolvedRuntimeConfig>> {
    let generation_defaults = RuntimeModelGenerationDefaults {
        maximum_output_tokens: 20_480,
        configured_maximum_output_tokens: None,
        temperature_thousandths: None,
        top_p_thousandths: None,
        repetition_penalty_thousandths: configured_repetition_penalty
            .map(|repetition_penalty| (repetition_penalty * 1_000.0).round() as u16),
    };
    let resolved_runtime_config = ResolvedRuntimeConfig {
        configuration_generation: "751".repeat(8),
        worker_executable_path: PathBuf::from("/tmp/astronomical-inference-worker"),
        discovered_models: Vec::new(),
        model_discovery_diagnostics: Vec::new(),
        configured_model_directories: Vec::new(),
        model_policy_catalog: Arc::new(HashMap::from([(
            "default".to_owned(),
            RuntimeModelPolicy {
                model_directory: PathBuf::from("/tmp/models/default"),
                generation_defaults,
                configured_maximum_context_tokens: None,
                default_maximum_context_tokens: 65_536,
                configured_chunking_fields: Default::default(),
                worker_model_configuration: WorkerModelConfiguration::Embeddings(
                    WorkerEmbeddingModelConfiguration {
                        model_id: "default".to_owned(),
                        model_family: WorkerEmbeddingModelFamily::ModernBert,
                        artifact_revision: "test-revision".to_owned(),
                        vector_width: 1,
                        maximum_input_tokens: 512,
                    },
                ),
            },
        )])),
        unmatched_model_config_ids: Vec::new(),
        maximum_mlx_memory_bytes: None,
        persistent_prompt_cache_enabled: false,
        configured_persistent_prompt_cache_enabled: None,
        configured_prompt_cache_maximum_size_bytes: None,
        performance_attribution_enabled: false,
        prompt_cache_config: PromptCacheConfig::new(PathBuf::from("/tmp/prompt-cache"), 1),
        bind_address: "127.0.0.1:6733".to_owned(),
        logging_config: LoggingConfig::new(
            PathBuf::from("/tmp/astronomical-logs"),
            LogLevel::Warn,
            7,
        ),
    };
    Arc::new(std::sync::RwLock::new(resolved_runtime_config))
}
