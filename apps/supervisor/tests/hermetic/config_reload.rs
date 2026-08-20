//! Pure config-reload diff logic tests.
//!
//! These tests verify the classification of changed config fields into
//! reload, worker-restart, and rest-api-restart categories without
//! starting a worker process or HTTP listener.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use astronomical_config::{LogLevel, LoggingConfig, PromptCacheConfig};
use astronomical_ipc_protocol::{WorkerChunkingConfiguration, WorkerModelConfiguration};
use astronomical_supervisor::{
    ConfigReloadDecision, ConfigReloadDiff, ResolvedConfigurationGeneration, ResolvedRuntimeConfig,
    ResolvedRuntimeConfigResolver, RuntimeModelPolicy,
};

#[test]
fn should_mark_bind_address_changes_as_rest_api_restart_required() {
    let current = resolved_config_with_bind_address("127.0.0.1:6733");
    let candidate = resolved_config_with_bind_address("127.0.0.1:6734");

    let decision = ConfigReloadDiff::compare(&current, &candidate);

    match decision {
        ConfigReloadDecision::RestApiRestartRequired {
            ref restart_required_fields,
            ..
        } if restart_required_fields.contains(&"supervisor.bind_address".to_owned()) => {}
        unexpected => {
            panic!("a bind address change must require a full REST API restart, got {unexpected:?}")
        }
    }
}

#[test]
fn should_mark_logging_changes_as_rest_api_restart_required() {
    let current = resolved_config_with_logging_directory("/tmp/astronomical-logs-a");
    let candidate = resolved_config_with_logging_directory("/tmp/astronomical-logs-b");

    let decision = ConfigReloadDiff::compare(&current, &candidate);

    match decision {
        ConfigReloadDecision::RestApiRestartRequired {
            ref restart_required_fields,
            ..
        } if restart_required_fields.contains(&"logging".to_owned()) => {}
        unexpected => panic!(
            "a logging directory change must require a full REST API restart, got {unexpected:?}"
        ),
    }
}

#[test]
fn should_classify_no_changes_as_no_worker_restart() {
    let current = sample_resolved_config();
    let candidate = sample_resolved_config();

    let decision = ConfigReloadDiff::compare(&current, &candidate);

    assert!(
        matches!(decision, ConfigReloadDecision::NoWorkerRestart { .. }),
        "identical configs must not trigger a worker restart, got {decision:?}"
    );
}

#[test]
fn should_change_resolved_generation_when_artifact_revision_changes() {
    let model_policy_catalog =
        HashMap::from([("default".to_owned(), sample_runtime_model_policy())]);
    let mut discovered_model = astronomical_config::DiscoveredModel {
        model_id: "default".to_owned(),
        model_family: astronomical_config::ModelFamily::Qwen3_5,
        revision: "revision-a".to_owned(),
        model_directory: PathBuf::from("/fictional/models/default"),
        context_window: 65_536,
        max_input_tokens: 65_535,
        max_output_tokens: 20_480,
        has_vision: false,
        supports_reasoning: true,
        supports_tool_calls: true,
        model_size_bytes: 1_000,
    };
    let first_generation = ResolvedConfigurationGeneration::derive(
        "document-generation",
        &[discovered_model.clone()],
        &model_policy_catalog,
        &[],
    )
    .expect("resolved generation should derive");
    discovered_model.revision = "revision-b".to_owned();

    let second_generation = ResolvedConfigurationGeneration::derive(
        "document-generation",
        &[discovered_model],
        &model_policy_catalog,
        &[],
    )
    .expect("resolved generation should derive");

    assert_ne!(first_generation, second_generation);
}

#[test]
fn should_restart_worker_when_a_discovered_artifact_revision_changes() {
    let mut current = sample_resolved_config();
    current.discovered_models = vec![sample_discovered_model("revision-a")];
    let mut candidate = current.clone();
    candidate.configuration_generation =
        "2222222222222222222222222222222222222222222222222222222222222222".to_owned();
    candidate.discovered_models[0].revision = "revision-b".to_owned();

    let decision = ConfigReloadDiff::compare(&current, &candidate);

    assert!(matches!(
        decision,
        ConfigReloadDecision::RestartWorker { ref reloaded_fields, .. }
            if reloaded_fields == &["discovered_model_artifacts".to_owned()]
    ));
}

#[test]
fn should_restart_worker_to_acknowledge_changed_dormant_model_policies() {
    let current = sample_resolved_config();
    let mut candidate = current.clone();
    candidate.configuration_generation =
        "2222222222222222222222222222222222222222222222222222222222222222".to_owned();
    candidate.unmatched_model_config_ids = vec!["temporarily-absent-model".to_owned()];

    let decision = ConfigReloadDiff::compare(&current, &candidate);

    assert!(matches!(
        decision,
        ConfigReloadDecision::RestartWorker { ref reloaded_fields, .. }
            if reloaded_fields == &["dormant_model_policies".to_owned()]
    ));
}

#[test]
fn should_restart_worker_when_dormant_policy_content_changes_without_changing_its_id() {
    let mut current = sample_resolved_config();
    current.unmatched_model_config_ids = vec!["temporarily-absent-model".to_owned()];
    let mut candidate = current.clone();
    candidate.configuration_generation =
        "2222222222222222222222222222222222222222222222222222222222222222".to_owned();

    let decision = ConfigReloadDiff::compare(&current, &candidate);

    assert!(matches!(
        decision,
        ConfigReloadDecision::RestartWorker { ref reloaded_fields, .. }
            if reloaded_fields == &["resolved_configuration".to_owned()]
    ));
}

#[test]
fn should_restart_worker_when_any_per_model_execution_policy_changes() {
    let current = sample_resolved_config();
    let mut candidate = sample_resolved_config();
    Arc::make_mut(&mut candidate.model_policy_catalog)
        .get_mut("default")
        .expect("sample policy should exist")
        .worker_model_configuration
        .chunking
        .fixed_prompt_processing_chunk_size_tokens = 4_096;

    let decision = ConfigReloadDiff::compare(&current, &candidate);

    assert!(matches!(
        decision,
        ConfigReloadDecision::RestartWorker { ref reloaded_fields, .. }
            if reloaded_fields == &["model_policies".to_owned()]
    ));
}

#[test]
fn should_classify_performance_attribution_change_as_worker_restart() {
    let mut current = sample_resolved_config();
    current.performance_attribution_enabled = false;
    let mut candidate = sample_resolved_config();
    candidate.performance_attribution_enabled = true;

    let decision = ConfigReloadDiff::compare(&current, &candidate);

    assert!(
        matches!(decision, ConfigReloadDecision::RestartWorker { ref reloaded_fields, .. } if reloaded_fields == &["performance_attribution_enabled".to_owned()]),
        "a performance attribution change must restart the worker, got {decision:?}"
    );
}

#[test]
fn should_classify_model_directories_change_as_worker_restart() {
    let mut current = sample_resolved_config();
    Arc::make_mut(&mut current.model_policy_catalog)
        .get_mut("default")
        .expect("sample policy should exist")
        .model_directory = PathBuf::from("/tmp/models-a");
    let mut candidate = sample_resolved_config();
    Arc::make_mut(&mut candidate.model_policy_catalog)
        .get_mut("default")
        .expect("sample policy should exist")
        .model_directory = PathBuf::from("/tmp/models-b");

    let decision = ConfigReloadDiff::compare(&current, &candidate);

    assert!(
        matches!(decision, ConfigReloadDecision::RestartWorker { ref reloaded_fields, .. } if reloaded_fields.contains(&"model_policies".to_owned())),
        "a model_directories change must trigger a worker restart, got {decision:?}"
    );
}

#[test]
fn should_restart_worker_when_empty_configured_model_root_changes() {
    let mut current = sample_resolved_config();
    current.configured_model_directories = vec![PathBuf::from("/tmp/empty-model-root-a")];
    let mut candidate = sample_resolved_config();
    candidate.configured_model_directories = vec![PathBuf::from("/tmp/empty-model-root-b")];

    let decision = ConfigReloadDiff::compare(&current, &candidate);

    assert!(
        matches!(decision, ConfigReloadDecision::RestartWorker { ref reloaded_fields, .. } if reloaded_fields.contains(&"model_directories".to_owned())),
        "configured model roots must participate in the worker-restart diff, got {decision:?}"
    );
}

#[test]
fn should_classify_prompt_cache_capacity_change_as_worker_restart() {
    let mut current = sample_resolved_config();
    current.prompt_cache_config =
        PromptCacheConfig::new(PathBuf::from("/tmp/prompt-cache"), 1_000_000_000);
    let mut candidate = sample_resolved_config();
    candidate.prompt_cache_config =
        PromptCacheConfig::new(PathBuf::from("/tmp/prompt-cache"), 2_000_000_000);

    let decision = ConfigReloadDiff::compare(&current, &candidate);

    assert!(
        matches!(decision, ConfigReloadDecision::RestartWorker { ref reloaded_fields, .. } if reloaded_fields.contains(&"prompt_cache".to_owned())),
        "a prompt-cache capacity change must restart the worker, got {decision:?}"
    );
}

#[test]
fn should_restart_worker_when_persistent_prompt_cache_flag_changes() {
    let current = sample_resolved_config();
    let mut candidate = sample_resolved_config();
    candidate.persistent_prompt_cache_enabled = false;

    let decision = ConfigReloadDiff::compare(&current, &candidate);

    assert!(
        matches!(decision, ConfigReloadDecision::RestartWorker { ref reloaded_fields, .. }
            if reloaded_fields.contains(&"persistent_prompt_cache_enabled".to_owned())),
        "changing the persistent prompt-cache flag must restart the worker, got {decision:?}"
    );
}

#[test]
fn should_mark_logging_level_change_as_rest_api_restart_required() {
    let mut current = sample_resolved_config();
    current.logging_config = LoggingConfig::new(PathBuf::from("/tmp/logs"), LogLevel::Warn, 7);
    let mut candidate = sample_resolved_config();
    candidate.logging_config = LoggingConfig::new(PathBuf::from("/tmp/logs"), LogLevel::Info, 7);

    let decision = ConfigReloadDiff::compare(&current, &candidate);

    assert!(
        matches!(decision, ConfigReloadDecision::RestApiRestartRequired { ref restart_required_fields, .. } if restart_required_fields.contains(&"logging".to_owned())),
        "a logging level change must require a REST restart, got {decision:?}"
    );
}

fn sample_resolved_config() -> ResolvedRuntimeConfig {
    ResolvedRuntimeConfig {
        configuration_generation:
            "1111111111111111111111111111111111111111111111111111111111111111".to_owned(),
        worker_executable_path: PathBuf::from("/tmp/astronomical-inference-worker"),
        discovered_models: Vec::new(),
        configured_model_directories: Vec::new(),
        model_policy_catalog: Arc::new(HashMap::from([(
            "default".to_owned(),
            sample_runtime_model_policy(),
        )])),
        unmatched_model_config_ids: Vec::new(),
        maximum_mlx_memory_bytes: None,
        persistent_prompt_cache_enabled: true,
        configured_persistent_prompt_cache_enabled: None,
        configured_prompt_cache_maximum_size_bytes: None,
        performance_attribution_enabled: false,
        prompt_cache_config: PromptCacheConfig::new(
            PathBuf::from("/tmp/prompt-cache"),
            50_000_000_000,
        ),
        bind_address: "127.0.0.1:6733".to_owned(),
        logging_config: LoggingConfig::new(
            PathBuf::from("/tmp/astronomical-logs"),
            LogLevel::Warn,
            7,
        ),
    }
}

fn sample_runtime_model_policy() -> RuntimeModelPolicy {
    RuntimeModelPolicy {
        model_directory: PathBuf::from("/tmp/models/default"),
        generation_defaults: astronomical_supervisor::RuntimeModelGenerationDefaults {
            maximum_output_tokens: 20_480,
            configured_maximum_output_tokens: None,
            temperature_thousandths: None,
            top_p_thousandths: None,
        },
        configured_maximum_context_tokens: None,
        default_maximum_context_tokens: 65_536,
        configured_chunking_fields: Default::default(),
        acceleration_availability: Default::default(),
        worker_model_configuration: WorkerModelConfiguration {
            model_id: "default".to_owned(),
            maximum_context_tokens: 65_536,
            maximum_output_tokens: 20_480,
            chunking: WorkerChunkingConfiguration {
                fixed_prompt_processing_chunk_size_tokens: 2_048,
                fixed_ssd_streaming_prompt_processing_chunk_size_tokens: None,
                full_attention_key_value_growth_tokens: 256,
                speculative_prefill_draft_forward_tokens: 2_048,
                prefill_graph_submission_layer_interval: 1,
                experimental_ssd_paging_generation_graph_submission_layer_interval: 3,
                prompt_cache_block_tokens: None,
                prompt_cache_common_prefix_stride_blocks: 4,
            },
            mtp_draft_depth: None,
            mtp_head_model: None,
            speculative_prefill: None,
        },
    }
}

fn sample_discovered_model(revision: &str) -> astronomical_config::DiscoveredModel {
    astronomical_config::DiscoveredModel {
        model_id: "default".to_owned(),
        model_family: astronomical_config::ModelFamily::Qwen3_5,
        revision: revision.to_owned(),
        model_directory: PathBuf::from("/fictional/models/default"),
        context_window: 65_536,
        max_input_tokens: 65_535,
        max_output_tokens: 20_480,
        has_vision: false,
        supports_reasoning: true,
        supports_tool_calls: true,
        model_size_bytes: 1_000,
    }
}

fn resolved_config_with_bind_address(bind_address: &str) -> ResolvedRuntimeConfig {
    let mut config = sample_resolved_config();
    config.bind_address = bind_address.to_owned();
    config
}

fn resolved_config_with_logging_directory(directory: &str) -> ResolvedRuntimeConfig {
    let mut config = sample_resolved_config();
    config.logging_config = LoggingConfig::new(PathBuf::from(directory), LogLevel::Warn, 7);
    config
}

#[test]
fn should_resolve_reload_config_from_the_config_file() {
    let config_home_directory = tempfile::tempdir()
        .expect("a config home should be created")
        .keep();
    let config_file_path = config_home_directory
        .join(".astronomical-dev")
        .join("config.json");
    std::fs::create_dir_all(
        config_file_path
            .parent()
            .expect("the config path should have a parent"),
    )
    .expect("the config directory should be created");
    std::fs::write(
        &config_file_path,
        r#"{
            "$schema": "./astronomical-config.schema.json",
            "schema_version": 1,
            "runtime": {
                "model_directories": []
            },
            "chunking": {
                "fixed_prompt_processing_chunk_size_tokens": 4096,
                "full_attention_key_value_growth_tokens": 192,
                "speculative_prefill_draft_forward_tokens": 1536,
                "prefill_graph_submission_layer_interval": 1,
                "experimental_ssd_paging_generation_graph_submission_layer_interval": 6,
                "prompt_cache_block_tokens": 768,
                "prompt_cache_common_prefix_stride_blocks": 8
            },
            "prompt_cache": {
                "maximum_size_gb": 1
            }
        }"#,
    )
    .expect("the config file should be written");
    let resolver = ResolvedRuntimeConfigResolver::for_development_home_directory(
        config_home_directory.clone(),
        PathBuf::from("/fallback/worker"),
    );

    let resolved_config = resolver.load().expect("the reload config should resolve");

    assert_eq!(resolved_config.bind_address, "127.0.0.1:6733");
    assert_eq!(
        resolved_config.worker_executable_path,
        PathBuf::from("/fallback/worker")
    );
    assert_eq!(
        resolved_config
            .prompt_cache_config
            .global_prompt_cache_root_directory(),
        &config_home_directory.join(".astronomical-dev/cache")
    );
}

#[test]
fn should_not_resolve_a_draft_model_when_speculative_prefill_is_disabled() {
    let config_home_directory = tempfile::tempdir().expect("a config home should be created");
    let config_file_path = config_home_directory
        .path()
        .join(".astronomical-dev")
        .join("config.json");
    std::fs::create_dir_all(
        config_file_path
            .parent()
            .expect("the config path should have a parent"),
    )
    .expect("the config directory should be created");
    std::fs::write(
        &config_file_path,
        r#"{
            "speculative_prefill": {
                "enabled": false,
                "draft_model_id": "astronomical/unused-draft"
            }
        }"#,
    )
    .expect("the config file should be written");
    let resolver = ResolvedRuntimeConfigResolver::for_development_home_directory(
        config_home_directory.path().to_path_buf(),
        PathBuf::from("/fallback/worker"),
    );

    let resolved_runtime_config = resolver
        .load()
        .expect("a disabled speculative-prefill draft must not require discovery");

    assert!(resolved_runtime_config.model_policy_catalog.is_empty());
}

#[test]
fn should_preserve_speculative_prefill_override_when_target_model_is_not_discovered() {
    let config_home_directory = tempfile::tempdir().expect("a config home should be created");
    let config_file_path = config_home_directory
        .path()
        .join(".astronomical-dev")
        .join("config.json");
    std::fs::create_dir_all(
        config_file_path
            .parent()
            .expect("the config path should have a parent"),
    )
    .expect("the config directory should be created");
    std::fs::write(
        &config_file_path,
        r#"{
            "$schema": "./astronomical-config.schema.json",
            "schema_version": 1,
            "runtime": {"model_directories": []},
            "models": {
                "target-model": {
                    "acceleration": {
                        "speculative_prefill": {
                            "draft_model_id": "draft-model",
                            "keep_percentage": 20
                        }
                    }
                }
            }
        }"#,
    )
    .expect("the config file should be written");
    let resolver = ResolvedRuntimeConfigResolver::for_development_home_directory(
        config_home_directory.path().to_path_buf(),
        PathBuf::from("/fallback/worker"),
    );

    let resolved_config = resolver
        .load()
        .expect("a dormant override must not prevent ordinary serving");
    assert_eq!(
        resolved_config.unmatched_model_config_ids,
        vec!["target-model".to_owned()]
    );
    assert!(resolved_config.model_policy_catalog.is_empty());
}
