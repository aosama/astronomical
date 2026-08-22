//! Pure config-reload diff logic tests.
//!
//! These tests verify the classification of changed config fields into
//! reload, worker-restart, and rest-api-restart categories without
//! starting a worker process or HTTP listener.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use astronomical_config::{
    ChatModelCapabilities, ImageGenerationCapabilities, LogLevel, LoggingConfig, ModelCapabilities,
    PromptCacheConfig,
};
use astronomical_ipc_protocol::{
    WorkerAutoregressiveModelConfiguration, WorkerChunkingConfiguration,
    WorkerFlux2KleinModelConfiguration, WorkerImageGenerationModelFamily, WorkerModelConfiguration,
};
use astronomical_supervisor::{
    ConfigReloadDecision, ConfigReloadDiff, ResolvedConfigurationGeneration, ResolvedRuntimeConfig,
    RuntimeModelPolicy,
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
        provider_model_id: None,
        model_family: astronomical_config::ModelFamily::Qwen3_5,
        revision: "revision-a".to_owned(),
        model_directory: PathBuf::from("/fictional/models/default"),
        capabilities: ModelCapabilities::Chat(ChatModelCapabilities {
            context_window: 65_536,
            max_input_tokens: 65_535,
            max_output_tokens: 20_480,
            supports_vision: false,
            supports_reasoning: true,
            supports_tool_calls: true,
        }),
        license: None,
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
fn should_change_resolved_generation_when_image_capability_changes() {
    let image_policy_catalog = HashMap::from([(
        "FLUX.2-klein-4B".to_owned(),
        image_runtime_model_policy("revision-a"),
    )]);
    let first_model = image_discovered_model("revision-a", false);
    let second_model = image_discovered_model("revision-a", true);

    let first_generation = ResolvedConfigurationGeneration::derive(
        "document-generation",
        &[first_model],
        &image_policy_catalog,
        &[],
    )
    .expect("first image generation should derive");
    let second_generation = ResolvedConfigurationGeneration::derive(
        "document-generation",
        &[second_model],
        &image_policy_catalog,
        &[],
    )
    .expect("second image generation should derive");

    assert_ne!(first_generation, second_generation);
}

#[test]
fn should_change_resolved_generation_when_image_artifact_revision_changes() {
    let first_policy_catalog = HashMap::from([(
        "FLUX.2-klein-4B".to_owned(),
        image_runtime_model_policy("revision-a"),
    )]);
    let second_policy_catalog = HashMap::from([(
        "FLUX.2-klein-4B".to_owned(),
        image_runtime_model_policy("revision-b"),
    )]);

    let first_generation = ResolvedConfigurationGeneration::derive(
        "document-generation",
        &[image_discovered_model("revision-a", false)],
        &first_policy_catalog,
        &[],
    )
    .expect("first image generation should derive");
    let second_generation = ResolvedConfigurationGeneration::derive(
        "document-generation",
        &[image_discovered_model("revision-b", false)],
        &second_policy_catalog,
        &[],
    )
    .expect("second image generation should derive");

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
        .autoregressive_mut()
        .expect("sample policy should be autoregressive")
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
        model_discovery_diagnostics: Vec::new(),
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
        worker_model_configuration: WorkerModelConfiguration::Autoregressive(
            WorkerAutoregressiveModelConfiguration {
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
        ),
    }
}

fn sample_discovered_model(revision: &str) -> astronomical_config::DiscoveredModel {
    astronomical_config::DiscoveredModel {
        model_id: "default".to_owned(),
        provider_model_id: None,
        model_family: astronomical_config::ModelFamily::Qwen3_5,
        revision: revision.to_owned(),
        model_directory: PathBuf::from("/fictional/models/default"),
        capabilities: ModelCapabilities::Chat(ChatModelCapabilities {
            context_window: 65_536,
            max_input_tokens: 65_535,
            max_output_tokens: 20_480,
            supports_vision: false,
            supports_reasoning: true,
            supports_tool_calls: true,
        }),
        license: None,
        model_size_bytes: 1_000,
    }
}

fn image_discovered_model(
    revision: &str,
    supports_image_editing: bool,
) -> astronomical_config::DiscoveredModel {
    astronomical_config::DiscoveredModel {
        model_id: "FLUX.2-klein-4B".to_owned(),
        provider_model_id: Some("black-forest-labs/FLUX.2-klein-4B".to_owned()),
        model_family: astronomical_config::ModelFamily::Flux2Klein,
        revision: revision.to_owned(),
        model_directory: PathBuf::from("/fictional/models/FLUX.2-klein-4B"),
        capabilities: ModelCapabilities::ImageGeneration(ImageGenerationCapabilities {
            supports_text_to_image: true,
            supports_image_editing,
            supports_multiple_reference_images: false,
        }),
        license: Some(astronomical_config::ModelLicense::Apache20),
        model_size_bytes: 4_000,
    }
}

fn image_runtime_model_policy(revision: &str) -> RuntimeModelPolicy {
    RuntimeModelPolicy {
        model_directory: PathBuf::from("/fictional/models/FLUX.2-klein-4B"),
        generation_defaults: astronomical_supervisor::RuntimeModelGenerationDefaults {
            maximum_output_tokens: 0,
            configured_maximum_output_tokens: None,
            temperature_thousandths: None,
            top_p_thousandths: None,
        },
        configured_maximum_context_tokens: None,
        default_maximum_context_tokens: 0,
        configured_chunking_fields: Default::default(),
        acceleration_availability: Default::default(),
        worker_model_configuration: WorkerModelConfiguration::Flux2Klein(
            WorkerFlux2KleinModelConfiguration {
                model_id: "FLUX.2-klein-4B".to_owned(),
                model_family: WorkerImageGenerationModelFamily::Flux2Klein,
                artifact_revision: revision.to_owned(),
            },
        ),
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
