//! The IPC chat admission gate: which model ID a request is dispatched under (#1898).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use astronomical_config::{LogLevel, LoggingConfig, PromptCacheConfig};
use astronomical_ipc_protocol::{
    WorkerEmbeddingModelConfiguration, WorkerEmbeddingModelFamily, WorkerModelConfiguration,
};

use super::{admitted_catalog_model_id, catalog_model_id_or_requested};
use crate::worker::health::{WorkerHealthSnapshot, WorkerHealthStatus};
use crate::{ResolvedRuntimeConfig, RuntimeModelGenerationDefaults, RuntimeModelPolicy};

const INSTALLED_LEAF_MODEL_ID: &str = "Ornith-1.5-35B-A3B-OptiQ-4bit";
const FULL_HUGGINGFACE_MODEL_ID: &str = "mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit";

#[test]
fn should_dispatch_a_full_huggingface_id_under_the_installed_leaf_key() {
    let live_config = Some(config_with_installed_model());

    let dispatched_model_id = admitted_catalog_model_id(
        &WorkerHealthSnapshot::ready_without_model(1),
        &live_config,
        FULL_HUGGINGFACE_MODEL_ID,
    );

    assert_eq!(dispatched_model_id, Ok(INSTALLED_LEAF_MODEL_ID.to_owned()));
}

#[test]
fn should_dispatch_the_leaf_id_unchanged() {
    let live_config = Some(config_with_installed_model());

    let dispatched_model_id = admitted_catalog_model_id(
        &WorkerHealthSnapshot::ready_without_model(1),
        &live_config,
        INSTALLED_LEAF_MODEL_ID,
    );

    assert_eq!(dispatched_model_id, Ok(INSTALLED_LEAF_MODEL_ID.to_owned()));
}

#[test]
fn should_reject_an_unknown_model_and_suggest_a_near_match() {
    let live_config = Some(config_with_installed_model());

    let rejection_reason = admitted_catalog_model_id(
        &WorkerHealthSnapshot::ready_without_model(1),
        &live_config,
        "ornith-1.5-35b",
    )
    .expect_err("a partial name is not an installed model");

    assert!(rejection_reason.contains("ornith-1.5-35b"));
    assert!(
        rejection_reason
            .to_ascii_lowercase()
            .contains(&INSTALLED_LEAF_MODEL_ID.to_ascii_lowercase())
    );
}

#[test]
fn should_reject_an_empty_model_id() {
    let live_config = Some(config_with_installed_model());

    let empty_id_admission = admitted_catalog_model_id(
        &WorkerHealthSnapshot::ready_without_model(1),
        &live_config,
        "",
    );

    assert!(empty_id_admission.is_err());
}

#[test]
fn should_reject_every_request_while_the_worker_is_unavailable() {
    let live_config = Some(config_with_installed_model());

    let rejection_reason = admitted_catalog_model_id(
        &WorkerHealthSnapshot::unavailable(WorkerHealthStatus::Unavailable),
        &live_config,
        INSTALLED_LEAF_MODEL_ID,
    )
    .expect_err("an unavailable worker serves nothing");

    assert!(rejection_reason.contains("not ready"));
}

#[test]
fn should_pass_the_requested_id_through_when_no_catalog_is_wired() {
    let dispatched_model_id = admitted_catalog_model_id(
        &WorkerHealthSnapshot::ready_without_model(1),
        &None,
        FULL_HUGGINGFACE_MODEL_ID,
    );

    assert_eq!(
        dispatched_model_id,
        Ok(FULL_HUGGINGFACE_MODEL_ID.to_owned())
    );
}

#[test]
fn should_resolve_a_full_huggingface_embedding_request_to_the_leaf_key() {
    let live_config = Some(config_with_installed_model());

    assert_eq!(
        catalog_model_id_or_requested(&live_config, FULL_HUGGINGFACE_MODEL_ID),
        INSTALLED_LEAF_MODEL_ID
    );
    assert_eq!(
        catalog_model_id_or_requested(&live_config, "someorg/not-installed"),
        "someorg/not-installed"
    );
    assert_eq!(
        catalog_model_id_or_requested(&None, FULL_HUGGINGFACE_MODEL_ID),
        FULL_HUGGINGFACE_MODEL_ID
    );
}

fn config_with_installed_model() -> Arc<RwLock<ResolvedRuntimeConfig>> {
    let installed_model_policy = RuntimeModelPolicy {
        model_directory: PathBuf::from("/fictional/models/installed"),
        generation_defaults: RuntimeModelGenerationDefaults {
            maximum_output_tokens: 20_480,
            configured_maximum_output_tokens: None,
            temperature_thousandths: None,
            top_p_thousandths: None,
            repetition_penalty_thousandths: None,
        },
        configured_maximum_context_tokens: None,
        default_maximum_context_tokens: 65_536,
        configured_chunking_fields: Default::default(),
        worker_model_configuration: WorkerModelConfiguration::Embeddings(
            WorkerEmbeddingModelConfiguration {
                model_id: INSTALLED_LEAF_MODEL_ID.to_owned(),
                model_family: WorkerEmbeddingModelFamily::ModernBert,
                artifact_revision: "test-revision".to_owned(),
                vector_width: 1,
                maximum_input_tokens: 512,
            },
        ),
    };
    Arc::new(RwLock::new(ResolvedRuntimeConfig {
        neural_engine_program_directory: "neural-engine-programs".into(),
        configuration_generation: "751".repeat(8),
        worker_executable_path: PathBuf::from("/fictional/worker"),
        discovered_models: Vec::new(),
        model_discovery_diagnostics: Vec::new(),
        configured_model_directories: Vec::new(),
        model_policy_catalog: Arc::new(HashMap::from([(
            INSTALLED_LEAF_MODEL_ID.to_owned(),
            installed_model_policy,
        )])),
        unmatched_model_config_ids: Vec::new(),
        maximum_mlx_memory_bytes: None,
        persistent_prompt_cache_enabled: false,
        configured_persistent_prompt_cache_enabled: None,
        metal_residency_refresh_enabled: false,
        configured_prompt_cache_maximum_size_bytes: None,
        performance_attribution_enabled: false,
        prompt_cache_config: PromptCacheConfig::new(PathBuf::from("/fictional/prompt-cache"), 1),
        bind_address: "127.0.0.1:6733".to_owned(),
        logging_config: LoggingConfig::new(PathBuf::from("/fictional/logs"), LogLevel::Warn, 7),
    }))
}
