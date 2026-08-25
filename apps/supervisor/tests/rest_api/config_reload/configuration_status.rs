use std::collections::HashMap;

use astronomical_ipc_protocol::{
    MtpDepthStatus, WorkerAutoregressiveModelConfiguration, WorkerChunkingConfiguration,
    WorkerModelConfiguration, WorkerRuntimeFeatureConfiguration,
};
use astronomical_supervisor::{RuntimeModelGenerationDefaults, RuntimeModelPolicy};

use super::*;

#[tokio::test]
async fn should_expose_configured_and_worker_effective_generation_with_path_free_model_summary() {
    let configured_generation = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    let effective_generation = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    let mut resolved_config = sample_resolved_config();
    resolved_config.configuration_generation = configured_generation.to_owned();
    resolved_config.unmatched_model_config_ids = vec!["fictional/dormant-model".to_owned()];
    resolved_config.model_discovery_diagnostics =
        vec![astronomical_config::ModelDiscoveryDiagnostic {
            model_id: "ambiguous-model".to_owned(),
            configured_root_numbers: vec![1, 3],
        }];
    let configured_worker_model_configuration = worker_model_configuration(false);
    let effective_worker_model_configuration = worker_model_configuration(true);
    resolved_config.model_policy_catalog = Arc::new(HashMap::from([(
        crate::common::MODEL_ID.to_owned(),
        RuntimeModelPolicy {
            model_directory: PathBuf::from("/fictional/private/target"),
            generation_defaults: RuntimeModelGenerationDefaults {
                maximum_output_tokens: 1_024,
                configured_maximum_output_tokens: Some(1_024),
                temperature_thousandths: Some(700),
                top_p_thousandths: Some(900),
            },
            configured_maximum_context_tokens: Some(16_384),
            default_maximum_context_tokens: 32_768,
            configured_chunking_fields: astronomical_config::ConfiguredChunkingFields {
                fixed_prompt_processing_chunk_size_tokens: true,
                ..Default::default()
            },
            acceleration_availability:
                astronomical_supervisor::RuntimeModelAccelerationAvailability {
                    configured_speculative_prefill: Some(
                        astronomical_supervisor::ConfiguredSpeculativePrefillPolicy {
                            draft_model_id: "fictional/missing-drafter".to_owned(),
                            keep_percentage: 50,
                            minimum_prompt_tokens: 1_024,
                        },
                    ),
                    speculative_prefill_unavailable_reason: Some(
                        "configured drafter is not currently available".to_owned(),
                    ),
                    configured_mtp_enabled: Some(false),
                    ..Default::default()
                },
            worker_model_configuration: configured_worker_model_configuration,
        },
    )]));
    let mut executor = ScriptedExecutor::ready(Vec::new());
    executor
        .health_snapshot
        .worker_runtime_feature_configuration = Some(WorkerRuntimeFeatureConfiguration {
        configuration_generation: effective_generation.to_owned(),
        persistent_prompt_cache_enabled: true,
        prompt_cache_maximum_size_bytes: 50_000_000_000,
        loaded_model: Some(effective_worker_model_configuration.runtime_configuration()),
    });
    executor.health_snapshot.mtp_depth_status = MtpDepthStatus {
        configured_draft_depth: Some(3),
        artifact_maximum_draft_depth: Some(3),
        artifact_default_draft_depth: Some(2),
        resolved_requested_draft_depth: Some(3),
        capped_draft_depth: Some(1),
        effective_execution_draft_depth: Some(1),
        resolution_reason: None,
    };
    let temporary_home = tempfile::tempdir().expect("status config home should exist");
    let application = build_development_application_with_reload(
        executor,
        Arc::new(RwLock::new(resolved_config)),
        temporary_home.path().to_path_buf(),
    );

    let status_response = application
        .oneshot(
            Request::builder()
                .uri("/v1/status")
                .body(Body::empty())
                .expect("status request should be valid"),
        )
        .await
        .expect("status response should be returned");
    let status_bytes = to_bytes(status_response.into_body(), 64 * 1024)
        .await
        .expect("status response should be readable");
    let status_document: serde_json::Value =
        serde_json::from_slice(&status_bytes).expect("status should contain JSON");

    assert_eq!(
        status_document["configured_generation"],
        configured_generation
    );
    assert_eq!(
        status_document["effective_generation"],
        effective_generation
    );
    assert_eq!(status_document["configuration"]["restart_required"], true);
    assert_eq!(
        status_document["configuration"]["unmatched_model_config_ids"],
        serde_json::json!(["fictional/dormant-model"])
    );
    assert_eq!(
        status_document["configuration"]["model_discovery_diagnostics"],
        serde_json::json!([{
            "code": "ambiguous_model_identity",
            "model_id": "ambiguous-model",
            "configured_root_numbers": [1, 3]
        }])
    );
    assert!(!String::from_utf8_lossy(&status_bytes).contains("/fictional/private"));
    assert_eq!(
        status_document["configuration"]["ready_model"]["mtp_enabled"]["default"],
        true
    );
    assert_eq!(
        status_document["configuration"]["ready_model"]["mtp_enabled"]["configured"],
        false
    );
    assert_eq!(
        status_document["configuration"]["ready_model"]["mtp_enabled"]["effective"],
        true
    );
    assert_eq!(
        status_document["configuration"]["ready_model"]["mtp_draft_depth"]["effective"],
        1
    );
    assert_eq!(
        status_document["configuration"]["ready_model"]["temperature"]["configured"],
        0.7
    );
    assert_eq!(
        status_document["configured_speculative_prefill_enabled"],
        true
    );
    assert_eq!(status_document["speculative_prefill_enabled"], true);
    assert_eq!(
        status_document["speculative_prefill_runtime_state"],
        "unavailable"
    );
    assert_eq!(
        status_document["speculative_prefill_unavailable_reason"],
        "configured drafter is not currently available"
    );
    assert_eq!(
        status_document["speculative_prefill_draft_model_id"],
        "fictional/missing-drafter"
    );
    assert_eq!(
        status_document["speculative_prefill_target_model_id"],
        crate::common::MODEL_ID
    );
    assert_eq!(
        status_document["configuration"]["ready_model"]["chunking"]["fixed_prompt_processing_chunk_size_tokens"]
            ["configured"],
        2_048
    );
    assert!(
        status_document["configuration"]["ready_model"]["chunking"]
            ["full_attention_key_value_growth_tokens"]["configured"]
            .is_null()
    );
    assert_eq!(
        status_document["configuration"]["ready_model"]["chunking"]["prompt_cache_block_tokens"]["is_configured"],
        false
    );
    let status_text = String::from_utf8(status_bytes.to_vec()).expect("status should be UTF-8");
    assert!(!status_text.contains("/fictional/private"));
    assert!(!status_text.contains("prompt-cache"));
}

fn worker_model_configuration(mtp_enabled: bool) -> WorkerModelConfiguration {
    WorkerModelConfiguration::Autoregressive(WorkerAutoregressiveModelConfiguration {
        model_id: crate::common::MODEL_ID.to_owned(),
        maximum_context_tokens: 16_384,
        maximum_output_tokens: 4_096,
        chunking: WorkerChunkingConfiguration {
            fixed_prompt_processing_chunk_size_tokens: 2_048,
            fixed_ssd_streaming_prompt_processing_chunk_size_tokens: Some(256),
            full_attention_key_value_growth_tokens: 256,
            speculative_prefill_draft_forward_tokens: 1_024,
            prefill_graph_submission_layer_interval: 1,
            experimental_ssd_paging_generation_graph_submission_layer_interval: 0,
            prompt_cache_block_tokens: Some(128),
            prompt_cache_common_prefix_stride_blocks: 4,
        },
        mtp_enabled,
        mtp_draft_depth: Some(3),
        speculative_prefill: None,
    })
}
