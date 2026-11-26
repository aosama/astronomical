//! Round-trip contract for the complete worker startup configuration boundary.

use std::path::PathBuf;

use crate::{
    ProtocolReader, ProtocolWriter, WorkerAutoregressiveModelConfiguration,
    WorkerChunkingConfiguration, WorkerCommand, WorkerDecodeAstroExpertCachePolicy,
    WorkerImageGenerationModelConfiguration, WorkerLoadedModelRuntimeConfiguration, WorkerLogLevel,
    WorkerModelConfiguration, WorkerStartupConfiguration,
};
use tokio::io::duplex;

const TEST_TRANSPORT_CAPACITY_BYTES: usize = 256 * 1024;

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn should_round_trip_worker_startup_configuration() {
    // Exercise every nested startup owner together so a newly added field cannot
    // silently disappear between the daemon and worker process boundary.
    let worker_command = WorkerCommand::InitializeWorker(WorkerStartupConfiguration {
        neural_engine_program_directory: "neural-engine-programs".into(),
        configuration_generation: "0123456789abcdef".repeat(4),
        global_astro_ssd_prompt_cache_root_directory: PathBuf::from(
            "/tmp/fictional-astro-ssd-prompt-cache",
        ),
        global_astro_ssd_prompt_cache_maximum_size_bytes: 50_000_000_000,
        astro_ssd_prompt_cache_enabled: false,
        configured_maximum_mlx_memory_bytes: Some(8_000_000_000),
        performance_attribution_enabled: false,
        decode_astro_expert_cache_policy: WorkerDecodeAstroExpertCachePolicy::Baseline,
        diagnostic_qwen3_5_engine_override: crate::WorkerDiagnosticQwen3_5EngineOverride::Automatic,
        logging_directory: PathBuf::from("/tmp/fictional-logs"),
        logging_level: WorkerLogLevel::Warn,
        retained_log_file_count: 1,
        metal_residency_refresh_enabled: false,
    });
    let (daemon_transport, worker_transport) = duplex(TEST_TRANSPORT_CAPACITY_BYTES);
    let mut daemon_writer = ProtocolWriter::new(daemon_transport);
    let mut worker_reader = ProtocolReader::new(worker_transport);

    daemon_writer
        .send_command(&worker_command)
        .await
        .expect("worker startup configuration should be written");

    assert_eq!(
        worker_reader
            .next_command()
            .await
            .expect("worker startup configuration should decode"),
        Some(worker_command)
    );
}

#[test]
fn should_serialize_the_candidate_decode_astro_expert_cache_policy() {
    assert_eq!(
        serde_json::to_value(WorkerDecodeAstroExpertCachePolicy::Candidate)
            .expect("candidate policy should serialize"),
        serde_json::json!("candidate")
    );
    assert_eq!(
        serde_json::from_value::<WorkerDecodeAstroExpertCachePolicy>(serde_json::json!(
            "candidate"
        ))
        .expect("candidate policy should deserialize"),
        WorkerDecodeAstroExpertCachePolicy::Candidate
    );
}

#[tokio::test(flavor = "current_thread", start_paused = true)]
async fn should_round_trip_selected_model_policy_on_swap_model() {
    let worker_command = WorkerCommand::SwapModel {
        model_directory: "/tmp/fictional-target".to_owned(),
        model_configuration: WorkerModelConfiguration::Autoregressive(
            WorkerAutoregressiveModelConfiguration {
                model_id: "organization/target".to_owned(),
                maximum_context_tokens: 32_768,
                maximum_output_tokens: 4_096,
                chunking: WorkerChunkingConfiguration {
                    fixed_prompt_processing_chunk_size_tokens: 4_096,
                    fixed_ssd_streaming_prompt_processing_chunk_size_tokens: 256,
                    full_attention_key_value_growth_tokens: 256,
                    prefill_graph_submission_layer_interval: 0,
                    experimental_ssd_paging_generation_graph_submission_layer_interval: 3,
                    ssd_prompt_cache_block_tokens: None,
                    ssd_prompt_cache_common_prefix_stride_blocks: 4,
                    experimental_in_memory_prefix_adoption_enabled: false,
                    experimental_neural_engine_prefill_enabled: false,
                    multi_token_prediction_draft_tokens: None,
                },
            },
        ),
    };
    let (daemon_transport, worker_transport) = duplex(TEST_TRANSPORT_CAPACITY_BYTES);
    let mut daemon_writer = ProtocolWriter::new(daemon_transport);
    let mut worker_reader = ProtocolReader::new(worker_transport);

    daemon_writer
        .send_command(&worker_command)
        .await
        .expect("swap policy should be written");

    assert_eq!(
        worker_reader
            .next_command()
            .await
            .expect("swap policy should decode"),
        Some(worker_command)
    );
}

#[test]
fn should_serialize_autoregressive_configuration_with_an_explicit_discriminator() {
    let model_configuration =
        WorkerModelConfiguration::Autoregressive(WorkerAutoregressiveModelConfiguration {
            model_id: "organization/target".to_owned(),
            maximum_context_tokens: 32_768,
            maximum_output_tokens: 4_096,
            chunking: WorkerChunkingConfiguration {
                fixed_prompt_processing_chunk_size_tokens: 4_096,
                fixed_ssd_streaming_prompt_processing_chunk_size_tokens: 2_048,
                full_attention_key_value_growth_tokens: 256,
                prefill_graph_submission_layer_interval: 0,
                experimental_ssd_paging_generation_graph_submission_layer_interval: 3,
                ssd_prompt_cache_block_tokens: None,
                ssd_prompt_cache_common_prefix_stride_blocks: 4,
                experimental_in_memory_prefix_adoption_enabled: false,
                experimental_neural_engine_prefill_enabled: false,
                multi_token_prediction_draft_tokens: None,
            },
        });

    let serialized_configuration =
        serde_json::to_value(model_configuration).expect("chat policy should serialize");

    assert_eq!(
        serialized_configuration,
        serde_json::json!({
            "kind": "autoregressive",
            "configuration": {
                "model_id": "organization/target",
                "maximum_context_tokens": 32_768,
                "maximum_output_tokens": 4_096,
                "chunking": {
                    "fixed_prompt_processing_chunk_size_tokens": 4_096,
                    "fixed_ssd_streaming_prompt_processing_chunk_size_tokens": 2_048,
                    "full_attention_key_value_growth_tokens": 256,
                    "prefill_graph_submission_layer_interval": 0,
                    "experimental_ssd_paging_generation_graph_submission_layer_interval": 3,
                    "ssd_prompt_cache_block_tokens": null,
                    "ssd_prompt_cache_common_prefix_stride_blocks": 4,
                    "experimental_in_memory_prefix_adoption_enabled": false,
                    "experimental_neural_engine_prefill_enabled": false,
                    "multi_token_prediction_draft_tokens": null
                }
            }
        })
    );
}

#[test]
fn should_reject_the_retired_untagged_model_configuration_wire_shape() {
    let retired_configuration = serde_json::json!({
        "model_id": "FLUX.2-klein-4B",
        "model_family": "flux2_klein",
        "artifact_revision": "reviewed-revision"
    });

    assert!(serde_json::from_value::<WorkerModelConfiguration>(retired_configuration).is_err());
    assert!(
        serde_json::from_value::<WorkerLoadedModelRuntimeConfiguration>(serde_json::json!({
            "model_id": "FLUX.2-klein-4B",
            "model_family": "flux2_klein",
            "artifact_revision": "reviewed-revision"
        }))
        .is_err()
    );
}

fn image_model_configuration(model_family: &str) -> WorkerImageGenerationModelConfiguration {
    WorkerImageGenerationModelConfiguration {
        model_id: format!("{model_family}-model"),
        model_family: model_family.to_owned(),
        artifact_revision: "reviewed-revision".to_owned(),
    }
}

#[test]
fn should_acknowledge_each_image_family_as_one_tagged_image_configuration() {
    for model_family in ["flux2_klein", "qwen_image_21"] {
        let model_configuration =
            WorkerModelConfiguration::ImageGeneration(image_model_configuration(model_family));

        let runtime_configuration = model_configuration.runtime_configuration();

        assert_eq!(
            runtime_configuration,
            WorkerLoadedModelRuntimeConfiguration::ImageGeneration(image_model_configuration(
                model_family
            ))
        );
        assert_eq!(
            serde_json::to_value(runtime_configuration).expect("image policy should serialize"),
            serde_json::json!({
                "kind": "image_generation",
                "configuration": {
                    "model_id": format!("{model_family}-model"),
                    "model_family": model_family,
                    "artifact_revision": "reviewed-revision"
                }
            })
        );
    }
}

#[test]
fn should_reject_the_retired_per_family_image_kinds() {
    for retired_kind in ["flux2_klein", "qwen_image21"] {
        let retired_configuration = serde_json::json!({
            "kind": retired_kind,
            "configuration": {
                "model_id": "image-model",
                "model_family": "flux2_klein",
                "artifact_revision": "reviewed-revision"
            }
        });

        assert!(
            serde_json::from_value::<WorkerModelConfiguration>(retired_configuration).is_err(),
            "{retired_kind}"
        );
    }
}
