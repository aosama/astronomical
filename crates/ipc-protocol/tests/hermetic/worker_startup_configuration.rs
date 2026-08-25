//! Round-trip contract for the complete worker startup configuration boundary.

use std::path::PathBuf;

use astronomical_ipc_protocol::{
    ProtocolReader, ProtocolWriter, WorkerAutoregressiveModelConfiguration,
    WorkerChunkingConfiguration, WorkerCommand, WorkerFlux2KleinModelConfiguration,
    WorkerImageGenerationModelFamily, WorkerLoadedModelRuntimeConfiguration, WorkerLogLevel,
    WorkerModelConfiguration, WorkerStartupConfiguration,
};
use tokio::io::duplex;

const TEST_TRANSPORT_CAPACITY_BYTES: usize = 256 * 1024;

#[tokio::test]
async fn should_round_trip_worker_startup_configuration() {
    // Exercise every nested startup owner together so a newly added field cannot
    // silently disappear between the supervisor and worker process boundary.
    let worker_command = WorkerCommand::InitializeWorker(WorkerStartupConfiguration {
        configuration_generation: "0123456789abcdef".repeat(4),
        global_prompt_cache_root_directory: PathBuf::from("/tmp/fictional-prompt-cache"),
        global_prompt_cache_maximum_size_bytes: 50_000_000_000,
        persistent_prompt_cache_enabled: false,
        configured_maximum_mlx_memory_bytes: Some(8_000_000_000),
        performance_attribution_enabled: false,
        logging_directory: PathBuf::from("/tmp/fictional-logs"),
        logging_level: WorkerLogLevel::Warn,
        retained_log_file_count: 1,
    });
    let (supervisor_transport, worker_transport) = duplex(TEST_TRANSPORT_CAPACITY_BYTES);
    let mut supervisor_writer = ProtocolWriter::new(supervisor_transport);
    let mut worker_reader = ProtocolReader::new(worker_transport);

    supervisor_writer
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

#[tokio::test]
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
                    fixed_ssd_streaming_prompt_processing_chunk_size_tokens: Some(256),
                    full_attention_key_value_growth_tokens: 256,
                    speculative_prefill_draft_forward_tokens: 2_048,
                    prefill_graph_submission_layer_interval: 1,
                    experimental_ssd_paging_generation_graph_submission_layer_interval: 3,
                    prompt_cache_block_tokens: None,
                    prompt_cache_common_prefix_stride_blocks: 4,
                },
                mtp_enabled: true,
                mtp_draft_depth: Some(2),
                speculative_prefill: None,
            },
        ),
    };
    let (supervisor_transport, worker_transport) = duplex(TEST_TRANSPORT_CAPACITY_BYTES);
    let mut supervisor_writer = ProtocolWriter::new(supervisor_transport);
    let mut worker_reader = ProtocolReader::new(worker_transport);

    supervisor_writer
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
                fixed_ssd_streaming_prompt_processing_chunk_size_tokens: None,
                full_attention_key_value_growth_tokens: 256,
                speculative_prefill_draft_forward_tokens: 2_048,
                prefill_graph_submission_layer_interval: 1,
                experimental_ssd_paging_generation_graph_submission_layer_interval: 3,
                prompt_cache_block_tokens: None,
                prompt_cache_common_prefix_stride_blocks: 4,
            },
            mtp_enabled: true,
            mtp_draft_depth: Some(2),
            speculative_prefill: None,
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
                    "full_attention_key_value_growth_tokens": 256,
                    "speculative_prefill_draft_forward_tokens": 2_048,
                    "prefill_graph_submission_layer_interval": 1,
                    "experimental_ssd_paging_generation_graph_submission_layer_interval": 3,
                    "prompt_cache_block_tokens": null,
                    "prompt_cache_common_prefix_stride_blocks": 4
                },
                "mtp_enabled": true,
                "mtp_draft_depth": 2,
                "speculative_prefill": null
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

#[test]
fn should_acknowledge_the_exact_tagged_flux_runtime_configuration_without_chat_fields() {
    let model_configuration =
        WorkerModelConfiguration::Flux2Klein(WorkerFlux2KleinModelConfiguration {
            model_id: "FLUX.2-klein-4B".to_owned(),
            model_family: WorkerImageGenerationModelFamily::Flux2Klein,
            artifact_revision: "reviewed-revision".to_owned(),
        });

    let runtime_configuration = model_configuration.runtime_configuration();
    assert_eq!(
        runtime_configuration,
        WorkerLoadedModelRuntimeConfiguration::Flux2Klein(WorkerFlux2KleinModelConfiguration {
            model_id: "FLUX.2-klein-4B".to_owned(),
            model_family: WorkerImageGenerationModelFamily::Flux2Klein,
            artifact_revision: "reviewed-revision".to_owned(),
        })
    );
    assert_eq!(
        serde_json::to_value(runtime_configuration).expect("FLUX policy should serialize"),
        serde_json::json!({
            "kind": "flux2_klein",
            "configuration": {
                "model_id": "FLUX.2-klein-4B",
                "model_family": "flux2_klein",
                "artifact_revision": "reviewed-revision"
            }
        })
    );
}
