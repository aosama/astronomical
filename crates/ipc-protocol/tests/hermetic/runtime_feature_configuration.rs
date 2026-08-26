use astronomical_ipc_protocol::{
    ChatModelCapabilities, MtpDepthResolutionReason, MtpDepthStatus, MtpRuntimeState,
    ProtocolReader, ProtocolWriter, SpeculativePrefillRuntimeState, WorkerEvent,
    WorkerModelCapabilities, WorkerRuntimeFeatureConfiguration,
};
use tokio::io::duplex;

const TEST_TRANSPORT_CAPACITY_BYTES: usize = 256 * 1024;

#[tokio::test]
async fn should_preserve_generation_and_full_path_free_model_configuration_in_acknowledgement() {
    use astronomical_ipc_protocol::{
        WorkerChunkingConfiguration, WorkerLoadedAutoregressiveModelRuntimeConfiguration,
        WorkerLoadedModelRuntimeConfiguration, WorkerSpeculativePrefillRuntimeConfiguration,
    };

    let configuration_generation = "abcdef0123456789".repeat(4);
    let worker_event = WorkerEvent::RuntimeFeatureConfigurationApplied {
        worker_runtime_feature_configuration: WorkerRuntimeFeatureConfiguration {
            configuration_generation: configuration_generation.clone(),
            persistent_prompt_cache_enabled: true,
            prompt_cache_maximum_size_bytes: 12_000_000_000,
            loaded_model: Some(WorkerLoadedModelRuntimeConfiguration::Autoregressive(
                WorkerLoadedAutoregressiveModelRuntimeConfiguration {
                    model_id: "fictional/target".to_owned(),
                    maximum_context_tokens: 32_768,
                    maximum_output_tokens: 4_096,
                    chunking: WorkerChunkingConfiguration {
                        fixed_prompt_processing_chunk_size_tokens: 2_048,
                        fixed_ssd_streaming_prompt_processing_chunk_size_tokens: 256,
                        full_attention_key_value_growth_tokens: 256,
                        speculative_prefill_draft_forward_tokens: 1_024,
                        prefill_graph_submission_layer_interval: 0,
                        experimental_ssd_paging_prefill_graph_submission_layer_interval: 1,
                        experimental_ssd_paging_generation_graph_submission_layer_interval: 0,
                        prompt_cache_block_tokens: Some(128),
                        prompt_cache_common_prefix_stride_blocks: 4,
                    },
                    mtp_enabled: true,
                    mtp_draft_depth: Some(3),
                    speculative_prefill_enabled: true,
                    speculative_prefill: Some(WorkerSpeculativePrefillRuntimeConfiguration {
                        draft_model_id: "fictional/draft".to_owned(),
                        minimum_prompt_tokens: 8_192,
                        keep_percentage: 20,
                    }),
                },
            )),
        },
    };
    let (supervisor_transport, worker_transport) = duplex(TEST_TRANSPORT_CAPACITY_BYTES);
    let mut worker_writer = ProtocolWriter::new(worker_transport);
    let mut supervisor_reader = ProtocolReader::new(supervisor_transport);

    worker_writer
        .send_event(&worker_event)
        .await
        .expect("acknowledgement should write");
    let decoded_event = supervisor_reader
        .next_event()
        .await
        .expect("acknowledgement should decode")
        .expect("acknowledgement should be present");

    assert_eq!(decoded_event, worker_event);
    let serialized_event = serde_json::to_string(&decoded_event).expect("event should serialize");
    assert!(serialized_event.contains(&configuration_generation));
    assert!(!serialized_event.contains("/tmp/"));
}

#[tokio::test]
async fn should_round_trip_loaded_model_mtp_depth_acknowledgement() {
    let worker_event = WorkerEvent::Ready {
        model_id: "fictional/qwen-model".to_owned(),
        capabilities: WorkerModelCapabilities::from(ChatModelCapabilities {
            supports_reasoning: true,
            supports_tool_calls: true,
            has_vision: false,
            max_input_tokens: 8_000,
            max_output_tokens: 1_000,
            context_window: 9_000,
        }),
        mtp_runtime_state: MtpRuntimeState::Active,
        mtp_unavailable_reason: None,
        mtp_depth_status: MtpDepthStatus {
            configured_draft_depth: Some(3),
            artifact_maximum_draft_depth: Some(3),
            artifact_default_draft_depth: Some(2),
            resolved_requested_draft_depth: Some(3),
            capped_draft_depth: Some(1),
            effective_execution_draft_depth: Some(1),
            resolution_reason: Some(
                MtpDepthResolutionReason::ConfiguredDepthClampedToArtifactMaximum,
            ),
        },
        speculative_prefill_runtime_state: SpeculativePrefillRuntimeState::Disabled,
        speculative_prefill_unavailable_reason: None,
        speculative_prefill_draft_model_id: None,
        speculative_prefill_draft_model_revision: None,
    };
    let (supervisor_transport, worker_transport) = duplex(TEST_TRANSPORT_CAPACITY_BYTES);
    let mut worker_writer = ProtocolWriter::new(worker_transport);
    let mut supervisor_reader = ProtocolReader::new(supervisor_transport);

    worker_writer
        .send_event(&worker_event)
        .await
        .expect("the loaded-model MTP depth acknowledgement should be written");

    assert_eq!(
        supervisor_reader
            .next_event()
            .await
            .expect("the loaded-model MTP depth acknowledgement should decode"),
        Some(worker_event)
    );
}
