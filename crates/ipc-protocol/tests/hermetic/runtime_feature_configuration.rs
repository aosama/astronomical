use astronomical_ipc_protocol::{
    ChatModelCapabilities, MtpDepthStatus, MtpRuntimeState, ProtocolReader, ProtocolWriter,
    SpeculativePrefillRuntimeState, WorkerEvent, WorkerRuntimeFeatureConfiguration,
};
use tokio::io::duplex;

const TEST_TRANSPORT_CAPACITY_BYTES: usize = 256 * 1024;

#[tokio::test]
async fn should_round_trip_worker_runtime_feature_configuration_acknowledgement() {
    let worker_event = WorkerEvent::RuntimeFeatureConfigurationApplied {
        worker_runtime_feature_configuration: WorkerRuntimeFeatureConfiguration {
            persistent_prompt_cache_enabled: true,
            mtp_enabled: true,
            mtp_draft_depth: Some(2),
            speculative_prefill_enabled: false,
        },
    };
    let (supervisor_transport, worker_transport) = duplex(TEST_TRANSPORT_CAPACITY_BYTES);
    let mut worker_writer = ProtocolWriter::new(worker_transport);
    let mut supervisor_reader = ProtocolReader::new(supervisor_transport);

    worker_writer
        .send_event(&worker_event)
        .await
        .expect("the runtime feature configuration acknowledgement should be written");

    assert_eq!(
        supervisor_reader
            .next_event()
            .await
            .expect("the runtime feature configuration acknowledgement should decode"),
        Some(worker_event)
    );
}

#[tokio::test]
async fn should_round_trip_loaded_model_mtp_depth_acknowledgement() {
    let worker_event = WorkerEvent::Ready {
        model_id: "fictional/qwen-model".to_owned(),
        capabilities: ChatModelCapabilities {
            supports_reasoning: true,
            supports_tool_calls: true,
            has_vision: false,
            max_input_tokens: 8_000,
            max_output_tokens: 1_000,
            context_window: 9_000,
        },
        mtp_runtime_state: MtpRuntimeState::Active,
        mtp_unavailable_reason: None,
        mtp_depth_status: MtpDepthStatus {
            configured_draft_depth: Some(3),
            artifact_maximum_draft_depth: Some(3),
            artifact_default_draft_depth: Some(2),
            resolved_requested_draft_depth: Some(3),
            effective_execution_draft_depth: Some(1),
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
