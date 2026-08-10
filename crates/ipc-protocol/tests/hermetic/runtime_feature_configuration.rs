use astronomical_ipc_protocol::{
    ProtocolReader, ProtocolWriter, WorkerEvent, WorkerRuntimeFeatureConfiguration,
};
use tokio::io::duplex;

const TEST_TRANSPORT_CAPACITY_BYTES: usize = 256 * 1024;

#[tokio::test]
async fn should_round_trip_worker_runtime_feature_configuration_acknowledgement() {
    let worker_event = WorkerEvent::RuntimeFeatureConfigurationApplied {
        worker_runtime_feature_configuration: WorkerRuntimeFeatureConfiguration {
            persistent_prompt_cache_enabled: true,
            mtp_enabled: true,
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
