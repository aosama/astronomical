//! Model-swap identity contract across the private supervisor/worker boundary.

use astronomical_ipc_protocol::{ProtocolReader, ProtocolWriter, WorkerCommand};
use tokio::io::duplex;

#[tokio::test]
async fn should_round_trip_distinct_requested_model_identity_and_directory() {
    let worker_command = WorkerCommand::SwapModel {
        model_id: "requestable-model-id".to_owned(),
        model_directory: "/tmp/fictional-model-snapshot".to_owned(),
        max_output_tokens: 512,
    };
    let (supervisor_transport, worker_transport) = duplex(64 * 1024);
    let mut supervisor_writer = ProtocolWriter::new(supervisor_transport);
    let mut worker_reader = ProtocolReader::new(worker_transport);

    supervisor_writer
        .send_command(&worker_command)
        .await
        .expect("model swap command should be written");

    assert_eq!(
        worker_reader
            .next_command()
            .await
            .expect("model swap command should decode"),
        Some(worker_command)
    );
}
