use astronomical_ipc_protocol::{
    MAX_IPC_FRAME_BYTES, ProtocolReader, ProtocolWriter, RequestId, WorkerCommand,
};
use tokio::io::duplex;

#[tokio::test]
async fn should_round_trip_a_generation_cancellation_command() {
    let cancellation_command = WorkerCommand::Cancel {
        request_id: RequestId::new(11),
    };
    let (supervisor_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
    let mut supervisor_writer = ProtocolWriter::new(supervisor_transport);
    let mut worker_reader = ProtocolReader::new(worker_transport);

    supervisor_writer
        .send_command(&cancellation_command)
        .await
        .expect("a bounded cancellation command should be written");

    let received_command = worker_reader
        .next_command()
        .await
        .expect("the cancellation command frame should decode")
        .expect("the transport should contain a command before it closes");

    assert_eq!(received_command, cancellation_command);
}
