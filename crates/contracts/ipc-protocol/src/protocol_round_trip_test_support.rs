use crate::{ProtocolReader, ProtocolWriter, WorkerCommand, WorkerEvent};
use tokio::io::duplex;

// Large enough that no single protocol frame under test can block the writer
// while the reader has not yet started draining the transport.
const TEST_TRANSPORT_CAPACITY_BYTES: usize = 256 * 1024;

/// Sends one daemon-to-worker command through a framed in-memory transport and
/// asserts that the worker side decodes an identical command.
pub(crate) async fn assert_command_round_trips(
    worker_command: &WorkerCommand,
    command_description: &str,
) {
    let (daemon_transport, worker_transport) = duplex(TEST_TRANSPORT_CAPACITY_BYTES);
    let mut daemon_writer = ProtocolWriter::new(daemon_transport);
    let mut worker_reader = ProtocolReader::new(worker_transport);

    daemon_writer
        .send_command(worker_command)
        .await
        .unwrap_or_else(|error| panic!("{command_description} should be written: {error:?}"));
    let decoded_command = worker_reader
        .next_command()
        .await
        .unwrap_or_else(|error| panic!("{command_description} should decode: {error:?}"));

    assert_eq!(decoded_command.as_ref(), Some(worker_command));
}

/// Sends one worker-to-daemon event through a framed in-memory transport and
/// asserts that the daemon side decodes an identical event.
pub(crate) async fn assert_event_round_trips(worker_event: &WorkerEvent, event_description: &str) {
    let (daemon_transport, worker_transport) = duplex(TEST_TRANSPORT_CAPACITY_BYTES);
    let mut worker_writer = ProtocolWriter::new(worker_transport);
    let mut daemon_reader = ProtocolReader::new(daemon_transport);

    worker_writer
        .send_event(worker_event)
        .await
        .unwrap_or_else(|error| panic!("{event_description} should be written: {error:?}"));
    let decoded_event = daemon_reader
        .next_event()
        .await
        .unwrap_or_else(|error| panic!("{event_description} should decode: {error:?}"));

    assert_eq!(decoded_event.as_ref(), Some(worker_event));
}
