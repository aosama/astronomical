//! The in-memory IPC pipe that connects a scripted daemon side to the worker
//! under test.
use ipc_protocol::{MAX_IPC_FRAME_BYTES, ProtocolReader, ProtocolWriter};
use tokio::io::{DuplexStream, ReadHalf, WriteHalf, duplex, split};

pub(crate) type DaemonReader = ProtocolReader<ReadHalf<DuplexStream>>;
pub(crate) type DaemonWriter = ProtocolWriter<WriteHalf<DuplexStream>>;

/// Opens one bidirectional pipe and returns the daemon's framed reader and
/// writer plus the two raw halves the worker's `run` consumes. The pipe holds
/// two maximum frames so a worker write never blocks on a daemon that has not
/// started reading yet.
pub(crate) fn open_ipc_worker_transport() -> (
    DaemonReader,
    DaemonWriter,
    ReadHalf<DuplexStream>,
    WriteHalf<DuplexStream>,
) {
    let (daemon_transport, worker_transport) = duplex(MAX_IPC_FRAME_BYTES * 2);
    let (daemon_reader_transport, daemon_writer_transport) = split(daemon_transport);
    let (worker_reader_transport, worker_writer_transport) = split(worker_transport);
    (
        ProtocolReader::new(daemon_reader_transport),
        ProtocolWriter::new(daemon_writer_transport),
        worker_reader_transport,
        worker_writer_transport,
    )
}
