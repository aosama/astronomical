use futures_util::StreamExt;
use tokio::io::AsyncRead;
use tokio_util::codec::{FramedRead, LengthDelimitedCodec};

use crate::{
    DaemonRequest, DaemonResponse, MAX_IPC_FRAME_BYTES, ProtocolError, WorkerCommand, WorkerEvent,
    decode_command, decode_daemon_request, decode_daemon_response, decode_event,
};

/// Receives bounded, length-delimited JSON commands from the supervisor.
pub struct ProtocolReader<ReadTransport> {
    framed_reader: FramedRead<ReadTransport, LengthDelimitedCodec>,
}

impl<ReadTransport> ProtocolReader<ReadTransport>
where
    ReadTransport: AsyncRead + Unpin,
{
    /// Creates a reader that rejects frames larger than [`MAX_IPC_FRAME_BYTES`].
    #[must_use]
    pub fn new(read_transport: ReadTransport) -> Self {
        let mut frame_codec = LengthDelimitedCodec::new();
        frame_codec.set_max_frame_length(MAX_IPC_FRAME_BYTES);

        Self {
            framed_reader: FramedRead::new(read_transport, frame_codec),
        }
    }

    /// Reads the next supervisor command, or `None` when the transport closes cleanly.
    pub async fn next_command(&mut self) -> Result<Option<WorkerCommand>, ProtocolError> {
        let Some(serialized_command) = self.next_frame().await? else {
            return Ok(None);
        };
        decode_command(&serialized_command).map(Some)
    }

    /// Reads the next worker event, or `None` when the transport closes cleanly.
    pub async fn next_event(&mut self) -> Result<Option<WorkerEvent>, ProtocolError> {
        let Some(serialized_event) = self.next_frame().await? else {
            return Ok(None);
        };
        decode_event(&serialized_event).map(Some)
    }

    /// Reads the next daemon request, or `None` when the transport closes cleanly.
    pub async fn next_daemon_request(&mut self) -> Result<Option<DaemonRequest>, ProtocolError> {
        let Some(serialized_request) = self.next_frame().await? else {
            return Ok(None);
        };
        decode_daemon_request(&serialized_request).map(Some)
    }

    /// Reads the next daemon response, or `None` when the transport closes cleanly.
    pub async fn next_daemon_response(&mut self) -> Result<Option<DaemonResponse>, ProtocolError> {
        let Some(serialized_response) = self.next_frame().await? else {
            return Ok(None);
        };
        decode_daemon_response(&serialized_response).map(Some)
    }

    async fn next_frame(&mut self) -> Result<Option<bytes::BytesMut>, ProtocolError> {
        let Some(frame_result) = self.framed_reader.next().await else {
            return Ok(None);
        };

        frame_result.map_err(ProtocolError::ReadFrame).map(Some)
    }
}
