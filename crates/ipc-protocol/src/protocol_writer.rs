use bytes::Bytes;
use futures_util::SinkExt;
use tokio::io::AsyncWrite;
use tokio_util::codec::{FramedWrite, LengthDelimitedCodec};

use crate::{MAX_IPC_FRAME_BYTES, ProtocolError, WorkerCommand, WorkerEvent, encode_event};

/// Sends bounded, length-delimited JSON commands to the worker.
pub struct ProtocolWriter<WriteTransport> {
    framed_writer: FramedWrite<WriteTransport, LengthDelimitedCodec>,
}

impl<WriteTransport> ProtocolWriter<WriteTransport>
where
    WriteTransport: AsyncWrite + Unpin,
{
    /// Creates a writer that never emits frames larger than [`MAX_IPC_FRAME_BYTES`].
    #[must_use]
    pub fn new(write_transport: WriteTransport) -> Self {
        let mut frame_codec = LengthDelimitedCodec::new();
        frame_codec.set_max_frame_length(MAX_IPC_FRAME_BYTES);

        Self {
            framed_writer: FramedWrite::new(write_transport, frame_codec),
        }
    }

    /// Serializes and transmits one supervisor command frame.
    pub async fn send_command(
        &mut self,
        worker_command: &WorkerCommand,
    ) -> Result<(), ProtocolError> {
        let serialized_command =
            serde_json::to_vec(worker_command).map_err(ProtocolError::SerializeMessage)?;
        self.send_serialized_message(serialized_command).await
    }

    /// Serializes and transmits one worker event frame.
    pub async fn send_event(&mut self, worker_event: &WorkerEvent) -> Result<(), ProtocolError> {
        let serialized_event = encode_event(worker_event)?;
        self.send_serialized_message(serialized_event).await
    }

    /// Flushes queued frames, then drops the owned write transport to deliver EOF.
    pub async fn close(mut self) -> Result<(), ProtocolError> {
        self.framed_writer
            .close()
            .await
            .map_err(ProtocolError::WriteFrame)
    }

    async fn send_serialized_message(
        &mut self,
        serialized_message: Vec<u8>,
    ) -> Result<(), ProtocolError> {
        if serialized_message.len() > MAX_IPC_FRAME_BYTES {
            return Err(ProtocolError::OutgoingMessageTooLarge {
                actual_message_bytes: serialized_message.len(),
                maximum_message_bytes: MAX_IPC_FRAME_BYTES,
            });
        }
        self.framed_writer
            .send(Bytes::from(serialized_message))
            .await
            .map_err(ProtocolError::WriteFrame)
    }
}
