use std::{io::ErrorKind, path::PathBuf};

use tokio::net::{UnixStream, unix::OwnedReadHalf, unix::OwnedWriteHalf};

use crate::{DaemonRequest, DaemonResponse, DaemonTransportError, ProtocolReader, ProtocolWriter};

/// Client one ephemeral CLI process uses to talk to the local daemon.
pub struct DaemonIpcClient {
    protocol_reader: ProtocolReader<OwnedReadHalf>,
    protocol_writer: ProtocolWriter<OwnedWriteHalf>,
}

impl DaemonIpcClient {
    /// Connects to the daemon socket, reporting a missing or dead daemon as
    /// [`DaemonTransportError::DaemonNotRunning`] instead of a raw IO failure.
    pub async fn connect(socket_path: PathBuf) -> Result<Self, DaemonTransportError> {
        let connection_stream = match UnixStream::connect(&socket_path).await {
            Ok(connection_stream) => connection_stream,
            Err(connection_error)
                if matches!(
                    connection_error.kind(),
                    ErrorKind::NotFound | ErrorKind::ConnectionRefused
                ) =>
            {
                return Err(DaemonTransportError::DaemonNotRunning { socket_path });
            }
            Err(connection_error) => {
                return Err(DaemonTransportError::ConnectFailed {
                    socket_path,
                    source: connection_error,
                });
            }
        };
        let (read_half, write_half) = connection_stream.into_split();
        Ok(Self {
            protocol_reader: ProtocolReader::new(read_half),
            protocol_writer: ProtocolWriter::new(write_half),
        })
    }

    /// Transmits one request to the daemon.
    pub async fn send_request(
        &mut self,
        daemon_request: &DaemonRequest,
    ) -> Result<(), DaemonTransportError> {
        self.protocol_writer
            .send_daemon_request(daemon_request)
            .await?;
        Ok(())
    }

    /// Reads the next response, or `None` when the daemon closes the connection.
    pub async fn next_response(&mut self) -> Result<Option<DaemonResponse>, DaemonTransportError> {
        let daemon_response = self.protocol_reader.next_daemon_response().await?;
        Ok(daemon_response)
    }
}
