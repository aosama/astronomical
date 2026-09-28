use std::{
    fs::Permissions,
    future::Future,
    io,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
};

use tokio::net::{UnixListener, UnixStream, unix::OwnedWriteHalf};

use crate::{DaemonRequest, DaemonResponse, DaemonTransportError, ProtocolReader, ProtocolWriter};

/// Unix-socket listener serving one ephemeral local request per connection.
///
/// The socket lives beside the instance lock file inside the instance state
/// directory so every runtime instance owns an isolated endpoint without any
/// extra configuration.
pub struct DaemonIpcListener {
    unix_listener: UnixListener,
    socket_path: PathBuf,
}

impl DaemonIpcListener {
    /// Binds the owner-only daemon socket, replacing a stale socket file left
    /// behind by a crashed daemon while refusing to take over a socket a live
    /// daemon still answers on.
    pub async fn bind(socket_path: PathBuf) -> Result<Self, DaemonTransportError> {
        let socket_parent_directory = socket_path.parent().filter(|parent_directory| {
            !parent_directory.as_os_str().is_empty() && parent_directory.exists()
        });
        if socket_parent_directory.is_none() {
            return Err(DaemonTransportError::SocketParentDirectoryMissing { socket_path });
        }
        if socket_path.exists() {
            refuse_live_socket_or_remove_stale_file(&socket_path).await?;
        }
        let unix_listener = UnixListener::bind(&socket_path).map_err(|source| {
            DaemonTransportError::BindFailed {
                socket_path: socket_path.clone(),
                source,
            }
        })?;
        std::fs::set_permissions(&socket_path, Permissions::from_mode(0o600)).map_err(
            |source| DaemonTransportError::SocketPermissionDenied {
                socket_path: socket_path.clone(),
                source,
            },
        )?;
        Ok(Self {
            unix_listener,
            socket_path,
        })
    }

    /// Socket file path this listener serves.
    #[must_use]
    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    /// Accepts one connection, answers one request, then closes the
    /// connection. A peer that disconnects before sending a request is not an
    /// error so the listener stays available for the next local client.
    pub async fn serve_next_request<RequestHandler, ResponseFuture>(
        &mut self,
        handle_request: RequestHandler,
    ) -> Result<(), DaemonTransportError>
    where
        RequestHandler: FnOnce(DaemonRequest) -> ResponseFuture,
        ResponseFuture: Future<Output = DaemonResponse> + Send,
    {
        let (connection_stream, _peer_address) = self
            .unix_listener
            .accept()
            .await
            .map_err(|source| DaemonTransportError::AcceptFailed { source })?;
        let (read_half, write_half) = connection_stream.into_split();
        let mut protocol_reader = ProtocolReader::new(read_half);
        let mut protocol_writer = ProtocolWriter::new(write_half);
        let Some(daemon_request) = protocol_reader.next_daemon_request().await? else {
            return Ok(());
        };
        let daemon_response = handle_request(daemon_request).await;
        protocol_writer
            .send_daemon_response(&daemon_response)
            .await?;
        protocol_writer.close().await?;
        Ok(())
    }

    /// Accepts one connection and hands the request plus the open write half
    /// to the handler, which streams every response frame before the
    /// connection closes. Like [`Self::serve_next_request`], a peer that
    /// disconnects before sending a request is not an error.
    pub async fn serve_streaming_request<StreamingHandler, StreamingHandlerFuture>(
        &mut self,
        handle_request: StreamingHandler,
    ) -> Result<(), DaemonTransportError>
    where
        StreamingHandler: FnOnce(DaemonRequest, StreamingResponseWriter) -> StreamingHandlerFuture,
        StreamingHandlerFuture: Future<Output = Result<(), DaemonTransportError>> + Send,
    {
        let (connection_stream, _peer_address) = self
            .unix_listener
            .accept()
            .await
            .map_err(|source| DaemonTransportError::AcceptFailed { source })?;
        let (read_half, write_half) = connection_stream.into_split();
        let mut protocol_reader = ProtocolReader::new(read_half);
        let protocol_writer = ProtocolWriter::new(write_half);
        let Some(daemon_request) = protocol_reader.next_daemon_request().await? else {
            return Ok(());
        };
        let streaming_response_writer =
            StreamingResponseWriter::from_protocol_writer(protocol_writer);
        handle_request(daemon_request, streaming_response_writer).await
    }
}

/// Owns the write half of one accepted daemon IPC connection so a handler can
/// stream multiple response frames before the connection closes.
pub struct StreamingResponseWriter {
    protocol_writer: ProtocolWriter<OwnedWriteHalf>,
}

impl StreamingResponseWriter {
    fn from_protocol_writer(protocol_writer: ProtocolWriter<OwnedWriteHalf>) -> Self {
        Self { protocol_writer }
    }

    /// Transmits one response frame, keeping the connection open.
    pub async fn send_response(
        &mut self,
        daemon_response: &DaemonResponse,
    ) -> Result<(), DaemonTransportError> {
        self.protocol_writer
            .send_daemon_response(daemon_response)
            .await?;
        Ok(())
    }

    /// Flushes queued frames, then drops the write half to deliver EOF.
    pub async fn close(self) -> Result<(), DaemonTransportError> {
        self.protocol_writer.close().await?;
        Ok(())
    }
}

async fn refuse_live_socket_or_remove_stale_file(
    socket_path: &Path,
) -> Result<(), DaemonTransportError> {
    match UnixStream::connect(socket_path).await {
        // A successful connect proves a live daemon still owns the socket; a
        // rename-based takeover would silently steal its endpoint otherwise.
        Ok(_live_connection) => Err(DaemonTransportError::BindFailed {
            socket_path: socket_path.to_path_buf(),
            source: io::Error::from(io::ErrorKind::AddrInUse),
        }),
        // No daemon answers the probe, so the socket file is stale residue.
        Err(_probe_error) => std::fs::remove_file(socket_path).map_err(|source| {
            DaemonTransportError::SocketCleanupFailed {
                socket_path: socket_path.to_path_buf(),
                source,
            }
        }),
    }
}
