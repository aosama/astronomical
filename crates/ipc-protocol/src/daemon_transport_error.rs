use std::{io, path::PathBuf};

use tokio::task::JoinError;

use crate::ProtocolError;

/// Failures of the local unix-socket transport between the daemon and a CLI process.
#[derive(Debug, thiserror::Error)]
pub enum DaemonTransportError {
    #[error("daemon IPC socket parent directory is missing: {socket_path}")]
    SocketParentDirectoryMissing { socket_path: PathBuf },

    #[error("daemon IPC socket could not be bound at {socket_path}: {source}")]
    BindFailed {
        socket_path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("daemon IPC socket permissions could not be restricted at {socket_path}: {source}")]
    SocketPermissionDenied {
        socket_path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("no daemon is running at the IPC socket {socket_path}")]
    DaemonNotRunning { socket_path: PathBuf },

    #[error("daemon IPC connection failed at {socket_path}: {source}")]
    ConnectFailed {
        socket_path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("daemon IPC listener failed to accept a connection: {source}")]
    AcceptFailed {
        #[source]
        source: io::Error,
    },

    #[error("daemon IPC socket file could not be cleaned up at {socket_path}: {source}")]
    SocketCleanupFailed {
        socket_path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("daemon IPC service task ended unexpectedly: {0}")]
    ServiceTaskFailed(#[from] JoinError),

    #[error("daemon IPC protocol failure: {0}")]
    Protocol(#[from] ProtocolError),
}
