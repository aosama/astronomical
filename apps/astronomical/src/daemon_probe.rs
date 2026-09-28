//! Shared daemon reachability for the one-shot IPC verbs: open the first
//! candidate instance socket, perform the protocol handshake, and read the
//! resident-model status. Both `respond` and `embed` start here, on
//! separate connections per request, and never touch the REST surface.

use std::{path::PathBuf, time::Duration};

use astronomical_ipc_protocol::{
    DAEMON_APPLICATION_NAME, DaemonIpcClient, DaemonRequest, DaemonResponse,
};
use tokio::time::timeout;

/// Why a verb could not learn the daemon's state before submitting work.
#[derive(Debug, thiserror::Error)]
pub enum DaemonProbeError {
    #[error("Astronomical isn't running — start it, then retry.")]
    DaemonNotRunning,
    #[error("The daemon stopped responding.")]
    DaemonStoppedResponding,
}

/// Socket candidates plus the per-stage bound both verbs share.
pub struct DaemonProbe {
    /// Instance sockets to try, most preferred first.
    pub candidate_socket_paths: Vec<PathBuf>,
    /// Bound for each protocol stage of the journey.
    pub request_timeout: Duration,
}

impl DaemonProbe {
    /// Connects, handshakes, and reports which model the daemon has resident.
    /// `None` means the daemon is running with no model ready.
    pub async fn ready_model_id(&self) -> Result<Option<String>, DaemonProbeError> {
        self.handshake_on_fresh_connection().await?;
        match self.status_on_fresh_connection().await? {
            DaemonResponse::Status { ready_model_id, .. } => Ok(ready_model_id),
            _ => Err(DaemonProbeError::DaemonStoppedResponding),
        }
    }

    /// Opens a connection the caller owns for its own request.
    pub async fn connect(&self) -> Result<DaemonIpcClient, DaemonProbeError> {
        for candidate_socket_path in &self.candidate_socket_paths {
            let daemon_client = timeout(
                self.request_timeout,
                DaemonIpcClient::connect(candidate_socket_path.clone()),
            )
            .await;
            if let Ok(Ok(daemon_client)) = daemon_client {
                return Ok(daemon_client);
            }
        }
        Err(DaemonProbeError::DaemonNotRunning)
    }

    async fn handshake_on_fresh_connection(&self) -> Result<(), DaemonProbeError> {
        let handshake_accepted = self
            .request_on_fresh_connection(&DaemonRequest::Handshake)
            .await
            .map_err(daemon_not_running_on_handshake)?;
        match handshake_accepted {
            DaemonResponse::HandshakeAccepted {
                application_name, ..
            } if application_name == DAEMON_APPLICATION_NAME => Ok(()),
            _ => Err(DaemonProbeError::DaemonNotRunning),
        }
    }

    async fn status_on_fresh_connection(&self) -> Result<DaemonResponse, DaemonProbeError> {
        self.request_on_fresh_connection(&DaemonRequest::Status)
            .await
            .map_err(|_probe_error| DaemonProbeError::DaemonStoppedResponding)
    }

    /// One request per connection: the daemon serves each socket connection
    /// as one exchange, so the probe and the verb's real request each open
    /// their own connection.
    async fn request_on_fresh_connection(
        &self,
        daemon_request: &DaemonRequest,
    ) -> Result<DaemonResponse, DaemonProbeError> {
        let mut daemon_client = self.connect().await?;
        timeout(self.request_timeout, async {
            daemon_client.send_request(daemon_request).await?;
            daemon_client.next_response().await
        })
        .await
        .map_err(|_elapsed| DaemonProbeError::DaemonStoppedResponding)?
        .map_err(|_transport_error| DaemonProbeError::DaemonStoppedResponding)?
        .ok_or(DaemonProbeError::DaemonStoppedResponding)
    }
}

/// A connect or transport failure during the handshake means no live
/// Astronomical answered; the same failure later means a daemon that went
/// away mid-conversation.
fn daemon_not_running_on_handshake(probe_error: DaemonProbeError) -> DaemonProbeError {
    match probe_error {
        DaemonProbeError::DaemonStoppedResponding => DaemonProbeError::DaemonNotRunning,
        already_not_running => already_not_running,
    }
}
