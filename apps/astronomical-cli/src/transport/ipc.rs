//! Shared daemon reachability for the one-shot IPC verbs: open the first
//! candidate instance socket and read status, model list, catalog, download
//! state, or persist a default model. Both `respond` and `embed` start here,
//! on separate connections per request, and never touch the REST surface.

use std::{path::PathBuf, time::Duration};

use astronomical_config::AstronomicalRuntimeInstance;
use astronomical_ipc_protocol::{
    DaemonCatalogEntry, DaemonDownloadJob, DaemonIpcClient, DaemonListedModel, DaemonRequest,
    DaemonResponse, DaemonWorkerStatus,
};
use tokio::time::timeout;

/// The single wording source for "this CLI's own-channel daemon is not
/// reachable": names the channel and every socket the probe tried, so the
/// failure explains itself when the other channel's app is running normally
/// (issue #1392 — channel binding follows the executable path, and strict
/// isolation means restarting the other channel's app cannot help). Every
/// verb error carries this text unchanged, so the guidance stays
/// byte-identical across `respond`, `embed`, `models`, and `status`.
#[must_use]
pub fn daemon_not_running_disclosure(
    runtime_instance: AstronomicalRuntimeInstance,
    probed_socket_paths: &[PathBuf],
) -> String {
    let probed_socket_list_text = if probed_socket_paths.is_empty() {
        "none".to_owned()
    } else {
        probed_socket_paths
            .iter()
            .map(|probed_socket_path| probed_socket_path.to_string_lossy())
            .collect::<Vec<_>>()
            .join(", ")
    };
    let socket_word = if probed_socket_paths.len() == 1 {
        "socket"
    } else {
        "sockets"
    };
    format!(
        "Astronomical ({} channel) isn't running — start it, then retry. \
         (probed {}: {})",
        runtime_instance.as_str(),
        socket_word,
        probed_socket_list_text
    )
}

/// Why a verb could not learn the daemon's state before submitting work.
#[derive(Debug, thiserror::Error)]
pub enum DaemonProbeError {
    #[error("{disclosure}")]
    DaemonNotRunning { disclosure: String },
    #[error("The daemon stopped responding.")]
    DaemonStoppedResponding,
    #[error("The daemon declined the request: {reason}")]
    DaemonRejected { reason: String },
}

/// Worker state plus the daemon's effective model ids from one Status frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DaemonStatusSnapshot {
    /// Whether the inference engine can serve generations right now.
    pub worker_status: DaemonWorkerStatus,
    /// Model currently resident in the worker, if any.
    pub ready_model_id: Option<String>,
    /// Effective default model: the persisted value, else the built-in.
    pub default_model_id: Option<String>,
}

/// Socket candidates plus the per-stage bound both verbs share.
pub struct DaemonProbe {
    /// Instance sockets to try: the binary's own channel only (strict channel
    /// isolation; no cross-channel fallback).
    pub candidate_socket_paths: Vec<PathBuf>,
    /// Bound for each protocol stage of the journey.
    pub request_timeout: Duration,
    /// Channel this probe belongs to: the not-running disclosure names it, so
    /// the failure says whose daemon is down instead of leaving the user to
    /// guess which of the two installed apps the error is about.
    pub runtime_instance: AstronomicalRuntimeInstance,
}

impl DaemonProbe {
    /// Worker state, resident model, and effective default model in one probe.
    pub async fn status_snapshot(&self) -> Result<DaemonStatusSnapshot, DaemonProbeError> {
        let status_response = self
            .request_on_fresh_connection(&DaemonRequest::Status)
            .await?;
        match status_response {
            DaemonResponse::Status {
                worker_status,
                ready_model_id,
                default_model_id,
            } => Ok(DaemonStatusSnapshot {
                worker_status,
                ready_model_id,
                default_model_id,
            }),
            _ => Err(DaemonProbeError::DaemonStoppedResponding),
        }
    }

    /// The models discovered on this Mac, with their resident markers.
    pub async fn models_list(&self) -> Result<Vec<DaemonListedModel>, DaemonProbeError> {
        match self
            .request_on_fresh_connection(&DaemonRequest::ModelsList)
            .await?
        {
            DaemonResponse::ModelsList { models } => Ok(models),
            DaemonResponse::RequestRejected { reason } => {
                Err(DaemonProbeError::DaemonRejected { reason })
            }
            _ => Err(DaemonProbeError::DaemonStoppedResponding),
        }
    }

    /// The release download catalog with local readiness per entry.
    pub async fn catalog(&self) -> Result<Vec<DaemonCatalogEntry>, DaemonProbeError> {
        match self
            .request_on_fresh_connection(&DaemonRequest::Catalog)
            .await?
        {
            DaemonResponse::Catalog { entries } => Ok(entries),
            DaemonResponse::RequestRejected { reason } => {
                Err(DaemonProbeError::DaemonRejected { reason })
            }
            _ => Err(DaemonProbeError::DaemonStoppedResponding),
        }
    }

    /// Starts (or resumes) a download; the daemon's refusal arrives as an error.
    pub async fn download_start(&self, model_id: &str) -> Result<(), DaemonProbeError> {
        match self
            .request_on_fresh_connection(&DaemonRequest::DownloadStart {
                model_id: model_id.to_owned(),
            })
            .await?
        {
            DaemonResponse::DownloadStarted { .. } => Ok(()),
            DaemonResponse::RequestRejected { reason } => {
                Err(DaemonProbeError::DaemonRejected { reason })
            }
            _ => Err(DaemonProbeError::DaemonStoppedResponding),
        }
    }

    /// The active download job, if one runs.
    pub async fn download_status(&self) -> Result<Option<DaemonDownloadJob>, DaemonProbeError> {
        match self
            .request_on_fresh_connection(&DaemonRequest::DownloadStatus)
            .await?
        {
            DaemonResponse::DownloadStatus { job } => Ok(job),
            DaemonResponse::RequestRejected { reason } => {
                Err(DaemonProbeError::DaemonRejected { reason })
            }
            _ => Err(DaemonProbeError::DaemonStoppedResponding),
        }
    }

    /// Persists a new default model id; the daemon's refusal arrives as an error.
    pub async fn default_model_set(&self, model_id: &str) -> Result<String, DaemonProbeError> {
        match self
            .request_on_fresh_connection(&DaemonRequest::DefaultModelSet {
                model_id: model_id.to_owned(),
            })
            .await?
        {
            DaemonResponse::DefaultModelSet { default_model_id } => Ok(default_model_id),
            DaemonResponse::RequestRejected { reason } => {
                Err(DaemonProbeError::DaemonRejected { reason })
            }
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
        Err(DaemonProbeError::DaemonNotRunning {
            disclosure: daemon_not_running_disclosure(
                self.runtime_instance,
                &self.candidate_socket_paths,
            ),
        })
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

#[cfg(test)]
mod ipc_tests;
