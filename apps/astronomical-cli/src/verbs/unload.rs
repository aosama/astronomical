//! The `astronomical unload` verb: unload the loaded model now and free its
//! memory, for a person or a tool that needs the Mac's memory back before
//! the idle unload would have done it.

use std::{io::Write, path::PathBuf, time::Duration};

use config::AstronomicalRuntimeInstance;
use ipc_protocol::DaemonModelUnloadStatus;
use thiserror::Error;

use crate::DaemonProbe;
use crate::transport::ipc::DaemonProbeError;

/// Collaborators the unload verb needs, injected so tests can stub them.
pub struct UnloadDependencies<'a> {
    /// Instance sockets to try, most preferred first.
    pub candidate_socket_paths: Vec<PathBuf>,
    /// Channel these sockets belong to, named when its daemon is down.
    pub runtime_instance: AstronomicalRuntimeInstance,
    /// Where the outcome line goes.
    pub stdout: &'a mut dyn Write,
    /// Bound for the exchange. Unloading replaces the worker process, so the
    /// caller passes a bound that covers a fresh worker starting.
    pub request_timeout: Duration,
}

/// Asks the daemon to unload its model and reports what happened. A model
/// left loaded because a request holds it is an error, so scripts can tell.
pub async fn run_unload(
    unload_dependencies: &mut UnloadDependencies<'_>,
) -> Result<(), UnloadError> {
    let daemon_probe = DaemonProbe {
        candidate_socket_paths: unload_dependencies.candidate_socket_paths.clone(),
        request_timeout: unload_dependencies.request_timeout,
        runtime_instance: unload_dependencies.runtime_instance,
    };
    let (unload_status, unloaded_model_id) = daemon_probe.model_unload().await?;
    let outcome_line = unload_outcome_line(unload_status, unloaded_model_id.as_deref())?;
    writeln!(unload_dependencies.stdout, "{outcome_line}").map_err(|output_error| {
        UnloadError::StdoutUnwritable {
            cause: output_error.to_string(),
        }
    })
}

/// The line to print for an unload that freed or found no memory, or the
/// error for one that could not.
pub fn unload_outcome_line(
    unload_status: DaemonModelUnloadStatus,
    unloaded_model_id: Option<&str>,
) -> Result<String, UnloadError> {
    let model_name = unloaded_model_id.unwrap_or("The model");
    match unload_status {
        DaemonModelUnloadStatus::Unloaded => {
            Ok(format!("Unloaded {model_name}. Its memory is free."))
        }
        DaemonModelUnloadStatus::NothingLoaded => Ok("No model was loaded.".to_owned()),
        DaemonModelUnloadStatus::UnloadedEngineRestarting => Ok(format!(
            "Unloaded {model_name}. Its memory is free; the engine is still restarting."
        )),
        DaemonModelUnloadStatus::Busy => Err(UnloadError::Busy),
        DaemonModelUnloadStatus::Unavailable => Err(UnloadError::EngineNotRunning),
    }
}

/// Failures of the `unload` verb after arguments have parsed. These exit 1.
#[derive(Error)]
pub enum UnloadError {
    #[error("{disclosure}")]
    DaemonNotRunning { disclosure: String },
    #[error("The daemon stopped responding.")]
    DaemonStoppedResponding,
    #[error("{reason}")]
    DaemonRejected { reason: String },
    #[error(
        "A request is running or waiting, so the model stays loaded. Try again when it finishes."
    )]
    Busy,
    #[error("The engine is not running, so there is nothing to unload.")]
    EngineNotRunning,
    #[error("Could not write to standard output: {cause}")]
    StdoutUnwritable { cause: String },
}

impl From<DaemonProbeError> for UnloadError {
    fn from(probe_error: DaemonProbeError) -> Self {
        match probe_error {
            DaemonProbeError::DaemonNotRunning { disclosure } => {
                UnloadError::DaemonNotRunning { disclosure }
            }
            DaemonProbeError::DaemonStoppedResponding => UnloadError::DaemonStoppedResponding,
            DaemonProbeError::DaemonRejected { reason } => UnloadError::DaemonRejected { reason },
        }
    }
}

impl std::fmt::Debug for UnloadError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.to_string())
    }
}

#[cfg(test)]
#[path = "unload_tests.rs"]
mod tests;
