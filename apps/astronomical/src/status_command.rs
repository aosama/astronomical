//! The `astronomical status` verb: worker state, resident model, effective
//! default model, and the active download job, if any.

use std::{fmt::Write as _, io::Write, path::PathBuf, time::Duration};

use astronomical_ipc_protocol::{DaemonDownloadJob, DaemonWorkerStatus};

use crate::{
    DaemonProbe, daemon_probe::DaemonStatusSnapshot, errors::StatusError,
    formatting::format_gigabytes,
};

/// Collaborators the status verb needs, injected so tests can stub them.
pub struct StatusDependencies<'a> {
    /// Instance sockets to try, most preferred first.
    pub candidate_socket_paths: Vec<PathBuf>,
    /// Where the status report goes.
    pub stdout: &'a mut dyn Write,
    /// Bound for each protocol exchange.
    pub request_timeout: Duration,
}

/// Runs the status journey against the resident daemon.
pub async fn run_status(
    status_dependencies: &mut StatusDependencies<'_>,
) -> Result<(), StatusError> {
    let daemon_probe = DaemonProbe {
        candidate_socket_paths: status_dependencies.candidate_socket_paths.clone(),
        request_timeout: status_dependencies.request_timeout,
    };
    let status_snapshot = daemon_probe.status_snapshot().await?;
    let active_download_job = daemon_probe.download_status().await?;
    let mut report = String::new();
    let _ = writeln!(
        report,
        "worker:   {}",
        render_worker_status(&status_snapshot)
    );
    let _ = writeln!(
        report,
        "default:  {}",
        status_snapshot
            .default_model_id
            .as_deref()
            .unwrap_or("none")
    );
    let _ = writeln!(
        report,
        "download: {}",
        render_download_line(active_download_job.as_ref())
    );
    status_dependencies
        .stdout
        .write_all(report.as_bytes())
        .map_err(|output_error| StatusError::StdoutUnwritable {
            cause: output_error.to_string(),
        })
}

fn render_worker_status(status_snapshot: &DaemonStatusSnapshot) -> String {
    let worker_state = match status_snapshot.worker_status {
        DaemonWorkerStatus::Ready => "ready",
        DaemonWorkerStatus::Loading => "loading",
        DaemonWorkerStatus::Unavailable => "unavailable",
    };
    match &status_snapshot.ready_model_id {
        Some(ready_model_id) => format!("{worker_state} (resident: {ready_model_id})"),
        None => worker_state.to_owned(),
    }
}

fn render_download_line(active_download_job: Option<&DaemonDownloadJob>) -> String {
    match active_download_job {
        None => "none".to_owned(),
        Some(active_download_job) => {
            if let Some(download_error) = &active_download_job.error {
                return format!(
                    "{} (failed: {download_error})",
                    active_download_job.huggingface_id
                );
            }
            if active_download_job.bytes_total == 0 {
                return format!(
                    "{} ({})",
                    active_download_job.huggingface_id, active_download_job.state
                );
            }
            format!(
                "{} — {} {} GB / {} GB",
                active_download_job.huggingface_id,
                active_download_job.state,
                format_gigabytes(active_download_job.bytes_completed),
                format_gigabytes(active_download_job.bytes_total),
            )
        }
    }
}
