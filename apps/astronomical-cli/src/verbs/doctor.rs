//! The `astronomical doctor` verb: a read-only diagnostic answering "which
//! channel is this `astronomical` binary bound to, and is each channel's
//! daemon up?" (issue #1391). It is the only command that deliberately
//! probes both channels: strict isolation still governs real verb traffic
//! (a doctor report routes nothing), and a binding mismatch is only visible
//! when both channels' liveness is printed side by side.

use std::path::{Path, PathBuf};
use std::time::Duration;
use std::{fmt::Write as _, io::Write};

use astronomical_config::AstronomicalRuntimeInstance;

use crate::instance::runtime_instance_from_executable_path;
use crate::transport::ipc::{DaemonProbe, DaemonProbeError};

/// Failures of the doctor report itself. A dead daemon is report data, not
/// an error: the verb succeeds whenever the report reaches standard output.
#[derive(Debug, thiserror::Error)]
pub enum DoctorError {
    #[error("Could not write the doctor report to standard output: {cause}")]
    StdoutUnwritable { cause: String },
}

/// Collaborators the doctor needs, injected so tests can stub them.
pub struct DoctorDependencies<'a> {
    /// Executable path this CLI runs from: the caller canonicalizes it when
    /// possible, because the channel binding follows the real binary.
    pub executable_path: PathBuf,
    /// One (channel, socket path) pair per runtime instance, resolved by the
    /// caller: the binary injects both channels' default locations, tests
    /// inject temporary paths.
    pub channel_socket_paths: Vec<(AstronomicalRuntimeInstance, PathBuf)>,
    /// Where the report goes.
    pub stdout: &'a mut dyn Write,
    /// Bound for each channel's status probe.
    pub request_timeout: Duration,
}

/// Runs the doctor journey: name the channel binding, then probe every
/// channel's daemon for liveness.
pub async fn run_doctor(
    doctor_dependencies: &mut DoctorDependencies<'_>,
) -> Result<(), DoctorError> {
    let own_runtime_instance =
        runtime_instance_from_executable_path(&doctor_dependencies.executable_path);
    let mut report = String::new();
    let _ = writeln!(
        report,
        "binary:  {}",
        doctor_dependencies.executable_path.display()
    );
    let _ = writeln!(
        report,
        "channel: {} (own channel of this binary)",
        own_runtime_instance.as_str()
    );
    for (runtime_instance, socket_path) in &doctor_dependencies.channel_socket_paths {
        let socket_exists = socket_path.exists();
        let status_probe_text = if socket_exists {
            probe_channel_liveness(
                *runtime_instance,
                socket_path,
                doctor_dependencies.request_timeout,
            )
            .await
        } else {
            "not running — socket file absent".to_owned()
        };
        let _ = writeln!(report, "{}:", runtime_instance.as_str());
        let _ = writeln!(report, "  socket: {}", socket_path.display());
        let _ = writeln!(
            report,
            "  socket exists: {}",
            if socket_exists { "yes" } else { "no" }
        );
        let _ = writeln!(report, "  status probe: {status_probe_text}");
    }
    doctor_dependencies
        .stdout
        .write_all(report.as_bytes())
        .map_err(|output_error| DoctorError::StdoutUnwritable {
            cause: output_error.to_string(),
        })
}

/// One channel's daemon liveness, phrased for the report. Only an existing
/// socket is probed: an absent socket file is already a definitive answer,
/// and probing it would only re-derive the same verdict more slowly.
async fn probe_channel_liveness(
    runtime_instance: AstronomicalRuntimeInstance,
    socket_path: &Path,
    request_timeout: Duration,
) -> String {
    let daemon_probe = DaemonProbe {
        candidate_socket_paths: vec![socket_path.to_path_buf()],
        request_timeout,
        runtime_instance,
    };
    match daemon_probe.status_snapshot().await {
        Ok(_status_snapshot) => "live — daemon answered a status request".to_owned(),
        Err(DaemonProbeError::DaemonNotRunning { .. }) => {
            "not running — no daemon accepted a connection".to_owned()
        }
        Err(_other_probe_error) => {
            "unresponsive — daemon accepted a connection but did not answer a status request"
                .to_owned()
        }
    }
}

#[cfg(test)]
mod doctor_tests;
