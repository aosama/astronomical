//! Turns the `runtime.metal_residency_refresh_enabled` setting into what MLX
//! actually reads when a worker starts.
//!
//! MLX exposes this control only as an environment variable, read once when the
//! worker creates its Metal device. The daemon sets it on the child's command
//! rather than having the worker set its own environment, because the worker
//! would be changing process-global state with its runtime threads running, and
//! could lose the race with the first Metal call.

use astronomical_ipc_protocol::WorkerStartupConfiguration;
use tokio::process::Command;

const MLX_RESIDENCY_REFRESH_VARIABLE: &str = "MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS";
// MLX documents one second as its example. Renewal submits no GPU work, so a
// finer interval would only shorten the window after macOS releases residency.
const RESIDENCY_REFRESH_INTERVAL_MILLISECONDS: &str = "1000";

/// Applies the setting to a worker command. A launch without a startup
/// configuration has no setting to apply, so its environment is left alone.
pub(crate) fn apply_metal_residency_refresh_policy(
    worker_command: &mut Command,
    worker_startup_configuration: Option<&WorkerStartupConfiguration>,
) {
    let Some(worker_startup_configuration) = worker_startup_configuration else {
        return;
    };
    if worker_startup_configuration.metal_residency_refresh_enabled {
        worker_command.env(
            MLX_RESIDENCY_REFRESH_VARIABLE,
            RESIDENCY_REFRESH_INTERVAL_MILLISECONDS,
        );
    } else {
        // Removed rather than left unset, so a variable inherited from the
        // daemon's own environment cannot contradict what the page shows.
        worker_command.env_remove(MLX_RESIDENCY_REFRESH_VARIABLE);
    }
}

#[cfg(test)]
#[path = "metal_residency_refresh_tests.rs"]
mod metal_residency_refresh_tests;
