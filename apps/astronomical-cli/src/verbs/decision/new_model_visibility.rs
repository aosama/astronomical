//! Making a just-written model folder servable, and saying so truthfully.
//!
//! The daemon looks for model folders when it starts and when its
//! configuration reloads, never in between, so a new folder is invisible
//! until something asks for a reload. `decision calibrate` asks, the way the
//! configuration page and the menu-bar agent do, and then checks the
//! daemon's own model list: whether discovery found the folder is the
//! daemon's call, and guessing it from the path here got it wrong once (it
//! scans more than one level deep).
//!
//! A reload never interrupts work: the daemon refuses one that would
//! restart a busy worker, and the folder then appears at the next reload.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use crate::DaemonProbe;
use crate::transport::loopback_http::post_loopback_json;

const CONFIGURATION_RELOAD_PATH: &str = "/v1/config/reload";

/// Whether the daemon serves the new model now.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NewModelVisibility {
    /// The daemon reloaded and lists the model.
    Listed,
    /// The daemon did not reload now (busy, or unreachable); it lists the
    /// model after its next reload or restart.
    ReloadDeclined,
    /// The daemon reloaded but does not list the model: the folder is outside
    /// every folder it looks in.
    NotFound,
}

/// Asks the daemon to look for models again, then reads whether it lists
/// `model_id`.
pub async fn make_new_model_visible(
    loopback_address: SocketAddr,
    daemon_probe: &DaemonProbe,
    model_id: &str,
) -> NewModelVisibility {
    let request_timeout = daemon_probe.request_timeout;
    let reload_outcome = tokio::task::spawn_blocking(move || {
        post_loopback_json(loopback_address, CONFIGURATION_RELOAD_PATH, request_timeout)
    })
    .await;
    if !matches!(reload_outcome, Ok(Ok(_))) {
        return NewModelVisibility::ReloadDeclined;
    }
    let Ok(listed_models) = daemon_probe.models_list().await else {
        return NewModelVisibility::ReloadDeclined;
    };
    for listed_model in &listed_models {
        if listed_model.model_id == model_id {
            return NewModelVisibility::Listed;
        }
    }
    NewModelVisibility::NotFound
}

/// The closing lines: where the model was written and how to ask for it.
#[must_use]
pub fn written_folder_text(
    output_directory: &Path,
    model_id: &str,
    visibility: NewModelVisibility,
    scanned_model_roots: &[PathBuf],
) -> String {
    let mut written_text = format!("\nWrote the new model to {}.\n", output_directory.display());
    let next_step_text = match visibility {
        NewModelVisibility::Listed => {
            format!("Astronomical serves it now; ask for it as {model_id}.\n")
        }
        NewModelVisibility::ReloadDeclined => format!(
            "Astronomical was busy and did not look for new models just now; ask for it as {model_id} once Astronomical restarts.\n"
        ),
        NewModelVisibility::NotFound => format!(
            "Astronomical does not look for models in {}. Move the folder into {}, or add its parent folder to the model folders in the instance configuration.\n",
            output_directory
                .parent()
                .unwrap_or(Path::new("."))
                .display(),
            scanned_model_roots
                .first()
                .map_or(Path::new("."), PathBuf::as_path)
                .display(),
        ),
    };
    written_text.push_str(&next_step_text);
    written_text
}

#[cfg(test)]
#[path = "new_model_visibility_tests.rs"]
mod tests;
