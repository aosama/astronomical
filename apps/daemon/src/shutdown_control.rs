//! Internal localhost shutdown controller for `POST /v1/control/shutdown`.
//!
//! Allows the menu bar app to request a graceful daemon shutdown over HTTP
//! instead of relying on OS signals alone. The controller uses a `tokio::sync::watch`
//! guarded by an `AtomicBool` so repeated requests cannot panic and the existing
//! Ctrl-C/SIGTERM handlers remain untouched.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use tokio::sync::watch;

use crate::application::ApplicationState;

/// Shared shutdown signal. `request_shutdown` returns whether this was the
/// first caller to trigger the shutdown.
#[derive(Clone)]
pub struct ShutdownController {
    shutdown_sender: Arc<watch::Sender<bool>>,
    shutdown_already_requested: Arc<AtomicBool>,
}

impl ShutdownController {
    /// Creates a new controller in the non-shutting-down state.
    #[must_use]
    pub fn new() -> Self {
        let (shutdown_sender, _shutdown_receiver) = watch::channel(false);
        Self {
            shutdown_sender: Arc::new(shutdown_sender),
            shutdown_already_requested: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Returns a receiver that observes the shutdown signal.
    pub fn subscribe(&self) -> watch::Receiver<bool> {
        self.shutdown_sender.subscribe()
    }

    /// Requests shutdown. Returns `true` if this was the first caller to
    /// trigger it, or `false` if shutdown was already requested.
    pub fn request_shutdown(&self) -> bool {
        let was_first = self.shutdown_already_requested.compare_exchange(
            false,
            true,
            Ordering::SeqCst,
            Ordering::SeqCst,
        );
        if was_first.is_ok() {
            // Unlike `send`, `send_replace` persists the request even if the
            // daemon has not subscribed to this controller yet.
            let _previous_shutdown_requested = self.shutdown_sender.send_replace(true);
            true
        } else {
            false
        }
    }
}

impl Default for ShutdownController {
    fn default() -> Self {
        Self::new()
    }
}

/// Triggers the daemon's existing graceful-shutdown path.
pub(crate) async fn request_supervisor_shutdown(
    State(application_state): State<ApplicationState>,
) -> Response {
    let Some(shutdown_controller) = application_state.shutdown_controller.as_ref() else {
        return (StatusCode::NOT_FOUND, "shutdown not supported").into_response();
    };
    shutdown_controller.request_shutdown();
    (
        StatusCode::ACCEPTED,
        Json(serde_json::json!({
            "status": "shutting_down",
            "message": "Astronomical daemon is shutting down",
        })),
    )
        .into_response()
}
