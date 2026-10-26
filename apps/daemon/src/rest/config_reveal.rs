//! Path-safe localhost control that reveals the active configuration in Finder.

use std::{process::Stdio, time::Duration};

use axum::{Router, extract::State, http::StatusCode, response::IntoResponse, routing::post};
use tokio::{process::Command, time::timeout};

use crate::application::ApplicationState;

const CONFIG_REVEAL_TIMEOUT: Duration = Duration::from_secs(5);

pub(crate) fn config_reveal_routes() -> Router<ApplicationState> {
    Router::new().route("/v1/config/reveal", post(reveal_config))
}

async fn reveal_config(State(application_state): State<ApplicationState>) -> impl IntoResponse {
    let Some(runtime_config_resolver) = application_state.runtime_config_resolver.as_ref() else {
        return StatusCode::NOT_FOUND;
    };
    let config_file_path = runtime_config_resolver.instance_paths().config_file_path();
    let reveal_outcome = timeout(
        CONFIG_REVEAL_TIMEOUT,
        Command::new("/usr/bin/open")
            .arg("-R")
            .arg(config_file_path)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status(),
    )
    .await;
    match reveal_outcome {
        Ok(Ok(exit_status)) if exit_status.success() => StatusCode::NO_CONTENT,
        Ok(Ok(_)) | Ok(Err(_)) | Err(_) => StatusCode::INTERNAL_SERVER_ERROR,
    }
}
