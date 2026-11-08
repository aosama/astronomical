//! Path-safe localhost control that reveals the active configuration in Finder.

use axum::{Router, extract::State, http::StatusCode, response::IntoResponse, routing::post};

use crate::application::ApplicationState;
use crate::rest::finder_reveal::reveal_path_in_finder;

pub(crate) fn config_reveal_routes() -> Router<ApplicationState> {
    Router::new().route("/v1/config/reveal", post(reveal_config))
}

async fn reveal_config(State(application_state): State<ApplicationState>) -> impl IntoResponse {
    let Some(runtime_config_resolver) = application_state.runtime_config_resolver.as_ref() else {
        return StatusCode::NOT_FOUND;
    };
    reveal_path_in_finder(&runtime_config_resolver.instance_paths().config_file_path()).await
}
