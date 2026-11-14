//! Localhost endpoints behind Observatory's Settings and Tuning pages.
//!
//! Saving reuses the reload endpoint, so a change takes the same validation and
//! worker transitions as a hand edit followed by a reload. The file is the source
//! of truth: a valid save stays on disk even when applying has to wait for an
//! active generation or a restart, and the response says which happened.

use axum::{
    Json,
    body::to_bytes,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};

use crate::application::ApplicationState;
use crate::config::config_reload::ResolvedRuntimeConfigResolver;
use crate::rest::loopback_browser_guard::refusal_for_foreign_request;

// A reload response is a small status document; this only bounds a surprise.
const MAXIMUM_RELOAD_RESPONSE_BYTES: usize = 1024 * 1024;
const RELOAD_UNREADABLE_STATUS: &str = "reload_response_unreadable";

#[derive(Serialize)]
struct ConfigDocumentResponse {
    document: Option<serde_json::Value>,
    read_error: Option<String>,
    document_digest: Option<String>,
    schema: serde_json::Value,
    /// Settings this instance cannot honour, as paths in the document. The
    /// console leaves them off its pages.
    unavailable_settings: &'static [&'static str],
}

/// The settings that name a folder of the person's choosing. A sandboxed
/// engine can open only its own container, so offering them in the Mac App
/// Store build would be offering something that cannot work (issue #2187
/// tracks giving the store build a folder picker instead).
const FOLDER_SETTINGS: [&str; 3] = [
    "runtime.model_directories",
    "runtime.model_download_directory",
    "runtime.neural_engine_program_directory",
];

fn unavailable_settings(
    instance_paths: &astronomical_config::AstronomicalInstancePaths,
) -> &'static [&'static str] {
    if instance_paths.keeps_state_in_app_container() {
        &FOLDER_SETTINGS
    } else {
        &[]
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SaveConfigDocumentRequest {
    document: serde_json::Value,
    expected_document_digest: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ResetConfigDocumentRequest {
    expected_document_digest: Option<String>,
}

#[derive(Serialize)]
struct ConfigWriteRefusal {
    status: &'static str,
    message: String,
}

#[derive(Serialize)]
struct ConfigWriteOutcome {
    status: &'static str,
    document_digest: String,
    /// HTTP status of the reload that followed the write.
    reload_http_status: u16,
    reload: serde_json::Value,
}

fn unavailable_response() -> Response {
    (StatusCode::NOT_FOUND, "config persistence is unavailable").into_response()
}

fn refusal(status_code: StatusCode, status: &'static str, message: String) -> Response {
    (status_code, Json(ConfigWriteRefusal { status, message })).into_response()
}

fn write_error_response(
    configuration_error: astronomical_config::AstronomicalConfigError,
) -> Response {
    use astronomical_config::AstronomicalConfigError as ConfigError;
    match configuration_error {
        ConfigError::ConfigChangedDuringUpdate => refusal(
            StatusCode::CONFLICT,
            "changed_elsewhere",
            "config.json changed since this page loaded it; reload the page and reapply your edits"
                .to_owned(),
        ),
        ConfigError::ReadConfigFile { .. }
        | ConfigError::WriteConfigFile { .. }
        | ConfigError::SerializeConfigFile { .. } => {
            tracing::error!(error = %configuration_error, "could not write the configuration document");
            refusal(
                StatusCode::INTERNAL_SERVER_ERROR,
                "write_failed",
                "the configuration could not be written; inspect local diagnostics".to_owned(),
            )
        }
        invalid_document => refusal(
            StatusCode::BAD_REQUEST,
            "invalid_config",
            invalid_document.to_string(),
        ),
    }
}

/// `GET /v1/config`: the document, its digest, and the schema that describes it.
pub(crate) async fn read_config_document(
    State(application_state): State<ApplicationState>,
    headers: HeaderMap,
) -> Response {
    if let Some(refusal_response) = refusal_for_foreign_request(&headers) {
        return refusal_response;
    }
    let Some(runtime_config_resolver) = application_state.runtime_config_resolver.as_ref() else {
        return unavailable_response();
    };
    let snapshot = match astronomical_config::read_config_document(
        runtime_config_resolver.state_directory(),
    ) {
        Ok(snapshot) => snapshot,
        Err(configuration_error) => return write_error_response(configuration_error),
    };
    let schema = match serde_json::from_slice(astronomical_config::config_schema_bytes()) {
        Ok(schema) => schema,
        Err(schema_error) => {
            tracing::error!(error = %schema_error, "the bundled configuration schema is invalid");
            return refusal(
                StatusCode::INTERNAL_SERVER_ERROR,
                "schema_unavailable",
                "the bundled configuration schema could not be read".to_owned(),
            );
        }
    };
    Json(ConfigDocumentResponse {
        document: snapshot.document,
        read_error: snapshot.read_error,
        document_digest: snapshot.document_digest,
        schema,
        unavailable_settings: unavailable_settings(runtime_config_resolver.instance_paths()),
    })
    .into_response()
}

/// A candidate must also resolve against the discovered models: a context limit
/// above an artifact's own passes the document's checks but fails at this step,
/// which would otherwise leave a file the daemon refuses to reload.
fn validate_candidate_resolves(
    runtime_config_resolver: &ResolvedRuntimeConfigResolver,
    candidate_document: &serde_json::Value,
) -> Result<(), Box<Response>> {
    if let Err(configuration_error) = astronomical_config::validate_config_document(
        runtime_config_resolver.state_directory(),
        candidate_document,
    ) {
        return Err(Box::new(write_error_response(configuration_error)));
    }
    let candidate_bytes = candidate_document.to_string().into_bytes();
    let candidate_config = match astronomical_config::AstronomicalConfig::load_v1_bytes(
        runtime_config_resolver.instance_paths().clone(),
        &candidate_bytes,
    ) {
        Ok(candidate_config) => candidate_config,
        Err(configuration_error) => {
            return Err(Box::new(write_error_response(configuration_error)));
        }
    };
    if let Err(resolution_error) = runtime_config_resolver.resolve(&candidate_config) {
        return Err(Box::new(refusal(
            StatusCode::BAD_REQUEST,
            "invalid_config",
            resolution_error.to_string(),
        )));
    }
    Ok(())
}

async fn reload_after_write(
    application_state: ApplicationState,
    document_digest: String,
) -> Response {
    let reload_response = crate::rest::config_reload::reload_config(State(application_state)).await;
    let reload_http_status = reload_response.status().as_u16();
    let reload = match to_bytes(reload_response.into_body(), MAXIMUM_RELOAD_RESPONSE_BYTES).await {
        Ok(reload_body) => match serde_json::from_slice(&reload_body) {
            Ok(reload_document) => reload_document,
            Err(_) => serde_json::json!({ "status": RELOAD_UNREADABLE_STATUS }),
        },
        Err(_) => serde_json::json!({ "status": RELOAD_UNREADABLE_STATUS }),
    };
    Json(ConfigWriteOutcome {
        status: "saved",
        document_digest,
        reload_http_status,
        reload,
    })
    .into_response()
}

/// `PUT /v1/config`: validates, atomically replaces the document, then reloads.
pub(crate) async fn save_config_document(
    State(application_state): State<ApplicationState>,
    headers: HeaderMap,
    Json(save_request): Json<SaveConfigDocumentRequest>,
) -> Response {
    if let Some(refusal_response) = refusal_for_foreign_request(&headers) {
        return refusal_response;
    }
    let Some(runtime_config_resolver) = application_state.runtime_config_resolver.clone() else {
        return unavailable_response();
    };
    let document_digest = {
        let _configuration_transition_guard =
            application_state.configuration_transition_lock.lock().await;
        if let Err(refusal_response) =
            validate_candidate_resolves(&runtime_config_resolver, &save_request.document)
        {
            return *refusal_response;
        }
        match astronomical_config::save_config_document(
            runtime_config_resolver.state_directory(),
            save_request.expected_document_digest.as_deref(),
            &save_request.document,
        ) {
            Ok(document_digest) => document_digest,
            Err(configuration_error) => return write_error_response(configuration_error),
        }
    };
    reload_after_write(application_state, document_digest).await
}

/// `POST /v1/config/reset`: first-run defaults, keeping the old file beside them.
pub(crate) async fn reset_config_document(
    State(application_state): State<ApplicationState>,
    headers: HeaderMap,
    Json(reset_request): Json<ResetConfigDocumentRequest>,
) -> Response {
    if let Some(refusal_response) = refusal_for_foreign_request(&headers) {
        return refusal_response;
    }
    let Some(runtime_config_resolver) = application_state.runtime_config_resolver.clone() else {
        return unavailable_response();
    };
    let document_digest = {
        let _configuration_transition_guard =
            application_state.configuration_transition_lock.lock().await;
        match astronomical_config::reset_config_document(
            runtime_config_resolver.state_directory(),
            reset_request.expected_document_digest.as_deref(),
        ) {
            Ok(document_digest) => document_digest,
            Err(configuration_error) => return write_error_response(configuration_error),
        }
    };
    reload_after_write(application_state, document_digest).await
}

#[cfg(test)]
#[path = "config_document_tests.rs"]
mod config_document_tests;
