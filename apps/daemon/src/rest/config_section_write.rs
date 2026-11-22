//! Writing a few named values of one `config.json` section for a caller that
//! is not Observatory's Settings page, such as the menu-bar agent or the chat
//! app. Each write takes the same lock, validation, save and reload path as
//! the Settings page, so every place shows the same stored value.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};

use crate::application::ApplicationState;
use crate::rest::config_document::{
    reload_after_write, validate_candidate_resolves, write_error_response,
};

/// The current document with `section_values` set inside `section_name`,
/// adding the section when it is missing. A document that is not a JSON
/// object is returned unchanged, so validation names the problem.
pub(super) fn document_with_section_values(
    mut config_document: serde_json::Value,
    section_name: &str,
    section_values: &[(&str, serde_json::Value)],
) -> serde_json::Value {
    if let Some(document_members) = config_document.as_object_mut() {
        if !document_members.contains_key(section_name) {
            document_members.insert(section_name.to_owned(), serde_json::json!({}));
        }
        if let Some(section_members) = document_members
            .get_mut(section_name)
            .and_then(serde_json::Value::as_object_mut)
        {
            for (setting_name, setting_value) in section_values {
                section_members.insert((*setting_name).to_owned(), setting_value.clone());
            }
        }
    }
    config_document
}

/// Saves `section_values` into `section_name` and reloads, refusing a value
/// the schema or the resolver rejects.
pub(super) async fn write_config_section(
    application_state: ApplicationState,
    section_name: &str,
    section_values: &[(&str, serde_json::Value)],
) -> Response {
    let Some(runtime_config_resolver) = application_state.runtime_config_resolver.clone() else {
        return (StatusCode::NOT_FOUND, "config persistence is unavailable").into_response();
    };
    let document_digest = {
        let _configuration_transition_guard =
            application_state.configuration_transition_lock.lock().await;
        let snapshot = match config::read_config_document(runtime_config_resolver.state_directory())
        {
            Ok(snapshot) => snapshot,
            Err(configuration_error) => return write_error_response(configuration_error),
        };
        let Some(current_document) = snapshot.document else {
            // A missing or unreadable file is repaired on the Settings page, which can show
            // what is wrong with it; writing one field over it here would hide that.
            return (
                StatusCode::CONFLICT,
                "config.json could not be read; open Settings in Observatory to repair it",
            )
                .into_response();
        };
        let candidate_document =
            document_with_section_values(current_document, section_name, section_values);
        if let Err(refusal_response) =
            validate_candidate_resolves(&runtime_config_resolver, &candidate_document)
        {
            return *refusal_response;
        }
        match config::save_config_document(
            runtime_config_resolver.state_directory(),
            snapshot.document_digest.as_deref(),
            &candidate_document,
        ) {
            Ok(document_digest) => document_digest,
            Err(configuration_error) => return write_error_response(configuration_error),
        }
    };
    reload_after_write(application_state, document_digest).await
}
