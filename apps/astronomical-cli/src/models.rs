//! Selects Library chat models from `/v1/models`. Image and embedding
//! advertisements are not coding-harness launch targets.

use serde_json::Value;

use crate::errors::LaunchError;

const CHAT_COMPLETIONS_ENDPOINT: &str = "/v1/chat/completions";

/// One chat model the launched session may use.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LibraryChatModel {
    pub model_id: String,
    pub context_window_tokens: Option<u32>,
}

/// Keeps advertised order so the picker matches Library listing.
pub fn chat_models_from_models_document(
    models_document: &Value,
) -> Result<Vec<LibraryChatModel>, LaunchError> {
    let advertised_models = models_document
        .get("data")
        .and_then(Value::as_array)
        .ok_or(LaunchError::ModelListUnavailable)?;

    let mut chat_models = Vec::new();
    for advertised_model in advertised_models {
        let Some(model_id) = advertised_model.get("id").and_then(Value::as_str) else {
            continue;
        };
        if !advertises_chat_completions(advertised_model) {
            continue;
        }
        let context_window_tokens = advertised_model
            .get("context_window")
            .and_then(Value::as_u64)
            .and_then(|context_window| u32::try_from(context_window).ok());
        chat_models.push(LibraryChatModel {
            model_id: model_id.to_owned(),
            context_window_tokens,
        });
    }
    Ok(chat_models)
}

fn advertises_chat_completions(advertised_model: &Value) -> bool {
    advertised_model
        .get("supported_endpoints")
        .and_then(Value::as_array)
        .is_some_and(|supported_endpoints| {
            supported_endpoints.iter().any(|supported_endpoint| {
                supported_endpoint.as_str() == Some(CHAT_COMPLETIONS_ENDPOINT)
            })
        })
}
