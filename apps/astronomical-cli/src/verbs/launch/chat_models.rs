//! Selects Library chat models from `/v1/models` for one harness. A model is
//! a launch target only when it advertises the endpoint that harness speaks,
//! so image and embedding models, and any chat model serving another
//! protocol only, never reach a harness that could not talk to them.

use serde_json::Value;

use crate::verbs::launch::LaunchError;

/// One chat model the launched session may use.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LibraryChatModel {
    pub model_id: String,
    pub context_window_tokens: Option<u32>,
}

/// Keeps advertised order so the picker matches Library listing.
pub fn chat_models_from_models_document(
    models_document: &Value,
    harness_endpoint: &str,
) -> Result<Vec<LibraryChatModel>, LaunchError> {
    let Some(advertised_models) = models_document.get("data").and_then(Value::as_array) else {
        return Err(LaunchError::ModelListUnavailable);
    };
    let mut chat_models = Vec::new();
    for advertised_model in advertised_models {
        let Some(model_id) = advertised_model.get("id").and_then(Value::as_str) else {
            continue;
        };
        if !advertises_endpoint(advertised_model, harness_endpoint) {
            continue;
        }
        chat_models.push(LibraryChatModel {
            model_id: model_id.to_owned(),
            context_window_tokens: advertised_context_window(advertised_model),
        });
    }
    Ok(chat_models)
}

fn advertised_context_window(advertised_model: &Value) -> Option<u32> {
    let context_window = advertised_model.get("context_window")?.as_u64()?;
    u32::try_from(context_window).ok()
}

fn advertises_endpoint(advertised_model: &Value, harness_endpoint: &str) -> bool {
    let Some(supported_endpoints) = advertised_model
        .get("supported_endpoints")
        .and_then(Value::as_array)
    else {
        return false;
    };
    for supported_endpoint in supported_endpoints {
        if supported_endpoint.as_str() == Some(harness_endpoint) {
            return true;
        }
    }
    false
}
