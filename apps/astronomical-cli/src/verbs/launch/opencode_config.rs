//! Builds process-scoped OpenCode config.
//!
//! Uses OpenCode's documented `OPENCODE_CONFIG_CONTENT` overlay
//! (https://opencode.ai/docs/config/) so this session does not write
//! `~/.config/opencode`. The provider follows OpenCode 2's `providers` shape
//! for an OpenAI-compatible runtime (https://opencode.ai/docs/providers/):
//! `package`, `settings` and per-model `limit`, which replaced the earlier
//! `provider` / `npm` / `options` shape. The provider id is the one the
//! OpenCode plugin in `integrations/opencode` registers, so a session started
//! here and the plugin name the same models the same way, and OpenCode lists
//! each model once when both are present.

use std::net::SocketAddr;

use serde_json::{Map, Value, json};

use super::chat_models::InstalledChatModel;

pub const OPENCODE_CONFIG_CONTENT_VARIABLE: &str = "OPENCODE_CONFIG_CONTENT";
const OPENCODE_PROVIDER_ID: &str = "astronomical";
const OPENCODE_OPENAI_COMPATIBLE_PACKAGE: &str = "@opencode/ai/providers/openai-compatible";

/// A cold model load can hold back the first response headers for longer than
/// OpenCode's default wait, so the session allows ten minutes, the same as the
/// OpenCode plugin.
const OPENCODE_INFERENCE_TIMEOUT_MILLISECONDS: u64 = 600_000;

/// The model entry: its name, and its real context window when the app
/// advertises one, so OpenCode plans compaction around the model's actual
/// capacity instead of a default.
fn opencode_model_entry(chat_model: &InstalledChatModel) -> Value {
    let mut model_entry = Map::new();
    model_entry.insert("name".to_owned(), json!(chat_model.model_id));
    if let Some(context_window_tokens) = chat_model.context_window_tokens {
        model_entry.insert(
            "limit".to_owned(),
            json!({ "context": context_window_tokens }),
        );
    }
    Value::Object(model_entry)
}

pub fn opencode_config_content(
    bind_address: SocketAddr,
    chat_model: &InstalledChatModel,
) -> Result<String, serde_json::Error> {
    let openai_base_url = format!("http://{bind_address}/v1");
    let provider_model_id = format!("{OPENCODE_PROVIDER_ID}/{}", chat_model.model_id);
    let mut provider_models = Map::new();
    provider_models.insert(
        chat_model.model_id.clone(),
        opencode_model_entry(chat_model),
    );
    let config_document = json!({
        "$schema": "https://opencode.ai/config.json",
        "providers": {
            OPENCODE_PROVIDER_ID: {
                "name": "Astronomical",
                "package": OPENCODE_OPENAI_COMPATIBLE_PACKAGE,
                "settings": {
                    "baseURL": openai_base_url,
                    "headerTimeout": OPENCODE_INFERENCE_TIMEOUT_MILLISECONDS,
                    "timeout": OPENCODE_INFERENCE_TIMEOUT_MILLISECONDS,
                },
                "models": provider_models,
            }
        },
        "model": provider_model_id,
    });
    serde_json::to_string(&config_document)
}

#[cfg(test)]
mod opencode_config_tests {
    use super::opencode_config_content;
    use crate::verbs::launch::chat_models::InstalledChatModel;

    fn config_document(context_window_tokens: Option<u32>) -> serde_json::Value {
        let chat_model = InstalledChatModel {
            model_id: "library-chat-model".to_owned(),
            context_window_tokens,
        };
        let config_text =
            opencode_config_content("127.0.0.1:6733".parse().expect("address"), &chat_model)
                .expect("config serializes");
        serde_json::from_str(&config_text).expect("config is JSON")
    }

    #[test]
    fn should_write_the_current_providers_shape_with_the_context_window() {
        let config = config_document(Some(262_144));
        let provider = &config["providers"]["astronomical"];
        assert_eq!(config["model"], "astronomical/library-chat-model");
        assert_eq!(provider["settings"]["baseURL"], "http://127.0.0.1:6733/v1");
        assert_eq!(provider["settings"]["headerTimeout"], 600_000);
        assert_eq!(
            provider["models"]["library-chat-model"]["limit"]["context"],
            262_144
        );
        assert!(config.get("provider").is_none());
    }

    #[test]
    fn should_leave_the_limit_out_when_the_app_advertises_no_context_window() {
        let config = config_document(None);
        let model_entry = &config["providers"]["astronomical"]["models"]["library-chat-model"];
        assert_eq!(model_entry["name"], "library-chat-model");
        assert!(model_entry.get("limit").is_none());
    }
}
