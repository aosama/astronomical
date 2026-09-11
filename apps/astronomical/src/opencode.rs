//! Builds process-scoped OpenCode config.
//!
//! Uses OpenCode's documented `OPENCODE_CONFIG_CONTENT` overlay
//! (https://opencode.ai/docs/config/) so this session does not write
//! `~/.config/opencode`. Provider shape follows the public OpenAI-compatible
//! custom-provider pattern used by Ollama launch and omlx integrations; this
//! file does not copy their source.

use std::net::SocketAddr;

use serde_json::{Map, json};

use crate::models::LibraryChatModel;

pub const OPENCODE_CONFIG_CONTENT_VARIABLE: &str = "OPENCODE_CONFIG_CONTENT";
const OPENCODE_PROVIDER_ID: &str = "astronomical";

pub fn opencode_config_content(
    bind_address: SocketAddr,
    chat_model: &LibraryChatModel,
) -> Result<String, serde_json::Error> {
    let openai_base_url = format!("http://{bind_address}/v1");
    let provider_model_id = format!("{OPENCODE_PROVIDER_ID}/{}", chat_model.model_id);
    let mut provider_models = Map::new();
    provider_models.insert(
        chat_model.model_id.clone(),
        json!({ "name": chat_model.model_id }),
    );
    let config_document = json!({
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            OPENCODE_PROVIDER_ID: {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Astronomical",
                "options": {
                    "baseURL": openai_base_url,
                },
                "models": provider_models,
            }
        },
        "model": provider_model_id,
    });
    serde_json::to_string(&config_document)
}
