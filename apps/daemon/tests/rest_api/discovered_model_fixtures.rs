//! Discovered-model fixtures shared by the endpoints that must refuse a
//! model of the wrong kind.

use std::path::PathBuf;

use config::{DiscoveredModel, ModelCapabilities};

/// A chat model with no other capability, used to prove that embeddings,
/// image, and decision endpoints reject chat-only models.
pub(super) fn chat_only_model() -> DiscoveredModel {
    DiscoveredModel {
        model_id: "chat-only-model".to_owned(),
        provider_model_id: None,
        model_family: config::QWEN3_5_MODEL_FAMILY,
        revision: "fixture-revision".to_owned(),
        model_directory: PathBuf::from("fixtures/models/chat-only"),
        capabilities: ModelCapabilities::Chat(config::ChatModelCapabilities {
            context_window: 2_048,
            max_input_tokens: 1_024,
            max_output_tokens: 128,
            supports_vision: false,
            supports_reasoning: true,
            supports_tool_calls: true,
        }),
        license: None,
        model_size_bytes: 1,
    }
}
