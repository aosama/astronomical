//! One discovered chat model for tests, built in one place so every test
//! that needs a model on this Mac describes it the same way.

use std::path::PathBuf;

use config::{ChatModelCapabilities, DiscoveredModel, ModelCapabilities, ModelFamily};

/// A Qwen 3.5 chat model with ordinary limits, at `model_directory`.
pub(crate) fn discovered_chat_model(
    model_id: &str,
    provider_model_id: Option<&str>,
    revision: &str,
    model_directory: &str,
) -> DiscoveredModel {
    DiscoveredModel {
        model_id: model_id.to_owned(),
        provider_model_id: provider_model_id.map(str::to_owned),
        model_family: ModelFamily::Qwen3_5,
        revision: revision.to_owned(),
        model_directory: PathBuf::from(model_directory),
        capabilities: ModelCapabilities::Chat(ChatModelCapabilities {
            context_window: 65_536,
            max_input_tokens: 65_535,
            max_output_tokens: 20_480,
            supports_vision: false,
            supports_reasoning: true,
            supports_tool_calls: true,
        }),
        license: None,
        model_size_bytes: 1_000,
    }
}
