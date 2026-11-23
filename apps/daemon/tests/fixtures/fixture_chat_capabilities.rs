//! The chat surface a scripted worker advertises for a swapped-in
//! autoregressive model, read from the configuration the daemon sent. The
//! idle and memory-demotion worker fixtures both advertise it.

use ipc_protocol::{ChatModelCapabilities, WorkerAutoregressiveModelConfiguration};

pub(super) fn swapped_chat_capabilities(
    autoregressive_configuration: &WorkerAutoregressiveModelConfiguration,
) -> ChatModelCapabilities {
    ChatModelCapabilities {
        supports_reasoning: true,
        supports_tool_calls: true,
        has_vision: false,
        max_input_tokens: autoregressive_configuration
            .maximum_context_tokens
            .saturating_sub(1),
        max_output_tokens: autoregressive_configuration.maximum_output_tokens,
        context_window: autoregressive_configuration.maximum_context_tokens,
    }
}
