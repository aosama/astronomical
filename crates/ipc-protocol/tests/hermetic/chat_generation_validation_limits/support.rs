use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, ChatToolDefinition,
    RequestId,
};

pub(super) fn command_with_tool_schema(
    request_id: u64,
    parameters_json: String,
) -> ChatGenerationCommand {
    ChatGenerationCommand {
        request_id: RequestId::new(request_id),
        model: "mlx-community/Ornith-1.0-35B-OptiQ-4bit".to_owned(),
        messages: user_messages(),
        tools: vec![ChatToolDefinition {
            name: "glob".to_owned(),
            description: None,
            parameters_json,
        }],
        tool_choice: ChatToolChoice::Auto,
        settings: standard_settings(),
        qwen_thinking_channel_seed: None,
    }
}
pub(super) fn standard_settings() -> ChatGenerationSettings {
    ChatGenerationSettings {
        max_output_tokens: 512,
        temperature_thousandths: Some(600),
        top_p_thousandths: Some(950),
        seed: Some(7),
        thinking_budget: None,
    }
}
pub(super) fn user_messages() -> Vec<ChatMessage> {
    vec![ChatMessage::User {
        content: "Inspect the repository.".to_owned(),
        images: Vec::new(),
    }]
}
