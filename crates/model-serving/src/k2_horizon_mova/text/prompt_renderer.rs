//! Family-owned IFM chat rendering.
//!
//! The shipped Jinja file is provenance only. This renderer covers the IFM
//! user/assistant/tool contract plus JSON tool presentation and XML tool-call
//! instructions, which is the template default for `tool_call_format`.

use astronomical_ipc_protocol::{
    ChatAssistantToolCall, ChatMessage, ChatToolChoice, ChatToolDefinition,
};

const BEGIN_OF_TEXT: &str = "<|ifm|begin_of_text|>";
const IM_START: &str = "<|ifm|im_start|>";
const IM_END: &str = "<|ifm|im_end|>";
const THINK_OPEN: &str = "<ifm|think>";
const THINK_CLOSE: &str = "</ifm|think>";
const XML_CALL_INSTRUCTIONS: &str = "Close thinking with </ifm|think> before any tool call or user-visible answer. Wrap all tool calls in a single <ifm|tool_calls></ifm|tool_calls> block immediately after that close. Never describe a tool call in prose. For each call, write the function name at the start of <ifm|tool_call>, followed by paired <ifm|arg_key> and <ifm|arg_value> tags for each argument:\n\n<ifm|tool_calls>\n<ifm|tool_call>$FUNCTION_NAME\n<ifm|arg_key>$PARAMETER_NAME</ifm|arg_key>\n<ifm|arg_value>$PARAMETER_VALUE</ifm|arg_value>\n...\n</ifm|tool_call>\n</ifm|tool_calls>\n\nString and scalar parameters should be written as plain text. Array and object parameters should be written as JSON literals.";

/// Renders IFM chat turns into one prompt string.
#[derive(Clone, Debug, Default)]
pub struct K2HorizonMoVAPromptRenderer;

impl K2HorizonMoVAPromptRenderer {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Renders history, optional tool catalog, and the assistant think channel.
    pub fn render(
        &self,
        messages: &[ChatMessage],
        tools: &[ChatToolDefinition],
        tool_choice: &ChatToolChoice,
    ) -> String {
        let mut prompt = String::from(BEGIN_OF_TEXT);
        let include_tools = !tools.is_empty() && !matches!(tool_choice, ChatToolChoice::None);
        let mut skip_first_system = false;
        if include_tools {
            let system_content = match messages.first() {
                Some(ChatMessage::System { content }) => {
                    skip_first_system = true;
                    content.as_str()
                }
                _ => "",
            };
            prompt.push_str(IM_START);
            prompt.push_str("system\n# Tools\nYou may call one or more tools to assist with the user query.\n\nAvailable tools are:\n\n<ifm|tools>\n");
            prompt.push_str(&render_tools_json(tools));
            prompt.push_str("\n</ifm|tools>\n\nWhen calling tools, you MUST follow the tool-call format below:\n\n");
            prompt.push_str(XML_CALL_INSTRUCTIONS);
            if !system_content.is_empty() {
                prompt.push_str("\n\n");
                prompt.push_str(system_content);
            }
            prompt.push_str(IM_END);
        }
        for (message_index, message) in messages.iter().enumerate() {
            if skip_first_system && message_index == 0 {
                continue;
            }
            match message {
                ChatMessage::System { content } => push_turn(&mut prompt, "system", content),
                ChatMessage::User { content, .. } => push_turn(&mut prompt, "user", content),
                ChatMessage::Assistant {
                    content,
                    reasoning_content,
                    tool_calls,
                } => {
                    prompt.push_str(IM_START);
                    prompt.push_str("assistant\n");
                    prompt.push_str(THINK_OPEN);
                    prompt.push('\n');
                    if let Some(reasoning_content) = reasoning_content {
                        prompt.push_str(reasoning_content);
                    }
                    prompt.push_str(THINK_CLOSE);
                    if let Some(content) = content {
                        prompt.push_str(content);
                    }
                    if !tool_calls.is_empty() {
                        prompt.push_str(&render_history_tool_calls(tool_calls));
                    }
                    prompt.push_str(IM_END);
                }
                ChatMessage::Tool { content, .. } => push_turn(&mut prompt, "tool", content),
            }
        }
        prompt.push_str(IM_START);
        prompt.push_str("assistant\n");
        prompt.push_str(THINK_OPEN);
        prompt.push('\n');
        prompt
    }
}

fn render_tools_json(tools: &[ChatToolDefinition]) -> String {
    let rendered = tools
        .iter()
        .map(|tool| {
            let parameters = serde_json::from_str::<serde_json::Value>(&tool.parameters_json)
                .unwrap_or(serde_json::json!({}));
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description.clone().unwrap_or_default(),
                    "parameters": parameters,
                }
            })
        })
        .collect::<Vec<_>>();
    serde_json::to_string(&rendered).unwrap_or_else(|_| "[]".to_owned())
}

fn render_history_tool_calls(tool_calls: &[ChatAssistantToolCall]) -> String {
    let mut rendered = String::from("\n<ifm|tool_calls>");
    for tool_call in tool_calls {
        rendered.push_str("\n<ifm|tool_call>");
        rendered.push_str(&tool_call.function.name);
        rendered.push('\n');
        if let Ok(serde_json::Value::Object(arguments)) =
            serde_json::from_str::<serde_json::Value>(&tool_call.function.arguments_json)
        {
            for (key, value) in arguments {
                rendered.push_str("<ifm|arg_key>");
                rendered.push_str(&key);
                rendered.push_str("</ifm|arg_key>\n<ifm|arg_value>");
                match value {
                    serde_json::Value::String(text) => rendered.push_str(&text),
                    other => rendered.push_str(&other.to_string()),
                }
                rendered.push_str("</ifm|arg_value>\n");
            }
        }
        rendered.push_str("</ifm|tool_call>");
    }
    rendered.push_str("\n</ifm|tool_calls>");
    rendered
}

fn push_turn(prompt: &mut String, role: &str, content: &str) {
    prompt.push_str(IM_START);
    prompt.push_str(role);
    prompt.push('\n');
    prompt.push_str(content);
    prompt.push_str(IM_END);
}
