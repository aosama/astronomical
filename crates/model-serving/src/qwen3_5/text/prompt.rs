use astronomical_ipc_protocol::{ChatAssistantToolCall, ChatMessage, ChatToolDefinition};
use serde_json::Value;
use thiserror::Error;

const IM_END: &str = "<|im_end|>";
const IM_START: &str = "<|im_start|>";
const THINK_END: &str = "</think>";
const THINK_START: &str = "<think>";
const TOOL_CALL_END: &str = "</tool_call>";
const TOOL_CALL_START: &str = "<tool_call>";
const TOOL_RESPONSE_END: &str = "</tool_response>";
const TOOL_RESPONSE_START: &str = "<tool_response>";
const TOOLS_END: &str = "</tools>";
const TOOLS_START: &str = "<tools>";

const VISION_END: &str = "<|vision_end|>";
const VISION_START: &str = "<|vision_start|>";
const IMAGE_PAD: &str = "<|image_pad|>";

const TOOL_INSTRUCTIONS: &str = r#"

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>"#;

/// Renders the pinned Qwen3.5 text chat template without executing model-provided Jinja.
#[derive(Debug)]
pub struct Qwen3_5PromptRenderer;

impl Qwen3_5PromptRenderer {
    /// Renders bounded typed chat history and the exact generation prefix for Qwen3.5.
    ///
    /// `image_token_counts_per_user_message` carries one `Vec<usize>` per `ChatMessage::User`
    /// entry, where each element is the number of `<|image_pad|>` tokens for one image
    /// after spatial merge. Empty for text-only user messages.
    pub fn render(
        messages: &[ChatMessage],
        tools: &[ChatToolDefinition],
        enable_thinking: bool,
        image_token_counts_per_user_message: &[Vec<usize>],
    ) -> Result<String, Qwen3_5PromptError> {
        if messages.is_empty() {
            return Err(Qwen3_5PromptError::MissingMessages);
        }

        let mut rendered_prompt = String::new();
        let has_initial_system_message =
            matches!(messages.first(), Some(ChatMessage::System { .. }));
        if tools.is_empty() {
            if let Some(ChatMessage::System { content }) = messages.first() {
                append_chat_message(&mut rendered_prompt, "system", content.trim());
            }
        } else {
            render_tool_system_preamble(&mut rendered_prompt, messages.first(), tools)?;
        }

        let mut user_message_image_index = 0usize;
        for (message_index, message) in messages.iter().enumerate() {
            match message {
                ChatMessage::System { .. } if message_index == 0 && has_initial_system_message => {}
                ChatMessage::System { .. } => {
                    return Err(Qwen3_5PromptError::SystemMessageMustBeFirst);
                }
                ChatMessage::User { content, .. } => {
                    rendered_prompt.push_str(IM_START);
                    rendered_prompt.push_str("user\n");
                    let image_token_counts = image_token_counts_per_user_message
                        .get(user_message_image_index)
                        .map(|counts| counts.as_slice())
                        .unwrap_or(&[]);
                    user_message_image_index += 1;
                    for per_image_token_count in image_token_counts {
                        rendered_prompt.push_str(VISION_START);
                        for _ in 0..*per_image_token_count {
                            rendered_prompt.push_str(IMAGE_PAD);
                        }
                        rendered_prompt.push_str(VISION_END);
                    }
                    append_template_safe_content(&mut rendered_prompt, content.trim());
                    rendered_prompt.push_str(IM_END);
                    rendered_prompt.push('\n');
                }
                ChatMessage::Assistant {
                    content,
                    reasoning_content,
                    tool_calls,
                } => render_assistant_message(
                    &mut rendered_prompt,
                    content.as_deref().unwrap_or_default(),
                    reasoning_content.as_deref().unwrap_or_default(),
                    tool_calls,
                )?,
                ChatMessage::Tool {
                    tool_call_id: _,
                    content,
                } => render_tool_response(
                    &mut rendered_prompt,
                    messages,
                    message_index,
                    content.trim(),
                ),
            }
        }

        rendered_prompt.push_str(IM_START);
        rendered_prompt.push_str("assistant\n");
        if enable_thinking {
            rendered_prompt.push_str(THINK_START);
            rendered_prompt.push('\n');
        } else {
            rendered_prompt.push_str(THINK_START);
            rendered_prompt.push_str("\n\n");
            rendered_prompt.push_str(THINK_END);
            rendered_prompt.push_str("\n\n");
        }
        Ok(rendered_prompt)
    }

    /// Renders server-generated feedback after a malformed model tool call, then reopens assistant generation.
    #[must_use]
    pub fn render_model_visible_correction(correction_text: &str, enable_thinking: bool) -> String {
        let mut rendered_correction = String::new();
        rendered_correction.push_str(IM_END);
        rendered_correction.push('\n');
        rendered_correction.push_str(IM_START);
        rendered_correction.push_str("user\n");
        rendered_correction.push_str(TOOL_RESPONSE_START);
        rendered_correction.push('\n');
        append_template_safe_content(&mut rendered_correction, correction_text.trim());
        rendered_correction.push('\n');
        rendered_correction.push_str(TOOL_RESPONSE_END);
        rendered_correction.push_str(IM_END);
        rendered_correction.push('\n');
        rendered_correction.push_str(IM_START);
        rendered_correction.push_str("assistant\n");
        rendered_correction.push_str(THINK_START);
        rendered_correction.push('\n');
        if !enable_thinking {
            rendered_correction.push('\n');
            rendered_correction.push_str(THINK_END);
            rendered_correction.push_str("\n\n");
        }
        rendered_correction
    }
}

fn render_tool_system_preamble(
    rendered_prompt: &mut String,
    first_message: Option<&ChatMessage>,
    tools: &[ChatToolDefinition],
) -> Result<(), Qwen3_5PromptError> {
    rendered_prompt.push_str(IM_START);
    rendered_prompt.push_str("system\n# Tools\n\nYou have access to the following functions:\n\n");
    rendered_prompt.push_str(TOOLS_START);
    for tool in tools {
        rendered_prompt.push('\n');
        append_template_safe_content(rendered_prompt, &render_tool_definition(tool)?);
    }
    rendered_prompt.push('\n');
    rendered_prompt.push_str(TOOLS_END);
    rendered_prompt.push_str(TOOL_INSTRUCTIONS);
    if let Some(ChatMessage::System { content }) = first_message {
        let trimmed_system_content = content.trim();
        if !trimmed_system_content.is_empty() {
            rendered_prompt.push_str("\n\n");
            append_template_safe_content(rendered_prompt, trimmed_system_content);
        }
    }
    rendered_prompt.push_str(IM_END);
    rendered_prompt.push('\n');
    Ok(())
}

fn render_assistant_message(
    rendered_prompt: &mut String,
    content: &str,
    reasoning_content: &str,
    tool_calls: &[ChatAssistantToolCall],
) -> Result<(), Qwen3_5PromptError> {
    let trimmed_content = content.trim();
    rendered_prompt.push_str(IM_START);
    rendered_prompt.push_str("assistant\n");
    rendered_prompt.push_str(THINK_START);
    rendered_prompt.push('\n');
    append_template_safe_content(rendered_prompt, reasoning_content.trim());
    rendered_prompt.push('\n');
    rendered_prompt.push_str(THINK_END);
    rendered_prompt.push_str("\n\n");
    append_template_safe_content(rendered_prompt, trimmed_content);

    for (tool_call_index, tool_call) in tool_calls.iter().enumerate() {
        if tool_call_index == 0 && !trimmed_content.is_empty() {
            rendered_prompt.push_str("\n\n");
        } else if tool_call_index > 0 {
            rendered_prompt.push('\n');
        }
        render_assistant_tool_call(rendered_prompt, tool_call)?;
    }
    rendered_prompt.push_str(IM_END);
    rendered_prompt.push('\n');
    Ok(())
}

fn render_assistant_tool_call(
    rendered_prompt: &mut String,
    tool_call: &ChatAssistantToolCall,
) -> Result<(), Qwen3_5PromptError> {
    let argument_values = serde_json::from_str::<Value>(&tool_call.function.arguments_json)
        .map_err(|source| Qwen3_5PromptError::InvalidToolArguments {
            function_name: tool_call.function.name.clone(),
            source,
        })?;
    let Value::Object(argument_values) = argument_values else {
        return Err(Qwen3_5PromptError::ToolArgumentsMustBeObject {
            function_name: tool_call.function.name.clone(),
        });
    };

    rendered_prompt.push_str(TOOL_CALL_START);
    rendered_prompt.push('\n');
    rendered_prompt.push_str("<function=");
    append_template_safe_content(rendered_prompt, &tool_call.function.name);
    rendered_prompt.push_str(">\n");
    for (argument_name, argument_value) in argument_values {
        rendered_prompt.push_str("<parameter=");
        append_template_safe_content(rendered_prompt, &argument_name);
        rendered_prompt.push_str(">\n");
        append_template_safe_content(rendered_prompt, &render_parameter_value(&argument_value)?);
        rendered_prompt.push_str("\n</parameter>\n");
    }
    rendered_prompt.push_str("</function>\n");
    rendered_prompt.push_str(TOOL_CALL_END);
    Ok(())
}

fn render_tool_response(
    rendered_prompt: &mut String,
    messages: &[ChatMessage],
    message_index: usize,
    content: &str,
) {
    let has_prior_tool_message = message_index
        .checked_sub(1)
        .and_then(|prior_index| messages.get(prior_index))
        .is_some_and(|prior_message| matches!(prior_message, ChatMessage::Tool { .. }));
    if !has_prior_tool_message {
        rendered_prompt.push_str(IM_START);
        rendered_prompt.push_str("user");
    }
    rendered_prompt.push('\n');
    rendered_prompt.push_str(TOOL_RESPONSE_START);
    rendered_prompt.push('\n');
    append_template_safe_content(rendered_prompt, content);
    rendered_prompt.push('\n');
    rendered_prompt.push_str(TOOL_RESPONSE_END);

    let next_message_is_tool = messages
        .get(message_index.saturating_add(1))
        .is_some_and(|next_message| matches!(next_message, ChatMessage::Tool { .. }));
    if !next_message_is_tool {
        rendered_prompt.push_str(IM_END);
        rendered_prompt.push('\n');
    }
}

fn render_tool_definition(tool: &ChatToolDefinition) -> Result<String, Qwen3_5PromptError> {
    let parameters = serde_json::from_str::<Value>(&tool.parameters_json).map_err(|source| {
        Qwen3_5PromptError::InvalidToolParameters {
            function_name: tool.name.clone(),
            source,
        }
    })?;
    let function_name_json = serde_json::to_string(&tool.name).map_err(|source| {
        Qwen3_5PromptError::SerializeToolDefinition {
            function_name: tool.name.clone(),
            source,
        }
    })?;
    let parameters_json = serde_json::to_string(&parameters).map_err(|source| {
        Qwen3_5PromptError::SerializeToolDefinition {
            function_name: tool.name.clone(),
            source,
        }
    })?;
    let description_json = tool
        .description
        .as_ref()
        .map(serde_json::to_string)
        .transpose()
        .map_err(|source| Qwen3_5PromptError::SerializeToolDefinition {
            function_name: tool.name.clone(),
            source,
        })?;

    let mut rendered_tool =
        format!("{{\"type\":\"function\",\"function\":{{\"name\":{function_name_json}");
    if let Some(description_json) = description_json {
        rendered_tool.push_str(",\"description\":");
        rendered_tool.push_str(&description_json);
    }
    rendered_tool.push_str(",\"parameters\":");
    rendered_tool.push_str(&parameters_json);
    rendered_tool.push_str("}}");
    Ok(rendered_tool)
}

fn render_parameter_value(parameter_value: &Value) -> Result<String, Qwen3_5PromptError> {
    match parameter_value {
        Value::String(string_value) => Ok(string_value.clone()),
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::Array(_) | Value::Object(_) => {
            serde_json::to_string(parameter_value)
                .map_err(Qwen3_5PromptError::SerializeToolArgument)
        }
    }
}

fn append_chat_message(rendered_prompt: &mut String, role: &str, content: &str) {
    rendered_prompt.push_str(IM_START);
    rendered_prompt.push_str(role);
    rendered_prompt.push('\n');
    append_template_safe_content(rendered_prompt, content);
    rendered_prompt.push_str(IM_END);
    rendered_prompt.push('\n');
}

fn append_template_safe_content(rendered_prompt: &mut String, untrusted_content: &str) {
    let mut remaining_content = untrusted_content;
    while let Some(marker_offset) = remaining_content.find('<') {
        rendered_prompt.push_str(&remaining_content[..marker_offset]);
        let marker_suffix = &remaining_content[marker_offset + 1..];
        if starts_reserved_template_marker(marker_suffix) {
            rendered_prompt.push_str("&lt;");
        } else {
            rendered_prompt.push('<');
        }
        remaining_content = marker_suffix;
    }
    rendered_prompt.push_str(remaining_content);
}

fn starts_reserved_template_marker(marker_suffix: &str) -> bool {
    marker_suffix.starts_with('|')
        || marker_suffix.starts_with("think>")
        || marker_suffix.starts_with("/think>")
        || marker_suffix.starts_with("tool_call>")
        || marker_suffix.starts_with("/tool_call>")
        || marker_suffix.starts_with("tool_response>")
        || marker_suffix.starts_with("/tool_response>")
        || marker_suffix.starts_with("tools>")
        || marker_suffix.starts_with("/tools>")
        || marker_suffix.starts_with("function=")
        || marker_suffix.starts_with("/function>")
        || marker_suffix.starts_with("parameter=")
        || marker_suffix.starts_with("/parameter>")
}

/// A typed error while reproducing the reviewed Qwen3.5 text template.
#[derive(Debug, Error)]
pub enum Qwen3_5PromptError {
    /// At least one user query or tool-response sequence is required by the template.
    #[error("Qwen3.5 prompt rendering requires at least one message")]
    MissingMessages,
    /// Only the first chat message may use the system role.
    #[error("Qwen3.5 system messages must be the first conversation message")]
    SystemMessageMustBeFirst,
    /// Historical assistant tool arguments were not valid JSON.
    #[error("assistant tool arguments for '{function_name}' are not valid JSON")]
    InvalidToolArguments {
        /// Function name that supplied invalid JSON.
        function_name: String,
        /// JSON parser failure.
        #[source]
        source: serde_json::Error,
    },
    /// Historical assistant tool arguments were valid JSON but not an object.
    #[error("assistant tool arguments for '{function_name}' must be a JSON object")]
    ToolArgumentsMustBeObject {
        /// Function name that supplied a non-object argument document.
        function_name: String,
    },
    /// A declared tool schema was not valid JSON.
    #[error("tool parameters for '{function_name}' are not valid JSON")]
    InvalidToolParameters {
        /// Function name that supplied invalid JSON.
        function_name: String,
        /// JSON parser failure.
        #[source]
        source: serde_json::Error,
    },
    /// A reviewed tool definition could not serialize.
    #[error("tool definition for '{function_name}' could not serialize")]
    SerializeToolDefinition {
        /// Function name that failed serialization.
        function_name: String,
        /// JSON serialization failure.
        #[source]
        source: serde_json::Error,
    },
    /// One tool argument could not serialize to its template representation.
    #[error("tool argument could not serialize")]
    SerializeToolArgument(#[source] serde_json::Error),
}
