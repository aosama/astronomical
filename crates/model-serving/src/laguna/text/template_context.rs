use astronomical_ipc_protocol::{ChatMessage, ChatToolDefinition};
use serde::Serialize;
use serde_json::Value;

use super::prompt_renderer::{LagunaPromptRendererError, parse_strict_tool_json};

/// Typed root passed to the strict artifact template for every render.
#[derive(Debug, Serialize)]
pub(super) struct LagunaTemplateContext {
    messages: Vec<LagunaTemplateMessage>,
    tools: Vec<LagunaTemplateTool>,
    add_generation_prompt: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    enable_thinking: Option<bool>,
    bos_token: String,
}

impl LagunaTemplateContext {
    pub(super) fn from_chat(
        messages: &[ChatMessage],
        tools: &[ChatToolDefinition],
        enable_thinking: Option<bool>,
        bos_token: &str,
    ) -> Result<Self, LagunaPromptRendererError> {
        let messages = messages
            .iter()
            .map(LagunaTemplateMessage::from_chat_message)
            .collect::<Result<Vec<_>, _>>()?;
        let tools = tools
            .iter()
            .map(LagunaTemplateTool::from_definition)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            messages,
            tools,
            add_generation_prompt: true,
            enable_thinking,
            bos_token: bos_token.to_owned(),
        })
    }

    pub(super) fn without_generation_prompt(mut self) -> Self {
        self.add_generation_prompt = false;
        self
    }
}

/// Uniform message fields prevent strict templates from observing accidental undefined values.
#[derive(Debug, Serialize)]
struct LagunaTemplateMessage {
    role: &'static str,
    content: String,
    reasoning: String,
    reasoning_content: String,
    tool_calls: Vec<LagunaTemplateToolCall>,
    tool_call_id: String,
}

impl LagunaTemplateMessage {
    fn from_chat_message(message: &ChatMessage) -> Result<Self, LagunaPromptRendererError> {
        match message {
            ChatMessage::System { content } => Ok(Self::plain("system", content)),
            ChatMessage::User { content, .. } => Ok(Self::plain("user", content)),
            ChatMessage::Tool {
                tool_call_id,
                content,
            } => Ok(Self {
                tool_call_id: tool_call_id.clone(),
                ..Self::plain("tool", content)
            }),
            ChatMessage::Assistant {
                content,
                reasoning_content,
                tool_calls,
            } => {
                let reasoning_content = reasoning_content.clone().unwrap_or_default();
                let tool_calls = tool_calls
                    .iter()
                    .map(|tool_call| {
                        let arguments = parse_strict_tool_json(
                            &tool_call.function.name,
                            tool_call.function.arguments_json.as_bytes(),
                        )?;
                        if !arguments.is_object() {
                            return Err(LagunaPromptRendererError::ToolArgumentsMustBeObject {
                                function_name: tool_call.function.name.clone(),
                            });
                        }
                        Ok(LagunaTemplateToolCall {
                            id: tool_call.id.clone(),
                            r#type: "function",
                            function: LagunaTemplateToolCallFunction {
                                name: tool_call.function.name.clone(),
                                arguments,
                            },
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Self {
                    role: "assistant",
                    content: content.clone().unwrap_or_default(),
                    reasoning: reasoning_content.clone(),
                    reasoning_content,
                    tool_calls,
                    tool_call_id: String::new(),
                })
            }
        }
    }

    fn plain(role: &'static str, content: &str) -> Self {
        Self {
            role,
            content: content.to_owned(),
            reasoning: String::new(),
            reasoning_content: String::new(),
            tool_calls: Vec::new(),
            tool_call_id: String::new(),
        }
    }
}

#[derive(Debug, Serialize)]
struct LagunaTemplateToolCall {
    id: String,
    r#type: &'static str,
    function: LagunaTemplateToolCallFunction,
}

#[derive(Debug, Serialize)]
struct LagunaTemplateToolCallFunction {
    name: String,
    arguments: Value,
}

/// OpenAI-style function declaration consumed directly by the artifact template.
#[derive(Debug, Serialize)]
struct LagunaTemplateTool {
    function: LagunaTemplateToolFunction,
    r#type: &'static str,
}

impl LagunaTemplateTool {
    fn from_definition(
        tool_definition: &ChatToolDefinition,
    ) -> Result<Self, LagunaPromptRendererError> {
        let parameters = parse_strict_tool_json(
            &tool_definition.name,
            tool_definition.parameters_json.as_bytes(),
        )?;
        if !parameters.is_object() {
            return Err(LagunaPromptRendererError::ToolArgumentsMustBeObject {
                function_name: tool_definition.name.clone(),
            });
        }
        Ok(Self {
            function: LagunaTemplateToolFunction {
                description: tool_definition.description.clone(),
                name: tool_definition.name.clone(),
                parameters,
            },
            r#type: "function",
        })
    }
}

#[derive(Debug, Serialize)]
struct LagunaTemplateToolFunction {
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    name: String,
    parameters: Value,
}
