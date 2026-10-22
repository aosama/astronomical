use serde::Deserialize;
use serde_json::Value;

use crate::image_input;
use crate::openai_request_budget::{
    MAX_OPENAI_IMAGE_PARTS_PER_REQUEST, MAX_OPENAI_INLINE_MEDIA_BYTES,
};
use crate::{OpenAiImageInput, OpenAiResponsesValidationError};

/// Stateless input accepted by the local Responses endpoint.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum OpenAiResponseInput {
    /// One compact user input string.
    Text(String),
    /// Ordered Responses input and prior-output items.
    Items(Vec<OpenAiResponseInputItem>),
}

/// One input item supported by the local Responses endpoint.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum OpenAiResponseInputItem {
    Message(OpenAiResponseMessageInput),
    Reasoning(OpenAiResponseReasoningInput),
    FunctionCall(OpenAiResponseFunctionCallInput),
    FunctionCallOutput(OpenAiResponseFunctionCallOutputInput),
    Unsupported(Value),
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OpenAiResponseMessageInput {
    #[serde(default, rename = "type")]
    message_type: Option<OpenAiResponseMessageType>,
    #[serde(default)]
    id: Option<String>,
    role: OpenAiResponseMessageRole,
    content: OpenAiResponseMessageContent,
    #[serde(default)]
    status: Option<OpenAiResponseItemStatus>,
    #[serde(default)]
    phase: Option<OpenAiResponseAssistantPhase>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum OpenAiResponseMessageType {
    Message,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum OpenAiResponseMessageRole {
    User,
    System,
    Developer,
    Assistant,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
enum OpenAiResponseMessageContent {
    Text(String),
    Parts(Vec<OpenAiResponseContentPart>),
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum OpenAiResponseContentPart {
    InputText {
        text: String,
    },
    OutputText {
        text: String,
        #[serde(default)]
        annotations: Vec<Value>,
        #[serde(default)]
        logprobs: Vec<Value>,
    },
    InputImage {
        image_url: String,
        #[serde(default)]
        detail: Option<String>,
    },
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OpenAiResponseReasoningInput {
    #[serde(rename = "type")]
    reasoning_type: OpenAiResponseReasoningType,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    summary: Vec<OpenAiResponseReasoningSummaryInput>,
    #[serde(default)]
    content: Vec<OpenAiResponseReasoningContent>,
    #[serde(default)]
    encrypted_content: Option<String>,
    #[serde(default)]
    status: Option<OpenAiResponseItemStatus>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum OpenAiResponseReasoningType {
    Reasoning,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum OpenAiResponseReasoningContent {
    ReasoningText { text: String },
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum OpenAiResponseReasoningSummaryInput {
    SummaryText { text: String },
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OpenAiResponseFunctionCallInput {
    #[serde(rename = "type")]
    function_call_type: OpenAiResponseFunctionCallType,
    #[serde(default)]
    id: Option<String>,
    call_id: String,
    name: String,
    arguments: String,
    #[serde(default)]
    status: Option<OpenAiResponseItemStatus>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum OpenAiResponseFunctionCallType {
    FunctionCall,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OpenAiResponseFunctionCallOutputInput {
    #[serde(rename = "type")]
    function_call_output_type: OpenAiResponseFunctionCallOutputType,
    #[serde(default)]
    id: Option<String>,
    call_id: String,
    output: String,
    #[serde(default)]
    status: Option<OpenAiResponseItemStatus>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum OpenAiResponseFunctionCallOutputType {
    FunctionCallOutput,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum OpenAiResponseItemStatus {
    InProgress,
    Completed,
    Incomplete,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum OpenAiResponseAssistantPhase {
    Commentary,
    FinalAnswer,
}

/// Validated Responses input ready for daemon translation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OpenAiResponseInputParts {
    Text(String),
    Items(Vec<OpenAiResponseInputItemParts>),
}

/// One validated input item ready for daemon translation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OpenAiResponseInputItemParts {
    SystemMessage {
        content: String,
    },
    DeveloperMessage {
        content: String,
    },
    UserMessage {
        content: String,
        images: Vec<OpenAiImageInput>,
    },
    AssistantMessage {
        content: String,
    },
    Reasoning {
        content: String,
    },
    FunctionCall {
        call_id: String,
        name: String,
        arguments_json: String,
    },
    FunctionCallOutput {
        call_id: String,
        output: String,
    },
}

impl OpenAiResponseInputItemParts {
    #[must_use]
    pub const fn kind_name(&self) -> &'static str {
        match self {
            Self::SystemMessage { .. } => "system_message",
            Self::DeveloperMessage { .. } => "developer_message",
            Self::UserMessage { .. } => "user_message",
            Self::AssistantMessage { .. } => "assistant_message",
            Self::Reasoning { .. } => "reasoning",
            Self::FunctionCall { .. } => "function_call",
            Self::FunctionCallOutput { .. } => "function_call_output",
        }
    }
}

impl OpenAiResponseInput {
    pub(super) fn into_parts(
        self,
    ) -> Result<OpenAiResponseInputParts, OpenAiResponsesValidationError> {
        match self {
            Self::Text(input_text) => Ok(OpenAiResponseInputParts::Text(input_text)),
            Self::Items(input_items) => {
                if input_items.is_empty() {
                    return Err(OpenAiResponsesValidationError::EmptyInputItems);
                }
                // The image-count budget is checked before any base64 decode
                // work, mirroring the chat endpoint's rejection ordering.
                validate_image_part_count(
                    input_items
                        .iter()
                        .map(OpenAiResponseInputItem::image_url_part_count)
                        .sum(),
                )?;
                // Inline media accumulates across every user message so the
                // portable media budget is per request, not per message.
                let mut decoded_inline_media_bytes = 0usize;
                let mut input_item_parts = Vec::with_capacity(input_items.len());
                for input_item in input_items {
                    let item_parts = input_item.into_parts()?;
                    if let OpenAiResponseInputItemParts::UserMessage { images, .. } = &item_parts {
                        for decoded_image in images {
                            decoded_inline_media_bytes += decoded_image.decoded_bytes().len();
                        }
                    }
                    input_item_parts.push(item_parts);
                }
                validate_inline_media_total(decoded_inline_media_bytes)?;
                Ok(OpenAiResponseInputParts::Items(input_item_parts))
            }
        }
    }
}

fn validate_image_part_count(
    image_part_count: usize,
) -> Result<(), OpenAiResponsesValidationError> {
    if image_part_count > MAX_OPENAI_IMAGE_PARTS_PER_REQUEST {
        return Err(
            OpenAiResponsesValidationError::ImagePartCountExceedsBudget {
                actual_image_part_count: image_part_count,
                maximum_image_part_count: MAX_OPENAI_IMAGE_PARTS_PER_REQUEST,
                excess_image_part_count: image_part_count - MAX_OPENAI_IMAGE_PARTS_PER_REQUEST,
            },
        );
    }
    Ok(())
}

fn validate_inline_media_total(
    decoded_inline_media_bytes: usize,
) -> Result<(), OpenAiResponsesValidationError> {
    if decoded_inline_media_bytes > MAX_OPENAI_INLINE_MEDIA_BYTES {
        return Err(OpenAiResponsesValidationError::InlineMediaExceedsBudget {
            actual_inline_media_bytes: decoded_inline_media_bytes,
            maximum_inline_media_bytes: MAX_OPENAI_INLINE_MEDIA_BYTES,
            excess_inline_media_bytes: decoded_inline_media_bytes - MAX_OPENAI_INLINE_MEDIA_BYTES,
        });
    }
    Ok(())
}

impl OpenAiResponseInputItem {
    /// Counts the image parts this input item carries.
    ///
    /// Only user message items can structurally carry image parts; the count
    /// feeds the request-level portable image-count budget.
    fn image_url_part_count(&self) -> usize {
        let Self::Message(message_input) = self else {
            return 0;
        };
        match &message_input.content {
            OpenAiResponseMessageContent::Text(_) => 0,
            OpenAiResponseMessageContent::Parts(content_parts) => content_parts
                .iter()
                .filter(|content_part| {
                    matches!(content_part, OpenAiResponseContentPart::InputImage { .. })
                })
                .count(),
        }
    }

    fn into_parts(self) -> Result<OpenAiResponseInputItemParts, OpenAiResponsesValidationError> {
        match self {
            Self::Message(message) => message.into_parts(),
            Self::Reasoning(reasoning) => reasoning.into_parts(),
            Self::FunctionCall(function_call) => Ok(OpenAiResponseInputItemParts::FunctionCall {
                call_id: function_call.call_id,
                name: function_call.name,
                arguments_json: function_call.arguments,
            }),
            Self::FunctionCallOutput(function_call_output) => {
                Ok(OpenAiResponseInputItemParts::FunctionCallOutput {
                    call_id: function_call_output.call_id,
                    output: function_call_output.output,
                })
            }
            Self::Unsupported(unsupported_input_item) => {
                let input_item_type = unsupported_input_item
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_owned();
                Err(OpenAiResponsesValidationError::UnsupportedInputItem { input_item_type })
            }
        }
    }
}

impl OpenAiResponseMessageInput {
    fn into_parts(self) -> Result<OpenAiResponseInputItemParts, OpenAiResponsesValidationError> {
        match self.role {
            OpenAiResponseMessageRole::User => {
                let (content, images) = self.content.into_user_content()?;
                Ok(OpenAiResponseInputItemParts::UserMessage { content, images })
            }
            OpenAiResponseMessageRole::System => Ok(OpenAiResponseInputItemParts::SystemMessage {
                content: self.content.into_text()?,
            }),
            OpenAiResponseMessageRole::Developer => {
                Ok(OpenAiResponseInputItemParts::DeveloperMessage {
                    content: self.content.into_text()?,
                })
            }
            OpenAiResponseMessageRole::Assistant => {
                Ok(OpenAiResponseInputItemParts::AssistantMessage {
                    content: self.content.into_text()?,
                })
            }
        }
    }
}

impl OpenAiResponseMessageContent {
    fn into_text(self) -> Result<String, OpenAiResponsesValidationError> {
        match self {
            Self::Text(content) => Ok(content),
            Self::Parts(content_parts) => {
                if content_parts.is_empty() {
                    return Err(OpenAiResponsesValidationError::EmptyContentParts);
                }
                let mut combined_text = String::new();
                for content_part in content_parts {
                    match content_part {
                        OpenAiResponseContentPart::InputText { text }
                        | OpenAiResponseContentPart::OutputText { text, .. } => {
                            combined_text.push_str(&text);
                        }
                        OpenAiResponseContentPart::InputImage { .. } => {
                            return Err(
                                OpenAiResponsesValidationError::ImageInputOutsideUserMessage,
                            );
                        }
                    }
                }
                Ok(combined_text)
            }
        }
    }

    fn into_user_content(
        self,
    ) -> Result<(String, Vec<OpenAiImageInput>), OpenAiResponsesValidationError> {
        match self {
            Self::Text(content) => Ok((content, Vec::new())),
            Self::Parts(content_parts) => {
                if content_parts.is_empty() {
                    return Err(OpenAiResponsesValidationError::EmptyContentParts);
                }
                let mut combined_text = String::new();
                let mut decoded_images = Vec::new();
                for content_part in content_parts {
                    match content_part {
                        OpenAiResponseContentPart::InputText { text }
                        | OpenAiResponseContentPart::OutputText { text, .. } => {
                            combined_text.push_str(&text);
                        }
                        OpenAiResponseContentPart::InputImage { image_url, .. } => {
                            image_input::validate_image_url_scheme(&image_url)
                                .map_err(OpenAiResponsesValidationError::ImageInput)?;
                            decoded_images.push(
                                image_input::decode_image_url(&image_url)
                                    .map_err(OpenAiResponsesValidationError::ImageInput)?,
                            );
                        }
                    }
                }
                Ok((combined_text, decoded_images))
            }
        }
    }
}

impl OpenAiResponseReasoningInput {
    fn into_parts(self) -> Result<OpenAiResponseInputItemParts, OpenAiResponsesValidationError> {
        if self.encrypted_content.is_some() {
            return Err(OpenAiResponsesValidationError::UnsupportedReasoningReplay);
        }
        let combined_reasoning_text = self
            .summary
            .into_iter()
            .map(|reasoning_summary| match reasoning_summary {
                OpenAiResponseReasoningSummaryInput::SummaryText { text } => text,
            })
            .chain(
                self.content
                    .into_iter()
                    .map(|reasoning_content| match reasoning_content {
                        OpenAiResponseReasoningContent::ReasoningText { text } => text,
                    }),
            )
            .collect::<String>();
        Ok(OpenAiResponseInputItemParts::Reasoning {
            content: combined_reasoning_text,
        })
    }
}
