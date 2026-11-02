//! Anthropic Messages input content: the message, system-prompt, and content
//! block shapes a Messages client sends, and their validation into
//! protocol-neutral parts the daemon translates onto the worker command.
//!
//! Block-level fields this server does not act on (`cache_control`,
//! `citations`, a thinking block's `signature`) are absorbed rather than
//! rejected: Anthropic clients attach them to nearly every block, and they
//! carry no instruction the local model could honor or violate.

use serde::Deserialize;
use serde_json::Value;

use crate::image_input::decode_base64_image_payload;
use crate::request_budget::{MAX_IMAGE_PARTS_PER_REQUEST, MAX_INLINE_MEDIA_BYTES};
use crate::{AnthropicMessagesValidationError, DecodedImageInput};

/// One conversation turn as an Anthropic client sends it.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct AnthropicInputMessage {
    role: String,
    content: AnthropicMessageContent,
}

/// A turn's content: the plain-string shorthand or an ordered block list.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum AnthropicMessageContent {
    Text(String),
    Blocks(Vec<AnthropicInputContentBlock>),
}

/// The top-level system prompt: a string or a list of text blocks.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum AnthropicSystemPrompt {
    Text(String),
    Blocks(Vec<AnthropicSystemBlock>),
}

/// One system-prompt block; only `text` blocks exist in the protocol.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct AnthropicSystemBlock {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(default)]
    text: String,
}

/// One content block inside a user or assistant turn.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicInputContentBlock {
    Text {
        text: String,
    },
    Image {
        source: AnthropicImageSource,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        #[serde(default)]
        content: Option<AnthropicMessageContent>,
        #[serde(default)]
        is_error: Option<bool>,
    },
    Thinking {
        thinking: String,
    },
    /// Claude Code announces deferred tools mid-conversation with system-role
    /// `tool_addition` blocks.
    ToolAddition {
        tool: AnthropicToolAdditionReference,
    },
    RedactedThinking {},
    /// Documents, server-tool results, search results, and future block types.
    #[serde(other)]
    Unsupported,
}

/// The tool a `tool_addition` block makes available. Only `tool_reference`
/// is served: it names a tool already declared in the top-level `tools`, and
/// every declared tool is already visible to the local model, so dropping the
/// reference changes nothing the model sees.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct AnthropicToolAdditionReference {
    #[serde(rename = "type")]
    reference_type: String,
    #[serde(default)]
    name: String,
}

/// An image block's source; only inline `base64` sources are served locally.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct AnthropicImageSource {
    #[serde(rename = "type")]
    source_type: String,
    #[serde(default)]
    media_type: Option<String>,
    #[serde(default)]
    data: Option<String>,
}

/// A validated turn ready for daemon translation.
#[derive(Clone, Debug, PartialEq)]
pub enum AnthropicMessageParts {
    User {
        segments: Vec<AnthropicUserSegmentParts>,
    },
    Assistant {
        text: String,
        thinking: String,
        tool_uses: Vec<AnthropicToolUseParts>,
    },
    /// A mid-conversation instruction; `text` is empty when the turn held
    /// only tool references. The referenced names are checked against the
    /// declared tools, so a reference to an undeclared tool fails loudly
    /// instead of being dropped.
    System {
        text: String,
        referenced_tool_names: Vec<String>,
    },
}

/// One ordered piece of a user turn.
#[derive(Clone, Debug, PartialEq)]
pub enum AnthropicUserSegmentParts {
    Text(String),
    Image(DecodedImageInput),
    ToolResult {
        tool_use_id: String,
        text: String,
        images: Vec<DecodedImageInput>,
        is_error: bool,
    },
}

/// One prior assistant tool call, its input re-serialized as canonical JSON.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnthropicToolUseParts {
    pub id: String,
    pub name: String,
    pub input_json: String,
}

/// Running image totals checked against the portable request budget.
#[derive(Default)]
pub(crate) struct AnthropicImageBudget {
    image_part_count: usize,
    inline_media_bytes: usize,
}

impl AnthropicSystemPrompt {
    /// Joins system blocks with a blank line, the separator Anthropic clients
    /// assume between independently cached system sections.
    pub(crate) fn into_text(self) -> Result<String, AnthropicMessagesValidationError> {
        match self {
            Self::Text(system_text) => Ok(system_text),
            Self::Blocks(system_blocks) => {
                let mut system_texts = Vec::with_capacity(system_blocks.len());
                for system_block in system_blocks {
                    if system_block.block_type != "text" {
                        return Err(AnthropicMessagesValidationError::UnsupportedContentBlock {
                            location: "system",
                        });
                    }
                    system_texts.push(system_block.text);
                }
                Ok(system_texts.join("\n\n"))
            }
        }
    }
}

impl AnthropicInputMessage {
    pub(crate) fn into_parts(
        self,
        image_budget: &mut AnthropicImageBudget,
    ) -> Result<AnthropicMessageParts, AnthropicMessagesValidationError> {
        let content_blocks = match self.content {
            AnthropicMessageContent::Text(text) => vec![AnthropicInputContentBlock::Text { text }],
            AnthropicMessageContent::Blocks(content_blocks) => content_blocks,
        };
        if content_blocks.is_empty() {
            return Err(AnthropicMessagesValidationError::EmptyMessageContent);
        }
        match self.role.as_str() {
            "user" => user_message_parts(content_blocks, image_budget),
            "assistant" => assistant_message_parts(content_blocks),
            "system" => system_message_parts(content_blocks),
            _ => Err(AnthropicMessagesValidationError::UnsupportedRole { role: self.role }),
        }
    }
}

fn system_message_parts(
    content_blocks: Vec<AnthropicInputContentBlock>,
) -> Result<AnthropicMessageParts, AnthropicMessagesValidationError> {
    let mut system_texts = Vec::new();
    let mut referenced_tool_names = Vec::new();
    for content_block in content_blocks {
        match content_block {
            AnthropicInputContentBlock::Text { text } => system_texts.push(text),
            AnthropicInputContentBlock::ToolAddition { tool }
                if tool.reference_type == "tool_reference" =>
            {
                referenced_tool_names.push(tool.name);
            }
            _ => {
                return Err(AnthropicMessagesValidationError::UnsupportedContentBlock {
                    location: "system message",
                });
            }
        }
    }
    Ok(AnthropicMessageParts::System {
        text: system_texts.join("\n\n"),
        referenced_tool_names,
    })
}

fn user_message_parts(
    content_blocks: Vec<AnthropicInputContentBlock>,
    image_budget: &mut AnthropicImageBudget,
) -> Result<AnthropicMessageParts, AnthropicMessagesValidationError> {
    let mut segments = Vec::with_capacity(content_blocks.len());
    for content_block in content_blocks {
        let segment = match content_block {
            AnthropicInputContentBlock::Text { text } => AnthropicUserSegmentParts::Text(text),
            AnthropicInputContentBlock::Image { source } => {
                AnthropicUserSegmentParts::Image(image_budget.admit(source)?)
            }
            AnthropicInputContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => tool_result_segment(tool_use_id, content, is_error, image_budget)?,
            _ => {
                return Err(AnthropicMessagesValidationError::UnsupportedContentBlock {
                    location: "user message",
                });
            }
        };
        segments.push(segment);
    }
    Ok(AnthropicMessageParts::User { segments })
}

fn tool_result_segment(
    tool_use_id: String,
    content: Option<AnthropicMessageContent>,
    is_error: Option<bool>,
    image_budget: &mut AnthropicImageBudget,
) -> Result<AnthropicUserSegmentParts, AnthropicMessagesValidationError> {
    let mut result_texts = Vec::new();
    let mut images = Vec::new();
    match content {
        None => {}
        Some(AnthropicMessageContent::Text(text)) => result_texts.push(text),
        Some(AnthropicMessageContent::Blocks(result_blocks)) => {
            for result_block in result_blocks {
                match result_block {
                    AnthropicInputContentBlock::Text { text } => result_texts.push(text),
                    AnthropicInputContentBlock::Image { source } => {
                        images.push(image_budget.admit(source)?);
                    }
                    _ => {
                        return Err(AnthropicMessagesValidationError::UnsupportedContentBlock {
                            location: "tool_result content",
                        });
                    }
                }
            }
        }
    }
    Ok(AnthropicUserSegmentParts::ToolResult {
        tool_use_id,
        text: result_texts.join("\n"),
        images,
        is_error: is_error == Some(true),
    })
}

fn assistant_message_parts(
    content_blocks: Vec<AnthropicInputContentBlock>,
) -> Result<AnthropicMessageParts, AnthropicMessagesValidationError> {
    let mut text = String::new();
    let mut thinking = String::new();
    let mut tool_uses = Vec::new();
    for content_block in content_blocks {
        match content_block {
            AnthropicInputContentBlock::Text { text: block_text } => text.push_str(&block_text),
            AnthropicInputContentBlock::Thinking {
                thinking: block_thinking,
            } => thinking.push_str(&block_thinking),
            AnthropicInputContentBlock::ToolUse { id, name, input } => {
                let Ok(input_json) = serde_json::to_string(&input) else {
                    return Err(AnthropicMessagesValidationError::UnserializableToolInput);
                };
                tool_uses.push(AnthropicToolUseParts {
                    id,
                    name,
                    input_json,
                });
            }
            AnthropicInputContentBlock::RedactedThinking {} => {
                return Err(AnthropicMessagesValidationError::UnsupportedReasoningReplay);
            }
            _ => {
                return Err(AnthropicMessagesValidationError::UnsupportedContentBlock {
                    location: "assistant message",
                });
            }
        }
    }
    Ok(AnthropicMessageParts::Assistant {
        text,
        thinking,
        tool_uses,
    })
}

impl AnthropicImageBudget {
    /// Decodes one image source and charges it against the same portable
    /// image-count and inline-media budgets the OpenAI surfaces enforce.
    fn admit(
        &mut self,
        image_source: AnthropicImageSource,
    ) -> Result<DecodedImageInput, AnthropicMessagesValidationError> {
        let (Some(media_type), Some(base64_payload)) = (image_source.media_type, image_source.data)
        else {
            return Err(AnthropicMessagesValidationError::UnsupportedImageSource {
                source_type: image_source.source_type,
            });
        };
        if image_source.source_type != "base64" || !media_type.starts_with("image/") {
            return Err(AnthropicMessagesValidationError::UnsupportedImageSource {
                source_type: image_source.source_type,
            });
        }
        self.image_part_count += 1;
        if self.image_part_count > MAX_IMAGE_PARTS_PER_REQUEST {
            return Err(
                AnthropicMessagesValidationError::ImagePartCountExceedsBudget {
                    maximum_image_part_count: MAX_IMAGE_PARTS_PER_REQUEST,
                },
            );
        }
        let decoded_image = decode_base64_image_payload(&media_type, &base64_payload)
            .map_err(AnthropicMessagesValidationError::ImageInput)?;
        self.inline_media_bytes += decoded_image.decoded_bytes().len();
        if self.inline_media_bytes > MAX_INLINE_MEDIA_BYTES {
            return Err(AnthropicMessagesValidationError::InlineMediaExceedsBudget {
                maximum_inline_media_bytes: MAX_INLINE_MEDIA_BYTES,
            });
        }
        Ok(decoded_image)
    }
}
