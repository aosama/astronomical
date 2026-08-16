use astronomical_ipc_protocol::{ChatMessage, ChatToolDefinition};
use serde_json::Value;
use thiserror::Error;

use crate::strict_json::{DUPLICATE_JSON_FIELD_MARKER, DuplicateAwareJsonValue};

use super::LagunaTextArtifactDescriptor;
use super::template_context::LagunaTemplateContext;
use super::template_program::LagunaTemplateProgramError;

const MAXIMUM_HISTORY_JSON_BYTES: usize = 256 * 1024;

/// A typed failure while rendering request data through the retained artifact template.
#[derive(Debug, Error)]
pub enum LagunaPromptRendererError {
    #[error("Laguna chat history must contain at least one message")]
    MissingMessages,
    #[error("Laguna tool JSON for '{function_name}' is invalid")]
    InvalidToolJson {
        function_name: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("Laguna tool JSON for '{function_name}' contains a duplicate field")]
    DuplicateToolJsonField { function_name: String },
    #[error("Laguna tool arguments for '{function_name}' must be a JSON object")]
    ToolArgumentsMustBeObject { function_name: String },
    #[error("Laguna tool JSON for '{function_name}' exceeds the bounded rendering limit")]
    ToolJsonTooLarge { function_name: String },
    #[error("Laguna tool definition for '{function_name}' could not be serialized")]
    SerializeToolDefinition {
        function_name: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("Laguna tool argument for '{function_name}' could not be serialized")]
    SerializeToolArgument {
        function_name: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("Laguna artifact template rendering failed")]
    TemplateRendering(#[source] minijinja::Error),
    #[error("Laguna rendered prompt exceeds the {maximum_bytes}-byte limit")]
    RenderedPromptTooLarge { maximum_bytes: usize },
    #[error("Laguna rendered prompt is not valid UTF-8")]
    RenderedPromptNotUtf8(#[source] std::string::FromUtf8Error),
}

/// Request renderer backed by the descriptor's startup-compiled template program.
#[derive(Debug)]
pub struct LagunaPromptRenderer<'a> {
    descriptor: &'a LagunaTextArtifactDescriptor,
}

impl<'a> LagunaPromptRenderer<'a> {
    /// Binds rendering to one canonical artifact descriptor.
    #[must_use]
    pub const fn new(descriptor: &'a LagunaTextArtifactDescriptor) -> Self {
        Self { descriptor }
    }

    /// Renders typed history, declared tools, and the assistant generation prefix.
    pub fn render(
        &self,
        messages: &[ChatMessage],
        tools: &[ChatToolDefinition],
        enable_thinking: bool,
    ) -> Result<String, LagunaPromptRendererError> {
        if messages.is_empty() {
            return Err(LagunaPromptRendererError::MissingMessages);
        }
        let context = LagunaTemplateContext::from_chat(
            messages,
            tools,
            Some(enable_thinking),
            &self.descriptor.template_contract().bos_token_content,
        )?;
        self.descriptor
            .template_program()
            .render(&context)
            .map_err(translate_program_error)
    }
}

pub(super) fn parse_strict_tool_json(
    function_name: &str,
    json_bytes: &[u8],
) -> Result<Value, LagunaPromptRendererError> {
    if json_bytes.len() > MAXIMUM_HISTORY_JSON_BYTES {
        return Err(LagunaPromptRendererError::ToolJsonTooLarge {
            function_name: function_name.to_owned(),
        });
    }
    serde_json::from_slice::<DuplicateAwareJsonValue>(json_bytes)
        .map(|strict_value| strict_value.0)
        .map_err(|source| {
            if source.to_string().contains(DUPLICATE_JSON_FIELD_MARKER) {
                LagunaPromptRendererError::DuplicateToolJsonField {
                    function_name: function_name.to_owned(),
                }
            } else {
                LagunaPromptRendererError::InvalidToolJson {
                    function_name: function_name.to_owned(),
                    source,
                }
            }
        })
}

fn translate_program_error(program_error: LagunaTemplateProgramError) -> LagunaPromptRendererError {
    match program_error {
        LagunaTemplateProgramError::Template(source) => {
            LagunaPromptRendererError::TemplateRendering(source)
        }
        LagunaTemplateProgramError::OutputTooLarge { maximum_bytes } => {
            LagunaPromptRendererError::RenderedPromptTooLarge { maximum_bytes }
        }
        LagunaTemplateProgramError::OutputNotUtf8(source) => {
            LagunaPromptRendererError::RenderedPromptNotUtf8(source)
        }
    }
}
