//! Images a tool returned, such as a screenshot, carried to the model.
//!
//! The worker's `ChatMessage::Tool` carries text only, and its prompt places
//! images for user messages only, so a tool's images follow its run of tool
//! messages in one user message, where a vision model sees them. (The
//! reference Qwen3.5 template could also render them inside the tool
//! response; that needs images on the worker's tool message.) They wait until
//! the run ends because the template renders consecutive tool messages as one
//! turn, and an image message in the middle would split that turn. A short
//! label says where they came from, so the model does not read them as a new
//! question from the person. The Anthropic endpoint also moves a
//! `tool_result` block's images into a following user message, but per result
//! and without the label; one placement for all three endpoints is #2320.

use ipc_protocol::{ChatImageInput, ChatMessage};
use rest_contract::DecodedImageInput;

/// The text of the user message that carries a tool run's images.
pub(crate) const TOOL_RESULT_IMAGES_LABEL: &str = "Images returned by the tool calls above.";

/// The images of the tool results translated since the last other message.
#[derive(Default)]
pub(crate) struct PendingToolResultImages {
    images: Vec<ChatImageInput>,
}

impl PendingToolResultImages {
    /// Keeps one tool result's images until its run of tool messages ends.
    pub(crate) fn hold(&mut self, tool_result_images: &[DecodedImageInput]) {
        self.images
            .extend(tool_result_images.iter().map(chat_image_input));
    }

    /// Called before each translated message: any message that is not a tool
    /// result ends the run of tool messages before it.
    pub(crate) fn end_run_before(
        &mut self,
        next_message_is_tool_result: bool,
        chat_messages: &mut Vec<ChatMessage>,
    ) {
        if !next_message_is_tool_result {
            self.flush_into(chat_messages);
        }
    }

    /// Ends a run of tool messages: the images it returned, if any, become one
    /// user message after it.
    pub(crate) fn flush_into(&mut self, chat_messages: &mut Vec<ChatMessage>) {
        if self.images.is_empty() {
            return;
        }
        chat_messages.push(ChatMessage::User {
            content: TOOL_RESULT_IMAGES_LABEL.to_owned(),
            images: std::mem::take(&mut self.images),
        });
    }
}

/// The worker's form of one decoded request image.
pub(crate) fn chat_image_input(image: &DecodedImageInput) -> ChatImageInput {
    ChatImageInput {
        mime_type: image.mime_type().to_owned(),
        decoded_bytes: image.decoded_bytes().to_vec(),
    }
}

#[cfg(test)]
#[path = "tool_result_images_tests.rs"]
mod tool_result_images_tests;
