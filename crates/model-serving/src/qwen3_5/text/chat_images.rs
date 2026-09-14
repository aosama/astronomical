//! User-image preparation for chat requests.
//!
//! Extracts one token-count vector per user message from the conversation
//! history and the decoded visual images the engine consumes, keeping image
//! handling out of the tokenizer's prompt path.

use astronomical_ipc_protocol::ChatMessage;

use super::tokenizer_error::Qwen3_5TokenizerError;
use crate::qwen3_5::Qwen3_5ImageProcessor;
use crate::qwen3_5::Qwen3_5ProcessedImage;

/// Extracts one token-count vector per user message from the conversation history.
///
/// Each user message may carry zero or more decoded images. For every image,
/// the Qwen3VL image processor computes the number of `<|image_pad|>` tokens
/// after spatial merge. Text-only user messages produce an empty vector.
pub(super) struct PreparedChatImages {
    pub(super) image_token_counts_per_user_message: Vec<Vec<usize>>,
    pub(super) processed_visual_images: Vec<Qwen3_5ProcessedImage>,
}

pub(super) fn prepare_chat_images(
    messages: &[ChatMessage],
    image_processor: Option<&Qwen3_5ImageProcessor>,
) -> Result<PreparedChatImages, Qwen3_5TokenizerError> {
    let mut image_token_counts_per_user_message = Vec::new();
    let mut processed_visual_images = Vec::new();
    for message in messages {
        if let ChatMessage::User { images, .. } = message {
            let mut per_image_token_counts = Vec::with_capacity(images.len());
            for image_input in images {
                let image_processor =
                    image_processor.ok_or(Qwen3_5TokenizerError::ImageInputUnsupported)?;
                let processed_image = image_processor
                    .process_image_bytes(&image_input.decoded_bytes)
                    .map_err(Qwen3_5TokenizerError::ImageProcessing)?;
                let image_token_count_after_spatial_merge =
                    processed_image.image_token_count_after_spatial_merge;
                per_image_token_counts.push(image_token_count_after_spatial_merge);
                processed_visual_images.push(processed_image);
            }
            image_token_counts_per_user_message.push(per_image_token_counts);
        }
    }
    Ok(PreparedChatImages {
        image_token_counts_per_user_message,
        processed_visual_images,
    })
}
