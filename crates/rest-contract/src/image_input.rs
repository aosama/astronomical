//! Data-URI image input decoding for the OpenAI-compatible chat endpoint.
//!
//! Only `data:image/<type>;base64,<payload>` URIs are accepted. HTTP(S) and
//! `file://` schemes are rejected to preserve the single-laptop privacy model
//! and avoid local-path attack surface.

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;

use crate::OpenAiChatCompletionValidationError;

/// Maximum accepted decoded image byte payload. Matches the chat body cap.
pub const MAX_OPENAI_IMAGE_BYTES: usize = 32 * 1024 * 1024;

/// One decoded image carried in a user chat message.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OpenAiImageInput {
    /// The MIME type parsed from the data URI, e.g. `image/png`.
    image_mime_type: String,
    /// The raw decoded image file bytes (PNG/JPEG/WebP payload before pixel decoding).
    decoded_image_bytes: Vec<u8>,
}

impl OpenAiImageInput {
    /// Returns the MIME type parsed from the data URI.
    #[must_use]
    pub fn mime_type(&self) -> &str {
        &self.image_mime_type
    }

    /// Returns the raw decoded image file bytes.
    #[must_use]
    pub fn decoded_bytes(&self) -> &[u8] {
        &self.decoded_image_bytes
    }
}

/// Decodes a pre-validated `data:image/<type>;base64,<payload>` URI into image bytes.
///
/// The caller must have already validated the scheme and MIME type via
/// `validate_image_url_scheme`. This function extracts the MIME type, decodes
/// the base64 payload, and enforces the byte size bound.
pub(crate) fn decode_image_url(
    image_url: &str,
) -> Result<OpenAiImageInput, OpenAiChatCompletionValidationError> {
    let data_uri_body = image_url
        .strip_prefix("data:")
        .ok_or(OpenAiChatCompletionValidationError::UnsupportedImageUrlScheme)?;
    let comma_position = data_uri_body
        .find(',')
        .ok_or(OpenAiChatCompletionValidationError::MalformedDataUri)?;
    let metadata = &data_uri_body[..comma_position];
    let base64_payload = &data_uri_body[comma_position + 1..];
    let (image_mime_type, _encoding) = metadata
        .split_once(';')
        .ok_or(OpenAiChatCompletionValidationError::MalformedDataUri)?;
    let decoded_image_bytes = BASE64_STANDARD
        .decode(base64_payload)
        .map_err(|_| OpenAiChatCompletionValidationError::InvalidBase64)?;
    if decoded_image_bytes.len() > MAX_OPENAI_IMAGE_BYTES {
        return Err(OpenAiChatCompletionValidationError::ImageTooLarge {
            actual_bytes: decoded_image_bytes.len(),
            maximum_bytes: MAX_OPENAI_IMAGE_BYTES,
        });
    }
    Ok(OpenAiImageInput {
        image_mime_type: image_mime_type.to_owned(),
        decoded_image_bytes,
    })
}

/// Validates a data-URI image before decoding its bounded payload.
pub(crate) fn validate_image_url_scheme(
    image_url: &str,
) -> Result<(), OpenAiChatCompletionValidationError> {
    let Some(data_uri_body) = image_url.strip_prefix("data:") else {
        return Err(OpenAiChatCompletionValidationError::UnsupportedImageUrlScheme);
    };
    let Some(comma_position) = data_uri_body.find(',') else {
        return Err(OpenAiChatCompletionValidationError::MalformedDataUri);
    };
    let metadata = &data_uri_body[..comma_position];
    let base64_payload = &data_uri_body[comma_position + 1..];
    let Some((mime_type, encoding)) = metadata.split_once(';') else {
        return Err(OpenAiChatCompletionValidationError::MalformedDataUri);
    };
    if encoding != "base64" {
        return Err(OpenAiChatCompletionValidationError::UnsupportedImageUrlScheme);
    }
    if !mime_type.starts_with("image/") {
        return Err(
            OpenAiChatCompletionValidationError::UnsupportedImageMimeType {
                actual_mime_type: mime_type.to_owned(),
            },
        );
    }
    if !base64_payload
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'/' | b'='))
    {
        return Err(OpenAiChatCompletionValidationError::InvalidBase64);
    }
    Ok(())
}
