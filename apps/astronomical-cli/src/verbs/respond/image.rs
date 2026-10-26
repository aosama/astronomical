//! Read and validate the `--image` inputs for `astronomical respond`.
//!
//! The CLI reads a raster image file and turns it into one decoded IPC image
//! input ready to cross the daemon boundary: it knows the bytes, the MIME type
//! the daemon expects, and rejects inputs before they ever reach the model.

use std::path::Path;

use astronomical_ipc_protocol::ChatImageInput;

use crate::verbs::respond::RespondError;

/// Decoded image bytes the CLI accepts in total for one request.
///
/// This leaves headroom for the roughly thirty-three percent base64 inflation
/// and framing the daemon's IPC frame then carries (the portable body budget,
/// see the rest-contract `openai_request_budget` module), so a request that
/// passes this check cannot itself exceed the daemon's IPC frame limit.
pub const MAX_TOTAL_IMAGE_DECODED_BYTES: usize = 16 * 1024 * 1024;

/// Supported image extensions paired with the wire MIME type the daemon expects.
const SUPPORTED_IMAGE_FORMATS: &[(&str, &str)] = &[
    ("png", "image/png"),
    ("jpg", "image/jpeg"),
    ("jpeg", "image/jpeg"),
    ("webp", "image/webp"),
];

/// The supported image extensions, for error messages.
const SUPPORTED_IMAGE_EXTENSIONS: &str = "png, jpg, jpeg, webp";

/// Reads and validates every image file for a request into decoded inputs.
///
/// Enforces the shared decoded-byte budget across all images so one request
/// cannot grow past what the daemon's IPC frame will carry.
pub fn read_image_inputs(
    paths: &[std::path::PathBuf],
) -> Result<Vec<ChatImageInput>, RespondError> {
    let mut inputs = Vec::with_capacity(paths.len());
    let mut total_bytes = 0usize;
    for path in paths {
        let input = read_image_input(path)?;
        total_bytes += input.decoded_bytes.len();
        if total_bytes > MAX_TOTAL_IMAGE_DECODED_BYTES {
            return Err(RespondError::ImageTooLarge {
                actual_bytes: total_bytes,
                maximum_bytes: MAX_TOTAL_IMAGE_DECODED_BYTES,
            });
        }
        inputs.push(input);
    }
    Ok(inputs)
}

/// Reads one image file into a decoded IPC input, or explains why it is rejected.
pub fn read_image_input(path: &Path) -> Result<ChatImageInput, RespondError> {
    let decoded_bytes =
        std::fs::read(path).map_err(|read_error| RespondError::ImageReadFailed {
            path: path.to_path_buf(),
            cause: read_error.to_string(),
        })?;
    let mime_type = mime_for_extension(path).ok_or(RespondError::UnsupportedImage {
        path: path.to_path_buf(),
        supported: supported_image_extensions().to_owned(),
    })?;
    Ok(ChatImageInput {
        mime_type: mime_type.to_owned(),
        decoded_bytes,
    })
}

/// Maps an image file extension to its wire MIME type, ignoring case.
#[must_use]
pub fn mime_for_extension(path: &Path) -> Option<&'static str> {
    let extension = path.extension()?.to_str()?.to_ascii_lowercase();
    SUPPORTED_IMAGE_FORMATS
        .iter()
        .find(|(supported_extension, _)| *supported_extension == extension)
        .map(|(_, mime_type)| *mime_type)
}

/// The supported image extensions, for error messages.
#[must_use]
pub fn supported_image_extensions() -> &'static str {
    SUPPORTED_IMAGE_EXTENSIONS
}

#[cfg(test)]
mod image_tests;
