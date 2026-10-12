//! Read and validate the `--image` inputs for `astronomical respond`.
//!
//! The CLI reads a raster image file and turns it into one decoded IPC image
//! input ready to cross the daemon boundary: it knows the bytes, the MIME type
//! the daemon expects, and rejects inputs before they ever reach the model.

use std::path::Path;

use astronomical_ipc_protocol::ChatImageInput;

use crate::errors::RespondError;

/// Decoded image bytes the CLI accepts in total for one request.
///
/// This leaves headroom for the roughly thirty-three percent base64 inflation
/// and framing the 32 MiB IPC frame then carries, so a request that passes this
/// check cannot itself exceed the daemon's IPC frame limit.
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
#[must_use]
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
mod tests {
    use super::*;

    fn write_image_file(test_directory: &Path, name: &str, contents: &[u8]) -> std::path::PathBuf {
        let path = test_directory.join(name);
        std::fs::write(&path, contents).expect("the fixture image should be writable");
        path
    }

    #[test]
    fn should_map_a_png_extension_to_the_png_mime_type() {
        assert_eq!(
            mime_for_extension(Path::new("snapshot.png")),
            Some("image/png")
        );
    }

    #[test]
    fn should_map_a_jpeg_extension_in_both_spellings_to_the_jpeg_mime_type() {
        assert_eq!(
            mime_for_extension(Path::new("scan.JPG")),
            Some("image/jpeg")
        );
        assert_eq!(
            mime_for_extension(Path::new("document.jpeg")),
            Some("image/jpeg")
        );
    }

    #[test]
    fn should_reject_an_unsupported_extension() {
        assert_eq!(mime_for_extension(Path::new("memo.txt")), None);
        assert_eq!(mime_for_extension(Path::new("notes.md")), None);
    }

    #[test]
    fn should_reject_a_file_with_no_extension() {
        assert_eq!(mime_for_extension(Path::new("image")), None);
    }

    #[test]
    fn should_read_a_supported_image_into_a_decoded_input() {
        let test_directory = std::env::temp_dir().join("respond_image_read");
        std::fs::create_dir_all(&test_directory).expect("the test directory should be creatable");
        let image_path = write_image_file(&test_directory, "picture.png", b"\x89PNG\r\n\x1a\nfake");
        let input = read_image_input(&image_path).expect("a supported image should read");
        assert_eq!(input.mime_type, "image/png");
        assert_eq!(input.decoded_bytes, b"\x89PNG\r\n\x1a\nfake");
        let _ = std::fs::remove_dir_all(&test_directory);
    }

    #[test]
    fn should_explain_a_missing_image_file() {
        let test_directory = std::env::temp_dir().join("respond_image_missing");
        std::fs::create_dir_all(&test_directory).expect("the test directory should be creatable");
        let missing_path = test_directory.join("nope.png");
        let outcome = read_image_input(&missing_path);
        assert!(
            matches!(outcome, Err(RespondError::ImageReadFailed { .. })),
            "a missing image should surface as a read failure: {outcome:?}"
        );
        let _ = std::fs::remove_dir_all(&test_directory);
    }

    #[test]
    fn should_reject_an_unsupported_image_file() {
        let test_directory = std::env::temp_dir().join("respond_image_unsupported");
        std::fs::create_dir_all(&test_directory).expect("the test directory should be creatable");
        let text_path = write_image_file(&test_directory, "memo.txt", b"hello");
        let outcome = read_image_input(&text_path);
        assert!(
            matches!(outcome, Err(RespondError::UnsupportedImage { .. })),
            "a non-image file should surface as unsupported: {outcome:?}"
        );
        let _ = std::fs::remove_dir_all(&test_directory);
    }

    #[test]
    fn should_enforce_the_shared_decoded_byte_budget_across_images() {
        let test_directory = std::env::temp_dir().join("respond_image_budget");
        std::fs::create_dir_all(&test_directory).expect("the test directory should be creatable");
        let first = write_image_file(&test_directory, "a.png", &[0u8; 10 * 1024 * 1024]);
        let second = write_image_file(&test_directory, "b.png", &[0u8; 10 * 1024 * 1020]);
        let outcome = read_image_inputs(&[first.clone(), second.clone()]);
        assert!(
            matches!(outcome, Err(RespondError::ImageTooLarge { .. })),
            "two images past the shared budget should surface as too large: {outcome:?}"
        );

        let small_second = write_image_file(&test_directory, "c.png", &[0u8; 2 * 1024 * 1024]);
        let inputs = read_image_inputs(&[first.clone(), small_second.clone()])
            .expect("in budget should read");
        assert_eq!(
            inputs.len(),
            2,
            "both images should be read when under budget"
        );
        let _ = std::fs::remove_dir_all(&test_directory);
    }

    #[test]
    fn should_read_no_images_when_the_paths_are_empty() {
        let inputs = read_image_inputs(&[]).expect("no images should read cleanly");
        assert!(inputs.is_empty(), "an empty request should carry no images");
    }

    #[test]
    fn should_accept_two_images_that_sum_exactly_to_the_budget() {
        const HALF: usize = MAX_TOTAL_IMAGE_DECODED_BYTES / 2;
        let test_directory = std::env::temp_dir().join("respond_image_boundary");
        std::fs::create_dir_all(&test_directory).expect("the test directory should be creatable");
        let first = write_image_file(&test_directory, "a.png", &[0u8; HALF]);
        let second = write_image_file(&test_directory, "b.png", &[0u8; HALF]);
        let inputs =
            read_image_inputs(&[first, second]).expect("summing exactly to the budget should read");
        assert_eq!(
            inputs.len(),
            2,
            "two images summing exactly to the budget should both read"
        );
        let _ = std::fs::remove_dir_all(&test_directory);
    }
}
