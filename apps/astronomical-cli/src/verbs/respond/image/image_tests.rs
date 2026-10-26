//! Unit tests for `crate::verbs::respond::image`: extension-to-mime mapping and the
//! `--image` file reading rules, including the shared decoded-byte budget.

use std::path::Path;

use crate::verbs::respond::RespondError;
use crate::verbs::respond::image::{self as respond_image, MAX_TOTAL_IMAGE_DECODED_BYTES};

fn write_image_file(test_directory: &Path, name: &str, contents: &[u8]) -> std::path::PathBuf {
    let path = test_directory.join(name);
    std::fs::write(&path, contents).expect("the fixture image should be writable");
    path
}

#[test]
fn should_map_a_png_extension_to_the_png_mime_type() {
    assert_eq!(
        respond_image::mime_for_extension(Path::new("snapshot.png")),
        Some("image/png")
    );
}

#[test]
fn should_map_a_jpeg_extension_in_both_spellings_to_the_jpeg_mime_type() {
    assert_eq!(
        respond_image::mime_for_extension(Path::new("scan.JPG")),
        Some("image/jpeg")
    );
    assert_eq!(
        respond_image::mime_for_extension(Path::new("document.jpeg")),
        Some("image/jpeg")
    );
}

#[test]
fn should_map_a_webp_extension_to_the_webp_mime_type() {
    assert_eq!(
        respond_image::mime_for_extension(Path::new("snapshot.webp")),
        Some("image/webp")
    );
}

#[test]
fn should_advertise_exactly_the_formats_the_decode_stack_honors() {
    // The vision image processor in crates/model-serving decodes every format
    // listed here; its `should_decode_every_format_the_cli_advertises` test
    // pins the same set from the decode side. Change both together.
    assert_eq!(
        respond_image::supported_image_extensions(),
        "png, jpg, jpeg, webp"
    );
    assert_eq!(
        respond_image::mime_for_extension(Path::new("a.png")),
        Some("image/png")
    );
    assert_eq!(
        respond_image::mime_for_extension(Path::new("a.jpg")),
        Some("image/jpeg")
    );
    assert_eq!(
        respond_image::mime_for_extension(Path::new("a.jpeg")),
        Some("image/jpeg")
    );
    assert_eq!(
        respond_image::mime_for_extension(Path::new("a.webp")),
        Some("image/webp")
    );
}

#[test]
fn should_advertise_the_image_help_extensions_matching_the_supported_set() {
    // The `--image` help line must advertise exactly the extensions the
    // extension-to-MIME mapping accepts, so the printed surface cannot
    // drift from what `read_image_input` enforces.
    let advertised_help_extensions = format!("({})", respond_image::supported_image_extensions());
    assert!(
        crate::arguments::help_text().contains(&advertised_help_extensions),
        "the --image help line should advertise {advertised_help_extensions}"
    );
}

#[test]
fn should_reject_an_unsupported_extension() {
    assert_eq!(
        respond_image::mime_for_extension(Path::new("memo.txt")),
        None
    );
    assert_eq!(
        respond_image::mime_for_extension(Path::new("notes.md")),
        None
    );
}

#[test]
fn should_reject_a_file_with_no_extension() {
    assert_eq!(respond_image::mime_for_extension(Path::new("image")), None);
}

#[test]
fn should_read_a_supported_image_into_a_decoded_input() {
    let test_directory = tempfile::tempdir().expect("the test directory should be creatable");
    let image_path = write_image_file(
        test_directory.path(),
        "picture.png",
        b"\x89PNG\r\n\x1a\nfake",
    );
    let input =
        respond_image::read_image_input(&image_path).expect("a supported image should read");
    assert_eq!(input.mime_type, "image/png");
    assert_eq!(input.decoded_bytes, b"\x89PNG\r\n\x1a\nfake");
}

#[test]
fn should_explain_a_missing_image_file() {
    let test_directory = tempfile::tempdir().expect("the test directory should be creatable");
    let missing_path = test_directory.path().join("nope.png");
    let outcome = respond_image::read_image_input(&missing_path);
    assert!(
        matches!(outcome, Err(RespondError::ImageReadFailed { .. })),
        "a missing image should surface as a read failure: {outcome:?}"
    );
}

#[test]
fn should_reject_an_unsupported_image_file() {
    let test_directory = tempfile::tempdir().expect("the test directory should be creatable");
    let text_path = write_image_file(test_directory.path(), "memo.txt", b"hello");
    let outcome = respond_image::read_image_input(&text_path);
    assert!(
        matches!(outcome, Err(RespondError::UnsupportedImage { .. })),
        "a non-image file should surface as unsupported: {outcome:?}"
    );
}

#[test]
fn should_enforce_the_shared_decoded_byte_budget_across_images() {
    let test_directory = tempfile::tempdir().expect("the test directory should be creatable");
    let first = write_image_file(test_directory.path(), "a.png", &[0u8; 10 * 1024 * 1024]);
    let second = write_image_file(test_directory.path(), "b.png", &[0u8; 10 * 1024 * 1020]);
    let outcome = respond_image::read_image_inputs(&[first.clone(), second.clone()]);
    assert!(
        matches!(outcome, Err(RespondError::ImageTooLarge { .. })),
        "two images past the shared budget should surface as too large: {outcome:?}"
    );

    let small_second = write_image_file(test_directory.path(), "c.png", &[0u8; 2 * 1024 * 1024]);
    let inputs = respond_image::read_image_inputs(&[first.clone(), small_second.clone()])
        .expect("in budget should read");
    assert_eq!(
        inputs.len(),
        2,
        "both images should be read when under budget"
    );
}

#[test]
fn should_read_no_images_when_the_paths_are_empty() {
    let inputs = respond_image::read_image_inputs(&[]).expect("no images should read cleanly");
    assert!(inputs.is_empty(), "an empty request should carry no images");
}

#[test]
fn should_accept_two_images_that_sum_exactly_to_the_budget() {
    const HALF: usize = MAX_TOTAL_IMAGE_DECODED_BYTES / 2;
    let test_directory = tempfile::tempdir().expect("the test directory should be creatable");
    let first = write_image_file(test_directory.path(), "a.png", &[0u8; HALF]);
    let second = write_image_file(test_directory.path(), "b.png", &[0u8; HALF]);
    let inputs = respond_image::read_image_inputs(&[first, second])
        .expect("summing exactly to the budget should read");
    assert_eq!(
        inputs.len(),
        2,
        "two images summing exactly to the budget should both read"
    );
}
