//! Unit tests for the `image` verb's pure parts: its grammar, the output-file
//! rules, and how failures are classified and worded. The journeys that cross
//! the real socket live in `tests/integration/image.rs`.

use std::ffi::OsString;
use std::path::PathBuf;

use astronomical_ipc_protocol::ImageGenerationFailureReason;

use super::image_arguments::{DEFAULT_IMAGE_SIDE_PIXELS, parse_image_arguments, parse_image_size};
use super::image_output::{discard_reserved_output, reserve_output_file, write_png};
use super::{ImageArguments, ImageError, image_failure_reason_text};
use crate::arguments::{CliCommand, UsageError, help_text, parse_command};

const PNG_SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];

fn parsed(arguments: &[&str]) -> Result<ImageArguments, UsageError> {
    let mut os_arguments = Vec::new();
    for argument in arguments {
        os_arguments.push(OsString::from(argument));
    }
    parse_image_arguments(&os_arguments)
}

#[test]
fn should_parse_a_prompt_alone_with_the_default_square_size() {
    let image_arguments = parsed(&["a lighthouse at dusk"]).expect("a bare prompt parses");

    assert_eq!(image_arguments.prompt, "a lighthouse at dusk");
    assert_eq!(image_arguments.width_pixels, DEFAULT_IMAGE_SIDE_PIXELS);
    assert_eq!(image_arguments.height_pixels, DEFAULT_IMAGE_SIDE_PIXELS);
    assert_eq!(image_arguments.model_id, None);
    assert_eq!(image_arguments.seed, None);
    assert_eq!(image_arguments.output_path, None);
}

#[test]
fn should_parse_every_flag_in_any_order() {
    let image_arguments = parsed(&[
        "--output",
        "out.png",
        "--seed",
        "7",
        "a lighthouse",
        "--size",
        "1024x576",
        "--model",
        "FLUX.2-klein-4B",
    ])
    .expect("every flag parses around the prompt");

    assert_eq!(image_arguments.prompt, "a lighthouse");
    assert_eq!(
        (image_arguments.width_pixels, image_arguments.height_pixels),
        (1_024, 576)
    );
    assert_eq!(image_arguments.model_id.as_deref(), Some("FLUX.2-klein-4B"));
    assert_eq!(image_arguments.seed, Some(7));
    assert_eq!(image_arguments.output_path, Some(PathBuf::from("out.png")));
}

#[test]
fn should_accept_an_uppercase_x_in_the_size() {
    assert_eq!(parse_image_size("1024X576"), Ok((1_024, 576)));
}

#[test]
fn should_accept_the_smallest_and_largest_seeds() {
    assert_eq!(
        parsed(&["a", "--seed", "0"]).expect("zero is a seed").seed,
        Some(0)
    );
    assert_eq!(
        parsed(&["a", "--seed", "18446744073709551615"])
            .expect("the largest u64")
            .seed,
        Some(u64::MAX)
    );
}

#[test]
fn should_require_a_prompt() {
    assert_eq!(parsed(&[]), Err(UsageError::ImagePromptRequired));
    assert_eq!(
        parsed(&["--size", "512x512"]),
        Err(UsageError::ImagePromptRequired)
    );
}

#[test]
fn should_refuse_a_blank_prompt() {
    assert_eq!(parsed(&["   "]), Err(UsageError::ImagePromptRequired));
    assert_eq!(parsed(&[""]), Err(UsageError::ImagePromptRequired));
}

#[test]
fn should_refuse_a_second_positional_as_an_unquoted_prompt() {
    assert_eq!(
        parsed(&["a", "lighthouse"]),
        Err(UsageError::UnknownArgument("lighthouse".to_owned()))
    );
}

#[test]
fn should_refuse_every_repeated_flag() {
    for (flag, value) in [
        ("--model", "m"),
        ("--size", "64x64"),
        ("--seed", "1"),
        ("--output", "a.png"),
    ] {
        let outcome = parsed(&["p", flag, value, flag, value]);

        assert!(
            matches!(outcome, Err(UsageError::RepeatedArgument(_))),
            "{flag}: {outcome:?}"
        );
    }
}

#[test]
fn should_refuse_an_unknown_flag() {
    assert_eq!(
        parsed(&["p", "--steps", "50"]),
        Err(UsageError::UnknownArgument("--steps".to_owned())),
        "the step count belongs to the model family and is not a flag"
    );
}

#[test]
fn should_refuse_a_flag_without_its_value() {
    for flag in ["--model", "--size", "--seed", "--output"] {
        assert_eq!(
            parsed(&["p", flag]),
            Err(UsageError::MissingValue(match flag {
                "--model" => "--model",
                "--size" => "--size",
                "--seed" => "--seed",
                _ => "--output",
            }))
        );
    }
}

#[test]
fn should_refuse_malformed_sizes() {
    for malformed_size in [
        "1024",
        "x512",
        "1024x",
        "axb",
        "0x512",
        "512x0",
        "1024x576x2",
        "1.5x2",
        "4294967296x1",
    ] {
        assert_eq!(
            parse_image_size(malformed_size),
            Err(UsageError::InvalidImageSize(malformed_size.to_owned())),
            "{malformed_size}"
        );
    }
}

#[test]
fn should_refuse_malformed_seeds() {
    for malformed_seed in ["abc", "1.5", "18446744073709551616"] {
        assert_eq!(
            parsed(&["p", "--seed", malformed_seed]),
            Err(UsageError::InvalidImageSeed(malformed_seed.to_owned())),
            "{malformed_seed}"
        );
    }
}

#[test]
fn should_route_the_image_command_through_the_cli_grammar() {
    let command = parse_command([
        OsString::from("astronomical"),
        OsString::from("image"),
        OsString::from("a lighthouse"),
    ])
    .expect("the image verb parses");

    assert!(matches!(command, CliCommand::Image(_)));
    assert_eq!(
        parse_command([OsString::from("astronomical"), OsString::from("image")]),
        Err(UsageError::ImagePromptRequired)
    );
}

#[test]
fn should_document_the_image_verb_and_its_flags_in_the_help_text() {
    let help = help_text();

    for expected_text in [
        "astronomical image PROMPT",
        "--size WIDTHxHEIGHT",
        "--seed N",
        "--output PATH",
    ] {
        assert!(help.contains(expected_text), "{expected_text}");
    }
}

#[test]
fn should_reserve_a_timestamped_png_name_in_the_default_directory() {
    let default_directory = tempfile::tempdir().expect("a scratch directory");

    let reserved_output =
        reserve_output_file(None, default_directory.path()).expect("the default name is free");
    let saved_path =
        write_png(reserved_output, &PNG_SIGNATURE).expect("a bare signature is a PNG for the CLI");

    let file_name = saved_path
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .expect("a file name")
        .to_owned();
    assert!(
        file_name.starts_with("astronomical-image-") && file_name.ends_with(".png"),
        "{file_name}"
    );
}

#[test]
fn should_accept_png_names_in_any_case_and_refuse_the_rest() {
    let output_directory = tempfile::tempdir().expect("a scratch directory");
    // Distinct stems: the default macOS filesystem is case-insensitive, so
    // `a.png` and `a.PNG` would be the same file and the claim would refuse.
    for accepted_name in ["one.png", "two.PNG", "three.PnG", "dir.v2.png"] {
        let reserved_output = reserve_output_file(
            Some(&output_directory.path().join(accepted_name)),
            output_directory.path(),
        );

        assert!(reserved_output.is_ok(), "{accepted_name}");
    }
    for refused_name in ["a.jpg", "a.png.txt", "a", "a."] {
        let outcome = reserve_output_file(
            Some(&output_directory.path().join(refused_name)),
            output_directory.path(),
        );

        assert!(
            matches!(outcome, Err(ImageError::OutputMustBePng { .. })),
            "{refused_name}"
        );
    }
}

#[test]
fn should_refuse_to_claim_a_path_that_already_exists() {
    let output_directory = tempfile::tempdir().expect("a scratch directory");
    let taken_path = output_directory.path().join("taken.png");
    std::fs::write(&taken_path, b"keep me").expect("the file should be writable");

    let outcome = reserve_output_file(Some(&taken_path), output_directory.path());

    assert!(matches!(outcome, Err(ImageError::OutputExists { .. })));
    assert_eq!(std::fs::read(&taken_path).expect("still there"), b"keep me");
}

#[test]
fn should_refuse_to_claim_a_path_in_a_missing_folder() {
    let output_directory = tempfile::tempdir().expect("a scratch directory");
    let orphan_path = output_directory.path().join("missing").join("a.png");

    let outcome = reserve_output_file(Some(&orphan_path), output_directory.path());

    assert!(matches!(outcome, Err(ImageError::OutputUnwritable { .. })));
}

#[test]
fn should_write_exactly_the_png_bytes() {
    let output_directory = tempfile::tempdir().expect("a scratch directory");
    let output_path = output_directory.path().join("a.png");
    let mut png_bytes = PNG_SIGNATURE.to_vec();
    png_bytes.extend_from_slice(b"payload");

    let reserved_output =
        reserve_output_file(Some(&output_path), output_directory.path()).expect("free");
    write_png(reserved_output, &png_bytes).expect("a PNG is written");

    assert_eq!(std::fs::read(&output_path).expect("saved"), png_bytes);
}

#[test]
fn should_refuse_bytes_that_are_not_a_png_and_remove_the_claim() {
    let output_directory = tempfile::tempdir().expect("a scratch directory");
    for not_a_png in [&b""[..], &b"GIF89a"[..], &PNG_SIGNATURE[..7]] {
        let output_path = output_directory.path().join("a.png");
        let reserved_output =
            reserve_output_file(Some(&output_path), output_directory.path()).expect("free");

        let outcome = write_png(reserved_output, not_a_png);

        assert!(
            matches!(outcome, Err(ImageError::ReturnedImageInvalid)),
            "{not_a_png:?}"
        );
        assert!(!output_path.exists(), "the empty claim must be removed");
    }
}

#[test]
fn should_remove_a_discarded_claim_so_the_path_is_free_again() {
    let output_directory = tempfile::tempdir().expect("a scratch directory");
    let output_path = output_directory.path().join("a.png");
    let reserved_output =
        reserve_output_file(Some(&output_path), output_directory.path()).expect("free");

    discard_reserved_output(reserved_output);

    assert!(!output_path.exists());
    assert!(reserve_output_file(Some(&output_path), output_directory.path()).is_ok());
}

#[test]
fn should_classify_only_command_fixable_failures_as_usage_errors() {
    let path = PathBuf::from("a.png");
    let usage_errors = [
        ImageError::ModelUnavailable {
            reason: "x".to_owned(),
        },
        ImageError::OutputExists { path: path.clone() },
        ImageError::OutputMustBePng { path: path.clone() },
    ];
    let runtime_errors = [
        ImageError::WorkerNotReady,
        ImageError::DaemonStoppedResponding,
        ImageError::ImageRejected {
            reason: "x".to_owned(),
        },
        ImageError::GenerationTimedOut { seconds: 1 },
        ImageError::ReturnedImageInvalid,
        ImageError::OutputUnwritable {
            path,
            cause: "x".to_owned(),
        },
    ];

    for usage_error in &usage_errors {
        assert!(usage_error.is_usage_error(), "{usage_error:?}");
    }
    for runtime_error in &runtime_errors {
        assert!(!runtime_error.is_usage_error(), "{runtime_error:?}");
    }
}

#[test]
fn should_word_every_worker_failure_for_a_person() {
    let expectations = [
        (ImageGenerationFailureReason::EngineBusy, "busy"),
        (ImageGenerationFailureReason::Cancelled, "cancelled"),
        (
            ImageGenerationFailureReason::ModelDoesNotSupportImageGeneration,
            "--model",
        ),
        (
            ImageGenerationFailureReason::invalid_request("bad size"),
            "bad size",
        ),
        (
            ImageGenerationFailureReason::EncodingFailed {
                reason: "disk".to_owned(),
            },
            "PNG",
        ),
        (
            ImageGenerationFailureReason::FatalExecution {
                reason: "oom".to_owned(),
            },
            "oom",
        ),
    ];

    for (failure_reason, expected_text) in expectations {
        let rendered_text = image_failure_reason_text(&failure_reason);

        assert!(rendered_text.contains(expected_text), "{rendered_text}");
    }
}

#[test]
fn should_render_the_same_text_in_display_and_debug() {
    let image_error = ImageError::OutputExists {
        path: PathBuf::from("a.png"),
    };

    assert_eq!(format!("{image_error:?}"), image_error.to_string());
    assert!(image_error.to_string().contains("nothing is overwritten"));
}
