//! Argument-parsing contracts for the standalone native-build tool.
//!
//! Pure CPU tests: they exercise the CLI surface that CI and the local
//! pre-warm scripts rely on, without touching CMake or the native build store.

use astronomical_native_build_tool::{NativeBuildToolArguments, parse_arguments};

const FICTIONAL_REPOSITORY_ROOT: &str = "/tmp/astronomical-native-build-tool-fixture";

fn parse_with_profile(profile_name: &str) -> Result<NativeBuildToolArguments, String> {
    parse_arguments(&[
        "--profile".to_owned(),
        profile_name.to_owned(),
        "--repository-root".to_owned(),
        FICTIONAL_REPOSITORY_ROOT.to_owned(),
    ])
}

#[test]
fn should_parse_core_profile_arguments() {
    let parsed_arguments = parse_with_profile("core").expect("core profile arguments should parse");

    assert_eq!(
        parsed_arguments.native_build_profile().identity_name(),
        "core"
    );
    assert_eq!(
        parsed_arguments.repository_root(),
        std::path::Path::new(FICTIONAL_REPOSITORY_ROOT)
    );
}

#[test]
fn should_parse_every_supported_profile_identity_name() {
    for profile_identity_name in [
        "core",
        "core+memory-contract",
        "core+experimental-aligned-expert-packs",
        "core+memory-contract+experimental-aligned-expert-packs",
    ] {
        let parsed_arguments =
            parse_with_profile(profile_identity_name).unwrap_or_else(|parse_error| {
                panic!("profile {profile_identity_name} should parse: {parse_error}")
            });

        assert_eq!(
            parsed_arguments.native_build_profile().identity_name(),
            profile_identity_name
        );
    }
}

#[test]
fn should_reject_unknown_profile_name() {
    let parse_error = parse_with_profile("core+unsupported-probe")
        .expect_err("an unknown profile name should be rejected");

    assert!(parse_error.contains("unsupported native build profile"));
    assert!(parse_error.contains("core+memory-contract"));
}

#[test]
fn should_reject_missing_profile_argument() {
    let parse_error = parse_arguments(&[
        "--repository-root".to_owned(),
        FICTIONAL_REPOSITORY_ROOT.to_owned(),
    ])
    .expect_err("a missing --profile argument should be rejected");

    assert!(parse_error.contains("missing required argument --profile"));
}

#[test]
fn should_reject_missing_repository_root_argument() {
    let parse_error = parse_arguments(&["--profile".to_owned(), "core".to_owned()])
        .expect_err("a missing --repository-root argument should be rejected");

    assert!(parse_error.contains("missing required argument --repository-root"));
}

#[test]
fn should_reject_relative_repository_root() {
    let parse_error = parse_arguments(&[
        "--profile".to_owned(),
        "core".to_owned(),
        "--repository-root".to_owned(),
        "relative/path".to_owned(),
    ])
    .expect_err("a relative repository root should be rejected");

    assert!(parse_error.contains("must be an absolute path"));
}

#[test]
fn should_reject_unsupported_argument() {
    let parse_error = parse_arguments(&[
        "--profile".to_owned(),
        "core".to_owned(),
        "--repository-root".to_owned(),
        FICTIONAL_REPOSITORY_ROOT.to_owned(),
        "--jobs".to_owned(),
        "4".to_owned(),
    ])
    .expect_err("an unsupported argument should be rejected");

    assert!(parse_error.contains("unsupported argument: --jobs"));
}
