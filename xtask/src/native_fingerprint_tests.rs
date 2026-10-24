use std::path::Path;

use astronomical_native_identity::probes::ProbeValue;
use astronomical_native_identity::{IdentityScope, NativeBuildProfile};

use super::{
    FingerprintOptions, host_from_rustc_version, one_line_text, parse_arguments, probe_source_text,
    resolve_target_triple,
};

fn fingerprint_options(arguments: &[&str]) -> FingerprintOptions {
    parse_arguments(
        Path::new("repository"),
        &arguments
            .iter()
            .map(|argument| (*argument).to_owned())
            .collect::<Vec<_>>(),
    )
    .unwrap()
    .unwrap()
}

fn argument_failure(arguments: &[&str]) -> String {
    parse_arguments(
        Path::new("repository"),
        &arguments
            .iter()
            .map(|argument| (*argument).to_owned())
            .collect::<Vec<_>>(),
    )
    .unwrap_err()
}

#[test]
fn should_parse_explicit_target_profile_explanation_and_repository_root() {
    let options = fingerprint_options(&[
        "--target",
        "x86_64-apple-darwin",
        "--profile",
        "core+memory-contract",
        "--explain",
        "selected-repository",
    ]);
    assert_eq!(options.repository_root, Path::new("selected-repository"));
    assert_eq!(options.profile, NativeBuildProfile::CoreMemoryContract);
    assert_eq!(options.scope, IdentityScope::Full);
    assert!(options.explain);
    assert_eq!(
        options.target_triple.as_deref(),
        Some("x86_64-apple-darwin")
    );
}

#[test]
fn should_keep_the_supplied_repository_root_and_live_target_by_default() {
    let options = fingerprint_options(&["--profile", "core"]);
    assert_eq!(options.repository_root, Path::new("repository"));
    assert_eq!(options.profile, NativeBuildProfile::Core);
    assert_eq!(options.scope, IdentityScope::Full);
    assert!(!options.explain);
    assert_eq!(options.target_triple, None);
}

#[test]
fn should_accept_help_without_a_profile() {
    for help_flag in ["--help", "-h"] {
        assert_eq!(
            parse_arguments(Path::new("repository"), &[help_flag.to_owned()]),
            Ok(None)
        );
    }
}

#[test]
fn should_require_an_explicit_supported_profile() {
    assert_eq!(argument_failure(&[]), "--profile is required");
    assert_eq!(
        argument_failure(&["--profile", "debug"]),
        "unsupported native build profile: debug"
    );
}

#[test]
fn should_reject_missing_blank_and_option_shaped_option_values() {
    for option_name in ["--profile", "--target"] {
        assert_eq!(
            argument_failure(&[option_name]),
            format!("{option_name} requires a value")
        );
        for invalid_value in ["", " \t", "--explain"] {
            assert_eq!(
                argument_failure(&[option_name, invalid_value]),
                format!("{option_name} requires a value")
            );
        }
    }
}

#[test]
fn should_reject_synthetic_toolchain_arguments() {
    for unsupported_option in [
        "--xcode",
        "--sdk",
        "--clang",
        "--cmake",
        "--rustc",
        "--build-type",
    ] {
        assert_eq!(
            argument_failure(&["--profile", "core", unsupported_option, "fixture"]),
            format!("unsupported native fingerprint option: {unsupported_option}")
        );
    }
}

#[test]
fn should_reject_repeated_targets_and_multiple_repository_roots() {
    assert_eq!(
        argument_failure(&[
            "--profile",
            "core",
            "--target",
            "aarch64-apple-darwin",
            "--target",
            "x86_64-apple-darwin",
        ]),
        "only one --target may be given"
    );
    assert_eq!(
        argument_failure(&["--profile", "core", "first-repository", "second-repository"]),
        "only one repository root may be given"
    );
}

#[test]
fn should_prefer_the_explicit_target_to_cargo_target_without_probing_the_host() {
    let target_triple = resolve_target_triple(
        Some("x86_64-apple-darwin"),
        Some("aarch64-apple-darwin"),
        || panic!("an explicit target must not probe the host"),
    )
    .unwrap();
    assert_eq!(target_triple, "x86_64-apple-darwin");
}

#[test]
fn should_use_the_standard_cargo_target_without_probing_the_host() {
    let target_triple = resolve_target_triple(None, Some("aarch64-apple-darwin"), || {
        panic!("Cargo's actual target must not be replaced by the host")
    })
    .unwrap();
    assert_eq!(target_triple, "aarch64-apple-darwin");
}

#[test]
fn should_probe_the_rustc_host_when_cargo_target_is_absent_or_empty() {
    for cargo_target in [None, Some("")] {
        let target_triple = resolve_target_triple(None, cargo_target, || {
            Ok("x86_64-unknown-linux-gnu".to_owned())
        })
        .unwrap();
        assert_eq!(target_triple, "x86_64-unknown-linux-gnu");
    }
}

#[test]
fn should_propagate_a_failed_host_probe() {
    assert_eq!(
        resolve_target_triple(None, None, || Err("host probe failed".to_owned())),
        Err("host probe failed".to_owned())
    );
}

#[test]
fn should_reject_a_blank_cargo_target_before_probing_the_host() {
    assert_eq!(
        resolve_target_triple(None, Some(" \t"), || {
            panic!("a supplied but blank target must fail")
        }),
        Err("the native build target triple must not be blank".to_owned())
    );
}

#[test]
fn should_extract_the_actual_rustc_host_from_verbose_version_output() {
    let rustc_version_output =
        "rustc 1.101.0\nbinary: rustc\nhost: x86_64-unknown-linux-gnu\nrelease: 1.101.0\n";
    assert_eq!(
        host_from_rustc_version(rustc_version_output),
        Ok("x86_64-unknown-linux-gnu".to_owned())
    );
}

#[test]
fn should_reject_rustc_output_without_a_nonblank_host() {
    for invalid_output in ["", "rustc 1.101.0\n", "host: \n", "host: \t\n"] {
        assert_eq!(
            host_from_rustc_version(invalid_output),
            Err("rustc host probe did not report a target triple".to_owned())
        );
    }
}

#[test]
fn should_construct_full_requests_with_real_toolchain_probes_and_release_policy() {
    let options = fingerprint_options(&["--profile", "core", "--target", "x86_64-apple-darwin"]);
    let request = options
        .request(None, || panic!("the explicit target needs no host probe"))
        .unwrap();
    assert_eq!(
        request.overrides.target_triple.as_deref(),
        Some("x86_64-apple-darwin")
    );
    assert_eq!(request.overrides.xcode, None);
    assert_eq!(request.overrides.sdk, None);
    assert_eq!(request.overrides.clang, None);
    assert_eq!(request.overrides.cmake, None);
    assert_eq!(request.overrides.rustc, None);
    assert_eq!(request.overrides.build_type, None);
}

#[test]
fn should_skip_target_resolution_for_source_only_requests() {
    let options = fingerprint_options(&["--source-only", "--profile", "core"]);
    let request = options
        .request(Some(" \t"), || {
            panic!("source-only identity must not probe a toolchain")
        })
        .unwrap();
    assert_eq!(request.scope, IdentityScope::SourceOnly);
    assert_eq!(request.overrides.target_triple, None);
}

#[test]
fn should_flatten_only_the_captured_probe_text_for_explanation() {
    assert_eq!(
        one_line_text("version=17.0.0\n target=aarch64-apple-darwin \n\n"),
        "version=17.0.0 | target=aarch64-apple-darwin"
    );
    assert_eq!(one_line_text("\n \n"), "");
}

#[test]
fn should_report_the_source_of_each_captured_probe() {
    for (from_override, expected_source) in [(true, "override"), (false, "live")] {
        let probe_entry = ProbeValue {
            name: "rustc",
            value: "rustc fixture\n".to_owned(),
            from_override,
        };
        assert_eq!(probe_source_text(&probe_entry), expected_source);
    }
}
