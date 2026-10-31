//! The macOS app-builder packaging journey as an integration test
//! (issue #1346): a sandboxed repository copy runs the real
//! `scripts/internal/build-macos-app.sh` with fixture-bin tool shims on PATH,
//! and the produced bundle is asserted through `plutil` extraction the same
//! way `validate-macos-app.sh` reads real bundles. Contracts the validator
//! already owns (URL scheme, icon key, build identity, post-signature
//! entitlements) live only there; this journey pins what the *builder* emits
//! and how it drives its tools. The sandbox and assertion machinery live in
//! `app_builder_support`.

#![cfg(target_os = "macos")]

mod app_builder_support;

use app_builder_support::journey::{
    APP_STORE_STATE_ROOT_FEATURE, BuilderJourney, EXPECTED_UPDATE_FEED_URL,
    SPARKLE_PUBLIC_ED25519_KEY, assert_disposable_target_was_removed, file_text,
    log_line_count_containing, plist_string, preparatory_development_journey, resource_text,
};
use app_builder_support::sandbox::AppBuilderSandbox;

#[test]
fn missing_metallib_source_is_rejected() {
    let sandbox = AppBuilderSandbox::install();
    let outcome = sandbox.run_builder(&BuilderJourney {
        channel: "development",
        signing_identity: None,
        metallib_path: sandbox.root_directory.join("absent.metallib"),
        journey_log_stem: "missing-metallib",
    });

    assert!(
        !outcome.succeeded,
        "the builder accepted a missing mlx.metallib source"
    );
    assert!(
        outcome
            .standard_error_text
            .contains("required MLX AOT metallib is unavailable"),
        "the builder did not report the missing mlx.metallib source; stderr:\n{}",
        outcome.standard_error_text
    );
}

#[test]
fn development_bundle_carries_channel_policy_without_distribution_signing() {
    let sandbox = AppBuilderSandbox::install();
    let outcome = sandbox.run_builder(&BuilderJourney {
        channel: "development",
        signing_identity: None,
        metallib_path: sandbox.fixture_metallib_path.clone(),
        journey_log_stem: "development",
    });

    assert!(
        outcome.succeeded,
        "the development journey failed; stderr:\n{}",
        outcome.standard_error_text
    );
    assert!(
        outcome
            .standard_output_text
            .contains("Astronomical development app ready"),
        "the development journey did not reach its success summary; stdout:\n{}",
        outcome.standard_output_text
    );
    let development_bundle = sandbox.development_bundle_path();
    assert!(
        plist_string(&development_bundle, "AstronomicalChannel").as_deref() == Some("development"),
        "the development bundle did not declare its channel"
    );
    assert!(
        plist_string(&development_bundle, "CFBundleIdentifier").as_deref()
            == Some("dev.astronomical.app.development"),
        "the development bundle did not carry its development identifier"
    );
    assert!(
        plist_string(&development_bundle, "AstronomicalStateDirectoryName").as_deref()
            == Some(".astronomical-dev"),
        "the development bundle did not carry its isolated state directory name"
    );
    assert!(
        plist_string(&development_bundle, "SUFeedURL").as_deref() == Some(EXPECTED_UPDATE_FEED_URL),
        "the development bundle did not carry the GitHub Pages update feed"
    );
    assert!(
        plist_string(&development_bundle, "SUPublicEDKey").as_deref()
            == Some(SPARKLE_PUBLIC_ED25519_KEY),
        "the development bundle did not pin the Sparkle public key"
    );
    assert_eq!(
        resource_text(&development_bundle, "share/mlx/mlx.metallib"),
        "fixture-metallib",
        "the packaged metallib is not the build-produced copy"
    );
    assert_eq!(
        resource_text(&development_bundle, "Astronomical.icns"),
        "development",
        "the development bundle did not request the Development icon identity"
    );
    let development_codesign_text = file_text(&outcome.codesign_log_path);
    assert!(
        !development_codesign_text.contains("--options runtime"),
        "the development bundle unexpectedly requested distribution signing"
    );
    assert!(
        !sandbox
            .cargo_arguments_text()
            .contains(APP_STORE_STATE_ROOT_FEATURE),
        "the development build must not enable the App Store state-root feature"
    );
    assert!(
        sandbox
            .swift_arguments_text()
            .contains("--product astronomical-menu"),
        "the development build did not build the standard menu product"
    );
    assert_disposable_target_was_removed(&outcome);
}

#[test]
fn stable_bundle_signs_for_distribution_and_keeps_the_development_artifact() {
    let sandbox = AppBuilderSandbox::install();
    let preparatory_development_outcome =
        sandbox.run_builder(&preparatory_development_journey(&sandbox));
    assert!(
        preparatory_development_outcome.succeeded,
        "the preparatory development journey failed; stderr:\n{}",
        preparatory_development_outcome.standard_error_text
    );
    let outcome = sandbox.run_builder(&BuilderJourney {
        channel: "stable",
        signing_identity: Some("Developer ID Application: Example (ABCDE12345)"),
        metallib_path: sandbox.fixture_metallib_path.clone(),
        journey_log_stem: "stable",
    });

    assert!(
        outcome.succeeded,
        "the stable journey failed; stderr:\n{}",
        outcome.standard_error_text
    );
    assert!(
        outcome
            .standard_output_text
            .contains("Astronomical stable app ready"),
        "the stable journey did not reach its success summary; stdout:\n{}",
        outcome.standard_output_text
    );
    let stable_bundle = sandbox.stable_bundle_path();
    assert_eq!(
        resource_text(&stable_bundle, "Astronomical.icns"),
        "stable",
        "the stable bundle did not request the Stable icon identity"
    );
    assert!(
        plist_string(&stable_bundle, "SUFeedURL").as_deref() == Some(EXPECTED_UPDATE_FEED_URL),
        "the stable bundle did not carry the GitHub Pages update feed"
    );
    assert!(
        sandbox.development_bundle_path().exists(),
        "building stable unexpectedly removed the development build artifact"
    );
    let stable_codesign_text = file_text(&outcome.codesign_log_path);
    assert!(
        stable_codesign_text.contains(
            "--sign Developer ID Application: Example (ABCDE12345) --options runtime --timestamp"
        ),
        "the stable bundle was not signed with Developer ID, hardened runtime, and timestamping"
    );
    assert!(
        !sandbox
            .cargo_arguments_text()
            .contains(APP_STORE_STATE_ROOT_FEATURE),
        "the stable build must not enable the App Store state-root feature"
    );
    assert_disposable_target_was_removed(&outcome);
    let lane_root_residue: Vec<_> = std::fs::read_dir(&sandbox.lane_root_directory)
        .expect("the cargo lane root should remain listable")
        .collect();
    assert!(
        lane_root_residue.is_empty(),
        "app builds retained files beneath the disposable Cargo lane root"
    );
}

/// The bundle-level identity contract: the channel and app-store bundle
/// identifier in the plist, the updater and state-directory keys expunged, no
/// Sparkle artifacts, the Sparkle-free menu product requested through SwiftPM,
/// and the Stable icon identity.
fn assert_app_store_bundle_identity(
    sandbox: &AppBuilderSandbox,
    app_store_bundle: &std::path::Path,
) {
    assert!(
        plist_string(app_store_bundle, "AstronomicalChannel").as_deref() == Some("app-store"),
        "the app-store bundle did not declare its channel"
    );
    assert!(
        plist_string(app_store_bundle, "CFBundleIdentifier").as_deref()
            == Some("app.astronomical.appstore"),
        "the app-store bundle did not carry the app-store bundle identifier"
    );
    for absent_plist_key in [
        "SUFeedURL",
        "SUPublicEDKey",
        "SUEnableAutomaticChecks",
        "AstronomicalStateDirectoryName",
    ] {
        assert!(
            plist_string(app_store_bundle, absent_plist_key).is_none(),
            "the app-store bundle unexpectedly embeds {absent_plist_key}"
        );
    }
    assert!(
        !app_store_bundle
            .join("Contents/Frameworks/Sparkle.framework")
            .exists(),
        "the app-store bundle unexpectedly embeds the Sparkle framework"
    );
    assert!(
        !app_store_bundle
            .join("Contents/Resources/SPARKLE_LICENSE")
            .exists(),
        "the app-store bundle unexpectedly carries the Sparkle license"
    );
    // The App Store menu binary excludes Sparkle by building a distinct
    // SwiftPM product; the fake toolchain cannot link-check a stub, so the
    // product request is the observable contract here (the real validator
    // owns the post-build entitlement and linkage checks on real bundles).
    assert!(
        sandbox
            .swift_arguments_text()
            .contains("--product astronomical-menu-app-store"),
        "the app-store build did not build the Sparkle-free menu product"
    );
    assert_eq!(
        resource_text(app_store_bundle, "Astronomical.icns"),
        "stable",
        "the app-store bundle did not request the Stable icon identity"
    );
}

/// The fake codesign only records its arguments, so the entitlement contract
/// is asserted from the signing log: the main executable and the bundle seal
/// carry the sandbox profile; helpers join it through inheritance; and the
/// CLI stays bare because it runs from the user's terminal — no entitlements
/// profile of any kind may ride on it (the trailing-object match keeps the
/// CLI distinct from astronomicald and the bundle seal).
fn assert_app_store_entitlements_contract(app_store_codesign_text: &str) {
    for (entitlements_file_name, code_object_name) in [
        ("app-store-application.entitlements", "astronomical-menu"),
        ("app-store-inherited.entitlements", "astronomicald"),
        (
            "app-store-inherited.entitlements",
            "astronomical-inference-worker",
        ),
        ("app-store-application.entitlements", "Astronomical.app"),
    ] {
        let mut object_signed_with_profile = false;
        for codesign_line in app_store_codesign_text.lines() {
            if codesign_line.contains(entitlements_file_name)
                && codesign_line.contains(code_object_name)
            {
                object_signed_with_profile = true;
            }
        }
        assert!(
            object_signed_with_profile,
            "the app-store {code_object_name} was not signed with {entitlements_file_name}"
        );
    }
    let mut cli_entitlements_line_count = 0_usize;
    for codesign_line in app_store_codesign_text.lines() {
        let carries_entitlements = codesign_line.contains("--entitlements");
        let names_cli = codesign_line.trim_end().ends_with("MacOS/astronomical");
        if carries_entitlements && names_cli {
            cli_entitlements_line_count += 1;
        }
    }
    assert_eq!(
        cli_entitlements_line_count, 0,
        "the app-store CLI was signed with an entitlements profile"
    );
}

#[test]
fn app_store_bundle_is_sparkle_free_and_assigns_sandbox_entitlements() {
    let sandbox = AppBuilderSandbox::install();
    let outcome = sandbox.run_builder(&BuilderJourney {
        channel: "app-store",
        signing_identity: Some("Apple Distribution: Example (ABCDE12345)"),
        metallib_path: sandbox.fixture_metallib_path.clone(),
        journey_log_stem: "app-store",
    });

    assert!(
        outcome.succeeded,
        "the app-store journey failed; stderr:\n{}",
        outcome.standard_error_text
    );
    assert!(
        outcome
            .standard_output_text
            .contains("Astronomical app-store app ready"),
        "the app-store journey did not reach its success summary; stdout:\n{}",
        outcome.standard_output_text
    );
    assert_app_store_bundle_identity(&sandbox, &sandbox.app_store_bundle_path());

    let app_store_codesign_text = file_text(&outcome.codesign_log_path);
    assert_app_store_entitlements_contract(&app_store_codesign_text);

    let cargo_arguments_text = sandbox.cargo_arguments_text();
    assert_eq!(
        log_line_count_containing(&cargo_arguments_text, APP_STORE_STATE_ROOT_FEATURE),
        1,
        "the app-store state-root feature must be enabled exactly once, for the app-store build only"
    );
    assert!(
        cargo_arguments_text.contains("bootstrap-native-dependencies --verify"),
        "the builder did not verify the native dependency archives through xtask"
    );
    assert_disposable_target_was_removed(&outcome);
}
