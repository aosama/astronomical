//! Release steps that install, package for the App Store, and accept a
//! Sparkle update, tested through their pure decisions and fixture
//! directories.

use std::path::{Path, PathBuf};

use super::app_store_package::{
    AppStorePackageRequest, app_store_channel_problem, default_output_package, identity_is_listed,
};
use super::command::{AppStoreCommand, ReleaseCommand, parse_release_command};
use super::sparkle_acceptance::{acceptance_info_plist, state_fingerprint};
use super::stable_installation::{
    PromotionTools, canonical_home_directory, promote_bundle, stable_promotion_problems,
};

fn arguments(argument_words: &[&str]) -> Vec<String> {
    let mut owned_arguments = Vec::new();
    for argument_word in argument_words {
        owned_arguments.push((*argument_word).to_owned());
    }
    owned_arguments
}

fn write_fixture_file(file_path: &Path, file_text: &str) {
    std::fs::create_dir_all(file_path.parent().expect("a parent")).expect("fixture directory");
    std::fs::write(file_path, file_text).expect("fixture file");
}

#[test]
fn should_resolve_a_real_home_directory_and_refuse_unsafe_ones() {
    let fixture_directory = tempfile::tempdir().expect("a fixture directory");
    let home_text = fixture_directory.path().display().to_string();

    assert_eq!(
        canonical_home_directory(Some(&home_text)),
        Ok(std::fs::canonicalize(fixture_directory.path()).expect("canonical fixture"))
    );
    assert!(canonical_home_directory(None).is_err());
    assert!(canonical_home_directory(Some("relative/home")).is_err());
    assert!(
        canonical_home_directory(Some(&format!("{home_text}/../escape")))
            .expect_err("parent")
            .contains("parent")
    );
    assert!(
        canonical_home_directory(Some(&format!("{home_text}/absent")))
            .expect_err("absent")
            .contains("existing")
    );
    assert!(
        canonical_home_directory(Some("/"))
            .expect_err("root")
            .contains("root")
    );
}

#[test]
fn should_promote_only_clean_stable_bundles() {
    let clean_stable =
        serde_json::json!({"AstronomicalChannel": "stable", "AstronomicalBuildDirty": false});
    let dirty_stable =
        serde_json::json!({"AstronomicalChannel": "stable", "AstronomicalBuildDirty": true});
    let development =
        serde_json::json!({"AstronomicalChannel": "development", "AstronomicalBuildDirty": false});

    assert!(stable_promotion_problems(&clean_stable).is_empty());
    assert_eq!(
        stable_promotion_problems(&dirty_stable),
        vec!["dirty builds cannot be promoted to Stable"]
    );
    assert_eq!(
        stable_promotion_problems(&development),
        vec!["only a Stable bundle may be promoted"]
    );
}

fn copy_fixture_bundle(source_bundle: &Path, staged_bundle: &Path) -> Result<(), String> {
    let source_marker =
        std::fs::read_to_string(source_bundle.join("marker")).map_err(drop_error)?;
    write_fixture_file(&staged_bundle.join("marker"), &source_marker);
    Ok(())
}

fn refuse_to_stage(_source_bundle: &Path, staged_bundle: &Path) -> Result<(), String> {
    write_fixture_file(&staged_bundle.join("partial"), "half-copied");
    Err("the staged copy does not verify".to_owned())
}

fn delete_fixture_backup(backup_bundle: &Path) -> Result<(), String> {
    std::fs::remove_dir_all(backup_bundle).map_err(drop_error)
}

fn drop_error(io_error: std::io::Error) -> String {
    io_error.to_string()
}

const FIXTURE_PROMOTION_TOOLS: PromotionTools = PromotionTools {
    stage_copy: copy_fixture_bundle,
    dispose_backup: delete_fixture_backup,
};

#[test]
fn should_install_fresh_then_replace_the_previous_app_leaving_no_staging_residue() {
    let fixture_directory = tempfile::tempdir().expect("a fixture directory");
    let applications_directory = fixture_directory.path().join("Applications");
    let first_bundle = fixture_directory.path().join("first.app");
    let second_bundle = fixture_directory.path().join("second.app");
    write_fixture_file(&first_bundle.join("marker"), "first");
    write_fixture_file(&second_bundle.join("marker"), "second");

    promote_bundle(
        &first_bundle,
        &applications_directory,
        &FIXTURE_PROMOTION_TOOLS,
    )
    .expect("a fresh install");
    promote_bundle(
        &second_bundle,
        &applications_directory,
        &FIXTURE_PROMOTION_TOOLS,
    )
    .expect("a replacement");

    let installed_marker =
        std::fs::read_to_string(applications_directory.join("Astronomical.app/marker"))
            .expect("installed");
    assert_eq!(installed_marker, "second");
    let mut directory_names = Vec::new();
    for directory_entry in std::fs::read_dir(&applications_directory)
        .expect("listable")
        .flatten()
    {
        directory_names.push(directory_entry.file_name().to_string_lossy().into_owned());
    }
    assert_eq!(
        directory_names,
        vec!["Astronomical.app"],
        "no stage or backup may remain"
    );
}

#[test]
fn should_leave_the_installed_app_untouched_when_staging_fails() {
    let fixture_directory = tempfile::tempdir().expect("a fixture directory");
    let applications_directory = fixture_directory.path().join("Applications");
    write_fixture_file(
        &applications_directory.join("Astronomical.app/marker"),
        "installed",
    );
    let failing_tools = PromotionTools {
        stage_copy: refuse_to_stage,
        dispose_backup: delete_fixture_backup,
    };

    let promotion_error = promote_bundle(
        &fixture_directory.path().join("any.app"),
        &applications_directory,
        &failing_tools,
    )
    .expect_err("must fail");

    assert!(
        promotion_error.contains("does not verify"),
        "{promotion_error}"
    );
    assert_eq!(
        std::fs::read_to_string(applications_directory.join("Astronomical.app/marker"))
            .expect("kept"),
        "installed"
    );
    assert_eq!(
        std::fs::read_dir(&applications_directory)
            .expect("listable")
            .count(),
        1,
        "the partial stage was removed"
    );
}

#[test]
fn should_parse_install_commands_with_an_optional_bundle_and_dry_run() {
    assert_eq!(
        parse_release_command(&arguments(&[
            "install-stable",
            "--dry-run",
            "--app-bundle",
            "/a.app"
        ])),
        Ok(ReleaseCommand::InstallStable {
            app_bundle: Some(PathBuf::from("/a.app")),
            is_dry_run: true
        })
    );
    assert_eq!(
        parse_release_command(&arguments(&["install-stable"])),
        Ok(ReleaseCommand::InstallStable {
            app_bundle: None,
            is_dry_run: false
        })
    );
    assert_eq!(
        parse_release_command(&arguments(&["build-and-install-stable", "--dry-run"])),
        Ok(ReleaseCommand::BuildAndInstallStable { is_dry_run: true })
    );
    assert!(
        parse_release_command(&arguments(&[
            "build-and-install-stable",
            "--app-bundle",
            "/a.app"
        ]))
        .is_err()
    );
    assert!(parse_release_command(&arguments(&["install-stable", "--force"])).is_err());
}

#[test]
fn should_package_only_app_store_channel_bundles() {
    assert_eq!(
        app_store_channel_problem(&serde_json::json!({"AstronomicalChannel": "app-store"})),
        None
    );
    assert!(
        app_store_channel_problem(&serde_json::json!({"AstronomicalChannel": "stable"}))
            .expect("stable is refused")
            .contains("this bundle declares stable")
    );
    assert!(
        app_store_channel_problem(&serde_json::json!({}))
            .expect("missing channel")
            .contains("does not declare")
    );
}

#[test]
fn should_match_the_installer_identity_by_its_full_quoted_name() {
    let identity_listing = "  1) ABCDEF0123456789 \"3rd Party Mac Developer Installer: Example (ABCDE12345)\"\n     1 valid identities found\n";

    assert!(identity_is_listed(
        identity_listing,
        "3rd Party Mac Developer Installer: Example (ABCDE12345)"
    ));
    assert!(!identity_is_listed(
        identity_listing,
        "Mac Developer Installer"
    ));
    assert!(!identity_is_listed(identity_listing, ""));
}

#[test]
fn should_write_the_package_beside_the_bundle_by_default() {
    assert_eq!(
        default_output_package(
            Path::new("/target/astronomical-macos-app-store.noindex/Astronomical.app"),
            "0.2.88"
        ),
        PathBuf::from(
            "/target/astronomical-macos-app-store.noindex/Astronomical-0.2.88-app-store.pkg"
        )
    );
}

#[test]
fn should_parse_the_app_store_package_command() {
    assert_eq!(
        parse_release_command(&arguments(&[
            "app-store-pkg",
            "--app-bundle",
            "/a.app",
            "--installer-identity",
            "Installer"
        ])),
        Ok(ReleaseCommand::AppStore(AppStoreCommand::Package(
            AppStorePackageRequest {
                app_bundle: PathBuf::from("/a.app"),
                installer_identity: "Installer".to_owned(),
                output_package: None,
            }
        )))
    );
    assert!(
        parse_release_command(&arguments(&["app-store-pkg", "--app-bundle", "/a.app"])).is_err()
    );
    assert!(
        parse_release_command(&arguments(&[
            "app-store-pkg",
            "--app-bundle",
            "/a.app",
            "--app-bundle",
            "/b.app",
            "--installer-identity",
            "I",
        ]))
        .is_err()
    );
}

#[cfg(target_os = "macos")]
#[test]
fn should_render_an_acceptance_plist_with_the_product_update_policy() {
    let fixture_directory = tempfile::tempdir().expect("a fixture directory");
    let plist_path = fixture_directory.path().join("Info.plist");
    let plist_text = acceptance_info_plist(
        "2",
        "2.0.0",
        "http://127.0.0.1:8123/appcast.xml?a=1&b=2",
        &fixture_directory.path().join("launch.log"),
        &fixture_directory.path().join("app.pid"),
    );
    std::fs::write(&plist_path, &plist_text).expect("plist");

    let lint_output = std::process::Command::new("plutil")
        .arg("-lint")
        .arg(&plist_path)
        .output()
        .expect("plutil runs");

    assert!(
        lint_output.status.success(),
        "{}",
        String::from_utf8_lossy(&lint_output.stdout)
    );
    assert!(plist_text.contains("<key>SURequireSignedFeed</key><true/>"));
    assert!(plist_text.contains("<key>SUVerifyUpdateBeforeExtraction</key><true/>"));
    assert!(plist_text.contains("<key>CFBundleVersion</key><string>2</string>"));
    assert!(plist_text.contains("a=1&amp;b=2"));
}

#[test]
fn should_fingerprint_external_state_and_refuse_a_missing_marker() {
    let state_directory = tempfile::tempdir().expect("a state directory");
    for marker_path in [
        "config/config.json",
        "models/model-marker",
        "cache/cache-marker",
        "logs/log-marker",
    ] {
        write_fixture_file(&state_directory.path().join(marker_path), marker_path);
    }

    let fingerprint_before =
        state_fingerprint(state_directory.path()).expect("all markers present");
    assert_eq!(
        state_fingerprint(state_directory.path()),
        Ok(fingerprint_before.clone())
    );
    write_fixture_file(
        &state_directory.path().join("cache/cache-marker"),
        "changed",
    );
    assert_ne!(
        state_fingerprint(state_directory.path()).expect("still present"),
        fingerprint_before
    );
    std::fs::remove_file(state_directory.path().join("logs/log-marker")).expect("remove a marker");
    assert!(state_fingerprint(state_directory.path()).is_err());
}

#[test]
fn should_parse_the_sparkle_acceptance_command_without_arguments() {
    assert_eq!(
        parse_release_command(&arguments(&["accept-sparkle-update"])),
        Ok(ReleaseCommand::AcceptSparkleUpdate)
    );
    assert!(parse_release_command(&arguments(&["accept-sparkle-update", "--fast"])).is_err());
}
