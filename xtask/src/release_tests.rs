use std::path::{Path, PathBuf};

use super::command::{ReleaseCommand, parse_release_command};
use super::disk_image_creation::{
    DiskImageRequest, FINDER_LAYOUT_SCRIPT, partial_output_path, validate_disk_image_request,
};
use super::disk_image_validation::{mounted_layout_problems, validate_disk_image};
use super::distribution_signature::{
    discover_code_objects, signature_problems, validate_distribution_app,
};
use super::notarization::{
    DEFAULT_NOTARIZATION_TIMEOUT_SECONDS, NotarizationRequest, notarization_plan,
    validate_notarization_request,
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

fn fixture_app_bundle(fixture_root: &Path) -> PathBuf {
    let app_bundle = fixture_root.join("Astronomical.app");
    write_fixture_file(
        &app_bundle.join("Contents/Resources/share/mlx/mlx.metallib"),
        "metallib",
    );
    app_bundle
}

#[test]
fn should_parse_the_disk_image_commands() {
    assert_eq!(
        parse_release_command(&arguments(&[
            "create-dmg",
            "--app-bundle",
            "/a.app",
            "--output",
            "/out.dmg"
        ])),
        Ok(ReleaseCommand::CreateDiskImage(DiskImageRequest {
            app_bundle: PathBuf::from("/a.app"),
            output_disk_image: PathBuf::from("/out.dmg"),
        }))
    );
    assert_eq!(
        parse_release_command(&arguments(&["validate-dmg", "--dmg", "/out.dmg"])),
        Ok(ReleaseCommand::ValidateDiskImage(PathBuf::from("/out.dmg")))
    );
}

#[test]
fn should_refuse_missing_repeated_or_unknown_release_arguments() {
    assert!(parse_release_command(&arguments(&[])).is_err());
    assert!(parse_release_command(&arguments(&["create-dmg", "--app-bundle", "/a.app"])).is_err());
    assert!(parse_release_command(&arguments(&["validate-dmg", "--dmg"])).is_err());
    assert!(
        parse_release_command(&arguments(&["validate-dmg", "--dmg", "/a", "--dmg", "/b"])).is_err()
    );
    assert!(parse_release_command(&arguments(&["validate-dmg", "--image", "/a"])).is_err());
    assert!(parse_release_command(&arguments(&["publish-everything"])).is_err());
}

#[test]
fn should_accept_a_bundle_with_its_metallib_and_a_fresh_output() {
    let fixture_directory = tempfile::tempdir().expect("a fixture directory");
    let disk_image_request = DiskImageRequest {
        app_bundle: fixture_app_bundle(fixture_directory.path()),
        output_disk_image: fixture_directory.path().join("release/Astronomical.dmg"),
    };

    assert_eq!(validate_disk_image_request(&disk_image_request), Ok(()));
}

fn request_for(app_bundle: &Path, output_disk_image: &Path) -> DiskImageRequest {
    DiskImageRequest {
        app_bundle: app_bundle.to_path_buf(),
        output_disk_image: output_disk_image.to_path_buf(),
    }
}

#[test]
fn should_refuse_a_missing_bundle_metallib_or_existing_output() {
    let fixture_directory = tempfile::tempdir().expect("a fixture directory");
    let app_bundle = fixture_app_bundle(fixture_directory.path());
    let existing_output = fixture_directory.path().join("existing.dmg");
    write_fixture_file(&existing_output, "image");
    let fresh_output = fixture_directory.path().join("fresh.dmg");

    assert!(
        validate_disk_image_request(&request_for(
            &fixture_directory.path().join("absent.app"),
            &fresh_output
        ))
        .is_err()
    );
    assert!(
        validate_disk_image_request(&request_for(&app_bundle, &existing_output))
            .expect_err("existing")
            .contains("already exists")
    );
    let metallib_path = app_bundle.join("Contents/Resources/share/mlx/mlx.metallib");
    std::fs::remove_file(&metallib_path).expect("remove the metallib");
    std::os::unix::fs::symlink("/dev/null", &metallib_path).expect("symlink the metallib");
    assert!(
        validate_disk_image_request(&request_for(&app_bundle, &fresh_output))
            .expect_err("symlinked")
            .contains("metallib")
    );
}

#[test]
fn should_stage_the_compressed_image_as_a_hidden_sibling_of_the_output() {
    let partial_path = partial_output_path(Path::new("/release/Astronomical-0.2.88.dmg"));

    assert_eq!(partial_path.parent(), Some(Path::new("/release")));
    let partial_name = partial_path
        .file_name()
        .expect("a file name")
        .to_string_lossy()
        .into_owned();
    assert!(
        partial_name.starts_with(".Astronomical-0.2.88.dmg.partial."),
        "{partial_name}"
    );
    assert!(partial_name.ends_with(".dmg"), "{partial_name}");
}

#[test]
fn should_place_the_app_and_applications_icons_on_either_side_of_the_arrow() {
    assert!(
        FINDER_LAYOUT_SCRIPT
            .contains("set position of item \"Astronomical.app\" of mountedVolume to {175, 205}")
    );
    assert!(
        FINDER_LAYOUT_SCRIPT
            .contains("set position of item \"Applications\" of mountedVolume to {455, 205}")
    );
    assert!(FINDER_LAYOUT_SCRIPT.contains("/.background/background.png"));
}

fn complete_mounted_volume(mount_point: &Path) {
    fixture_app_bundle(mount_point);
    std::os::unix::fs::symlink("/Applications", mount_point.join("Applications"))
        .expect("applications link");
    write_fixture_file(&mount_point.join(".DS_Store"), "layout");
    write_fixture_file(&mount_point.join(".background/background.png"), "png");
}

#[test]
fn should_accept_a_complete_installer_volume() {
    let mount_directory = tempfile::tempdir().expect("a mount fixture");
    complete_mounted_volume(mount_directory.path());

    assert!(mounted_layout_problems(mount_directory.path()).is_empty());
}

#[test]
fn should_report_every_missing_or_misdirected_installer_item() {
    let mount_directory = tempfile::tempdir().expect("a mount fixture");
    std::os::unix::fs::symlink("/Users", mount_directory.path().join("Applications"))
        .expect("wrong link");

    let layout_problems = mounted_layout_problems(mount_directory.path());

    assert_eq!(
        layout_problems,
        vec![
            "DMG does not contain Astronomical.app",
            "DMG Astronomical.app is missing mlx.metallib",
            "DMG Applications link does not target /Applications",
            "DMG does not contain Finder layout metadata",
            "DMG does not contain installation guidance",
        ]
    );
}

#[test]
fn should_refuse_to_validate_an_empty_or_missing_disk_image() {
    let fixture_directory = tempfile::tempdir().expect("a fixture directory");
    let empty_image = fixture_directory.path().join("empty.dmg");
    write_fixture_file(&empty_image, "");

    assert!(
        validate_disk_image(&empty_image)
            .expect_err("empty")
            .contains("nonempty")
    );
    assert!(validate_disk_image(&fixture_directory.path().join("absent.dmg")).is_err());
}

const DISTRIBUTABLE_SIGNATURE: &str = "Executable=/fixture/astronomicald\nCodeDirectory v=20500 size=1 flags=0x10000(runtime) hashes=1+7\nAuthority=Developer ID Application: Example (ABCDE12345)\nTimestamp=3 Nov 2026 at 01:00:00\nTeamIdentifier=ABCDE12345\n";

#[test]
fn should_accept_a_developer_id_runtime_timestamped_signature_of_the_release_team() {
    assert!(signature_problems(DISTRIBUTABLE_SIGNATURE, "ABCDE12345").is_empty());
}

#[test]
fn should_name_each_missing_property_of_a_distributable_signature() {
    let ad_hoc_signature = "Executable=/fixture/astronomicald\nCodeDirectory v=20500 size=1 flags=0x2(adhoc) hashes=1+7\nSignature=adhoc\nTeamIdentifier=not set\n";

    assert_eq!(
        signature_problems(ad_hoc_signature, "ABCDE12345"),
        vec![
            "code object is not signed by Developer ID Application",
            "code object Team ID does not match release configuration",
            "code object does not enable Hardened Runtime",
            "code object does not have a secure timestamp",
        ]
    );
    assert_eq!(
        signature_problems(DISTRIBUTABLE_SIGNATURE, "ZZZZZ99999"),
        vec!["code object Team ID does not match release configuration"]
    );
}

#[test]
fn should_discover_bundles_frameworks_services_and_executables_without_following_symlinks() {
    let fixture_directory = tempfile::tempdir().expect("a fixture directory");
    let app_bundle = fixture_directory.path().join("Astronomical.app");
    let framework_versions = app_bundle.join("Contents/Frameworks/Sparkle.framework/Versions");
    write_fixture_file(
        &framework_versions.join("B/XPCServices/Installer.xpc/Contents/Info.plist"),
        "plist",
    );
    write_fixture_file(&app_bundle.join("Contents/MacOS/astronomicald"), "binary");
    write_fixture_file(&app_bundle.join("Contents/Resources/LICENSE"), "text");
    let executable_permissions = std::os::unix::fs::PermissionsExt::from_mode(0o755);
    std::fs::set_permissions(
        app_bundle.join("Contents/MacOS/astronomicald"),
        executable_permissions,
    )
    .expect("executable bit");
    std::os::unix::fs::symlink("B", framework_versions.join("Current")).expect("Current link");

    let code_objects = discover_code_objects(&app_bundle).expect("the bundle should be walkable");

    let mut relative_objects = Vec::new();
    for code_object in &code_objects {
        relative_objects.push(
            code_object
                .strip_prefix(fixture_directory.path())
                .expect("inside the fixture")
                .display()
                .to_string(),
        );
    }
    assert_eq!(
        relative_objects,
        vec![
            "Astronomical.app",
            "Astronomical.app/Contents/Frameworks/Sparkle.framework",
            "Astronomical.app/Contents/Frameworks/Sparkle.framework/Versions/B/XPCServices/Installer.xpc",
            "Astronomical.app/Contents/MacOS/astronomicald",
        ]
    );
}

#[test]
fn should_refuse_distribution_validation_before_running_codesign() {
    let fixture_directory = tempfile::tempdir().expect("a fixture directory");
    let app_bundle = fixture_app_bundle(fixture_directory.path());

    assert!(
        validate_distribution_app(&fixture_directory.path().join("absent.app"), "ABCDE12345")
            .is_err()
    );
    assert!(
        validate_distribution_app(&app_bundle, "ABC")
            .expect_err("short team")
            .contains("--team-id")
    );
    assert!(
        validate_distribution_app(&app_bundle, "ABCDE12345")
            .expect_err("no Sparkle")
            .contains("code object is unavailable")
    );
}

fn notarization_request(disk_image: &Path, timeout_seconds: u64) -> NotarizationRequest {
    NotarizationRequest {
        disk_image: disk_image.to_path_buf(),
        signing_identity: "Developer ID Application: Example (ABCDE12345)".to_owned(),
        notary_profile: "astronomical-notary".to_owned(),
        timeout_seconds,
    }
}

#[test]
fn should_plan_sign_submit_staple_validate_and_assess_in_order() {
    let planned_steps = notarization_plan(&notarization_request(Path::new("/release/A.dmg"), 180));

    let mut step_names = Vec::new();
    for planned_step in &planned_steps {
        step_names.push(planned_step.step_name);
    }
    assert_eq!(
        step_names,
        vec![
            "sign-dmg",
            "submit-dmg-for-notarization",
            "staple-notarization-ticket",
            "validate-stapled-ticket",
            "assess-notarized-dmg"
        ]
    );
    assert_eq!(
        planned_steps[0].program_arguments.join(" "),
        "--force --sign Developer ID Application: Example (ABCDE12345) --timestamp /release/A.dmg"
    );
    assert_eq!(
        planned_steps[1].program_arguments.join(" "),
        "notarytool submit /release/A.dmg --keychain-profile astronomical-notary --wait --timeout 180s"
    );
}

#[test]
fn should_default_the_notarization_timeout_and_refuse_bad_values() {
    let parsed_command = parse_release_command(&arguments(&[
        "notarize-dmg",
        "--dmg",
        "/a.dmg",
        "--signing-identity",
        "Dev",
        "--notary-profile",
        "p",
    ]));
    assert_eq!(
        parsed_command,
        Ok(ReleaseCommand::NotarizeDiskImage(NotarizationRequest {
            disk_image: PathBuf::from("/a.dmg"),
            signing_identity: "Dev".to_owned(),
            notary_profile: "p".to_owned(),
            timeout_seconds: DEFAULT_NOTARIZATION_TIMEOUT_SECONDS,
        }))
    );
    assert!(
        parse_release_command(&arguments(&[
            "notarize-dmg",
            "--dmg",
            "/a.dmg",
            "--signing-identity",
            "Dev"
        ]))
        .is_err()
    );
    assert!(
        parse_release_command(&arguments(&[
            "notarize-dmg",
            "--dmg",
            "/a.dmg",
            "--signing-identity",
            "Dev",
            "--notary-profile",
            "p",
            "--timeout-seconds",
            "soon",
        ]))
        .is_err()
    );

    let fixture_directory = tempfile::tempdir().expect("a fixture directory");
    let disk_image = fixture_directory.path().join("A.dmg");
    write_fixture_file(&disk_image, "image");
    assert!(
        validate_notarization_request(&notarization_request(&disk_image, 0))
            .expect_err("zero")
            .contains("positive")
    );
    assert!(
        validate_notarization_request(&notarization_request(
            &fixture_directory.path().join("absent.dmg"),
            180
        ))
        .is_err()
    );
    assert_eq!(
        validate_notarization_request(&notarization_request(&disk_image, 180)),
        Ok(())
    );
}

/// The distribution signature check must look for the agent exactly where the
/// packager puts it, innermost object first.
#[test]
fn should_require_the_stable_agent_before_the_bundle_that_contains_it() {
    let required_paths = super::distribution_signature::required_code_object_paths();
    let stable_profile =
        crate::app_bundle::ChannelProfile::for_channel(crate::app_bundle::Channel::Stable);
    let position_of = |required_path: &str| {
        required_paths
            .iter()
            .position(|listed_path| listed_path == required_path)
    };

    let agent_executable_position = position_of(&stable_profile.agent_executable_relative_path())
        .expect("the agent's executable is required");
    let agent_bundle_position = position_of(&stable_profile.agent_bundle_relative_path())
        .expect("the agent's bundle is required");

    assert!(agent_executable_position < agent_bundle_position);
    assert!(position_of("Contents/MacOS/astronomical-app").is_some());
    assert!(
        position_of("Contents/MacOS/astronomical-menu").is_none(),
        "the agent no longer lives in the main bundle's Contents/MacOS"
    );
    assert_eq!(required_paths.len(), 11);
}
