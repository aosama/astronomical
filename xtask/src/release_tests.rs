use std::path::{Path, PathBuf};

use super::command::{ReleaseCommand, parse_release_command};
use super::disk_image_creation::{
    DiskImageRequest, FINDER_LAYOUT_SCRIPT, partial_output_path, validate_disk_image_request,
};
use super::disk_image_validation::{mounted_layout_problems, validate_disk_image};

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
