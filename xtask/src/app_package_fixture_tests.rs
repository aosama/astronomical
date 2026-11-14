//! Checks run against a fixture bundle assembled by the real assembly code,
//! so layout, hidden-entry and agent checks see what the pipeline writes.

use std::path::{Path, PathBuf};

use super::app_package_tests::{DEVELOPER_ID, release_identity};
use super::application_group::{application_group_identifier, stamp_application_group};
use super::bundle_assembly::{BundleInputs, BundlePlists, assemble_bundle, missing_inputs};
use super::bundle_check::{bundled_agent_problems, hidden_entry_problems, layout_problems};
use super::channel_profile::{Channel, ChannelProfile};
use super::info_plist::{render_agent_info_plist, render_info_plist};

/// Assembles a fixture bundle through the real assembly code so layout
/// checks run against what the pipeline actually writes.
fn assembled_fixture_bundle(channel: Channel) -> (tempfile::TempDir, PathBuf) {
    let fixture_directory = tempfile::tempdir().expect("a fixture directory");
    let fixture_root = fixture_directory.path();
    let channel_profile = ChannelProfile::for_channel(channel);
    let swift_build = fixture_root.join("apps/astronomical-menu/.build");
    write_fixture_file(
        &swift_build
            .join("release")
            .join(channel_profile.application_swift_product),
        "application",
        true,
    );
    write_fixture_file(
        &swift_build.join("release/astronomical-menu"),
        "agent",
        true,
    );
    write_fixture_file(
        &swift_build.join("release/ThinTalk_ThinTalkCanvas.bundle/index.html"),
        "<html>",
        false,
    );
    write_fixture_file(
        &swift_build.join("checkouts/Sparkle/LICENSE"),
        "sparkle",
        false,
    );
    write_fixture_file(
        &swift_build.join("release/Sparkle.framework/Versions/B/Sparkle"),
        "sparkle",
        false,
    );
    for notice_path in [
        "LICENSE",
        "third-party/THIRD_PARTY_NOTICES",
        "third-party/RUST_DEPENDENCY_NOTICES",
        "mlx.metallib",
        "icon.icns",
        "scripts/internal/privacy/PrivacyInfo.xcprivacy",
    ] {
        write_fixture_file(&fixture_root.join(notice_path), "notice", false);
    }
    for executable_name in [
        "astronomicald",
        "astronomical-inference-worker",
        "astronomical",
    ] {
        write_fixture_file(
            &fixture_root.join("release").join(executable_name),
            "binary",
            true,
        );
    }
    let bundle_inputs = BundleInputs::for_repository(
        fixture_root,
        &channel_profile,
        &fixture_root.join("release"),
        fixture_root.join("mlx.metallib"),
        fixture_root.join("icon.icns"),
    );
    assert!(
        missing_inputs(&bundle_inputs).is_empty(),
        "{:?}",
        missing_inputs(&bundle_inputs)
    );
    let bundle_path = fixture_root
        .join("out")
        .join(channel_profile.bundle_directory_name());
    let plist_text = render_info_plist(
        &channel_profile,
        &release_identity(false),
        "https://example-owner.github.io/astronomical/appcast.xml",
    );
    assemble_bundle_without_rpath(&bundle_path, &bundle_inputs, &plist_text, &channel_profile);
    (fixture_directory, bundle_path)
}

/// The two plists of a fixture bundle. A sandboxed channel's plists both name
/// the app group, as packaging stamps them once the signing identity is known.
fn fixture_plist_texts(
    application_plist_text: &str,
    channel_profile: &ChannelProfile,
) -> (String, String) {
    let agent_plist_text = render_agent_info_plist(channel_profile, &release_identity(false));
    match application_group_identifier(channel_profile, Some(DEVELOPER_ID))
        .expect("the fixture identity names a team")
    {
        Some(group_identifier) => (
            stamp_application_group(application_plist_text, &group_identifier)
                .expect("the fixture plist is stampable"),
            stamp_application_group(&agent_plist_text, &group_identifier)
                .expect("the fixture agent plist is stampable"),
        ),
        None => (application_plist_text.to_owned(), agent_plist_text),
    }
}

/// `install_name_tool` cannot edit a text fixture, so the Sparkle
/// framework input is dropped before assembly and copied afterwards; the
/// layout under test is otherwise exactly what `assemble_bundle` writes.
fn assemble_bundle_without_rpath(
    bundle_path: &Path,
    bundle_inputs: &BundleInputs,
    plist_text: &str,
    channel_profile: &ChannelProfile,
) {
    let mut rpath_free_inputs = bundle_inputs.clone();
    rpath_free_inputs.sparkle_framework = None;
    let (application_plist_text, agent_plist_text) =
        fixture_plist_texts(plist_text, channel_profile);
    assemble_bundle(
        bundle_path,
        &rpath_free_inputs,
        &BundlePlists {
            application_info_plist_text: &application_plist_text,
            agent_info_plist_text: &agent_plist_text,
            agent_bundle_relative_path: &channel_profile.agent_bundle_relative_path(),
        },
    )
    .expect("the fixture bundle should assemble");
    if channel_profile.ships_sparkle {
        write_fixture_file(
            &bundle_path.join("Contents/Frameworks/Sparkle.framework/Versions/B/Sparkle"),
            "sparkle",
            false,
        );
    }
}

fn write_fixture_file(file_path: &Path, file_text: &str, is_executable: bool) {
    std::fs::create_dir_all(file_path.parent().expect("a parent")).expect("fixture directory");
    std::fs::write(file_path, file_text).expect("fixture file");
    if is_executable {
        let executable_permissions = std::os::unix::fs::PermissionsExt::from_mode(0o755);
        std::fs::set_permissions(file_path, executable_permissions).expect("executable bit");
    }
}

#[test]
fn should_find_no_layout_problem_in_freshly_assembled_bundles() {
    for channel in [Channel::Development, Channel::Stable, Channel::AppStore] {
        let (_fixture_directory, bundle_path) = assembled_fixture_bundle(channel);

        assert!(
            layout_problems(&bundle_path, &ChannelProfile::for_channel(channel)).is_empty(),
            "{channel:?}"
        );
    }
}

#[test]
fn should_report_a_missing_metallib_executable_bit_and_stray_sparkle() {
    let (_fixture_directory, bundle_path) = assembled_fixture_bundle(Channel::Stable);
    std::fs::remove_file(bundle_path.join("Contents/Resources/share/mlx/mlx.metallib"))
        .expect("remove metallib");
    let plain_permissions = std::os::unix::fs::PermissionsExt::from_mode(0o644);
    std::fs::set_permissions(
        bundle_path.join("Contents/MacOS/astronomicald"),
        plain_permissions,
    )
    .expect("drop the executable bit");

    let stable_problems =
        layout_problems(&bundle_path, &ChannelProfile::for_channel(Channel::Stable));
    let store_problems = layout_problems(
        &bundle_path,
        &ChannelProfile::for_channel(Channel::AppStore),
    );

    assert!(
        stable_problems.contains(&"required bundled MLX AOT metallib is unavailable".to_owned()),
        "{stable_problems:?}"
    );
    assert!(
        stable_problems.contains(&"bundled executable is unavailable: astronomicald".to_owned()),
        "{stable_problems:?}"
    );
    assert!(
        store_problems
            .contains(&"an App Store bundle must not embed Sparkle or its license".to_owned()),
        "{store_problems:?}"
    );
}

/// Both applications of a store bundle carry the privacy manifest, a direct
/// bundle carries none, and a store bundle that lost one is refused.
#[test]
fn should_require_the_privacy_manifest_in_both_store_applications_only() {
    let manifest_relative_paths = [
        "Contents/Resources/PrivacyInfo.xcprivacy",
        "Contents/Library/LoginItems/Astronomical Agent.app/Contents/Resources/PrivacyInfo.xcprivacy",
    ];
    let (_store_fixture_directory, store_bundle_path) = assembled_fixture_bundle(Channel::AppStore);
    let (_stable_fixture_directory, stable_bundle_path) = assembled_fixture_bundle(Channel::Stable);
    for manifest_relative_path in manifest_relative_paths {
        assert!(
            store_bundle_path.join(manifest_relative_path).is_file(),
            "{manifest_relative_path}"
        );
        assert!(
            !stable_bundle_path.join(manifest_relative_path).exists(),
            "{manifest_relative_path}"
        );
    }
    std::fs::remove_file(store_bundle_path.join(manifest_relative_paths[1]))
        .expect("remove the agent's manifest");

    let store_problems = layout_problems(
        &store_bundle_path,
        &ChannelProfile::for_channel(Channel::AppStore),
    );

    assert_eq!(
        store_problems,
        ["the menu-bar agent of an App Store bundle is missing its privacy manifest"]
    );
}

/// A store bundle is refused when a terminal command is found in it, and a
/// direct bundle is refused when its terminal command is missing.
#[test]
fn should_refuse_a_terminal_command_in_the_store_bundle_and_require_it_elsewhere() {
    let (_store_fixture_directory, store_bundle_path) = assembled_fixture_bundle(Channel::AppStore);
    let store_profile = ChannelProfile::for_channel(Channel::AppStore);
    let terminal_command_path = store_bundle_path.join("Contents/MacOS/astronomical");
    assert!(
        !terminal_command_path.exists(),
        "assembly must leave the terminal command out of a store bundle"
    );
    write_fixture_file(&terminal_command_path, "binary", true);

    let store_problems = layout_problems(&store_bundle_path, &store_profile);

    assert!(
        store_problems.contains(
            &"this channel's bundle must not carry the terminal command, which cannot be sandboxed"
                .to_owned()
        ),
        "{store_problems:?}"
    );

    let (_stable_fixture_directory, stable_bundle_path) = assembled_fixture_bundle(Channel::Stable);
    std::fs::remove_file(stable_bundle_path.join("Contents/MacOS/astronomical"))
        .expect("remove the terminal command");

    let stable_problems = layout_problems(
        &stable_bundle_path,
        &ChannelProfile::for_channel(Channel::Stable),
    );

    assert!(
        stable_problems.contains(&"bundled executable is unavailable: astronomical".to_owned()),
        "{stable_problems:?}"
    );
}

/// The v0.2.88 leak: a git-ignored Playwright folder inside the Thin Talk
/// web resources was copied whole into the shipped bundle.
#[test]
fn should_report_a_leaked_playwright_folder_inside_the_canvas_resources() {
    let (_fixture_directory, bundle_path) = assembled_fixture_bundle(Channel::Stable);
    let leaked_folder =
        bundle_path.join("Contents/Resources/ThinTalk_ThinTalkCanvas.bundle/.playwright-cli");
    write_fixture_file(&leaked_folder.join("console.log"), "debug", false);

    let stable_problems =
        layout_problems(&bundle_path, &ChannelProfile::for_channel(Channel::Stable));

    assert_eq!(
        stable_problems,
        vec![
            "hidden entry inside the bundle: Contents/Resources/ThinTalk_ThinTalkCanvas.bundle/.playwright-cli"
                .to_owned()
        ]
    );
}

#[test]
fn should_report_a_hidden_file_at_any_depth_of_the_bundle() {
    let (_fixture_directory, bundle_path) = assembled_fixture_bundle(Channel::Development);
    write_fixture_file(&bundle_path.join("Contents/.DS_Store"), "finder", false);

    assert_eq!(
        hidden_entry_problems(&bundle_path),
        vec!["hidden entry inside the bundle: Contents/.DS_Store".to_owned()]
    );
}

/// Sparkle's framework links `Versions/Current`; following symlinks could
/// walk out of the bundle and blame it for files it does not contain.
#[test]
fn should_not_follow_a_symlink_out_of_the_bundle() {
    let (fixture_directory, bundle_path) = assembled_fixture_bundle(Channel::Stable);
    let outside_directory = fixture_directory.path().join("outside");
    write_fixture_file(&outside_directory.join(".hidden"), "outside", false);
    std::os::unix::fs::symlink(&outside_directory, bundle_path.join("Contents/linked"))
        .expect("symlink out of the bundle");

    assert!(hidden_entry_problems(&bundle_path).is_empty());
}

#[test]
fn should_cap_the_hidden_entry_report() {
    let (_fixture_directory, bundle_path) = assembled_fixture_bundle(Channel::Stable);
    for hidden_index in 0..15 {
        write_fixture_file(
            &bundle_path.join(format!("Contents/Resources/.stray-{hidden_index:02}")),
            "stray",
            false,
        );
    }

    assert_eq!(hidden_entry_problems(&bundle_path).len(), 10);
}

/// The main plist a fixture bundle of `channel` carries, as the checker reads it.
#[cfg(target_os = "macos")]
fn fixture_application_plist_document(bundle_path: &Path) -> serde_json::Value {
    let plist_json_output = std::process::Command::new("plutil")
        .args(["-convert", "json", "-o", "-"])
        .arg(bundle_path.join("Contents/Info.plist"))
        .output()
        .expect("plutil should run on macOS");
    serde_json::from_slice(&plist_json_output.stdout).expect("the fixture plist converts to JSON")
}

#[cfg(target_os = "macos")]
#[test]
fn should_find_no_agent_problem_in_an_assembled_bundle_of_any_channel() {
    for channel in [Channel::Development, Channel::Stable, Channel::AppStore] {
        let (_fixture_directory, bundle_path) = assembled_fixture_bundle(channel);

        let agent_problems = bundled_agent_problems(
            &bundle_path,
            &fixture_application_plist_document(&bundle_path),
            &ChannelProfile::for_channel(channel),
        );

        assert_eq!(agent_problems, Ok(Vec::new()), "{channel:?}");
    }
}

/// One run must list everything wrong: a bundle that lost its agent plist is
/// one more problem, not a tool error that hides the others.
#[cfg(target_os = "macos")]
#[test]
fn should_report_a_missing_agent_plist_as_a_problem_rather_than_a_tool_error() {
    let (_fixture_directory, bundle_path) = assembled_fixture_bundle(Channel::Stable);
    let stable_profile = ChannelProfile::for_channel(Channel::Stable);
    let application_plist_document = fixture_application_plist_document(&bundle_path);
    std::fs::remove_file(
        bundle_path
            .join(stable_profile.agent_bundle_relative_path())
            .join("Contents/Info.plist"),
    )
    .expect("the fixture agent plist should be removable");

    assert_eq!(
        bundled_agent_problems(&bundle_path, &application_plist_document, &stable_profile),
        Ok(vec![
            "the menu-bar agent's Info.plist is unavailable".to_owned()
        ])
    );
}

#[test]
fn should_report_a_bundle_whose_menu_bar_agent_is_missing() {
    let (_fixture_directory, bundle_path) = assembled_fixture_bundle(Channel::Stable);
    let stable_profile = ChannelProfile::for_channel(Channel::Stable);
    assert!(layout_problems(&bundle_path, &stable_profile).is_empty());

    std::fs::remove_file(bundle_path.join(stable_profile.agent_executable_relative_path()))
        .expect("the fixture agent executable should be removable");

    assert_eq!(
        layout_problems(&bundle_path, &stable_profile),
        vec!["the menu-bar agent's executable is unavailable".to_owned()]
    );
}
