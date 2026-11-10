//! Contracts of the two-process bundle: the main app is a regular
//! application, the menu-bar agent nested inside it is an accessory, and the
//! two belong to the same instance. Neither ever changes role while running,
//! which is what keeps the main app's menu bar reachable.

use super::build_identity::BuildIdentity;
use super::bundle_check::process_role_problems;
use super::channel_profile::{
    AGENT_EXECUTABLE_NAME, APPLICATION_EXECUTABLE_NAME, Channel, ChannelProfile,
};
use super::info_plist::{render_agent_info_plist, render_info_plist};
use super::running_instance::{agent_process_pattern, application_process_pattern};
use super::smoke_test::agent_executable_path;

const EVERY_CHANNEL: [Channel; 3] = [Channel::Development, Channel::Stable, Channel::AppStore];

fn sample_build_identity() -> BuildIdentity {
    BuildIdentity {
        application_version: "0.2.88".to_owned(),
        repository_url: "https://github.com/example-owner/astronomical".to_owned(),
        build_commit: "49af41bd9bba".to_owned(),
        build_number: "4242".to_owned(),
        build_date_utc: "20261102".to_owned(),
        is_dirty: false,
    }
}

/// The plist documents a correct bundle of `channel` carries, as the checker
/// reads them.
fn matching_plist_documents(channel: Channel) -> (serde_json::Value, serde_json::Value) {
    let channel_profile = ChannelProfile::for_channel(channel);
    let shared_identity = serde_json::json!({
        "AstronomicalChannel": channel_profile.channel.as_channel_name(),
        "AstronomicalDaemonPort": channel_profile.daemon_port,
        "AstronomicalStateDirectoryName": channel_profile.state_directory_name,
        "CFBundleVersion": "4242",
        "CFBundleShortVersionString": "0.2.88",
        "AstronomicalBuildCommit": "49af41bd9bba",
        "AstronomicalBuildDirty": false,
    });
    let mut application_document = shared_identity.clone();
    application_document["CFBundleExecutable"] = serde_json::json!(APPLICATION_EXECUTABLE_NAME);
    application_document["CFBundleIdentifier"] =
        serde_json::json!(channel_profile.bundle_identifier);
    let mut agent_document = shared_identity;
    agent_document["CFBundleExecutable"] = serde_json::json!(AGENT_EXECUTABLE_NAME);
    agent_document["CFBundleIdentifier"] =
        serde_json::json!(channel_profile.agent_bundle_identifier());
    agent_document["LSUIElement"] = serde_json::json!(true);
    (application_document, agent_document)
}

#[test]
fn should_render_the_main_plist_as_a_regular_application_with_its_own_executable() {
    for channel in EVERY_CHANNEL {
        let plist_text = render_info_plist(
            &ChannelProfile::for_channel(channel),
            &sample_build_identity(),
            "https://example-owner.github.io/astronomical/appcast.xml",
        );

        assert!(!plist_text.contains("LSUIElement"), "{channel:?}");
        assert!(
            plist_text.contains("<key>CFBundleExecutable</key><string>astronomical-app</string>"),
            "{channel:?}"
        );
    }
}

#[test]
fn should_render_the_agent_plist_as_an_accessory_of_the_same_instance() {
    let agent_plist_text = render_agent_info_plist(
        &ChannelProfile::for_channel(Channel::Development),
        &sample_build_identity(),
    );

    for expected_fragment in [
        "<key>LSUIElement</key><true/>",
        "<key>CFBundleExecutable</key><string>astronomical-menu</string>",
        "<key>CFBundleIdentifier</key><string>dev.astronomical.app.development.agent</string>",
        "<key>CFBundleName</key><string>Astronomical Development Agent</string>",
        "<key>AstronomicalChannel</key><string>development</string>",
        "<key>AstronomicalDaemonPort</key><integer>6733</integer>",
        "<key>AstronomicalStateDirectoryName</key><string>.astronomical-dev</string>",
        "<key>CFBundleVersion</key><string>4242</string>",
    ] {
        assert!(
            agent_plist_text.contains(expected_fragment),
            "{expected_fragment}"
        );
    }
}

/// Links and updates belong to the main app: an agent that registered the
/// chat scheme could receive the link meant to open a window, and one that
/// carried Sparkle keys would suggest it updates itself.
#[test]
fn should_keep_links_and_update_keys_out_of_the_agent_plist_on_every_channel() {
    for channel in EVERY_CHANNEL {
        let agent_plist_text = render_agent_info_plist(
            &ChannelProfile::for_channel(channel),
            &sample_build_identity(),
        );

        assert!(
            !agent_plist_text.contains("CFBundleURLTypes"),
            "{channel:?}"
        );
        assert!(!agent_plist_text.contains("SUFeedURL"), "{channel:?}");
    }
}

#[test]
fn should_omit_the_state_directory_from_the_app_store_agent_plist() {
    let agent_plist_text = render_agent_info_plist(
        &ChannelProfile::for_channel(Channel::AppStore),
        &sample_build_identity(),
    );

    assert!(!agent_plist_text.contains("AstronomicalStateDirectoryName"));
}

#[cfg(target_os = "macos")]
#[test]
fn should_render_an_agent_plist_that_plutil_accepts_for_every_channel() {
    let plist_directory = tempfile::tempdir().expect("a temporary directory should be available");
    for channel in EVERY_CHANNEL {
        let plist_path = plist_directory
            .path()
            .join(format!("{}-agent.plist", channel.as_channel_name()));
        std::fs::write(
            &plist_path,
            render_agent_info_plist(
                &ChannelProfile::for_channel(channel),
                &sample_build_identity(),
            ),
        )
        .expect("the plist should be writable");

        let lint_output = std::process::Command::new("plutil")
            .arg("-lint")
            .arg(&plist_path)
            .output()
            .expect("plutil should run on macOS");

        assert!(
            lint_output.status.success(),
            "plutil rejected the {channel:?} agent plist"
        );
    }
}

#[test]
fn should_derive_the_agent_name_identifier_and_place_from_the_channel() {
    let stable_profile = ChannelProfile::for_channel(Channel::Stable);
    let store_profile = ChannelProfile::for_channel(Channel::AppStore);

    assert_eq!(stable_profile.agent_bundle_name(), "Astronomical Agent");
    assert_eq!(
        stable_profile.agent_bundle_identifier(),
        "dev.astronomical.app.agent"
    );
    assert_eq!(
        stable_profile.agent_executable_relative_path(),
        "Contents/Library/LoginItems/Astronomical Agent.app/Contents/MacOS/astronomical-menu"
    );
    // The store build shares Stable's name, yet its helper must extend the
    // store identifier, not Stable's.
    assert_eq!(
        store_profile.agent_bundle_identifier(),
        "app.astronomical.appstore.agent"
    );
}

#[test]
fn should_accept_plists_of_one_instance_on_every_channel() {
    for channel in EVERY_CHANNEL {
        let (application_document, agent_document) = matching_plist_documents(channel);

        assert_eq!(
            process_role_problems(
                &application_document,
                &agent_document,
                &ChannelProfile::for_channel(channel)
            ),
            Vec::<String>::new(),
            "{channel:?}"
        );
    }
}

/// The defect this layout exists to prevent: a main app marked LSUIElement
/// has no menu bar of its own.
#[test]
fn should_refuse_a_main_app_marked_as_an_agent_and_an_agent_that_is_not() {
    let stable_profile = ChannelProfile::for_channel(Channel::Stable);
    let (mut application_document, mut agent_document) = matching_plist_documents(Channel::Stable);
    application_document["LSUIElement"] = serde_json::json!(true);
    agent_document["LSUIElement"] = serde_json::json!(false);

    let role_problems =
        process_role_problems(&application_document, &agent_document, &stable_profile);

    assert!(
        role_problems
            .contains(&"the main app must be a regular application, not LSUIElement".to_owned())
    );
    assert!(role_problems.contains(&"the menu-bar agent must be LSUIElement".to_owned()));
}

#[test]
fn should_refuse_an_agent_of_another_instance_or_another_build() {
    let stable_profile = ChannelProfile::for_channel(Channel::Stable);
    let (application_document, mut agent_document) = matching_plist_documents(Channel::Stable);
    agent_document["AstronomicalDaemonPort"] = serde_json::json!(6733);
    agent_document["AstronomicalBuildCommit"] = serde_json::json!("000000000000");
    agent_document["CFBundleIdentifier"] =
        serde_json::json!("dev.astronomical.app.development.agent");

    let role_problems =
        process_role_problems(&application_document, &agent_document, &stable_profile);

    for expected_problem in [
        "the agent and the main app disagree on AstronomicalDaemonPort",
        "the agent and the main app disagree on AstronomicalBuildCommit",
        "the agent's CFBundleIdentifier does not match the bundle",
    ] {
        assert!(
            role_problems.contains(&expected_problem.to_owned()),
            "{expected_problem}: {role_problems:?}"
        );
    }
}

#[test]
fn should_refuse_an_agent_that_registers_a_link_scheme() {
    let stable_profile = ChannelProfile::for_channel(Channel::Stable);
    let (application_document, mut agent_document) = matching_plist_documents(Channel::Stable);
    agent_document["CFBundleURLTypes"] =
        serde_json::json!([{"CFBundleURLSchemes": ["astronomical"]}]);

    let role_problems =
        process_role_problems(&application_document, &agent_document, &stable_profile);

    assert_eq!(
        role_problems,
        vec!["links belong to the main app; the agent must register none".to_owned()]
    );
}

/// Quitting one of the two must never wait on, or count, the other: the
/// patterns are matched against whole command lines by `pgrep -f`.
#[test]
fn should_tell_the_main_app_and_the_agent_apart_by_command_line() {
    for channel in EVERY_CHANNEL {
        let channel_profile = ChannelProfile::for_channel(channel);
        let bundle_root = format!(
            "/worktree/target/{}/{}.app",
            channel_profile.output_directory_name, channel_profile.bundle_name
        );
        let application_command_line = format!("{bundle_root}/Contents/MacOS/astronomical-app");
        let agent_command_line = format!(
            "{bundle_root}/{}",
            channel_profile.agent_executable_relative_path()
        );
        let terminal_command_line = format!("{bundle_root}/Contents/MacOS/astronomical status");

        let application_pattern = application_process_pattern(&channel_profile);
        let agent_pattern = agent_process_pattern(&channel_profile);

        assert!(application_command_line.contains(&application_pattern));
        assert!(agent_command_line.contains(&agent_pattern));
        assert!(!agent_command_line.contains(&application_pattern));
        assert!(!application_command_line.contains(&agent_pattern));
        assert!(
            !terminal_command_line.contains(&application_pattern),
            "the terminal command must not be mistaken for the main app"
        );
    }
}

fn write_executable_fixture(executable_path: &std::path::Path) {
    std::fs::create_dir_all(executable_path.parent().expect("a parent directory"))
        .expect("the fixture directory should be creatable");
    std::fs::write(executable_path, "fixture").expect("the fixture should be writable");
}

/// The smoke test is handed only a bundle path, so it finds the agent by
/// looking in LoginItems rather than by a channel's name.
#[test]
fn should_find_the_agent_executable_whatever_the_agent_bundle_is_called() {
    let bundle_directory = tempfile::tempdir().expect("a temporary directory");
    let agent_executable = bundle_directory
        .path()
        .join("Contents/Library/LoginItems/Any Name Agent.app/Contents/MacOS/astronomical-menu");
    write_executable_fixture(&agent_executable);

    assert_eq!(
        agent_executable_path(bundle_directory.path()),
        Ok(agent_executable)
    );
}

#[test]
fn should_say_a_bundle_has_no_agent_when_login_items_is_missing_or_holds_none() {
    let bundle_without_login_items = tempfile::tempdir().expect("a temporary directory");
    let bundle_with_other_helper = tempfile::tempdir().expect("a temporary directory");
    write_executable_fixture(
        &bundle_with_other_helper
            .path()
            .join("Contents/Library/LoginItems/Other.app/Contents/MacOS/some-other-helper"),
    );

    let missing_directory_error = agent_executable_path(bundle_without_login_items.path())
        .expect_err("a bundle without LoginItems has no agent");
    let missing_agent_error = agent_executable_path(bundle_with_other_helper.path())
        .expect_err("a helper that is not the agent does not count");

    assert!(missing_directory_error.contains("the bundle has no menu-bar agent"));
    assert!(missing_agent_error.contains("no menu-bar agent executable under"));
}
