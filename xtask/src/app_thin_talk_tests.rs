//! `cargo xtask app thin-talk`: its argument rule, the plist that points the
//! standalone chat app at the Development engine, and the canvas check that
//! refuses a build whose conversation would open blank.

use std::fs;
use std::path::Path;

use super::channel_profile::{Channel, ChannelProfile};
use super::command::{AppCommand, DevOpenPlacement, parse_app_command};
use super::thin_talk_pipeline::{
    check_canvas_resources, missing_canvas_client_assets, render_thin_talk_info_plist,
    thin_talk_bundle_path, thin_talk_open_arguments, thin_talk_process_pattern,
};

fn development_plist() -> String {
    render_thin_talk_info_plist(&ChannelProfile::for_channel(Channel::Development))
}

#[test]
fn should_parse_thin_talk_without_arguments() {
    assert_eq!(
        parse_app_command(&["thin-talk".to_owned()]),
        Ok(AppCommand::ThinTalk(DevOpenPlacement::InFront))
    );
    assert_eq!(
        parse_app_command(&["thin-talk".to_owned(), "--background".to_owned()]),
        Ok(AppCommand::ThinTalk(DevOpenPlacement::InBackground))
    );
}

#[test]
fn should_open_behind_other_windows_only_when_asked() {
    let bundle_path = Path::new("/checkout/Thin Talk Development.app");
    assert_eq!(
        thin_talk_open_arguments(bundle_path, DevOpenPlacement::InFront),
        vec![std::ffi::OsString::from(bundle_path)]
    );
    assert_eq!(
        thin_talk_open_arguments(bundle_path, DevOpenPlacement::InBackground),
        vec![
            std::ffi::OsString::from("-g"),
            std::ffi::OsString::from(bundle_path)
        ]
    );
}

#[test]
fn should_refuse_a_channel_or_any_other_argument_for_thin_talk() {
    for refused_argument in ["--channel", "stable", "--front"] {
        match parse_app_command(&["thin-talk".to_owned(), refused_argument.to_owned()]) {
            Err(refusal) => assert!(
                refusal.contains("takes only --background"),
                "{refused_argument}: {refusal}"
            ),
            Ok(parsed_command) => panic!("{refused_argument} was accepted: {parsed_command:?}"),
        }
    }
}

#[test]
fn should_still_refuse_an_unknown_app_subcommand() {
    assert!(parse_app_command(&["thin-talks".to_owned()]).is_err());
}

#[test]
fn should_point_the_plist_at_the_development_engine() {
    let plist_text = development_plist();
    for expected_entry in [
        "<key>AstronomicalChannel</key><string>development</string>",
        "<key>AstronomicalDaemonPort</key><integer>6733</integer>",
        "<key>AstronomicalStateDirectoryName</key><string>.astronomical-dev</string>",
    ] {
        assert!(plist_text.contains(expected_entry), "{expected_entry}");
    }
}

#[test]
fn should_name_the_standalone_executable_identifier_and_icon_in_the_plist() {
    let plist_text = development_plist();
    for expected_entry in [
        "<key>CFBundleName</key><string>Thin Talk Development</string>",
        "<key>CFBundleExecutable</key><string>ThinTalk</string>",
        "<key>CFBundleIdentifier</key><string>dev.astronomical.thin-talk.development</string>",
        "<key>CFBundleIconFile</key><string>Astronomical.icns</string>",
    ] {
        assert!(plist_text.contains(expected_entry), "{expected_entry}");
    }
    assert!(plist_text.starts_with("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"));
    assert!(plist_text.ends_with("</dict>\n</plist>\n"));
}

#[test]
fn should_build_the_bundle_outside_spotlight_under_target() {
    assert_eq!(
        thin_talk_bundle_path(Path::new("/checkout")),
        Path::new("/checkout/target/thin-talk-development.noindex/Thin Talk Development.app")
    );
}

#[test]
fn should_report_both_client_assets_missing_from_an_unbuilt_canvas() {
    let canvas_web_directory = tempfile::tempdir().expect("a canvas folder should be creatable");
    assert_eq!(
        missing_canvas_client_assets(canvas_web_directory.path()),
        vec!["bundle.js", "bundle.css"]
    );
}

#[test]
fn should_count_an_empty_client_asset_as_missing() {
    let canvas_web_directory = tempfile::tempdir().expect("a canvas folder should be creatable");
    fs::write(canvas_web_directory.path().join("bundle.js"), "").expect("writable");
    fs::write(canvas_web_directory.path().join("bundle.css"), "body{}").expect("writable");
    assert_eq!(
        missing_canvas_client_assets(canvas_web_directory.path()),
        vec!["bundle.js"]
    );
}

#[test]
fn should_accept_a_built_canvas() {
    let canvas_web_directory = tempfile::tempdir().expect("a canvas folder should be creatable");
    fs::write(canvas_web_directory.path().join("bundle.js"), "x=1").expect("writable");
    fs::write(canvas_web_directory.path().join("bundle.css"), "body{}").expect("writable");
    assert!(missing_canvas_client_assets(canvas_web_directory.path()).is_empty());
}

#[test]
fn should_refuse_a_build_without_the_canvas_resource_bundle() {
    let release_directory = tempfile::tempdir().expect("a release folder should be creatable");
    let refusal = check_canvas_resources(&release_directory.path().join("Missing.bundle"))
        .expect_err("a missing resource bundle must be refused");
    assert!(
        refusal.contains("canvas resources are unavailable"),
        "{refusal}"
    );
}

#[test]
fn should_name_the_recovery_command_when_the_web_client_was_never_built() {
    let resource_bundle = tempfile::tempdir().expect("a resource bundle should be creatable");
    fs::create_dir_all(resource_bundle.path().join("Contents/Resources/web"))
        .expect("the web folder should be creatable");
    let refusal = check_canvas_resources(resource_bundle.path())
        .expect_err("an unbuilt web client must be refused");
    assert!(
        refusal.starts_with("bundle.js and bundle.css missing"),
        "{refusal}"
    );
    assert!(
        refusal.contains("npm install && npm run build"),
        "{refusal}"
    );
}

#[test]
fn should_match_the_executable_of_every_worktree_build() {
    assert_eq!(
        thin_talk_process_pattern(),
        "/thin-talk-development.noindex/Thin Talk Development.app/Contents/MacOS/ThinTalk"
    );
}
