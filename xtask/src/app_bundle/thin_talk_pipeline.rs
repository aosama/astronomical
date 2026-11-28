//! `cargo xtask app thin-talk`: builds the standalone Thin Talk chat app from
//! this checkout and opens it, for iterating on the chat window without
//! rebuilding the engine. Typed port of `apps/thin-talk/run-thin-talk.sh`
//! (issue #2635).
//!
//! It only ever talks to the Development instance: a Thin Talk built from a
//! checkout and pointed at Stable looked identical and caused constant
//! confusion, so there is no channel to choose. Its port and state folder come
//! from the Development channel profile, the same facts `app dev` builds with,
//! rather than from copies the shell script had to keep in sync.
//!
//! Thin Talk is a thin client of the running engine's REST endpoints, so the
//! bundle carries only the chat executable, its canvas resources and the
//! Development icon: no engine, worker, Sparkle or MLX library.
//!
//! The web client is rebuilt first, the same step `app dev` runs. Without it
//! the window kept whatever `bundle.js` an earlier build left behind, so a
//! change under `apps/thin-talk/web/src` did not show up and the old screen
//! looked like a broken change.

use std::path::{Path, PathBuf};
use std::process::Command;

use super::build_steps::{
    THIN_TALK_LOCKFILE_STAMP_FILE_NAME, build_thin_talk_web_assets,
    swift_package_product_build_command,
};
use super::bundle_assembly::{copy_file, copy_tree, create_directory, write_file};
use super::channel_profile::{Channel, ChannelProfile};
use super::command::DevOpenPlacement;
use super::info_plist::escape_xml_text;
use super::rendered_icon_memo::rendered_channel_icon;
use super::running_instance::quit_standalone_application;
use super::step_reporter::StepReporter;
use super::tool_command::{run_quiet_tool, run_tool};

const THIN_TALK_PACKAGE_DIRECTORY: &str = "apps/thin-talk";
const THIN_TALK_EXECUTABLE_NAME: &str = "ThinTalk";
const THIN_TALK_BUNDLE_NAME: &str = "Thin Talk Development";
const THIN_TALK_BUNDLE_IDENTIFIER: &str = "dev.astronomical.thin-talk.development";
const THIN_TALK_OUTPUT_DIRECTORY: &str = "target/thin-talk-development.noindex";
/// Shared with `app dev`, so both reuse one rendered Development icon.
const APP_DEV_CACHE_DIRECTORY: &str = "target/app-dev-cache";
const ICON_FILE_NAME: &str = "Astronomical.icns";
const CANVAS_RESOURCE_BUNDLE_NAME: &str = "ThinTalk_ThinTalkCanvas.bundle";
/// The generated web client the canvas renders with. SwiftPM copies
/// `Resources/web` verbatim, so the resource bundle exists even on a fresh
/// clone where the client was never built; without these files the window
/// opens a blank conversation.
const CANVAS_CLIENT_ASSETS: [&str; 2] = ["bundle.js", "bundle.css"];
/// The web client comes before the Swift build: SwiftPM copies
/// `Resources/web` into the canvas resource bundle as it finds it.
pub(crate) const THIN_TALK_STEP_NAMES: [&str; 5] =
    ["thin-talk-web", "swift-build", "assemble", "sign", "open"];

pub(crate) fn run_thin_talk_pipeline(
    repository_root: &Path,
    open_placement: DevOpenPlacement,
) -> Result<(), String> {
    let channel_profile = ChannelProfile::for_channel(Channel::Development);
    let cache_directory = repository_root.join(APP_DEV_CACHE_DIRECTORY);
    create_directory(&cache_directory)?;
    let mut step_reporter = StepReporter::new(
        "app-thin-talk",
        THIN_TALK_STEP_NAMES.to_vec(),
        cache_directory.join("thin-talk-step-timings.json"),
    );
    let package_directory = repository_root.join(THIN_TALK_PACKAGE_DIRECTORY);
    build_web_client_and_executable(
        repository_root,
        &package_directory,
        &cache_directory,
        &mut step_reporter,
    )?;
    step_reporter.start_step("assemble");
    // A running copy would keep the old build on screen: `open` only brings
    // an app that is already running forward.
    quit_standalone_application(
        "Thin Talk",
        THIN_TALK_BUNDLE_IDENTIFIER,
        thin_talk_process_pattern(),
    )?;
    let bundle_path = thin_talk_bundle_path(repository_root);
    let icon_path = rendered_channel_icon(
        repository_root,
        &cache_directory.join("icons"),
        channel_profile.icon_channel,
    )?;
    assemble_thin_talk_bundle(
        &package_directory,
        &bundle_path,
        &icon_path,
        &channel_profile,
    )?;
    step_reporter.finish_step();
    step_reporter.start_step("sign");
    run_quiet_tool(
        Command::new("plutil")
            .arg("-lint")
            .arg(info_plist_path(&bundle_path)),
    )?;
    run_quiet_tool(
        Command::new("codesign")
            .args(["--force", "--sign", "-"])
            .arg(&bundle_path),
    )?;
    step_reporter.finish_step();
    step_reporter.start_step("open");
    run_quiet_tool(
        Command::new("open").args(thin_talk_open_arguments(&bundle_path, open_placement)),
    )?;
    step_reporter.finish_step();
    step_reporter.save_timings();
    println!(
        "[app-thin-talk] {THIN_TALK_BUNDLE_NAME} ready and opened: {}",
        bundle_path.display()
    );
    Ok(())
}

/// The two build steps: the web client the canvas renders, then the Swift
/// executable whose resource bundle copies it.
fn build_web_client_and_executable(
    repository_root: &Path,
    package_directory: &Path,
    cache_directory: &Path,
    step_reporter: &mut StepReporter,
) -> Result<(), String> {
    step_reporter.start_step("thin-talk-web");
    build_thin_talk_web_assets(
        repository_root,
        &cache_directory.join(THIN_TALK_LOCKFILE_STAMP_FILE_NAME),
        step_reporter,
    )?;
    step_reporter.finish_step();
    step_reporter.start_step("swift-build");
    let macos_sdk_path =
        run_quiet_tool(Command::new("xcrun").args(["--sdk", "macosx", "--show-sdk-path"]))?;
    let mut swift_command = swift_package_product_build_command(
        package_directory,
        THIN_TALK_EXECUTABLE_NAME,
        macos_sdk_path.trim(),
    );
    run_tool(&mut swift_command, step_reporter)?;
    step_reporter.finish_step();
    Ok(())
}

/// `open -g` launches the app behind every other window, for a rebuild in the
/// middle of other work.
pub(crate) fn thin_talk_open_arguments(
    bundle_path: &Path,
    open_placement: DevOpenPlacement,
) -> Vec<std::ffi::OsString> {
    match open_placement {
        DevOpenPlacement::InFront => vec![bundle_path.into()],
        DevOpenPlacement::InBackground => vec!["-g".into(), bundle_path.into()],
    }
}

pub(crate) fn thin_talk_bundle_path(repository_root: &Path) -> PathBuf {
    repository_root
        .join(THIN_TALK_OUTPUT_DIRECTORY)
        .join(format!("{THIN_TALK_BUNDLE_NAME}.app"))
}

/// The executable path inside every worktree's Thin Talk Development bundle.
pub(crate) fn thin_talk_process_pattern() -> String {
    let output_directory_name = Path::new(THIN_TALK_OUTPUT_DIRECTORY)
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .unwrap_or(THIN_TALK_OUTPUT_DIRECTORY);
    format!(
        "/{output_directory_name}/{THIN_TALK_BUNDLE_NAME}.app/Contents/MacOS/{THIN_TALK_EXECUTABLE_NAME}"
    )
}

fn info_plist_path(bundle_path: &Path) -> PathBuf {
    bundle_path.join("Contents/Info.plist")
}

/// Lays the bundle out afresh: a stale bundle could keep a resource this
/// build no longer has.
fn assemble_thin_talk_bundle(
    package_directory: &Path,
    bundle_path: &Path,
    icon_path: &Path,
    channel_profile: &ChannelProfile,
) -> Result<(), String> {
    let release_directory = package_directory.join(".build/release");
    let canvas_resource_bundle = release_directory.join(CANVAS_RESOURCE_BUNDLE_NAME);
    check_canvas_resources(&canvas_resource_bundle)?;
    if bundle_path.exists() {
        std::fs::remove_dir_all(bundle_path).map_err(|remove_error| {
            format!("could not remove {}: {remove_error}", bundle_path.display())
        })?;
    }
    let executable_directory = bundle_path.join("Contents/MacOS");
    let resources_directory = bundle_path.join("Contents/Resources");
    create_directory(&executable_directory)?;
    create_directory(&resources_directory)?;
    write_file(
        &info_plist_path(bundle_path),
        &render_thin_talk_info_plist(channel_profile),
    )?;
    copy_file(
        &release_directory.join(THIN_TALK_EXECUTABLE_NAME),
        &executable_directory.join(THIN_TALK_EXECUTABLE_NAME),
    )?;
    copy_tree(
        &canvas_resource_bundle,
        &resources_directory.join(CANVAS_RESOURCE_BUNDLE_NAME),
    )?;
    copy_file(icon_path, &resources_directory.join(ICON_FILE_NAME))
}

/// Refuses a build whose canvas would open blank, naming how to recover.
pub(crate) fn check_canvas_resources(canvas_resource_bundle: &Path) -> Result<(), String> {
    if !canvas_resource_bundle.is_dir() {
        return Err(format!(
            "conversation canvas resources are unavailable at {}",
            canvas_resource_bundle.display()
        ));
    }
    let missing_assets =
        missing_canvas_client_assets(&canvas_resource_bundle.join("Contents/Resources/web"));
    if missing_assets.is_empty() {
        return Ok(());
    }
    Err(format!(
        "{} missing from the canvas resources. The web client is generated, not committed; build it once with:\n  (cd apps/thin-talk/web && npm install && npm run build)",
        missing_assets.join(" and ")
    ))
}

/// The client assets that are absent or empty under the canvas web folder.
pub(crate) fn missing_canvas_client_assets(canvas_web_directory: &Path) -> Vec<&'static str> {
    let mut missing_assets = Vec::new();
    for client_asset in CANVAS_CLIENT_ASSETS {
        let asset_is_present = match std::fs::metadata(canvas_web_directory.join(client_asset)) {
            Ok(asset_metadata) => asset_metadata.len() > 0,
            Err(_) => false,
        };
        if !asset_is_present {
            missing_assets.push(client_asset);
        }
    }
    missing_assets
}

/// The standalone app's plist. The `Astronomical*` keys are how Thin Talk
/// finds the engine it talks to, so they come from the Development profile.
pub(crate) fn render_thin_talk_info_plist(channel_profile: &ChannelProfile) -> String {
    let state_directory_name = channel_profile.state_directory_name.unwrap_or_default();
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n\
         <plist version=\"1.0\">\n<dict>\n\
         \x20 <key>CFBundleName</key><string>{bundle_name}</string>\n\
         \x20 <key>CFBundleDisplayName</key><string>{bundle_name}</string>\n\
         \x20 <key>CFBundleExecutable</key><string>{THIN_TALK_EXECUTABLE_NAME}</string>\n\
         \x20 <key>CFBundleIdentifier</key><string>{THIN_TALK_BUNDLE_IDENTIFIER}</string>\n\
         \x20 <key>CFBundleVersion</key><string>1</string>\n\
         \x20 <key>CFBundleShortVersionString</key><string>0.1.0</string>\n\
         \x20 <key>LSMinimumSystemVersion</key><string>14.0</string>\n\
         \x20 <key>CFBundleIconFile</key><string>{ICON_FILE_NAME}</string>\n\
         \x20 <key>AstronomicalChannel</key><string>{channel_name}</string>\n\
         \x20 <key>AstronomicalDaemonPort</key><integer>{daemon_port}</integer>\n\
         \x20 <key>AstronomicalStateDirectoryName</key><string>{state_directory}</string>\n\
         \x20 <key>CFBundlePackageType</key><string>APPL</string>\n\
         </dict>\n</plist>\n",
        bundle_name = escape_xml_text(THIN_TALK_BUNDLE_NAME),
        channel_name = channel_profile.channel.as_channel_name(),
        daemon_port = channel_profile.daemon_port,
        state_directory = escape_xml_text(state_directory_name),
    )
}
