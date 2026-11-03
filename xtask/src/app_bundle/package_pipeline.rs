//! `cargo xtask app package --channel stable|app-store [--signing-identity NAME]`:
//! builds a release bundle from a clean commit. Typed port of the retired
//! `build-macos-app.sh` release path.
//!
//! The command re-enters itself inside the `app-release` disposable target
//! lane, as the shell builder did: several Cargo invocations must share one
//! owned release target directory that is removed afterwards, so a release
//! never links against artifacts a previous build left behind.

use std::path::{Path, PathBuf};
use std::process::Command;

use super::build_identity::{BuildIdentity, github_pages_feed_url};
use super::bundle_assembly::{BundleInputs, assemble_bundle, missing_inputs};
use super::bundle_check::check_bundle;
use super::channel_profile::{Channel, ChannelProfile};
use super::code_signing::{SigningRequest, sign_bundle};
use super::icon_cache::cached_channel_icon;
use super::info_plist::render_info_plist;
use super::native_metallib::packaged_metallib_source;
use super::release_build_steps::ReleaseBuildContext;
use super::step_reporter::StepReporter;
use super::tool_command::run_quiet_tool;

const DISPOSABLE_LIFECYCLE_VARIABLE: &str = "ASTRONOMICAL_CARGO_TARGET_LIFECYCLE";
const RELEASE_LANE_NAME: &str = "app-release";

pub(crate) const PACKAGE_STEP_NAMES: [&str; 8] = [
    "build-identity",
    "native-dependencies",
    "native-runtime",
    "rust-executables",
    "thin-talk-web",
    "menu-app",
    "assemble-and-sign",
    "check-bundle",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PackageRequest {
    pub(crate) channel: Channel,
    pub(crate) signing_identity: Option<String>,
}

/// The Development channel is built by `app dev`; packaging is for the
/// channels that ship.
pub(crate) fn validate_package_request(package_request: &PackageRequest) -> Result<(), String> {
    match package_request.channel {
        Channel::Development => Err(
            "app package builds stable or app-store; build Development with cargo xtask app dev"
                .to_owned(),
        ),
        Channel::Stable | Channel::AppStore => Ok(()),
    }
}

pub(crate) fn run_package(
    repository_root: &Path,
    package_request: &PackageRequest,
) -> Result<(), String> {
    validate_package_request(package_request)?;
    if std::env::var(DISPOSABLE_LIFECYCLE_VARIABLE).as_deref() != Ok("disposable") {
        return reenter_in_disposable_target(repository_root, package_request);
    }
    let Ok(cargo_target_directory) = std::env::var("CARGO_TARGET_DIR") else {
        return Err("a disposable release build is missing CARGO_TARGET_DIR".to_owned());
    };
    build_release_bundle(
        repository_root,
        package_request,
        Path::new(&cargo_target_directory),
    )
}

pub(crate) fn disposable_reentry_arguments(
    repository_root: &Path,
    package_request: &PackageRequest,
    xtask_executable: &Path,
) -> Vec<String> {
    let mut reentry_arguments = Vec::new();
    for argument_word in [
        "xtask",
        "disposable-target",
        "--lane",
        RELEASE_LANE_NAME,
        "--",
    ] {
        reentry_arguments.push(argument_word.to_owned());
    }
    reentry_arguments.push(xtask_executable.display().to_string());
    reentry_arguments.push("--repository-root".to_owned());
    reentry_arguments.push(repository_root.display().to_string());
    for argument_word in [
        "app",
        "package",
        "--channel",
        package_request.channel.as_channel_name(),
    ] {
        reentry_arguments.push(argument_word.to_owned());
    }
    if let Some(signing_identity) = &package_request.signing_identity {
        reentry_arguments.push("--signing-identity".to_owned());
        reentry_arguments.push(signing_identity.clone());
    }
    reentry_arguments
}

fn reenter_in_disposable_target(
    repository_root: &Path,
    package_request: &PackageRequest,
) -> Result<(), String> {
    let xtask_executable = match std::env::current_exe() {
        Ok(xtask_executable) => xtask_executable,
        Err(executable_error) => {
            return Err(format!(
                "could not locate the xtask executable: {executable_error}"
            ));
        }
    };
    let reentry_status = Command::new("cargo")
        .args(disposable_reentry_arguments(
            repository_root,
            package_request,
            &xtask_executable,
        ))
        .current_dir(repository_root)
        .status();
    match reentry_status {
        Ok(exit_status) if exit_status.success() => Ok(()),
        Ok(exit_status) => Err(format!(
            "the release build failed inside the disposable target ({exit_status})"
        )),
        Err(spawn_error) => Err(format!(
            "could not start the disposable release build: {spawn_error}"
        )),
    }
}

fn build_release_bundle(
    repository_root: &Path,
    package_request: &PackageRequest,
    cargo_target_directory: &Path,
) -> Result<(), String> {
    let channel_profile = ChannelProfile::for_channel(package_request.channel);
    let timings_directory = repository_root.join("target/app-package-cache");
    create_directory(&timings_directory)?;
    let mut step_reporter = StepReporter::new(
        "app-package",
        PACKAGE_STEP_NAMES.to_vec(),
        timings_directory.join(format!(
            "{}-step-timings.json",
            package_request.channel.as_channel_name()
        )),
    );
    step_reporter.start_step("build-identity");
    let build_identity = BuildIdentity::gather(repository_root)?;
    if channel_profile.requires_clean_worktree && build_identity.is_dirty {
        return Err(format!(
            "{} builds require a clean worktree; build Development or commit the intended changes",
            package_request.channel.as_channel_name()
        ));
    }
    step_reporter.finish_step();
    let release_build = ReleaseBuildContext {
        repository_root,
        channel_profile: &channel_profile,
        build_identity: &build_identity,
        host_target_triple: &host_target_triple()?,
        cargo_target_directory,
        logical_cpu_count: run_quiet_tool(Command::new("sysctl").args(["-n", "hw.logicalcpu"]))?,
    };
    run_release_compile_steps(&release_build, &mut step_reporter)?;
    let bundle_path =
        assemble_and_sign_release_bundle(&release_build, package_request, &mut step_reporter)?;
    step_reporter.start_step("check-bundle");
    check_bundle(&bundle_path, &channel_profile, &build_identity)?;
    step_reporter.finish_step();
    step_reporter.save_timings();
    println!(
        "[app-package] Astronomical {} app ready in {:.0}s\n[app-package] {}\n[app-package] Launch: open \"{}\"",
        package_request.channel.as_channel_name(),
        step_reporter.total_elapsed_seconds(),
        bundle_path.display(),
        bundle_path.display()
    );
    Ok(())
}

fn run_release_compile_steps(
    release_build: &ReleaseBuildContext<'_>,
    step_reporter: &mut StepReporter,
) -> Result<(), String> {
    step_reporter.start_step("native-dependencies");
    crate::bootstrap_native_dependencies::run(
        release_build.repository_root,
        &["--verify".to_owned()],
    )?;
    step_reporter.finish_step();
    step_reporter.start_step("native-runtime");
    crate::prewarm_native_build::run(
        release_build.repository_root,
        &["--profile".to_owned(), "core".to_owned()],
    )?;
    step_reporter.finish_step();
    step_reporter.start_step("rust-executables");
    release_build.build_rust_executables(step_reporter)?;
    step_reporter.finish_step();
    step_reporter.start_step("thin-talk-web");
    release_build.build_thin_talk_web_assets(step_reporter)?;
    step_reporter.finish_step();
    step_reporter.start_step("menu-app");
    release_build.build_menu_application(step_reporter)?;
    step_reporter.finish_step();
    Ok(())
}

fn assemble_and_sign_release_bundle(
    release_build: &ReleaseBuildContext<'_>,
    package_request: &PackageRequest,
    step_reporter: &mut StepReporter,
) -> Result<PathBuf, String> {
    step_reporter.start_step("assemble-and-sign");
    let bundle_inputs = release_bundle_inputs(release_build)?;
    let update_feed_url = github_pages_feed_url(&release_build.build_identity.repository_url)?;
    let info_plist_text = render_info_plist(
        release_build.channel_profile,
        release_build.build_identity,
        &update_feed_url,
    );
    let bundle_path = release_build
        .repository_root
        .join("target")
        .join(release_build.channel_profile.output_directory_name)
        .join(release_build.channel_profile.bundle_directory_name());
    remove_previous_bundle(&bundle_path)?;
    assemble_bundle(&bundle_path, &bundle_inputs, &info_plist_text)?;
    sign_bundle(&SigningRequest {
        bundle_path: &bundle_path,
        channel_profile: release_build.channel_profile,
        signing_identity: package_request.signing_identity.as_deref(),
        repository_root: release_build.repository_root,
    })?;
    step_reporter.finish_step();
    Ok(bundle_path)
}

/// The icon is rendered into the disposable target directory, so a release
/// never reuses an icon the edit loop cached.
fn release_bundle_inputs(release_build: &ReleaseBuildContext<'_>) -> Result<BundleInputs, String> {
    let Ok(home_directory) = std::env::var("HOME") else {
        return Err("HOME is required to locate the native build store".to_owned());
    };
    let metallib = packaged_metallib_source(
        release_build.repository_root,
        release_build.host_target_triple,
        Path::new(&home_directory),
    )?;
    let icon_directory = release_build.cargo_target_directory.join("app-icons");
    create_directory(&icon_directory)?;
    let icon = cached_channel_icon(
        release_build.repository_root,
        &icon_directory,
        release_build.channel_profile.icon_channel,
    )?;
    let bundle_inputs = BundleInputs::for_repository(
        release_build.repository_root,
        release_build.channel_profile,
        &release_build.rust_executable_directory(),
        metallib,
        icon,
    );
    let missing_paths = missing_inputs(&bundle_inputs);
    if !missing_paths.is_empty() {
        let mut missing_lines = String::new();
        for missing_path in missing_paths {
            missing_lines.push_str(&format!("\n  {}", missing_path.display()));
        }
        return Err(format!("bundle inputs are missing:{missing_lines}"));
    }
    Ok(bundle_inputs)
}

fn host_target_triple() -> Result<String, String> {
    let host_target_triple = run_quiet_tool(Command::new("rustc").args(["--print", "host-tuple"]))?;
    let is_valid_triple =
        !host_target_triple.is_empty() && host_target_triple.bytes().all(is_target_triple_byte);
    if is_valid_triple {
        Ok(host_target_triple)
    } else {
        Err("rustc did not return a valid host target triple".to_owned())
    }
}

fn is_target_triple_byte(triple_byte: u8) -> bool {
    triple_byte.is_ascii_alphanumeric() || matches!(triple_byte, b'_' | b'.' | b'-')
}

/// Only the channel's own generated bundle path is ever removed.
fn remove_previous_bundle(bundle_path: &Path) -> Result<(), String> {
    if !bundle_path.exists() {
        return Ok(());
    }
    match std::fs::remove_dir_all(bundle_path) {
        Ok(()) => Ok(()),
        Err(remove_error) => Err(format!(
            "could not remove the previous bundle {}: {remove_error}",
            bundle_path.display()
        )),
    }
}

fn create_directory(directory_path: &Path) -> Result<(), String> {
    match std::fs::create_dir_all(directory_path) {
        Ok(()) => Ok(()),
        Err(create_error) => Err(format!(
            "could not create {}: {create_error}",
            directory_path.display()
        )),
    }
}
