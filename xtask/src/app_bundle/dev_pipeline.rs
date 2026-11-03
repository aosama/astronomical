//! `cargo xtask app dev`: rebuild the Development app and open it.
//!
//! The order matters for the person using the running app: everything is
//! built, assembled, signed, and checked in a staging directory first, and
//! the running instance is stopped only for the final swap. A failed build
//! therefore leaves the working app running instead of a stopped one, and a
//! slow-exiting daemon delays only the launch, never the swap.

use std::path::{Path, PathBuf};

use super::build_identity::{BuildIdentity, github_pages_feed_url};
use super::build_steps::{
    build_menu_application, build_rust_executables, build_thin_talk_web_assets,
    prewarm_native_runtime, rust_executable_directory,
};
use super::bundle_assembly::{BundleInputs, assemble_bundle, missing_inputs};
use super::bundle_check::check_bundle;
use super::channel_profile::{Channel, ChannelProfile};
use super::code_signing::{SigningRequest, sign_bundle};
use super::icon_cache::cached_channel_icon;
use super::info_plist::render_info_plist;
use super::native_metallib::packaged_metallib_source;
use super::running_instance::{open_bundle, quit_menu_app, wait_for_daemon_exit};
use super::step_reporter::StepReporter;
use super::tool_command::run_quiet_tool;

pub(crate) const DEV_PIPELINE_STEP_NAMES: [&str; 8] = [
    "build-identity",
    "native-runtime",
    "rust-executables",
    "thin-talk-web",
    "menu-app",
    "assemble-and-sign",
    "check-bundle",
    "swap-and-open",
];

/// Paths the edit loop keeps between runs. They live beside, not inside,
/// Cargo's `target/app-dev` profile directory, which Cargo owns.
pub(crate) struct DevPipelinePaths {
    pub(crate) bundle_path: PathBuf,
    pub(crate) staging_bundle_path: PathBuf,
    pub(crate) previous_bundle_path: PathBuf,
    pub(crate) cache_directory: PathBuf,
}

impl DevPipelinePaths {
    pub(crate) fn for_profile(repository_root: &Path, channel_profile: &ChannelProfile) -> Self {
        let output_directory = repository_root
            .join("target")
            .join(channel_profile.output_directory_name);
        let bundle_directory_name = channel_profile.bundle_directory_name();
        Self {
            bundle_path: output_directory.join(&bundle_directory_name),
            staging_bundle_path: output_directory
                .join(".staging")
                .join(&bundle_directory_name),
            previous_bundle_path: output_directory
                .join(".previous")
                .join(&bundle_directory_name),
            cache_directory: repository_root.join("target/app-dev-cache"),
        }
    }
}

pub(crate) fn run_dev_pipeline(repository_root: &Path) -> Result<(), String> {
    let channel_profile = ChannelProfile::for_channel(Channel::Development);
    let pipeline_paths = DevPipelinePaths::for_profile(repository_root, &channel_profile);
    create_directory(&pipeline_paths.cache_directory)?;
    let mut step_reporter = StepReporter::new(
        "app-dev",
        DEV_PIPELINE_STEP_NAMES.to_vec(),
        pipeline_paths.cache_directory.join("step-timings.json"),
    );
    step_reporter.start_step("build-identity");
    let build_identity = BuildIdentity::gather(repository_root)?;
    if channel_profile.requires_clean_worktree && build_identity.is_dirty {
        return Err("this channel requires a clean worktree".to_owned());
    }
    step_reporter.finish_step();
    run_compile_steps(
        repository_root,
        &channel_profile,
        &build_identity,
        &pipeline_paths,
        &mut step_reporter,
    )?;
    run_bundle_steps(
        repository_root,
        &channel_profile,
        &build_identity,
        &pipeline_paths,
        &mut step_reporter,
    )?;
    step_reporter.save_timings();
    println!(
        "[app-dev] {} {} ({}{}) ready and opened in {:.1}s\n[app-dev] {}",
        channel_profile.bundle_name,
        build_identity.application_version,
        build_identity.build_commit,
        if build_identity.is_dirty {
            ", dirty"
        } else {
            ""
        },
        step_reporter.total_elapsed_seconds(),
        pipeline_paths.bundle_path.display()
    );
    Ok(())
}

fn run_compile_steps(
    repository_root: &Path,
    channel_profile: &ChannelProfile,
    build_identity: &BuildIdentity,
    pipeline_paths: &DevPipelinePaths,
    step_reporter: &mut StepReporter,
) -> Result<(), String> {
    step_reporter.start_step("native-runtime");
    prewarm_native_runtime(repository_root)?;
    step_reporter.finish_step();
    step_reporter.start_step("rust-executables");
    build_rust_executables(
        repository_root,
        channel_profile,
        build_identity,
        step_reporter,
    )?;
    step_reporter.finish_step();
    step_reporter.start_step("thin-talk-web");
    let lockfile_stamp_path = pipeline_paths
        .cache_directory
        .join("thin-talk-package-lock.sha256");
    build_thin_talk_web_assets(repository_root, &lockfile_stamp_path, step_reporter)?;
    step_reporter.finish_step();
    step_reporter.start_step("menu-app");
    build_menu_application(repository_root, channel_profile, step_reporter)?;
    step_reporter.finish_step();
    Ok(())
}

fn run_bundle_steps(
    repository_root: &Path,
    channel_profile: &ChannelProfile,
    build_identity: &BuildIdentity,
    pipeline_paths: &DevPipelinePaths,
    step_reporter: &mut StepReporter,
) -> Result<(), String> {
    step_reporter.start_step("assemble-and-sign");
    let bundle_inputs = resolve_bundle_inputs(repository_root, channel_profile, pipeline_paths)?;
    let update_feed_url = github_pages_feed_url(&build_identity.repository_url)?;
    let info_plist_text = render_info_plist(channel_profile, build_identity, &update_feed_url);
    remove_directory_if_present(&pipeline_paths.staging_bundle_path)?;
    assemble_bundle(
        &pipeline_paths.staging_bundle_path,
        &bundle_inputs,
        &info_plist_text,
    )?;
    sign_bundle(&SigningRequest {
        bundle_path: &pipeline_paths.staging_bundle_path,
        channel_profile,
        signing_identity: None,
        repository_root,
    })?;
    step_reporter.finish_step();
    step_reporter.start_step("check-bundle");
    check_bundle(
        &pipeline_paths.staging_bundle_path,
        channel_profile,
        build_identity,
    )?;
    step_reporter.finish_step();
    step_reporter.start_step("swap-and-open");
    quit_menu_app(channel_profile)?;
    swap_in_staged_bundle(pipeline_paths)?;
    wait_for_daemon_exit(channel_profile)?;
    open_bundle(&pipeline_paths.bundle_path)?;
    step_reporter.finish_step();
    Ok(())
}

fn resolve_bundle_inputs(
    repository_root: &Path,
    channel_profile: &ChannelProfile,
    pipeline_paths: &DevPipelinePaths,
) -> Result<BundleInputs, String> {
    let host_target_triple =
        run_quiet_tool(std::process::Command::new("rustc").args(["--print", "host-tuple"]))?;
    let Ok(home_directory) = std::env::var("HOME") else {
        return Err("HOME is required to locate the native build store".to_owned());
    };
    let metallib = packaged_metallib_source(
        repository_root,
        &host_target_triple,
        Path::new(&home_directory),
    )?;
    let icon_cache_directory = pipeline_paths.cache_directory.join("icons");
    create_directory(&icon_cache_directory)?;
    let icon = cached_channel_icon(
        repository_root,
        &icon_cache_directory,
        channel_profile.icon_channel,
    )?;
    let bundle_inputs = BundleInputs::for_repository(
        repository_root,
        channel_profile,
        &rust_executable_directory(repository_root),
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

/// Two renames instead of a delete-then-copy, so the window in which no
/// Development app exists on disk is as short as the filesystem allows.
fn swap_in_staged_bundle(pipeline_paths: &DevPipelinePaths) -> Result<(), String> {
    remove_directory_if_present(&pipeline_paths.previous_bundle_path)?;
    if pipeline_paths.bundle_path.exists() {
        if let Some(previous_parent) = pipeline_paths.previous_bundle_path.parent() {
            create_directory(previous_parent)?;
        }
        rename_path(
            &pipeline_paths.bundle_path,
            &pipeline_paths.previous_bundle_path,
        )?;
    }
    if let Err(swap_error) = rename_path(
        &pipeline_paths.staging_bundle_path,
        &pipeline_paths.bundle_path,
    ) {
        // Put the previous app back so a failed swap never leaves the
        // Development bundle path empty.
        if pipeline_paths.previous_bundle_path.exists() {
            rename_path(
                &pipeline_paths.previous_bundle_path,
                &pipeline_paths.bundle_path,
            )?;
        }
        return Err(swap_error);
    }
    remove_directory_if_present(&pipeline_paths.previous_bundle_path)
}

fn rename_path(source_path: &Path, destination_path: &Path) -> Result<(), String> {
    match std::fs::rename(source_path, destination_path) {
        Ok(()) => Ok(()),
        Err(rename_error) => Err(format!(
            "could not move {} to {}: {rename_error}",
            source_path.display(),
            destination_path.display()
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

/// Removes only generated bundle directories under `target/`; the paths come
/// from `DevPipelinePaths`, never from user input.
fn remove_directory_if_present(directory_path: &Path) -> Result<(), String> {
    if !directory_path.exists() {
        return Ok(());
    }
    match std::fs::remove_dir_all(directory_path) {
        Ok(()) => Ok(()),
        Err(remove_error) => Err(format!(
            "could not remove {}: {remove_error}",
            directory_path.display()
        )),
    }
}
