//! The compile steps of a release bundle (Stable or App Store).
//!
//! Unlike the Development loop, a release build starts cold on purpose: it
//! runs inside the `app-release` disposable target lane with the shipped
//! `release` profile (thin LTO), always reinstalls npm dependencies from the
//! lockfile, and never reuses a cached icon, so the bundle depends only on
//! the commit it was built from.

use std::path::{Path, PathBuf};
use std::process::Command;

use super::build_identity::BuildIdentity;
use super::build_steps::build_menu_application;
use super::channel_profile::ChannelProfile;
use super::step_reporter::StepReporter;
use super::tool_command::run_tool;

/// `cargo build` arguments for the three shipped executables. The App Store
/// channel adds the daemon feature that derives state from Application
/// Support; no other channel may carry it.
pub(crate) fn release_cargo_arguments(
    channel_profile: &ChannelProfile,
    host_target_triple: &str,
) -> Vec<String> {
    let mut cargo_arguments: Vec<String> = Vec::new();
    for argument_word in ["build", "--release", "--target", host_target_triple] {
        cargo_arguments.push(argument_word.to_owned());
    }
    for (package_name, binary_name) in [
        (
            "astronomical-inference-worker",
            "astronomical-inference-worker",
        ),
        ("astronomical-daemon", "astronomicald"),
        ("astronomical-cli", "astronomical"),
    ] {
        for argument_word in ["-p", package_name, "--bin", binary_name] {
            cargo_arguments.push(argument_word.to_owned());
        }
    }
    if let Some(daemon_cargo_features) = channel_profile.daemon_cargo_features {
        cargo_arguments.push("--features".to_owned());
        cargo_arguments.push(daemon_cargo_features.to_owned());
    }
    cargo_arguments
}

pub(crate) struct ReleaseBuildContext<'context> {
    pub(crate) repository_root: &'context Path,
    pub(crate) channel_profile: &'context ChannelProfile,
    pub(crate) build_identity: &'context BuildIdentity,
    pub(crate) host_target_triple: &'context str,
    pub(crate) cargo_target_directory: &'context Path,
    pub(crate) logical_cpu_count: String,
}

impl ReleaseBuildContext<'_> {
    pub(crate) fn rust_executable_directory(&self) -> PathBuf {
        self.cargo_target_directory
            .join(self.host_target_triple)
            .join("release")
    }

    pub(crate) fn build_rust_executables(
        &self,
        step_reporter: &StepReporter,
    ) -> Result<(), String> {
        let mut cargo_command = Command::new("cargo");
        cargo_command
            .current_dir(self.repository_root)
            .args(release_cargo_arguments(
                self.channel_profile,
                self.host_target_triple,
            ))
            .env("CARGO_TARGET_DIR", self.cargo_target_directory)
            .env("CARGO_BUILD_JOBS", &self.logical_cpu_count);
        for (identity_variable, identity_value) in self.build_identity.cargo_environment() {
            cargo_command.env(identity_variable, identity_value);
        }
        run_tool(&mut cargo_command, step_reporter)
    }

    /// `npm ci` every time: a release must not inherit whatever a developer
    /// last installed into node_modules.
    pub(crate) fn build_thin_talk_web_assets(
        &self,
        step_reporter: &StepReporter,
    ) -> Result<(), String> {
        let web_directory = self.repository_root.join("apps/thin-talk/web");
        run_tool(
            Command::new("npm").arg("ci").current_dir(&web_directory),
            step_reporter,
        )?;
        run_tool(
            Command::new("npm")
                .args(["run", "build"])
                .current_dir(&web_directory),
            step_reporter,
        )
    }

    pub(crate) fn build_menu_application(
        &self,
        step_reporter: &StepReporter,
    ) -> Result<(), String> {
        build_menu_application(self.repository_root, self.channel_profile, step_reporter)
    }
}
