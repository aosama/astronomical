//! The compile steps of the Development app: the native MLX runtime, the
//! Rust executables, the Thin Talk web assets, and the two Swift executables
//! (the main app and its menu-bar agent).
//!
//! Each step keeps its output between runs. The release lane deliberately
//! starts cold from a disposable target directory; the edit loop must not,
//! or every rebuild pays for the whole program again.

use std::path::{Path, PathBuf};
use std::process::Command;

use astronomical_native_identity::digest_bytes;

use super::build_identity::BuildIdentity;
use super::channel_profile::{AGENT_EXECUTABLE_NAME, ChannelProfile};
use super::step_reporter::StepReporter;
use super::tool_command::{run_quiet_tool, run_tool};

/// The Cargo profile in the workspace manifest tuned for the edit loop.
pub(crate) const APP_DEV_CARGO_PROFILE: &str = "app-dev";

/// Pre-warms the machine-wide native build store, so the Rust build script
/// finds MLX already compiled and never invokes CMake. Called in-process:
/// the prewarm command is part of this same xtask binary, and it streams the
/// native build's own progress lines while it runs.
pub(crate) fn prewarm_native_runtime(repository_root: &Path) -> Result<(), String> {
    crate::prewarm_native_build::run(
        repository_root,
        &["--profile".to_owned(), "core".to_owned()],
    )
}

pub(crate) fn build_rust_executables(
    repository_root: &Path,
    channel_profile: &ChannelProfile,
    build_identity: &BuildIdentity,
    step_reporter: &StepReporter,
) -> Result<(), String> {
    let mut cargo_command = Command::new("cargo");
    cargo_command
        .current_dir(repository_root)
        .args(["build", "--profile", APP_DEV_CARGO_PROFILE, "--target-dir"])
        .arg(repository_root.join("target"))
        .args([
            "-p",
            "astronomical-inference-worker",
            "--bin",
            "astronomical-inference-worker",
        ])
        .args(["-p", "astronomical-daemon", "--bin", "astronomicald"])
        .args(["-p", "astronomical-cli", "--bin", "astronomical"])
        // sccache refuses incremental compilation, and incremental is the
        // point of this profile, so the wrapper is dropped for this build.
        .env_remove("RUSTC_WRAPPER");
    if let Some(daemon_cargo_features) = channel_profile.daemon_cargo_features {
        cargo_command.args(["--features", daemon_cargo_features]);
    }
    for (identity_variable, identity_value) in build_identity.cargo_environment() {
        cargo_command.env(identity_variable, identity_value);
    }
    run_tool(&mut cargo_command, step_reporter)
}

pub(crate) fn rust_executable_directory(repository_root: &Path) -> PathBuf {
    repository_root.join("target").join(APP_DEV_CARGO_PROFILE)
}

/// Builds the Thin Talk offline shell. `npm ci` deletes and reinstalls
/// `node_modules`, which the shell builder paid on every run; here it runs
/// only when the lockfile changed since the last install.
pub(crate) fn build_thin_talk_web_assets(
    repository_root: &Path,
    lockfile_stamp_path: &Path,
    step_reporter: &StepReporter,
) -> Result<(), String> {
    let web_directory = repository_root.join("apps/thin-talk/web");
    let lockfile_digest = file_sha256_hex(&web_directory.join("package-lock.json"))?;
    let installed_digest = std::fs::read_to_string(lockfile_stamp_path).unwrap_or_default();
    if !npm_install_is_current(
        &lockfile_digest,
        &installed_digest,
        web_directory.join("node_modules").is_dir(),
    ) {
        run_tool(
            Command::new("npm").arg("ci").current_dir(&web_directory),
            step_reporter,
        )?;
        if let Err(write_error) = std::fs::write(lockfile_stamp_path, &lockfile_digest) {
            return Err(format!(
                "could not record the npm lockfile stamp: {write_error}"
            ));
        }
    } else {
        println!("[app-dev] npm dependencies match package-lock.json; skipping npm ci");
    }
    run_tool(
        Command::new("npm")
            .args(["run", "build"])
            .current_dir(&web_directory),
        step_reporter,
    )
}

pub(crate) fn npm_install_is_current(
    lockfile_digest: &str,
    installed_digest: &str,
    node_modules_exists: bool,
) -> bool {
    node_modules_exists && !lockfile_digest.is_empty() && lockfile_digest == installed_digest.trim()
}

/// Builds the main app, then the agent. They share one package and most of
/// their modules, so the second build only compiles what the first did not.
///
/// Warnings fail the Swift build, mirroring the Rust rule that a warning is
/// a defect, so none survives into a bundle.
pub(crate) fn build_swift_applications(
    repository_root: &Path,
    channel_profile: &ChannelProfile,
    step_reporter: &StepReporter,
) -> Result<(), String> {
    let macos_sdk_path =
        run_quiet_tool(Command::new("xcrun").args(["--sdk", "macosx", "--show-sdk-path"]))?;
    for swift_product in [
        channel_profile.application_swift_product,
        AGENT_EXECUTABLE_NAME,
    ] {
        let mut swift_command =
            swift_product_build_command(repository_root, swift_product, macos_sdk_path.trim());
        run_tool(&mut swift_command, step_reporter)?;
    }
    Ok(())
}

/// The `swift build` for one product of the app's Swift package.
///
/// The SDK is handed to the link step's clang as well as to swiftc. Under the
/// swiftbuild backend the link runs without SDKROOT, so clang cannot read the
/// SDK's version and stamps the binary with the deployment target (14.0)
/// instead. macOS then treats the app as built for macOS 14 and draws its
/// windows in the old style: smaller window controls and plain toolbar icons
/// rather than the current design every other app on the Mac uses.
///
/// The SDK path travels as one argument. The tests prove that for the `swift`
/// command line; an SDK path containing spaces has not been tried through the
/// swiftbuild backend, which receives it from SwiftPM.
pub(crate) fn swift_product_build_command(
    repository_root: &Path,
    swift_product: &str,
    macos_sdk_path: &str,
) -> Command {
    let mut swift_command = Command::new("swift");
    swift_command
        .args(["build", "--configuration", "release", "--package-path"])
        .arg(repository_root.join("apps/astronomical-menu"))
        .args(["--product", swift_product])
        .args(["-Xswiftc", "-warnings-as-errors"])
        .args(["-Xswiftc", "-Xclang-linker", "-Xswiftc", "-isysroot"])
        .args(["-Xswiftc", "-Xclang-linker", "-Xswiftc", macos_sdk_path]);
    swift_command
}

pub(crate) fn file_sha256_hex(file_path: &Path) -> Result<String, String> {
    match std::fs::read(file_path) {
        Ok(file_bytes) => Ok(digest_bytes(&file_bytes)),
        Err(read_error) => Err(format!(
            "could not read {}: {read_error}",
            file_path.display()
        )),
    }
}
