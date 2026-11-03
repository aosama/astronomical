//! Lays out an `.app` bundle from already-built inputs.
//!
//! Assembly knows nothing about channels: the profile decides which inputs
//! exist (Sparkle or not), and this only places them. That split is what
//! lets a future `package` command reuse the exact layout the Development
//! app is tested with.

use std::path::{Path, PathBuf};
use std::process::Command;

use super::build_steps::RUST_EXECUTABLE_NAMES;
use super::channel_profile::ChannelProfile;
use super::tool_command::run_quiet_tool;

/// Everything copied into the bundle, resolved before anything is written so
/// a missing input fails before the old bundle is touched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BundleInputs {
    pub(crate) menu_executable: PathBuf,
    pub(crate) thin_talk_resource_bundle: PathBuf,
    pub(crate) rust_executables: Vec<(&'static str, PathBuf)>,
    pub(crate) metallib: PathBuf,
    pub(crate) icon: PathBuf,
    /// Destination file name in `Contents/Resources`, then source path.
    pub(crate) notice_files: Vec<(&'static str, PathBuf)>,
    pub(crate) sparkle_framework: Option<PathBuf>,
}

impl BundleInputs {
    pub(crate) fn for_repository(
        repository_root: &Path,
        channel_profile: &ChannelProfile,
        rust_executable_directory: &Path,
        metallib: PathBuf,
        icon: PathBuf,
    ) -> Self {
        let swift_build_directory = repository_root.join("apps/astronomical-menu/.build");
        let mut rust_executables = Vec::new();
        for executable_name in RUST_EXECUTABLE_NAMES {
            rust_executables.push((
                executable_name,
                rust_executable_directory.join(executable_name),
            ));
        }
        let mut notice_files = vec![
            ("LICENSE", repository_root.join("LICENSE")),
            (
                "THIRD_PARTY_NOTICES",
                repository_root.join("third-party/THIRD_PARTY_NOTICES"),
            ),
            (
                "RUST_DEPENDENCY_NOTICES",
                repository_root.join("third-party/RUST_DEPENDENCY_NOTICES"),
            ),
        ];
        let mut sparkle_framework = None;
        if channel_profile.ships_sparkle {
            notice_files.push((
                "SPARKLE_LICENSE",
                swift_build_directory.join("checkouts/Sparkle/LICENSE"),
            ));
            sparkle_framework = Some(swift_build_directory.join("release/Sparkle.framework"));
        }
        Self {
            menu_executable: swift_build_directory
                .join("release")
                .join(channel_profile.menu_swift_product),
            thin_talk_resource_bundle: swift_build_directory
                .join("out/Products/Release/ThinTalk_ThinTalkCanvas.bundle"),
            rust_executables,
            metallib,
            icon,
            notice_files,
            sparkle_framework,
        }
    }

    /// Every input path, so one check can name all missing inputs at once.
    pub(crate) fn every_source_path(&self) -> Vec<&Path> {
        let mut source_paths: Vec<&Path> = vec![
            &self.menu_executable,
            &self.thin_talk_resource_bundle,
            &self.metallib,
            &self.icon,
        ];
        for (_, executable_path) in &self.rust_executables {
            source_paths.push(executable_path);
        }
        for (_, notice_path) in &self.notice_files {
            source_paths.push(notice_path);
        }
        if let Some(sparkle_framework) = &self.sparkle_framework {
            source_paths.push(sparkle_framework);
        }
        source_paths
    }
}

pub(crate) fn missing_inputs(bundle_inputs: &BundleInputs) -> Vec<PathBuf> {
    let mut missing_paths = Vec::new();
    for source_path in bundle_inputs.every_source_path() {
        if !source_path.exists() {
            missing_paths.push(source_path.to_path_buf());
        }
    }
    missing_paths
}

pub(crate) fn assemble_bundle(
    bundle_path: &Path,
    bundle_inputs: &BundleInputs,
    info_plist_text: &str,
) -> Result<(), String> {
    let contents_path = bundle_path.join("Contents");
    for bundle_subdirectory in ["MacOS", "Frameworks", "Resources/share/mlx"] {
        create_directory(&contents_path.join(bundle_subdirectory))?;
    }
    write_file(&contents_path.join("Info.plist"), info_plist_text)?;
    copy_file(
        &bundle_inputs.menu_executable,
        &contents_path.join("MacOS/astronomical-menu"),
    )?;
    for (executable_name, executable_path) in &bundle_inputs.rust_executables {
        copy_file(
            executable_path,
            &contents_path.join("MacOS").join(executable_name),
        )?;
    }
    for (notice_name, notice_path) in &bundle_inputs.notice_files {
        copy_file(
            notice_path,
            &contents_path.join("Resources").join(notice_name),
        )?;
    }
    copy_file(
        &bundle_inputs.metallib,
        &contents_path.join("Resources/share/mlx/mlx.metallib"),
    )?;
    copy_file(
        &bundle_inputs.icon,
        &contents_path.join("Resources/Astronomical.icns"),
    )?;
    copy_tree(
        &bundle_inputs.thin_talk_resource_bundle,
        &contents_path.join("Resources/ThinTalk_ThinTalkCanvas.bundle"),
    )?;
    if let Some(sparkle_framework) = &bundle_inputs.sparkle_framework {
        copy_tree(
            sparkle_framework,
            &contents_path.join("Frameworks/Sparkle.framework"),
        )?;
        // SwiftPM links Sparkle through @loader_path; a bundle keeps
        // frameworks in Contents/Frameworks, so the standard rpath is added
        // before signing seals the executable.
        run_quiet_tool(
            Command::new("install_name_tool")
                .args(["-add_rpath", "@executable_path/../Frameworks"])
                .arg(contents_path.join("MacOS/astronomical-menu")),
        )?;
    }
    Ok(())
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

fn write_file(file_path: &Path, file_text: &str) -> Result<(), String> {
    match std::fs::write(file_path, file_text) {
        Ok(()) => Ok(()),
        Err(write_error) => Err(format!(
            "could not write {}: {write_error}",
            file_path.display()
        )),
    }
}

/// `fs::copy` keeps the executable permission bits, so no `chmod` follows.
fn copy_file(source_path: &Path, destination_path: &Path) -> Result<(), String> {
    match std::fs::copy(source_path, destination_path) {
        Ok(_) => Ok(()),
        Err(copy_error) => Err(format!(
            "could not copy {} to {}: {copy_error}",
            source_path.display(),
            destination_path.display()
        )),
    }
}

/// `ditto` keeps the symlinks a framework's `Versions/Current` layout relies
/// on; a plain recursive copy would duplicate them and break the signature.
fn copy_tree(source_path: &Path, destination_path: &Path) -> Result<(), String> {
    run_quiet_tool(Command::new("ditto").arg(source_path).arg(destination_path)).map(drop)
}
