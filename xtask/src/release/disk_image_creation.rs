//! `cargo xtask release create-dmg --app-bundle PATH --output PATH`: builds
//! the drag-to-Applications disk image with Finder-owned layout metadata.
//! Typed port of the retired `create-dmg.sh`.
//!
//! The image is staged writable, laid out by Finder (only Finder writes a
//! `.DS_Store` that Finder honours), then compressed to a temporary name and
//! renamed into place, so an interrupted run never leaves a partial image
//! at the output path.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use super::scratch_space::{MountedVolume, ScratchDirectory};
use super::system_step::{require_commands, run_system_step};
use crate::step_log::start_step;

const VOLUME_NAME: &str = "Astronomical";
const BACKGROUND_RENDERER_PATH: &str = "scripts/release/render-dmg-background.swift";

/// Window bounds and icon positions match the rendered background's arrow,
/// so the two icons sit on either side of it.
pub(crate) const FINDER_LAYOUT_SCRIPT: &str = r#"on run arguments
    set mountedVolumePath to item 1 of arguments
    set backgroundImage to POSIX file (mountedVolumePath & "/.background/background.png") as alias
    tell application "Finder"
        set mountedVolume to POSIX file mountedVolumePath as alias
        open mountedVolume
        set volumeWindow to container window of mountedVolume
        set current view of volumeWindow to icon view
        set toolbar visible of volumeWindow to false
        set statusbar visible of volumeWindow to false
        set bounds of volumeWindow to {180, 180, 820, 600}
        set iconOptions to icon view options of volumeWindow
        set arrangement of iconOptions to not arranged
        set icon size of iconOptions to 128
        set text size of iconOptions to 14
        set background picture of iconOptions to backgroundImage
        set position of item "Astronomical.app" of mountedVolume to {175, 205}
        set position of item "Applications" of mountedVolume to {455, 205}
        update mountedVolume without registering applications
        delay 1
        close volumeWindow
    end tell
end run
"#;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DiskImageRequest {
    pub(crate) app_bundle: PathBuf,
    pub(crate) output_disk_image: PathBuf,
}

/// Everything that can be refused before any tool runs.
pub(crate) fn validate_disk_image_request(
    disk_image_request: &DiskImageRequest,
) -> Result<(), String> {
    if !disk_image_request.app_bundle.is_dir() {
        return Err("--app-bundle must name an application bundle".to_owned());
    }
    let bundled_metallib = disk_image_request
        .app_bundle
        .join("Contents/Resources/share/mlx/mlx.metallib");
    let has_metallib = match std::fs::symlink_metadata(&bundled_metallib) {
        Ok(metallib_metadata) => {
            metallib_metadata.file_type().is_file() && metallib_metadata.len() > 0
        }
        Err(_) => false,
    };
    if !has_metallib {
        return Err("required bundled MLX AOT metallib is unavailable".to_owned());
    }
    if disk_image_request.output_disk_image.exists() {
        return Err(format!(
            "output already exists: {}",
            disk_image_request.output_disk_image.display()
        ));
    }
    // Finder lays out whichever volume carries this name; a second one
    // mounted already would receive the layout instead.
    if Path::new("/Volumes").join(VOLUME_NAME).exists() {
        return Err(
            "an Astronomical volume is already mounted; eject it before building a DMG".to_owned(),
        );
    }
    Ok(())
}

pub(crate) fn create_disk_image(
    repository_root: &Path,
    disk_image_request: &DiskImageRequest,
) -> Result<(), String> {
    validate_disk_image_request(disk_image_request)?;
    require_commands(&["chflags", "ditto", "hdiutil", "osascript", "swift", "sync"])?;
    let scratch_directory = ScratchDirectory::create("astronomical-dmg")?;
    let volume_root = scratch_directory.directory_path.join("volume-root");
    let writable_disk_image = scratch_directory
        .directory_path
        .join("Astronomical-writable.dmg");
    stage_installation_volume(
        repository_root,
        &disk_image_request.app_bundle,
        &volume_root,
    )?;
    run_system_step(
        "create-writable-dmg",
        Command::new("hdiutil")
            .args(["create", "-volname", VOLUME_NAME, "-srcfolder"])
            .arg(&volume_root)
            .args(["-fs", "HFS+", "-format", "UDRW"])
            .arg(&writable_disk_image),
    )?;
    apply_finder_layout(
        &writable_disk_image,
        &scratch_directory.directory_path.join("mounted-volume"),
    )?;
    compress_into_place(&writable_disk_image, &disk_image_request.output_disk_image)?;
    println!(
        "Created drag-to-Applications DMG: {}",
        disk_image_request.output_disk_image.display()
    );
    Ok(())
}

fn stage_installation_volume(
    repository_root: &Path,
    app_bundle: &Path,
    volume_root: &Path,
) -> Result<(), String> {
    let stage_step = start_step("stage-installation-volume");
    create_directory(&volume_root.join(".background"))?;
    run_quiet(
        Command::new("ditto")
            .arg(app_bundle)
            .arg(volume_root.join("Astronomical.app")),
    )?;
    if let Err(link_error) =
        std::os::unix::fs::symlink("/Applications", volume_root.join("Applications"))
    {
        return Err(format!("could not link Applications: {link_error}"));
    }
    run_quiet(
        Command::new("swift")
            .arg(repository_root.join(BACKGROUND_RENDERER_PATH))
            .arg(volume_root.join(".background/background.png")),
    )?;
    // Spotlight must never index an installer volume.
    write_empty_file(&volume_root.join(".metadata_never_index"))?;
    run_quiet(
        Command::new("chflags")
            .arg("hidden")
            .arg(volume_root.join(".background"))
            .arg(volume_root.join(".metadata_never_index")),
    )?;
    stage_step.finish("success");
    Ok(())
}

fn apply_finder_layout(writable_disk_image: &Path, mount_point: &Path) -> Result<(), String> {
    let layout_step = start_step("apply-finder-layout");
    create_directory(mount_point)?;
    let mounted_volume = MountedVolume::attach(
        writable_disk_image,
        mount_point,
        &["-readwrite", "-noverify", "-noautoopen"],
    )?;
    // No fsevents log on a read-only installer volume.
    create_directory(&mount_point.join(".fseventsd"))?;
    write_empty_file(&mount_point.join(".fseventsd/no_log"))?;
    run_quiet(
        Command::new("chflags")
            .arg("hidden")
            .arg(mount_point.join(".fseventsd")),
    )?;
    run_finder_layout_script(mount_point)?;
    run_quiet(&mut Command::new("sync"))?;
    mounted_volume.detach()?;
    layout_step.finish("success");
    Ok(())
}

fn run_finder_layout_script(mount_point: &Path) -> Result<(), String> {
    let mut osascript_process = match Command::new("osascript")
        .arg("-")
        .arg(mount_point)
        .stdin(Stdio::piped())
        .spawn()
    {
        Ok(osascript_process) => osascript_process,
        Err(spawn_error) => return Err(format!("could not start osascript: {spawn_error}")),
    };
    if let Some(mut script_input) = osascript_process.stdin.take() {
        use std::io::Write;
        if let Err(write_error) = script_input.write_all(FINDER_LAYOUT_SCRIPT.as_bytes()) {
            return Err(format!(
                "could not send the Finder layout script: {write_error}"
            ));
        }
    }
    match osascript_process.wait() {
        Ok(exit_status) if exit_status.success() => Ok(()),
        _ => Err("Finder could not lay out the installer volume".to_owned()),
    }
}

fn compress_into_place(writable_disk_image: &Path, output_disk_image: &Path) -> Result<(), String> {
    let compress_step = start_step("compress-and-verify-dmg");
    let partial_disk_image = partial_output_path(output_disk_image);
    if let Some(output_parent) = output_disk_image.parent() {
        create_directory(output_parent)?;
    }
    let compression_result = compress_and_verify(writable_disk_image, &partial_disk_image);
    if compression_result.is_err() {
        let _ = std::fs::remove_file(&partial_disk_image);
        return compression_result;
    }
    if let Err(rename_error) = std::fs::rename(&partial_disk_image, output_disk_image) {
        let _ = std::fs::remove_file(&partial_disk_image);
        return Err(format!("could not publish the DMG: {rename_error}"));
    }
    compress_step.finish("success");
    Ok(())
}

fn compress_and_verify(
    writable_disk_image: &Path,
    partial_disk_image: &Path,
) -> Result<(), String> {
    run_quiet(
        Command::new("hdiutil")
            .arg("convert")
            .arg(writable_disk_image)
            .args(["-format", "UDZO", "-imagekey", "zlib-level=9", "-o"])
            .arg(partial_disk_image),
    )?;
    run_quiet(
        Command::new("hdiutil")
            .arg("verify")
            .arg(partial_disk_image),
    )?;
    match std::fs::metadata(partial_disk_image) {
        Ok(image_metadata) if image_metadata.len() > 0 => Ok(()),
        _ => Err("compressed DMG was not created".to_owned()),
    }
}

/// A hidden sibling of the output, so the rename into place is atomic on
/// one filesystem.
pub(crate) fn partial_output_path(output_disk_image: &Path) -> PathBuf {
    let output_name = match output_disk_image.file_name() {
        Some(output_name) => output_name.to_string_lossy().into_owned(),
        None => "Astronomical.dmg".to_owned(),
    };
    let partial_name = format!(".{output_name}.partial.{}.dmg", std::process::id());
    match output_disk_image.parent() {
        Some(output_parent) => output_parent.join(partial_name),
        None => PathBuf::from(partial_name),
    }
}

fn run_quiet(tool_command: &mut Command) -> Result<(), String> {
    match tool_command.output() {
        Ok(tool_output) if tool_output.status.success() => Ok(()),
        Ok(tool_output) => Err(format!(
            "{} failed: {}",
            tool_command.get_program().to_string_lossy(),
            String::from_utf8_lossy(&tool_output.stderr).trim()
        )),
        Err(spawn_error) => Err(format!(
            "{} could not start: {spawn_error}",
            tool_command.get_program().to_string_lossy()
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

fn write_empty_file(file_path: &Path) -> Result<(), String> {
    match std::fs::write(file_path, "") {
        Ok(()) => Ok(()),
        Err(write_error) => Err(format!(
            "could not write {}: {write_error}",
            file_path.display()
        )),
    }
}
