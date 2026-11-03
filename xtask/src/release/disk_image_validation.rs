//! `cargo xtask release validate-dmg --dmg PATH`: mounts the final disk
//! image and proves its installation journey and Apple trust chain, the way
//! a person installs it: checksum, stapled ticket, Gatekeeper on the image,
//! the volume's layout, the app's signature on the volume, and again after
//! copying it to an Applications folder. Typed port of the retired
//! `validate-dmg.sh`.

use std::path::Path;
use std::process::Command;

use super::scratch_space::{MountedVolume, ScratchDirectory};
use super::system_step::{require_commands, run_system_step};

pub(crate) fn validate_disk_image(disk_image_path: &Path) -> Result<(), String> {
    let has_contents = match std::fs::metadata(disk_image_path) {
        Ok(image_metadata) => image_metadata.is_file() && image_metadata.len() > 0,
        Err(_) => false,
    };
    if !has_contents {
        return Err("--dmg must name a nonempty disk image".to_owned());
    }
    require_commands(&["codesign", "ditto", "hdiutil", "spctl", "xcrun"])?;
    run_system_step(
        "verify-dmg-checksum",
        Command::new("hdiutil").arg("verify").arg(disk_image_path),
    )?;
    run_system_step(
        "validate-stapled-ticket",
        Command::new("xcrun")
            .args(["stapler", "validate"])
            .arg(disk_image_path),
    )?;
    run_system_step(
        "assess-notarized-dmg",
        Command::new("spctl")
            .args([
                "--assess",
                "--type",
                "open",
                "--context",
                "context:primary-signature",
                "--verbose=4",
            ])
            .arg(disk_image_path),
    )?;
    let scratch_directory = ScratchDirectory::create("astronomical-dmg-validation")?;
    let mount_point = scratch_directory.directory_path.join("mounted-volume");
    if let Err(create_error) = std::fs::create_dir_all(&mount_point) {
        return Err(format!("could not create the mount point: {create_error}"));
    }
    let mounted_volume = MountedVolume::attach(
        disk_image_path,
        &mount_point,
        &["-readonly", "-nobrowse", "-noautoopen"],
    )?;
    println!(
        "{} step=mount-dmg status=success",
        crate::disposable_target::local_timestamp()
    );
    let mounted_layout_problems = mounted_layout_problems(&mounted_volume.mount_point);
    if let Some(first_problem) = mounted_layout_problems.first() {
        return Err(first_problem.clone());
    }
    assess_app(
        &mounted_volume.mount_point.join("Astronomical.app"),
        "mounted",
    )?;
    let installed_app = scratch_directory
        .directory_path
        .join("Applications/Astronomical.app");
    run_system_step(
        "copy-app-to-applications",
        Command::new("ditto")
            .arg(mounted_volume.mount_point.join("Astronomical.app"))
            .arg(&installed_app),
    )?;
    assess_app(&installed_app, "installed")?;
    mounted_volume.detach()?;
    println!(
        "Validated notarized drag-to-Applications DMG: {}",
        disk_image_path.display()
    );
    Ok(())
}

/// What a mounted installer volume must contain, in the order the shell
/// checked it; the first problem is reported.
pub(crate) fn mounted_layout_problems(mount_point: &Path) -> Vec<String> {
    let mut layout_problems = Vec::new();
    if !mount_point.join("Astronomical.app").is_dir() {
        layout_problems.push("DMG does not contain Astronomical.app".to_owned());
    }
    if !is_non_empty_regular_file(
        &mount_point.join("Astronomical.app/Contents/Resources/share/mlx/mlx.metallib"),
    ) {
        layout_problems.push("DMG Astronomical.app is missing mlx.metallib".to_owned());
    }
    match std::fs::read_link(mount_point.join("Applications")) {
        Ok(link_target) if link_target == Path::new("/Applications") => {}
        Ok(_) => {
            layout_problems.push("DMG Applications link does not target /Applications".to_owned())
        }
        Err(_) => layout_problems.push("DMG does not contain an Applications link".to_owned()),
    }
    if !is_non_empty_regular_file(&mount_point.join(".DS_Store")) {
        layout_problems.push("DMG does not contain Finder layout metadata".to_owned());
    }
    if !is_non_empty_regular_file(&mount_point.join(".background/background.png")) {
        layout_problems.push("DMG does not contain installation guidance".to_owned());
    }
    layout_problems
}

fn assess_app(app_path: &Path, location_name: &'static str) -> Result<(), String> {
    let (signature_step, assessment_step) = if location_name == "mounted" {
        ("verify-mounted-app-signature", "assess-mounted-app")
    } else {
        ("verify-installed-app-signature", "assess-installed-app")
    };
    run_system_step(
        signature_step,
        Command::new("codesign")
            .args(["--verify", "--deep", "--strict", "--verbose=2"])
            .arg(app_path),
    )?;
    run_system_step(
        assessment_step,
        Command::new("spctl")
            .args(["--assess", "--type", "execute", "--verbose=4"])
            .arg(app_path),
    )
}

fn is_non_empty_regular_file(file_path: &Path) -> bool {
    match std::fs::symlink_metadata(file_path) {
        Ok(file_metadata) => file_metadata.file_type().is_file() && file_metadata.len() > 0,
        Err(_) => false,
    }
}
