//! `cargo xtask release install-stable [--app-bundle PATH] [--dry-run]`:
//! promotes one validated Stable bundle to `~/Applications/Astronomical.app`,
//! and `release build-and-install-stable [--dry-run]` packages Stable first.
//! Typed port of the retired `install-stable-app.sh` and
//! `build-and-install-stable-app.sh`.
//!
//! The bundle is re-checked with the same layout and plist rules that built
//! it, then staged beside the destination, verified there, and swapped in
//! by rename with the previous app kept as a backup until the swap holds.
//! The previous app then goes to the Trash rather than being deleted, so a
//! promotion can always be undone by hand. The running app is never
//! restarted.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::app_bundle::{
    Channel, ChannelProfile, layout_problems, plist_identity_problems, version_text_reports,
};
use crate::step_log::start_step;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct InstallRequest {
    pub(crate) source_bundle: PathBuf,
    pub(crate) is_dry_run: bool,
}

/// The bundle `app package --channel stable` writes; generated bundles live
/// below `.noindex` so Spotlight only ever shows the installed copy.
pub(crate) fn default_stable_bundle(repository_root: &Path) -> PathBuf {
    repository_root.join("target/astronomical-macos-stable.noindex/Astronomical.app")
}

pub(crate) fn install_stable(install_request: &InstallRequest) -> Result<(), String> {
    let home_directory = canonical_home_directory(std::env::var("HOME").ok().as_deref())?;
    let (application_version, build_commit) =
        validate_stable_bundle(&install_request.source_bundle)?;
    let destination_directory = home_directory.join("Applications");
    let destination_bundle = destination_directory.join("Astronomical.app");
    if install_request.is_dry_run {
        println!(
            "Would install {application_version} Stable ({build_commit}) at {}",
            destination_bundle.display()
        );
        return Ok(());
    }
    let promotion_step = start_step("stage-and-promote-stable");
    promote_bundle(
        &install_request.source_bundle,
        &destination_directory,
        &PROMOTION_TOOLS,
    )?;
    promotion_step.finish("success");
    println!(
        "Installed {application_version} Stable ({build_commit}). The running app was not restarted."
    );
    Ok(())
}

/// HOME must be an absolute, existing, non-root directory without `.` or
/// `..` components; it is resolved through symlinks so the promotion never
/// lands somewhere a crafted HOME points it.
pub(crate) fn canonical_home_directory(home_text: Option<&str>) -> Result<PathBuf, String> {
    let Some(home_text) = home_text.filter(is_absolute_path_text) else {
        return Err("HOME must be an absolute non-root directory".to_owned());
    };
    for home_component in home_text.split('/') {
        if home_component == "." || home_component == ".." {
            return Err("HOME must not contain parent or current-directory components".to_owned());
        }
    }
    let canonical_home = match std::fs::canonicalize(home_text) {
        Ok(canonical_home) if canonical_home.is_dir() => canonical_home,
        _ => return Err("HOME must identify an existing directory".to_owned()),
    };
    if canonical_home == Path::new("/") {
        return Err("HOME must not resolve to the filesystem root".to_owned());
    }
    Ok(canonical_home)
}

fn is_absolute_path_text(home_text: &&str) -> bool {
    home_text.starts_with('/')
}

/// Returns the bundle's version and commit once it passes every check.
fn validate_stable_bundle(source_bundle: &Path) -> Result<(String, String), String> {
    if !source_bundle.is_dir() {
        return Err(format!(
            "Stable app bundle not found: {}",
            source_bundle.display()
        ));
    }
    let validation_step = start_step("validate-stable-bundle");
    let plist_document = read_plist_document(source_bundle)?;
    let stable_profile = ChannelProfile::for_channel(Channel::Stable);
    let mut bundle_problems = stable_promotion_problems(&plist_document);
    bundle_problems.extend(layout_problems(source_bundle, &stable_profile));
    bundle_problems.extend(plist_identity_problems(&plist_document, &stable_profile));
    if let Some(first_problem) = bundle_problems.first() {
        return Err(first_problem.clone());
    }
    let application_version = plist_text(&plist_document, "CFBundleShortVersionString")?;
    let build_commit = plist_text(&plist_document, "AstronomicalBuildCommit")?;
    verify_seal(source_bundle)?;
    let daemon_version = tool_version(&source_bundle.join("Contents/MacOS/astronomicald"))?;
    if !version_text_reports(&daemon_version, &application_version, &build_commit) {
        return Err("Stable app and bundled daemon identities do not match".to_owned());
    }
    if !tool_version(&source_bundle.join("Contents/MacOS/astronomical"))?
        .contains(&application_version)
    {
        return Err("Stable app and bundled CLI identities do not match".to_owned());
    }
    validation_step.finish("success");
    Ok((application_version, build_commit))
}

/// The promotion-specific refusals, worded for the person promoting: only a
/// clean Stable build may replace the installed app.
pub(crate) fn stable_promotion_problems(plist_document: &serde_json::Value) -> Vec<String> {
    let mut promotion_problems = Vec::new();
    if plist_document["AstronomicalChannel"].as_str() != Some("stable") {
        promotion_problems.push("only a Stable bundle may be promoted".to_owned());
    }
    if plist_document["AstronomicalBuildDirty"].as_bool() != Some(false) {
        promotion_problems.push("dirty builds cannot be promoted to Stable".to_owned());
    }
    promotion_problems
}

/// The two side effects of a promotion that touch the outside world; tests
/// substitute plain copies and keep the previous app in place.
pub(crate) struct PromotionTools {
    /// Copies the source bundle to the staging path and verifies the copy.
    pub(crate) stage_copy: fn(&Path, &Path) -> Result<(), String>,
    /// Disposes of the previous app once the new one is in place.
    pub(crate) dispose_backup: fn(&Path) -> Result<(), String>,
}

const PROMOTION_TOOLS: PromotionTools = PromotionTools {
    stage_copy: stage_and_verify_copy,
    dispose_backup: move_backup_to_trash,
};

/// Stage beside the destination (same filesystem, so the swap is a rename),
/// verify the staged copy, back up the current app, swap, and roll back if
/// the swap fails.
pub(crate) fn promote_bundle(
    source_bundle: &Path,
    destination_directory: &Path,
    promotion_tools: &PromotionTools,
) -> Result<(), String> {
    create_directory(destination_directory)?;
    let destination_bundle = destination_directory.join("Astronomical.app");
    let staged_bundle =
        destination_directory.join(format!(".Astronomical.stage.{}", std::process::id()));
    let backup_bundle =
        destination_directory.join(format!(".Astronomical.backup.{}", std::process::id()));
    if staged_bundle.exists() || backup_bundle.exists() {
        return Err("a promotion staging or backup path already exists".to_owned());
    }
    if let Err(stage_error) = (promotion_tools.stage_copy)(source_bundle, &staged_bundle) {
        let _ = std::fs::remove_dir_all(&staged_bundle);
        return Err(stage_error);
    }
    let has_previous_app = destination_bundle.exists();
    if has_previous_app {
        rename_path(&destination_bundle, &backup_bundle)?;
    }
    if let Err(swap_error) = rename_path(&staged_bundle, &destination_bundle) {
        let _ = std::fs::remove_dir_all(&staged_bundle);
        return Err(rollback_message(
            &swap_error,
            has_previous_app,
            &backup_bundle,
            &destination_bundle,
        ));
    }
    if has_previous_app
        && let Err(disposal_error) = (promotion_tools.dispose_backup)(&backup_bundle)
    {
        println!(
            "the previous app is kept at {} ({disposal_error})",
            backup_bundle.display()
        );
    }
    Ok(())
}

fn rollback_message(
    swap_error: &str,
    has_previous_app: bool,
    backup_bundle: &Path,
    destination_bundle: &Path,
) -> String {
    if !has_previous_app {
        return format!("Stable promotion failed: {swap_error}");
    }
    match rename_path(backup_bundle, destination_bundle) {
        Ok(()) => format!("Stable promotion failed; the prior app was restored ({swap_error})"),
        Err(_) => format!(
            "Stable promotion and rollback failed; the prior app is preserved at {} ({swap_error})",
            backup_bundle.display()
        ),
    }
}

fn stage_and_verify_copy(source_bundle: &Path, staged_bundle: &Path) -> Result<(), String> {
    run_checked(
        Command::new("ditto").arg(source_bundle).arg(staged_bundle),
        "could not stage the Stable bundle",
    )?;
    verify_seal(staged_bundle)
}

/// The previous app goes to the Trash, never straight to deletion.
fn move_backup_to_trash(backup_bundle: &Path) -> Result<(), String> {
    run_checked(
        Command::new("trash").arg(backup_bundle),
        "could not move the previous app to the Trash",
    )
}

fn verify_seal(bundle_path: &Path) -> Result<(), String> {
    run_checked(
        Command::new("codesign")
            .args(["--verify", "--deep", "--strict"])
            .arg(bundle_path),
        &format!("the bundle seal does not verify: {}", bundle_path.display()),
    )
}

fn tool_version(executable_path: &Path) -> Result<String, String> {
    match Command::new(executable_path).arg("--version").output() {
        Ok(version_output) if version_output.status.success() => {
            Ok(String::from_utf8_lossy(&version_output.stdout).into_owned())
        }
        _ => Err(format!(
            "could not read the version of {}",
            executable_path.display()
        )),
    }
}

fn read_plist_document(bundle_path: &Path) -> Result<serde_json::Value, String> {
    let plist_output = Command::new("plutil")
        .args(["-convert", "json", "-o", "-"])
        .arg(bundle_path.join("Contents/Info.plist"))
        .output();
    let plist_json_text = match plist_output {
        Ok(plist_output) if plist_output.status.success() => {
            String::from_utf8_lossy(&plist_output.stdout).into_owned()
        }
        _ => return Err("the bundle Info.plist could not be read".to_owned()),
    };
    match serde_json::from_str(&plist_json_text) {
        Ok(plist_document) => Ok(plist_document),
        Err(parse_error) => Err(format!("Info.plist did not convert to JSON: {parse_error}")),
    }
}

fn plist_text(plist_document: &serde_json::Value, plist_key: &str) -> Result<String, String> {
    match plist_document[plist_key].as_str() {
        Some(plist_value) if !plist_value.is_empty() => Ok(plist_value.to_owned()),
        _ => Err(format!("Stable bundle {plist_key} is unavailable")),
    }
}

fn run_checked(tool_command: &mut Command, failure_message: &str) -> Result<(), String> {
    match tool_command.output() {
        Ok(tool_output) if tool_output.status.success() => Ok(()),
        _ => Err(failure_message.to_owned()),
    }
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

/// Packages Stable from this clean commit, then installs it (or previews
/// the installation), as one release-only command.
pub(crate) fn build_and_install_stable(
    repository_root: &Path,
    is_dry_run: bool,
) -> Result<(), String> {
    let build_step = start_step("build-and-validate-stable");
    let package_arguments = [
        "package".to_owned(),
        "--channel".to_owned(),
        "stable".to_owned(),
    ];
    if let Err(package_error) = crate::app_bundle::run(repository_root, &package_arguments) {
        build_step.finish("failed");
        return Err(package_error);
    }
    build_step.finish("success");
    let install_step = start_step(if is_dry_run {
        "preview-stable-installation"
    } else {
        "install-stable"
    });
    let install_result = install_stable(&InstallRequest {
        source_bundle: default_stable_bundle(repository_root),
        is_dry_run,
    });
    install_step.finish(if install_result.is_ok() {
        "success"
    } else {
        "failed"
    });
    install_result
}
