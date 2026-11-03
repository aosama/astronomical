//! `cargo xtask release validate-distribution-app --app-bundle PATH
//! --team-id ID`: proves every code object in a Stable bundle carries a
//! Developer ID signature from the release team, with Hardened Runtime and a
//! secure timestamp, before the bundle goes to notarization (which would
//! reject it later and slower). Typed port of the retired
//! `validate-distribution-app.sh`.
//!
//! The required objects are checked for presence by name; the signature
//! checks then run once over every code object found in the bundle, which
//! covers the named ones too. The shell validated the named list and then
//! the discovered list again, signing-checking most objects twice.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::step_log::{log_line, start_step};

/// The objects a Stable bundle must contain, innermost first.
const REQUIRED_CODE_OBJECT_PATHS: [&str; 9] = [
    "Contents/Frameworks/Sparkle.framework/Versions/B/Autoupdate",
    "Contents/Frameworks/Sparkle.framework/Versions/B/Updater.app",
    "Contents/Frameworks/Sparkle.framework/Versions/B/XPCServices/Downloader.xpc",
    "Contents/Frameworks/Sparkle.framework/Versions/B/XPCServices/Installer.xpc",
    "Contents/Frameworks/Sparkle.framework",
    "Contents/MacOS/astronomical-menu",
    "Contents/MacOS/astronomicald",
    "Contents/MacOS/astronomical-inference-worker",
    "Contents/MacOS/astronomical",
];

/// A Team ID is ten characters; anything shorter is a typo, not a team.
const MINIMUM_TEAM_IDENTIFIER_LENGTH: usize = 7;

pub(crate) fn validate_distribution_app(
    app_bundle: &Path,
    team_identifier: &str,
) -> Result<(), String> {
    if !app_bundle.is_dir() {
        return Err("--app-bundle must name an application bundle".to_owned());
    }
    if !is_non_empty_regular_file(&app_bundle.join("Contents/Resources/share/mlx/mlx.metallib")) {
        return Err("required bundled MLX AOT metallib is unavailable".to_owned());
    }
    if team_identifier.len() < MINIMUM_TEAM_IDENTIFIER_LENGTH {
        return Err("--team-id is required".to_owned());
    }
    for required_path in REQUIRED_CODE_OBJECT_PATHS {
        if !app_bundle.join(required_path).exists() {
            return Err(format!(
                "code object is unavailable: {}",
                app_bundle.join(required_path).display()
            ));
        }
    }
    let seal_step = start_step("verify-app-seal");
    run_codesign_verify(app_bundle)?;
    seal_step.finish("success");
    let code_objects = discover_code_objects(app_bundle)?;
    for code_object_path in &code_objects {
        validate_code_object(code_object_path, team_identifier)?;
    }
    println!(
        "Validated Developer ID distribution signature for every bundled code object ({}).",
        code_objects.len()
    );
    Ok(())
}

fn validate_code_object(code_object_path: &Path, team_identifier: &str) -> Result<(), String> {
    let signature_output = match Command::new("codesign")
        .args(["--display", "--verbose=4"])
        .arg(code_object_path)
        .output()
    {
        Ok(signature_output) => signature_output,
        Err(spawn_error) => return Err(format!("codesign could not start: {spawn_error}")),
    };
    // codesign writes the signature details to stderr.
    let mut signature_details = String::from_utf8_lossy(&signature_output.stderr).into_owned();
    signature_details.push_str(&String::from_utf8_lossy(&signature_output.stdout));
    if let Some(signature_problem) = signature_problems(&signature_details, team_identifier).first()
    {
        return Err(format!(
            "{signature_problem}: {}",
            code_object_path.display()
        ));
    }
    log_line(&format!(
        "operation=validate-code-signature object={:?} status=success",
        file_name_of(code_object_path)
    ));
    Ok(())
}

/// What a distributable signature must show, worded as the shell worded it.
pub(crate) fn signature_problems(signature_details: &str, team_identifier: &str) -> Vec<String> {
    let mut signature_problems = Vec::new();
    if !signature_details.contains("Authority=Developer ID Application:") {
        signature_problems.push("code object is not signed by Developer ID Application".to_owned());
    }
    if !signature_details.contains(&format!("TeamIdentifier={team_identifier}")) {
        signature_problems
            .push("code object Team ID does not match release configuration".to_owned());
    }
    if !has_hardened_runtime_flag(signature_details) {
        signature_problems.push("code object does not enable Hardened Runtime".to_owned());
    }
    if !signature_details.contains("Timestamp=") {
        signature_problems.push("code object does not have a secure timestamp".to_owned());
    }
    signature_problems
}

/// The runtime flag appears in the `flags=0x10000(runtime)` field.
fn has_hardened_runtime_flag(signature_details: &str) -> bool {
    for detail_line in signature_details.lines() {
        if let Some(flags_position) = detail_line.find("flags=")
            && detail_line[flags_position..].contains("(runtime)")
        {
            return true;
        }
    }
    false
}

/// Every bundle, framework, and XPC service directory plus every executable
/// regular file, as `find -type d -name '*.app' ... -o -type f -perm -111`
/// found them; symlinks such as `Versions/Current` are not followed.
pub(crate) fn discover_code_objects(bundle_path: &Path) -> Result<Vec<PathBuf>, String> {
    let mut code_objects = vec![bundle_path.to_path_buf()];
    collect_code_objects(bundle_path, &mut code_objects)?;
    code_objects.sort();
    Ok(code_objects)
}

fn collect_code_objects(
    directory_path: &Path,
    code_objects: &mut Vec<PathBuf>,
) -> Result<(), String> {
    let directory_entries = match std::fs::read_dir(directory_path) {
        Ok(directory_entries) => directory_entries,
        Err(read_error) => {
            return Err(format!(
                "could not list {}: {read_error}",
                directory_path.display()
            ));
        }
    };
    for directory_entry in directory_entries.flatten() {
        let entry_path = directory_entry.path();
        let Ok(entry_metadata) = std::fs::symlink_metadata(&entry_path) else {
            continue;
        };
        if entry_metadata.is_dir() {
            if is_code_bundle_directory(&entry_path) {
                code_objects.push(entry_path.clone());
            }
            collect_code_objects(&entry_path, code_objects)?;
        } else if entry_metadata.is_file()
            && std::os::unix::fs::PermissionsExt::mode(&entry_metadata.permissions()) & 0o111 != 0
        {
            code_objects.push(entry_path);
        }
    }
    Ok(())
}

fn is_code_bundle_directory(directory_path: &Path) -> bool {
    match directory_path.extension() {
        Some(directory_extension) => matches!(
            directory_extension.to_str(),
            Some("app" | "xpc" | "framework")
        ),
        None => false,
    }
}

fn run_codesign_verify(app_bundle: &Path) -> Result<(), String> {
    match Command::new("codesign")
        .args(["--verify", "--deep", "--strict", "--verbose=2"])
        .arg(app_bundle)
        .status()
    {
        Ok(exit_status) if exit_status.success() => Ok(()),
        _ => Err(format!(
            "the app seal does not verify: {}",
            app_bundle.display()
        )),
    }
}

fn file_name_of(code_object_path: &Path) -> String {
    match code_object_path.file_name() {
        Some(file_name) => file_name.to_string_lossy().into_owned(),
        None => code_object_path.display().to_string(),
    }
}

fn is_non_empty_regular_file(file_path: &Path) -> bool {
    match std::fs::symlink_metadata(file_path) {
        Ok(file_metadata) => file_metadata.file_type().is_file() && file_metadata.len() > 0,
        Err(_) => false,
    }
}
