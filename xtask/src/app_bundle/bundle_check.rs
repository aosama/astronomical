//! Checks an assembled bundle without launching it: layout, plist identity,
//! signatures, and that the bundled daemon is this build's. Typed port of
//! the `--bundle-only` half of the retired `validate-macos-app.sh`.
//!
//! Launching is not a check here: in the edit loop opening the new app *is*
//! the launch, and a launch-then-kill rehearsal before it forced the old
//! wrapper to stop and restart the running instance around every build. The
//! launch rehearsal is `cargo xtask app smoke-test`.
//!
//! Layout and plist checks collect every problem instead of stopping at the
//! first, so one run names everything wrong with a bundle.

use std::path::Path;
use std::process::Command;

use super::build_identity::BuildIdentity;
use super::channel_profile::{ChannelProfile, SandboxPolicy};
use super::tool_command::run_quiet_tool;

const BUNDLED_EXECUTABLE_NAMES: [&str; 4] = [
    "astronomical-menu",
    "astronomicald",
    "astronomical-inference-worker",
    "astronomical",
];
const REQUIRED_RESOURCE_NAMES: [&str; 4] = [
    "LICENSE",
    "THIRD_PARTY_NOTICES",
    "RUST_DEPENDENCY_NOTICES",
    "Astronomical.icns",
];
const SPARKLE_PLIST_KEYS: [&str; 3] = ["SUFeedURL", "SUPublicEDKey", "SUEnableAutomaticChecks"];

pub(crate) fn check_bundle(
    bundle_path: &Path,
    channel_profile: &ChannelProfile,
    build_identity: &BuildIdentity,
) -> Result<(), String> {
    run_quiet_tool(
        Command::new("plutil")
            .arg("-lint")
            .arg(bundle_path.join("Contents/Info.plist")),
    )?;
    let plist_json_text = run_quiet_tool(
        Command::new("plutil")
            .args(["-convert", "json", "-o", "-"])
            .arg(bundle_path.join("Contents/Info.plist")),
    )?;
    let plist_document: serde_json::Value = match serde_json::from_str(&plist_json_text) {
        Ok(plist_document) => plist_document,
        Err(parse_error) => {
            return Err(format!("Info.plist did not convert to JSON: {parse_error}"));
        }
    };
    let mut bundle_problems = layout_problems(bundle_path, channel_profile);
    bundle_problems.extend(plist_identity_problems(&plist_document, channel_profile));
    if !bundle_problems.is_empty() {
        return Err(format!(
            "the bundle failed its checks:\n  {}",
            bundle_problems.join("\n  ")
        ));
    }
    check_signatures(bundle_path, channel_profile)?;
    check_daemon_identity(bundle_path, build_identity)
}

/// Files that must, or for the App Store must not, be in the bundle.
pub(crate) fn layout_problems(bundle_path: &Path, channel_profile: &ChannelProfile) -> Vec<String> {
    let contents_path = bundle_path.join("Contents");
    let mut layout_problems = Vec::new();
    for executable_name in BUNDLED_EXECUTABLE_NAMES {
        if !is_executable_file(&contents_path.join("MacOS").join(executable_name)) {
            layout_problems.push(format!(
                "bundled executable is unavailable: {executable_name}"
            ));
        }
    }
    for resource_name in REQUIRED_RESOURCE_NAMES {
        if !is_non_empty_file(&contents_path.join("Resources").join(resource_name)) {
            layout_problems.push(format!(
                "required bundled resource is unavailable: {resource_name}"
            ));
        }
    }
    if !is_non_empty_file(&contents_path.join("Resources/share/mlx/mlx.metallib")) {
        layout_problems.push("required bundled MLX AOT metallib is unavailable".to_owned());
    }
    let has_sparkle_framework = contents_path.join("Frameworks/Sparkle.framework").is_dir();
    let has_sparkle_license = is_non_empty_file(&contents_path.join("Resources/SPARKLE_LICENSE"));
    if channel_profile.ships_sparkle && !(has_sparkle_framework && has_sparkle_license) {
        layout_problems.push("the Sparkle framework or its license is missing".to_owned());
    }
    if !channel_profile.ships_sparkle && (has_sparkle_framework || has_sparkle_license) {
        layout_problems
            .push("an App Store bundle must not embed Sparkle or its license".to_owned());
    }
    layout_problems.extend(hidden_entry_problems(bundle_path));
    layout_problems
}

/// Hidden entries reported before the list is cut short; enough to name a
/// stray directory without burying the other layout problems.
const MAXIMUM_REPORTED_HIDDEN_ENTRIES: usize = 10;

/// Every file or directory inside the bundle whose name starts with a dot.
///
/// A correct bundle has none, Sparkle's framework included. They arrive when
/// a git-ignored file sits inside a directory a build copies whole: the
/// clean-worktree check cannot see ignored files, and v0.2.88 shipped two
/// Playwright debug logs from a `.playwright-cli/` folder inside the Thin
/// Talk canvas resources that way. Symlinks are reported, never followed, so
/// the walk stays inside the bundle.
pub(crate) fn hidden_entry_problems(bundle_path: &Path) -> Vec<String> {
    let mut hidden_entry_problems = Vec::new();
    let mut pending_directories = vec![bundle_path.to_path_buf()];
    while let Some(directory_path) = pending_directories.pop() {
        let Ok(directory_entries) = std::fs::read_dir(&directory_path) else {
            continue;
        };
        for directory_entry in directory_entries.flatten() {
            let entry_path = directory_entry.path();
            if directory_entry
                .file_name()
                .to_string_lossy()
                .starts_with('.')
            {
                if hidden_entry_problems.len() < MAXIMUM_REPORTED_HIDDEN_ENTRIES {
                    let relative_path = entry_path.strip_prefix(bundle_path).unwrap_or(&entry_path);
                    hidden_entry_problems.push(format!(
                        "hidden entry inside the bundle: {}",
                        relative_path.display()
                    ));
                }
                continue;
            }
            match std::fs::symlink_metadata(&entry_path) {
                Ok(entry_metadata) if entry_metadata.is_dir() => {
                    pending_directories.push(entry_path)
                }
                _ => {}
            }
        }
    }
    hidden_entry_problems.sort();
    hidden_entry_problems
}

/// The plist must describe exactly the channel it was built for; a stray
/// key here would point a Stable app at the Development state directory.
pub(crate) fn plist_identity_problems(
    plist_document: &serde_json::Value,
    channel_profile: &ChannelProfile,
) -> Vec<String> {
    let mut identity_problems = Vec::new();
    expect_plist_text(
        plist_document,
        "AstronomicalChannel",
        channel_profile.channel.as_channel_name(),
        &mut identity_problems,
    );
    expect_plist_text(
        plist_document,
        "CFBundleIdentifier",
        channel_profile.bundle_identifier,
        &mut identity_problems,
    );
    expect_plist_text(
        plist_document,
        "CFBundleIconFile",
        "Astronomical.icns",
        &mut identity_problems,
    );
    if plist_document["AstronomicalDaemonPort"].as_u64()
        != Some(u64::from(channel_profile.daemon_port))
    {
        identity_problems.push("bundle channel and daemon port disagree".to_owned());
    }
    match channel_profile.state_directory_name {
        Some(state_directory_name) => expect_plist_text(
            plist_document,
            "AstronomicalStateDirectoryName",
            state_directory_name,
            &mut identity_problems,
        ),
        None if !plist_document["AstronomicalStateDirectoryName"].is_null() => identity_problems
            .push("an App Store bundle must not embed a state directory name".to_owned()),
        None => {}
    }
    if !is_all_digits(plist_document["CFBundleVersion"].as_str(), None) {
        identity_problems.push("invalid bundle build number".to_owned());
    }
    if !is_all_digits(plist_document["AstronomicalBuildDate"].as_str(), Some(8)) {
        identity_problems.push("bundle build date must use YYYYMMDD".to_owned());
    }
    if channel_profile.requires_clean_worktree
        && plist_document["AstronomicalBuildDirty"].as_bool() != Some(false)
    {
        identity_problems.push("a release bundle must not be dirty".to_owned());
    }
    if !registers_astronomical_url_scheme(plist_document) {
        identity_problems.push(
            "the bundle must register the astronomical URL scheme for the chat handoff".to_owned(),
        );
    }
    sparkle_key_problems(plist_document, channel_profile, &mut identity_problems);
    identity_problems
}

fn sparkle_key_problems(
    plist_document: &serde_json::Value,
    channel_profile: &ChannelProfile,
    identity_problems: &mut Vec<String>,
) {
    for sparkle_plist_key in SPARKLE_PLIST_KEYS {
        let has_sparkle_key = !plist_document[sparkle_plist_key].is_null();
        if channel_profile.ships_sparkle && !has_sparkle_key {
            identity_problems.push(format!(
                "the bundle is missing the {sparkle_plist_key} update key"
            ));
        }
        if !channel_profile.ships_sparkle && has_sparkle_key {
            identity_problems.push(format!(
                "an App Store bundle must not contain the {sparkle_plist_key} update key"
            ));
        }
    }
}

fn expect_plist_text(
    plist_document: &serde_json::Value,
    plist_key: &str,
    expected_text: &str,
    identity_problems: &mut Vec<String>,
) {
    if plist_document[plist_key].as_str() != Some(expected_text) {
        identity_problems.push(format!("{plist_key} must be {expected_text:?}"));
    }
}

fn is_all_digits(candidate_text: Option<&str>, required_length: Option<usize>) -> bool {
    let Some(candidate_text) = candidate_text else {
        return false;
    };
    let has_required_length = match required_length {
        Some(required_length) => candidate_text.len() == required_length,
        None => !candidate_text.is_empty(),
    };
    if !has_required_length {
        return false;
    }
    for candidate_byte in candidate_text.bytes() {
        if !candidate_byte.is_ascii_digit() {
            return false;
        }
    }
    true
}

fn registers_astronomical_url_scheme(plist_document: &serde_json::Value) -> bool {
    let Some(url_types) = plist_document["CFBundleURLTypes"].as_array() else {
        return false;
    };
    for url_type in url_types {
        let Some(url_schemes) = url_type["CFBundleURLSchemes"].as_array() else {
            continue;
        };
        for url_scheme in url_schemes {
            if url_scheme.as_str() == Some("astronomical") {
                return true;
            }
        }
    }
    false
}

/// Strict seal verification, plus for the App Store the sandbox
/// entitlements on every executable that must run sandboxed, and proof the
/// menu app does not link Sparkle.
fn check_signatures(bundle_path: &Path, channel_profile: &ChannelProfile) -> Result<(), String> {
    run_quiet_tool(
        Command::new("codesign")
            .args(["--verify", "--deep", "--strict"])
            .arg(bundle_path),
    )?;
    if channel_profile.sandbox_policy != SandboxPolicy::AppStoreSandbox {
        return Ok(());
    }
    let macos_directory = bundle_path.join("Contents/MacOS");
    for (executable_name, entitlement_keys) in [
        (
            "astronomical-menu",
            vec![
                "com.apple.security.app-sandbox",
                "com.apple.security.network.client",
                "com.apple.security.network.server",
            ],
        ),
        ("astronomicald", vec!["com.apple.security.inherit"]),
        (
            "astronomical-inference-worker",
            vec!["com.apple.security.inherit"],
        ),
    ] {
        let entitlements_text =
            signature_entitlements_text(&macos_directory.join(executable_name))?;
        for entitlement_key in entitlement_keys {
            if !entitlements_text.contains(entitlement_key) {
                return Err(format!(
                    "App Store executable {executable_name} is missing the {entitlement_key} entitlement"
                ));
            }
        }
    }
    let linked_libraries = run_quiet_tool(
        Command::new("otool")
            .arg("-L")
            .arg(macos_directory.join("astronomical-menu")),
    )?;
    if linked_libraries.to_lowercase().contains("sparkle") {
        return Err("the App Store menu executable must not link the Sparkle framework".to_owned());
    }
    Ok(())
}

/// codesign prints the decoded DER entitlements on stdout; the bare key
/// name matches either the decoded or the XML form.
fn signature_entitlements_text(executable_path: &Path) -> Result<String, String> {
    run_quiet_tool(
        Command::new("codesign")
            .args(["-d", "--entitlements", "-"])
            .arg(executable_path),
    )
}

fn check_daemon_identity(bundle_path: &Path, build_identity: &BuildIdentity) -> Result<(), String> {
    let daemon_version_text = run_quiet_tool(
        Command::new(bundle_path.join("Contents/MacOS/astronomicald")).arg("--version"),
    )?;
    if !daemon_version_matches(&daemon_version_text, build_identity) {
        return Err(format!(
            "the bundled daemon reports {daemon_version_text:?}, not version {} at commit {}",
            build_identity.application_version, build_identity.build_commit
        ));
    }
    Ok(())
}

/// The daemon must report this build's version and commit, so an app can
/// never ship a daemon left over from an earlier build.
pub(crate) fn daemon_version_matches(
    daemon_version_text: &str,
    build_identity: &BuildIdentity,
) -> bool {
    version_text_reports(
        daemon_version_text,
        &build_identity.application_version,
        &build_identity.build_commit,
    )
}

/// True when a `--version` line names this version followed by this commit;
/// shared with Stable installation, which reads both from the bundle plist.
pub(crate) fn version_text_reports(
    version_text: &str,
    application_version: &str,
    build_commit: &str,
) -> bool {
    let Some(version_position) = version_text.find(application_version) else {
        return false;
    };
    version_text[version_position..].contains(build_commit)
}

fn is_executable_file(file_path: &Path) -> bool {
    match std::fs::metadata(file_path) {
        Ok(file_metadata) => {
            file_metadata.is_file()
                && std::os::unix::fs::PermissionsExt::mode(&file_metadata.permissions()) & 0o111
                    != 0
        }
        Err(_) => false,
    }
}

/// A symlink does not count: a bundle must carry its own bytes, not a
/// pointer into a cache macOS may purge.
fn is_non_empty_file(file_path: &Path) -> bool {
    match std::fs::symlink_metadata(file_path) {
        Ok(file_metadata) => file_metadata.file_type().is_file() && file_metadata.len() > 0,
        Err(_) => false,
    }
}
