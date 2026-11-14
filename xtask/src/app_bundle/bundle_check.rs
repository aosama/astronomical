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

use super::application_group::{APPLICATION_GROUP_PLIST_KEY, APPLICATION_GROUPS_ENTITLEMENT_KEY};
use super::build_identity::BuildIdentity;
use super::bundle_assembly::PRIVACY_MANIFEST_FILE_NAME;
use super::channel_profile::{
    AGENT_EXECUTABLE_NAME, APPLICATION_EXECUTABLE_NAME, ChannelProfile, DAEMON_EXECUTABLE_NAME,
    SandboxPolicy, TERMINAL_COMMAND_EXECUTABLE_NAME, WORKER_EXECUTABLE_NAME,
};
use super::tool_command::run_quiet_tool;

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
    let plist_document = read_plist_document(&bundle_path.join("Contents/Info.plist"))?;
    let mut bundle_problems = layout_problems(bundle_path, channel_profile);
    bundle_problems.extend(plist_identity_problems(&plist_document, channel_profile));
    bundle_problems.extend(bundled_agent_problems(
        bundle_path,
        &plist_document,
        channel_profile,
    )?);
    if !bundle_problems.is_empty() {
        return Err(format!(
            "the bundle failed its checks:\n  {}",
            bundle_problems.join("\n  ")
        ));
    }
    check_signatures(bundle_path, channel_profile)?;
    check_daemon_identity(bundle_path, build_identity)
}

/// The role and identity problems of the agent inside `bundle_path`. A missing
/// agent plist is reported as one more problem rather than as a tool error,
/// so a single run still lists everything wrong with the bundle.
pub(crate) fn bundled_agent_problems(
    bundle_path: &Path,
    application_plist_document: &serde_json::Value,
    channel_profile: &ChannelProfile,
) -> Result<Vec<String>, String> {
    let agent_plist_path = bundle_path
        .join(channel_profile.agent_bundle_relative_path())
        .join("Contents/Info.plist");
    if !agent_plist_path.is_file() {
        return Ok(vec![
            "the menu-bar agent's Info.plist is unavailable".to_owned(),
        ]);
    }
    let agent_plist_document = read_plist_document(&agent_plist_path)?;
    let mut agent_problems = process_role_problems(
        application_plist_document,
        &agent_plist_document,
        channel_profile,
    );
    agent_problems.extend(application_group_problems(
        application_plist_document,
        &agent_plist_document,
        channel_profile,
    ));
    Ok(agent_problems)
}

/// A sandboxed channel's two applications each have a container of their
/// own, so both plists must name one app group, the same one, ending in the
/// channel's group name: with different groups, or none, the chat app and the
/// engine would each keep separate state. An unsandboxed channel must name
/// none, because its state stays in the home directory.
pub(crate) fn application_group_problems(
    application_plist_document: &serde_json::Value,
    agent_plist_document: &serde_json::Value,
    channel_profile: &ChannelProfile,
) -> Vec<String> {
    let application_group = application_plist_document[APPLICATION_GROUP_PLIST_KEY].as_str();
    let agent_group = agent_plist_document[APPLICATION_GROUP_PLIST_KEY].as_str();
    let Some(application_group_name) = channel_profile.application_group_name else {
        if application_group.is_some() || agent_group.is_some() {
            return vec!["an unsandboxed bundle must not name an app group".to_owned()];
        }
        return Vec::new();
    };
    let mut group_problems = Vec::new();
    let expected_group_suffix = format!(".{application_group_name}");
    match application_group {
        Some(application_group)
            if application_group.ends_with(&expected_group_suffix)
                && application_group.len() > expected_group_suffix.len() => {}
        _ => group_problems.push(format!(
            "the main app must name an app group ending in {expected_group_suffix}"
        )),
    }
    if agent_group != application_group {
        group_problems.push("the agent and the main app must name the same app group".to_owned());
    }
    group_problems
}

fn read_plist_document(plist_path: &Path) -> Result<serde_json::Value, String> {
    let plist_json_text = run_quiet_tool(
        Command::new("plutil")
            .args(["-convert", "json", "-o", "-"])
            .arg(plist_path),
    )?;
    match serde_json::from_str(&plist_json_text) {
        Ok(plist_document) => Ok(plist_document),
        Err(parse_error) => Err(format!(
            "{} did not convert to JSON: {parse_error}",
            plist_path.display()
        )),
    }
}

/// Each process keeps one role for its whole life: the main app is a regular
/// application and the agent an accessory. A main app marked `LSUIElement`
/// would have no menu bar, which is the defect this layout exists to prevent;
/// an agent without it would put a second icon in the Dock. The agent must
/// also belong to the same instance and build as the app around it, or its
/// engine would serve another channel's port and state directory.
pub(crate) fn process_role_problems(
    application_plist_document: &serde_json::Value,
    agent_plist_document: &serde_json::Value,
    channel_profile: &ChannelProfile,
) -> Vec<String> {
    let mut role_problems = Vec::new();
    if application_plist_document["LSUIElement"].as_bool() == Some(true) {
        role_problems
            .push("the main app must be a regular application, not LSUIElement".to_owned());
    }
    expect_plist_text(
        application_plist_document,
        "CFBundleExecutable",
        APPLICATION_EXECUTABLE_NAME,
        &mut role_problems,
    );
    if agent_plist_document["LSUIElement"].as_bool() != Some(true) {
        role_problems.push("the menu-bar agent must be LSUIElement".to_owned());
    }
    if !agent_plist_document["CFBundleURLTypes"].is_null() {
        role_problems.push("links belong to the main app; the agent must register none".to_owned());
    }
    role_problems.extend(agent_identity_problems(
        agent_plist_document,
        channel_profile,
    ));
    role_problems.extend(shared_build_identity_problems(
        application_plist_document,
        agent_plist_document,
    ));
    role_problems
}

/// The agent's own naming: its executable, its identifier (which the main app
/// registers at login), and the channel it serves.
fn agent_identity_problems(
    agent_plist_document: &serde_json::Value,
    channel_profile: &ChannelProfile,
) -> Vec<String> {
    let mut identity_problems = Vec::new();
    for (agent_plist_key, expected_text) in [
        ("CFBundleExecutable", AGENT_EXECUTABLE_NAME.to_owned()),
        (
            "CFBundleIdentifier",
            channel_profile.agent_bundle_identifier(),
        ),
        (
            "AstronomicalChannel",
            channel_profile.channel.as_channel_name().to_owned(),
        ),
    ] {
        if agent_plist_document[agent_plist_key].as_str() != Some(expected_text.as_str()) {
            identity_problems.push(format!(
                "the agent's {agent_plist_key} does not match the bundle"
            ));
        }
    }
    identity_problems
}

/// Both processes read their instance and build from their own plist, so the
/// two plists must agree on every such value.
fn shared_build_identity_problems(
    application_plist_document: &serde_json::Value,
    agent_plist_document: &serde_json::Value,
) -> Vec<String> {
    let mut disagreement_problems = Vec::new();
    for shared_identity_key in [
        "AstronomicalDaemonPort",
        "AstronomicalStateDirectoryName",
        "CFBundleVersion",
        "CFBundleShortVersionString",
        "AstronomicalBuildCommit",
        "AstronomicalBuildDirty",
    ] {
        if agent_plist_document[shared_identity_key]
            != application_plist_document[shared_identity_key]
        {
            disagreement_problems.push(format!(
                "the agent and the main app disagree on {shared_identity_key}"
            ));
        }
    }
    disagreement_problems
}

/// The agent's executable must be in its nested bundle, where macOS looks for
/// a helper the main app registers at login.
fn agent_layout_problems(bundle_path: &Path, channel_profile: &ChannelProfile) -> Vec<String> {
    if is_executable_file(&bundle_path.join(channel_profile.agent_executable_relative_path())) {
        return Vec::new();
    }
    vec!["the menu-bar agent's executable is unavailable".to_owned()]
}

/// The executables of the main bundle's `Contents/MacOS`; the agent's lives
/// in its own nested bundle and is checked there. A channel that does not
/// bundle the terminal command must not carry one either: a stray copy in a
/// store bundle is an executable outside the sandbox, which App Store
/// Connect refuses.
fn executable_layout_problems(
    macos_directory: &Path,
    channel_profile: &ChannelProfile,
) -> Vec<String> {
    let mut executable_problems = Vec::new();
    let mut required_executable_names = vec![APPLICATION_EXECUTABLE_NAME];
    required_executable_names.extend(channel_profile.bundled_rust_executable_names());
    for executable_name in required_executable_names {
        if !is_executable_file(&macos_directory.join(executable_name)) {
            executable_problems.push(format!(
                "bundled executable is unavailable: {executable_name}"
            ));
        }
    }
    if !channel_profile.bundles_terminal_command
        && macos_directory
            .join(TERMINAL_COMMAND_EXECUTABLE_NAME)
            .exists()
    {
        executable_problems.push(
            "this channel's bundle must not carry the terminal command, which cannot be sandboxed"
                .to_owned(),
        );
    }
    executable_problems
}

/// Files that must, or for the App Store must not, be in the bundle.
pub(crate) fn layout_problems(bundle_path: &Path, channel_profile: &ChannelProfile) -> Vec<String> {
    let contents_path = bundle_path.join("Contents");
    let mut layout_problems = agent_layout_problems(bundle_path, channel_profile);
    layout_problems.extend(executable_layout_problems(
        &contents_path.join("MacOS"),
        channel_profile,
    ));
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
    layout_problems.extend(privacy_manifest_problems(bundle_path, channel_profile));
    layout_problems.extend(hidden_entry_problems(bundle_path));
    layout_problems
}

/// A sandboxed channel's two applications must each carry the privacy
/// manifest, which is part of what App Store Connect reviews.
fn privacy_manifest_problems(bundle_path: &Path, channel_profile: &ChannelProfile) -> Vec<String> {
    if channel_profile.sandbox_policy != SandboxPolicy::AppStoreSandbox {
        return Vec::new();
    }
    let mut manifest_problems = Vec::new();
    let application_contents = [
        ("the main app", bundle_path.join("Contents")),
        (
            "the menu-bar agent",
            bundle_path
                .join(channel_profile.agent_bundle_relative_path())
                .join("Contents"),
        ),
    ];
    for (application_label, contents_path) in application_contents {
        let manifest_path = contents_path
            .join("Resources")
            .join(PRIVACY_MANIFEST_FILE_NAME);
        if !is_non_empty_file(&manifest_path) {
            manifest_problems.push(format!(
                "{application_label} of an App Store bundle is missing its privacy manifest"
            ));
        }
    }
    manifest_problems
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
    if registered_url_schemes(plist_document) != [channel_profile.chat_url_scheme] {
        identity_problems.push(format!(
            "the bundle must register only the {} URL scheme for the chat handoff",
            channel_profile.chat_url_scheme
        ));
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

fn registered_url_schemes(plist_document: &serde_json::Value) -> Vec<&str> {
    let mut registered_schemes = Vec::new();
    let Some(url_types) = plist_document["CFBundleURLTypes"].as_array() else {
        return registered_schemes;
    };
    for url_type in url_types {
        let Some(url_schemes) = url_type["CFBundleURLSchemes"].as_array() else {
            continue;
        };
        for url_scheme in url_schemes {
            if let Some(url_scheme) = url_scheme.as_str() {
                registered_schemes.push(url_scheme);
            }
        }
    }
    registered_schemes
}

/// Strict seal verification, plus for the App Store the sandbox
/// entitlements on every executable that must run sandboxed, and proof that
/// neither Swift executable links Sparkle.
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
    let application_executable = macos_directory.join(APPLICATION_EXECUTABLE_NAME);
    let agent_executable = bundle_path.join(channel_profile.agent_executable_relative_path());
    // Both applications must belong to the app group, or the sandbox would
    // refuse one of them the container their shared state lives in.
    let sandbox_entitlement_keys = vec![
        "com.apple.security.app-sandbox",
        "com.apple.security.network.client",
        "com.apple.security.network.server",
        APPLICATION_GROUPS_ENTITLEMENT_KEY,
    ];
    check_entitlements(&[
        (
            application_executable.clone(),
            sandbox_entitlement_keys.clone(),
        ),
        (agent_executable.clone(), sandbox_entitlement_keys),
        (
            macos_directory.join(DAEMON_EXECUTABLE_NAME),
            INHERITED_SANDBOX_ENTITLEMENT_KEYS.to_vec(),
        ),
        (
            macos_directory.join(WORKER_EXECUTABLE_NAME),
            INHERITED_SANDBOX_ENTITLEMENT_KEYS.to_vec(),
        ),
    ])?;
    check_no_sparkle_linkage(&[application_executable, agent_executable])
}

/// What the engine and the worker must be signed with. App Store Connect
/// refuses an executable that only inherits: validation of the first store
/// package failed with error 90296 until both keys were present. The pair
/// means "sandboxed, with the sandbox of the application that started me".
const INHERITED_SANDBOX_ENTITLEMENT_KEYS: [&str; 2] = [
    "com.apple.security.app-sandbox",
    "com.apple.security.inherit",
];

/// Every listed executable must carry every entitlement listed with it.
fn check_entitlements(
    required_entitlements: &[(std::path::PathBuf, Vec<&str>)],
) -> Result<(), String> {
    for (executable_path, entitlement_keys) in required_entitlements {
        let entitlements_text = signature_entitlements_text(executable_path)?;
        for entitlement_key in entitlement_keys {
            if !entitlements_text.contains(entitlement_key) {
                return Err(format!(
                    "App Store executable {} is missing the {entitlement_key} entitlement",
                    executable_path.display()
                ));
            }
        }
    }
    Ok(())
}

/// A store build must contain no self-updating code (App Review guideline
/// 2.4.5(vii)), so neither Swift executable may link Sparkle.
fn check_no_sparkle_linkage(swift_executables: &[std::path::PathBuf]) -> Result<(), String> {
    for swift_executable in swift_executables {
        let linked_libraries =
            run_quiet_tool(Command::new("otool").arg("-L").arg(swift_executable))?;
        if linked_libraries.to_lowercase().contains("sparkle") {
            return Err(format!(
                "an App Store executable must not link the Sparkle framework: {}",
                swift_executable.display()
            ));
        }
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
