//! Structural checks on an assembled bundle, without launching it.
//!
//! Launching is not a check in the edit loop: opening the new app *is* the
//! launch, and a launch-then-kill rehearsal before it is what forced the
//! shell wrapper to stop and restart the running instance around every
//! build. The launch rehearsal belongs to CI and release.

use std::path::Path;
use std::process::Command;

use super::build_identity::BuildIdentity;
use super::tool_command::run_quiet_tool;

pub(crate) fn check_bundle(
    bundle_path: &Path,
    build_identity: &BuildIdentity,
) -> Result<(), String> {
    run_quiet_tool(
        Command::new("plutil")
            .arg("-lint")
            .arg(bundle_path.join("Contents/Info.plist")),
    )?;
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
    let Some(version_position) = daemon_version_text.find(&build_identity.application_version)
    else {
        return false;
    };
    daemon_version_text[version_position..].contains(&build_identity.build_commit)
}
