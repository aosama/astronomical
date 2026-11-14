//! Proof that the daemon inside a bundle is the one this build compiled, so
//! an app can never ship a daemon left over from an earlier build.
//!
//! A direct bundle's daemon is asked for its `--version`. A store bundle's
//! daemon cannot be asked: it is signed to inherit its sandbox from the agent
//! that starts it, and macOS kills it during sandbox setup when anything else
//! starts it. From a build it exits on SIGTRAP while the system log reports
//! that the sandbox registration failed. The version and commit it would
//! print are compiled into the file as text, so for a store bundle they are
//! read from there.

use std::path::Path;
use std::process::Command;

use super::build_identity::BuildIdentity;
use super::bundle_check::daemon_version_matches;
use super::channel_profile::{ChannelProfile, DAEMON_EXECUTABLE_NAME, SandboxPolicy};
use super::tool_command::run_quiet_tool;

pub(crate) fn check_daemon_identity(
    bundle_path: &Path,
    build_identity: &BuildIdentity,
    channel_profile: &ChannelProfile,
) -> Result<(), String> {
    let daemon_path = bundle_path
        .join("Contents/MacOS")
        .join(DAEMON_EXECUTABLE_NAME);
    if channel_profile.sandbox_policy == SandboxPolicy::AppStoreSandbox {
        return check_embedded_identity(&daemon_path, build_identity);
    }
    check_reported_identity(&daemon_path, build_identity)
}

fn check_reported_identity(
    daemon_path: &Path,
    build_identity: &BuildIdentity,
) -> Result<(), String> {
    let daemon_version_text = run_quiet_tool(Command::new(daemon_path).arg("--version"))?;
    if !daemon_version_matches(&daemon_version_text, build_identity) {
        return Err(format!(
            "the bundled daemon reports {daemon_version_text:?}, not version {} at commit {}",
            build_identity.application_version, build_identity.build_commit
        ));
    }
    Ok(())
}

fn check_embedded_identity(
    daemon_path: &Path,
    build_identity: &BuildIdentity,
) -> Result<(), String> {
    let daemon_bytes = match std::fs::read(daemon_path) {
        Ok(daemon_bytes) => daemon_bytes,
        Err(read_error) => {
            return Err(format!(
                "could not read {}: {read_error}",
                daemon_path.display()
            ));
        }
    };
    if !executable_embeds_build_identity(&daemon_bytes, build_identity) {
        return Err(format!(
            "the bundled daemon does not carry version {} at commit {}",
            build_identity.application_version, build_identity.build_commit
        ));
    }
    Ok(())
}

/// True when the executable's bytes hold both this build's commit and its
/// version. The commit alone tells a leftover daemon from a fresh one; the
/// version is required as well so that an identity which was never compiled
/// in cannot pass on a chance match of one short string.
pub(crate) fn executable_embeds_build_identity(
    executable_bytes: &[u8],
    build_identity: &BuildIdentity,
) -> bool {
    // Lossy decoding keeps the embedded ASCII intact and lets the standard
    // substring search do the work. The daemon is about 13 MB, so the copy
    // costs a few milliseconds once per package build.
    let executable_text = String::from_utf8_lossy(executable_bytes);
    !build_identity.build_commit.is_empty()
        && !build_identity.application_version.is_empty()
        && executable_text.contains(&build_identity.build_commit)
        && executable_text.contains(&build_identity.application_version)
}
