//! Ad-hoc signing of an assembled bundle, leaf first.
//!
//! A seal covers everything beneath it, so nested code is signed before the
//! code that contains it: Sparkle's helpers, then the framework, then each
//! executable, then the app. Signing outward-in would invalidate the outer
//! seal the moment an inner signature changed.

use std::path::{Path, PathBuf};
use std::process::Command;

use super::build_steps::RUST_EXECUTABLE_NAMES;
use super::channel_profile::{ChannelProfile, SandboxPolicy};
use super::tool_command::run_quiet_tool;

/// Sparkle's nested code, innermost first.
const SPARKLE_NESTED_CODE_PATHS: [&str; 4] = [
    "Versions/B/Autoupdate",
    "Versions/B/Updater.app",
    "Versions/B/XPCServices/Downloader.xpc",
    "Versions/B/XPCServices/Installer.xpc",
];

pub(crate) fn ad_hoc_signing_order(
    bundle_path: &Path,
    channel_profile: &ChannelProfile,
) -> Vec<PathBuf> {
    let contents_path = bundle_path.join("Contents");
    let mut signing_order = Vec::new();
    if channel_profile.ships_sparkle {
        let sparkle_framework = contents_path.join("Frameworks/Sparkle.framework");
        for nested_code_path in SPARKLE_NESTED_CODE_PATHS {
            signing_order.push(sparkle_framework.join(nested_code_path));
        }
        signing_order.push(sparkle_framework);
    }
    signing_order.push(contents_path.join("MacOS/astronomical-menu"));
    for executable_name in RUST_EXECUTABLE_NAMES {
        signing_order.push(contents_path.join("MacOS").join(executable_name));
    }
    signing_order.push(bundle_path.to_path_buf());
    signing_order
}

/// Store builds need sandbox entitlements over a real identity; that stays
/// with the release packaging path, which this edit-loop command is not.
pub(crate) fn sign_bundle_ad_hoc(
    bundle_path: &Path,
    channel_profile: &ChannelProfile,
) -> Result<(), String> {
    if channel_profile.sandbox_policy == SandboxPolicy::AppStoreSandbox {
        return Err(
            "App Store bundles require a signing identity and sandbox entitlements".to_owned(),
        );
    }
    for code_object_path in ad_hoc_signing_order(bundle_path, channel_profile) {
        run_quiet_tool(
            Command::new("codesign")
                .args(["--force", "--sign", "-", "--preserve-metadata=entitlements"])
                .arg(&code_object_path),
        )?;
    }
    run_quiet_tool(
        Command::new("codesign")
            .args(["--verify", "--deep", "--strict"])
            .arg(bundle_path),
    )
    .map(drop)
}
