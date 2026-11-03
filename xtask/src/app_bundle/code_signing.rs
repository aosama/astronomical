//! Signing of an assembled bundle, leaf first.
//!
//! A seal covers everything beneath it, so nested code is signed before the
//! code that contains it: Sparkle's helpers, then the framework, then each
//! executable, then the app. Signing outward-in would invalidate the outer
//! seal the moment an inner signature changed.
//!
//! The plan is pure data, so every channel's signing contract (Developer ID
//! with hardened runtime and timestamp, App Store sandbox entitlements, a
//! bare CLI) is tested without running `codesign`.

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

pub(crate) const APPLICATION_ENTITLEMENTS_PATH: &str =
    "scripts/internal/entitlements/app-store-application.entitlements";
pub(crate) const INHERITED_ENTITLEMENTS_PATH: &str =
    "scripts/internal/entitlements/app-store-inherited.entitlements";

/// One `codesign` invocation: the code object and the arguments before it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SigningInvocation {
    pub(crate) code_object_path: PathBuf,
    pub(crate) codesign_arguments: Vec<String>,
}

/// Who signs: `None` signs ad hoc; `Some` signs with that identity, which
/// adds the hardened runtime and a secure timestamp notarization requires.
pub(crate) struct SigningRequest<'request> {
    pub(crate) bundle_path: &'request Path,
    pub(crate) channel_profile: &'request ChannelProfile,
    pub(crate) signing_identity: Option<&'request str>,
    pub(crate) repository_root: &'request Path,
}

pub(crate) fn signing_plan(
    signing_request: &SigningRequest<'_>,
) -> Result<Vec<SigningInvocation>, String> {
    let is_app_store =
        signing_request.channel_profile.sandbox_policy == SandboxPolicy::AppStoreSandbox;
    if is_app_store && signing_request.signing_identity.is_none() {
        return Err(
            "App Store builds require --signing-identity because every executable must carry sandbox entitlements over a real signature"
                .to_owned(),
        );
    }
    let contents_path = signing_request.bundle_path.join("Contents");
    let mut signing_invocations = Vec::new();
    if signing_request.channel_profile.ships_sparkle {
        let sparkle_framework = contents_path.join("Frameworks/Sparkle.framework");
        for nested_code_path in SPARKLE_NESTED_CODE_PATHS {
            signing_invocations.push(invocation(
                signing_request,
                sparkle_framework.join(nested_code_path),
                None,
            ));
        }
        signing_invocations.push(invocation(signing_request, sparkle_framework, None));
    }
    for (executable_name, entitlements_path) in executable_entitlements(is_app_store) {
        let executable_path = contents_path.join("MacOS").join(executable_name);
        signing_invocations.push(invocation(
            signing_request,
            executable_path,
            entitlements_path,
        ));
    }
    let bundle_entitlements = if is_app_store {
        Some(APPLICATION_ENTITLEMENTS_PATH)
    } else {
        None
    };
    signing_invocations.push(invocation(
        signing_request,
        signing_request.bundle_path.to_path_buf(),
        bundle_entitlements,
    ));
    Ok(signing_invocations)
}

/// The menu app carries the sandbox profile; the daemon and worker it spawns
/// join that sandbox through inherited entitlements. The CLI runs from the
/// user's terminal, never spawned by the sandboxed app, so it stays bare.
fn executable_entitlements(is_app_store: bool) -> Vec<(&'static str, Option<&'static str>)> {
    let mut executable_entitlements = vec![(
        "astronomical-menu",
        if is_app_store {
            Some(APPLICATION_ENTITLEMENTS_PATH)
        } else {
            None
        },
    )];
    for executable_name in RUST_EXECUTABLE_NAMES {
        let joins_sandbox = is_app_store && executable_name != "astronomical";
        executable_entitlements.push((
            executable_name,
            if joins_sandbox {
                Some(INHERITED_ENTITLEMENTS_PATH)
            } else {
                None
            },
        ));
    }
    executable_entitlements
}

fn invocation(
    signing_request: &SigningRequest<'_>,
    code_object_path: PathBuf,
    entitlements_path: Option<&str>,
) -> SigningInvocation {
    let mut codesign_arguments = vec!["--force".to_owned(), "--sign".to_owned()];
    match signing_request.signing_identity {
        Some(signing_identity) => {
            codesign_arguments.push(signing_identity.to_owned());
            codesign_arguments.extend([
                "--options".to_owned(),
                "runtime".to_owned(),
                "--timestamp".to_owned(),
            ]);
        }
        None => codesign_arguments.push("-".to_owned()),
    }
    match entitlements_path {
        Some(entitlements_path) => {
            codesign_arguments.push("--entitlements".to_owned());
            codesign_arguments.push(
                signing_request
                    .repository_root
                    .join(entitlements_path)
                    .display()
                    .to_string(),
            );
        }
        None => codesign_arguments.push("--preserve-metadata=entitlements".to_owned()),
    }
    SigningInvocation {
        code_object_path,
        codesign_arguments,
    }
}

/// Runs the plan, then verifies the whole bundle strictly.
pub(crate) fn sign_bundle(signing_request: &SigningRequest<'_>) -> Result<(), String> {
    for signing_invocation in signing_plan(signing_request)? {
        run_quiet_tool(
            Command::new("codesign")
                .args(&signing_invocation.codesign_arguments)
                .arg(&signing_invocation.code_object_path),
        )?;
    }
    run_quiet_tool(
        Command::new("codesign")
            .args(["--verify", "--deep", "--strict"])
            .arg(signing_request.bundle_path),
    )
    .map(drop)
}
