//! Signing of an assembled bundle, leaf first.
//!
//! A seal covers everything beneath it, so nested code is signed before the
//! code that contains it: Sparkle's helpers, then the framework, then the
//! menu-bar agent (its executable, then its bundle), then each executable of
//! the main bundle, then the app. Signing outward-in would invalidate the outer
//! seal the moment an inner signature changed.
//!
//! The plan is pure data, so every channel's signing contract (Developer ID
//! with hardened runtime and timestamp, App Store sandbox entitlements, a
//! bare CLI) is tested without running `codesign`.

use std::path::{Path, PathBuf};
use std::process::Command;

use super::application_group::{application_group_entitlements, application_group_identifier};
use super::channel_profile::{
    APPLICATION_EXECUTABLE_NAME, ChannelProfile, SandboxPolicy, TERMINAL_COMMAND_EXECUTABLE_NAME,
};
use super::tool_command::run_quiet_tool;

/// Sparkle's nested code, innermost first.
const SPARKLE_NESTED_CODE_PATHS: [&str; 4] = [
    "Versions/B/Autoupdate",
    "Versions/B/Updater.app",
    "Versions/B/XPCServices/Downloader.xpc",
    "Versions/B/XPCServices/Installer.xpc",
];

/// The checked-in sandbox profile of the two applications. It is a template:
/// each signing adds membership of the app group, which is named after the
/// signing team and so cannot be checked in.
pub(crate) const APPLICATION_ENTITLEMENTS_PATH: &str =
    "scripts/internal/entitlements/app-store-application.entitlements";
pub(crate) const INHERITED_ENTITLEMENTS_PATH: &str =
    "scripts/internal/entitlements/app-store-inherited.entitlements";

/// The entitlements one code object is signed with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SignedEntitlements {
    /// Unsandboxed code keeps whatever its build gave it.
    Preserved,
    /// A sandboxed application: the main app, the agent, or their bundles.
    SandboxedApplication,
    /// Code a sandboxed application spawns, which joins its sandbox.
    Inherited,
}

/// Where a signing writes the entitlements it renders for the two sandboxed
/// applications: beside the bundle, never inside it, so the file is not sealed
/// into the product.
pub(crate) fn rendered_application_entitlements_path(bundle_path: &Path) -> PathBuf {
    let mut entitlements_file_name = bundle_path
        .file_stem()
        .map(|bundle_stem| bundle_stem.to_os_string())
        .unwrap_or_default();
    entitlements_file_name.push(".application.entitlements");
    bundle_path.with_file_name(entitlements_file_name)
}

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
    let application_entitlements = application_entitlements_for(signing_request)?;
    let is_app_store = application_entitlements == SignedEntitlements::SandboxedApplication;
    let contents_path = signing_request.bundle_path.join("Contents");
    let mut signing_invocations = sparkle_signing_invocations(signing_request);
    signing_invocations.extend(agent_signing_invocations(
        signing_request,
        application_entitlements,
    ));
    for (executable_name, executable_entitlements) in
        executable_entitlements(signing_request.channel_profile, is_app_store)
    {
        signing_invocations.push(invocation(
            signing_request,
            contents_path.join("MacOS").join(executable_name),
            executable_entitlements,
        ));
    }
    signing_invocations.push(invocation(
        signing_request,
        signing_request.bundle_path.to_path_buf(),
        application_entitlements,
    ));
    Ok(signing_invocations)
}

/// What the two applications are signed with, after refusing a request that
/// cannot be honoured. A store build needs a real identity, because the
/// sandbox is established from signature entitlements, and that identity must
/// name its team, because the app group the two applications share is named
/// after it. Both are refused here, before anything is signed.
fn application_entitlements_for(
    signing_request: &SigningRequest<'_>,
) -> Result<SignedEntitlements, String> {
    if signing_request.channel_profile.sandbox_policy != SandboxPolicy::AppStoreSandbox {
        return Ok(SignedEntitlements::Preserved);
    }
    if signing_request.signing_identity.is_none() {
        return Err(
            "App Store builds require --signing-identity because every executable must carry sandbox entitlements over a real signature"
                .to_owned(),
        );
    }
    application_group_identifier(
        signing_request.channel_profile,
        signing_request.signing_identity,
    )?;
    Ok(SignedEntitlements::SandboxedApplication)
}

/// Sparkle's nested code and then the framework, for a channel that ships it.
fn sparkle_signing_invocations(signing_request: &SigningRequest<'_>) -> Vec<SigningInvocation> {
    let mut sparkle_invocations = Vec::new();
    if !signing_request.channel_profile.ships_sparkle {
        return sparkle_invocations;
    }
    let sparkle_framework = signing_request
        .bundle_path
        .join("Contents/Frameworks/Sparkle.framework");
    for nested_code_path in SPARKLE_NESTED_CODE_PATHS {
        sparkle_invocations.push(invocation(
            signing_request,
            sparkle_framework.join(nested_code_path),
            SignedEntitlements::Preserved,
        ));
    }
    sparkle_invocations.push(invocation(
        signing_request,
        sparkle_framework,
        SignedEntitlements::Preserved,
    ));
    sparkle_invocations
}

/// The agent's executable, then its bundle. The agent is an application in
/// its own right: it carries the sandbox profile itself, and the daemon it
/// spawns inherits that sandbox.
fn agent_signing_invocations(
    signing_request: &SigningRequest<'_>,
    application_entitlements: SignedEntitlements,
) -> Vec<SigningInvocation> {
    let mut agent_invocations = Vec::new();
    for agent_code_path in [
        signing_request
            .channel_profile
            .agent_executable_relative_path(),
        signing_request.channel_profile.agent_bundle_relative_path(),
    ] {
        agent_invocations.push(invocation(
            signing_request,
            signing_request.bundle_path.join(agent_code_path),
            application_entitlements,
        ));
    }
    agent_invocations
}

/// The main app carries the sandbox profile, as the agent does; the daemon
/// and worker the agent spawns join its sandbox through inherited
/// entitlements. The terminal command runs from the user's terminal, never
/// spawned by a sandboxed process, so it could only be signed bare; a
/// sandboxed channel therefore does not bundle it at all, and every
/// executable of a store bundle ends up sandboxed.
fn executable_entitlements(
    channel_profile: &ChannelProfile,
    is_app_store: bool,
) -> Vec<(&'static str, SignedEntitlements)> {
    let mut executable_entitlements = vec![(
        APPLICATION_EXECUTABLE_NAME,
        if is_app_store {
            SignedEntitlements::SandboxedApplication
        } else {
            SignedEntitlements::Preserved
        },
    )];
    for executable_name in channel_profile.bundled_rust_executable_names() {
        let joins_sandbox = is_app_store && executable_name != TERMINAL_COMMAND_EXECUTABLE_NAME;
        executable_entitlements.push((
            executable_name,
            if joins_sandbox {
                SignedEntitlements::Inherited
            } else {
                SignedEntitlements::Preserved
            },
        ));
    }
    executable_entitlements
}

fn invocation(
    signing_request: &SigningRequest<'_>,
    code_object_path: PathBuf,
    signed_entitlements: SignedEntitlements,
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
    match entitlements_path_for(signing_request, signed_entitlements) {
        Some(entitlements_path) => {
            codesign_arguments.push("--entitlements".to_owned());
            codesign_arguments.push(entitlements_path.display().to_string());
        }
        None => codesign_arguments.push("--preserve-metadata=entitlements".to_owned()),
    }
    SigningInvocation {
        code_object_path,
        codesign_arguments,
    }
}

/// The entitlements file a code object is signed with, or none when its
/// existing entitlements are kept.
fn entitlements_path_for(
    signing_request: &SigningRequest<'_>,
    signed_entitlements: SignedEntitlements,
) -> Option<PathBuf> {
    match signed_entitlements {
        SignedEntitlements::Preserved => None,
        SignedEntitlements::SandboxedApplication => Some(rendered_application_entitlements_path(
            signing_request.bundle_path,
        )),
        SignedEntitlements::Inherited => Some(
            signing_request
                .repository_root
                .join(INHERITED_ENTITLEMENTS_PATH),
        ),
    }
}

/// The signing plan, with the entitlements it refers to written to disk.
fn prepared_signing_plan(
    signing_request: &SigningRequest<'_>,
) -> Result<Vec<SigningInvocation>, String> {
    let signing_invocations = signing_plan(signing_request)?;
    render_application_entitlements(signing_request)?;
    Ok(signing_invocations)
}

/// Writes the entitlements the two sandboxed applications are signed with:
/// the checked-in sandbox profile plus membership of the app group named
/// after this signing's team. Nothing is written for an unsandboxed channel.
fn render_application_entitlements(signing_request: &SigningRequest<'_>) -> Result<(), String> {
    let Some(application_group_identifier) = application_group_identifier(
        signing_request.channel_profile,
        signing_request.signing_identity,
    )?
    else {
        return Ok(());
    };
    let sandbox_profile_path = signing_request
        .repository_root
        .join(APPLICATION_ENTITLEMENTS_PATH);
    let sandbox_profile_text = match std::fs::read_to_string(&sandbox_profile_path) {
        Ok(sandbox_profile_text) => sandbox_profile_text,
        Err(read_error) => {
            return Err(format!(
                "could not read {}: {read_error}",
                sandbox_profile_path.display()
            ));
        }
    };
    let rendered_entitlements_path =
        rendered_application_entitlements_path(signing_request.bundle_path);
    let rendered_entitlements_text =
        application_group_entitlements(&sandbox_profile_text, &application_group_identifier)?;
    match std::fs::write(&rendered_entitlements_path, rendered_entitlements_text) {
        Ok(()) => Ok(()),
        Err(write_error) => Err(format!(
            "could not write {}: {write_error}",
            rendered_entitlements_path.display()
        )),
    }
}

/// Runs the plan, then verifies the whole bundle strictly.
pub(crate) fn sign_bundle(signing_request: &SigningRequest<'_>) -> Result<(), String> {
    for signing_invocation in prepared_signing_plan(signing_request)? {
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
