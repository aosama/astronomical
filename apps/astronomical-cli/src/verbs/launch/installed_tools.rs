//! Supported launch tools for this slice. Only registered slugs are
//! launchable; other names fail closed instead of showing a fake picker.

use std::{
    ffi::OsStr,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
};

use crate::verbs::launch::LaunchError;

pub const OPENCODE_SLUG: &str = "opencode";
pub const OPENCODE_BINARY_NAME: &str = "opencode";
pub const CODEX_SLUG: &str = "codex";
pub const CODEX_BINARY_NAME: &str = "codex";
pub const CLAUDE_CODE_SLUG: &str = "claude";
pub const CLAUDE_CODE_BINARY_NAME: &str = "claude";

/// The launchable harnesses this registry knows. Exhaustive on purpose:
/// adding a variant fails to compile until the environment building in
/// `launch.rs` handles it, so a registry entry can never launch without a
/// prepared session configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LaunchToolKind {
    OpenCode,
    Codex,
    ClaudeCode,
}

/// A supported harness that is present on PATH.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedLaunchTool {
    pub kind: LaunchToolKind,
    pub slug: String,
    /// Human-facing harness name for tool-specific advisories ("OpenCode",
    /// "Codex", "Claude Code").
    pub display_name: &'static str,
    pub program: PathBuf,
}

pub fn resolve_launch_tool(
    requested_tool_slug: Option<&str>,
    path_value: &OsStr,
) -> Result<ResolvedLaunchTool, LaunchError> {
    let (kind, binary_name, display_name) = launch_tool_registration(requested_tool_slug)?;
    let program = find_executable(binary_name, path_value).ok_or(missing_tool_error(kind))?;
    Ok(ResolvedLaunchTool {
        kind,
        slug: binary_name.to_owned(),
        display_name,
        program,
    })
}

/// Pure: the registry entry for a requested slug, as (kind, binary name,
/// display name). A bare launch still means the default harness.
fn launch_tool_registration(
    requested_tool_slug: Option<&str>,
) -> Result<(LaunchToolKind, &'static str, &'static str), LaunchError> {
    match requested_tool_slug {
        None | Some(OPENCODE_SLUG) => {
            Ok((LaunchToolKind::OpenCode, OPENCODE_BINARY_NAME, "OpenCode"))
        }
        Some(CODEX_SLUG) => Ok((LaunchToolKind::Codex, CODEX_BINARY_NAME, "Codex")),
        Some(CLAUDE_CODE_SLUG) => Ok((
            LaunchToolKind::ClaudeCode,
            CLAUDE_CODE_BINARY_NAME,
            "Claude Code",
        )),
        Some(requested_tool) => Err(LaunchError::UnknownTool {
            requested_tool: requested_tool.to_owned(),
        }),
    }
}

/// Pure: the generation endpoint each harness speaks, which decides which
/// installed models it can be launched on.
pub fn harness_endpoint(launch_tool_kind: LaunchToolKind) -> &'static str {
    match launch_tool_kind {
        LaunchToolKind::OpenCode => "/v1/chat/completions",
        LaunchToolKind::Codex => "/v1/responses",
        LaunchToolKind::ClaudeCode => "/v1/messages",
    }
}

/// Pure: the install hint for a registered harness that is not on PATH.
fn missing_tool_error(launch_tool_kind: LaunchToolKind) -> LaunchError {
    match launch_tool_kind {
        LaunchToolKind::Codex => LaunchError::CodexMissing,
        LaunchToolKind::OpenCode => LaunchError::OpenCodeMissing,
        LaunchToolKind::ClaudeCode => LaunchError::ClaudeCodeMissing,
    }
}

fn find_executable(binary_name: &str, path_value: &OsStr) -> Option<PathBuf> {
    let path_text = path_value.to_str()?;
    for directory in path_text.split(':') {
        if directory.is_empty() {
            continue;
        }
        let candidate_path = Path::new(directory).join(binary_name);
        if is_executable_file(&candidate_path) {
            return Some(candidate_path);
        }
    }
    None
}

fn is_executable_file(candidate_path: &Path) -> bool {
    // Follows PATH shims so a symlink to a harness still counts as installed.
    let Ok(metadata) = candidate_path.metadata() else {
        return false;
    };
    metadata.is_file() && metadata.permissions().mode() & 0o111 != 0
}
