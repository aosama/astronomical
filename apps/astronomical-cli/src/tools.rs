//! Supported launch tools for this slice. Only OpenCode is launchable; other
//! names fail closed instead of showing a fake picker.

use std::{
    ffi::OsStr,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
};

use crate::errors::LaunchError;

pub const OPENCODE_SLUG: &str = "opencode";
pub const OPENCODE_BINARY_NAME: &str = "opencode";

/// A supported harness that is present on PATH.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedLaunchTool {
    pub slug: String,
    pub program: PathBuf,
}

pub fn resolve_launch_tool(
    requested_tool_slug: Option<&str>,
    path_value: &OsStr,
) -> Result<ResolvedLaunchTool, LaunchError> {
    if let Some(requested_tool_slug) = requested_tool_slug {
        if requested_tool_slug != OPENCODE_SLUG {
            return Err(LaunchError::UnknownTool {
                requested_tool: requested_tool_slug.to_owned(),
            });
        }
    }
    let program =
        find_executable(OPENCODE_BINARY_NAME, path_value).ok_or(LaunchError::OpenCodeMissing)?;
    Ok(ResolvedLaunchTool {
        slug: OPENCODE_SLUG.to_owned(),
        program,
    })
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
    // Follows PATH shims so a symlink to OpenCode still counts as installed.
    let Ok(metadata) = candidate_path.metadata() else {
        return false;
    };
    metadata.is_file() && metadata.permissions().mode() & 0o111 != 0
}
