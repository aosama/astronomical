//! The app icon, rendered once per channel and icon-renderer revision.
//!
//! The shell builder ran the Swift icon renderer and `iconutil` on every
//! build, although the icon depends only on the channel and the renderer's
//! source. Keying the cached `.icns` on both makes an edited renderer redraw
//! automatically and every other build reuse the file.

use std::path::{Path, PathBuf};
use std::process::Command;

use super::build_steps::file_sha256_hex;
use super::tool_command::run_quiet_tool;

const ICON_RENDERER_PATH: &str = "scripts/internal/render-macos-app-icon.swift";

pub(crate) fn cached_channel_icon(
    repository_root: &Path,
    icon_cache_directory: &Path,
    icon_channel: &str,
) -> Result<PathBuf, String> {
    let renderer_digest = file_sha256_hex(&repository_root.join(ICON_RENDERER_PATH))?;
    let cached_icon_path =
        icon_cache_directory.join(cached_icon_file_name(icon_channel, &renderer_digest));
    if cached_icon_path.is_file() {
        return Ok(cached_icon_path);
    }
    let iconset_directory = icon_cache_directory.join(format!("{icon_channel}.iconset"));
    if iconset_directory.exists()
        && let Err(remove_error) = std::fs::remove_dir_all(&iconset_directory)
    {
        return Err(format!("could not clear a stale iconset: {remove_error}"));
    }
    run_quiet_tool(
        Command::new("swift")
            .arg(repository_root.join(ICON_RENDERER_PATH))
            .arg("--output-directory")
            .arg(&iconset_directory)
            .args(["--channel", icon_channel]),
    )?;
    run_quiet_tool(
        Command::new("iconutil")
            .args(["--convert", "icns", "--output"])
            .arg(&cached_icon_path)
            .arg(&iconset_directory),
    )?;
    Ok(cached_icon_path)
}

pub(crate) fn cached_icon_file_name(icon_channel: &str, renderer_digest: &str) -> String {
    let short_digest_length = renderer_digest.len().min(16);
    format!(
        "{icon_channel}-{}.icns",
        &renderer_digest[..short_digest_length]
    )
}
