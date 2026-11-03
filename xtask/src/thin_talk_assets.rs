//! `cargo xtask vendor-thin-talk-assets [--verify-only]` — vendors the
//! pinned third-party web assets the Thin Talk conversation canvas renders
//! with. The canvas must work with no network at all, so every asset is
//! committed and served to the web view through a private scheme handler,
//! never a CDN. Typed port of the retired
//! `scripts/vendor-thin-talk-canvas-assets.sh`.
//!
//! An asset is downloaded only when it is missing or fails its pin, written
//! to a temporary sibling, and renamed into place, so an interrupted
//! download never leaves a partial file under its real name.

mod pinned_assets;

use std::path::{Path, PathBuf};
use std::process::Command;

use astronomical_native_identity::digest_bytes;

pub(crate) use pinned_assets::{PINNED_ASSETS, PINNED_FONTS, PINNED_LICENSES};

use crate::step_log::log_line;

pub(crate) const VENDOR_DIRECTORY: &str =
    "apps/thin-talk/Sources/ThinTalkCanvas/Resources/web/vendor";
const DOWNLOAD_URL_PREFIX: &str = "https://cdn.jsdelivr.net/npm";

/// One vendored file: where it lives under the vendor directory, where it
/// comes from on the CDN, and the digest it must match (`None` for
/// licenses, which only need to be present).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PinnedAsset {
    pub(crate) vendored_name: &'static str,
    pub(crate) package_path: &'static str,
    pub(crate) sha256: Option<&'static str>,
}

pub fn run(repository_root: &Path, arguments: &[String]) -> Result<(), String> {
    let is_verify_only = match arguments {
        [] => false,
        [verify_flag] if verify_flag == "--verify-only" => true,
        _ => return Err("usage: cargo xtask vendor-thin-talk-assets [--verify-only]".to_owned()),
    };
    let vendor_directory = repository_root.join(VENDOR_DIRECTORY);
    log_line(&format!(
        "operation=vendor-canvas-assets status=start directory={}",
        vendor_directory.display()
    ));
    let mut problem_count = 0;
    for pinned_asset in every_pinned_asset() {
        if let Err(asset_problem) = ensure_asset(&vendor_directory, &pinned_asset, is_verify_only) {
            eprintln!("Error: {asset_problem}");
            problem_count += 1;
        }
    }
    if problem_count > 0 {
        log_line(&format!(
            "operation=vendor-canvas-assets status=failed mismatched={problem_count}"
        ));
        return Err(format!(
            "{problem_count} vendored asset(s) are missing or fail their pin"
        ));
    }
    log_line(&format!(
        "operation=vendor-canvas-assets status=success assets={}",
        every_pinned_asset().len()
    ));
    Ok(())
}

pub(crate) fn every_pinned_asset() -> Vec<PinnedAsset> {
    let mut pinned_assets = PINNED_ASSETS.to_vec();
    pinned_assets.extend(PINNED_FONTS);
    pinned_assets.extend(PINNED_LICENSES);
    pinned_assets
}

/// Verifies one asset, downloading it first unless `--verify-only` or it
/// already satisfies its pin.
pub(crate) fn ensure_asset(
    vendor_directory: &Path,
    pinned_asset: &PinnedAsset,
    is_verify_only: bool,
) -> Result<(), String> {
    let asset_path = vendor_directory.join(pinned_asset.vendored_name);
    if !is_verify_only && asset_problem(&asset_path, pinned_asset).is_some() {
        log_line(&format!(
            "operation=vendor-canvas-assets status=download asset={}",
            pinned_asset.vendored_name
        ));
        download_asset(&asset_path, pinned_asset.package_path)?;
    }
    if let Some(asset_problem) = asset_problem(&asset_path, pinned_asset) {
        return Err(asset_problem);
    }
    let asset_bytes = std::fs::metadata(&asset_path).map_or(0, file_length);
    log_line(&format!(
        "operation=vendor-canvas-assets status=verified asset={} bytes={asset_bytes}",
        pinned_asset.vendored_name
    ));
    Ok(())
}

fn file_length(file_metadata: std::fs::Metadata) -> u64 {
    file_metadata.len()
}

/// What is wrong with a vendored file, or `None` when it satisfies its pin.
pub(crate) fn asset_problem(asset_path: &Path, pinned_asset: &PinnedAsset) -> Option<String> {
    let Ok(asset_bytes) = std::fs::read(asset_path) else {
        return Some(format!("{} is missing", pinned_asset.vendored_name));
    };
    match pinned_asset.sha256 {
        Some(_) if asset_bytes.is_empty() => {
            Some(format!("{} is empty", pinned_asset.vendored_name))
        }
        Some(pinned_digest) => {
            let actual_digest = digest_bytes(&asset_bytes);
            if actual_digest == pinned_digest {
                None
            } else {
                Some(format!(
                    "{} digest mismatch: expected {pinned_digest}, found {actual_digest}",
                    pinned_asset.vendored_name
                ))
            }
        }
        None if asset_bytes.is_empty() => {
            Some(format!("{} is missing", pinned_asset.vendored_name))
        }
        None => None,
    }
}

fn download_asset(asset_path: &Path, package_path: &str) -> Result<(), String> {
    if let Some(asset_parent) = asset_path.parent()
        && let Err(create_error) = std::fs::create_dir_all(asset_parent)
    {
        return Err(format!(
            "could not create {}: {create_error}",
            asset_parent.display()
        ));
    }
    let partial_path = partial_download_path(asset_path);
    let download_status = Command::new("curl")
        .args([
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--output",
        ])
        .arg(&partial_path)
        .arg(format!("{DOWNLOAD_URL_PREFIX}/{package_path}"))
        .status();
    if !matches!(download_status, Ok(exit_status) if exit_status.success()) {
        let _ = std::fs::remove_file(&partial_path);
        return Err(format!("could not download {package_path}"));
    }
    match std::fs::rename(&partial_path, asset_path) {
        Ok(()) => Ok(()),
        Err(rename_error) => Err(format!(
            "could not place {}: {rename_error}",
            asset_path.display()
        )),
    }
}

pub(crate) fn partial_download_path(asset_path: &Path) -> PathBuf {
    let mut partial_name = asset_path.file_name().unwrap_or_default().to_os_string();
    partial_name.push(".download");
    asset_path.with_file_name(partial_name)
}
